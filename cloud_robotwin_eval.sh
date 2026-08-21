#!/usr/bin/env bash
# ============================================================================
# LingBot-VA 云上 RoboTwin 评测：从零到跑通（单卡推理）
#
# 用法:
#   bash cloud_robotwin_eval.sh setup    # 检查环境 + 装依赖 + 下权重 + 改配置
#   bash cloud_robotwin_eval.sh robowin  # 单独装 RoboTwin-2.0 环境（最费时）
#   bash cloud_robotwin_eval.sh run      # 起服务端 + 客户端
#   bash cloud_robotwin_eval.sh all      # 一口气全做（默认）
#
# 建议: 先 setup -> robowin -> run 分阶段跑，出问题好定位。
# ============================================================================
set -uo pipefail

# ---------- 可配置变量（改成你自己的） ----------
LINGBOT_DIR="${LINGBOT_DIR:-$HOME/lingbot-va}"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$HOME/RoboTwin}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/lingbot-va-posttrain-robotwin}"
DL_SOURCE="${DL_SOURCE:-modelscope}"        # modelscope | huggingface（国内优先 modelscope）
TASK_NAME="${TASK_NAME:-adjust_bottle}"     # RoboTwin 50 任务之一，见 CLOUD_DEPLOY.md
SAVE_ROOT="${SAVE_ROOT:-results}"
ENABLE_OFFLOAD="${ENABLE_OFFLOAD:-true}"    # 24GB 卡必须 true；40GB+ 卡可 false
# --------------------------------------------------

cd "$LINGBOT_DIR" || { echo "找不到 $LINGBOT_DIR"; exit 1; }

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m[FAIL] %s\033[0m\n' "$*"; exit 1; }

# ---------- Phase 0: 环境检查 ----------
phase_check() {
  say "Phase 0: 环境检查"
  command -v nvidia-smi >/dev/null || die "没有 nvidia-smi，确认这是带 NVIDIA GPU 的实例"
  nvidia-smi
  command -v python >/dev/null || die "没有 python，先装 Python 3.10.16"
  python -V
  command -v git >/dev/null || die "没有 git"
}

# ---------- Phase 1: 装 lingbot-va 依赖 ----------
phase_deps() {
  say "Phase 1: 安装 lingbot-va 依赖"
  pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
      --index-url https://download.pytorch.org/whl/cu126 || die "torch 安装失败"
  pip install websockets einops diffusers==0.36.0 transformers==4.55.2 \
      accelerate msgpack opencv-python matplotlib ftfy easydict || die "依赖安装失败"
  pip install flash-attn --no-build-isolation || say "flash-attn 装失败，见 CLOUD_DEPLOY.md 兜底方案"
}

# ---------- Phase 2: 下载权重 ----------
phase_download() {
  say "Phase 2: 下载权重 -> $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  if [ "$DL_SOURCE" = "modelscope" ]; then
    pip install modelscope -q
    modelscope download --model Robbyant/lingbot-va-posttrain-robotwin --local_dir "$MODEL_DIR" \
      || die "ModelScope 下载失败"
  else
    pip install -U "huggingface_hub[cli]" -q
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    huggingface-cli download robbyant/lingbot-va-posttrain-robotwin --local-dir "$MODEL_DIR" \
      || die "HuggingFace 下载失败"
  fi
  [ -d "$MODEL_DIR/transformer" ] || die "权重目录缺少 transformer/，检查是否下全（应含 vae/ tokenizer/ text_encoder/ transformer/）"
}

# ---------- Phase 3: 改配置 ----------
phase_config() {
  say "Phase 3: 写配置（模型路径 / RoboTwin 路径 / offload / attn_mode）"

  # 3.1 模型路径写入服务端 cfg
  sed -i "s|/path/to/pretrained/model|$MODEL_DIR|g" wan_va/configs/va_robotwin_cfg.py

  # 3.2 客户端 RoboTwin 根目录
  sed -i "s|/path/to/your/robowin|$ROBOTWIN_DIR|g" evaluation/robotwin/eval_polict_client_openpi.py

  # 3.3 显存 offload（24GB 卡必须开，否则 VAE+text_encoder 常驻显存会 OOM）
  if [ "$ENABLE_OFFLOAD" = "true" ]; then
    grep -q "enable_offload = True" wan_va/configs/va_robotwin_cfg.py || \
      echo "va_robotwin_cfg.enable_offload = True" >> wan_va/configs/va_robotwin_cfg.py
    say "已开启 enable_offload（VAE+text_encoder 卸到 CPU）"
  fi

  # 3.4 attn_mode -> torch（README 强制要求，推理必须）
  python - "$MODEL_DIR/transformer/config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
if d.get("attn_mode") != "torch":
    d["attn_mode"] = "torch"
    json.dump(d, open(p, "w"), indent=2)
    print("attn_mode -> torch")
else:
    print("attn_mode 已是 torch")
PY

  say "配置完成。确认 grep 结果："
  grep -n "wan22_pretrained_model_name_or_path" wan_va/configs/va_robotwin_cfg.py
  grep -n "robowin_root" evaluation/robotwin/eval_polict_client_openpi.py
}

# ---------- Phase 4: 装 RoboTwin-2.0 环境 ----------
phase_robowin() {
  say "Phase 4: 安装 RoboTwin-2.0 环境（最耗时，可单独跑）"
  sudo apt update
  sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools || die "Vulkan 安装失败"

  if [ ! -d "$ROBOTWIN_DIR" ]; then
    git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_DIR" || die "clone RoboTwin 失败"
  fi
  cd "$ROBOTWIN_DIR"
  git checkout 2eeec322 || die "checkout 2eeec322 失败"

  # 替换 requirements.txt（README 指定内容）
  cat > script/requirements.txt <<'EOF'
transforms3d==0.4.2
sapien==3.0.0b1
scipy==1.10.1
mplib==0.2.1
gymnasium==0.29.1
trimesh==4.4.3
open3d==0.18.0
imageio==2.34.2
pydantic
zarr
openai
huggingface_hub==0.36.2
h5py
azure==4.0.0
azure-ai-inference
pyglet<2
wandb
moviepy
imageio
termcolor
av
matplotlib
ffmpeg
EOF

  # pytorch3d 安装行修正（README 要求改 _install.sh 第 8 行）
  if grep -q "pytorch3d" script/_install.sh; then
    sed -i 's|.*pytorch3d.*|pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation|' script/_install.sh
  else
    say "未在 script/_install.sh 找到 pytorch3d 行，请手动按 README 改第 8 行"
  fi

  bash script/_install.sh || die "_install.sh 失败"
  bash script/_download_assets.sh || die "下载 assets 失败"
  cd "$LINGBOT_DIR"
  say "RoboTwin 环境装好"
}

# ---------- Phase 5: 起服务端 + 客户端 ----------
phase_run() {
  say "Phase 5: 启动服务端与客户端"

  mkdir -p logs
  nohup bash evaluation/robotwin/launch_server.sh > logs/server.log 2>&1 &
  echo $! > logs/server.pid
  say "服务端已后台启动 (pid $(cat logs/server.pid))，日志 logs/server.log"

  # 等端口 29056 就绪（最多 10 分钟）
  python - <<'PY'
import socket, time, sys
for _ in range(120):
    try:
        socket.create_connection(("127.0.0.1", 29056), timeout=1).close()
        print("服务端端口就绪"); sys.exit(0)
    except OSError:
        time.sleep(5)
print("等待超时，看 logs/server.log"); sys.exit(1)
PY
  sleep 10

  say "启动客户端：task=$TASK_NAME save_root=$SAVE_ROOT"
  bash evaluation/robotwin/launch_client.sh "$SAVE_ROOT" "$TASK_NAME"
}

# ---------- main ----------
MODE="${1:-all}"
case "$MODE" in
  setup)   phase_check; phase_deps; phase_download; phase_config ;;
  robowin) phase_robowin ;;
  run)     phase_run ;;
  all)     phase_check; phase_deps; phase_download; phase_config; phase_robowin; phase_run ;;
  *) echo "用法: $0 [setup|robowin|run|all]"; exit 1 ;;
esac

say "完成"
