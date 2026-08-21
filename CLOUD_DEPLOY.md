# 云上从零跑通 RoboTwin 评测（LingBot-VA 推理）

目标：在一台 Linux + NVIDIA GPU 的云服务器上，跑通 RoboTwin-2.0 推理评测。
配套脚本：`cloud_robotwin_eval.sh`（自动完成依赖、下载、配置、启动）。

---

## 1. 租卡（控制台操作，不脚本化）

**选型：单卡 ≥24GB 显存即可**（推理用，offload 模式）。

| 档位 | 型号 | 是否需 offload |
|---|---|---|
| 24GB 卡 | RTX 4090 / A10 / L4 | ✅ 必须开 offload（脚本默认已开） |
| 40GB+ 卡 | A100 40G / A6000 48G | 可选关 |

- **镜像**：选带 **CUDA 12.6** 的 Linux 镜像，或 `pytorch 2.9 + cu126` 现成镜像（省得装驱动）。
- **平台**：国内优先 AutoDL（按小时、便宜、ModelScope 直下）；阿里云/腾讯云适合长期挂。
- **存储**：权重约几十 GB，数据集另算，留足空间。

## 2. 上传代码

把本地 `lingbot-va` 目录传上云（或用 `git clone https://github.com/robbyant/lingbot-va` 重新拉）：

```bash
scp -r lingbot-va user@<服务器IP>:~/
ssh user@<服务器IP>
cd ~/lingbot-va
```

## 3. 一键 / 分阶段跑

```bash
cd ~/lingbot-va

# 方式 A：全自动（环境检查→依赖→权重→配置→RoboTwin→启动）
bash cloud_robotwin_eval.sh all

# 方式 B：分阶段（推荐，出问题好定位）
bash cloud_robotwin_eval.sh setup     # 检查+装依赖+下权重+改配置
bash cloud_robotwin_eval.sh robowin   # 单独装 RoboTwin-2.0（最费时）
bash cloud_robotwin_eval.sh run       # 起服务端+客户端
```

**可配置变量**（在脚本顶部，或运行时用环境变量覆盖）：

```bash
MODEL_DIR=~/models/lingbot-va-posttrain-robotwin  # 权重目录
DL_SOURCE=modelscope                              # 国内默认 modelscope
TASK_NAME=adjust_bottle                           # 评测任务
ENABLE_OFFLOAD=true                               # 24GB 卡必须 true
```

## 4. 脚本做了什么（对照源码里的关键坑）

脚本会自动改这几处源码里写死/默认不对的地方：

| 文件 | 改什么 | 为什么 |
|---|---|---|
| `wan_va/configs/va_robotwin_cfg.py` | `wan22_pretrained_model_name_or_path` → 实际权重路径 | 服务端用 argparse，**不支持命令行 override**，必须改文件 |
| `evaluation/robotwin/eval_polict_client_openpi.py` | `robowin_root` → 实际 RoboTwin 路径 | 客户端里写死 `/path/to/your/robowin` |
| `wan_va/configs/va_robotwin_cfg.py` | 追加 `enable_offload = True` | 默认是 `False`，24GB 卡会 OOM |
| `<权重>/transformer/config.json` | `attn_mode` → `torch` | README 强制要求，推理不能用 `flex` |

## 5. 验证结果

- 服务端日志：`logs/server.log`，看到模型加载完成、`Serve` 相关输出即正常。
- 结果目录：`/path/to/your/RoboTwin/${SAVE_ROOT}`（默认 `results/`），含每任务评测视频/指标。
- 还有个 `eval_result` 目录是 RoboTwin 原生输出，和 `results/` 内容一致，可忽略。

## 6. 常见问题

**flash-attn 装失败（PEP 517）**
```bash
pip install --upgrade pip setuptools wheel
pip install flash-attn --no-build-isolation
# 或
pip install git+https://github.com/Dao-AILab/flash-attention.git
```

**OOM / CUDA out of memory**
- 确认 `enable_offload = True` 已生效（`grep enable_offload wan_va/configs/va_robotwin_cfg.py`）。
- 或换 40GB+ 卡。

**下载慢 / 连不上 HuggingFace**
- 用 ModelScope（脚本默认）；或 HF 设镜像：`export HF_ENDPOINT=https://hf-mirror.com`。

**服务端端口 29056 没起来**
- 看 `logs/server.log`；确认服务端和客户端**在同一台机器**。

**RoboTwin `_install.sh` 里没找到 pytorch3d 行**
- 手动按 README 把 `_install.sh` 第 8 行改成：
  `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation`

## 7. 评测任务名

`TASK_NAME` 是 RoboTwin 50 任务之一，全部清单见 `evaluation/robotwin/launch_client.sh` 的 `task_groups` 数组，例如 `adjust_bottle`、`stack_bowls_three`、`open_microwave` 等。多卡评测参考 `launch_client_multigpus.sh`（按 8 卡分 7 组）。
