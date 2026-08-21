# LingBot-VA 部署与运行清单

> 配套文件：一键云上评测脚本 [cloud_robotwin_eval.sh](cloud_robotwin_eval.sh) · 云上部署指南 [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md)

---

## 0. 硬性门槛（先对号入座）

| 项目 | 要求 | 说明 |
|---|---|---|
| 系统 | **Linux**（+ Vulkan） | Windows 不兼容 |
| Python | **3.10.16** | 精确版本 |
| PyTorch | **2.9.0**（cu126） | torchvision 0.24.0 / torchaudio 2.9.0 |
| CUDA | **12.6** | |
| 显存 | 推理 **18–24GB**；训练 **8 卡** | 24GB 卡需开 offload |

## 1. 安装

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation    # 若 PEP 517 报错，改用 git 安装
```

## 2. 下载模型权重

按用途二选一（HuggingFace / ModelScope 均可，国内优先 ModelScope）：

```bash
# 后训练基座（做微调用）
huggingface-cli download robbyant/lingbot-va-base --local-dir ./lingbot-va-base

# 已后训练好的推理权重（做 RoboTwin 评测用）
huggingface-cli download robbyant/lingbot-va-posttrain-robotwin --local-dir ./lingbot-va-posttrain-robotwin
```

把下载路径填进配置：`wan_va/configs/va_robotwin_cfg.py`（或对应 cfg）里的 `wan22_pretrained_model_name_or_path`。

## 3. ⚠️ 最容易踩的坑：`attn_mode`

模型是从 `transformer/config.json` 里读 `attn_mode` 的，**必须手动改**：

| 用途 | 改成 | 备注 |
|---|---|---|
| **训练** | `"flex"` | 推理时用 flex 会报错 |
| **推理/评测** | `"torch"` 或 `"flashattn"` | |

> 改 `<你的模型路径>/transformer/config.json` 里的 `"attn_mode"` 字段。

## 4. 推理部署（Server-Client 架构）

服务端和客户端必须跑在**同一台机器**上。先起服务端：

**RoboTwin 评测**（先按 README 装 RoboTwin-2.0 环境）：
```bash
# 服务端
bash evaluation/robotwin/launch_server.sh
# 客户端（指定任务）
task_name="adjust_bottle"; save_root="results/"
bash evaluation/robotwin/launch_client.sh ${save_root} ${task_name}
```
> 显存：约 **24GB**（offload 模式，VAE + text_encoder 卸到 CPU）

**LIBERO 评测**：
```bash
bash evaluation/libero/launch_server.sh
bash evaluation/libero/launch_client.sh
```

**图像→视频-动作生成（i2va）**：
```bash
NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh
```
> 显存：约 **18GB**

## 5. 后训练（微调自定义数据集）

```bash
pip install lerobot==0.3.3 scipy wandb --no-deps

# 下载官方后训练数据
huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot --local-dir /path/to/dataset

# RoboTwin 训练（8卡）
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh
# LIBERO 训练
NGPU=8 CONFIG_NAME='libero_train' bash script/run_va_posttrain.sh
```

自定义数据 4 步：**① 转 LeRobot 格式 → ② 在 `episodes.jsonl` 加 `action_config` 动作分段 → ③ 用 Wan2.2 VAE 抽 latent 放到 `latents/` → ④ 按 30 维动作格式对齐**。
动作格式固定为 30 维（左右臂 EEF 各 7 + 左右臂关节各 7 + 左右夹爪各 1）。

## 6. 配置注册名速查（`VA_CONFIGS` 的 key）

| 用途 | `CONFIG_NAME` |
|---|---|
| RoboTwin 推理 | `robotwin` |
| RoboTwin 训练 | `robotwin_train` |
| RoboTwin i2va | `robotwin_i2av` |
| LIBERO 推理/训练/i2va | `libero` / `libero_train` / `libero_i2av` |
| Franka（真实机械臂） | `franka` / `franka_i2av` |
| Demo（示例） | `demo` / `demo_train` / `demo_i2av` |

---

## 附：源码里的三处关键硬编码（部署前必须确认）

| 文件 | 占位符/默认值 | 说明 |
|---|---|---|
| `wan_va/configs/va_robotwin_cfg.py` | `wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"` | 服务端用 argparse，**不支持命令行 override** |
| `evaluation/robotwin/eval_polict_client_openpi.py` | `robowin_root = Path("/path/to/your/robowin")` | 客户端写死 RoboTwin 路径 |
| `wan_va/configs/shared_config.py` | `enable_offload = False` | 24GB 卡需改成 `True` 防 OOM |

> 用 [cloud_robotwin_eval.sh](cloud_robotwin_eval.sh) 可自动完成以上三处改动。
