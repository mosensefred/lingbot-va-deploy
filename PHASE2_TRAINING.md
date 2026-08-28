# LingBot-VA 阶段二：后训练（本地单卡）

> 本文档承接 `LOCAL_DEPLOY_SUMMARY.md` 的「六、阶段二：后训练」，记录后训练的实际执行结果与后续训练步骤。
> 状态：**数据集就绪、配置就绪、训练冒烟测试进行中**。

---

## 一、当前进度总览

| 步骤 | 状态 | 说明 |
|---|---|---|
| 数据集下载 | ✅ 完成 | ModelScope 分卷 `.aa`(52.4G) + `.ab`(52.1G)，共约 104GB |
| 数据集解压 | ✅ 完成 | 流式解压到数据盘，共 **415GB** |
| `dataset_path` 配置 | ✅ 完成 | 指向 Data2TB 数据盘 |
| `enable_wandb` | ✅ 关闭 | 原配置为占位 key，训练前必须关 |
| `save_root` 配置 | ✅ 完成 | 指向 Data2TB（避开根分区，checkpoint 大） |
| 单任务数据加载 | ✅ 通过 | latents/actions/text_emb 张量规格全部对齐 |
| 单卡训练冒烟测试 | 🔄 进行中 | 3 步验证，正反向 + FSDP 全链路 |

---

## 二、关键结论（本次执行新发现）

### 2.1 数据集解压后是 415GB，不是 130GB ⚠️

`LOCAL_DEPLOY_SUMMARY.md` 原估算「解压约 100~130GB」**严重偏低**。实际：

| 项 | 大小 |
|---|---|
| 压缩包（.aa + .ab） | 104GB |
| **解压后** | **415GB** |

根分区 `/` 仅 115GB 可用，**根本装不下**。这是本次执行遇到的最大的、文档未预料到的障碍。

### 2.2 解决方案：改用 Data2TB 数据盘

本机第二块盘 `/media/mosense/Data2TB`（1.8T，机械盘 WD20EZBX）有 539GB 可用，足够放下 415GB 数据集。

最终数据存放路径（**新建独立目录，不触碰盘上已有 `Projects/` 等项目**）：

```
/media/mosense/Data2TB/fred/lingbot-va-data/robotwin-clean-and-aug-lerobot/
```

### 2.3 数据集真实结构

```
robotwin-clean-and-aug-lerobot/
├── empty_emb.pt                    # 空文本 embedding [512, 4096] bf16
├── README.md
├── lerobot_robotwin_eef_aug_500/   # 增强集，50 个任务
│   └── <task>/
│       ├── data/chunk-000/         # *.parquet
│       ├── latents/                # 已提取的 VAE latent（*.pth）
│       ├── meta/                   # info.json + episodes.jsonl
│       └── videos/                 # 原始视频（训练不读）
└── lerobot_robotwin_eef_clean_50/  # 干净集，50 个任务
```

- **共 100 个子数据集**（50 aug + 50 clean），每个任务是一个独立 LeRobot v2.1 数据集。
- `dataset_path` 指向 `robotwin-clean-and-aug-lerobot/`，`recursive_find_file('info.json')` 会自动发现全部 100 个。
- 单任务实测：500 episode → `len(ds)=498`（2 个缺 latent 被 `_check_meta` 过滤）。

### 2.4 训练入口不依赖 torchft

`train.py` 用标准 `dist.init_process_group(backend="nccl", init_method="env://")`，读 `RANK/LOCAL_RANK/WORLD_SIZE` 环境变量，**不 import torchft**。因此单卡直接 `torch.distributed.run` 即可，无需 `run_va_posttrain.sh` 里的 `--local-ranks-filter` 等 torchft 参数。

### 2.5 训练时 `attn_mode` 无需改 config.json

`train.py` 加载 transformer 时**硬编码 `attn_mode="flex"`**（[train.py:88](../../lingbot-va/wan_va/train.py#L88)），会覆盖模型目录 `transformer/config.json` 里的 `"torch"`。所以训练**不需要**手动改 config.json（那个只在推理时读取）。

---

## 三、单任务数据加载验证（已通过）

实测单个任务 `adjust_bottle-aloha-agilex_randomized_500-1000`：

| 字段 | shape | dtype |
|---|---|---|
| latents | `[48, 9, 24, 20]` | bfloat16 |
| text_emb | `[512, 4096]` | bfloat16 |
| actions | `[30, 9, 16, 1]` | float32 |
| actions_mask | `[30, 9, 16, 1]` | bool |

与阶段一推理验证的张量规格一致，数据链路完整。

---

## 四、配置改动记录

文件 `configs/va_robotwin_train_cfg.py` 本次实际改动：

| 配置项 | 原值 | 现值 |
|---|---|---|
| `dataset_path` | `/path/to/your/dataset` | `/media/mosense/Data2TB/fred/lingbot-va-data/robotwin-clean-and-aug-lerobot` |
| `save_root` | `./train_out`（相对，会写根分区） | `/media/mosense/Data2TB/fred/lingbot-va-train-out` |
| `enable_wandb` | `True` | `False` |
| `num_steps` | `50000` | `3`（冒烟测试临时值，测试后改回） |

其余训练超参保持默认：`lr=1e-5`、`batch_size=1`、`gradient_accumulation_steps=1`、`save_interval=1000`、`gc_interval=50`、`cfg_prob=0.1`。

---

## 五、启动命令

### 5.1 单卡冒烟测试（当前运行中）

```bash
cd ~/lingbot-va
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lingbot
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false \
python -m torch.distributed.run --nproc_per_node=1 --master_port 29501 \
    -m wan_va.train --config-name robotwin_train
```

### 5.2 正式训练（冒烟通过后）

```bash
# 1) 把 num_steps 从 3 改回 50000
# 2) 启动
cd ~/lingbot-va
source ~/anaconda3/etc/profile.d/conda.sh && conda activate lingbot
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false \
python -m torch.distributed.run --nproc_per_node=1 --master_port 29501 \
    -m wan_va.train --config-name robotwin_train
```

---

## 六、后续训练步骤（待办）

1. **等冒烟测试出 loss 结果**：确认正反向 + FSDP 全链路通过，记录首步 latent_loss / action_loss。
2. **把 `num_steps` 改回 50000**。
3. **解决 checkpoint 磁盘问题** ⚠️：
   - 单 checkpoint ≈ **9.5GB**（bf16 transformer）。
   - `save_interval=1000` × `num_steps=50000` → **50 个 checkpoint ≈ 475GB**。
   - Data2TB 解压后仅剩 **124GB**，根分区仅剩 114GB，**都存不下 475GB**。
   - 需要决策：减少 `save_interval`、只保留最后 N 个 checkpoint、或换更大磁盘。
4. **监控训练指标**：wandb 已关闭，需靠终端 `progress_bar` 输出观察 loss；若要曲线图需配真实 wandb 凭据。
5. **机械盘 IO 注意**：Data2TB 是机械盘，数据集初始化（100 个子数据集的 latent 文件存在性校验）和 DataLoader 读 latent 都较慢，首次启动需耐心等待。

---

## 七、磁盘现状（截至本文档）

| 磁盘 | 容量 | 已用 | 可用 | 用途 |
|---|---|---|---|---|
| `/`（nvme，系统盘） | 657G | 510G | 114G | 源码 / 权重 / conda 环境 |
| `/media/mosense/Data2TB`（sda，机械盘） | 1.8T | 1.6T | **124G** | 数据集 415G + 训练输出 |

> ⚠️ 数据盘剩余 124G 不足以存下 50 个 checkpoint（475G），正式训练前必须先定 checkpoint 策略。
