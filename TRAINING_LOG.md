# 训练运行日志（Training Log）

> 本文件记录 LingBot-VA 阶段二正式训练的实时进度。训练期间由 Claude Code 定期更新并推送。
> 训练启动：2026-08-29 14:33（本地时间，UTC+8）

## 训练配置快照

| 项 | 值 |
|---|---|
| 模型 | LingBot-VA posttrain-robotwin（transformer 9.5GB） |
| 数据集 | robotwin-clean-and-aug-lerobot（100 子数据集，27500 episode） |
| 单卡 | RTX PRO 6000 Blackwell（96GB） |
| num_steps | 50000 |
| batch_size / grad_accum | 1 / 1 |
| learning_rate | 1e-5 |
| save_interval | 10000（5 个 checkpoint） |
| 数据加载 | ThreadPool（已修复 Pool 多进程 hang） |
| 步速（实测） | ~1.18 秒/步 |
| 预计总时长 | ~16 小时 |

---

## 进度记录

### 2026-08-29 15:47（启动后 1h14m）

- 进度：**3716 / 50000**（7.4%）
- 步速：1.18 s/it
- 预计剩余：~15h14m
- loss：latent=0.2049，action=0.0017，grad_norm=0.72
- GPU：温度 **90°C**，利用率 100%，显存 87.4GB
- 状态：✅ 正常
- 备注：GPU 温度从 87°C 升到 90°C，仍属满载正常偏高区间，持续观察

<!-- 后续记录在此追加 -->
