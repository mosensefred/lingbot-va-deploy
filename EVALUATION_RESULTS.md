# 评测结果（Evaluation Results）

> 记录 LingBot-VA 在 RoboTwin 仿真环境上的评测结果。
> 评测进行中：5 任务 × 各 20 episodes，随跑随更。
> 说明：回放视频文件较大，**不上传 GitHub**，保存在本地（路径见「四、数据文件结构」）。

## 一、评测计划

评测脚本 `run_eval_5tasks_20.sh`，5 个任务、每个 20 episodes（`--test_num 20`）：

| # | 任务 | 说明 | 状态 |
|---|---|---|---|
| 1 | `stack_bowls_three` | 叠三个碗 | 🟢 进行中 |
| 2 | `handover_block` | 递积木 | ⏳ 待跑 |
| 3 | `hanging_mug` | 挂杯 | ⏳ 待跑 |
| 4 | `scan_object` | 扫描物体 | ⏳ 待跑 |
| 5 | `lift_pot` | 举锅 | ⏳ 待跑 |

## 二、已完成结果

### adjust_bottle（抓取/举起瓶子，5 episodes）

| 指标 | 值 |
|---|---|
| 成功率 | **5 / 5 = 100%** |
| res.json | [`results/adjust_bottle/res.json`](results/adjust_bottle/res.json) |

5 个任务变体（不同瓶子、左右臂）全部一次通过，回放视频本地保存（5 段，均 `_True`）。

## 三、进行中

### stack_bowls_three（叠碗，20 episodes）

- 当前 `res.json`：1 / 1（部分结果，完整成功率待 20 episodes 跑完后统计）
- 回放视频本地保存（含成功/失败，见文件名 `_True` / `_False` 后缀）

## 四、数据文件结构

**已上传 GitHub**（小文件）：

```
results/
├── adjust_bottle/res.json     # { succ_num, total_num, succ_rate }
└── stack_bowls_three/res.json
```

**回放视频本地保存**（不上传 GitHub）：

```
/home/mosense/RoboTwin/results/stseed-10000/visualization/
├── adjust_bottle/      # 5 段，均 _True
└── stack_bowls_three/  # 含 _True / _False
```

> 训练 checkpoint 权重（5 个 step checkpoint，共 48GB）同样不上传，保存在本地
> `/media/mosense/Data2TB/fred/lingbot-va-train-out/checkpoints/`。训练过程与 loss 分析见
> [TRAINING_REPORT.md](TRAINING_REPORT.md)。
