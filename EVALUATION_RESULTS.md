# 评测结果（Evaluation Results）

> 记录 LingBot-VA 在 RoboTwin 仿真环境上的评测结果。
> 评测进行中：5 任务 × 各 20 episodes，随跑随更。
> 说明：回放视频文件较大，**不上传 GitHub**，保存在本地（路径见文末）。

## 一、评测计划

评测脚本 `run_eval_5tasks_20.sh`，5 个任务、每个 20 episodes（`--test_num 20`）：

| # | 任务 | 说明 |
|---|---|---|
| 1 | `stack_bowls_three` | 叠三个碗 |
| 2 | `handover_block` | 递积木 |
| 3 | `hanging_mug` | 挂杯 |
| 4 | `scan_object` | 扫描物体 |
| 5 | `lift_pot` | 举锅 |

## 二、当前结果（截至 2026-09-01 09:23）

| 任务 | 成功率 | 状态 |
|---|---|---|
| `adjust_bottle` | **5/5 = 100%** | ✅ 完成（更早单独跑的 5 episodes） |
| `stack_bowls_three` | 3/3 = 100% | ⏸ 卡死前结果，待补跑 |
| `handover_block` | — | ⏳ 待补跑（并发冲突导致假失败，无结果） |
| `hanging_mug` | 0/1 = 0% | 🟢 进行中 |
| `scan_object` | — | ⏳ 未开始 |
| `lift_pot` | — | ⏳ 未开始 |

> `stack_bowls_three` / `handover_block` 因「两个 client 并发连同一 server」导致 VAE 状态污染而假失败，计划用 `run_eval_2tasks.sh` 补跑。

## 三、当天遇到的问题（简要）

- **坑 11**：ffmpeg 视频录制管道死锁 → 后台线程 + 有界队列（已修复）
- **坑 12**：client↔server websocket 通信僵局 → `asyncio.to_thread`（已修复）
- **并发 client**：漏 kill 旧脚本导致双 client 踩坏 VAE 状态 → 已清理，需补跑 2 任务

详细诊断与修复见 [EVALUATION_LOG.md](EVALUATION_LOG.md) 和 [EVALUATION_TROUBLESHOOTING.md](EVALUATION_TROUBLESHOOTING.md)。

## 四、hanging_mug 失败原因分析（14/14 失败）

**结论：判定逻辑 bug（不是能力不足）** ✅ 已定位

**bug 位置**（`envs/hanging_mug.py` 的 `check_success`）：

```python
rack_middle_pose = (rack_pose + rack_function_pose) / 2   # ⚠️ 用了「中点」
return abs((mug_function_pose - rack_middle_pose)[:2]) < 0.02
```

**根因**：

- 操作目标（`play_once`）是把杯子挂到**挂杆末端** `rack_function_pose`；
- 判定却要求杯子在**架子和挂杆末端的「中点」** `rack_middle_pose` 的 2cm 内；
- 挂杆末端相对架子中心偏移约 **(6.5cm, 19.5cm)**，中点离挂杆末端约 **10cm**，远超 2cm 容差；
- 所以即使杯子精确挂在正确位置（挂杆末端），也会因离「中点」差 10cm 而被判失败 → 14/14 全失败。

**修复**（一行）：`rack_middle_pose = rack_function_pose`（直接对齐挂杆末端）。

> 之前曾误判为「模型精细操作能力不足」，实为判定位置用错（中点 vs 挂杆末端）。待修复后重跑验证。

## 五、数据文件结构

**已上传 GitHub**（小文件，`res.json`）：

```
results/
├── adjust_bottle/res.json
├── stack_bowls_three/res.json
└── hanging_mug/res.json
```

**回放视频本地保存**（不上传 GitHub）：

```
/home/mosense/RoboTwin/results/stseed-10000/visualization/
├── adjust_bottle/      # 5 段，均 _True
└── stack_bowls_three/  # 含 _True / _False
```

> 训练 checkpoint 权重（48GB）也不上传，本地路径见 [TRAINING_REPORT.md](TRAINING_REPORT.md)。
