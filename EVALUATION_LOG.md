# RoboTwin 评测记录（2026-08-31 ~ 09-01）

> 本次评测的目标、进展、遇到的卡死问题与根治过程。随跑随更。

## 一、评测概览

| 项 | 值 |
|---|---|
| 评测脚本 | `run_eval_5tasks_20.sh`（5 任务 × 各 20 episodes） |
| 任务集 | stack_bowls_three / handover_block / hanging_mug / scan_object / lift_pot |
| 架构 | 双环境：server（lingbot，LingBot-VA 推理）+ client（robotwin，RoboTwin 仿真），websocket 通信 |
| 策略 | ACT，`--video_guidance_scale 5 --action_guidance_scale 1` |
| 单卡 | RTX PRO 6000 Blackwell |
| 单 episode 耗时 | 约 2.5–3.5 分钟（复杂任务如 stack_bowls_three 有 1200 步上限，更久） |

## 二、当前结果

| 任务 | 成功率 | 状态 |
|---|---|---|
| `adjust_bottle` | **5/5 = 100%** | ✅ 完成（更早单独跑的 5 episodes） |
| `stack_bowls_three` | **3/3 = 100%** | ⏸ 卡死在第 4 个 episode（最新批次） |
| `handover_block` | — | ⏳ 未开始 |
| `hanging_mug` | — | ⏳ 未开始 |
| `scan_object` | — | ⏳ 未开始 |
| `lift_pot` | — | ⏳ 未开始 |

## 三、遇到的问题与根治（两个卡死）

### 坑 11：ffmpeg 视频录制管道死锁（已修复 ✅）

- **现象**：评测跑完 episode1 后永久卡死，client 卡 `poll`、ffmpeg 子进程卡 `futex`（CPU 0 秒）。
- **根因**：`_base_task.py` 里帧写入是同步阻塞的 `stdin.write()`（每帧 225KB），ffmpeg stdin 管道缓冲仅 64KB，ffmpeg 一转码慢/卡住就写满管道，主流程永久阻塞。
- **修复**：改成「后台守护线程 + 有界队列」——`_set_eval_video_ffmpeg` 启动写入线程，帧写入走 `_push_video_frame`（`put_nowait` 永不阻塞，队列满丢帧），`_del_eval_video_ffmpeg` 加 `wait(timeout=10)` + `kill()` 兜底。
- **验证**：重启后顺利跑过之前卡死的 episode1→episode2。

### 坑 12：client↔server websocket 通信僵局（已修复 ✅）

- **现象**：第 3 个 episode 卡死，client 卡在 `infer()` 的 `recv()`、server 卡在 `ep_poll`，两边互等，连接 ESTAB 不断、socket 无积压。
- **根因**：`websocket_policy_server.py` 的 async handler 里，`self._policy.infer(obs)` 是**同步阻塞**的 GPU 推理（单次几分钟），直接在 asyncio 事件循环里调用 → 推理期间整个事件循环冻结，底层 websocket 事件停摆，与同步 client 失步。
- **修复**：改成 `await asyncio.to_thread(self._policy.infer, obs)`，把同步推理丢进线程池，事件循环不再冻结。
- **验证**：待重启 server 后确认。

### 共性问题

两个坑同源：**同步阻塞塞进了异步/多进程的通信链路**，与训练阶段的「坑 1 Pool hang」是同一家族。详见 [EVALUATION_TROUBLESHOOTING.md](EVALUATION_TROUBLESHOOTING.md)。

## 四、修复涉及的代码文件

| 文件 | 改动 |
|---|---|
| `RoboTwin/envs/_base_task.py` | 视频帧写入 → 后台线程 + 有界队列（备份 `.bak`） |
| `lingbot-va/wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py` | `infer` 同步调用 → `asyncio.to_thread` |

## 五、当前状态与下一步

- ⏸ client + server 仍在卡死状态（从 08-31 22:00 卡至 09-01 09:08，已跨夜）。
- 两个修复代码已就绪，**待重启 server + client 后验证**：
  1. kill 卡死的 client + server；
  2. 重启 server（重新加载模型，约几分钟）；
  3. 重启 client（从 stack_bowls_three 从头跑）。
- 验证通过后继续跑完 5 任务，再评估是否扩展到论文对齐规模（50 任务 × 100，单卡约 10 天，需并行）。

## 六、回放视频

视频文件不上传 GitHub，保存在本地：

```
/home/mosense/RoboTwin/results/stseed-10000/visualization/
├── adjust_bottle/      # 5 段，均 _True
└── stack_bowls_three/  # 含 _True / _False
```

指标 `res.json` 见 `results/` 目录。
