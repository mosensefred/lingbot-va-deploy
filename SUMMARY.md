# RoboTwin 评测完整总结（2026-08-31 ~ 09-01）

## 一、评测目标与背景

- **目标**：验证 LingBot-VA 后训练模型（50k 步，`lingbot-va-posttrain-robotwin`）在 RoboTwin 仿真上的操作能力。
- **最终目标**：对齐论文基准 **Easy 92.9 / Hard 91.6**（50 任务 × 100 episodes）。
- **当前阶段**：第一批快速验证，5 任务 × 各 20 episodes（`run_eval_5tasks_20.sh`）。

## 二、环境配置

| 项 | 值 |
|---|---|
| 架构 | 双环境：server（lingbot，LingBot-VA 推理）+ client（robotwin，RoboTwin 仿真），websocket 通信 |
| server | torch 2.7.1 + cu128（Blackwell sm_120 必需），监听 :29056 |
| client | torch 2.7.1 + cu128，`eval_polict_client_openpi` |
| 策略 | ACT，`--video_guidance_scale 5 --action_guidance_scale 1` |
| 单卡 | RTX PRO 6000 Blackwell（96GB），评测时温度 ~62°C、利用率 ~50% |

## 三、完整时间线

| 时间 | 事件 |
|---|---|
| 08-31 18:44 | 第一次正式评测启动（stack_bowls_three × 20） |
| 08-31 18:47 | **卡死**（坑 11：ffmpeg 管道死锁） |
| 08-31 21:49 | 修复 ffmpeg 后重启评测 |
| 08-31 22:00 | **再次卡死**（坑 12：通信僵局），卡一整夜 |
| 09-01 09:08 | 诊断通信僵局，修复（`asyncio.to_thread`） |
| 09-01 09:14 | 重启 server + client |
| 09-01 09:16 | 发现**并发 client 冲突**（漏 kill 旧脚本），清理 |
| 09-01 09:23~10:43 | hanging_mug 跑 14 个 episode，全失败（0/14） |

## 四、各任务结果（截至 09-01 10:43）

| 任务 | 成功率 | 状态 |
|---|---|---|
| `adjust_bottle` | **5/5 = 100%** | ✅ 完成（单独跑的 5 episodes） |
| `stack_bowls_three` | 3/3 = 100% | ⏸ 卡死前结果，待补跑 |
| `handover_block` | — | ⏳ 待补跑（并发冲突假失败） |
| `hanging_mug` | 0/14 = 0% | 🟢 进行中（还剩 6 个） |
| `scan_object` | — | ⏳ 未开始 |
| `lift_pot` | — | ⏳ 未开始 |

## 五、三个问题的完整记录

### 坑 11：ffmpeg 视频录制管道死锁（已修复 ✅）

- **现象**：跑完 episode1 后永久卡死，client 卡 `poll`、ffmpeg 子进程卡 `futex`（CPU 0 秒），2.5 小时无进展。
- **根因**：`_base_task.py` 里 `self.eval_video_ffmpeg.stdin.write(rgb.tobytes())` 同步阻塞写帧（每帧 225KB），ffmpeg stdin 管道缓冲仅 64KB，转码一慢就写满管道，主流程永久阻塞。
- **修复**：后台守护线程 + 有界队列（`queue.Queue(maxsize=200)`）；`_push_video_frame` 用 `put_nowait`（队列满丢帧、永不阻塞）；`_del_eval_video_ffmpeg` 加 `wait(timeout=10)` + `kill()` 兜底。
- **验证**：重启后顺利跑过之前卡死的 episode1→episode2。

### 坑 12：client↔server websocket 通信僵局（已修复 ✅）

- **现象**：第 3 个 episode 卡死，client 卡在 `infer()` 的 `recv()`、server 卡在 `ep_poll`，两边互等，连接 ESTAB 不断、socket Recv-Q/Send-Q 均 0。
- **根因**：`websocket_policy_server.py` 的 async `_handler` 里，`self._policy.infer(obs)` 是同步阻塞的 GPU 推理（单次几分钟），直接在 asyncio 事件循环里调用，推理期间事件循环冻结、底层 websocket 事件停摆，与同步 client 失步。
- **修复**：改成 `await asyncio.to_thread(self._policy.infer, obs)`，把同步推理丢进线程池。
- **验证**：待重启 server 后确认。

### 并发 client 踩坏 VAE 状态（已清理）

- **现象**：重启后 stack_bowls_three/handover_block 快速失败，server 报 `RuntimeError: The size of tensor a (8) must match the size of tensor b (4)`。
- **根因**：kill 卡死 client 时**漏 kill 旧脚本**，旧脚本 for 循环继续启动新 client，与我的新 client **同时连同一个 server**，并发调用 `infer` 踩坏 `streaming_vae` 共享状态。
- **处理**：清理旧脚本 + client，只保留一个 client；受影响的两个任务需补跑。

## 六、hanging_mug 0% 失败原因（双重原因）

**判定 bug（已修）+ 模型精细对准能力不足**，两者叠加：

1. **判定 bug**：`check_success` 误用「中点」`(rack_pose + rack_function_pose)/2` 而非「挂杆末端」，修复前差 10cm 必挂；
2. **修复后仍失败**：诊断日志显示杯柄离挂杆稳定停在 **4cm**（判定要求 2cm），模型能举到附近、但套不上挂杆——精细对准精度不足。

详见 [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) 第四章。

## 七、代码改动详情

**1. `RoboTwin/envs/_base_task.py`**（视频帧写入异步化）：

```python
def _set_eval_video_ffmpeg(self, ffmpeg):
    self.eval_video_ffmpeg = ffmpeg
    self._eval_video_queue = queue.Queue(maxsize=200)
    def _writer():
        while True:
            frame = self._eval_video_queue.get()
            if frame is None: break
            self.eval_video_ffmpeg.stdin.write(frame)
    self._eval_video_writer = threading.Thread(target=_writer, daemon=True).start()

def _push_video_frame(self, frame_bytes):
    try: self._eval_video_queue.put_nowait(frame_bytes)  # 满则丢帧
    except queue.Full: pass
```

**2. `websocket_policy_server.py`**（推理不阻塞事件循环）：

```python
# 改前
action = self._policy.infer(obs)
# 改后
action = await asyncio.to_thread(self._policy.infer, obs)
```

## 八、当前状态与下一步

- 🟢 评测进行中：hanging_mug 跑第 15/20 个 episode。
- ⏳ 待办：
  1. 跑完 hanging_mug → scan_object → lift_pot；
  2. 补跑 stack_bowls_three + handover_block（`run_eval_2tasks.sh` 已备好）；
  3. 评估扩展论文对齐规模（50 任务 × 100，单卡约 10 天，需并行）。

## 九、关联文档

- [EVALUATION_LOG.md](EVALUATION_LOG.md) — 评测记录
- [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) — 结果 + 失败分析
- [EVALUATION_TROUBLESHOOTING.md](EVALUATION_TROUBLESHOOTING.md) — 踩坑实录（坑 1~12）
