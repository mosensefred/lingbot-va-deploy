# 评测可视化总览（Visualization）

> 用图说话。架构、结果、卡死诊断、时间线一页看懂。

## 一、评测双环境架构（同卡竞争 CUDA × Vulkan）

```mermaid
graph LR
    subgraph srv["server（lingbot 环境）"]
        A["LingBot-VA 推理<br/>torch 2.7.1+cu128<br/>显存 ~31GB"]
    end
    subgraph cli["client（robotwin 环境）"]
        B["RoboTwin 仿真 + ACT 策略"]
        C["curobo 运动规划<br/>(CUDA)"]
        D["相机渲染<br/>(Vulkan)"]
    end
    A <-->|"websocket :29056"| B
    B --> C
    B --> D
    A -.->|"共享"| E["同一块 GPU<br/>RTX PRO 6000 96GB"]
    C -.-> E
    D -.-> E
    style E fill:#4FC3A1,color:#07130F
```

> ⚠️ 坑 13 的根因就在这张图里：CUDA（server 推理 + client curobo）和 Vulkan（client 相机渲染）挤在同一块 GPU 上，竞争导致相机 `get_picture` 偶发卡死。

## 二、各任务成功率

![成功率对比](assets/success_rate.png)

## 三、hanging_mug 判定距离诊断（一图看懂「差 2cm」）

![dist 分布](assets/dist_vs_threshold.png)

> 模型把杯子举到挂杆附近后，杯柄距离稳定停在 **~4cm**，够不到 2cm 判定阈值——「最终精确对准」能力不足，非判定 bug（判定 bug 已修复，见 EVALUATION_RESULTS.md）。

## 四、三个卡死的诊断流程（现象 → 定位 → 根因 → 修复）

### 坑 11：ffmpeg 视频录制管道死锁

```mermaid
graph TD
    A["现象：跑完 episode1 永久卡死<br/>client=poll, ffmpeg=futex(0 CPU)"] --> B["wchan 定位：ffmpeg 卡在 futex 不在 read"]
    B --> C["根因：stdin.write 同步阻塞<br/>管道 64KB 写满"]
    C --> D["修复：后台线程 + 有界队列<br/>put_nowait 永不阻塞"]
    style D fill:#4FC3A1,color:#07130F
```

### 坑 12：client↔server websocket 通信僵局

```mermaid
graph TD
    A["现象：client recv 卡 + server ep_poll 互等<br/>socket 无积压"] --> B["定位：server 主循环 ep_poll<br/>说明响应已发但 client 没收到"]
    B --> C["根因：同步 infer 阻塞 async 事件循环<br/>底层 websocket 事件停摆"]
    C --> D["修复：asyncio.to_thread + client recv 超时重连"]
    style D fill:#4FC3A1,color:#07130F
```

### 坑 13：sapien 相机渲染 get_picture 卡死

```mermaid
graph TD
    A["现象：step 426 冻结，GPU 仅 6% 空闲"] --> B["py-spy 定位：主线程卡在 camera.get_picture"]
    B --> C["根因：CUDA × Vulkan 同卡竞争<br/>渲染前的等待卡死"]
    C --> D["待修：渲染进程隔离/超时，或分卡"]
    style D fill:#E05656,color:#fff
```

## 五、评测时间线

```mermaid
gantt
    title RoboTwin 评测时间线
    dateFormat YYYY-MM-DD HH:mm
    axisFormat %m-%d %H:%M
    section 评测
    评测启动           :2026-08-31 18:44, 3m
    坑11 ffmpeg 卡死    :2026-08-31 18:47, 180m
    修复后重启          :2026-08-31 21:49, 1m
    坑12 通信卡死       :2026-08-31 22:00, 660m
    修复坑12 重启       :2026-09-01 09:14, 2m
    并发 client 冲突    :2026-09-01 09:16, 60m
    hanging_mug 诊断    :2026-09-01 15:50, 30m
```

## 六、卡死定位方法论（递进式）

```mermaid
graph TD
    A["进程卡住了"] --> B["1. CPU 时间增量：15 秒是否在涨？"]
    B -->|"不涨 = 冻结"| C["2. wchan：卡在什么系统调用？"]
    B -->|"在涨 = 只是慢"| Z["正常慢，不是卡死"]
    C --> D["3. /proc/fd + ss：等哪个 IO / 连接？"]
    D --> E["4. py-spy dump：Python 卡在哪一行？"]
    E --> F["根因定位 + 修复"]
    style F fill:#4FC3A1,color:#07130F
    style Z fill:#7B8CFF,color:#fff
```

> 关键教训：`wchan`/`/proc` 只能推断，**py-spy 才能一锤定音**（坑 13 靠它纠正了「curobo」的误判）。
