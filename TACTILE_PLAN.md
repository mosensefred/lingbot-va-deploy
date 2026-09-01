# 触觉二次开发方案（TACTILE_PLAN）

> 状态：规划中。基于 `TACTILE_ANALYSIS.md`（6 处改动点）扩展，聚焦「无传感器时怎么启动」。

## 一、战略判断

- **触觉 × 世界模型是 MoSense 的核心差异化**——这是唯一能形成自主专利 + 技术壁垒的方向，比跑分更有价值。
- **当前无触觉传感器硬件**，但这不阻塞启动：先用**仿真触觉（proxy）**跑通整条管道，验证「触觉引导预测」是否有效，硬件到位后再替换数据源（代码管道不变）。

## 二、触觉插入的两条路线

```mermaid
graph TD
    A["触觉插入"] --> B["路线 A：作为条件输入<br/>（condition，不加噪）"]
    A --> C["路线 B：作为预测目标<br/>（加噪，要预测触觉）"]
    B --> D["✅ 首选：简单，像 text_emb 一样<br/>触觉序列直接引导预测"]
    C --> E["暂不做：复杂"]
    style D fill:#4FC3A1,color:#07130F
```

## 三、仿真触觉 proxy（无传感器的替代）

RoboTwin 物理引擎（SAPIEN）能提供的「触觉代理信号」：

| 可用信号 | 代码位置 | 作为触觉的含义 |
|---|---|---|
| 接触对 `scene.get_contacts()` | `_base_task.py` | 接触点 / 法线 / 接触力 |
| 夹爪-物体接触检测 | `get_gripper_actor_contact_position` | 抓取是否稳固 |
| 两物体接触判断 | `check_actors_contact` | 物体是否碰桌/碰架 |

→ 这些可拼成一条「伪触觉」序列（如 `[接触标志, 接触力, 接触位置]`），替代真实传感器，先跑通管道。

```mermaid
graph LR
    A["RoboTwin 物理引擎<br/>scene.get_contacts()"] --> B["提取触觉代理<br/>接触力/位置/标志"]
    B --> C["伪触觉序列<br/>替代真实传感器"]
    C --> D["喂给世界模型<br/>触觉作为条件"]
    style C fill:#7B8CFF,color:#fff
```

## 四、分阶段路线图

```mermaid
graph TD
    S0["阶段0：定触觉数据格式<br/>（proxy：接触力[N]+接触标志）"] --> S1["阶段1：数据管道<br/>__getitem__ 加 tactile 字段<br/>_prepare_input_dict"]
    S1 --> S2["阶段2：模型<br/>tactile_embedder + _input_embed<br/>+ forward_train 拼接"]
    S2 --> S3["阶段3：攻克 mask 难点<br/>FlexAttnFunc 加触觉段"]
    S3 --> S4["阶段4：训练 + 验证<br/>触觉是否改善预测"]
    S4 --> S5["阶段5：推理部署<br/>wan_va_server.py 加触觉"]
    S5 --> S6["有真实传感器后<br/>替换数据源（管道不变）"]
    style S3 fill:#E05656,color:#fff
    style S6 fill:#4FC3A1,color:#07130F
```

## 五、关键前置（动手前必须定）

1. **触觉代理的具体格式**：接触力向量 `[N]`？接触点位置 `[3]`？还是两者拼接 `[N+3]`？
2. **时序对齐**：触觉采样帧和相机帧、动作帧怎么对齐（同帧率？还是插值）？
3. **作为 condition 还是加噪**：proxy 建议走 condition（不加噪），最简。

## 六、关联文档

- `TACTILE_ANALYSIS.md` — 6 处代码改动点的原始分析
- `EVALUATION_RESULTS.md` — 评测能力画像（粗放强、精细弱，触觉正可补精细短板）
