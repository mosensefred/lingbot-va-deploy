# 触觉模块插入点分析（草稿，待训练完成后整合进 DEVELOPMENT_GUIDE.md）

> 状态：代码探索阶段的准备笔记，未定稿。训练完成后重读 lingbot-va-deploy 仓库再整合。

## 一、数据流全景（从数据进到模型出）

```
[数据集 __getitem__]                    [训练 _prepare_input_dict]           [模型 forward_train]
lerobot_latent_dataset.py               train.py                           model.py
─────────────────────────────────────────────────────────────────────────────────────────────
latents  [48,9,24,20] bf16    ──加噪──► noisy_latents ──patch_embedding_mlp──┐
actions  [30,9,16,1]  f32     ──加噪──► noisy_actions ──action_embedder──────┤
text_emb [512,4096]   bf16    ─────────► text_emb      ──text_embedder───────┤ cat → hidden_states
actions_mask [30,9,16,1] bool                                               │  (30 层 WanTransformerBlock)
                                                                             ▼
                                                     split 回 latent/action
                                                     ├─ proj_out        → 视频预测 (latent_pred)
                                                     └─ action_proj_out → 动作预测 (action_pred)
```

## 二、模型架构（多模态共享 backbone）

`WanTransformer3DModel`（model.py，906 行）：
- `inner_dim = 24 heads × 128 head_dim = 3072`
- 三种输入模态共享同一个 30 层 transformer backbone（`WanTransformerBlock`）
- 每层 block：self-attn（attn1）+ cross-attn（attn2，与文本）+ ffn

各 embedding 层（`__init__` 第 601-654 行）：
| 组件 | 定义 | 输入维度 → 输出 |
|---|---|---|
| `patch_embedding_mlp` | `Linear(192, 3072)` | 视频 latent（48×1×2×2=192） |
| `action_embedder` | `Linear(30, 3072)` | 动作（30 维） |
| `condition_embedder.text_embedder` | 文本 | text_emb（4096） |
| `proj_out` | `Linear(3072, 192)` | 输出视频预测 |
| `action_proj_out` | `Linear(3072, 30)` | 输出动作预测 |

## 三、触觉插入点（关键）

触觉最自然的方式是作为**新的输入模态 token 段**，和视频 latent、动作并列进共享 backbone。需改 6 处：

### 1. 数据加载 `lerobot_latent_dataset.py` `__getitem__`（第 287-309 行）
返回 dict 新增触觉字段，例如 `tactile`（形状自定，如 [tactile_dim, F] 或 [tactile_dim]）。

### 2. 训练准备 `train.py` `_prepare_input_dict`（第 220-248 行）
把触觉放进 `input_dict`。触觉若作为 condition（不加噪），需像 text_emb 一样直接传入；若要预测触觉，才走 `_add_noise`。

### 3. 模型定义 `model.py` `__init__`（第 601-654 行）
新增 `self.tactile_embedder = nn.Linear(tactile_dim, inner_dim)`。

### 4. 输入 embedding `model.py` `_input_embed`（第 674-690 行）
加分支：
```python
elif input_type == 'tactile':
    hidden_states = self.tactile_embedder(latents)
```

### 5. 模型 forward `model.py` `forward_train`（第 705-801 行）
- cat 进 `hidden_states`（第 724 行）
- 更新 `split_list`（第 762-766 行）
- 更新 `full_grid_id` 拼接（第 730-732 行）

### 6. 配置 `configs/va_robotwin_cfg.py` 等
加 `tactile_dim` 等字段。

## 四、关键难点（待深入）

**`FlexAttnFunc.init_mask`（model.py 第 148-204 行）**：模型用 flex_attention，序列被分成 `[latent, cond_latent, action, cond_action]` 几段，掩码按段控制注意力模式。新增触觉段需要理解并修改掩码逻辑（`_get_mask_mod` 第 157 行、`_get_cross_mask_mod` 第 148 行）。这是插入触觉最复杂的部分，尚未深入。

## 五、尚未读的文件

- `wan_va_server.py`（731 行，推理服务，触觉部署也要改）
- `modules/utils.py`（load_transformer 等）
- `utils/scheduler.py`、`utils/utils.py`（get_mesh_id 等）
- `FlexAttnFunc` 的 mask 逻辑细节
