# NanoLM 模型架构

## 模型配置

### 训练使用的精确配置

| 参数 | 短上下文 (pretrain) | 长上下文 (midtrain/SFT/PPO) |
|---|---|---|
| `vocab_size` | tokenizer 词表大小 | 同左 |
| `hidden_size` | 768 | 768 |
| `intermediate_size` | 2048 | 2048 |
| `num_hidden_layers` | 12 | 12 |
| `num_attention_heads` | 12 | 12 |
| `num_key_value_heads` | 4 | 4 |
| `head_dim` | 64 (=768/12) | 64 |
| `max_position_embeddings` | 512 | 2048 |
| `original_max_position_embeddings` | — | 512 |
| `rope_type` | `default` | `yarn` |
| `rope_theta` | 10000.0 | 10000.0 |
| `attention_dropout` | 0.0 | 0.0 |
| `tie_word_embeddings` | True | True |
| `use_qk_norm` | True | True |
| `initializer_range` | 0.02 | 0.02 |

### 模型参数统计

- 总参数量: **81,809,664** (~82M)
- 模型文件大小: ~327 MB (float32)

---

## 模型组件

### 1. RMSNorm

```python
RMSNorm(hidden_size=768, eps=1e-6)
```

对 hidden_states 做 RMS 归一化：
- 转为 float32 计算
- `output = x * rsqrt(mean(x²) + eps) * weight`

### 2. MLP (SwiGLU)

```
hidden_states (768) ─┬─→ gate_proj (768→2048) → SiLU ─┐
                     │                                  ├─→ × ─→ down_proj (2048→768)
                     └─→ up_proj   (768→2048) ─────────┘
```

- gate_proj: Linear(768, 2048, bias=False)
- up_proj: Linear(768, 2048, bias=False)
- down_proj: Linear(2048, 768, bias=False)

### 3. Attention (GQA + SDPA)

**Group Query Attention**:
- 12 query heads, 4 KV heads → 每组 3 个 query head 共享 1 个 KV head
- head_dim = 64

**投影层**:
- q_proj: Linear(768, 12×64, bias=False)
- k_proj: Linear(768, 4×64, bias=False)
- v_proj: Linear(768, 4×64, bias=False)
- o_proj: Linear(12×64, 768, bias=False)

**QK Norm** (可选):
- 在 RoPE 之前对 Q、K 应用 RMSNorm(head_dim=64)

**SDPA (Scaled Dot-Product Attention)**:
- 优先使用 PyTorch `F.scaled_dot_product_attention()`
- 支持 Flash Attention（当 `is_causal=True` 且无 past KV cache）
- PyTorch ≥ 2.3 时支持 GQA 的 `enable_gqa=True`
- Fallback: 手动 GQA 扩展 + standard attention

**Attention 前向流程**:
```
hidden_states → [Q,K,V]投影 → reshape → QK Norm(可选) → RoPE → KV Cache(推理) →
SDPA(is_causal/Flash) → transpose → reshape → o_proj
```

### 4. RoPE 位置编码

三种模式：

| 模式 | 说明 | 使用场景 |
|---|---|---|
| `default` | 标准 RoPE，inv_freq = 1/(10000^(2i/d)) | pretrain (seq=512) |
| `dynamic` | 动态 NTK 缩放，序列超长时自动调整 base | — |
| `yarn` | YaRN 插值+外推混合，线性 ramp 平滑过渡 | midtrain/SFT/PPO (seq=2048) |

**YaRN 参数**:
- beta_fast=32, beta_slow=1
- attention_scaling 通过 `get_mscale(scale)` 计算

### 5. DecoderLayer

```
         ┌──────────────────────────────┐
x ──────→│ RMSNorm → Attention → + ────→│ RMSNorm → MLP (或 MoE) → + ──→ output
         └──────────────────────────────┘
```

Pre-norm 结构，残差连接在每个子层后。

### 6. MoE (混合专家，可选)

**启用条件**: `moe_config` 中所有字段都设置了，且 `layer_idx >= n_dense_layer`

**MoEGate**:
- `Linear(hidden_size, n_routed_experts)`
- Top-k 选择（softmax + topk）
- 可选 norm_topk_prob
- 辅助负载均衡 loss（seq_aux 或 batch_aux）+ z_loss

**MoE 层**:
- n_routed_experts 个独立 FFN（intermediate_size 可配置）
- n_shared_experts 个共享 FFN（输出加到所有 token）
- 训练时: repeat_interleave + 按 expert 分组计算
- 推理时: 按 expert 分配 tokens + scatter 还原

### 7. LlmModel (完整模型)

```
input_ids → Embedding(vocab_size, 768)
    → ×12 DecoderLayer (每个含 Attention + MLP/MoE)
    → RMSNorm (head_norm)
    → lm_head Linear(768, vocab_size)  [与 embedding 权重绑定]
```

**forward() 返回值**:
```python
{
    'logits':          # (batch, seq_len, vocab_size)
    'hidden_states':   # (batch, seq_len, hidden_size)
    'past_key_values': # KVCache 或 None
    'aux_loss':        # MoE 辅助 loss（无 MoE 时为 None）
}
```

**梯度检查点**: 通过 DeepSpeed checkpointing 减少显存

---

## VLM 扩展 (VlmModel)

继承 LlmModel，新增视觉编码能力。

### VLMConfig 额外字段

| 字段 | 值 |
|---|---|
| `image_tok` | `<image>` 的 token ID |
| `image_size` | 224 |
| `patch_size` | 16 |
| `tokens_per_image` | 196 |
| `vision_hidden_size` | 视觉编码器输出维度 |
| `vision_tower` | 视觉编码器 callable |

### MultiModalProjector

```
vision_outputs (B, 196, vision_hidden)
    → transpose + reshape → (B, vision_hidden, 14, 14)
    → AvgPool2d → (B, vision_hidden, tokens_per_side, tokens_per_side)
    → flatten + transpose → (B, tokens_per_image, vision_hidden)
    → RMSNorm(vision_hidden)
    → matmul(input_projection_weight) → (B, tokens_per_image, hidden_size)
```

### get_input_embeddings() 重写

用 `masked_scatter` 将 `<image>` token 的 embedding 替换为 projected image features。

---

## KV Cache

### KVCache 类

- **Pre-allocated 模式**: `max_capacity > 0` 时，首次 update 预分配 `(batch, num_heads, max_capacity, head_size)` 零张量，后续按位置写入并返回切片视图
- **Dynamic 模式**: `max_capacity == 0` 时，每次 update 用 `torch.cat` 沿 seq_len 维度拼接
- **Per-layer 存储**: key_cache 和 value_cache 是按 layer_idx 索引的列表

### 推理流程

1. Prefill: 一次性传入完整 prompt，KV Cache 保存所有层的 K、V
2. Decode: 每次只传入上一步生成的 token，KV Cache 追加新的 K、V
3. 返回的 cache 只到当前有效长度（pre-allocated 模式下通过 lengths 数组追踪）

---

## Tokenizer

### 词表

- 自训练 tokenizer，存放在 `./tokens/`
- 基于 HuggingFace `AutoTokenizer` 加载

### 特殊 Token

| Token 字符串 | 语义 |
|---|---|
| `</s>` | 句子结束 / 文档边界 |
| `<pad>` | 填充 |
| `<unk>` | 未知词 |
| `<user>` | 用户消息开头 |
| `<assistant>` | 助手消息开头 |
| `<system>` | 系统提示开头 |
| `<think>` / `</think>` | 思考过程包裹 |
| `<answer>` / `</answer>` | 回答内容包裹 |
| `<image>` | 图像占位符（VLM） |

### Chat Template

```
<system>{system}</s><user>{msg}</s><assistant>{reply}</s>
```

带思考模式:
```
<assistant><think>{思考}</think><answer>{回答}</answer></s>
```
