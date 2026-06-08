# NanoLM 训练管线

## 训练四阶段

```
Pretrain → Midtrain → SFT → PPO
  (512)     (2048)    (2048)  (2048)
```

---

## 阶段 1: Pretrain（预训练）

**目标**: 海量无标注文本学习基础知识
**脚本**: `train_pretrain.py`
**训练器**: `Trainer`
**数据集**: `pretrain_data_0.npy`, `pretrain_data_1.npy`（短文本 token 序列）
**模型配置**: max_position_embeddings=512, rope_type='default'

### 训练参数

| 参数 | 值 | 说明 |
|---|---|---|
| `n_epochs` | 1 | 训练轮数 |
| `real_batch_size` | 76 | 每 GPU 每步样本数 |
| `max_lr` | 6e-4 | 峰值学习率 |
| `initial_lr` | 1e-7 | 初始学习率（warmup 起点） |
| `min_lr` | 6e-5 | 最低学习率（max_lr × 0.1） |
| `gradient_accumulation_steps` | 3 | 梯度累积步数 |
| `effective_batch_size` | 76 × 4 GPU × 3 = 912 | 有效批次大小 |
| `warmup_iters` | ~8541 | warmup 步数（10% 总步数） |
| `cosine_annealing_batches` | ~76870 | cosine 衰减步数（90% 总步数） |
| `all_data_size` | 6,532,762 | 总 token 数 |
| `eval_batch_interval` | 100 | 每 100 步评估一次 |
| `optim_type` | adam | 优化器类型 |
| `weight_decay` | 0.01 | 权重衰减 |
| `betas` | (0.9, 0.999) | Adam betas |
| `gradient_clipping` | 1.0 | 梯度裁剪阈值 |

### DeepSpeed 配置

| 参数 | 值 |
|---|---|
| ZeRO stage | 1 |
| BF16 | 启用（优先）|
| FP16 | 降级备用 |

### 学习率曲线

```
lr
 │
 │  max_lr=6e-4  ┌──────────────────────────────┐
 │              ╱                                ╲
 │             ╱                                  ╲___
 │  initial_lr=1e-7                      min_lr=6e-5
 └─────────────────────────────────────────────────→ steps
       warmup (~10%)         cosine decay (~90%)
```

---

## 阶段 2: Midtrain（长文适应）

**目标**: 扩展上下文窗口至 2048，适应长文本
**脚本**: `train_midtrain.py`
**训练器**: `Trainer`
**数据集**: `midtrain_data_0.npy`（长文本 token 序列）
**模型配置**: max_position_embeddings=2048, rope_type='yarn', original_max=512
**输入检查点**: `last_checkpoint.bin`（来自 pretrain）

### 训练参数

| 参数 | 值 |
|---|---|
| `n_epochs` | 1 |
| `real_batch_size` | 18 |
| `max_lr` | 8e-5 |
| `initial_lr` | 1e-7 |
| `min_lr` | 8e-6 |
| `all_data_size` | 1,147,192 |
| `gradient_accumulation_steps` | 3 |
| `eval_batch_interval` | 100 |

---

## 阶段 3: SFT（指令微调）

**目标**: 赋予对话能力，遵循指令格式
**脚本**: `train_sft.py`
**训练器**: `SFTTrainer`
**数据集**: `sft_data.npy`（对话 token 序列，含 self-cognition）
**模型配置**: max_position_embeddings=2048, rope_type='yarn'
**输入检查点**: `last_checkpoint.bin`（来自 midtrain）

### 训练参数

| 参数 | 值 |
|---|---|
| `n_epochs` | 1 |
| `real_batch_size` | 15 |
| `max_lr` | 2e-5 |
| `initial_lr` | 1e-7 |
| `min_lr` | 2e-6 |
| `all_data_size` | 2,430,781 |
| `gradient_accumulation_steps` | 3 |
| `eval_batch_interval` | 100 |
| `mask_prompt` | True |

### Prompt Masking

SFT 阶段只对 assistant 的回复部分计算 loss：
- `<system>` prompt 部分 → mask
- `<user>` 消息部分 → mask
- `<assistant>` 标签本身 → mask
- `<think>` / `<answer>` 内容和实际回复文本 → 计算 loss

### Self-Cognition 注入

- 从 `self_cognition.jsonl` 读取身份对话数据
- 替换 `{{AUTHOR}}` → `QB`，`{{NAME}}` → `NanoLM`
- 重复 20× 注入 SFT 数据集中

---

## 阶段 4: PPO（强化学习对齐）

**目标**: 通过 Reward Model 优化回复质量，对齐人类偏好
**脚本**: `train_ppo.py`
**训练器**: `PPOTrainer`
**数据集**: `ppo_data.npy`（prompt 数据）
**模型配置**: max_position_embeddings=2048, rope_type='yarn'
**输入检查点**: `last_checkpoint.bin`（来自 SFT）
**参考检查点**: `sft.bin`（SFT 结果备份）

### 双模型架构

```
Policy Model (trainable)        Value Model (trainable)
       │                               │
       ├── LlmModel ───────────────→ BaseModel
       │                               │
       └── (生成 rollout)              └── ValueHead(768→1)
                                               │
                                      V(s) ∈ R (标量值)
Reference Model (frozen)
       │
       └── LlmModel (SFT checkpoint)
            → 计算 KL 散度约束
```

### 训练参数

| 参数 | 值 |
|---|---|
| `n_epochs` | 2 |
| `real_batch_size` | 50 |
| `gradient_accumulation_steps` | 10 |
| `eval_batch_interval` | 10 |

### PPO 优化参数

| 参数 | 值 | 说明 |
|---|---|---|
| `ppo_epochs` | 2 | 每批 rollout 的 PPO 更新轮数 |
| `ppo_batch_size` | 5 | PPO minibatch 大小 |
| `clip_eps` | 0.1 | PPO clip 范围 |
| `vf_coef` | 0.5 | Value loss 权重 |
| `kl_beta` | 0.01 | KL 散度惩罚系数 |
| `kl_estimator` | k3 | KL 估计器: `exp(logr)-1-logr` |
| `gamma` | 1.0 | 折扣因子 |
| `lam` | 0.95 | GAE λ 参数 |

### 学习率（Policy 与 Value 分离）

| 参数组 | initial_lr | 调度器 |
|---|---|---|
| Policy | 1e-5 | 常量（无调度）|
| Value | 5e-5 | 常量（无调度）|

### 生成参数（Rollout）

| 参数 | 值 |
|---|---|
| `gen_max_seq_len` | 2048 |
| `gen_temperature` | 1.0 |
| `gen_p` | 0.9 (top-p 核采样) |
| `gen_k` | None（不使用 top-k）|

### 奖励计算

使用 `Shanghai_AI_Laboratory/internlm2-1_8b-reward` 作为 Reward Model。

**奖励组成**:
1. **EOS 惩罚**: 回复不以 `</s>` 结尾时施加 -5.0 惩罚
2. **RM 分数**: Reward Model 打分 × 0.5 权重，截断到 [-5.0, 5.0]
3. **KL 惩罚**: `-kl_beta × KL(policy || reference)` 每 token

**奖励归一化**:
- 方法: RunningMeanStd（Welford 算法 + all-reduce）
- 逐 batch 更新均值和方差

### GAE 优势估计

```python
# 反向遍历每个 token
for t in reversed(range(seq_len)):
    next_values = 0 if done else values[t+1]
    delta = rewards[t] + gamma * next_values - values[t]
    advantage[t] = delta + gamma * lam * advantage[t+1]

returns = advantages + values  # 用于 value loss
```

### PPO Loss

**Actor Loss**:
```
ratio = exp(log_probs - old_log_probs)
surr1 = ratio × advantages
surr2 = clamp(ratio, 1-eps, 1+eps) × advantages
actor_loss = -mean(min(surr1, surr2))
```

**Value Loss** (clipped):
```
v_clipped = old_values + clamp(v - old_values, -eps, eps)
value_loss = 0.5 × mean(max((v-returns)², (v_clipped-returns)²))
```

**Total**:
```
loss = actor_loss + vf_coef × value_loss
```

---

## PPO vs SFT 效果

| 阶段 | 平均得分 | 说明 |
|---|---|---|
| SFT | -0.73 | 初步具备对话能力，但回复质量一般 |
| PPO | +0.82 | PPO 对齐后显著提升，更符合人类偏好 |

---

## Checkpoint 管理

### 转换流程

每个训练阶段结束后，DeepSpeed checkpoint 需转为标准 `.bin` 文件：

```bash
cd ./ckpt_dir
python3 zero_to_fp32.py ./ ../
cd ..
mv pytorch_model.bin last_checkpoint.bin
```

PPO 完成后还需提取 policy 权重：
```bash
python3 extract_ppo_result.py  # 输出 ppo_policy.bin
```

### 断点续训

- `ckpt.pth`: 模型+优化器状态
- `steps.pt`: epoch/file_idx/batch_idx + RNG 状态
- `CKPT_MAX_TO_KEEP=2`: 最多保留 2 个旧 checkpoint

---

## 监控

```bash
# 查看训练指标
vis_log ./log/log.txt

# 查看学习率变化
vis_lr ./log/lr.txt
```

日志格式：
```
epoch: 0, file: 1/2, batch: 100/1000 -> loss: 2.345, moe_aux_loss: 0.001
```
