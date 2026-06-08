# NanoLM 数据预处理

## 快速开始

```bash
# 完整管线
python -m toolkit full

# 单独阶段
python -m toolkit download
python -m toolkit shuffle
python -m toolkit encode

# 列出所有阶段
python -m toolkit list

# 跳过质量校验 / 强制重新运行
python -m toolkit full --skip-quality --force --seed 42
```

---

## 数据流

```
ModelScope 下载
     │
     ├── gongjy/minimind_dataset
     └── swift/self-cognition
            │
            ▼
data/raw/*.jsonl              ← 原始 JSONL 文件
            │
            ▼ [shuffle]
data/tmp/shuffle_*.jsonl      ← 打乱后的文件
            │
            ▼ [split]
shuffle_sft_mini_2048.jsonl   ← SFT 部分
shuffle_pretrain_2048.jsonl   ← 预训练部分
            │
            ▼ [preprocess] 按文本长度分类
pretrain_data_short.jsonl     ← ≤768 字符（pretrain）
pretrain_data_long.jsonl      ← >768 字符（midtrain）
            │
            ▼ [shuffle2]
shuffle_pretrain_data_short.jsonl
shuffle_pretrain_data_long.jsonl
            │
            ▼ [encode]
pretrain_data_0.npy           ← 短文本 token 序列（分 2 块）
pretrain_data_1.npy
midtrain_data_0.npy           ← 长文本 token 序列
            │
            ▼ [merge]
sft_data.jsonl                ← SFT 合并文件
            │
            ▼ [encode_sft]
sft_data.npy                  ← SFT token 数组（含 self-cognition×20）
            │
            ▼ [encode_ppo]
ppo_data.npy                  ← PPO prompt token 数组
```

---

## 各阶段详解

### 阶段 1: download — 下载原始数据

从 ModelScope 下载两个数据集到 `data/raw/`：

| 数据集 | 内容 |
|---|---|
| `gongjy/minimind_dataset` | 预训练 HQ 文本、SFT 对话（512/1024/2048）、R1 混合数据、RL 数据 |
| `swift/self-cognition` | 身份认知对话（你是谁/你叫什么） |

### 阶段 2: shuffle — 打乱

用 Python `random.shuffle`（确定性 seed=42）替代 Linux 专属的 `terashuf`。

**特点**:
- 全内存操作（MB 级数据完全可放入内存）
- 可通过 `--seed` 指定种子确保可复现
- macOS/Linux/Windows 通用

### 阶段 3: split — 拆分 SFT

将 `shuffle_sft_2048.jsonl` 按 `shuffle_sft_mini_512.jsonl` 的行数拆分为两部分：
- 前 N 行 → SFT 微调数据
- 剩余 → 混入预训练数据

### 阶段 4: preprocess — 文本分类

1. 读取 `shuffle_pretrain_hq.jsonl` → 直接按长度分类
2. 读取 4 个 SFT 文件 → 通过 `sft_to_text()` 转为纯文本
   - 解析 `<think>` / `<answer>` 标签
   - 替换模型名称占位符（MiniMind → NanoLM）
3. 按 `SHORT_TEXT_THRESHOLD=768` 分为短/长两类

### 阶段 5: shuffle2 — 再次打乱

对分类后的 `pretrain_data_short.jsonl` 和 `pretrain_data_long.jsonl` 再次打乱。

### 阶段 6: encode — Tokenize 为 .npy

**流程**:
1. 批量读取 JSONL（PRETRAIN_BATCH_SIZE=50000 行）
2. 每行追加 `</s>` 结束符
3. `tokenizer.batch_encode()` 批量编码
4. 写入临时 `.bin` 文件
5. 包装为 `.npy` 格式（`np.lib.format.write_array_header_1_0`）

**输出格式**: flat token array
- dtype: `np.uint16`（vocab < 65535）或 `np.uint32`
- 支持 `np.load(mmap_mode='r')` 内存映射加载
- pretrain_data 分 2 块，midtrain_data 不分块

### 阶段 7: merge — 合并 SFT

将 `shuffle_sft_mini_2048.jsonl` 和 `shuffle_sft_mini_512.jsonl` 合并为 `sft_data.jsonl`。

### 阶段 8: encode_sft — SFT Tokenize

**流程**:
1. 逐行读取对话 JSON
2. 构建 `[system, user, assistant]` 对话模板
3. `tokenizer.apply_chat_template(tokenizer=True)` 直接返回 token IDs
4. 过滤长度超过 2048 的序列
5. 注入 self-cognition 数据 ×20
6. `sklearn.utils.shuffle` 打乱
7. 保存为 `sft_data.npy`（object array: `np.array(tokens, dtype=object)`）

**Self-Cognition 数据**:
```json
{"query": "你叫什么名字？", "response": "我叫{{NAME}}"}
{"query": "你是谁开发的？", "response": "我是由{{AUTHOR}}开发的"}
```
替换后: `{{AUTHOR}}` → `QB`, `{{NAME}}` → `NanoLM`

### 阶段 9: encode_ppo — PPO Tokenize

1. 读取 `rlaif-mini.jsonl`
2. 提取 user content → 构建 `[system, user]` 模板 → `apply_chat_template`
3. 追加 `<assistant>` token → encode
4. 保存为 `ppo_data.npy`

---

## 输出文件与训练对应

| 文件 | 类型 | 训练阶段 | 数据集类 |
|---|---|---|---|
| `pretrain_data_0.npy` | flat uint16/32 | Pretrain | `PretrainFileDataset` |
| `pretrain_data_1.npy` | flat uint16/32 | Pretrain | `PretrainFileDataset` |
| `midtrain_data_0.npy` | flat uint16/32 | Midtrain | `MidtrainFileDataset` |
| `sft_data.npy` | object array | SFT | `SFTFileDataset` |
| `ppo_data.npy` | object array | PPO | `PPODataset` |

---

## 文件数据集系统

`file_dataset.py` 提供自动下载和缓存管理：

- **自动下载**: 文件不存在时从 ModelScope `qibin0506/NanoLM-3.0-data` 下载
- **预取**: 后台线程预下载下一个文件
- **自动清理**: 处理完的文件自动删除以节省磁盘空间

---

## 可配置参数

通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NANOLM_DATA_DIR` | `./data/` | 数据根目录 |
| `NANOLM_SHORT_THRESHOLD` | `768` | 短/长文本分界（字符数） |
| `NANOLM_MAX_SFT_LEN` | `2048` | SFT 序列最大长度 |
| `NANOLM_SELF_COG_MULT` | `20` | self-cognition 重复倍数 |
| `NANOLM_SHUFFLE_SEED` | `42` | 打乱种子 |

---

## 数据集类

### PretrainDataset

支持三种格式：
- `.npy`: `np.load(mmap_mode='r')`，内存高效
- `.jsonl`: 即时 tokenize
- `.pkl`: pickle 反序列化

采样方式: stride 滑动窗口，`len = (total - block_size) // stride + 1`

### SFTDataset

- `.npy`: 变长 token 序列数组
- `.jsonl`: 对话 dict，`__getitem__` 时 `apply_chat_template`
- `.pkl`: 预 tokenize 序列

可选 VLM 支持: 通过 `image_tags_file_path` 加载图像标签

### DPODataset

返回 `{'chosen': [tokens], 'rejected': [tokens]}` 配对数据。

### RLDataset

返回 `{'prompt': tensor, 'answer': tensor | None}`，用于 PPO/GRPO。

---

## 数据质量校验

```bash
python -m toolkit validate
```

校验内容：
- 文件完整性（`.npy` 能否正常加载）
- 数据类型匹配（flat array vs object array）
- Token 值范围合法性
- 序列长度分布统计
- 总 token 数统计
