# NanoLM 工具集

数据预处理、训练启动、日志分析的一站式 CLI。

## 快速开始

```bash
# 数据预处理
python -m toolkit full

# 启动训练
python -m toolkit train pretrain

# 查看日志
python -m toolkit analyze loss

# 列出所有命令
python -m toolkit list
```

## CLI 命令

### 数据处理

| 命令 | 说明 |
|---|---|
| `full` | 运行完整 9 阶段管线 |
| `download` | 从 ModelScope 下载原始数据集 |
| `shuffle` | 打乱所有原始 JSONL（Python 原生，无需 terashuf） |
| `split` | 拆分 SFT-2048 为 SFT 和 pretrain 两部分 |
| `preprocess` | 按文本长度分为短/长两类 |
| `shuffle2` | 再次打乱分类后的预训练数据 |
| `encode` | Tokenize pretrain/midtrain 数据 → `.npy` |
| `merge` | 合并 SFT 数据文件 |
| `encode_sft` | Tokenize SFT 数据 + self-cognition 注入 |
| `encode_ppo` | Tokenize PPO 提示数据 |
| `validate` | 校验输出文件的完整性和数据质量 |
| `list` | 列出所有可用阶段 |

### 选项

```bash
# 强制重新运行（覆盖已有输出）
python -m toolkit full --force

# 跳过质量校验
python -m toolkit full --skip-quality

# 指定打乱种子（确保可复现）
python -m toolkit full --seed 123

# 单独运行某个阶段
python -m toolkit shuffle
python -m toolkit encode
```

### 训练启动

| 命令 | 说明 |
|---|---|
| `train pretrain` | 启动预训练 |
| `train midtrain` | 启动长文适应 |
| `train sft` | 启动指令微调 |
| `train ppo` | 启动强化学习对齐 |

```bash
python -m toolkit train pretrain
python -m toolkit train sft
python -m toolkit train ppo --extra-args "--some-deepspeed-flag"
```

自动检测 DeepSpeed，有则用 DeepSpeed 分布式训练，无则退化为单机训练。

### 日志分析

| 命令 | 说明 |
|---|---|
| `analyze loss` | 绘制 loss 等训练指标曲线 |
| `analyze lr` | 绘制学习率曲线 |

```bash
python -m toolkit analyze loss
python -m toolkit analyze loss -f ./other_log.txt
python -m toolkit analyze lr
```

需要 `matplotlib`，未安装时会提示。

## 管线流程

```
ModelScope 下载 → data/raw/*.jsonl
  ↓ shuffle
data/tmp/shuffle_*.jsonl
  ↓ split
data/tmp/shuffle_sft_mini_2048.jsonl + shuffle_pretrain_2048.jsonl
  ↓ preprocess
data/tmp/pretrain_data_short.jsonl + pretrain_data_long.jsonl
  ↓ shuffle2
data/tmp/shuffle_pretrain_data_short.jsonl + shuffle_pretrain_data_long.jsonl
  ↓ encode
data/pretrain_data_0.npy + pretrain_data_1.npy  (短文本)
data/midtrain_data_0.npy                       (长文本)
  ↓ merge
data/tmp/sft_data.jsonl
  ↓ encode_sft
data/sft_data.npy         (含 self-cognition)
  ↓ encode_ppo
data/ppo_data.npy
```

## 可配置参数

通过环境变量覆盖默认值：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `NANOLM_DATA_DIR` | `./data/` | 数据根目录 |
| `NANOLM_SHORT_THRESHOLD` | `768` | 短/长文本分界长度 |
| `NANOLM_MAX_SFT_LEN` | `2048` | SFT 最大 token 数 |
| `NANOLM_SELF_COG_MULT` | `20` | self-cognition 数据重复倍数 |
| `NANOLM_SHUFFLE_SEED` | `42` | 打乱随机种子 |

```bash
# 示例：调大 self-cognition 权重
NANOLM_SELF_COG_MULT=50 python -m toolkit encode_sft
```

## 数据质量校验

```bash
# 校验所有输出文件
python -m toolkit validate
```

校验内容：
- 文件能否正常加载
- flat array：token 总数、min/max 值是否在合法范围
- object array：序列数、token 总数、seq 长度分布
- dtype 是否匹配预期

## 目录结构

```
toolkit/
  __init__.py       # 包标记 + 公共 API
  __main__.py       # python -m toolkit 入口
  cli.py            # argparse CLI
  pipeline.py       # 管线编排器
  config.py         # 集中化配置
  download.py       # ModelScope 下载
  shuffle.py        # Python 原生打乱
  split.py          # 数据拆分
  preprocess.py     # 文本分类
  encode.py         # pretrain/midtrain tokenize
  merge.py          # SFT 合并
  encode_sft.py     # SFT tokenize
  encode_ppo.py     # PPO tokenize
  quality.py        # 质量校验
  utils.py          # 共享工具
```
