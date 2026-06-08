# NanoLM 3.0 — 项目总览

## 项目定位

从零构建大语言模型的全流程训练项目，目标是让个人开发者也能负担 LLM 训练成本。

- **模型规模**: 80M 参数 Dense 模型
- **硬件需求**: 4× RTX 4090，全流程训练约 7 小时
- **训练流程**: Pretrain → Midtrain → SFT → PPO 四阶段

## 项目结构

```
NanoLM/
├── app.py                  # Web 推理服务 (Bottle + SSE 流式)
├── utils.py                # 训练配置工厂函数
│
├── model/              # 模型定义
│   ├── lm.py               # Llama 风格 Decoder-only 模型
│   ├── model_config.py     # 模型配置 dataclass
│   ├── rope.py             # RoPE 位置编码（default/dynamic/yarn）
│   ├── kv_cache.py         # KV Cache（推理加速）
│   ├── attention_masks.py  # 因果注意力掩码
│   ├── sparse_moe.py       # 稀疏 MoE（混合专家）
│   └── vlm_model.py        # 多模态 VLM 扩展
│
├── train/            # 训练框架
│   ├── base_trainer.py     # 训练基类（DeepSpeed/梯度累积/断点续训）
│   ├── trainer.py          # 预训练器
│   ├── sft_trainer.py      # SFT 训练器
│   ├── dpo_trainer.py      # DPO 训练器
│   ├── ppo_trainer.py      # PPO 训练器（Policy+Value）
│   ├── grpo_trainer.py     # GRPO 训练器
│   ├── train_configs.py    # 所有训练配置 dataclass
│   ├── dataset.py          # 数据集类
│   ├── loss.py             # 损失函数（LM/KD/DPO/PPO/GRPO）
│   ├── scheduler.py        # 学习率调度器
│   ├── checkpoint.py       # 断点管理
│   ├── ds_checkpoint.py    # DeepSpeed 断点
│   ├── parallel.py         # 分布式并行（DeepSpeed/DDP/单机）
│   ├── generate_utils.py   # 自回归生成（采样/束搜索）
│   ├── eval.py             # 训练中评估
│   ├── partition_utils.py  # DeepSpeed 模型解包
│   ├── tokenizer.py        # Tokenizer 封装
│   ├── tools.py            # 工具单例
│   ├── utils.py            # 辅助函数（collate/whiten/pad）
│   └── log.py              # 日志系统
│
├── toolkit/             # 工具集（数据 + 训练 + 分析）
│   ├── cli.py              # argparse CLI 统一入口
│   ├── pipeline.py         # 管线编排器
│   ├── config.py           # 集中化配置
│   ├── file_dataset.py     # 文件数据集（自动下载/缓存管理）
│   ├── train.py            # 训练启动器
│   ├── analyze.py          # 日志可视化
│   ├── download.py         # 阶段1: ModelScope 下载
│   ├── shuffle.py          # 阶段2+5: Python 打乱
│   ├── split.py            # 阶段3: SFT 数据拆分
│   ├── preprocess.py       # 阶段4: 短/长文本分类
│   ├── encode.py           # 阶段6: Tokenize → .npy
│   ├── merge.py            # 阶段7: 合并 SFT
│   ├── encode_sft.py       # 阶段8: SFT Tokenize
│   ├── encode_ppo.py       # 阶段9: PPO Tokenize
│   ├── quality.py          # 数据质量校验
│   └── utils.py            # 共享工具
│
├── eval/                    # 模型评估与后处理
│   ├── compare.py            # SFT vs PPO 对比
│   └── extract.py            # PPO 权重提取
│
├── tokens/                 # Tokenizer 文件
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── static/index.html       # Web 聊天前端
├── bin/ppo_policy.bin      # PPO 策略模型权重
├── images/                 # 训练曲线图
└── log/                    # 训练日志
```

## 核心设计

### 模型

Llama 风格 Decoder-only Transformer：
- hidden_size=768, 12 层, 12 注意力头, 4 KV 头（GQA）
- SwiGLU FFN（intermediate_size=2048）
- RMSNorm pre-norm
- RoPE 位置编码（短上下文 default，长上下文 YaRN）
- QK Norm、词表 embedding 与 lm_head 权重绑定

### 训练框架

- 基于 DeepSpeed ZeRO-1 分布式训练
- 支持 BF16/FP16 混合精度
- Warmup + Cosine Annealing 学习率调度
- 梯度累积、梯度裁剪、断点续训
- 训练中周期性评估生成质量

### 数据管线

- 原始 JSONL → 打乱 → 按长度分类 → Tokenize → .npy
- Pretrain/Midtrain 用 flat token array（mmap 加载）
- SFT/PPO 用 object array（变长序列）
- Self-cognition 数据 20× 过采样注入 SFT
