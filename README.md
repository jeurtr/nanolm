<br />

## 项目简介

**NanoLM** 是一个面向个人开发者的微型语言模型训练项目。不需要几百张显卡，也不需要几周的等待时间——在 4 张 RTX 4090 上，一个完整的训练周期只需要一个下午。

这个项目的目标是降低 LLM 训练的门槛。所有代码完全开源，每个模块独立可读，方便你理解、修改和复用。

### 核心特点

- **个人可承受的成本**：82M 参数 Dense 架构，4×RTX 4090 全流程约 7 小时。
- **覆盖完整训练链路**：Pretrain → Midtrain → SFT → PPO，从随机权重到对齐人类偏好。
- **模块化设计**：模型定义、训练框架、数据管线、推理服务各自独立，按需取用。
- **自带 Web 交互**：训练完成后一键启动对话服务，直接在你的浏览器里和模型对话。

### 版本

当前版本 **0.1.1**，82M Dense 模型，支持 PPO 对齐训练。

## 最近更新

- 采用自训练 8192 词表 SentencePiece Tokenizer，中文编码效率更高。
- 训练框架重构，支持断点续训和自动 checkpoint 管理。
- 去除思考链模式，专注标准对话流程。
- 新增 GRPO 和 DPO 训练支持。
- CLI 统一入口，数据处理、训练、评估、Web 服务一个命令搞定。

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- 推荐 NVIDIA GPU（至少 8GB 显存），也支持 Apple Silicon MPS 和纯 CPU 推理

### 安装与启动

```bash
git clone <repo-url>
cd NanoLM
pip install -r requirements.txt
python -m toolkit web
```

首次运行会自动下载模型权重。启动后在浏览器打开 `http://localhost:8080` 即可对话。

如果需要生产环境部署，可启用 Waitress：

```bash
USE_WAITRESS=1 python -m toolkit web
```

## 训练流程

### 数据预处理

从 ModelScope 下载 [Minimind Dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset)，自动完成清洗、分桶和序列化。

```bash
python -m toolkit full      # 一键执行完整预处理管线
python -m toolkit list      # 查看所有可用的数据处理子步骤
```

### 四阶段训练

| 阶段       | 上下文长度 | 目标                   |
| -------- | ----- | -------------------- |
| Pretrain | 512   | 在海量文本上学习语言的基础知识      |
| Midtrain | 2048  | 将上下文窗口从 512 扩展到 2048 |
| SFT      | 2048  | 用对话数据教模型进行多轮交互       |
| PPO      | 2048  | 通过奖励模型让输出更符合人类偏好     |

每个阶段训练完成后，需要将 DeepSpeed checkpoint 转换为标准 PyTorch 权重文件，供下一阶段加载。

#### Pretrain — 预训练

```bash
python -m toolkit train pretrain

# 训练完成后转换权重
cd ckpt_dir && python zero_to_fp32.py . ../ && cd ..
mv pytorch_model.bin last_checkpoint.bin
```

#### Midtrain — 长上下文适应

```bash
python -m toolkit train midtrain

# 自动加载上一阶段的 last_checkpoint.bin
cd ckpt_dir && python zero_to_fp32.py . ../ && cd ..
mv pytorch_model.bin last_checkpoint.bin
```

#### SFT — 监督微调

```bash
python -m toolkit train sft
cd ckpt_dir && python zero_to_fp32.py . ../ && cd ..
mv pytorch_model.bin last_checkpoint.bin
cp last_checkpoint.bin sft.bin
```

#### PPO — 强化学习对齐

此阶段同时维护 Policy 网络和 Value 网络，使用 InternLM2-1.8B 作为奖励模型。

```bash
python -m toolkit train ppo
cd ckpt_dir && python zero_to_fp32.py . ../ && cd ..
mv pytorch_model.bin ppo.bin

# 从联合权重中提取纯 Policy 部分
python -m toolkit eval extract
```

### 训练监控

训练日志输出到 `./log/` 目录，可以通过 CLI 查看可视化曲线：

```bash
python -m toolkit analyze loss      # loss 下降曲线
python -m toolkit analyze lr        # 学习率变化曲线
```

### 效果评估

PPO 训练完成后，可以对比 SFT 和 PPO 模型的回复质量：

```bash
python -m toolkit eval compare
```

| 阶段  | 平均奖励分数 | 说明               |
| --- | ------ | ---------------- |
| SFT | -0.73  | 能对话，但质量不稳定       |
| PPO | +0.82  | 经 RL 对齐后回复质量明显提升 |

## 项目结构

```
NanoLM/
├── cli.py                  # 统一命令行入口
├── nanolm/                 # 核心包（API、推理服务、设备工具、配置工具）
├── model/                  # 模型定义（Llama 风格 Decoder-only Transformer）
├── train/                  # 训练框架（Trainer、损失函数、并行策略、数据加载）
├── toolkit/                # 数据处理管线与训练启动器
├── eval/                   # 模型评估与权重后处理
├── configs/                # YAML 训练配置文件
├── tests/                  # 单元测试
├── docs/                   # 详细文档
├── static/                 # Web 对话界面
└── tokens/                 # Tokenizer 模型文件
```

## 致谢

本项目在开发和训练过程中参考了以下开源成果，谨此致谢。

### 训练数据

| 数据集                                                                             | 来源         | 用途               |
| ------------------------------------------------------------------------------- | ---------- | ---------------- |
| [Minimind Dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset)  | ModelScope | 预训练、SFT、RL 全阶段数据 |
| [Cortex-3.0-data](https://www.modelscope.cn/datasets/qibin0506/Cortex-3.0-data) | ModelScope | 预处理结果归档与分发       |

### 模型与架构

| 模型                                                                  | 来源        | 用途                         |
| ------------------------------------------------------------------- | --------- | -------------------------- |
| [InternLM2-1.8B Reward Model](https://github.com/InternLM/InternLM) | 上海人工智能实验室 | PPO 训练中的奖励信号               |
| Llama 架构                                                            | Meta      | GQA、SwiGLU、RMSNorm、RoPE 参考 |

### 论文与技术参考

| 技术                                                    | 出处                                                           | 在本项目中的用途         |
| ----------------------------------------------------- | ------------------------------------------------------------ | ---------------- |
| RoPE                                                  | [Su et al., 2021](https://arxiv.org/abs/2104.09864)          | 旋转位置编码           |
| YaRN                                                  | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)         | 长上下文位置编码外推       |
| DPO                                                   | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)         | 直接偏好优化           |
| CPO                                                   | [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)         | 对比偏好优化           |
| CDPO                                                  | [ericmitchell.ai/cdpo.pdf](https://ericmitchell.ai/cdpo.pdf) | DPO 校准策略         |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)      | —                                                            | PPO/GRPO 损失函数参考  |
| [HuggingFace TRL](https://github.com/huggingface/trl) | —                                                            | DeepSpeed 模型准备逻辑 |
| DeepSpeed ZeRO                                        | Microsoft                                                    | 分布式训练显存优化        |

## 许可证

[Apache 2.0](LICENSE)
