"""
NanoLM 工具集 — 集中化配置

所有路径、阈值、魔术数字统一管理。
可通过环境变量覆盖默认值。
"""

import os

# --- 基础常量 --- #

developer_name = 'QB'
assistant_name = 'NanoLM'

image_size = 224       # VLM 图像尺寸
patch_size = 16        # VLM patch 尺寸
tokens_per_image = 196 # VLM 每张图 token 数

def data_root_dir():
    dir_name = os.environ.get('NANOLM_DATA_DIR', './data/')
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def raw_dir():
    d = os.path.join(data_root_dir(), 'raw')
    os.makedirs(d, exist_ok=True)
    return d

def tmp_dir():
    d = os.path.join(data_root_dir(), 'tmp')
    os.makedirs(d, exist_ok=True)
    return d

# --- 目录结构 --- #

RAW_DIR = os.path.join(data_root_dir(), 'raw')
TMP_DIR = os.path.join(data_root_dir(), 'tmp')
OUTPUT_DIR = data_root_dir()  # 最终 .npy 直接放在 data/ 下


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# --- 文件列表 --- #

# shuffle_all_data 后出现在 tmp/ 的 SFT 相关文件
SFT_SPLIT_FILES = [
    'shuffle_pretrain_2048.jsonl',
    'shuffle_sft_512.jsonl',
    'shuffle_sft_1024.jsonl',
    'shuffle_r1_mix_1024.jsonl',
]

# merge_sft_data 的输入
SFT_MERGE_INPUTS = [
    'shuffle_sft_mini_2048.jsonl',
    'shuffle_sft_mini_512.jsonl',
]

# --- 阈值与参数 --- #

SHORT_TEXT_THRESHOLD = int(os.environ.get('NANOLM_SHORT_THRESHOLD', '768'))
MAX_SFT_TOKEN_LENGTH = int(os.environ.get('NANOLM_MAX_SFT_LEN', '2048'))
SELF_COGNITION_MULTIPLIER = int(os.environ.get('NANOLM_SELF_COG_MULT', '20'))
VOCAB_THRESHOLD_UINT16 = 65535

# 预训练数据分块数（pretrain=2, midtrain=1）
PRETRAIN_SPLIT_COUNT = 2
MIDTRAIN_SPLIT_COUNT = 1

# --- 批次大小 --- #

PRETRAIN_BATCH_SIZE = 50000
SFT_BATCH_SIZE = 10000

# --- ModelScope 数据集 ID --- #

MINIMIND_DATASET = 'gongjy/minimind_dataset'
SELF_COGNITION_DATASET = 'swift/self-cognition'
NANOLM_DATA_REPO = 'qibin0506/NanoLM-3.0-data'

# --- 输出文件名（与 file_dataset.py 保持一致） --- #

PRETRAIN_OUTPUT_PREFIX = 'pretrain_data'
MIDTRAIN_OUTPUT_PREFIX = 'midtrain_data'
SFT_OUTPUT = 'sft_data.npy'
PPO_OUTPUT = 'ppo_data.npy'

# --- 打乱随机种子 --- #

SHUFFLE_SEED = int(os.environ.get('NANOLM_SHUFFLE_SEED', '42'))
