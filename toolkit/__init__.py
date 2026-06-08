"""
NanoLM 工具集

用法：
    python -m toolkit full          # 完整管线
    python -m toolkit download      # 仅下载
    python -m toolkit shuffle       # 仅打乱
    ...
"""

from .download import download_raw_datasets
from .encode import encode_pretrain_and_midtrain
from .encode_ppo import encode_ppo_data
from .encode_sft import encode_sft_data
from .merge import merge_sft_files
from .preprocess import preprocess_pretrain_data
from .quality import validate_pipeline_outputs
from .shuffle import shuffle_all_raw_files, shuffle_file
from .split import split_sft_data

__all__ = [
    'download_raw_datasets',
    'encode_pretrain_and_midtrain',
    'encode_ppo_data',
    'encode_sft_data',
    'merge_sft_files',
    'preprocess_pretrain_data',
    'validate_pipeline_outputs',
    'shuffle_all_raw_files',
    'shuffle_file',
    'split_sft_data',
]
