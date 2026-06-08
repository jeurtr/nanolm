"""
NanoLM 模型评估与后处理

用法：
    python -m toolkit eval compare     # SFT vs PPO 效果对比
    python -m toolkit eval extract     # 从 PPO checkpoint 提取 policy 权重
"""

from .compare import compare_sft_ppo
from .extract import extract_policy
from .reward import reward_func

__all__ = ['compare_sft_ppo', 'extract_policy', 'reward_func']
