"""
从 PPO 联合 checkpoint 提取 policy 权重

PPO 训练输出的 ppo.bin 包含 policy_model + value_model 的权重。
此脚本提取 policy 部分，保存为 ppo_policy.bin，供推理使用。
"""

import torch

from nanolm.utils import get_model_config, init_env
from train import extract_policy_weights_from_ppo


def extract_policy(ppo_path='./bin/ppo.bin', output_path='./bin/ppo_policy.bin'):
    init_env()
    ppo_weights = torch.load(ppo_path, weights_only=True)
    policy_weights = extract_policy_weights_from_ppo(get_model_config(), ppo_weights)
    torch.save(policy_weights, output_path)
    print(f"[extract] {ppo_path} → {output_path}")


if __name__ == '__main__':
    extract_policy()
