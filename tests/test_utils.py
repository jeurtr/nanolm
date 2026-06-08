"""验证 train.utils 工具函数"""

import torch

from train.utils import calc_position_ids, truncate_sequences_at_eos


def test_calc_position_ids_left_padding():
    """Left padding: mask 前面的 0 应映射为 position 0"""
    mask = torch.tensor([
        [0, 0, 1, 1, 1],   # 前两个为 pad，后三个为有效 token
        [0, 1, 1, 1, 1],   # 第一个为 pad
    ])
    position_ids = calc_position_ids(mask)
    expected = torch.tensor([
        [0, 0, 0, 1, 2],
        [0, 0, 1, 2, 3],
    ])
    assert torch.equal(position_ids, expected)


def test_calc_position_ids_no_padding():
    """无 padding 时 position 从 0 连续递增"""
    mask = torch.ones(2, 5, dtype=torch.bool)
    position_ids = calc_position_ids(mask)
    expected = torch.tensor([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ])
    assert torch.equal(position_ids, expected)


def test_calc_position_ids_all_pad():
    """全为 pad 时全为 0"""
    mask = torch.zeros(2, 3, dtype=torch.bool)
    position_ids = calc_position_ids(mask)
    assert torch.equal(position_ids, torch.zeros(2, 3, dtype=torch.long))


def test_truncate_sequences_at_eos_single():
    """EOS 及之后的内容替换为 pad_token_id（EOS 本身也不保留）"""
    sequences = torch.tensor([[1, 2, 5, 3, 4]])  # token 5 是 EOS
    result = truncate_sequences_at_eos(sequences, eos_token_id=5, pad_token_id=0)
    expected = torch.tensor([[1, 2, 0, 0, 0]])
    assert torch.equal(result, expected)


def test_truncate_sequences_at_eos_no_eos():
    """没有 EOS 的序列应保持不变"""
    sequences = torch.tensor([[1, 2, 3, 4]])
    result = truncate_sequences_at_eos(sequences, eos_token_id=99, pad_token_id=0)
    assert torch.equal(result, sequences)


def test_truncate_sequences_at_eos_batch():
    """批处理中部分序列有 EOS"""
    sequences = torch.tensor([
        [1, 5, 3, 4],   # EOS 在位置 1
        [1, 2, 3, 4],   # 无 EOS
    ])
    result = truncate_sequences_at_eos(sequences, eos_token_id=5, pad_token_id=0)
    expected = torch.tensor([
        [1, 0, 0, 0],
        [1, 2, 3, 4],
    ])
    assert torch.equal(result, expected)
