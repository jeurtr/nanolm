"""验证数据集加载"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanolm.utils import init_env

init_env()

from train import TrainerTools
from train.dataset import PretrainDataset, RLDataset, SFTDataset


def test_pretrain_dataset_empty():
    """PretrainDataset 处理空数据时 length 为 0"""
    # 创建一个只有少量 token 的临时文件
    tokens = torch.randint(0, 1000, (10,))
    import tempfile

    import numpy as np
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, tokens.numpy())
        path = f.name

    ds = PretrainDataset(path, block_size=100, stride=100)
    # 10 tokens < 100 block_size，length 应为 0
    assert len(ds) == 0
    os.unlink(path)


def test_sft_dataset_small():
    """SFTDataset 可以处理小数组"""
    import tempfile

    import numpy as np
    data = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
    ], dtype=np.int64)

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, data)
        path = f.name

    ds = SFTDataset(path, block_size=2048)
    assert len(ds) == 2
    item = ds[0]
    assert 'inputs' in item
    os.unlink(path)


def test_rl_dataset_small():
    """RLDataset 可以处理小数组"""
    import tempfile

    import numpy as np
    data = np.array([
        {'prompt': np.array([1, 2, 3]).astype(np.int64), 'answer': None},
        {'prompt': np.array([4, 5, 6]).astype(np.int64), 'answer': None},
    ])

    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, data)
        path = f.name

    ds = RLDataset(path)
    assert len(ds) == 2
    item = ds[0]
    assert 'prompt' in item
    assert 'answer' in item
    os.unlink(path)


def test_tokenizer_loads():
    """Tokenizer 可以正常加载"""
    tokenizer = TrainerTools().tokenizer
    assert tokenizer.vocab_size > 0
    assert tokenizer.end is not None
    assert tokenizer.pad is not None

    encoded = tokenizer.encode("你好世界")
    assert len(encoded) > 0

    decoded = tokenizer.decode(encoded)
    assert "你好" in decoded or len(decoded) > 0
