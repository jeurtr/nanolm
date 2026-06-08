"""验证模型创建和前向传播"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanolm.utils import get_model_config, init_env

init_env()

from model import LlmModel


def test_model_creation():
    """模型可以成功创建"""
    config = get_model_config(long_context=False)
    model = LlmModel(config)
    assert model is not None
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1_000_000
    assert total_params < 100_000_000  # ~82M


def test_model_forward():
    """模型可以执行前向传播"""
    config = get_model_config(long_context=False)
    model = LlmModel(config)
    model.eval()

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    assert 'logits' in output
    assert output['logits'].shape == (batch_size, seq_len, config.vocab_size)
    assert 'hidden_states' in output
    assert output['hidden_states'].shape == (batch_size, seq_len, config.hidden_size)


def test_model_with_attention_mask():
    """带 attention mask 的前向传播"""
    config = get_model_config(long_context=False)
    model = LlmModel(config)
    model.eval()

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attention_mask[0, -4:] = False  # 部分 padding

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    assert output['logits'].shape[0] == batch_size


def test_long_context_model():
    """长上下文模型配置可以创建"""
    config = get_model_config(long_context=True)
    model = LlmModel(config)
    assert config.max_position_embeddings == 2048
    assert config.rope_config.rope_type == 'yarn'
    assert model is not None
