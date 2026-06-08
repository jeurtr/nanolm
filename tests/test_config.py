"""验证 YAML 配置加载"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanolm.utils import init_env

init_env()

from toolkit.config_loader import load_config


def test_load_default_pretrain():
    cfg = load_config('pretrain')
    assert cfg.n_epochs == 1
    assert cfg.batch_size == 76
    assert cfg.model_config.hidden_size == 768
    assert cfg.model_config.max_position_embeddings == 512
    assert cfg.model_config.rope_config.rope_type == 'default'


def test_load_default_midtrain():
    cfg = load_config('midtrain')
    assert cfg.n_epochs == 1
    assert cfg.batch_size == 18
    assert cfg.model_config.max_position_embeddings == 2048
    assert cfg.model_config.rope_config.rope_type == 'yarn'


def test_load_default_sft():
    cfg = load_config('sft')
    assert cfg.n_epochs == 1
    assert cfg.batch_size == 15
    assert cfg.sft_config.mask_prompt is True


def test_load_default_ppo():
    cfg = load_config('ppo')
    assert cfg.n_epochs == 2
    assert cfg.batch_size == 50
    assert cfg.ppo_config is not None
    assert cfg.ppo_config.kl_beta == 0.01
    assert cfg.ppo_config.vf_coef == 0.5
    assert cfg.optim_config.enable_lr_scheduler is False


def test_all_stage_configs():
    """所有阶段配置都能成功加载"""
    for stage in ['pretrain', 'midtrain', 'sft', 'ppo']:
        cfg = load_config(stage)
        assert cfg.model_config.hidden_size == 768
        assert cfg.model_config.num_hidden_layers == 12
        assert cfg.file_dataset is not None
