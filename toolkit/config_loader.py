"""
YAML 配置加载器

加载 configs/ 目录的 YAML 文件，支持继承（extends），
合并后转换为 TrainConfig dataclass 对象。
"""

import math
import os

import yaml

from model import ModelConfig, RoPEConfig
from train import TrainerTools
from train.train_configs import (
    DataLoaderConfig,
    DsConfig,
    DsZero0Config,
    DsZero1Config,
    DsZero2Config,
    DsZero3Config,
    EvalConfig,
    LossConfig,
    OptimConfig,
    PPOConfig,
    PretrainConfig,
    SFTConfig,
    TrainConfig,
)

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')


def _deep_merge(base, override):
    """递归合并两个 dict，override 覆盖 base"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_extends(data, base_dir):
    """递归解析 extends 链，返回合并后的完整配置"""
    if 'extends' not in data:
        return data
    parent_name = data.pop('extends')
    parent_path = os.path.join(base_dir, f'{parent_name}.yaml')
    if not os.path.exists(parent_path):
        parent_path = os.path.join(base_dir, parent_name)
    parent_data = _load_yaml(parent_path)
    parent_data = _resolve_extends(parent_data, base_dir)
    return _deep_merge(parent_data, data)


def _build_model_config(cfg):
    m = cfg.get('model', {})
    rope = m.get('rope', {})

    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=m.get('hidden_size', 768),
        intermediate_size=m.get('intermediate_size', 2048),
        num_hidden_layers=m.get('num_hidden_layers', 12),
        num_attention_heads=m.get('num_attention_heads', 12),
        num_key_value_heads=m.get('num_key_value_heads', 4),
        max_position_embeddings=m.get('max_position_embeddings', 512),
        original_max_position_embeddings=m.get('original_max_position_embeddings'),
        attention_dropout=m.get('attention_dropout', 0.0),
        attention_implementation=m.get('attention_implementation', 'auto'),
        tie_word_embeddings=m.get('tie_word_embeddings', False),
        use_qk_norm=m.get('use_qk_norm', True),
        initializer_range=m.get('initializer_range', 0.02),
        rope_config=RoPEConfig(
            rope_type=rope.get('rope_type', 'default'),
            rope_theta=rope.get('rope_theta', 10000.0),
            factor=rope.get('factor', 1.0),
        ),
    )


def _build_optim_config(cfg):
    opt = cfg.get('optimizer', {})
    lr = cfg.get('lr', {})
    total_steps = _calc_total_steps(cfg)

    if lr.get('scheduler', True):
        warmup_iters = int(opt.get('warmup_ratio', 0.1) * total_steps)
        cosine_period = math.ceil(total_steps - warmup_iters)
        max_lr = lr.get('max', lr.get('initial', 1e-5))
        min_lr = max_lr * opt.get('min_lr_ratio', 0.1)
    else:
        warmup_iters = None
        cosine_period = None
        max_lr = None
        min_lr = None

    return OptimConfig(
        optim_type=opt.get('type', 'adam'),
        enable_lr_scheduler=lr.get('scheduler', True),
        initial_lr=lr.get('initial', 1e-5),
        warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=min_lr,
        cosine_annealing_period=cosine_period,
        weight_decay=opt.get('weight_decay'),
        betas=tuple(opt['betas']) if 'betas' in opt else None,
    )


def _calc_total_steps(cfg):
    epochs = cfg.get('n_epochs', 1)
    data_size = cfg.get('data_size', 1)
    batch_size = cfg.get('batch_size', 1)
    grad_acc = cfg.get('gradient_accumulation_steps', 1)
    world_size = TrainerTools().parallel.world_size
    return int(epochs * data_size / batch_size / world_size / grad_acc)


def _build_ds_config(cfg):
    ds = cfg.get('ds', {})
    stage = ds.get('zero_stage', 1)
    zero_cls = {0: DsZero0Config, 1: DsZero1Config, 2: DsZero2Config, 3: DsZero3Config}
    return DsConfig(
        zero_config=zero_cls.get(stage, DsZero1Config)(),
        gradient_clipping=ds.get('gradient_clipping', 1.0),
    )


def _build_data_loader_config(cfg):
    dl = cfg.get('dataloader', {})
    return DataLoaderConfig(
        data_loader_pin_memory=dl.get('pin_memory', True),
        data_loader_num_workers=dl.get('num_workers', 0),
        data_loader_shuffle=dl.get('shuffle', False),
        data_loader_drop_last=dl.get('drop_last', True),
    )


def _build_loss_config(cfg):
    ls = cfg.get('loss', {})
    return LossConfig(
        critical_tokens=ls.get('critical_tokens'),
        critical_alpha=ls.get('critical_alpha', 1.0),
        aux_loss_coef=ls.get('aux_loss_coef', 0.001),
    )


def _build_eval_config(cfg, model_config):
    ev = cfg.get('eval', {})
    return EvalConfig(
        max_seq_len=model_config.max_position_embeddings,
        eval_batch_interval=cfg.get('eval_batch_interval', 100),
        temperature=ev.get('temperature', 1.0),
        top_p=ev.get('top_p', 0.95),
        top_k=ev.get('top_k'),
    )


def load_config(stage, custom_path=None):
    """
    加载并合并 YAML 配置，返回 TrainConfig。

    Args:
        stage: 'pretrain' | 'midtrain' | 'sft' | 'ppo'
        custom_path: 可选的自定义 YAML 路径（覆盖默认值）
    """
    # 1. 加载 stage YAML（自动解析 extends）
    base_path = os.path.join(CONFIGS_DIR, f'{stage}.yaml')
    cfg = _load_yaml(base_path)
    cfg = _resolve_extends(cfg, CONFIGS_DIR)

    # 2. 合并自定义配置
    if custom_path:
        custom_cfg = _load_yaml(custom_path)
        custom_cfg = _resolve_extends(custom_cfg, os.path.dirname(custom_path) or '.')
        cfg = _deep_merge(cfg, custom_cfg)

    # 3. 构建核心对象
    model_config = _build_model_config(cfg)
    optim_config = _build_optim_config(cfg)
    ds_config = _build_ds_config(cfg)

    # 4. 初始化状态字典
    init_state_dict = None
    ref_checkpoint = None
    paths = cfg.get('paths', {})
    init_path = paths.get('init_checkpoint', './last_checkpoint.bin')
    ref_path = paths.get('ref_checkpoint', './sft.bin')

    if os.path.exists(init_path):
        import torch
        init_state_dict = torch.load(init_path, weights_only=True)
    if stage == 'ppo' and os.path.exists(ref_path):
        import torch
        ref_checkpoint = torch.load(ref_path, weights_only=True)

    # 5. 文件数据集
    from toolkit.file_dataset import (
        MidtrainFileDataset,
        PPODataset,
        PretrainFileDataset,
        SFTFileDataset,
    )
    file_dataset_map = {
        'pretrain': PretrainFileDataset,
        'midtrain': MidtrainFileDataset,
        'sft': SFTFileDataset,
        'ppo': PPODataset,
    }
    file_dataset = file_dataset_map[stage]()

    # 6. 阶段特定配置
    pretrain_config = None
    sft_config = None
    ppo_config = None

    grad_acc = cfg.get('gradient_accumulation_steps', 1)

    if stage in ('pretrain', 'midtrain'):
        pretrain_config = PretrainConfig(
            gradient_accumulation_steps=grad_acc,
            kd_config=None,
        )
    elif stage == 'sft':
        sft_cfg = cfg.get('sft', {})
        sft_config = SFTConfig(
            mask_prompt=sft_cfg.get('mask_prompt', True),
            gradient_accumulation_steps=grad_acc,
            kd_config=None,
        )
    elif stage == 'ppo':
        ppo_cfg = cfg.get('ppo', {})
        lr_cfg = cfg.get('lr', {})
        value_lr = lr_cfg.get('value', {})

        ppo_config = PPOConfig(
            ppo_epochs=ppo_cfg.get('epochs', 2),
            ppo_batch_size=ppo_cfg.get('batch_size', 5),
            gradient_accumulation_steps=grad_acc,
            ref_model_checkpoint=ref_checkpoint,
            vf_coef=ppo_cfg.get('vf_coef', 0.5),
            kl_beta=ppo_cfg.get('kl_beta', 0.01),
            kl_estimator=ppo_cfg.get('kl_estimator', 'k3'),
            normalize_rewards=ppo_cfg.get('normalize_rewards', True),
            normalize_method=ppo_cfg.get('normalize_method', 'RunningMeanStd'),
            clip_eps=ppo_cfg.get('clip_eps', 0.1),
            gamma=ppo_cfg.get('gamma', 1.0),
            lam=ppo_cfg.get('lam', 0.95),
            gen_max_seq_len=ppo_cfg.get('gen', {}).get('max_seq_len', 2048),
            gen_temperature=ppo_cfg.get('gen', {}).get('temperature', 1.0),
            gen_p=ppo_cfg.get('gen', {}).get('p', 0.9),
            value_optim_config=OptimConfig(
                enable_lr_scheduler=value_lr.get('scheduler', False),
                initial_lr=value_lr.get('initial', 5e-5),
            ),
        )

    return TrainConfig(
        n_epochs=cfg.get('n_epochs', 1),
        batch_size=cfg.get('batch_size', 16),
        model_config=model_config,
        file_dataset=file_dataset,
        dataset_block_size=model_config.max_position_embeddings,
        loss_config=_build_loss_config(cfg),
        optim_config=optim_config,
        ds_config=ds_config,
        data_loader_config=_build_data_loader_config(cfg),
        init_state_dict=init_state_dict,
        eval_config=_build_eval_config(cfg, model_config),
        pretrain_config=pretrain_config,
        sft_config=sft_config,
        ppo_config=ppo_config,
    )
