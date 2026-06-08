from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from model import ModelConfig, VLMConfig

from .tools import FileDataset


@dataclass(kw_only=True)
class DsOffloadConfig:
    device: str = 'cpu'
    pin_memory: bool = True


@dataclass(kw_only=True)
class DsActivationCheckpointingConfig:
    partition_activations: bool = True
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = True
    number_checkpoints: int | None = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False


@dataclass(kw_only=True)
class DsZeROConfig:
    stage: int
    allgather_partitions: bool | None = True
    allgather_bucket_size: int | None = 5e8
    overlap_comm: bool | None = True
    reduce_scatter: bool | None = True
    reduce_bucket_size: str | int | None = 5e8
    contiguous_gradients: bool | None = True

@dataclass(kw_only=True)
class DsZero0Config(DsZeROConfig):
    stage: int = field(default=0, init=False)


@dataclass(kw_only=True)
class DsZero1Config(DsZeROConfig):
    stage: int = field(default=1, init=False)


@dataclass(kw_only=True)
class DsZero2Config(DsZeROConfig):
    stage: int = field(default=2, init=False)
    offload_optimizer: DsOffloadConfig | None = None
    offload_param: DsOffloadConfig | None = None


@dataclass(kw_only=True)
class DsZero3Config(DsZeROConfig):
    stage: int = field(default=3, init=False)
    sub_group_size: int | None = 1e9
    stage3_prefetch_bucket_size: str | int | None = 'auto'
    stage3_param_persistence_threshold: str | int | None = 'auto'
    stage3_max_live_parameters: int | None = 1e9
    stage3_max_reuse_distance: int | None = 1e9
    stage3_gather_16bit_weights_on_model_save: bool | None = True
    offload_optimizer: DsOffloadConfig | None = None
    offload_param: DsOffloadConfig | None = None


@dataclass(kw_only=True)
class DsFp16Config:
    enabled: str | bool = 'auto'
    loss_scale: int = 0
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: int = 1
    fp16_opt_level: str | None = 'O2'


@dataclass(kw_only=True)
class DsBf16Config:
    enabled: bool = True


@dataclass(kw_only=True)
class DsConfig:
    zero_config: DsZeROConfig | None = field(default_factory=DsZero3Config)
    fp16_config: DsFp16Config | None = field(default_factory=DsFp16Config)
    bf16_config: DsBf16Config | None = field(default_factory=DsBf16Config)
    gradient_clipping: float | None = 1.0
    activation_checkpointing: DsActivationCheckpointingConfig | None = None


@dataclass(kw_only=True)
class DataLoaderConfig:
    """
        data loader配置项
        Args:
            data_loader_pin_memory (`bool`, *optional*, default is None):
                data_loader pin_memory config
            data_loader_num_workers (`int`, *optional*, default is 0):
                data_loader num_workers config
            data_loader_shuffle (`bool`, *optional*, default is False):
                是否需要shuffle数据
            data_loader_drop_last (`bool`, default is False):
                最后一个batch不满足batch_size时，是否丢弃
    """
    data_loader_pin_memory: bool = False
    data_loader_num_workers: int = 0
    data_loader_shuffle: bool = False
    data_loader_drop_last: bool = True


@dataclass(kw_only=True)
class OptimConfig:
    optim_type: str = 'adam' # or 'lion'
    enable_lr_scheduler: bool = False
    initial_lr: float
    weight_decay: float | None = None
    betas: tuple[float, float] | None = None
    warmup_iters: int | None = None
    max_lr: float | None = None
    min_lr: float | None = None
    cosine_annealing_period: int | None = None
    cosine_annealing_period_mul: int = 0


@dataclass(kw_only=True)
class LossConfig:
    critical_tokens: list[int] | None = None
    critical_alpha: float = 1.0
    aux_loss_coef: float | None = 0.001


@dataclass(kw_only=True)
class KDConfig:
    """
        知识蒸馏模式配置项

        Args:
            teacher_logits_provider (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                知识蒸馏教师模型logits的提供者
            kd_coef (`float`, *optional*, default is 0.4):
                蒸馏loss的占比，loss = kd_coef * kd_loss + (1 - kd_coef) * lm_loss
    """
    teacher_logits_provider: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    kd_coef: float = 0.4


@dataclass(kw_only=True)
class EvalConfig:
    """
    训练参数配置项

    Args:
        eval_batch_interval (`int`, default is 100):
            每隔多少个batch进行模型eval
    """
    max_seq_len: int

    eval_batch_interval: int = 100
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: float | None = None


@dataclass(kw_only=True)
class PretrainConfig:
    """
    训练参数配置项

    Args:
        gradient_accumulation_steps (`int`, *Optional*, default is 1):
            梯度累积步数，为0时不使用梯度累积
            目前仅适用于pretrain、sft、dpo，不适用于ppo、grpo、gspo
        kd_config: (`KDConfig`, *Optional*, default is None):
            知识蒸馏配置项，为None时不使用知识蒸馏
    """
    gradient_accumulation_steps: int = 1
    kd_config: KDConfig | None = None


@dataclass(kw_only=True)
class SFTConfig:
    """
    训练参数配置项

    Args:
        mask_prompt (`bool`)
            指定是否mask prompt部分的token
        gradient_accumulation_steps (`int`, *Optional*, default is 1):
            梯度累积步数，为0时不使用梯度累积
            目前仅适用于pretrain、sft、dpo，不适用于ppo、grpo、gspo
        kd_config: (`KDConfig`, *Optional*, default is None):
            知识蒸馏配置项，为None时不使用知识蒸馏
        pixel_values_provider: (`Callable[[list[str]], torch.Tensor]`, *Optional*, default is None):
            训练vlm时根据image_tag提供pixel_values信息
        freeze_model:
            是否冻结llm部分model参数，用于训练vlm
    """
    mask_prompt: bool = True
    gradient_accumulation_steps: int = 1
    kd_config: KDConfig | None = None
    image_tags_file_dataset: FileDataset | None = None
    pixel_values_provider: Callable[[list[str]], torch.Tensor] | None = None
    freeze_model: bool = False


@dataclass(kw_only=True)
class DPOConfig:
    """
    训练参数配置项

    Args:
        mask_prompt (`bool`)
            指定是否mask prompt部分的token
        gradient_accumulation_steps (`int`, *Optional*, default is 1):
            梯度累积步数，为0时不使用梯度累积
            目前仅适用于pretrain、sft、dpo，不适用于ppo、grpo、gspo
    """
    ref_model_checkpoint: Mapping[str, Any]
    mask_prompt: bool = True
    gradient_accumulation_steps: int = 1
    loss_beta: float
    loss_label_smoothing: float = 0.0
    loss_ipo: bool = False
    nll_loss_coef: float | None = None


@dataclass(kw_only=True)
class PPOConfig:
    ppo_epochs: int
    ppo_batch_size: int
    ref_model_checkpoint: Mapping[str, Any]
    value_model_checkpoint: Mapping[str, Any] | None = None
    value_optim_config: Optional['OptimConfig'] = None
    gradient_accumulation_steps: int = 1
    gamma: float = 1.0
    lam: float = 0.95
    clip_eps: float = 0.1
    vf_coef: float = 0.5
    kl_beta: float = 0.02
    kl_estimator: str = 'k1' # or k3
    missing_eos_penalty: float | None = None
    normalize_rewards: bool = False
    normalize_method: str = 'RunningMeanStd' # RunningMeanStd or BatchStd
    whiten_rewards: bool = False
    gen_max_seq_len: int
    gen_temperature: float | None = None
    gen_k: int | None = None
    gen_p: float | None = None
    gen_suppress_tokens: list[int] | None = None


@dataclass(kw_only=True)
class GRPOConfig:
    grpo_steps: int = 1
    group_size: int = 12
    mixup_alpha: float = 1.0
    loss_beta: float = 0.0 # or 0.04 for grpo
    loss_clip_eps: float = 3e-4
    loss_clip_eps_high: float | None = 4e-4
    loss_delta: float | None = None
    loss_importance_sampling_level: str = 'seq' # token or seq
    loss_type: str = 'grpo' # grpo or bnpo or dr_grpo
    gen_max_seq_len: int
    gen_temperature: float | None = None
    gen_k: int | None = None
    gen_p: float | None = None
    gen_suppress_tokens: list[int] | None = None


@dataclass(kw_only=True)
class TrainConfig:
    """
    训练参数配置项

    Args:
        n_epochs (`int`):
            训练epochs
        batch_size (`int`):
            每个batch的大小
        model_config (`ModelConfig`):
            模型的配置
        init_state_dict:
            初始化检查点
        file_dataset (`FileDataset`):
            训练文件dataset
        dataset_block_size (`int`, default is None)
            训练序列最大长度，为None时取model的max_position_embedding
        data_loader_config: (`DataLoaderConfig`):
            data loader配置项
        loss_config:
            配置loss
        ds_config:
            配置deepspeed
        eval_config:
            配置eval
        optim_config (`OptimConfig`):
            optim配置项
        pretrain_config:
            预训练配置项，仅适用于使用Trainer
        sft_config:
            sft配置项，仅适用于使用SFTTrainer
        dpo_config:
            dpo配置项，仅适用于使用DPOTrainer
        ppo_config:
            ppo配置项，仅适用于使用PPOTrainer
        grpo_config:
            grpo配置项，仅适用于使用GRPOTrainer
    """
    n_epochs: int
    batch_size: int
    model_config: ModelConfig | VLMConfig
    init_state_dict: Mapping[str, Any] | None = None

    file_dataset: FileDataset
    dataset_block_size: int
    data_loader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    loss_config: LossConfig = field(default_factory=LossConfig)
    optim_config: OptimConfig = field(default_factory=OptimConfig)
    ds_config: DsConfig = field(default_factory=DsConfig)

    eval_config: EvalConfig = field(default_factory=EvalConfig)

    pretrain_config: PretrainConfig | None = None
    sft_config: SFTConfig | None = None
    dpo_config: DPOConfig | None = None
    ppo_config: PPOConfig | None = None
    grpo_config: GRPOConfig | None = None
