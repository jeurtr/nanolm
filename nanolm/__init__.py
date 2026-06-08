"""
NanoLM — 从零构建小型大语言模型

用法:
    from nanolm import load_model, generate, chat

    model = load_model("./bin/ppo_policy.bin")
    print(generate(model, "你好"))
"""

__version__ = '0.1.1'

from .api import chat, generate, load_model
from .device import (
    empty_cache,
    get_autocast_dtype,
    get_optimal_backend,
    get_optimal_device,
    is_bf16_supported,
    is_fp16_supported,
    resolve_device_type,
)
from .utils import get_eval_prompt, get_model_config, init_env

__all__ = [
    'chat',
    'generate',
    'load_model',
    'empty_cache',
    'get_autocast_dtype',
    'get_optimal_backend',
    'get_optimal_device',
    'is_bf16_supported',
    'is_fp16_supported',
    'resolve_device_type',
    'get_eval_prompt',
    'get_model_config',
    'init_env',
]
