"""
训练启动器 — 从 YAML 配置加载参数并启动训练

用法:
    python -m toolkit train pretrain
    python -m toolkit train sft -c configs/custom/my_sft.yaml
"""

import os
import sys

from nanolm.utils import init_env as _base_init_env

# 训练配置工厂的 import 链需要 TOKEN_DIR 等环境变量已设置
_base_init_env()

from train import PPOTrainer, SFTTrainer, Trainer  # noqa: E402

from .config_loader import load_config  # noqa: E402

EVAL_PROMPTS_GENERIC = [
    '初中阶段是学生身心发育的一个突变期',
    '癜风病人调节心理要偶尔也要屈服',
    '列举出五种古代建筑的设计特点',
    '吕宽，西汉末年平帝时期人'
]

EVAL_PROMPTS_CONVERSATION = [
]


def _init_eval_prompts():
    from nanolm.utils import get_eval_prompt
    return [
        get_eval_prompt('写一篇介绍太阳系行星的科普文章'),
        get_eval_prompt('生态环境是人类的生存和发展的空间，所以人类是不是应当尽可能地去改变生态环境？'),
        get_eval_prompt('水资源主要是被工业用水消耗，我在生活中节约用水有意义吗？'),
        get_eval_prompt('作为历史初学者，我该如何开始我的历史学习之旅？'),
        get_eval_prompt('如果Python中的父类和子类分别定义在不同的文件里，怎样导入才能避免出现循环导入的问题呢？'),
        get_eval_prompt('你叫什么？'),
        get_eval_prompt('你是谁？'),
    ]


def _setup_parallel():
    try:
        import deepspeed  # noqa: F401
        parallel_type = 'ds'
    except ImportError:
        parallel_type = 'none'
    os.environ['PARALLEL_TYPE'] = parallel_type
    print(f"[train] backend={parallel_type}")


def _do_pretrain(config_path=None):
    _setup_parallel()
    config = load_config('pretrain', custom_path=config_path)
    Trainer(train_config=config, eval_prompts=EVAL_PROMPTS_GENERIC).train()


def _do_midtrain(config_path=None):
    _setup_parallel()
    config = load_config('midtrain', custom_path=config_path)
    Trainer(train_config=config, eval_prompts=EVAL_PROMPTS_GENERIC).train()


def _do_sft(config_path=None):
    _setup_parallel()
    config = load_config('sft', custom_path=config_path)
    SFTTrainer(train_config=config, eval_prompts=_init_eval_prompts()).train()


def _do_ppo(config_path=None):
    _setup_parallel()
    config = load_config('ppo', custom_path=config_path)
    from eval.reward import reward_func
    PPOTrainer(
        train_config=config,
        reward_func=reward_func,
        eval_prompts=_init_eval_prompts(),
    ).train()


STAGE_MAP = {
    'pretrain': _do_pretrain,
    'midtrain': _do_midtrain,
    'sft':     _do_sft,
    'ppo':     _do_ppo,
}


def run_training(stage, config_path=None):
    if stage not in STAGE_MAP:
        print(f"未知训练阶段: {stage}")
        print(f"可用: {', '.join(STAGE_MAP.keys())}")
        sys.exit(1)
    STAGE_MAP[stage](config_path)
