"""
NanoLM Python API — 一行调用推理
"""

import torch

from model import LlmModel
from nanolm.device import get_optimal_device
from nanolm.utils import get_model_config, init_env


def load_model(path="./bin/ppo_policy.bin", device=None, long_context=True):
    """
    加载 NanoLM 模型。

    Args:
        path: 模型权重路径
        device: 'cuda' | 'mps' | 'cpu'，None 则自动选择
        long_context: 是否使用长上下文配置（2048 tokens）

    Returns:
        LlmModel 实例（已设为 eval 模式）
    """
    init_env()

    if device is None:
        device = get_optimal_device()

    model = LlmModel(get_model_config(long_context=long_context)).to(device=device)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def generate(model, prompt, max_tokens=512, temperature=1.0, top_p=0.95, top_k=None, device=None):
    """
    单次文本生成。

    Args:
        model: LlmModel 实例
        prompt: 字符串 prompt
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: nucleus sampling 概率
        top_k: top-k 采样，None 则不用
        device: 设备

    Returns:
        生成的完整文本（不含 prompt）
    """
    from train import TrainerTools, streaming_generate

    if device is None:
        device = next(model.parameters()).device

    prompt_tokens = torch.tensor(
        TrainerTools().tokenizer.encode(prompt),
        device=device
    )

    generator = streaming_generate(
        model=model,
        prompt=prompt_tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
        k=top_k,
        p=top_p,
        device=device,
        return_token=True,
    )

    response_tokens = []
    for token in generator:
        if token == TrainerTools().tokenizer.end:
            break
        response_tokens.append(token)

    return TrainerTools().tokenizer.decode(torch.tensor(response_tokens))


def chat(model, history, max_tokens=512, temperature=1.0, top_p=0.95):
    """
    多轮对话。

    Args:
        model: LlmModel 实例
        history: 对话历史 [{"role": "user", "content": "..."}, ...]
        max_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: nucleus sampling 概率

    Returns:
        assistant 回复文本
    """
    from train import TrainerTools

    system_tokens = TrainerTools().tokenizer.encode('<system> </s>')
    max_user_tokens = 2048 - max_tokens

    # 反转历史，从最新到最旧
    reversed_history = list(reversed(history))
    chat_tokens = []

    for turn in reversed_history:
        role = '<user>' if turn['role'] == 'user' else '<assistant>'
        turn_tokens = TrainerTools().tokenizer.encode(f"{role}{turn['content']}</s>")
        if len(system_tokens) + len(chat_tokens) + len(turn_tokens) >= max_user_tokens:
            break
        chat_tokens.append(turn_tokens)

    chat_tokens.reverse()
    flat_tokens = [t for sublist in chat_tokens for t in sublist]
    flat_tokens.append(TrainerTools().tokenizer.assistant)
    flat_tokens = system_tokens + flat_tokens

    return generate(
        model,
        prompt=TrainerTools().tokenizer.decode(flat_tokens),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
