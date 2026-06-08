"""
PPO 奖励函数

使用 InternLM2-1.8B Reward Model 计算回复质量分数。
奖励 = EOS 惩罚 + RM 模型打分（加权并截断）。
"""


import torch
from modelscope import snapshot_download
from transformers import AutoModel, AutoTokenizer

from nanolm.device import empty_cache
from train import TrainerTools

MODEL_ID = "Shanghai_AI_Laboratory/internlm2-1_8b-reward"
LOCAL_CACHE_DIR = "./rm_models"

SCORE_EOS_PENALTY = -5.0   # 未以 </s> 结尾的惩罚
RM_WEIGHT = 0.5            # RM 分数权重
RM_SCORE_CLIP = 5.0        # RM 分数截断范围

_model = None
_tokenizer = None


def _get_rm():
    global _model, _tokenizer
    if _model is None:
        model_dir = snapshot_download(MODEL_ID, cache_dir=LOCAL_CACHE_DIR, revision='master')
        _model = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.float16,
            device_map='cpu', trust_remote_code=True,
        ).eval()
        _tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _model, _tokenizer


def _replace_spec_tokens(text: str) -> str:
    text = text.replace('<system> </s>', '')
    spec_tokens = TrainerTools().tokenizer.get_special_tokens_dict().keys()
    for tk in spec_tokens:
        text = text.replace(tk, '')
    return text.strip()


def reward_func(
        prompt_ids: list[torch.Tensor],
        completion_ids: torch.Tensor,
        answers: list[torch.Tensor | None]) -> list[float]:
    """
    PPO 训练使用的奖励函数签名。

    Returns:
        每个样本的标量奖励值
    """
    rm, rm_tokenizer = _get_rm()
    rm_device = TrainerTools().parallel.device

    prompts_text = TrainerTools().tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
    completions_text = TrainerTools().tokenizer.batch_decode(completion_ids, skip_special_tokens=False)

    batch_size = len(prompts_text)
    total_scores = [0.0] * batch_size
    rm_inputs_text = []
    rm_indices = []

    for idx, (prompt, completion) in enumerate(zip(prompts_text, completions_text, strict=False)):
        completion = completion.replace("<pad>", '')
        has_eos = completion.endswith('</s>')

        if not has_eos:
            total_scores[idx] += SCORE_EOS_PENALTY

        clean_prompt = _replace_spec_tokens(prompt)
        clean_completion = _replace_spec_tokens(completion)

        chat = [
            {"role": "user", "content": clean_prompt},
            {"role": "assistant", "content": clean_completion}
        ]
        rm_inputs_text.append(rm_tokenizer.apply_chat_template(chat, tokenize=False))
        rm_indices.append(idx)

    if rm_inputs_text:
        rm.to(rm_device)
        try:
            inputs = rm_tokenizer(
                rm_inputs_text, return_tensors="pt",
                padding=True, truncation=True, max_length=2048,
            ).to(rm_device)

            with torch.no_grad():
                scores_tensor = rm(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).logits
                batch_rm_scores = scores_tensor.float().cpu().numpy().flatten()

            for i, original_idx in enumerate(rm_indices):
                raw = float(batch_rm_scores[i])
                clipped = max(min(raw, RM_SCORE_CLIP), -RM_SCORE_CLIP)
                total_scores[original_idx] += clipped * RM_WEIGHT

        except Exception as e:
            print(f"[reward] RM error: {e}")
            for original_idx in rm_indices:
                total_scores[original_idx] -= 2.0
        finally:
            rm.to('cpu')
            empty_cache()

    if TrainerTools().parallel.is_main_process:
        with open('./log/reward.txt', 'a', encoding='utf-8') as f:
            f.write("-" * 65 + "\n")
            f.write(f"Reward: {total_scores[0]:.4f}\n")

    return total_scores
