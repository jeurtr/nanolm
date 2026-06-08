"""
SFT vs PPO 效果对比

用 InternLM2-1.8B Reward Model 分别评估 SFT 和 PPO 模型在 12 个 prompt 上的回复质量，
输出平均得分。
"""

import torch
from modelscope import snapshot_download
from transformers import AutoModel, AutoTokenizer

from model import LlmModel
from nanolm.utils import get_eval_prompt, get_model_config, init_env
from train import TrainerTools, streaming_generate
from train.utils import set_seed

MODEL_ID = "Shanghai_AI_Laboratory/internlm2-1_8b-reward"
LOCAL_CACHE_DIR = "./rm_models"

PROMPTS = [
    '请写一个关于一只在大城市里迷路的流浪猫的短篇故事，结局要温馨。',
    '帮我构思一个悬疑故事的开头，背景设定在一家深夜的便利店。',
    '我最近总是失眠，有什么非药物的助眠小技巧吗？',
    '给一个刚毕业的大学生三条关于职场沟通的建议。',
    '请用通俗易懂的语言解释一下为什么天空是蓝色的。',
    '什么是"蝴蝶效应"？请举个例子说明。',
    '树上骑个猴，地上一个猴，一共几个猴？',
    '为什么在高速公路上不能突然停车？',
    '扮演一个外星人，第一次吃到冰淇淋时的反应。',
    '你是一个耐心的心理咨询师，安慰一个因为考试失利而沮丧的学生。',
    '将"快乐、公园、跑步、遇见、老朋友"这几个词串成一个通顺的句子。',
    '给"人工智能的发展"这个主题拟定三个不同的文章标题。',
]


def _get_rm():
    model_dir = snapshot_download(MODEL_ID, cache_dir=LOCAL_CACHE_DIR, revision='master')
    rm = AutoModel.from_pretrained(
        model_dir, torch_dtype=torch.float16,
        device_map='cpu', trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return rm, tokenizer


def _eval_one_model(model_type, rm, rm_tokenizer, device):
    print(f"[eval] 评估 {model_type} 模型 ...")
    ckpt = 'sft.bin' if model_type == 'sft' else 'ppo_policy.bin'

    model = LlmModel(get_model_config(long_context=True)).to(device)
    model.load_state_dict(torch.load(f'./bin/{ckpt}', weights_only=True))
    model.eval()

    batch_eval_template = []
    for prompt in PROMPTS:
        chat_template = get_eval_prompt(prompt)
        chat_tokens = TrainerTools().tokenizer.encode(chat_template, covert_tensor=True)

        generator = streaming_generate(
            model=model, prompt=chat_tokens,
            max_new_tokens=2048 - chat_tokens.shape[0],
            temperature=1.0, p=0.95,
            suppress_tokens=None, device=device, return_token=True,
        )

        response_tokens = []
        for item in generator:
            response_tokens.append(item)

        response = TrainerTools().tokenizer.decode(torch.tensor(response_tokens))
        batch_eval_template.append(
            rm_tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.replace('</s>', '')}
            ], tokenize=False)
        )

    inputs = rm_tokenizer(
        batch_eval_template, return_tensors="pt",
        padding=True, truncation=True, max_length=2048,
    ).to(device)

    with torch.no_grad():
        output = rm(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        score = torch.mean(output.logits).item()
        print(f"  {model_type} avg score = {score:.4f}")
        return score


def compare_sft_ppo():
    set_seed()
    init_env()

    from nanolm.device import get_optimal_device
    device = get_optimal_device()
    rm, rm_tokenizer = _get_rm()
    rm.to(device)

    sft_score = _eval_one_model('sft', rm, rm_tokenizer, device)
    ppo_score = _eval_one_model('ppo', rm, rm_tokenizer, device)

    print(f"\n[结果] SFT: {sft_score:.4f}  |  PPO: {ppo_score:.4f}  |  提升: {ppo_score - sft_score:+.4f}")


if __name__ == '__main__':
    compare_sft_ppo()
