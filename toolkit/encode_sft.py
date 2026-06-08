"""
阶段 8: Tokenize SFT 数据 + self-cognition 注入

修复: 直接调用 apply_chat_template(tokenizer=True) 避免字符串往返的二次 encode。
"""

import json
import os

import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

from train import TrainerTools

from .config import (
    MAX_SFT_TOKEN_LENGTH,
    OUTPUT_DIR,
    RAW_DIR,
    SELF_COGNITION_MULTIPLIER,
    SFT_OUTPUT,
    TMP_DIR,
    assistant_name,
    developer_name,
)
from .utils import determine_dtype


def _get_self_cognition_tokens(dtype):
    """读取 self_cognition.jsonl 并 tokenize"""
    raw_path = os.path.join(RAW_DIR, 'self_cognition.jsonl')
    tokens_list = []

    with open(raw_path) as f:
        for line in f:
            item = json.loads(line)
            user = item['query']
            content = item['response'].replace('{{AUTHOR}}', developer_name).replace('{{NAME}}', assistant_name)

            chat_template = [
                {'role': 'system', 'content': ' '},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': content.strip()}
            ]

            ids = TrainerTools().tokenizer.apply_chat_template(
                chat_template, add_answer_tag_for_assistant=False
            )
            tokens_list.append(np.array(ids, dtype=dtype))

    return tokens_list


def encode_sft_data(tmp_dir=None, output_dir=None):
    """Tokenize SFT 对话数据，注入 self-cognition，保存为 .npy"""
    tmp_dir = tmp_dir or TMP_DIR
    output_dir = output_dir or OUTPUT_DIR
    dtype = determine_dtype()
    tokenizer = TrainerTools().tokenizer

    input_path = os.path.join(tmp_dir, 'sft_data.jsonl')
    if not os.path.exists(input_path):
        print(f"[encode-sft] {input_path} 不存在，跳过")
        return

    tokens = []
    total_lines = sum(1 for _ in open(input_path))

    print(f"[encode-sft] 处理 {input_path} ({total_lines} 行) ...")

    with open(input_path) as f:
        for line in tqdm(f, total=total_lines):
            conversations = json.loads(line)['conversations']

            chat_template = [{'role': 'system', 'content': ' '}]
            for item in conversations:
                chat_template.append({
                    'role': item['role'],
                    'content': item['content'].strip()
                })

            # 直接 tokenize，避免字符串往返（修复 redundant encode）
            token_ids = tokenizer.apply_chat_template(
                chat_template,
                add_answer_tag_for_assistant=False
            )

            if len(token_ids) <= MAX_SFT_TOKEN_LENGTH:
                tokens.append(np.array(token_ids, dtype=dtype))

    # 注入 self-cognition 数据
    self_cog_tokens = _get_self_cognition_tokens(dtype)
    tokens.extend(self_cog_tokens * SELF_COGNITION_MULTIPLIER)
    print(f"[encode-sft] self-cognition: {len(self_cog_tokens)} samples × {SELF_COGNITION_MULTIPLIER}")

    # 打乱并保存
    print(f"[encode-sft] 打乱 {len(tokens)} 条 ...")
    tokens = shuffle(tokens)

    output_path = os.path.join(output_dir, SFT_OUTPUT)
    np.save(output_path, np.array(tokens, dtype=object))
    print(f"[encode-sft] 保存到 {output_path}")

    # 清理临时文件
    os.remove(input_path)
