"""
阶段 9: Tokenize PPO 数据
"""

import json
import os

import numpy as np
from sklearn.utils import shuffle

from train import TrainerTools

from .config import OUTPUT_DIR, PPO_OUTPUT, RAW_DIR


def encode_ppo_data(output_dir=None):
    """
    从 data/raw/rlaif-mini.jsonl 编码 PPO 提示数据。
    每行为 {'prompt': token_ids}
    """
    output_dir = output_dir or OUTPUT_DIR

    input_path = os.path.join(RAW_DIR, 'rlaif-mini.jsonl')

    if not os.path.exists(input_path):
        print(f"[encode-ppo] {input_path} 不存在，跳过")
        return

    tokens_list = []
    print(f"[encode-ppo] 处理 {input_path} ...")

    with open(input_path) as f:
        for line in f:
            user_content = json.loads(line)['conversations'][0]['content']
            chat_template = [
                {'role': 'system', 'content': ' '},
                {'role': 'user', 'content': user_content.strip()}
            ]
            item = TrainerTools().tokenizer.apply_chat_template(
                chat_template, tokenizer=False
            )
            tokens_list.append({
                'prompt': TrainerTools().tokenizer.encode(f'{item}<assistant>')
            })

    tokens_list = shuffle(tokens_list)

    output_path = os.path.join(output_dir, PPO_OUTPUT)
    np.save(output_path, np.array(tokens_list, dtype=object))
    print(f"[encode-ppo] {len(tokens_list)} 条保存到 {output_path}")
