"""
阶段 4: 短/长文本分类，SFT 对话转纯文本
"""

import json
import os

from tqdm import tqdm

from .config import (
    SFT_SPLIT_FILES,
    SHORT_TEXT_THRESHOLD,
    TMP_DIR,
)
from .utils import sft_to_text


def preprocess_pretrain_data(tmp_dir=None):
    """
    1. 读取 shuffle_pretrain_hq.jsonl → 按 SHORT_TEXT_THRESHOLD 分为短/长
    2. 读取 SFT_SPLIT_FILES → 转为纯文本 → 同样分为短/长
    3. 合并输出 pretrain_data_short.jsonl 和 pretrain_data_long.jsonl
    4. 处理后删除所有中间文件
    """
    tmp_dir = tmp_dir or TMP_DIR
    threshold = SHORT_TEXT_THRESHOLD

    short_data = []
    long_data = []

    # --- 处理 pretrain HQ 数据 ---
    hq_path = os.path.join(tmp_dir, 'shuffle_pretrain_hq.jsonl')
    if os.path.exists(hq_path):
        print("[preprocess] 处理 pretrain_hq.jsonl ...")
        with open(hq_path) as f:
            for line in f:
                text = json.loads(line)['text'].replace('<|im_end|>', '')
                record = json.dumps({'text': text}, ensure_ascii=False)
                if len(text) <= threshold:
                    short_data.append(f"{record}\n")
                else:
                    long_data.append(f"{record}\n")
        os.remove(hq_path)
        print(f"[preprocess] pretrain_hq: short={len(short_data)}, long={len(long_data)}")

    # --- 处理 SFT 文件 ---
    for filename in SFT_SPLIT_FILES:
        filepath = os.path.join(tmp_dir, filename)
        if not os.path.exists(filepath):
            continue

        print(f"[preprocess] 处理 {filename} ...")
        with open(filepath) as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=filename):
            text = sft_to_text(line)
            record = json.dumps({'text': text}, ensure_ascii=False)
            if len(text) <= threshold:
                short_data.append(f"{record}\n")
            else:
                long_data.append(f"{record}\n")

        os.remove(filepath)

    # --- 写入输出 ---
    short_out = os.path.join(tmp_dir, 'pretrain_data_short.jsonl')
    long_out = os.path.join(tmp_dir, 'pretrain_data_long.jsonl')

    with open(short_out, 'w') as f:
        f.writelines(short_data)
    del short_data

    with open(long_out, 'w') as f:
        f.writelines(long_data)
    del long_data

    print(f"[preprocess] 输出: {short_out}, {long_out}")
