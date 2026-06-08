"""
阶段 2: Python 原生打乱（替代 terashuf）
"""

import glob
import os
import random

from .config import RAW_DIR, SHUFFLE_SEED, TMP_DIR


def shuffle_file(input_path, output_path, seed=None):
    """
    读取 input_path 所有行，随机打乱，写入 output_path。

    对于 MB 级小文件，直接全读入内存后 random.shuffle。
    足够简单、可移植，且结果可复现（固定 seed）。
    """
    if seed is None:
        seed = SHUFFLE_SEED
    rng = random.Random(seed)

    with open(input_path) as f:
        lines = f.readlines()

    rng.shuffle(lines)

    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"[shuffle] {input_path} → {output_path} ({len(lines)} lines, seed={seed})")


def shuffle_all_raw_files(raw_dir=None, tmp_dir=None):
    """
    打乱 data/raw/*.jsonl 中的所有文件，输出到 data/tmp/shuffle_<name>.jsonl
    """
    raw_dir = raw_dir or RAW_DIR
    tmp_dir = tmp_dir or TMP_DIR
    os.makedirs(tmp_dir, exist_ok=True)

    for file in sorted(glob.glob(os.path.join(raw_dir, '*.jsonl'))):
        basename = os.path.basename(file)
        output = os.path.join(tmp_dir, f'shuffle_{basename}')
        if not os.path.exists(output):
            shuffle_file(file, output)


def shuffle_pretrain_files(tmp_dir=None):
    """
    第 5 阶段：对 pretrain_data_short.jsonl 和 pretrain_data_long.jsonl 再次打乱，
    生成 shuffle_pretrain_data_short.jsonl 和 shuffle_pretrain_data_long.jsonl。
    打乱后删除原始文件。
    """
    tmp_dir = tmp_dir or TMP_DIR

    for name in ('pretrain_data_short.jsonl', 'pretrain_data_long.jsonl'):
        src = os.path.join(tmp_dir, name)
        dst = os.path.join(tmp_dir, f'shuffle_{name}')
        if not os.path.exists(dst) and os.path.exists(src):
            shuffle_file(src, dst)
            os.remove(src)
