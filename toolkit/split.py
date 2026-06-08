"""
阶段 3: 拆分 SFT-2048 数据
"""

import os

from .config import TMP_DIR
from .utils import get_file_line_count


def split_sft_data(tmp_dir=None):
    """
    将 shuffle_sft_2048.jsonl 按 shuffle_sft_mini_512.jsonl 的行数拆分：
    前 N 行 → shuffle_sft_mini_2048.jsonl (SFT)
    剩余   → shuffle_pretrain_2048.jsonl (预训练)

    处理后删除原始 shuffle_sft_2048.jsonl。
    """
    tmp_dir = tmp_dir or TMP_DIR

    ref_path = os.path.join(tmp_dir, 'shuffle_sft_mini_512.jsonl')
    sft_2048_path = os.path.join(tmp_dir, 'shuffle_sft_2048.jsonl')
    sft_output = os.path.join(tmp_dir, 'shuffle_sft_mini_2048.jsonl')
    pretrain_output = os.path.join(tmp_dir, 'shuffle_pretrain_2048.jsonl')

    if not os.path.exists(sft_2048_path):
        print(f"[split] {sft_2048_path} 不存在，跳过")
        return

    line_count = get_file_line_count(ref_path)

    sft_lines = []
    pretrain_lines = []

    with open(sft_2048_path) as f:
        for idx, line in enumerate(f):
            if idx < line_count:
                sft_lines.append(line)
            else:
                pretrain_lines.append(line)

    with open(sft_output, 'w') as f:
        f.writelines(sft_lines)
    with open(pretrain_output, 'w') as f:
        f.writelines(pretrain_lines)

    os.remove(sft_2048_path)

    print(f"[split] SFT: {len(sft_lines)} 行 → {sft_output}")
    print(f"[split] Pretrain: {len(pretrain_lines)} 行 → {pretrain_output}")
