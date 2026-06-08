"""
阶段 7: 合并 SFT 数据文件
"""

import os

from .config import SFT_MERGE_INPUTS, TMP_DIR


def merge_sft_files(tmp_dir=None):
    """
    将 SFT_MERGE_INPUTS 中列出的文件合并为 sft_data.jsonl。
    合并后删除输入文件。
    """
    tmp_dir = tmp_dir or TMP_DIR
    output_path = os.path.join(tmp_dir, 'sft_data.jsonl')
    lines = []

    for filename in SFT_MERGE_INPUTS:
        filepath = os.path.join(tmp_dir, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            lines.extend(f.readlines())
        os.remove(filepath)

    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"[merge] {len(lines)} 行合并到 {output_path}")
