"""
阶段 1: 从 ModelScope 下载原始数据集
"""

import os

from modelscope import dataset_snapshot_download

from .config import MINIMIND_DATASET, RAW_DIR, SELF_COGNITION_DATASET


def download_raw_datasets(force=False):
    """
    下载 minimind_dataset 和 self-cognition 原始数据到 data/raw/

    Args:
        force: 为 True 时强制重新下载
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"[download] 下载 {MINIMIND_DATASET} → {RAW_DIR}")
    dataset_snapshot_download(
        MINIMIND_DATASET,
        local_dir=RAW_DIR,
        allow_file_pattern='*.jsonl',
    )

    print(f"[download] 下载 {SELF_COGNITION_DATASET} → {RAW_DIR}")
    dataset_snapshot_download(
        SELF_COGNITION_DATASET,
        local_dir=RAW_DIR,
        allow_file_pattern='*.jsonl',
    )

    print("[download] 完成")
