"""
数据质量校验：长度分布统计、去重、输出文件验证
"""

import json
import os

import numpy as np

from .config import (
    MIDTRAIN_OUTPUT_PREFIX,
    OUTPUT_DIR,
    PPO_OUTPUT,
    PRETRAIN_OUTPUT_PREFIX,
    SFT_OUTPUT,
    VOCAB_THRESHOLD_UINT16,
)


def compute_length_distribution(file_path):
    """对 JSONL 文件的文本长度进行统计"""
    lengths = []

    with open(file_path) as f:
        for line in f:
            item = json.loads(line)
            text = item.get('text', '')
            lengths.append(len(text))

    if not lengths:
        print("[quality] 文件为空")
        return

    arr = np.array(lengths)
    print(f"[quality] {file_path}:")
    print(f"  样本数: {len(arr)}")
    print(f"  最小: {arr.min()}")
    print(f"  最大: {arr.max()}")
    print(f"  均值: {arr.mean():.1f}")
    print(f"  中位数: {np.percentile(arr, 50):.0f}")
    print(f"  P95: {np.percentile(arr, 95):.0f}")
    print(f"  P99: {np.percentile(arr, 99):.0f}")

    # 直方图
    bins = [0, 128, 256, 512, 768, 1024, 2048, 4096, 10**6]
    hist, _ = np.histogram(arr, bins=bins)
    print(f"  长度分布: {dict(zip([f'<{b}' for b in bins[1:]], hist, strict=False))}")


def deduplicate_file(input_path, output_path, field='text'):
    """基于指定字段去重 JSONL 文件"""
    seen = set()
    kept = 0
    removed = 0

    with open(input_path) as f_in:
        with open(output_path, 'w') as f_out:
            for line in f_in:
                item = json.loads(line)
                key = item.get(field, line)
                if key in seen:
                    removed += 1
                else:
                    seen.add(key)
                    f_out.write(line)
                    kept += 1

    print(f"[quality] 去重: {input_path} → {output_path}")
    print(f"  保留: {kept}, 移除: {removed} ({100*removed/(kept+removed):.1f}%)")


def validate_tokenized_npy(npy_path):
    """验证 .npy 文件的完整性和 token 合法性"""
    print(f"[quality] 验证 {npy_path} ...")

    try:
        data = np.load(npy_path, allow_pickle=True, mmap_mode='r')
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return False

    if data.size == 0:
        print("  ✗ 文件为空")
        return False

    print(f"  dtype: {data.dtype}, shape: {data.shape}")

    # flat array (pretrain/midtrain)
    if data.dtype != np.dtype('object'):
        total_tokens = data.size
        # 检查 token 范围
        max_token = int(data.max())
        min_token = int(data.min())
        print(f"  tokens: {total_tokens:,}, min: {min_token}, max: {max_token}")
        if max_token >= VOCAB_THRESHOLD_UINT16 * 2:
            print(f"  ⚠ max token ({max_token}) 可能超出 vocab 范围")
    else:
        # object array (SFT/PPO)
        total_seqs = data.size
        lengths = [len(item) for item in data]
        total_tokens = sum(lengths)
        print(f"  sequences: {total_seqs:,}, total tokens: {total_tokens:,}")
        print(f"  seq len: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.0f}")

    print("  ✓ 通过")
    return True


def validate_pipeline_outputs(output_dir=None):
    """验证所有管线输出文件"""
    output_dir = output_dir or OUTPUT_DIR

    print("\n========== 数据质量校验 ==========\n")

    for prefix in [PRETRAIN_OUTPUT_PREFIX, MIDTRAIN_OUTPUT_PREFIX]:
        found = False
        for i in range(10):
            path = os.path.join(output_dir, f'{prefix}_{i}.npy')
            if os.path.exists(path):
                validate_tokenized_npy(path)
                found = True
        if not found:
            print(f"[quality] {prefix}_*.npy 未找到")

    for name in [SFT_OUTPUT, PPO_OUTPUT]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            validate_tokenized_npy(path)
        else:
            print(f"[quality] {name} 未找到")

    print("\n==================================\n")
