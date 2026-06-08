"""
阶段 6: Tokenize 预训练和 midtrain 数据 → .npy
"""

import os

import numpy as np
from tqdm import tqdm

from train import TrainerTools

from .config import (
    MIDTRAIN_OUTPUT_PREFIX,
    MIDTRAIN_SPLIT_COUNT,
    OUTPUT_DIR,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_OUTPUT_PREFIX,
    PRETRAIN_SPLIT_COUNT,
    TMP_DIR,
)
from .utils import convert_bin_to_npy, determine_dtype, get_file_line_count


def _process_and_save_stream(input_path, output_prefix, split_count, batch_size, dtype):
    """流式读取 JSONL、批量 tokenize、写入分块 .npy 文件"""
    file_lines = get_file_line_count(input_path)
    lines_per_chunk = file_lines // split_count if split_count > 0 else file_lines

    current_chunk_idx = 0
    current_token_count = 0

    bin_path = f"{output_prefix}_{current_chunk_idx}.bin"
    bin_file = open(bin_path, "wb")
    text_buffer = []

    tokenizer = TrainerTools().tokenizer

    def _flush_buffer(bin_fh):
        nonlocal current_token_count
        if not text_buffer:
            return
        batch_encodings = tokenizer.batch_encode(text_buffer)
        for input_ids in batch_encodings:
            arr = np.array(input_ids, dtype=dtype)
            bin_fh.write(arr.tobytes())
            current_token_count += len(arr)
        text_buffer.clear()

    print(f"[encode] 处理 {input_path} ({file_lines} 行) ...")

    with open(input_path) as f:
        for idx, line in enumerate(tqdm(f, total=file_lines)):
            try:
                import orjson
                text = f"{orjson.loads(line)['text'].strip()}</s>"
            except Exception:
                import json
                text = f"{json.loads(line)['text'].strip()}</s>"
            text_buffer.append(text)

            if len(text_buffer) >= batch_size:
                _flush_buffer(bin_file)

            # 是否需要分块切换
            is_chunk_boundary = (
                split_count > 1
                and current_chunk_idx < split_count - 1
                and (idx + 1) % lines_per_chunk == 0
            )

            if is_chunk_boundary:
                _flush_buffer(bin_file)
                bin_file.close()
                print(f"[encode] 分块 {current_chunk_idx}: {current_token_count} tokens")
                npy_path = f"{output_prefix}_{current_chunk_idx}.npy"
                convert_bin_to_npy(bin_path, npy_path, dtype, current_token_count)
                os.remove(bin_path)

                current_chunk_idx += 1
                current_token_count = 0
                bin_path = f"{output_prefix}_{current_chunk_idx}.bin"
                bin_file = open(bin_path, "wb")

    # 处理最后的 buffer
    _flush_buffer(bin_file)
    bin_file.close()

    if current_token_count > 0:
        print(f"[encode] 分块 {current_chunk_idx}: {current_token_count} tokens")
        npy_path = f"{output_prefix}_{current_chunk_idx}.npy"
        convert_bin_to_npy(bin_path, npy_path, dtype, current_token_count)
        os.remove(bin_path)


def encode_pretrain_and_midtrain(tmp_dir=None, output_dir=None):
    """编码 pretrain（短文本）和 midtrain（长文本）数据"""
    tmp_dir = tmp_dir or TMP_DIR
    output_dir = output_dir or OUTPUT_DIR
    dtype = determine_dtype()

    # --- Pretrain (短文本) ---
    short_path = os.path.join(tmp_dir, 'shuffle_pretrain_data_short.jsonl')
    if os.path.exists(short_path):
        _process_and_save_stream(
            short_path,
            output_prefix=os.path.join(output_dir, PRETRAIN_OUTPUT_PREFIX),
            split_count=PRETRAIN_SPLIT_COUNT,
            batch_size=PRETRAIN_BATCH_SIZE,
            dtype=dtype,
        )
        os.remove(short_path)
    else:
        print("[encode] pretrain short data 不存在，跳过")

    # --- Midtrain (长文本) ---
    long_path = os.path.join(tmp_dir, 'shuffle_pretrain_data_long.jsonl')
    if os.path.exists(long_path):
        _process_and_save_stream(
            long_path,
            output_prefix=os.path.join(output_dir, MIDTRAIN_OUTPUT_PREFIX),
            split_count=MIDTRAIN_SPLIT_COUNT,
            batch_size=PRETRAIN_BATCH_SIZE,
            dtype=dtype,
        )
        os.remove(long_path)
    else:
        print("[encode] midtrain long data 不存在，跳过")
