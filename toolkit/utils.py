"""
数据预处理共享工具函数

从 process_data.py 提取，纯函数，无全局副作用。
"""

import json
import os
import re

import numpy as np

from train import TrainerTools


def get_file_line_count(file_path):
    with open(file_path) as f:
        return sum(1 for _ in f)


def determine_dtype(tokenizer=None):
    """根据词表大小选择 numpy dtype"""
    if tokenizer is None:
        tokenizer = TrainerTools().tokenizer
    return np.uint16 if tokenizer.vocab_size < 65535 else np.uint32


def _extra_think_and_answer(text: str):
    """解析 NanoLM 2.5 的 <think>...</think><answer>...</answer> 格式"""
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if not match:
        return '', text
    think_data = match.group(1)
    content = match.group(2)
    answer_match = re.search(r"<answer>(.*?)</answer>(.*)", content, re.DOTALL)
    if answer_match:
        content = answer_match.group(1)
    return think_data, content


def sft_to_text(line, assistant_name='NanoLM'):
    """将 SFT 对话 JSONL 行转换为纯文本（用于混入预训练数据）"""
    text = ''
    conversations = json.loads(line)['conversations']
    for conversation in conversations:
        content = conversation['content']
        if '<think>' in content:
            _, content = _extra_think_and_answer(content)
        text = f'{text}\n\n{content}'
    text = text.strip()
    text = text.replace('MiniMind-R1', assistant_name).replace('MiniMind', assistant_name)
    return text


def convert_bin_to_npy(bin_path, npy_path, dtype, shape):
    """将 raw binary token 文件包装为 .npy 格式（带 header）"""
    import shutil
    with open(bin_path, 'rb') as f_bin:
        with open(npy_path, 'wb') as f_npy:
            header = {
                'descr': np.dtype(dtype).str,
                'fortran_order': False,
                'shape': (shape,),
            }
            np.lib.format.write_array_header_1_0(f_npy, header)
            shutil.copyfileobj(f_bin, f_npy)


def setup_preprocessing_env():
    """配置预处理环境（与训练环境不同，需要 tokenizer 并行加速）"""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ['TOKEN_DIR'] = './tokens'


def iter_jsonl(file_path):
    """延迟读取 JSONL 文件，yield 每行的 dict"""
    with open(file_path) as f:
        for line in f:
            yield json.loads(line)
