"""
文件数据集 — 训练脚本与预处理数据文件之间的桥梁

提供自动下载、预取、缓存清理能力。
训练配置工厂 (utils.py) 引用此模块来定义各阶段的数据文件列表。
"""

import os
import threading

from modelscope import dataset_snapshot_download

from train import FileDataset, TrainerTools

from .config import data_root_dir


class FileDatasetBase(FileDataset):
    """
    文件数据集基类。

    功能：
    - __getitem__(idx) 返回文件的完整路径
    - 文件不存在时自动从 ModelScope 下载
    - 后台线程预取下一个文件，处理完后删除前一个文件
    """

    def __init__(self, file_names: list):
        self.file_names = file_names

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx) -> str:
        file_path = f"{data_root_dir()}{self.file_names[idx]}"

        # 下载当前文件（如果不存在）
        if not os.path.exists(file_path):
            if TrainerTools().parallel.is_main_process:
                dataset_snapshot_download(
                    'qibin0506/Cortex-3.0-data',
                    allow_file_pattern=[self.file_names[idx]],
                    local_dir=data_root_dir()
                )
            TrainerTools().parallel.wait()

        # 预下载下一个文件
        if idx < len(self.file_names) - 1 and TrainerTools().parallel.is_main_process:
            next_file = self.file_names[idx + 1]
            dst_file = f'{data_root_dir()}{next_file}'
            if not os.path.exists(dst_file):
                threading.Thread(
                    target=dataset_snapshot_download,
                    kwargs={
                        'dataset_id': 'qibin0506/Cortex-3.0-data',
                        'allow_file_pattern': [next_file],
                        'local_dir': data_root_dir()
                    }
                ).start()

        # 删除前一个文件
        if idx > 0 and TrainerTools().parallel.is_main_process:
            prev_file = self.file_names[idx - 1]
            if os.path.exists(f'{data_root_dir()}{prev_file}'):
                os.remove(f'{data_root_dir()}{prev_file}')

        return file_path


class PretrainFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'pretrain_data_0.npy',
            'pretrain_data_1.npy',
        ])


class MidtrainFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'midtrain_data_0.npy',
        ])


class SFTFileDataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'sft_data.npy',
        ])


class PPODataset(FileDatasetBase):
    def __init__(self):
        super().__init__([
            'ppo_data.npy',
        ])
