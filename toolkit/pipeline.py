"""
数据预处理管线编排器

支持全量运行或按阶段选择性执行。
每阶段会检查输出是否已存在，存在则跳过（除非 force=True）。
"""

import os

from .config import OUTPUT_DIR, RAW_DIR, TMP_DIR, ensure_dirs
from .download import download_raw_datasets
from .encode import encode_pretrain_and_midtrain
from .encode_ppo import encode_ppo_data
from .encode_sft import encode_sft_data
from .merge import merge_sft_files
from .preprocess import preprocess_pretrain_data
from .quality import validate_pipeline_outputs
from .shuffle import shuffle_all_raw_files, shuffle_pretrain_files
from .split import split_sft_data
from .utils import setup_preprocessing_env

STAGES = [
    ('download',  '下载原始数据集'),
    ('shuffle',   '打乱所有原始 JSONL'),
    ('split',     '拆分 SFT-2048'),
    ('preprocess','短/长文本分类'),
    ('shuffle2',  '再次打乱预训练数据'),
    ('encode',    'Tokenize pretrain/midtrain'),
    ('merge',     '合并 SFT 文件'),
    ('encode_sft','Tokenize SFT'),
    ('encode_ppo','Tokenize PPO'),
    ('validate',  '数据质量校验'),
]


class DataPipeline:
    def __init__(self, force=False, skip_quality=False, seed=None):
        self.force = force
        self.skip_quality = skip_quality
        if seed is not None:
            os.environ['NANOLM_SHUFFLE_SEED'] = str(seed)

    def run_full(self):
        """运行完整管线"""
        setup_preprocessing_env()
        ensure_dirs(RAW_DIR, TMP_DIR, OUTPUT_DIR)

        print("=" * 50)
        print("NanoLM 数据预处理管线")
        print("=" * 50)

        self._do_download()
        self._do_shuffle()
        self._do_split()
        self._do_preprocess()
        self._do_shuffle2()
        self._do_encode()
        self._do_merge()
        self._do_encode_sft()
        self._do_encode_ppo()

        if not self.skip_quality:
            self._do_validate()

        print("\n✓ 管线完成")

    def run_stages(self, stage_names):
        """运行指定阶段"""
        setup_preprocessing_env()
        ensure_dirs(RAW_DIR, TMP_DIR, OUTPUT_DIR)

        stage_map = {name: method for name, method in [
            ('download',    self._do_download),
            ('shuffle',     self._do_shuffle),
            ('split',       self._do_split),
            ('preprocess',  self._do_preprocess),
            ('shuffle2',    self._do_shuffle2),
            ('encode',      self._do_encode),
            ('merge',       self._do_merge),
            ('encode_sft',  self._do_encode_sft),
            ('encode_ppo',  self._do_encode_ppo),
            ('validate',    self._do_validate),
        ]}

        for name in stage_names:
            if name in stage_map:
                stage_map[name]()
            else:
                print(f"未知阶段: {name}")

    def _do_download(self):
        print("\n--- 阶段 1: 下载原始数据集 ---")
        # 检查是否已存在
        if not self.force and os.path.exists(RAW_DIR) and os.listdir(RAW_DIR):
            print("[跳过] 原始数据已存在，使用 --force 强制重新下载")
            return
        download_raw_datasets()

    def _do_shuffle(self):
        print("\n--- 阶段 2: 打乱原始数据 ---")
        shuffle_all_raw_files()

    def _do_split(self):
        print("\n--- 阶段 3: 拆分 SFT-2048 ---")
        split_sft_data()

    def _do_preprocess(self):
        print("\n--- 阶段 4: 短/长文本分类 ---")
        preprocess_pretrain_data()

    def _do_shuffle2(self):
        print("\n--- 阶段 5: 再次打乱 ---")
        shuffle_pretrain_files()

    def _do_encode(self):
        print("\n--- 阶段 6: Tokenize pretrain/midtrain ---")
        encode_pretrain_and_midtrain()

    def _do_merge(self):
        print("\n--- 阶段 7: 合并 SFT ---")
        merge_sft_files()

    def _do_encode_sft(self):
        print("\n--- 阶段 8: Tokenize SFT ---")
        encode_sft_data()

    def _do_encode_ppo(self):
        print("\n--- 阶段 9: Tokenize PPO ---")
        encode_ppo_data()

    def _do_validate(self):
        print("\n--- 质量校验 ---")
        validate_pipeline_outputs()

    @classmethod
    def list_stages(cls):
        """列出所有可用阶段"""
        for name, desc in STAGES:
            print(f"  {name:<12} {desc}")
