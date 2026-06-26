import os

from model import ModelConfig, RoPEConfig


def init_env():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKEN_DIR'] = './tokens'
    os.environ['LOG_DIR'] = './log/'

    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'

    os.environ['CKPT_MAX_TO_KEEP'] = '2'
    os.environ['SAVE_BEST_CHECKPOINT'] = '0'  # or '1'


def get_eval_prompt(content: str, tokenizer) -> str:
    chat_template = [
        {'role': 'system', 'content': ' '},
        {'role': 'user', 'content': content}
    ]

    chat_template = tokenizer.apply_chat_template(chat_template, tokenizer=False)
    return f'{chat_template}<assistant>'


def get_model_config(long_context=False, vocab_size=8192):
    max_position_embeddings = 2048 if long_context else 512
    original_max_position_embeddings = 512 if long_context else None
    rope_type = 'yarn' if long_context else 'default'

    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=768,
        intermediate_size=2048,

        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,

        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        attention_dropout=0.0,
        tie_word_embeddings=True,

        rope_config=RoPEConfig(
            rope_type=rope_type,
            rope_theta=10000.0,
        )
    )
