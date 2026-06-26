import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from .tools import TrainerTools


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device_type):
    if TrainerTools().use_amp:
        from nanolm.device import get_autocast_dtype
        return torch.autocast(
            device_type=device_type,
            dtype=get_autocast_dtype(),
            enabled=True,
            cache_enabled=None
        )
    else:
        return nullcontext()




def create_doc_boundary_mask(
        input_ids: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    """
    根据文档结束符 (eot) 的位置，创建一个 attention mask 来阻止跨文档的注意力。

    这个函数生成的 mask 会阻止一个 token 关注 (attend to) 属于前面文档的 tokens。
    例如，对于输入 `[[1, 2, eot, 3, 4, eot]]`，
    tokens `3` 和 `4` 将无法关注 `1`, `2`, 和第一个 `eot`。

    Args:
        input_ids (torch.Tensor): 输入的 token ID 张量，形状为 (bsz, seq_len)。
        dtype (torch.dtype): 数据类型。

    Returns:
        torch.Tensor: 符合 attention 机制要求的 mask 张量，
                      形状为 (bsz, 1, seq_len, seq_len)。
                      值为 -inf 的位置表示被屏蔽，值为 0 的位置表示允许注意力。
    """
    # 获取 batch size 和 sequence length
    bsz, seq_len = input_ids.shape

    # 1. 确定每个 eot_token 的位置
    # is_eot 是一个布尔张量，形状为 (bsz, seq_len)
    is_eot = (input_ids == TrainerTools().tokenizer.end)

    # 2. 为每个 token 分配一个文档 ID
    # 我们使用 cumsum (累加和) 来创建递增的文档 ID。一个 token 所属的文档 ID，
    # 取决于它前面有多少个 eot。
    # 示例:
    # input_ids:        [[1, 2, 3, eot, 4, 5, eot]]
    # is_eot:           [F, F, F, T, F, F, T] -> [0, 0, 0, 1, 0, 0, 1]
    # doc_ids_ending:   [0, 0, 0, 1, 1, 1, 2] (cumsum 的结果)
    # doc_ids:          [0, 0, 0, 0, 1, 1, 1] (向右移位后的结果)
    # 这个结果正确地将文档 0 分配给了前四个 token，将文档 1 分配给了后三个 token。
    doc_ids_ending = torch.cumsum(is_eot, dim=-1)
    doc_ids = F.pad(doc_ids_ending[:, :-1], (1, 0), value=0)

    # 3. 通过比较 query 和 key 的文档 ID 来创建 mask
    # 我们的目标是：当 query token 所在的文档 ID 大于 key token 所在的文档 ID 时，进行屏蔽。
    # query_doc_ids 形状: (bsz, seq_len, 1)
    # key_doc_ids 形状:   (bsz, 1, seq_len)
    query_doc_ids = doc_ids.unsqueeze(2)
    key_doc_ids = doc_ids.unsqueeze(1)

    # 利用 PyTorch 的广播机制，`query_doc_ids > key_doc_ids` 会创建一个
    # 形状为 (bsz, seq_len, seq_len) 的布尔张量。
    # 当 query 的文档 ID 大于 key 的文档 ID 时，值为 True，这正是我们需要屏蔽的位置。
    boundary_mask = query_doc_ids > key_doc_ids

    # 4. 将布尔 mask 转换为 attention 机制所需的浮点数 mask (-inf 和 0)
    final_mask = torch.zeros(
        (bsz, seq_len, seq_len), device=input_ids.device, dtype=dtype
    )
    final_mask.masked_fill_(boundary_mask, torch.finfo(dtype).min)

    # 5. 增加一个维度以匹配 attention head 的输入要求 (bsz, num_heads, seq_len, seq_len)
    #    这里我们只生成一个 mask，它可以被广播到所有的 head。
    return final_mask.unsqueeze(1)


def generate_position_ids(input_ids: torch.Tensor):
    """
    为打包序列生成 position_ids 张量。

    参数:
      input_ids (torch.Tensor): 输入的 token ID 张量 (batch_size, sequence_length)。
      end_of_text_id (int): 代表文本结束的特殊 token ID。

    返回:
      torch.Tensor: 生成的 position_ids 张量。
    """
    # 获取输入张量的形状
    batch_size, seq_length = input_ids.shape

    # 创建一个与输入形状相同，全为0的张量来存储position_ids
    # 第一个token的位置永远是0，所以这个初始化是正确的
    position_ids = torch.zeros_like(input_ids, dtype=torch.long)

    # 从第二个时间步 (t=1) 开始遍历整个序列
    for t in range(1, seq_length):
        # 检查前一个时间步 (t-1) 的token是否为 EOT token
        # 这会为批次中的每个序列生成一个布尔值
        is_reset_token = (input_ids[:, t - 1] == TrainerTools().tokenizer.end)

        # 获取前一个时间步的位置ID
        prev_position_ids = position_ids[:, t - 1]

        # 如果前一个token是EOT，当前位置重置为0；否则，在前一个位置上加1
        # torch.where 会根据 is_reset_token 的布尔值进行选择
        position_ids[:, t] = torch.where(is_reset_token, 0, prev_position_ids + 1)

    return position_ids


def calc_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    根据 attention_mask 计算 position_ids，主要用于 Left Padding 场景。
    mask: [0, 0, 1, 1, 1] -> position_ids: [0, 0, 0, 1, 2]
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def repeat_image_tok(
        tokens: torch.Tensor,
        tokens_per_image: int,
        attention_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # tokens_per_image=3 -> <image>...xxxx -> <image><image><image>...xxx
    image_tok = TrainerTools().tokenizer.image
    mask = (tokens == image_tok)
    if not mask.any():
        return tokens, attention_mask

    # 计算每个位置的重复次数：默认为1，image token 位置为 tokens_per_image
    repeats = torch.ones_like(tokens, dtype=torch.long)
    repeats[mask] = tokens_per_image

    # 使用 repeat_interleave 进行高效扩展
    new_tokens = torch.repeat_interleave(tokens, repeats, dim=0)

    if attention_mask is not None:
        # 对 mask 做同样的操作
        new_mask = torch.repeat_interleave(attention_mask, repeats, dim=0)
        return new_tokens, new_mask

    return new_tokens, None


def batch_repeat_image_tok(
        tokens: torch.Tensor,
        tokens_per_image: int,
        attention_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    new_tokens_list = []
    new_masks_list = []
    has_mask = attention_mask is not None

    for i in range(len(tokens)):
        token_seq = tokens[i]
        mask_seq = attention_mask[i] if has_mask else None

        if has_mask:
            new_tok, new_mask = repeat_image_tok(token_seq, tokens_per_image, mask_seq)
            new_tokens_list.append(new_tok)
            new_masks_list.append(new_mask)
        else:
            new_tok, _ = repeat_image_tok(token_seq, tokens_per_image)
            new_tokens_list.append(new_tok)

    ret_tokens = torch.stack(new_tokens_list, dim=0)
    if has_mask:
        ret_masks = torch.stack(new_masks_list, dim=0)
        return ret_tokens, ret_masks

    return ret_tokens, None


_use_memory_efficient_log_softmax = True
def log_softmax(logits, index) -> torch.Tensor:
    if _use_memory_efficient_log_softmax:
        return _selective_log_softmax(logits, index)

    # Convert raw logits into log probabilities along the vocabulary axis.
    # [batch_size, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    # Reshape input_ids from (batch_size, seq_len) to (batch_size, seq_len, 1) for gathering.
    # Then, gather the log probability for each token in input_ids.
    selected_log_probs = log_probs.gather(dim=-1, index=index.unsqueeze(-1))

    # Remove the extra last dimension to get back to shape (batch_size, seq_len).
    return selected_log_probs.squeeze(-1)


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = _masked_mean(values, mask), _masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def truncate_sequences_at_eos(
        sequences: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int
) -> torch.Tensor:
    """
    高效地将批处理中的序列在第一个EOS标记处截断。
    第一个EOS标记之后的所有内容（不包括EOS自身）将被替换为pad_token_id。

    这是一个向量化的实现，以确保在GPU上的性能。
    它使用 torch.where，因此不依赖于 pad_token_id 必须为0。

    Args:
        sequences (torch.Tensor): 批处理序列, 形状为 (batch_size, seq_len)。
        eos_token_id (int): 句子结束标记的ID。
        pad_token_id (int): 填充标记的ID。

    Returns:
        torch.Tensor: 截断后的序列，形状与输入相同。
    """
    # 创建一个布尔掩码，标记所有EOS token的位置
    eos_mask = (sequences == eos_token_id)

    # 找到每行中第一个True（即第一个EOS token）的索引
    # .int() 是为了兼容旧版torch，argmax需要非布尔类型
    first_eos_indices = torch.argmax(eos_mask.int(), dim=1)

    # 检查哪些序列确实包含了EOS token。
    # 如果某一行完全没有EOS, argmax会返回0, 这会产生歧义。
    has_eos = eos_mask.any(dim=1)

    # 对于没有EOS token的序列，将截断索引设置为序列最大长度，以防错误截断
    first_eos_indices[~has_eos] = sequences.shape[1]

    # 创建一个 [0, 1, 2, ..., seq_len-1] 的索引张量
    indices_mask = torch.arange(sequences.shape[1], device=sequences.device)

    # 利用广播机制创建一个掩码，标记所有应保留的token
    # 对于每个序列，当 token_index < first_eos_index 时为True
    keep_mask = indices_mask < first_eos_indices.unsqueeze(1)

    # 使用 torch.where 进行安全替换
    # 如果 keep_mask 为 True，则保留原始序列的token，否则替换为 pad_token_id
    truncated_sequences = torch.where(
        keep_mask,
        sequences,
        pad_token_id
    )

    return truncated_sequences


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: bool | None = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def _masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = _masked_mean(values, mask)
    centered_values = values - mean
    variance = _masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=values.device, dtype=values.dtype)

        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def _selective_log_softmax(logits, index) -> torch.Tensor:
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=False):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps




