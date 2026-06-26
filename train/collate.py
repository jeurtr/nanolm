"""Collate functions and batch manipulation utilities for training."""

import torch
from torch.nn.utils.rnn import pad_sequence

from .tools import TrainerTools


def pretrain_collate_fn(batch_data):
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
    labels = pad_sequence(batch_data, batch_first=True, padding_value=-100)
    return {
        'inputs': inputs,
        'labels': labels
    }


def get_sft_collate_fn(mask_prompt: bool):
    def sft_collate_fn(batch_data):
        batch_train_data = []
        image_tags = []
        for item in batch_data:
            batch_train_data.append(item['inputs'])
            image_tags.append(item['image_tag'])

        inputs = pad_sequence(batch_train_data, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
        labels = pad_sequence(batch_train_data, batch_first=True, padding_value=-100)

        if mask_prompt:
            labels = _mask_prompt(labels)

        return {
            'inputs': inputs,
            'labels': labels,
            'image_tags': image_tags
        }

    return sft_collate_fn


def get_dpo_collate_fn(mask_prompt: bool):
    def dpo_collate_fn(batch_data):
        chosen_inputs = []
        chosen_labels = []
        rejected_inputs = []
        rejected_labels = []

        max_len = 0
        for key in ['chosen', 'rejected']:
            max_len = max(max(len(item[key]) for item in batch_data), max_len)

        for item in batch_data:
            chosen_sequence = item['chosen']
            chosen_inputs.append(chosen_sequence + [TrainerTools().tokenizer.pad] * (max_len - len(chosen_sequence)))
            chosen_labels.append(chosen_sequence + [-100] * (max_len - len(chosen_sequence)))

            rejected_sequence = item['rejected']
            rejected_inputs.append(rejected_sequence + [TrainerTools().tokenizer.pad] * (max_len - len(rejected_sequence)))
            rejected_labels.append(rejected_sequence + [-100] * (max_len - len(rejected_sequence)))

        chosen_inputs = torch.tensor(chosen_inputs).long()
        chosen_labels = torch.tensor(chosen_labels).long()
        if mask_prompt:
            chosen_labels = _mask_prompt(chosen_labels)

        rejected_inputs = torch.tensor(rejected_inputs).long()
        rejected_labels = torch.tensor(rejected_labels).long()
        if mask_prompt:
            rejected_labels = _mask_prompt(rejected_labels)

        return {
            'chosen_inputs': chosen_inputs,
            'chosen_labels': chosen_labels,
            'rejected_inputs': rejected_inputs,
            'rejected_labels': rejected_labels
        }

    return dpo_collate_fn


def split_batch(data_per_batch: dict) -> list[dict]:
    group_size = data_per_batch['sequence_ids'].size(0)
    group_data = [{} for _ in range(group_size)]

    keys = (
        'sequence_ids',
        'old_log_probs',
        'ref_log_probs',
        'advantages',
        'attention_mask',
        'mask',
    )

    for key in keys:
        value = data_per_batch[key]
        if value is None:
            vals = [None] * group_size
        else:
            vals = torch.unbind(value)

        for i, v in enumerate(vals):
            group_data[i][key] = v

    return group_data


def join_batch(batch_data: list[dict]) -> dict:
    result = {}
    keys = (
        'sequence_ids',
        'old_log_probs',
        'ref_log_probs',
        'advantages',
        'attention_mask',
        'mask',
    )

    for key in keys:
        vals = [item[key] for item in batch_data]
        if all(v is not None for v in vals):
            data = _zero_pad_sequences(vals, 'left')
        else:
            data = None
        result[key] = data

    return result


_use_origin_pad_sequence = True


def left_pad_sequence(
        sequences: torch.Tensor | list[torch.Tensor],
        padding_value: float,
) -> torch.Tensor:
    global _use_origin_pad_sequence

    if _use_origin_pad_sequence:
        try:
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value, padding_side='left')
        except TypeError:
            _use_origin_pad_sequence = False
            return left_pad_sequence(sequences, padding_value)
    else:
        reversed_sequences = [seq.flip(dims=(0,)) for seq in sequences]
        padded_reversed = pad_sequence(reversed_sequences, batch_first=True, padding_value=padding_value)
        return padded_reversed.flip(dims=(1,))


def _zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = 'left'
) -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(torch.nn.functional.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def _mask_prompt(labels):
    tokenizer = TrainerTools().tokenizer
    system_id = tokenizer.system
    user_id = tokenizer.user
    end_id = tokenizer.end
    assistant_id = tokenizer.assistant
    ignore_index = -100

    for i in range(labels.shape[0]):
        row = labels[i]
        seq_len = len(row)

        starts = torch.nonzero((row == system_id) | (row == user_id)).view(-1)
        ends = torch.nonzero(row == end_id).view(-1)

        if starts.numel() == 0:
            continue

        start_idx_ptr = 0
        while start_idx_ptr < len(starts):
            s_pos = starts[start_idx_ptr].item()

            e_idx = torch.searchsorted(ends, s_pos, right=True).item()

            if e_idx >= len(ends):
                row[s_pos:] = ignore_index
                break

            e_pos = ends[e_idx].item()
            mask_end = e_pos

            if e_pos + 1 < seq_len and row[e_pos + 1] == assistant_id:
                mask_end = e_pos + 1

            row[s_pos: mask_end + 1] = ignore_index

            next_s_idx = torch.searchsorted(starts, mask_end, right=True).item()
            start_idx_ptr = next_s_idx

    return labels
