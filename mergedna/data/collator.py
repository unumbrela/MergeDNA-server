"""Data collators for MergeDNA pre-training and fine-tuning."""

import torch
from typing import Dict, List, Any


class PretrainCollator:
    """Collator for pre-training. Pads sequences to the same length within a batch."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids = []
        attention_mask = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            pad_len = max_len - ids.shape[0]

            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])

            input_ids.append(ids)
            attention_mask.append(mask)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
        }


class FineTuneCollator:
    """Collator for fine-tuning. Pads sequences and collates labels."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            pad_len = max_len - ids.shape[0]

            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])

            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(f["labels"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
