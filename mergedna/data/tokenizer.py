"""Character-level DNA tokenizer for MergeDNA."""

import torch
from typing import List, Dict, Optional


class DNACharTokenizer:
    """Character-level DNA tokenizer mapping each nucleotide to a token ID.

    Vocabulary:
        [PAD]=0, [CLS]=1, [SEP]=2, [MASK]=3, [UNK]=4,
        A=5, T=6, C=7, G=8, N=9
    """

    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[MASK]": 3,
        "[UNK]": 4,
    }
    NUCLEOTIDE_TOKENS = {"A": 5, "T": 6, "C": 7, "G": 8, "N": 9}

    def __init__(self, max_length: int = 4096):
        self.max_length = max_length
        self.vocab = {**self.SPECIAL_TOKENS, **self.NUCLEOTIDE_TOKENS}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.SPECIAL_TOKENS["[PAD]"]
        self.cls_token_id = self.SPECIAL_TOKENS["[CLS]"]
        self.sep_token_id = self.SPECIAL_TOKENS["[SEP]"]
        self.mask_token_id = self.SPECIAL_TOKENS["[MASK]"]
        self.unk_token_id = self.SPECIAL_TOKENS["[UNK]"]

    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        """Encode a DNA sequence string to token IDs."""
        ids = []
        if add_special_tokens:
            ids.append(self.cls_token_id)

        for c in sequence.upper():
            ids.append(self.vocab.get(c, self.unk_token_id))

        if add_special_tokens:
            ids.append(self.sep_token_id)

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to a DNA sequence string."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, "")
            if tok not in self.SPECIAL_TOKENS:
                tokens.append(tok)
        return "".join(tokens)

    def __call__(
        self,
        sequences: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of DNA sequences.

        Returns dict with 'input_ids' and 'attention_mask'.
        """
        max_len = max_length or self.max_length

        batch_ids = []
        batch_mask = []

        for seq in sequences:
            ids = self.encode(seq, add_special_tokens=False)

            if truncation and len(ids) > max_len:
                ids = ids[:max_len]

            attn_mask = [1] * len(ids)

            if padding and len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                attn_mask = attn_mask + [0] * pad_len

            batch_ids.append(ids)
            batch_mask.append(attn_mask)

        result = {
            "input_ids": batch_ids,
            "attention_mask": batch_mask,
        }

        if return_tensors == "pt":
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}

        return result
