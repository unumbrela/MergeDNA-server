"""Shared utilities for downstream experiment runners."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mergedna.data.tokenizer import DNACharTokenizer
from mergedna.model.mergedna import MergeDNA, MergeDNAConfig


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_result(
    result_path: str | Path,
    experiment_id: str,
    metrics: dict,
    started_at: str,
    finished_at: str,
    extra: Optional[dict] = None,
) -> dict:
    payload = {
        "experiment_id": experiment_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": max(
            0.0,
            datetime.fromisoformat(finished_at).timestamp()
            - datetime.fromisoformat(started_at).timestamp(),
        ),
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    save_json(result_path, payload)
    return payload


def load_result(result_path: str | Path) -> Optional[dict]:
    path = Path(result_path)
    if not path.exists():
        return None
    return load_json(path)


def load_backbone(config: dict) -> tuple[MergeDNA, DNACharTokenizer, torch.device]:
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    model_config = MergeDNAConfig(
        vocab_size=config.get("vocab_size", 10),
        embed_dim=config.get("embed_dim", 1024),
        num_heads=config.get("num_heads", 16),
        local_encoder_layers=config.get("local_encoder_layers", 4),
        latent_encoder_layers=config.get("latent_encoder_layers", 20),
        latent_decoder_layers=config.get("latent_decoder_layers", 4),
        local_decoder_layers=config.get("local_decoder_layers", 2),
        window_size=config.get("window_size", 16),
        dropout=config.get("dropout", 0.0),
        use_flash_attn=config.get("use_flash_attn", True),
        max_seq_length=config.get("max_seq_length", 4096),
        lambda_latent=config.get("lambda_latent", 0.25),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    )
    model = MergeDNA(model_config).to(device)
    ckpt_path = config.get("pretrain_ckpt")
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    tokenizer = DNACharTokenizer(max_length=config.get("max_seq_length", 4096))
    model.eval()
    return model, tokenizer, device


def deterministic_target_length(model: MergeDNA, length: int) -> int:
    mean = int(length * model.config.compression_target)
    mean = max(int(0.4 * length), min(mean, int(0.6 * length)))
    window = model.config.window_size
    return max(window, (mean // window) * window)


@dataclass
class WindowSpec:
    start: int
    end: int


def sliding_windows(
    seq_len: int,
    window_len: int,
    stride: Optional[int] = None,
) -> List[WindowSpec]:
    if seq_len <= window_len:
        return [WindowSpec(0, seq_len)]
    stride = stride or window_len
    windows = []
    start = 0
    while start < seq_len:
        end = min(start + window_len, seq_len)
        if end - start < window_len and end == seq_len:
            start = max(0, seq_len - window_len)
            end = seq_len
        windows.append(WindowSpec(start, end))
        if end == seq_len:
            break
        start += stride
    deduped = []
    seen = set()
    for w in windows:
        key = (w.start, w.end)
        if key not in seen:
            deduped.append(w)
            seen.add(key)
    return deduped


class LongSequenceEmbedder:
    """Extracts frozen sequence embeddings with optional sliding-window pooling."""

    def __init__(self, config: dict):
        self.model, self.tokenizer, self.device = load_backbone(config)
        self.window_len = config.get(
            "embedding_window_length", config.get("max_seq_length", 4096)
        )
        self.window_stride = config.get("embedding_window_stride", self.window_len)

    @torch.no_grad()
    def embed_sequence(
        self,
        sequence: str,
        window_len: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        sequence = sequence.upper()
        window_len = window_len or self.window_len
        stride = stride or self.window_stride
        windows = sliding_windows(len(sequence), window_len, stride)
        pooled = []
        for window in windows:
            seq_chunk = sequence[window.start:window.end]
            encoded = self.tokenizer(
                [seq_chunk],
                max_length=window_len,
                padding=True,
                truncation=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            pooled_chunk = self.model.forward_encode_only(input_ids, attention_mask)
            pooled.append(pooled_chunk.squeeze(0).detach().cpu().float().numpy())
        return np.stack(pooled).mean(axis=0)

    @torch.no_grad()
    def masked_base_log_probs(
        self,
        sequence: str,
        masked_positions: Sequence[int],
        window_len: Optional[int] = None,
    ) -> np.ndarray:
        """Predict log-probs at selected nucleotide positions.

        Positions are 0-based and relative to the input sequence.
        """
        window_len = window_len or self.window_len
        if not masked_positions:
            return np.zeros((0,), dtype=np.float32)

        sorted_positions = sorted(set(int(pos) for pos in masked_positions))
        groups: list[list[int]] = []
        current_group: list[int] = []
        for pos in sorted_positions:
            if not current_group or pos - current_group[0] < window_len:
                current_group.append(pos)
            else:
                groups.append(current_group)
                current_group = [pos]
        if current_group:
            groups.append(current_group)

        values = []
        for group in groups:
            start = max(0, min(group) - window_len // 2)
            end = min(len(sequence), max(group) + window_len // 2 + 1)
            if end - start > window_len:
                end = start + window_len
            if end - start < window_len:
                start = max(0, end - window_len)
                end = min(len(sequence), start + window_len)

            window_seq = sequence[start:end].upper()
            local_positions = [pos - start for pos in group if start <= pos < end]
            encoded = self.tokenizer(
                [window_seq],
                max_length=window_len,
                padding=True,
                truncation=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            masked_input_ids = input_ids.clone()
            mask_token_id = self.model.config.mask_token_id
            masked_input_ids[0, local_positions] = mask_token_id

            target_length = deterministic_target_length(self.model, input_ids.shape[1])
            z_L, source, mask_L = self.model.local_encoder(
                masked_input_ids, attention_mask, target_length
            )
            z_L_prime = self.model.latent_encoder(z_L, mask_L)
            z_hat_L = self.model.latent_decoder(z_L_prime, mask_L)
            logits, _ = self.model.local_decoder(z_hat_L, source, attention_mask)
            log_probs = F.log_softmax(logits, dim=-1)

            for local_pos in local_positions:
                token_id = int(input_ids[0, local_pos].item())
                values.append(float(log_probs[0, local_pos, token_id].item()))
        return np.asarray(values, dtype=np.float32)


DNA_CODON_TABLE = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGT", "TGC"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "*": ["TAA", "TAG", "TGA"],
}


def translate_cds(cds: str) -> str:
    aa = []
    for idx in range(0, len(cds), 3):
        codon = cds[idx:idx + 3]
        if len(codon) < 3:
            break
        found = "X"
        for residue, codons in DNA_CODON_TABLE.items():
            if codon in codons:
                found = residue
                break
        aa.append(found)
    return "".join(aa)


def choose_mutant_codon(wt_codon: str, target_aa: str) -> str:
    candidates = DNA_CODON_TABLE[target_aa]
    wt_codon = wt_codon.upper()
    return min(
        candidates,
        key=lambda codon: (
            sum(a != b for a, b in zip(wt_codon, codon)),
            codon,
        ),
    )


def apply_aa_mutations_to_cds(
    wt_cds: str,
    mutant: str,
    start_idx: int = 1,
) -> tuple[str, List[int]]:
    """Applies ProteinGym-style AA mutations to a CDS.

    Returns the mutated CDS and the mutated nucleotide positions.
    """
    wt_cds = wt_cds.upper()
    mutated = list(wt_cds)
    mutated_positions: List[int] = []
    for item in mutant.split(":"):
        item = item.strip()
        if not item:
            continue
        wt_aa = item[0]
        alt_aa = item[-1]
        aa_pos = int(item[1:-1]) - start_idx
        codon_start = aa_pos * 3
        wt_codon = wt_cds[codon_start:codon_start + 3]
        if len(wt_codon) != 3:
            raise ValueError(f"mutation outside CDS range: {item}")
        translated = translate_cds(wt_codon)
        if translated and translated[0] != wt_aa:
            raise ValueError(
                f"wild-type amino acid mismatch for {item}: CDS gives {translated[0]}"
            )
        alt_codon = choose_mutant_codon(wt_codon, alt_aa)
        for offset, base in enumerate(alt_codon):
            pos = codon_start + offset
            if mutated[pos] != base:
                mutated_positions.append(pos)
            mutated[pos] = base
    return "".join(mutated), sorted(set(mutated_positions))


def batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


class Timer:
    def __enter__(self):
        self.start = time.time()
        self.started_at = utc_now_iso()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()
        self.finished_at = utc_now_iso()
        self.duration_seconds = self.end - self.start
        return False
