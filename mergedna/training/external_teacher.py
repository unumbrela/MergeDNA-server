"""External teacher model wrappers for knowledge distillation.

Wraps publicly available DNA foundation models (DNABERT-2, Nucleotide
Transformer v2, Evo2) into a unified interface for distillation into
EfficientMergeDNA.

Each wrapper produces:
- hidden_states: [B, N, D_teacher] per-position representations
- logits: [B, N, V_teacher] or None (MLM logits if available)

The distillation framework handles dimension/length mismatches via
projection layers and adaptive pooling.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ExternalTeacherWrapper(nn.Module):
    """Unified wrapper for external DNA foundation model teachers.

    Supported models:
    - dnabert2: zhihan1996/DNABERT-2-117M (BPE tokenization, 768-dim)
    - ntv2: InstaDeepAI/nucleotide-transformer-v2-{size}-multi-species (6-mer, 512/768/1024-dim)
    - evo2: arcinstitute/evo2_1b_base (character-level, decoder-only)

    All models are loaded from HuggingFace, frozen, and wrapped to produce
    per-position hidden states at the original input resolution N.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 4096,
    ):
        super().__init__()
        self.model_name = model_name
        self.device_str = device
        self.max_length = max_length

        self._load_model(model_name)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def _load_model(self, model_name: str):
        """Load model and tokenizer from HuggingFace."""
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

        logger.info(f"Loading external teacher: {model_name}")

        if "dnabert" in model_name.lower() or "DNABERT" in model_name:
            # DNABERT-2: BPE tokenization, encoder-only
            self.model_type = "dnabert2"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.hidden_dim = self.model.config.hidden_size  # 768

        elif "nucleotide-transformer" in model_name.lower():
            # Nucleotide Transformer: 6-mer tokenization, encoder-only
            self.model_type = "ntv2"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True
            )
            # NT models have a bert-like structure
            if hasattr(self.model.config, "hidden_size"):
                self.hidden_dim = self.model.config.hidden_size
            else:
                self.hidden_dim = self.model.config.d_model

        elif "evo" in model_name.lower():
            # Evo2: character-level, decoder-only
            self.model_type = "evo2"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.hidden_dim = self.model.config.hidden_size

        else:
            raise ValueError(
                f"Unknown teacher model: {model_name}. "
                "Supported: dnabert2, nucleotide-transformer, evo2"
            )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Teacher loaded: {model_name} ({n_params/1e6:.1f}M params, "
            f"hidden_dim={self.hidden_dim}, type={self.model_type})"
        )

    def _dna_ids_to_sequence(self, input_ids: torch.Tensor) -> list:
        """Convert MergeDNA character-level token IDs back to DNA strings.

        MergeDNA vocab: PAD=0, CLS=1, SEP=2, MASK=3, UNK=4, A=5, T=6, C=7, G=8, N=9
        """
        id_to_base = {5: "A", 6: "T", 7: "C", 8: "G", 9: "N"}
        sequences = []
        for seq_ids in input_ids.cpu().tolist():
            bases = []
            for tid in seq_ids:
                if tid in id_to_base:
                    bases.append(id_to_base[tid])
                # Skip special tokens (PAD, CLS, SEP, MASK, UNK)
            sequences.append("".join(bases))
        return sequences

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run teacher forward and return per-position representations.

        Args:
            input_ids: [B, N] MergeDNA-format token IDs (character-level).
            attention_mask: [B, N] attention mask.

        Returns:
            Dict with:
                hidden_states: [B, N, D_teacher] interpolated to input length.
                logits: [B, N, V_teacher] or None.
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Convert MergeDNA token IDs to DNA sequences
        sequences = self._dna_ids_to_sequence(input_ids)

        # Re-tokenize for the external model
        teacher_inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        # Forward through teacher
        if self.model_type == "dnabert2":
            outputs = self.model(**teacher_inputs)
            # outputs[0] = last_hidden_state: [B, L_bpe, 768]
            hidden = outputs[0]
            logits = None

        elif self.model_type == "ntv2":
            outputs = self.model(
                **teacher_inputs, output_hidden_states=True
            )
            # Use last hidden state from the base model
            hidden = outputs.hidden_states[-1]  # [B, L_kmer, D]
            logits = outputs.logits if hasattr(outputs, "logits") else None

        elif self.model_type == "evo2":
            outputs = self.model(**teacher_inputs)
            hidden = outputs[0]  # [B, L, D]
            logits = None

        # Interpolate hidden states to original input length N
        # hidden: [B, L_teacher, D] -> [B, N, D]
        L_teacher = hidden.shape[1]
        if L_teacher != N:
            # Use linear interpolation along sequence dimension
            # [B, L_teacher, D] -> [B, D, L_teacher] -> pool -> [B, D, N] -> [B, N, D]
            hidden = F.interpolate(
                hidden.transpose(1, 2),
                size=N,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        # Similarly interpolate logits if available
        if logits is not None:
            L_logits = logits.shape[1]
            if L_logits != N:
                logits = F.interpolate(
                    logits.transpose(1, 2),
                    size=N,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

        return {
            "hidden_states": hidden,
            "logits": logits,
        }

    def get_hidden_dim(self) -> int:
        """Return the teacher's hidden dimension."""
        return self.hidden_dim
