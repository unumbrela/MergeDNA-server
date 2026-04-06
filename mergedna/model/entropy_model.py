"""Lightweight local entropy estimator for Entropy-Guided Token Merging.

Estimates per-position information content using causal 1D convolutions.
High-entropy positions (information-rich) resist merging; low-entropy
positions (repetitive/predictable) are merged aggressively.

Inspired by BLT (Byte Latent Transformer, Meta 2024) entropy-based patching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalEntropyEstimator(nn.Module):
    """Lightweight entropy estimator (~0.5-1M parameters).

    Uses causal 1D convolutions to estimate how "surprising" each position is
    based on its local context.  The output is a score in [0, 1] where higher
    values indicate higher information content (harder to predict from
    neighbours).

    Architecture:
        embed -> Conv1d(causal) x 3 -> Linear -> sigmoid

    Args:
        embed_dim: Input embedding dimension D.
        hidden_dim: Convolution hidden dimension (default 128).
        kernel_size: Convolution kernel size (default 9, covers ~1 codon on
                     each side).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 128,
        kernel_size: int = 9,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        pad = kernel_size - 1  # causal padding

        self.proj_in = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.convs = nn.Sequential(
            nn.ConstantPad1d((pad, 0), 0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=1),
            nn.SiLU(),
            nn.ConstantPad1d((pad, 0), 0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=1),
            nn.SiLU(),
            nn.ConstantPad1d((pad, 0), 0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=1),
            nn.SiLU(),
        )
        self.proj_out = nn.Linear(hidden_dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate per-position entropy score.

        Args:
            x: [B, N, D] token embeddings (from nucleotide embedding layer).

        Returns:
            entropy_scores: [B, N] values in [0, 1].  High = information-rich.
        """
        h = self.proj_in(x)           # [B, N, hidden]
        h = h.transpose(1, 2)         # [B, hidden, N]  (conv1d expects channels-first)
        h = self.convs(h)             # [B, hidden, N]
        h = h.transpose(1, 2)         # [B, N, hidden]
        scores = self.proj_out(h).squeeze(-1)  # [B, N]
        return torch.sigmoid(scores)
