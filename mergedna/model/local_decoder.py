"""Local Decoder and Token Unmerging for MergeDNA.

The Local Decoder E_zeta maps the Latent Decoder's output Z_hat_L back to
the original base space. It performs:
1. Token unmerging U(Z_hat_L, S) to restore length N using the source matrix.
2. Local-window attention to refine local details.
3. Output the reconstructed sequence X_hat.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import LocalWindowAttention, RMSNorm


def token_unmerge(
    z_hat_L: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Unmerge tokens back to original length using the source matrix.

    Given Z_hat_L ∈ R^{B×L×D} and source S ∈ R^{B×N×L} (where S[b,i,j]=1
    means original position i belongs to merged token j), compute:
        Z_N = S^T @ Z_hat_L  (but using the source matrix for assignment)

    Actually, for each original position i, we find which merged token j
    it belongs to and copy that token's representation.

    In matrix form: Z_N[b, i, :] = sum_j S[b, i, j] * Z_hat_L[b, j, :] / sum_j S[b, i, j]

    Args:
        z_hat_L: [B, L, D] decoded merged tokens.
        source: [B, N, L] source matrix.

    Returns:
        z_N: [B, N, D] unmerged tokens at original resolution.
    """
    # source: [B, N, L], z_hat_L: [B, L, D]
    # z_N = source @ z_hat_L / source.sum(dim=-1, keepdim=True)
    source_sum = source.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, N, 1]
    source_norm = source / source_sum  # [B, N, L] normalized
    z_N = torch.bmm(source_norm, z_hat_L)  # [B, N, D]
    return z_N


class LocalDecoder(nn.Module):
    """Local Decoder E_zeta.

    Takes the decoded token-level representations, unmerges them to the
    original sequence length, and applies local-window attention to refine
    local details before outputting the reconstructed sequence.

    Paper config: 2 layers of local-window attention, D=1024, w=16.

    Args:
        embed_dim: Hidden dimension D.
        vocab_size: Output vocabulary size (for reconstruction head).
        num_layers: Number of local-window attention layers.
        num_heads: Number of attention heads.
        window_size: Local window size w.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        vocab_size: int = 10,
        num_layers: int = 2,
        num_heads: int = 16,
        window_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Local-window attention layers for refinement
        self.layers = nn.ModuleList([
            LocalWindowAttention(embed_dim, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

        # Output projection: predict nucleotide at each position
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        z_hat_L: torch.Tensor,
        source: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_hat_L: [B, L, D] decoded merged tokens from Latent Decoder.
            source: [B, N, L] source matrix from Local Encoder.
            attention_mask: [B, N] original attention mask.

        Returns:
            logits: [B, N, vocab_size] reconstruction logits.
            z_N: [B, N, D] unmerged representations.
        """
        # 1. Token unmerging: restore to original length N
        z_N = token_unmerge(z_hat_L, source)  # [B, N, D]

        # 2. Local-window attention refinement
        x = z_N
        for layer in self.layers:
            x, _ = layer(x, attention_mask)
        x = self.norm(x)

        # 3. Output head
        logits = self.output_head(x)  # [B, N, vocab_size]

        return logits, x
