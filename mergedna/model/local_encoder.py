"""Local Encoder for MergeDNA.

The Local Encoder serves as a learnable DNA tokenizer. It consists of
stacked LocalToMeAttention layers, each containing:
1. Local-window self-attention for capturing local patterns
2. Token merging for progressively reducing sequence length

The output is a tokenized sequence Z_L and a source matrix S tracking
which original positions were merged into each token.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .transformer import LocalWindowAttention
from .token_merging import LocalWindowTokenMerging


class LocalToMeAttentionLayer(nn.Module):
    """Single layer of Local Encoder: local-window attention followed by token merging.

    Args:
        embed_dim: Hidden dimension D.
        num_heads: Number of attention heads.
        window_size: Local window size w.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 16,
        window_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.local_attn = LocalWindowAttention(
            embed_dim, num_heads, window_size, dropout
        )
        self.token_merge = LocalWindowTokenMerging(embed_dim, window_size)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        r: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, N, D] input tokens.
            source: [B, N_orig, N] source matrix.
            r: Number of tokens to remove per window.
            attention_mask: [B, N].

        Returns:
            x_merged: [B, N', D] merged tokens.
            source_updated: [B, N_orig, N'] updated source matrix.
            attention_mask_new: [B, N'] updated attention mask.
        """
        # 1. Local-window self-attention
        x, _ = self.local_attn(x, attention_mask)

        # 2. Token merging
        x_merged, source_updated = self.token_merge(
            x, source, r, attention_mask
        )

        # Update attention mask
        new_len = x_merged.shape[1]
        if attention_mask is not None:
            # Derive new mask from source matrix: a merged token is valid
            # if any of its constituent original tokens were valid
            attention_mask_new = (source_updated.sum(dim=1) > 0).float()
        else:
            attention_mask_new = None

        return x_merged, source_updated, attention_mask_new


class LocalEncoder(nn.Module):
    """Local Encoder E_phi: Learnable DNA tokenizer.

    Stacks multiple LocalToMeAttention layers to progressively merge
    adjacent bases into words. Produces tokenized sequence Z_L ∈ R^{L×D}
    and source matrix S ∈ {0,1}^{L×N}.

    Paper config: 4 layers, D=1024, window_size=16, num_heads=16.

    Args:
        vocab_size: Size of DNA vocabulary (10: PAD,CLS,SEP,MASK,UNK,A,T,C,G,N).
        embed_dim: Hidden dimension D.
        num_layers: Number of LocalToMeAttention layers.
        num_heads: Number of attention heads.
        window_size: Local window size w.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 10,
        embed_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        window_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.window_size = window_size

        # Nucleotide embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Stack of LocalToMeAttention layers
        self.layers = nn.ModuleList([
            LocalToMeAttentionLayer(embed_dim, num_heads, window_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, N] token IDs.
            attention_mask: [B, N] attention mask.
            target_length: Target output length L. If None, uses N // 2.

        Returns:
            z_L: [B, L, D] tokenized sequence.
            source: [B, N, L] source matrix (transposed from paper for convenience:
                    source[b, i, j] = 1 means original position i belongs to merged token j).
            attention_mask: [B, L] updated attention mask.
        """
        B, N = input_ids.shape

        # Embed nucleotides
        x = self.embedding(input_ids)  # [B, N, D]

        # Initialize source matrix as identity: S^(0) = I_N
        # source[b, i, j] = 1 means original position i is in token j
        source = torch.eye(N, device=x.device, dtype=x.dtype)
        source = source.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        # Compute per-layer merge rate
        if target_length is None:
            target_length = N // 2

        # Calculate r per layer such that after num_layers merges,
        # we reduce from N to approximately target_length
        # Each layer reduces by r tokens per window
        # Total reduction = num_layers * r * num_windows
        # We distribute the reduction evenly across layers
        current_len = N
        for i, layer in enumerate(self.layers):
            # Calculate how many tokens to remove this layer
            remaining_layers = self.num_layers - i
            tokens_to_remove = current_len - target_length
            r_total = max(0, tokens_to_remove // remaining_layers)

            # r per window
            num_windows = (current_len + self.window_size - 1) // self.window_size
            r_per_window = max(1, r_total // max(num_windows, 1))
            r_per_window = min(r_per_window, self.window_size // 2)

            x, source, attention_mask = layer(
                x, source, r_per_window, attention_mask
            )
            current_len = x.shape[1]

        return x, source, attention_mask
