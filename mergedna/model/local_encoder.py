"""Local Encoder for MergeDNA.

The Local Encoder serves as a learnable DNA tokenizer. It consists of
stacked LocalToMeAttention layers, each containing:
1. Local-window self-attention for capturing local patterns
2. Token merging for progressively reducing sequence length

The output is a tokenized sequence Z_L and a source matrix S tracking
which original positions were merged into each token.

MergeDNA-Long extensions:
- Entropy-guided token merging (high-information positions resist merging)
- Learned per-layer compression schedule
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
        entropy_weight: Weight for entropy penalty (0 = disabled).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 16,
        window_size: int = 16,
        dropout: float = 0.0,
        entropy_weight: float = 0.0,
    ):
        super().__init__()
        self.local_attn = LocalWindowAttention(
            embed_dim, num_heads, window_size, dropout
        )
        self.token_merge = LocalWindowTokenMerging(
            embed_dim, window_size, entropy_weight=entropy_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        r: int,
        attention_mask: Optional[torch.Tensor] = None,
        entropy_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, N, D] input tokens.
            source: [B, N_orig, N] source matrix.
            r: Number of tokens to remove per window.
            attention_mask: [B, N].
            entropy_scores: [B, N] per-position entropy (MergeDNA-Long).

        Returns:
            x_merged: [B, N', D] merged tokens.
            source_updated: [B, N_orig, N'] updated source matrix.
            attention_mask_new: [B, N'] updated attention mask.
        """
        # 1. Local-window self-attention
        x, _ = self.local_attn(x, attention_mask)

        # 2. Token merging (with optional entropy guidance)
        x_merged, source_updated = self.token_merge(
            x, source, r, attention_mask, entropy_scores=entropy_scores,
        )

        # Update attention mask
        if attention_mask is not None:
            attention_mask_new = (source_updated.sum(dim=1) > 0).float()
        else:
            attention_mask_new = None

        return x_merged, source_updated, attention_mask_new


class LearnedCompressionSchedule(nn.Module):
    """Learned per-layer compression rate (MergeDNA-Long).

    Each layer has a learnable logit that maps (via sigmoid) to a merge rate
    in [r_min, r_max].  During training Gumbel noise encourages exploration.

    Reference: DiffRate (Chen et al., ICCV 2023).
    """

    def __init__(self, num_layers: int, r_min: int = 1, r_max: int = 8):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        # Initialise near the midpoint
        init_val = 0.0  # sigmoid(0) = 0.5 → midpoint of [r_min, r_max]
        self.r_logits = nn.Parameter(torch.full((num_layers,), init_val))

    def forward(self, layer_idx: int) -> int:
        """Return the integer merge-rate for *layer_idx*."""
        p = torch.sigmoid(self.r_logits[layer_idx])
        r_float = self.r_min + (self.r_max - self.r_min) * p
        if self.training:
            # Gumbel noise for exploration (small: stddev ~0.3)
            noise = torch.randn_like(r_float) * 0.3
            r_float = r_float + noise
        return int(r_float.clamp(self.r_min, self.r_max).round().item())

    def get_all_rates(self) -> list:
        """Return current rates for all layers (useful for logging)."""
        with torch.no_grad():
            p = torch.sigmoid(self.r_logits)
            rates = self.r_min + (self.r_max - self.r_min) * p
            return rates.round().int().tolist()

    def compression_loss(self, target_r: float) -> torch.Tensor:
        """Differentiable loss encouraging the mean rate to match *target_r*.

        This is the only path through which gradients reach ``r_logits``,
        since the forward() discretisation (round + item) is non-differentiable.
        Following DiffRate (Chen et al., ICCV 2023), we penalise the gap
        between the continuous (sigmoid-based) mean merge rate and a target.

        Args:
            target_r: Desired average merge rate (e.g. window_size * compression_target).

        Returns:
            Scalar MSE loss on the continuous rates.
        """
        p = torch.sigmoid(self.r_logits)  # differentiable
        r_continuous = self.r_min + (self.r_max - self.r_min) * p
        mean_r = r_continuous.mean()
        return (mean_r - target_r) ** 2


class LocalEncoder(nn.Module):
    """Local Encoder E_phi: Learnable DNA tokenizer.

    Stacks multiple LocalToMeAttention layers to progressively merge
    adjacent bases into words. Produces tokenized sequence Z_L in R^{LxD}
    and source matrix S in {0,1}^{LxN}.

    Paper config: 4 layers, D=1024, window_size=16, num_heads=16.

    MergeDNA-Long extensions:
    - entropy_weight > 0 enables entropy-guided merging
    - use_learned_compression=True enables learned per-layer rates

    Args:
        vocab_size: Size of DNA vocabulary (10: PAD,CLS,SEP,MASK,UNK,A,T,C,G,N).
        embed_dim: Hidden dimension D.
        num_layers: Number of LocalToMeAttention layers.
        num_heads: Number of attention heads.
        window_size: Local window size w.
        dropout: Dropout rate.
        entropy_weight: Weight for entropy penalty (0 = original MergeDNA).
        use_learned_compression: Whether to use learned per-layer rates.
        r_min_per_window: Minimum merge rate per window (learned schedule).
        r_max_per_window: Maximum merge rate per window (learned schedule).
    """

    def __init__(
        self,
        vocab_size: int = 10,
        embed_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        window_size: int = 16,
        dropout: float = 0.0,
        entropy_weight: float = 0.0,
        use_learned_compression: bool = False,
        r_min_per_window: int = 1,
        r_max_per_window: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.use_learned_compression = use_learned_compression

        # Nucleotide embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Stack of LocalToMeAttention layers
        self.layers = nn.ModuleList([
            LocalToMeAttentionLayer(
                embed_dim, num_heads, window_size, dropout,
                entropy_weight=entropy_weight,
            )
            for _ in range(num_layers)
        ])

        # Learned compression schedule (MergeDNA-Long)
        self.compression_schedule = None
        if use_learned_compression:
            self.compression_schedule = LearnedCompressionSchedule(
                num_layers,
                r_min=r_min_per_window,
                r_max=min(r_max_per_window, window_size - 1),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        entropy_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, N] token IDs.
            attention_mask: [B, N] attention mask.
            target_length: Target output length L. If None, uses N // 2.
            entropy_scores: [B, N] per-position entropy (MergeDNA-Long).

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
        source = torch.eye(N, device=x.device, dtype=x.dtype)
        source = source.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        if target_length is None:
            target_length = N // 2

        current_len = N
        # Track current entropy scores (they shrink as tokens merge)
        current_entropy = entropy_scores

        for i, layer in enumerate(self.layers):
            if self.compression_schedule is not None:
                # Learned per-layer rate
                r_per_window = self.compression_schedule(i)
            else:
                # Heuristic: distribute reduction evenly across remaining layers
                remaining_layers = self.num_layers - i
                tokens_to_remove = current_len - target_length
                r_total = max(0, tokens_to_remove // remaining_layers)
                num_windows = (current_len + self.window_size - 1) // self.window_size
                r_per_window = max(1, r_total // max(num_windows, 1))
                r_per_window = min(r_per_window, self.window_size // 2)

            x, source, attention_mask = layer(
                x, source, r_per_window, attention_mask,
                entropy_scores=current_entropy,
            )

            # After merging, entropy scores must be re-derived for the new
            # (shorter) token set.  We approximate by max-pooling the original
            # entropy through the source matrix (a merged token inherits the
            # *highest* entropy of its constituents).
            if current_entropy is not None:
                # source: [B, N_orig, current_len'], current_entropy: [B, prev_len]
                # We need entropy for the new tokens.  Use source matrix:
                # new_entropy[j] = max over original positions i that map to j
                new_len = x.shape[1]
                # source[:, :, :new_len] maps original positions to new tokens
                # Weighted max: for each new token j, take max entropy of its sources
                # entropy_orig: [B, N_orig] (the *original* input entropy)
                if entropy_scores is not None:
                    ent = entropy_scores.unsqueeze(-1)  # [B, N_orig, 1]
                    # source: [B, N_orig, new_len]
                    # Multiply and take max over N_orig dimension
                    weighted = source * ent  # [B, N_orig, new_len]
                    current_entropy = weighted.max(dim=1).values  # [B, new_len]
                else:
                    current_entropy = None

            current_len = x.shape[1]

        return x, source, attention_mask
