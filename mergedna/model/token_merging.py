"""Token Merging modules for MergeDNA.

Implements:
1. LocalWindowTokenMerging: Used in the Local Encoder. Operates within local
   windows to merge adjacent DNA tokens based on similarity (DTEM grouping).
2. GlobalTokenMerging: Used in the Latent Encoder for selecting K salient
   tokens during pre-training (Adaptive Context Modeling).
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalWindowTokenMerging(nn.Module):
    """Local-window token merging for the Local Encoder.

    At each layer, within each local window of size w, this module:
    1. Computes similarity scores between adjacent token pairs using a
       lightweight grouping embedding (DTEM).
    2. Selects the top-r most similar pairs to merge via bipartite matching.
    3. Performs soft merging: the "keeper" absorbs the "merged" token via
       weighted average.
    4. Updates the source matrix S to track which original positions merged.

    Args:
        embed_dim: Hidden dimension D.
        window_size: Local window size w (default 16).
        entropy_weight: Weight for entropy penalty in merge scores (0 = disabled).
    """

    def __init__(self, embed_dim: int, window_size: int = 16, entropy_weight: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.entropy_weight = entropy_weight
        # Lightweight grouping embedding (DTEM) for computing merge scores
        self.group_proj = nn.Linear(embed_dim, embed_dim // 4, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        r: int,
        attention_mask: Optional[torch.Tensor] = None,
        entropy_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Token embeddings [B, N, D].
            source: Source matrix [B, N_orig, N] tracking merging history.
                    S[i,j]=1 means original position i is in merged token j.
            r: Number of tokens to remove per window.
            attention_mask: [B, N], 1 for valid tokens.
            entropy_scores: [B, N] per-position entropy in [0,1].  High values
                            penalise merging (information-rich positions are
                            kept).  None disables entropy guidance.

        Returns:
            x_merged: Merged tokens [B, N-total_r, D].
            source_merged: Updated source matrix [B, N_orig, N-total_r].
        """
        B, N, D = x.shape
        w = self.window_size

        # Ensure N is divisible by window_size by padding
        pad_len = (w - N % w) % w
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            if entropy_scores is not None:
                entropy_scores = F.pad(entropy_scores, (0, pad_len), value=0)
            # Pad source matrix columns
            source = F.pad(source, (0, pad_len))

        N_padded = x.shape[1]
        num_windows = N_padded // w

        # Reshape into windows: [B*num_windows, w, D]
        x_win = x.reshape(B * num_windows, w, D)

        # Compute grouping metric using DTEM projection
        metric = self.group_proj(x_win)  # [B*nw, w, D//4]
        metric = F.normalize(metric, dim=-1)

        # Reshape entropy scores into windows if provided
        entropy_win = None
        if entropy_scores is not None and self.entropy_weight > 0:
            entropy_win = entropy_scores.reshape(B * num_windows, w)

        # Bipartite soft matching within each window
        # Split into even/odd sets
        r_per_window = min(r, w // 2)
        if r_per_window <= 0:
            return x[:, :N, :], source[:, :, :N]

        merge_fn, unmerge_info = self._bipartite_soft_matching(
            metric, r_per_window, attention_mask, B, num_windows, w,
            entropy_win=entropy_win,
        )

        # Apply merge
        x_merged_win, sizes = self._merge_wavg(merge_fn, x_win)

        # Reshape back: [B, num_windows * (w - r_per_window), D]
        new_w = w - r_per_window
        x_merged = x_merged_win.reshape(B, num_windows * new_w, D)

        # Update source matrix
        source_merged = self._update_source(
            source, merge_fn, B, num_windows, w, new_w
        )

        return x_merged, source_merged

    def _bipartite_soft_matching(
        self,
        metric: torch.Tensor,
        r: int,
        attention_mask: Optional[torch.Tensor],
        B: int,
        num_windows: int,
        w: int,
        entropy_win: Optional[torch.Tensor] = None,
    ):
        """Bipartite soft matching within local windows.

        When *entropy_win* is provided, high-entropy tokens are penalised so
        they are less likely to be selected for merging (they stay as keepers).

        Returns merge function and info needed for source matrix update.
        """
        Bw = metric.shape[0]  # B * num_windows

        with torch.no_grad():
            # Split into alternating sets
            a = metric[:, ::2, :]   # [Bw, w//2, d]
            b = metric[:, 1::2, :]  # [Bw, w//2, d]

            # Compute similarity scores
            scores = torch.bmm(a, b.transpose(-1, -2))  # [Bw, w//2, w//2]

            # Mask invalid positions if attention_mask provided
            if attention_mask is not None:
                mask_win = attention_mask.reshape(B * num_windows, w)
                mask_a = mask_win[:, ::2]   # [Bw, w//2]
                mask_b = mask_win[:, 1::2]  # [Bw, w//2]
                # Mask invalid pairs
                invalid = ~(mask_a.unsqueeze(-1).bool() & mask_b.unsqueeze(-2).bool())
                scores = scores.masked_fill(invalid, -math.inf)

            # --- Entropy-guided penalty ---
            # For each token in set A, subtract its entropy score from the
            # best-match similarity.  This means high-entropy (informative)
            # tokens get *lower* effective similarity → sorted later →
            # become keepers rather than merge sources.
            if entropy_win is not None:
                entropy_a = entropy_win[:, ::2]  # [Bw, w//2]
                entropy_b = entropy_win[:, 1::2]  # [Bw, w//2]
                # Penalise pairs where *either* token is high-entropy
                pair_entropy = entropy_a.unsqueeze(-1) + entropy_b.unsqueeze(-2)  # [Bw, w//2, w//2]
                scores = scores - self.entropy_weight * pair_entropy

            # Find best match for each token in set A
            node_max, node_idx = scores.max(dim=-1)  # [Bw, w//2]
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            # Split into merged and unmerged
            unm_idx = edge_idx[:, r:, :]   # Unmerged tokens from set A
            src_idx = edge_idx[:, :r, :]   # Source tokens to merge
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # Dest in set B

        def merge(x_in: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = x_in[:, ::2, :], x_in[:, 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src_gathered = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(
                -2, dst_idx.expand(n, r, c), src_gathered, reduce=mode
            )
            return torch.cat([unm, dst], dim=1)

        return merge, (unm_idx, src_idx, dst_idx, r)

    def _merge_wavg(self, merge_fn, x: torch.Tensor):
        """Weighted average merge tracking token sizes."""
        size = torch.ones(
            x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype
        )
        x = merge_fn(x * size, mode="sum")
        size = merge_fn(size, mode="sum")
        x = x / size.clamp(min=1)
        return x, size

    def _update_source(self, source, merge_fn, B, num_windows, w, new_w):
        """Update source matrix after merging."""
        N_orig = source.shape[1]
        N_old = source.shape[2]

        # Reshape source columns into windows
        # source: [B, N_orig, N_old] -> need to apply merge to the token dimension
        # Process each window's columns
        source_win = source.reshape(B, N_orig, num_windows, w)
        source_win = source_win.permute(0, 2, 1, 3)  # [B, nw, N_orig, w]
        source_win = source_win.reshape(B * num_windows, N_orig, w)

        # Apply merge along the last dimension (token dim)
        # Transpose so merge operates on token dim
        source_t = source_win.permute(0, 2, 1)  # [B*nw, w, N_orig]
        source_t_merged = merge_fn(source_t, mode="amax")  # [B*nw, new_w, N_orig]

        # Reshape back
        source_merged = source_t_merged.permute(0, 2, 1)  # [B*nw, N_orig, new_w]
        source_merged = source_merged.reshape(B, num_windows, N_orig, new_w)
        source_merged = source_merged.permute(0, 2, 1, 3)  # [B, N_orig, nw, new_w]
        source_merged = source_merged.reshape(B, N_orig, num_windows * new_w)

        return source_merged


class GlobalTokenMerging(nn.Module):
    """Global token merging for the Latent Encoder (used during pre-training).

    Selects K most salient tokens from L merged tokens using ToMe-style
    attention-based merging at the global scale. Used for computing the
    latent MTR loss and the AMTM masking strategy.

    Args:
        embed_dim: Hidden dimension D.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.group_proj = nn.Linear(embed_dim, embed_dim // 4, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        K: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select K salient tokens from the input.

        Args:
            x: Input tokens [B, L, D].
            K: Number of tokens to keep.
            attention_mask: [B, L], 1 for valid.

        Returns:
            x_selected: Selected tokens [B, K, D].
            source_prime: Source matrix [B, K, L] indicating which tokens
                          in the original L were selected/merged.
        """
        B, L, D = x.shape
        r = L - K  # Number of tokens to remove

        if r <= 0:
            source = torch.eye(L, device=x.device).unsqueeze(0).expand(B, -1, -1)
            return x, source

        metric = self.group_proj(x)
        metric = F.normalize(metric, dim=-1)

        with torch.no_grad():
            a = metric[:, ::2, :]
            b = metric[:, 1::2, :]
            scores = torch.bmm(a, b.transpose(-1, -2))

            if attention_mask is not None:
                mask_a = attention_mask[:, ::2].bool()
                mask_b = attention_mask[:, 1::2].bool()
                invalid = ~(mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2))
                scores = scores.masked_fill(invalid, -math.inf)

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            r_actual = min(r, a.shape[1])
            unm_idx = edge_idx[:, r_actual:, :]
            src_idx = edge_idx[:, :r_actual, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        # Build merge function
        def merge(x_in, mode="mean"):
            src, dst = x_in[:, ::2, :], x_in[:, 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r_actual, c))
            src_g = src.gather(dim=-2, index=src_idx.expand(n, r_actual, c))
            dst = dst.scatter_reduce(
                -2, dst_idx.expand(n, r_actual, c), src_g, reduce=mode
            )
            return torch.cat([unm, dst], dim=1)

        # Weighted average merge
        size = torch.ones(B, L, 1, device=x.device, dtype=x.dtype)
        x_merged = merge(x * size, mode="sum")
        size_merged = merge(size, mode="sum")
        x_merged = x_merged / size_merged.clamp(min=1)

        # Build source matrix S' ∈ {0,1}^{K×L}
        source_eye = torch.eye(L, device=x.device).unsqueeze(0).expand(B, -1, -1)
        source_t = source_eye.permute(0, 2, 1)  # [B, L, L]
        source_merged_t = merge(source_t, mode="amax")  # [B, K, L]
        source_prime = source_merged_t

        return x_merged, source_prime
