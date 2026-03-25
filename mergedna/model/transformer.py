"""LLaMA-style Transformer building blocks for MergeDNA.

Implements:
- RMSNorm
- Rotary Positional Embedding (RoPE)
- SwiGLU Feed-Forward Network
- Multi-Head Attention (with optional Flash Attention)
- TransformerBlock (full attention)
- LocalWindowAttention (windowed attention for Local Encoder/Decoder)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange

# Try importing flash attention
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, dim: int, max_position: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (LLaMA style).

    FFN(x) = SiLU(W1 @ x) * (W3 @ x) then W2 @ ...
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to nearest multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional Flash Attention support.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
        use_flash_attn: Whether to use Flash Attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_key_metric: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, N, D]
            attention_mask: [B, N], 1 for valid tokens.
            return_key_metric: If True, return mean of keys as merge metric.

        Returns:
            output: [B, N, D]
            key_metric: [B, N, D//num_heads] if return_key_metric else None
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self.rotary_emb(N, x.device)
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, N, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)

        q_r = q.permute(0, 2, 1, 3)  # [B, H, N, hd]
        k_r = k.permute(0, 2, 1, 3)

        q_r, k_r = apply_rotary_pos_emb(
            q_r,
            k_r,
            cos.permute(0, 2, 1, 3),
            sin.permute(0, 2, 1, 3),
        )

        key_metric = None
        if return_key_metric:
            key_metric = k_r.mean(dim=1)  # [B, N, head_dim]

        if self.use_flash_attn:
            # Flash attention expects [B, N, H, hd] in fp16/bf16
            orig_dtype = x.dtype
            fa_dtype = torch.bfloat16
            q_fa = q_r.permute(0, 2, 1, 3).to(fa_dtype).contiguous()
            k_fa = k_r.permute(0, 2, 1, 3).to(fa_dtype).contiguous()
            v_fa = v.to(fa_dtype).contiguous()

            out = flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
            )
            out = out.reshape(B, N, -1).to(orig_dtype)
        else:
            # Standard attention
            q_r = q_r  # [B, H, N, hd]
            k_r = k_r
            v_r = v.permute(0, 2, 1, 3)  # [B, H, N, hd]

            attn = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
                attn = attn.masked_fill(~attn_mask.bool(), float("-inf"))

            # Prevent NaN from all-padding windows: if all positions are masked,
            # replace -inf with 0 before softmax so it produces uniform attention
            all_masked = (attention_mask.sum(dim=-1) == 0) if attention_mask is not None else None
            if all_masked is not None and all_masked.any():
                # [B] -> [B, 1, 1, 1], broadcast to fill
                attn = attn.masked_fill(all_masked[:, None, None, None], 0.0)

            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)

            out = torch.matmul(attn, v_r)  # [B, H, N, hd]
            out = out.permute(0, 2, 1, 3).reshape(B, N, -1)

        out = self.o_proj(out)
        return out, key_metric


class TransformerBlock(nn.Module):
    """Standard Transformer block with full attention (LLaMA style).

    Pre-norm architecture: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads, dropout, use_flash_attn
        )
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden_dim, dropout)
        self.gradient_checkpointing = gradient_checkpointing

    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_key_metric: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, key_metric = self.attn(
            x_norm, attention_mask, return_key_metric
        )
        x = residual + attn_out

        # Pre-norm FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x, key_metric

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_key_metric: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.gradient_checkpointing and self.training and not return_key_metric:
            # Checkpoint cannot return tuples with None easily, wrap it
            x = torch_checkpoint(
                self._ckpt_forward, x, attention_mask,
                use_reentrant=False,
            )
            return x, None
        return self._forward_impl(x, attention_mask, return_key_metric)

    def _ckpt_forward(self, x, attention_mask):
        # Preserve autocast context inside gradient checkpoint
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            out, _ = self._forward_impl(x, attention_mask, False)
        return out


class LocalWindowAttention(nn.Module):
    """Local-window self-attention for Local Encoder/Decoder.

    Applies attention within non-overlapping local windows of size w.
    This provides linear complexity O(N * w) instead of O(N^2).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        window_size: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.attn_block = TransformerBlock(
            dim, num_heads, dropout=dropout, use_flash_attn=False
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_key_metric: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, N, D]
            attention_mask: [B, N]
        Returns:
            out: [B, N, D]
            key_metric: Optional merge metric
        """
        B, N, D = x.shape
        w = self.window_size

        # Pad to multiple of window size
        pad_len = (w - N % w) % w
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

        N_padded = x.shape[1]
        num_windows = N_padded // w

        # Reshape into windows
        x_win = x.reshape(B * num_windows, w, D)
        mask_win = None
        if attention_mask is not None:
            mask_win = attention_mask.reshape(B * num_windows, w)

        # Apply attention within each window
        out_win, key_metric_win = self.attn_block(
            x_win, mask_win, return_key_metric
        )

        # Reshape back
        out = out_win.reshape(B, N_padded, D)[:, :N, :]

        key_metric = None
        if key_metric_win is not None:
            key_metric = key_metric_win.reshape(B, N_padded, -1)[:, :N, :]

        return out, key_metric
