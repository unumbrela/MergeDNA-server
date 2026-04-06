"""Hybrid SSM-Attention layers for MergeDNA-Long.

Provides wrappers around Gated DeltaNet (flash-linear-attention) and
Mamba-2 that share the same interface as the existing TransformerBlock,
enabling drop-in replacement within the Latent Encoder.

Reference:
- Gated DeltaNet: Yang et al., ICLR 2025, arxiv.org/abs/2412.06464
- Mamba-2: Dao & Gu, 2024, arxiv.org/abs/2405.21060
- HybriDNA: Ma et al., 2025, arxiv.org/abs/2502.10807
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .transformer import RMSNorm, SwiGLU


class GatedDeltaNetBlock(nn.Module):
    """Gated DeltaNet block with the same interface as TransformerBlock.

    Pre-norm architecture:
        RMSNorm -> GatedDeltaNet -> Residual -> RMSNorm -> SwiGLU FFN -> Residual

    Args:
        embed_dim: Model dimension D.
        num_heads: Number of attention/SSM heads.
        ffn_hidden_dim: FFN hidden dimension (auto-computed if None).
        dropout: Dropout rate.
        gradient_checkpointing: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        from fla.layers import GatedDeltaNet

        head_dim = max(embed_dim // num_heads, 64)
        actual_heads = embed_dim // head_dim

        self.norm1 = RMSNorm(embed_dim)
        self.ssm = GatedDeltaNet(
            hidden_size=embed_dim,
            num_heads=actual_heads,
            head_dim=head_dim,
            use_short_conv=True,
            conv_size=4,
            layer_idx=layer_idx,
        )
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ffn_hidden_dim, dropout)

    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        # Pre-norm SSM
        residual = x
        h = self.norm1(x)
        # GatedDeltaNet returns (output, kv_state, hidden_state)
        h, _, _ = self.ssm(h)
        x = residual + h

        # Pre-norm FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x, None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_key_metric: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        if self.gradient_checkpointing and self.training:
            x = torch_checkpoint(
                self._ckpt_forward, x, attention_mask,
                use_reentrant=False,
            )
            return x, None
        return self._forward_impl(x, attention_mask)

    def _ckpt_forward(self, x, attention_mask):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            out, _ = self._forward_impl(x, attention_mask)
        return out


class Mamba2Block(nn.Module):
    """Mamba-2 block wrapper (fallback when GatedDeltaNet is unavailable).

    Same interface as TransformerBlock / GatedDeltaNetBlock.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        from mamba_ssm.modules.mamba2 import Mamba2

        self.norm1 = RMSNorm(embed_dim)
        self.ssm = Mamba2(
            d_model=embed_dim,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64,
            layer_idx=layer_idx,
        )
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ffn_hidden_dim, dropout)

    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        residual = x
        h = self.norm1(x)
        h = self.ssm(h)
        x = residual + h

        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x, None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_key_metric: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        if self.gradient_checkpointing and self.training:
            x = torch_checkpoint(
                self._ckpt_forward, x, attention_mask,
                use_reentrant=False,
            )
            return x, None
        return self._forward_impl(x, attention_mask)

    def _ckpt_forward(self, x, attention_mask):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            out, _ = self._forward_impl(x, attention_mask)
        return out


def build_ssm_block(
    ssm_type: str,
    embed_dim: int,
    num_heads: int = 16,
    ffn_hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    gradient_checkpointing: bool = False,
    layer_idx: Optional[int] = None,
) -> nn.Module:
    """Factory function to create an SSM block by type name."""
    if ssm_type == "gated_deltanet":
        return GatedDeltaNetBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
            layer_idx=layer_idx,
        )
    elif ssm_type == "mamba2":
        return Mamba2Block(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
            layer_idx=layer_idx,
        )
    else:
        raise ValueError(f"Unknown SSM type: {ssm_type}. Choose 'gated_deltanet' or 'mamba2'.")
