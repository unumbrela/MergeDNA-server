"""Latent Encoder and Latent Decoder for MergeDNA.

- Latent Encoder E_psi: Full-attention Transformer that captures global context
  of the merged tokens from the Local Encoder. 20 layers, D=1024.
- HybridLatentEncoder (MergeDNA-Long): Mixes SSM layers (Gated DeltaNet /
  Mamba-2) with sparse full-attention layers for O(N) complexity.
- Latent Decoder E_omega: Lightweight Transformer that maps latent representations
  back toward the token space. 4 layers, symmetric to encoder.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .transformer import TransformerBlock, RMSNorm
from .token_merging import GlobalTokenMerging


class LatentEncoder(nn.Module):
    """Latent Encoder E_psi.

    Full-attention Transformer encoder that processes the tokenized sequence Z_L
    to capture long-range dependencies across the entire input.

    During pre-training, it also supports an additional round with ToMe-style
    attention to select K salient tokens for the latent MTR loss.

    Args:
        embed_dim: Hidden dimension D.
        num_layers: Number of Transformer blocks.
        num_heads: Number of attention heads.
        ffn_hidden_dim: FFN hidden dimension.
        dropout: Dropout rate.
        use_flash_attn: Whether to use Flash Attention.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_layers: int = 20,
        num_heads: int = 16,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, ffn_hidden_dim, dropout,
                use_flash_attn, gradient_checkpointing,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

        # Global token merging for selecting K salient tokens (pre-training only)
        self.global_tome = GlobalTokenMerging(embed_dim)

    def forward(
        self,
        z_L: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard forward pass (inference / downstream tasks).

        Args:
            z_L: [B, L, D] tokenized sequence from Local Encoder.
            attention_mask: [B, L].

        Returns:
            z_L_prime: [B, L, D] contextually enriched tokens.
        """
        x = z_L
        for layer in self.layers:
            x, _ = layer(x, attention_mask)
        x = self.norm(x)
        return x

    def forward_with_selection(
        self,
        z_L: torch.Tensor,
        K: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with token selection for latent MTR loss.

        Runs the encoder, then applies global ToMe to select K tokens.

        Args:
            z_L: [B, L, D] tokenized sequence.
            K: Number of tokens to keep.
            attention_mask: [B, L].

        Returns:
            z_L_prime: [B, L, D] full encoder output (for standard path).
            z_K_prime: [B, K, D] selected salient tokens.
            source_prime: [B, K, L] selection source matrix S'.
        """
        # Full encoder forward
        x = z_L
        for layer in self.layers:
            x, _ = layer(x, attention_mask)
        z_L_prime = self.norm(x)

        # Select K salient tokens via global ToMe
        z_K_prime, source_prime = self.global_tome(
            z_L_prime, K, attention_mask
        )

        return z_L_prime, z_K_prime, source_prime


class LatentDecoder(nn.Module):
    """Latent Decoder E_omega.

    Symmetric to the Latent Encoder but lighter (4 layers).
    Transforms Z'_L back toward the token space of Z_L, producing Z_hat_L.
    Together with the Latent Encoder, forms an autoencoder at the token level.

    Args:
        embed_dim: Hidden dimension D.
        num_layers: Number of Transformer blocks.
        num_heads: Number of attention heads.
        ffn_hidden_dim: FFN hidden dimension.
        dropout: Dropout rate.
        use_flash_attn: Whether to use Flash Attention.
        gradient_checkpointing: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, ffn_hidden_dim, dropout,
                use_flash_attn, gradient_checkpointing,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)

    def forward(
        self,
        z_L_prime: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_L_prime: [B, L, D] encoder output.
            attention_mask: [B, L].

        Returns:
            z_hat_L: [B, L, D] decoded token representations.
        """
        x = z_L_prime
        for layer in self.layers:
            x, _ = layer(x, attention_mask)
        x = self.norm(x)
        return x


class HybridLatentEncoder(nn.Module):
    """Hybrid SSM-Attention Latent Encoder (MergeDNA-Long).

    Replaces the full-attention Latent Encoder with a mix of SSM layers
    (Gated DeltaNet or Mamba-2) and sparse full-attention layers.

    Default layout (20 layers, attention at 5/11/17):
        SSM SSM SSM SSM SSM ATT SSM SSM SSM SSM SSM ATT SSM SSM SSM SSM SSM ATT SSM SSM

    This gives O(N) complexity for 17/20 layers while preserving global
    recall capability at 3 strategic positions.

    Args:
        embed_dim: Hidden dimension D.
        num_layers: Total number of layers.
        num_heads: Number of attention heads.
        ffn_hidden_dim: FFN hidden dimension.
        dropout: Dropout rate.
        use_flash_attn: Use Flash Attention for the attention layers.
        gradient_checkpointing: Use gradient checkpointing.
        ssm_type: SSM variant ('gated_deltanet' or 'mamba2').
        attention_layer_indices: Which layers use full attention.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_layers: int = 20,
        num_heads: int = 16,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        gradient_checkpointing: bool = False,
        ssm_type: str = "gated_deltanet",
        attention_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        if attention_layer_indices is None:
            attention_layer_indices = [5, 11, 17]
        attn_set = set(attention_layer_indices)

        from .hybrid_layers import build_ssm_block

        self.layers = nn.ModuleList()
        self.layer_types = []  # "ssm" or "attn" for logging
        for i in range(num_layers):
            if i in attn_set:
                self.layers.append(
                    TransformerBlock(
                        embed_dim, num_heads, ffn_hidden_dim, dropout,
                        use_flash_attn, gradient_checkpointing,
                    )
                )
                self.layer_types.append("attn")
            else:
                self.layers.append(
                    build_ssm_block(
                        ssm_type=ssm_type,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ffn_hidden_dim=ffn_hidden_dim,
                        dropout=dropout,
                        gradient_checkpointing=gradient_checkpointing,
                        layer_idx=i,
                    )
                )
                self.layer_types.append("ssm")

        self.norm = RMSNorm(embed_dim)
        self.global_tome = GlobalTokenMerging(embed_dim)

    def forward(
        self,
        z_L: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard forward pass."""
        x = z_L
        for layer in self.layers:
            x, _ = layer(x, attention_mask)
        x = self.norm(x)
        return x

    def forward_with_selection(
        self,
        z_L: torch.Tensor,
        K: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with global ToMe selection (same as LatentEncoder)."""
        x = z_L
        for layer in self.layers:
            x, _ = layer(x, attention_mask)
        z_L_prime = self.norm(x)
        z_K_prime, source_prime = self.global_tome(z_L_prime, K, attention_mask)
        return z_L_prime, z_K_prime, source_prime

