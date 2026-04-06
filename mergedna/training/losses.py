"""Pre-training loss functions for MergeDNA.

Implements three losses (Eq. 8 in the paper):
L_total = L_MTR(theta) + lambda * L_MTR(theta without phi) + L_AMTM(theta)

1. L_MTR: Merged Token Reconstruction - Cross-entropy between reconstructed
   sequence X_hat and original input X. Trains both Local Encoder and the
   full autoencoder pipeline.

2. Latent L_MTR (L_MTR without Local Encoder): Same reconstruction loss but with the Local
   Encoder frozen. The Latent Encoder uses global ToMe to select K tokens,
   forcing the latent model to compress and reconstruct from fewer tokens.

3. L_AMTM: Adaptive Masked Token Modeling - Masks tokens identified as
   important by the merging outcome S', then predicts the masked tokens.
   Uses an importance-weighted masking strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class MergeDNAPretrainLoss(nn.Module):
    """Combined pre-training loss for MergeDNA.

    Args:
        vocab_size: Vocabulary size for cross-entropy.
        lambda_latent: Weight for latent MTR loss (default 0.25).
        pad_token_id: Token ID for padding (excluded from loss).
    """

    def __init__(
        self,
        vocab_size: int = 10,
        lambda_latent: float = 0.25,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.lambda_latent = lambda_latent
        self.pad_token_id = pad_token_id
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_token_id, reduction="mean"
        )

    def compute_mtr_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Merged Token Reconstruction loss (Eq. 6).

        L_MTR = -(1/N) * sum_i log P(X_hat_i | X_i; θ)

        Args:
            logits: [B, N, vocab_size] reconstruction logits.
            targets: [B, N] original input token IDs.
            attention_mask: [B, N] mask for valid tokens.

        Returns:
            loss: Scalar cross-entropy loss.
        """
        B, N, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        if attention_mask is not None:
            # Set masked positions to pad_token_id so they're ignored
            mask_flat = attention_mask.reshape(-1).bool()
            targets_flat = targets_flat.clone()
            targets_flat[~mask_flat] = self.pad_token_id

        return self.ce_loss(logits_flat, targets_flat)

    def compute_amtm_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask_N: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Adaptive Masked Token Modeling loss (Eq. 7).

        L_AMTM = -(1/K) * sum_{i: M_N(i)=1} log P(X_hat_i | X * M_N; θ)

        Only computes loss on the K masked (important) positions.

        Args:
            logits: [B, N, vocab_size] prediction logits from masked input.
            targets: [B, N] original input token IDs.
            mask_N: [B, N] binary mask (1 = masked/to predict, 0 = visible).
            attention_mask: [B, N] valid token mask.

        Returns:
            loss: Scalar masked prediction loss.
        """
        B, N, V = logits.shape

        # Only compute loss on masked positions
        combined_mask = mask_N.bool()
        if attention_mask is not None:
            combined_mask = combined_mask & attention_mask.bool()

        if combined_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Gather predictions and targets at masked positions
        logits_masked = logits[combined_mask]  # [K_total, V]
        targets_masked = targets[combined_mask]  # [K_total]

        return F.cross_entropy(logits_masked, targets_masked)

    def compute_adaptive_mask(
        self,
        source_prime: torch.Tensor,
        source: torch.Tensor,
        K_mask: int,
    ) -> torch.Tensor:
        """Compute adaptive masking strategy from merging outcome S'.

        The key idea: tokens in large merged groups (highly merged) get low
        masking probability, while tokens in singleton or small groups get
        high masking probability. This focuses the model on predicting
        information-rich tokens.

        Args:
            source_prime: [B, K, L] from global ToMe selection.
            source: [B, N, L] from Local Encoder.
            K_mask: Number of local tokens to mask.

        Returns:
            mask_N: [B, N] binary mask in input space (1 = masked).
        """
        B, K_sel, L = source_prime.shape
        _, N, _ = source.shape

        # Compute group size g_i for each local token i
        # g_i = number of original tokens merged into local token i
        # g_i = sum_j S'[j, i] indicates how many selected tokens include token i
        g = source_prime.sum(dim=1)  # [B, L] group sizes in selection space

        # Importance weight: inversely proportional to group size
        # w_i = 1 / g_i (tokens in small groups are more important)
        w = 1.0 / g.clamp(min=1.0)  # [B, L]

        # Normalize to get probability distribution P_L over local tokens
        P_L = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, L]

        # Sample K_mask tokens to mask (without replacement)
        K_mask = min(K_mask, L)
        # Use Gumbel-top-k for differentiable-ish sampling
        noise = torch.rand_like(P_L)
        noise = noise.clamp(min=1e-8)
        scores = P_L.log() - (-noise.log()).log()  # Gumbel noise
        _, mask_indices = scores.topk(K_mask, dim=-1)  # [B, K_mask]

        # Create mask in local token space
        M_L = torch.zeros(B, L, device=source.device)
        M_L.scatter_(1, mask_indices, 1.0)  # [B, L]

        # Map mask from local token space to input space using source matrix
        # M_N = U(M_L, S): if a merged token is masked, all its constituent
        # original bases are masked
        # source: [B, N, L], M_L: [B, L] -> [B, L, 1]
        mask_N = torch.bmm(source, M_L.unsqueeze(-1)).squeeze(-1)  # [B, N]
        mask_N = (mask_N > 0).float()

        return mask_N

    def compute_random_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.15,
    ) -> torch.Tensor:
        """Compute a random mask in input space for vanilla MTM ablations."""
        noise = torch.rand_like(input_ids.float())
        if attention_mask is not None:
            noise = noise.masked_fill(~attention_mask.bool(), 2.0)
        k_mask = max(1, int(mask_ratio * input_ids.shape[1]))
        mask = torch.zeros_like(input_ids, dtype=torch.float32)
        _, indices = noise.topk(k_mask, dim=-1, largest=False)
        mask.scatter_(1, indices, 1.0)
        if attention_mask is not None:
            mask = mask * attention_mask.float()
        return mask

    def forward(
        self,
        # Standard reconstruction path
        logits_mtr: Optional[torch.Tensor] = None,
        # Latent reconstruction path (Local Encoder frozen)
        logits_latent_mtr: Optional[torch.Tensor] = None,
        # AMTM path
        logits_amtm: Optional[torch.Tensor] = None,
        mask_N: Optional[torch.Tensor] = None,
        # Targets
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total pre-training loss.

        L_total = L_MTR + λ * L_latent_MTR + L_AMTM

        Returns dict with individual and total losses.
        """
        # 1. Standard MTR loss
        assert input_ids is not None, "input_ids are required for pre-training losses"

        zero = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        l_mtr = (
            self.compute_mtr_loss(logits_mtr, input_ids, attention_mask)
            if logits_mtr is not None
            else zero
        )

        # 2. Latent MTR loss (Local Encoder frozen)
        l_latent_mtr = (
            self.compute_mtr_loss(logits_latent_mtr, input_ids, attention_mask)
            if logits_latent_mtr is not None
            else zero
        )

        # 3. AMTM loss
        l_amtm = (
            self.compute_amtm_loss(logits_amtm, input_ids, mask_N, attention_mask)
            if logits_amtm is not None and mask_N is not None
            else zero
        )

        # Total loss
        l_total = l_mtr + self.lambda_latent * l_latent_mtr + l_amtm

        return {
            "loss": l_total,
            "loss_mtr": l_mtr,
            "loss_latent_mtr": l_latent_mtr,
            "loss_amtm": l_amtm,
        }
