"""Knowledge distillation loss functions for EfficientMergeDNA.

Three complementary distillation signals from teacher to student:
1. MergePatternDistillLoss  — aligns tokenization decisions (source matrices)
2. LatentRepresentationDistillLoss — aligns latent space representations
3. OutputDistillLoss — standard soft-label KD on reconstruction logits

Reference: FastAST (Interspeech 2024) combined ToMe + Cross-Model KD for
audio. This is the first application of merge-aware KD to genomic models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MergePatternDistillLoss(nn.Module):
    """Distill teacher's merge pattern (source matrix S) into the student.

    Challenge: teacher and student have different compression targets,
    so source matrices have different shapes ([B, N, L_t] vs [B, N, L_s]).

    Solution: normalise each row of S to a probability distribution over
    merged tokens, pool both to a shared dimension, then compute KL
    divergence.  This measures whether teacher and student "agree" on
    which original bases should belong together.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_source: torch.Tensor,
        teacher_source: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_source: [B, N, L_s] student merge source matrix.
            teacher_source: [B, N, L_t] teacher merge source matrix.
            attention_mask: [B, N] valid positions in the input.

        Returns:
            Scalar KL divergence loss.
        """
        B, N, L_s = student_source.shape
        _, _, L_t = teacher_source.shape

        # Normalise rows to probability distributions
        # Each row tells us "how is original position i distributed across merged tokens?"
        s_prob = student_source / student_source.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        t_prob = teacher_source / teacher_source.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Pool both to the same length (min of L_s, L_t) for comparison
        shared_len = min(L_s, L_t)
        # Use adaptive average pooling along the last dimension
        # Reshape for pooling: [B*N, 1, L] -> pool -> [B*N, 1, shared_len]
        s_pooled = F.adaptive_avg_pool1d(
            s_prob.reshape(B * N, 1, L_s), shared_len
        ).reshape(B, N, shared_len)
        t_pooled = F.adaptive_avg_pool1d(
            t_prob.reshape(B * N, 1, L_t), shared_len
        ).reshape(B, N, shared_len)

        # Re-normalise after pooling
        s_pooled = s_pooled / s_pooled.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        t_pooled = t_pooled / t_pooled.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Apply temperature
        s_log = (s_pooled / self.temperature).log_softmax(dim=-1)
        t_soft = (t_pooled / self.temperature).softmax(dim=-1)

        # KL divergence per position
        kl = F.kl_div(s_log, t_soft, reduction="none").sum(dim=-1)  # [B, N]

        # Mask invalid positions
        if attention_mask is not None:
            kl = kl * attention_mask
            return kl.sum() / attention_mask.sum().clamp(min=1)
        return kl.mean()


class LatentRepresentationDistillLoss(nn.Module):
    """Distill teacher's latent representations into the student.

    Handles mismatched dimensions (embed_dim) and sequence lengths (L)
    via a learnable linear projection and adaptive pooling.

    Uses smooth-L1 (Huber) loss for robustness to outlier tokens.
    """

    def __init__(self, student_dim: int = 512, teacher_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(
        self,
        student_z: torch.Tensor,
        teacher_z: torch.Tensor,
        student_mask: Optional[torch.Tensor] = None,
        teacher_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_z: [B, L_s, D_s] student latent representations.
            teacher_z: [B, L_t, D_t] teacher latent representations.
            student_mask: [B, L_s] student attention mask.
            teacher_mask: [B, L_t] teacher attention mask.

        Returns:
            Scalar representation matching loss.
        """
        B, L_s, D_s = student_z.shape
        _, L_t, D_t = teacher_z.shape

        # Project student to teacher dimension
        s_proj = self.proj(student_z)  # [B, L_s, D_t]

        # Pool both to the same sequence length
        shared_len = min(L_s, L_t)

        # [B, D_t, L_s] -> pool -> [B, D_t, shared_len] -> [B, shared_len, D_t]
        s_pooled = F.adaptive_avg_pool1d(
            s_proj.transpose(1, 2), shared_len
        ).transpose(1, 2)
        t_pooled = F.adaptive_avg_pool1d(
            teacher_z.transpose(1, 2), shared_len
        ).transpose(1, 2)

        # L2-normalise along feature dim for scale-invariant comparison
        s_norm = F.normalize(s_pooled, dim=-1)
        t_norm = F.normalize(t_pooled, dim=-1)

        # Smooth L1 (Huber) loss
        loss = F.smooth_l1_loss(s_norm, t_norm, reduction="mean")
        return loss


class OutputDistillLoss(nn.Module):
    """Standard soft-label knowledge distillation on reconstruction logits.

    KL(softmax(teacher/T) || softmax(student/T)) * T^2

    Both logits are at the original input resolution [B, N, vocab_size].
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, N, V] student reconstruction logits.
            teacher_logits: [B, N, V] teacher reconstruction logits.
            attention_mask: [B, N] valid positions.

        Returns:
            Scalar KD loss.
        """
        T = self.temperature

        s_log = F.log_softmax(student_logits / T, dim=-1)
        t_soft = F.softmax(teacher_logits / T, dim=-1)

        # KL per position: [B, N]
        kl = F.kl_div(s_log, t_soft, reduction="none").sum(dim=-1)

        if attention_mask is not None:
            kl = kl * attention_mask
            loss = kl.sum() / attention_mask.sum().clamp(min=1)
        else:
            loss = kl.mean()

        return loss * (T ** 2)
