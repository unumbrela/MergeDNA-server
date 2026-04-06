"""MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization.

Full model assembling Local Encoder, Latent Encoder, Latent Decoder,
and Local Decoder into a hierarchical autoencoder-style architecture.

Supports three modes:
1. Pre-training: Three forward passes for L_MTR, latent L_MTR, and L_AMTM.
2. Encoder-only (classification): Use Latent Encoder output for downstream tasks.
3. Encoder-decoder (generation): Full autoencoder for sequence reconstruction.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .local_encoder import LocalEncoder
from .latent_encoder import LatentEncoder, LatentDecoder, HybridLatentEncoder
from .local_decoder import LocalDecoder, token_unmerge
from ..training.losses import MergeDNAPretrainLoss


@dataclass
class MergeDNAConfig:
    """Configuration for MergeDNA model."""

    # Vocabulary
    vocab_size: int = 10  # PAD, CLS, SEP, MASK, UNK, A, T, C, G, N
    pad_token_id: int = 0
    mask_token_id: int = 3

    # Model dimensions
    embed_dim: int = 1024
    num_heads: int = 16
    ffn_hidden_dim: Optional[int] = None  # Auto: 8/3 * embed_dim

    # Local Encoder
    local_encoder_layers: int = 4
    window_size: int = 16

    # Latent Encoder
    latent_encoder_layers: int = 20

    # Latent Decoder
    latent_decoder_layers: int = 4

    # Local Decoder
    local_decoder_layers: int = 2

    # Training
    dropout: float = 0.0
    use_flash_attn: bool = True
    max_seq_length: int = 4096

    # Pre-training
    compression_target: float = 0.5  # Target L/N ratio
    compression_variance: float = 0.1  # Variance for random L sampling
    lambda_latent: float = 0.25  # Weight for latent MTR loss
    K_ratio: float = 0.5  # K/L ratio for token selection
    gradient_checkpointing: bool = False
    use_mtr: bool = True
    use_latent_mtr: bool = True
    use_amtm: bool = True
    amtm_masking_strategy: str = "adaptive"
    random_mask_ratio: float = 0.15

    # === MergeDNA-Long extensions ===

    # Entropy-guided merging
    use_entropy_guided_merging: bool = False
    entropy_weight: float = 0.5
    entropy_model_hidden_dim: int = 128
    entropy_model_kernel_size: int = 9

    # Learned compression schedule
    use_learned_compression: bool = False
    r_min_per_window: int = 1
    r_max_per_window: int = 8

    # Entropy auxiliary loss weight (trains entropy model via reconstruction signal)
    entropy_aux_loss_weight: float = 0.1

    # Compression loss weight (learned schedule regularization)
    compression_loss_weight: float = 0.1

    # Hybrid Latent Encoder
    latent_encoder_type: str = "transformer"  # "transformer" | "hybrid"
    ssm_type: str = "gated_deltanet"          # "gated_deltanet" | "mamba2"
    attention_layer_indices: list = field(
        default_factory=lambda: [5, 11, 17]
    )


@dataclass
class EfficientMergeDNAConfig(MergeDNAConfig):
    """Configuration for EfficientMergeDNA (~50M params).

    Lightweight variant designed for knowledge distillation from a
    full-size MergeDNA teacher. Key differences:
    - Halved embed_dim (512 vs 1024)
    - Fewer layers (3/8/2/1 vs 4/20/4/2)
    - Aggressive compression (80% vs 50%)
    - Entropy-guided merging + learned compression on by default
    - Hybrid SSM-Attention latent encoder

    Target: ~45-50M parameters, competitive with 380M teacher.
    """

    # Model dimensions (halved)
    embed_dim: int = 512
    num_heads: int = 8

    # Fewer layers
    local_encoder_layers: int = 3
    latent_encoder_layers: int = 8
    latent_decoder_layers: int = 2
    local_decoder_layers: int = 1

    # Aggressive compression (80% token reduction)
    compression_target: float = 0.2
    compression_variance: float = 0.05

    # Entropy-guided merging on by default
    use_entropy_guided_merging: bool = True
    entropy_model_hidden_dim: int = 64  # smaller than teacher's 128

    # Learned compression on by default
    use_learned_compression: bool = True
    r_min_per_window: int = 2
    r_max_per_window: int = 14  # up to 14/16 removed per window

    # Hybrid SSM-Attention latent encoder by default
    latent_encoder_type: str = "hybrid"
    attention_layer_indices: list = field(
        default_factory=lambda: [3, 7]  # 2 attention layers among 8
    )


class MergeDNA(nn.Module):
    """MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization.

    Architecture:
        Input X (N bases) -> Local Encoder -> Z_L (L tokens) + Source S
        Z_L -> Latent Encoder -> Z'_L (L tokens, contextually enriched)
        Z'_L -> Latent Decoder -> Z_hat_L (L tokens)
        Z_hat_L -> Unmerge(S) -> Z_N -> Local Decoder -> X_hat (N bases)
    """

    def __init__(self, config: MergeDNAConfig):
        super().__init__()
        self.config = config

        # Entropy model (MergeDNA-Long)
        self.entropy_model = None
        if config.use_entropy_guided_merging:
            from .entropy_model import LocalEntropyEstimator
            self.entropy_model = LocalEntropyEstimator(
                embed_dim=config.embed_dim,
                hidden_dim=config.entropy_model_hidden_dim,
                kernel_size=config.entropy_model_kernel_size,
            )

        # Local Encoder (learnable tokenizer)
        self.local_encoder = LocalEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_layers=config.local_encoder_layers,
            num_heads=config.num_heads,
            window_size=config.window_size,
            dropout=config.dropout,
            entropy_weight=config.entropy_weight if config.use_entropy_guided_merging else 0.0,
            use_learned_compression=config.use_learned_compression,
            r_min_per_window=config.r_min_per_window,
            r_max_per_window=config.r_max_per_window,
        )

        # Latent Encoder: full attention (original) or hybrid SSM-Attention (Long)
        if config.latent_encoder_type == "hybrid":
            self.latent_encoder = HybridLatentEncoder(
                embed_dim=config.embed_dim,
                num_layers=config.latent_encoder_layers,
                num_heads=config.num_heads,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
                use_flash_attn=config.use_flash_attn,
                gradient_checkpointing=config.gradient_checkpointing,
                ssm_type=config.ssm_type,
                attention_layer_indices=config.attention_layer_indices,
            )
        else:
            self.latent_encoder = LatentEncoder(
                embed_dim=config.embed_dim,
                num_layers=config.latent_encoder_layers,
                num_heads=config.num_heads,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
                use_flash_attn=config.use_flash_attn,
                gradient_checkpointing=config.gradient_checkpointing,
            )

        # Latent Decoder
        self.latent_decoder = LatentDecoder(
            embed_dim=config.embed_dim,
            num_layers=config.latent_decoder_layers,
            num_heads=config.num_heads,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            use_flash_attn=config.use_flash_attn,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        # Local Decoder (detokenizer)
        self.local_decoder = LocalDecoder(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            num_layers=config.local_decoder_layers,
            num_heads=config.num_heads,
            window_size=config.window_size,
            dropout=config.dropout,
        )

        # Pre-training loss
        self.pretrain_loss = MergeDNAPretrainLoss(
            vocab_size=config.vocab_size,
            lambda_latent=config.lambda_latent,
            pad_token_id=config.pad_token_id,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following LLaMA conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _sample_target_length(self, N: int) -> int:
        """Sample target compression length L during pre-training.

        L is sampled from a Gaussian centered at N * compression_target,
        clamped to [compression_target - 2*variance, compression_target + 2*variance].
        """
        ct = self.config.compression_target
        cv = self.config.compression_variance
        mean = N * ct
        std = N * cv
        L = int(torch.normal(torch.tensor(mean), torch.tensor(std)).item())
        lo = max(1, int((ct - 2 * cv) * N))
        hi = int((ct + 2 * cv) * N)
        L = max(lo, min(L, hi))
        # Round to nearest multiple of window_size for clean windowing
        w = self.config.window_size
        L = max(w, (L // w) * w)
        return L

    def _compute_entropy(
        self,
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute per-position entropy scores (MergeDNA-Long)."""
        if self.entropy_model is None:
            return None
        # Use the embedding layer to get token representations for entropy
        x = self.local_encoder.embedding(input_ids)  # [B, N, D]
        return self.entropy_model(x)  # [B, N]

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run Local Encoder + Latent Encoder.

        Returns:
            z_L_prime: [B, L, D] contextually enriched tokens.
            z_L: [B, L, D] tokenized sequence (before latent encoder).
            source: [B, N, L] source matrix.
            mask_L: [B, L] attention mask for merged tokens.
        """
        # Compute entropy scores (MergeDNA-Long)
        entropy_scores = self._compute_entropy(input_ids)

        # Local Encoder: tokenize (with optional entropy guidance)
        z_L, source, mask_L = self.local_encoder(
            input_ids, attention_mask, target_length,
            entropy_scores=entropy_scores,
        )

        # Latent Encoder: global context
        z_L_prime = self.latent_encoder(z_L, mask_L)

        return z_L_prime, z_L, source, mask_L

    def decode(
        self,
        z_L_prime: torch.Tensor,
        source: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_L: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Latent Decoder + Local Decoder.

        Returns:
            logits: [B, N, vocab_size] reconstruction logits.
            z_N: [B, N, D] unmerged representations.
        """
        # Latent Decoder
        z_hat_L = self.latent_decoder(z_L_prime, mask_L)

        # Local Decoder: unmerge + refine
        logits, z_N = self.local_decoder(z_hat_L, source, attention_mask)

        return logits, z_N

    def forward_pretrain(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pre-training forward pass with three objectives.

        Performs three forward passes:
        1. Standard reconstruction (L_MTR)
        2. Latent reconstruction with Local Encoder frozen (latent L_MTR)
        3. Adaptive masked token modeling (L_AMTM)

        Args:
            input_ids: [B, N] input token IDs.
            attention_mask: [B, N] attention mask.

        Returns:
            Dict with 'loss', 'loss_mtr', 'loss_latent_mtr', 'loss_amtm'.
        """
        _, N = input_ids.shape
        target_length = self._sample_target_length(N)
        K = max(self.config.window_size, target_length // 2)

        logits_mtr = None
        logits_latent_mtr = None
        logits_amtm = None
        mask_N = None
        z_L = source = mask_L = None
        source_detached = None
        source_prime = None

        # Compute entropy scores once (shared across all passes)
        entropy_scores = self._compute_entropy(input_ids)

        need_standard_pass = (
            self.config.use_mtr
            or self.config.use_latent_mtr
            or (
                self.config.use_amtm
                and self.config.amtm_masking_strategy == "adaptive"
            )
        )

        if need_standard_pass:
            z_L, source, mask_L = self.local_encoder(
                input_ids, attention_mask, target_length,
                entropy_scores=entropy_scores,
            )
            z_L_prime = self.latent_encoder(z_L, mask_L)

            if self.config.use_mtr:
                z_hat_L = self.latent_decoder(z_L_prime, mask_L)
                logits_mtr, _ = self.local_decoder(z_hat_L, source, attention_mask)

        if self.config.use_latent_mtr or (
            self.config.use_amtm and self.config.amtm_masking_strategy == "adaptive"
        ):
            with torch.no_grad():
                z_L_detached = z_L.detach()
                source_detached = source.detach()

            _, z_K_prime, source_prime = self.latent_encoder.forward_with_selection(
                z_L_detached, K, mask_L
            )
            z_L_from_K = token_unmerge(z_K_prime, source_prime.permute(0, 2, 1))

            if self.config.use_latent_mtr:
                z_hat_L_latent = self.latent_decoder(z_L_from_K, mask_L)
                logits_latent_mtr, _ = self.local_decoder(
                    z_hat_L_latent, source_detached, attention_mask
                )

        if self.config.use_amtm:
            if self.config.amtm_masking_strategy == "adaptive":
                mask_N = self.pretrain_loss.compute_adaptive_mask(
                    source_prime, source_detached, K
                )
            else:
                mask_N = self.pretrain_loss.compute_random_mask(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_ratio=self.config.random_mask_ratio,
                )

            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_N.bool()] = self.config.mask_token_id

            # Recompute entropy on masked input (positions are different)
            entropy_masked = self._compute_entropy(masked_input_ids)
            z_L_masked, source_masked, mask_L_masked = self.local_encoder(
                masked_input_ids, attention_mask, target_length,
                entropy_scores=entropy_masked,
            )
            z_L_prime_masked = self.latent_encoder(z_L_masked, mask_L_masked)
            z_hat_L_masked = self.latent_decoder(z_L_prime_masked, mask_L_masked)
            logits_amtm, _ = self.local_decoder(
                z_hat_L_masked, source_masked, attention_mask
            )

        # ===== Compute losses =====
        losses = self.pretrain_loss(
            logits_mtr=logits_mtr,
            logits_latent_mtr=logits_latent_mtr,
            logits_amtm=logits_amtm,
            mask_N=mask_N,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Entropy auxiliary loss (BLT-style): train entropy model to predict
        # per-position reconstruction difficulty.  The matching step is inside
        # torch.no_grad() so entropy_model receives NO gradient from the main
        # loss.  This auxiliary loss is the only training signal for it.
        if (self.entropy_model is not None
                and logits_mtr is not None
                and self.config.entropy_aux_loss_weight > 0):
            B_l, N_l, V = logits_mtr.shape
            with torch.no_grad():
                per_pos_ce = F.cross_entropy(
                    logits_mtr.reshape(-1, V),
                    input_ids.reshape(-1),
                    reduction="none",
                ).reshape(B_l, N_l)
                # Normalise to [0, 1] per sample for stable targets
                lo = per_pos_ce.min(dim=-1, keepdim=True).values
                hi = per_pos_ce.max(dim=-1, keepdim=True).values
                target_entropy = (per_pos_ce - lo) / (hi - lo + 1e-8)
                if attention_mask is not None:
                    target_entropy = target_entropy * attention_mask

            # Re-run entropy model (or reuse the scores computed earlier).
            # The scores must be from a graph-connected forward to get grads.
            entropy_pred = entropy_scores  # already computed with grad
            if entropy_pred is not None:
                entropy_aux = F.mse_loss(
                    entropy_pred * (attention_mask if attention_mask is not None else 1.0),
                    target_entropy,
                )
                losses["loss_entropy_aux"] = entropy_aux
                losses["loss"] = losses["loss"] + self.config.entropy_aux_loss_weight * entropy_aux

        # Learned compression regularization (DiffRate-style)
        if (self.config.use_learned_compression
                and self.local_encoder.compression_schedule is not None):
            # Target merge rate per window = window_size * (1 - compression_target)
            target_r = self.config.window_size * (1 - self.config.compression_target)
            comp_loss = self.local_encoder.compression_schedule.compression_loss(target_r)
            losses["loss_compression"] = comp_loss
            losses["loss"] = losses["loss"] + self.config.compression_loss_weight * comp_loss

        return losses

    def forward_with_intermediates(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning all intermediate representations for distillation.

        Runs the standard MTR path (encode -> latent encode -> latent decode ->
        local decode) and returns every intermediate tensor needed by the
        knowledge distillation framework.

        Returns:
            Dict with keys:
                z_L: [B, L, D] Local Encoder output.
                source: [B, N, L] source matrix.
                mask_L: [B, L] merged attention mask.
                z_L_prime: [B, L, D] Latent Encoder output.
                z_hat_L: [B, L, D] Latent Decoder output.
                logits: [B, N, vocab_size] reconstruction logits.
                entropy_scores: [B, N] or None.
        """
        _, N = input_ids.shape
        if target_length is None:
            target_length = self._sample_target_length(N)

        entropy_scores = self._compute_entropy(input_ids)

        z_L, source, mask_L = self.local_encoder(
            input_ids, attention_mask, target_length,
            entropy_scores=entropy_scores,
        )
        z_L_prime = self.latent_encoder(z_L, mask_L)
        z_hat_L = self.latent_decoder(z_L_prime, mask_L)
        logits, _ = self.local_decoder(z_hat_L, source, attention_mask)

        return {
            "z_L": z_L,
            "source": source,
            "mask_L": mask_L,
            "z_L_prime": z_L_prime,
            "z_hat_L": z_hat_L,
            "logits": logits,
            "entropy_scores": entropy_scores,
        }

    def forward_encode_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encoder-only forward for classification tasks.

        Returns the Latent Encoder output, averaged over valid tokens.

        Args:
            input_ids: [B, N]
            attention_mask: [B, N]

        Returns:
            pooled: [B, D] pooled representation.
        """
        z_L_prime, _, _, mask_L = self.encode(input_ids, attention_mask)

        # Mean pooling over valid tokens
        if mask_L is not None:
            mask = mask_L.unsqueeze(-1)  # [B, L, 1]
            pooled = (z_L_prime * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = z_L_prime.mean(dim=1)

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "pretrain",
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified forward pass.

        Args:
            input_ids: [B, N]
            attention_mask: [B, N]
            mode: 'pretrain', 'encode', or 'decode' (full autoencoder).
            labels: [B] classification labels (for 'encode' mode with cls head).

        Returns:
            Dict with outputs depending on mode.
        """
        if mode == "pretrain":
            return self.forward_pretrain(input_ids, attention_mask)
        elif mode == "encode":
            pooled = self.forward_encode_only(input_ids, attention_mask)
            return {"pooled_output": pooled}
        elif mode == "decode":
            z_L_prime, z_L, source, mask_L = self.encode(
                input_ids, attention_mask
            )
            logits, z_N = self.decode(z_L_prime, source, attention_mask, mask_L)
            return {"logits": logits, "pooled_output": z_L_prime.mean(dim=1)}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return total number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.local_encoder.embedding.weight.numel()
        return n_params


class MergeDNAForSequenceClassification(nn.Module):
    """MergeDNA with a classification head for downstream tasks.

    Uses the Latent Encoder output (encoder-only mode) with a linear
    classification head. Supports LoRA fine-tuning.
    """

    def __init__(self, config: MergeDNAConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.mergedna = MergeDNA(config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled = self.mergedna.forward_encode_only(input_ids, attention_mask)
        logits = self.classifier(pooled)

        result = {"logits": logits}

        if labels is not None:
            if self.num_classes == 1:
                loss = F.mse_loss(logits.squeeze(-1), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        return result

    def apply_lora(self, rank: int = 8, alpha: int = 16):
        """Apply LoRA to the model for parameter-efficient fine-tuning.

        Applies LoRA to all linear layers in the Latent Encoder.
        Requires the peft library.
        """
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.mergedna = get_peft_model(self.mergedna, lora_config)
        except ImportError:
            raise ImportError(
                "peft library required for LoRA. Install: pip install peft"
            )


class MergeDNAForTokenClassification(nn.Module):
    """MergeDNA with a token-level classification head.

    Uses the full autoencoder (encoder-decoder) to get per-position
    representations for token-level tasks like splice site prediction.
    """

    def __init__(self, config: MergeDNAConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.mergedna = MergeDNA(config)
        self.token_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.embed_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Full autoencoder forward
        z_L_prime, z_L, source, mask_L = self.mergedna.encode(
            input_ids, attention_mask
        )
        z_hat_L = self.mergedna.latent_decoder(z_L_prime, mask_L)
        _, z_N = self.mergedna.local_decoder(z_hat_L, source, attention_mask)

        logits = self.token_classifier(z_N)  # [B, N, num_classes]

        result = {"logits": logits}

        if labels is not None:
            logits_flat = logits.view(-1, self.num_classes)
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(
                logits_flat, labels_flat, ignore_index=-100
            )
            result["loss"] = loss

        return result
