"""Sparse Autoencoder (SAE) for interpreting merged token representations.

Trains an overcomplete sparse autoencoder on the latent representations
produced by MergeDNA's Latent Encoder. Each SAE feature (latent dimension)
ideally corresponds to an interpretable biological concept (motif, structural
element, regulatory pattern).

Reference:
- InterPLM (Simon & Zou, Nature Methods 2025): SAE on protein LMs
- Evo 2 SAE (Goodfire / Arc Institute, 2025): SAE on DNA foundation model
- Cunningham et al., "Sparse Autoencoders Find Highly Interpretable
  Features in Language Models" (ICLR 2024)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """TopK Sparse Autoencoder for merged token representations.

    Architecture:
        z ∈ R^D  →  encoder(z) = ReLU(W_enc @ z + b_enc) ∈ R^H  (sparse)
        h ∈ R^H  →  decoder(h) = W_dec @ h + b_dec ∈ R^D  (reconstruct)

    Loss = MSE(z, z_hat) + λ * L1(h)

    Args:
        input_dim: Dimension of input representations (embed_dim).
        hidden_dim: Dictionary size, typically 4x-16x input_dim.
        sparsity_lambda: L1 penalty coefficient on hidden activations.
        top_k: If > 0, use TopK activation instead of ReLU + L1.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 4096,
        sparsity_lambda: float = 1e-3,
        top_k: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_lambda = sparsity_lambda
        self.top_k = top_k

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Tie decoder weights to encoder (optional, improves quality)
        # self.decoder.weight = nn.Parameter(self.encoder.weight.t())

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode to sparse hidden activations."""
        h = self.encoder(z)
        if self.top_k > 0:
            # TopK activation: zero out all but top-k
            topk_vals, topk_idx = h.topk(self.top_k, dim=-1)
            h_sparse = torch.zeros_like(h)
            h_sparse.scatter_(-1, topk_idx, F.relu(topk_vals))
            return h_sparse
        return F.relu(h)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [..., D] representations (any batch shape).

        Returns:
            Dict with 'z_hat', 'h_sparse', 'loss_recon', 'loss_sparse', 'loss'.
        """
        h_sparse = self.encode(z)
        z_hat = self.decoder(h_sparse)

        loss_recon = F.mse_loss(z_hat, z)
        loss_sparse = h_sparse.abs().mean()
        loss = loss_recon + self.sparsity_lambda * loss_sparse

        return {
            "z_hat": z_hat,
            "h_sparse": h_sparse,
            "loss_recon": loss_recon,
            "loss_sparse": loss_sparse,
            "loss": loss,
        }

    def get_feature_activations(self, z: torch.Tensor) -> torch.Tensor:
        """Get sparse feature activations for analysis (no grad)."""
        with torch.no_grad():
            return self.encode(z)

    def get_top_features(
        self, z: torch.Tensor, k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get indices and values of top-k most active features."""
        h = self.get_feature_activations(z)
        # Mean activation across batch/sequence
        mean_act = h.reshape(-1, self.hidden_dim).mean(dim=0)
        vals, idx = mean_act.topk(k)
        return idx, vals


class SAETrainer:
    """Train a Sparse Autoencoder on collected model representations.

    Pipeline:
    1. Run frozen model on dataset, collect z_L_prime from target layer
    2. Train SAE on collected representations
    3. Analyze learned features

    Args:
        config: Dict with sae_hidden_dim, sae_sparsity_lambda, sae_top_k,
                sae_lr, sae_epochs, sae_batch_size, etc.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def collect_representations(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        max_batches: int = 1000,
    ) -> torch.Tensor:
        """Collect latent representations from a frozen model.

        Args:
            model: Frozen MergeDNA model.
            dataloader: Pre-training or evaluation dataloader.
            max_batches: Maximum number of batches to process.

        Returns:
            Tensor of shape [N_total, D] — flattened token representations.
        """
        model.eval()
        all_reps = []

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = model.forward_with_intermediates(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            z = out["z_L_prime"]  # [B, L, D]
            mask = out["mask_L"]  # [B, L] or None

            if mask is not None:
                # Only keep valid tokens
                for b in range(z.shape[0]):
                    valid = mask[b].bool()
                    all_reps.append(z[b, valid].cpu())
            else:
                all_reps.append(z.reshape(-1, z.shape[-1]).cpu())

            if (i + 1) % 100 == 0:
                logger.info(f"Collected {i+1}/{max_batches} batches")

        reps = torch.cat(all_reps, dim=0)
        logger.info(f"Collected {reps.shape[0]} token representations, dim={reps.shape[1]}")
        return reps

    def train_sae(
        self,
        representations: torch.Tensor,
    ) -> SparseAutoencoder:
        """Train SAE on collected representations.

        Args:
            representations: [N, D] token representations.

        Returns:
            Trained SparseAutoencoder.
        """
        D = representations.shape[1]
        hidden_dim = self.config.get("sae_hidden_dim", D * 8)
        sparsity_lambda = self.config.get("sae_sparsity_lambda", 1e-3)
        top_k = self.config.get("sae_top_k", 0)
        lr = self.config.get("sae_lr", 1e-3)
        epochs = self.config.get("sae_epochs", 50)
        batch_size = self.config.get("sae_batch_size", 2048)

        sae = SparseAutoencoder(
            input_dim=D,
            hidden_dim=hidden_dim,
            sparsity_lambda=sparsity_lambda,
            top_k=top_k,
        ).to(self.device)

        dataset = TensorDataset(representations)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

        logger.info(
            f"Training SAE: input_dim={D}, hidden_dim={hidden_dim}, "
            f"λ={sparsity_lambda}, top_k={top_k}, epochs={epochs}"
        )

        for epoch in range(epochs):
            total_loss = 0.0
            total_recon = 0.0
            total_sparse = 0.0
            n_batches = 0

            for (batch_z,) in loader:
                batch_z = batch_z.to(self.device)
                out = sae(batch_z)

                optimizer.zero_grad()
                out["loss"].backward()
                optimizer.step()

                total_loss += out["loss"].item()
                total_recon += out["loss_recon"].item()
                total_sparse += out["loss_sparse"].item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {total_loss/n_batches:.6f} "
                    f"(recon: {total_recon/n_batches:.6f}, "
                    f"sparse: {total_sparse/n_batches:.6f})"
                )

        return sae

    def save_sae(self, sae: SparseAutoencoder, path: str):
        """Save trained SAE."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": sae.state_dict(),
            "input_dim": sae.input_dim,
            "hidden_dim": sae.hidden_dim,
            "sparsity_lambda": sae.sparsity_lambda,
            "top_k": sae.top_k,
        }, path)
        logger.info(f"Saved SAE to {path}")

    def load_sae(self, path: str) -> SparseAutoencoder:
        """Load trained SAE."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        sae = SparseAutoencoder(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            sparsity_lambda=ckpt["sparsity_lambda"],
            top_k=ckpt.get("top_k", 0),
        ).to(self.device)
        sae.load_state_dict(ckpt["state_dict"])
        return sae
