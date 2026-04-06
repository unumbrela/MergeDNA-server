#!/usr/bin/env python3
"""Train a Sparse Autoencoder on MergeDNA / EfficientMergeDNA representations.

Usage:
    python scripts/train_sae.py \
        --checkpoint ./outputs/efficient_distill/checkpoint-50000.pt \
        --data_path ./data/pretrain/multi_species_genomes \
        --output_dir ./outputs/sae_analysis \
        --max_batches 500 \
        --sae_hidden_dim 4096 \
        --sae_epochs 50
"""

import os
import sys
import argparse
import logging
import yaml

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mergedna.model.mergedna import MergeDNA, MergeDNAConfig
from mergedna.data.dataset import MultiSpeciesGenomeDataset
from mergedna.data.tokenizer import DNACharTokenizer
from mergedna.data.collator import PretrainCollator
from mergedna.analysis.sparse_autoencoder import SAETrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_model_from_checkpoint(ckpt_path: str, device: str = "cuda") -> MergeDNA:
    """Load a MergeDNA model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config_dict = ckpt.get("config", {})

    model_config = MergeDNAConfig(
        embed_dim=config_dict.get("embed_dim", 512),
        num_heads=config_dict.get("num_heads", 8),
        local_encoder_layers=config_dict.get("local_encoder_layers", 3),
        latent_encoder_layers=config_dict.get("latent_encoder_layers", 8),
        latent_decoder_layers=config_dict.get("latent_decoder_layers", 2),
        local_decoder_layers=config_dict.get("local_decoder_layers", 1),
        window_size=config_dict.get("window_size", 16),
        max_seq_length=config_dict.get("max_seq_length", 4096),
        compression_target=config_dict.get("compression_target", 0.2),
        compression_variance=config_dict.get("compression_variance", 0.05),
        use_entropy_guided_merging=config_dict.get("use_entropy_guided_merging", False),
        entropy_weight=config_dict.get("entropy_weight", 0.5),
        entropy_model_hidden_dim=config_dict.get("entropy_model_hidden_dim", 64),
        use_learned_compression=config_dict.get("use_learned_compression", False),
        r_min_per_window=config_dict.get("r_min_per_window", 1),
        r_max_per_window=config_dict.get("r_max_per_window", 8),
        latent_encoder_type=config_dict.get("latent_encoder_type", "transformer"),
        ssm_type=config_dict.get("ssm_type", "gated_deltanet"),
        attention_layer_indices=config_dict.get("attention_layer_indices", [3, 7]),
    )

    model = MergeDNA(model_config)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    logger.info(f"Loaded model: {model.get_num_params()/1e6:.1f}M params")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train SAE on MergeDNA representations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/sae_analysis")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=500)
    parser.add_argument("--sae_hidden_dim", type=int, default=4096)
    parser.add_argument("--sae_sparsity_lambda", type=float, default=1e-3)
    parser.add_argument("--sae_top_k", type=int, default=32)
    parser.add_argument("--sae_lr", type=float, default=1e-3)
    parser.add_argument("--sae_epochs", type=int, default=50)
    parser.add_argument("--sae_batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = build_model_from_checkpoint(args.checkpoint, args.device)

    # Build dataloader
    tokenizer = DNACharTokenizer(max_length=args.max_seq_length)
    dataset = MultiSpeciesGenomeDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        split="train",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=PretrainCollator(pad_token_id=0),
    )

    # Train SAE
    trainer = SAETrainer(config={
        "device": args.device,
        "sae_hidden_dim": args.sae_hidden_dim,
        "sae_sparsity_lambda": args.sae_sparsity_lambda,
        "sae_top_k": args.sae_top_k,
        "sae_lr": args.sae_lr,
        "sae_epochs": args.sae_epochs,
        "sae_batch_size": args.sae_batch_size,
    })

    logger.info("Step 1: Collecting representations...")
    reps = trainer.collect_representations(model, dataloader, max_batches=args.max_batches)

    logger.info("Step 2: Training SAE...")
    sae = trainer.train_sae(reps)

    # Save
    sae_path = os.path.join(args.output_dir, "sae.pt")
    trainer.save_sae(sae, sae_path)

    # Quick analysis: top features
    logger.info("Step 3: Analysing top features...")
    sample_reps = reps[:10000].to(args.device)
    top_idx, top_vals = sae.get_top_features(sample_reps, k=20)
    logger.info(f"Top 20 most active SAE features: {top_idx.tolist()}")
    logger.info(f"Mean activations: {top_vals.tolist()}")

    # Sparsity report
    h = sae.get_feature_activations(sample_reps)
    active_frac = (h > 0).float().mean(dim=0)
    n_active = (active_frac > 0.01).sum().item()
    logger.info(f"Features active in >1% of tokens: {n_active}/{sae.hidden_dim}")

    logger.info(f"SAE training complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
