"""Step-by-step walkthrough of MergeDNA's core idea.

This script is intentionally small and pedagogical. It traces the main path:

1. DNA bases -> Local Encoder -> dynamic merged tokens
2. merged tokens -> Latent Encoder -> global context
3. latent tokens -> Local Decoder -> base-level reconstruction
4. latent token selection -> adaptive mask -> masked modeling targets

Usage:
    python scripts/core_idea_walkthrough.py
    python scripts/core_idea_walkthrough.py --sequence ATATATATACGTGCTAAGTCGGTAGGGGCCCCGGCCGCGC
    python scripts/core_idea_walkthrough.py --ckpt outputs/pretrain_quicktest/checkpoint-50.pt
"""

import argparse
import os
import sys
from dataclasses import fields
from typing import Dict, Iterable

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mergedna.data.tokenizer import DNACharTokenizer
from mergedna.model.local_decoder import token_unmerge
from mergedna.model.mergedna import MergeDNA, MergeDNAConfig


def build_demo_config() -> MergeDNAConfig:
    """A small CPU-friendly config for understanding the execution flow."""
    return MergeDNAConfig(
        vocab_size=10,
        embed_dim=128,
        num_heads=4,
        local_encoder_layers=2,
        latent_encoder_layers=2,
        latent_decoder_layers=1,
        local_decoder_layers=1,
        window_size=8,
        dropout=0.0,
        use_flash_attn=False,
        max_seq_length=64,
        compression_target=0.5,
        compression_variance=0.0,
        lambda_latent=0.25,
    )


def filter_config_dict(raw: Dict) -> Dict:
    """Keep only keys that exist in MergeDNAConfig."""
    valid_keys = {f.name for f in fields(MergeDNAConfig)}
    return {k: v for k, v in raw.items() if k in valid_keys}


def load_model(ckpt_path: str | None) -> MergeDNA:
    """Create a model, optionally loading a checkpoint."""
    if ckpt_path is None:
        model = MergeDNA(build_demo_config())
        model.eval()
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_config = ckpt.get("config", {})
    config = MergeDNAConfig(**filter_config_dict(raw_config))
    config.use_flash_attn = False
    config.gradient_checkpointing = False
    model = MergeDNA(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def default_sequence() -> str:
    """Toy sequence with repeated and non-repeated regions."""
    return (
        "ATATATATATATATAT"
        "ACGTGCTAAGTCGGTA"
        "GGGGCCCCGGCCGCGC"
    )


def nonzero_positions(mask: torch.Tensor) -> Iterable[int]:
    """Convert a 1D binary tensor to a list of selected indices."""
    return mask.nonzero(as_tuple=False).squeeze(-1).tolist()


def render_prediction(ids: Iterable[int], tokenizer: DNACharTokenizer) -> str:
    """Render predicted token ids as a fixed-length DNA-like string."""
    out = []
    for idx in ids:
        token = tokenizer.id_to_token.get(int(idx), "[UNK]")
        if token in {"A", "T", "C", "G", "N"}:
            out.append(token)
        else:
            out.append("?")
    return "".join(out)


def render_masked_sequence(sequence: str, mask: torch.Tensor) -> str:
    """Show masked positions using '*' while keeping sequence length unchanged."""
    chars = list(sequence)
    for pos in nonzero_positions(mask):
        chars[pos] = "*"
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default=default_sequence())
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = load_model(args.ckpt)
    config = model.config
    tokenizer = DNACharTokenizer(max_length=max(len(args.sequence), config.max_seq_length))

    batch = tokenizer([args.sequence], max_length=len(args.sequence))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    N = input_ids.shape[1]
    target_length = max(config.window_size, N // 2)
    K = max(config.window_size, target_length // 2)

    print("=" * 72)
    print("MergeDNA core-idea walkthrough")
    print("=" * 72)
    print(f"sequence        : {args.sequence}")
    print(f"sequence length : N = {N}")
    print(f"target merged L : {target_length}")
    print(f"selected latent K: {K}")
    print(f"checkpoint      : {args.ckpt or 'random init demo model'}")
    print()

    with torch.no_grad():
        z_L, source, mask_L = model.local_encoder(
            input_ids,
            attention_mask,
            target_length=target_length,
        )
        print("[1] Local Encoder = dynamic tokenization")
        print(f"    input_ids shape : {tuple(input_ids.shape)}")
        print(f"    z_L shape       : {tuple(z_L.shape)}")
        print(f"    source shape    : {tuple(source.shape)}")
        print(f"    mask_L shape    : {tuple(mask_L.shape) if mask_L is not None else None}")
        raw_token_lengths = source[0].sum(dim=0).long().tolist()
        local_token_lengths = [length for length in raw_token_lengths if length > 0]
        print(f"    token lengths   : {local_token_lengths}")
        print()

        z_L_prime = model.latent_encoder(z_L, mask_L)
        print("[2] Latent Encoder = global context modeling")
        print(f"    z_L_prime shape : {tuple(z_L_prime.shape)}")
        print()

        z_hat_L = model.latent_decoder(z_L_prime, mask_L)
        logits_mtr, z_N = model.local_decoder(z_hat_L, source, attention_mask)
        recon_ids = logits_mtr.argmax(dim=-1)[0].tolist()
        recon_seq = render_prediction(recon_ids, tokenizer)
        print("[3] Local Decoder = token unmerge + reconstruction")
        print(f"    z_hat_L shape   : {tuple(z_hat_L.shape)}")
        print(f"    z_N shape       : {tuple(z_N.shape)}")
        print(f"    logits shape    : {tuple(logits_mtr.shape)}")
        print(f"    recon sequence  : {recon_seq}")
        print("    note            : '?' means the random/logit-best token was a special token")
        print()

        _, z_K_prime, source_prime = model.latent_encoder.forward_with_selection(
            z_L, K, mask_L
        )
        z_L_from_K = token_unmerge(z_K_prime, source_prime.permute(0, 2, 1))
        print("[4] Global token selection = keep salient latent tokens")
        print(f"    z_K_prime shape : {tuple(z_K_prime.shape)}")
        print(f"    requested K     : {K}")
        print(f"    actual K        : {z_K_prime.shape[1]}")
        print(f"    source_prime    : {tuple(source_prime.shape)}")
        selected_group_sizes = source_prime[0].sum(dim=-1).long().tolist()
        print(f"    group sizes     : {selected_group_sizes}")
        print(f"    upsampled shape : {tuple(z_L_from_K.shape)}")
        print()

        mask_N = model.pretrain_loss.compute_adaptive_mask(source_prime, source, K)
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_N.bool()] = config.mask_token_id
        masked_seq = render_masked_sequence(args.sequence, mask_N[0])
        print("[5] Adaptive masking = mask what global merging thinks is important")
        print(f"    mask_N shape    : {tuple(mask_N.shape)}")
        print(f"    masked positions: {nonzero_positions(mask_N[0])}")
        print(f"    masked sequence : {masked_seq}")
        print()

        losses = model.forward_pretrain(input_ids, attention_mask)
        print("[6] Pretraining objective = MTR + latent MTR + AMTM")
        print(f"    total loss      : {losses['loss'].item():.4f}")
        print(f"    loss_mtr        : {losses['loss_mtr'].item():.4f}")
        print(f"    loss_latent_mtr : {losses['loss_latent_mtr'].item():.4f}")
        print(f"    loss_amtm       : {losses['loss_amtm'].item():.4f}")
        print()

    print("Reading guide:")
    print("  1. local_encoder.py: why token count shrinks from N to L")
    print("  2. token_merging.py: how source/source_prime are updated")
    print("  3. latent_encoder.py: why L tokens become contextualized")
    print("  4. local_decoder.py: how merged tokens return to base resolution")
    print("  5. losses.py: why adaptive masking focuses on high-information tokens")


if __name__ == "__main__":
    main()
