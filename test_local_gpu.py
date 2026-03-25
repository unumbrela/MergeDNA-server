"""Smoke test: run a few pretrain steps on GPU to verify VRAM fits.

Usage: python test_local_gpu.py
"""

import torch
import time


def test_gpu_pretrain():
    """Test pretrain forward+backward on GPU with local config."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    print()

    from mergedna.model.mergedna import MergeDNA, MergeDNAConfig

    # Local config: ~95M params
    config = MergeDNAConfig(
        vocab_size=10,
        embed_dim=512,
        num_heads=8,
        local_encoder_layers=4,
        latent_encoder_layers=12,
        latent_decoder_layers=4,
        local_decoder_layers=2,
        window_size=16,
        dropout=0.0,
        use_flash_attn=True,
        max_seq_length=1024,
        gradient_checkpointing=True,
    )

    model = MergeDNA(config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M parameters")
    print(f"Gradient checkpointing: {config.gradient_checkpointing}")
    print()

    # Test with batch_size=2, seq_len=1024 (matches local config)
    B, N = 2, 1024
    input_ids = torch.randint(5, 9, (B, N), device=device)
    attention_mask = torch.ones(B, N, device=device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Forward
    print("Running pretrain forward pass...")
    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        losses = model.forward_pretrain(input_ids, attention_mask)
    torch.cuda.synchronize()
    t_fwd = time.time() - t0

    mem_fwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Forward: {t_fwd:.2f}s | Peak VRAM: {mem_fwd:.2f} GB")
    print(f"  Loss: {losses['loss'].item():.4f}")

    # Backward
    print("Running backward pass...")
    t0 = time.time()
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(losses["loss"]).backward()
    torch.cuda.synchronize()
    t_bwd = time.time() - t0

    mem_bwd = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Backward: {t_bwd:.2f}s | Peak VRAM: {mem_bwd:.2f} GB")

    # Optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    mem_final = torch.cuda.max_memory_allocated() / 1e9
    print(f"  After optimizer step: Peak VRAM: {mem_final:.2f} GB")
    print()

    if mem_final < vram_gb:
        print(f"OK - Peak {mem_final:.2f} GB fits in {vram_gb:.1f} GB VRAM")
    else:
        print(f"WARNING - Peak {mem_final:.2f} GB exceeds {vram_gb:.1f} GB VRAM!")

    # Cleanup
    del model, losses, optimizer, scaler
    torch.cuda.empty_cache()

    print()
    print("=" * 50)
    print("Smoke test PASSED")


def test_gpu_finetune():
    """Test finetune forward+backward on GPU."""
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    from mergedna.model.mergedna import MergeDNAConfig, MergeDNAForSequenceClassification

    config = MergeDNAConfig(
        vocab_size=10,
        embed_dim=512,
        num_heads=8,
        local_encoder_layers=4,
        latent_encoder_layers=12,
        latent_decoder_layers=4,
        local_decoder_layers=2,
        window_size=16,
        gradient_checkpointing=True,
    )

    model = MergeDNAForSequenceClassification(config, num_classes=2).to(device)
    print()
    print("Fine-tune test (batch=16, seq=1024, 2 classes)...")

    torch.cuda.reset_peak_memory_stats()
    B, N = 16, 1024
    input_ids = torch.randint(5, 9, (B, N), device=device)
    attention_mask = torch.ones(B, N, device=device)
    labels = torch.randint(0, 2, (B,), device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids, attention_mask, labels)

    out["loss"].backward()
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak VRAM: {mem:.2f} GB")
    print(f"  Loss: {out['loss'].item():.4f}")
    print("  Fine-tune smoke test PASSED")


if __name__ == "__main__":
    test_gpu_pretrain()
    test_gpu_finetune()
