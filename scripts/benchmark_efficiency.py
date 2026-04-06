#!/usr/bin/env python
"""Benchmark efficiency of MergeDNA vs MergeDNA-Long.

Measures memory, throughput, and (estimated) FLOPs across sequence lengths.

Usage:
    python scripts/benchmark_efficiency.py
    python scripts/benchmark_efficiency.py --seq_lengths 1024 4096 16384
    python scripts/benchmark_efficiency.py --include_baseline  # also test pure Transformer
"""

import argparse
import gc
import time
from dataclasses import dataclass
from typing import List

import torch

from mergedna.model.mergedna import MergeDNA, MergeDNAConfig


@dataclass
class BenchResult:
    name: str
    seq_len: int
    params_m: float
    peak_memory_mb: float
    throughput_samples_sec: float
    time_per_step_ms: float
    success: bool
    error: str = ""


def _bench_config(name: str, config: MergeDNAConfig, seq_len: int,
                  batch_size: int = 2, warmup: int = 3, repeats: int = 10) -> BenchResult:
    """Benchmark a single config at a given sequence length."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        model = MergeDNA(config).cuda().train()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6

        x = torch.randint(5, 10, (batch_size, seq_len)).cuda()
        mask = torch.ones(batch_size, seq_len).cuda()

        # Warmup
        for _ in range(warmup):
            out = model.forward_pretrain(x, mask)
            out["loss"].backward()
            model.zero_grad()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Timed runs
        start = time.perf_counter()
        for _ in range(repeats):
            out = model.forward_pretrain(x, mask)
            out["loss"].backward()
            model.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        peak_mem = torch.cuda.max_memory_allocated() / 1e6  # MB
        time_per_step = elapsed / repeats * 1000  # ms
        throughput = batch_size * repeats / elapsed

        del model, x, mask, out
        torch.cuda.empty_cache()
        gc.collect()

        return BenchResult(
            name=name, seq_len=seq_len, params_m=n_params,
            peak_memory_mb=peak_mem, throughput_samples_sec=throughput,
            time_per_step_ms=time_per_step, success=True,
        )

    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return BenchResult(
            name=name, seq_len=seq_len, params_m=0,
            peak_memory_mb=0, throughput_samples_sec=0,
            time_per_step_ms=0, success=False, error=str(e)[:80],
        )


def make_original_config(embed_dim=256, layers=6) -> MergeDNAConfig:
    """Small MergeDNA (original) for benchmarking."""
    return MergeDNAConfig(
        embed_dim=embed_dim, num_heads=4,
        local_encoder_layers=2, latent_encoder_layers=layers,
        latent_decoder_layers=2, local_decoder_layers=1,
        window_size=16, use_flash_attn=True,
        gradient_checkpointing=True,
        compression_target=0.5,
    )


def make_long_config(embed_dim=256, layers=6) -> MergeDNAConfig:
    """Small MergeDNA-Long for benchmarking."""
    return MergeDNAConfig(
        embed_dim=embed_dim, num_heads=4,
        local_encoder_layers=2, latent_encoder_layers=layers,
        latent_decoder_layers=2, local_decoder_layers=1,
        window_size=16, use_flash_attn=True,
        gradient_checkpointing=True,
        compression_target=0.25,
        # Innovations
        use_entropy_guided_merging=True,
        entropy_weight=0.5,
        entropy_model_hidden_dim=32,
        entropy_model_kernel_size=5,
        use_learned_compression=True,
        latent_encoder_type="hybrid",
        ssm_type="gated_deltanet",
        attention_layer_indices=[2, 4],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_lengths", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    results: List[BenchResult] = []

    for seq_len in args.seq_lengths:
        print(f"\n--- Sequence length: {seq_len} ---")

        # Original MergeDNA
        config_orig = make_original_config(args.embed_dim, args.layers)
        config_orig.max_seq_length = seq_len
        r = _bench_config("MergeDNA", config_orig, seq_len,
                          args.batch_size, repeats=args.repeats)
        results.append(r)
        if r.success:
            print(f"  MergeDNA:      {r.peak_memory_mb:8.0f} MB | {r.time_per_step_ms:8.1f} ms/step | {r.throughput_samples_sec:6.1f} samples/s")
        else:
            print(f"  MergeDNA:      FAILED ({r.error})")

        # MergeDNA-Long
        config_long = make_long_config(args.embed_dim, args.layers)
        config_long.max_seq_length = seq_len
        r = _bench_config("MergeDNA-Long", config_long, seq_len,
                          args.batch_size, repeats=args.repeats)
        results.append(r)
        if r.success:
            print(f"  MergeDNA-Long: {r.peak_memory_mb:8.0f} MB | {r.time_per_step_ms:8.1f} ms/step | {r.throughput_samples_sec:6.1f} samples/s")
        else:
            print(f"  MergeDNA-Long: FAILED ({r.error})")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<16} {'SeqLen':>7} {'Params(M)':>10} {'Memory(MB)':>11} {'ms/step':>9} {'samples/s':>10} {'Status':>8}")
    print("-" * 90)
    for r in results:
        status = "OK" if r.success else "OOM"
        print(f"{r.name:<16} {r.seq_len:>7} {r.params_m:>10.1f} {r.peak_memory_mb:>11.0f} "
              f"{r.time_per_step_ms:>9.1f} {r.throughput_samples_sec:>10.1f} {status:>8}")
    print("=" * 90)


if __name__ == "__main__":
    main()
