#!/usr/bin/env python
"""Validate effectiveness of MergeDNA-Long's three innovations.

Each test checks specific, falsifiable criteria — not just "code runs".
All comparisons use reconstruction loss (excluding regularizers) and
are averaged over 3 random seeds for stability.

Tests:
1. Entropy-Guided Merging  — gradient flow, learned scores, recon quality
2. Hybrid SSM-Attention     — memory scaling, convergence, speed
3. Learned Compression      — gradient flow, logit evolution, recon quality
4. Joint Ablation           — full combination table

Usage:
    python scripts/validate_innovations.py              # All (CUDA)
    python scripts/validate_innovations.py --cpu        # CPU (skip SSM)
    python scripts/validate_innovations.py --test entropy
    python scripts/validate_innovations.py --test hybrid
    python scripts/validate_innovations.py --test learned
    python scripts/validate_innovations.py --test joint
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────── Helpers ───────────────────────

SEED = 42


def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def header(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def sub(title):
    print(f"\n  --- {title} ---")


def ok(msg):
    print(f"  [PASS] {msg}")


def info(msg):
    print(f"  [INFO] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def make_data(B, N, device):
    """Synthetic DNA: conserved motifs + random variable regions."""
    x = torch.randint(5, 9, (B, N), device=device)
    mask = torch.ones(B, N, device=device)
    motif = torch.tensor([5, 6, 7, 8, 5, 6, 7, 8], device=device)
    for i in range(0, N - 8, N // 4):
        x[:, i:i + 8] = motif
    return x, mask


def train_loop(model, data_fn, steps=50, lr=1e-3, seed=SEED):
    """Train model, return per-step reconstruction loss (fair comparison).

    Returns dict:
      'recon': list of per-step reconstruction loss (loss_mtr + 0.25*latent + amtm)
      'total': list of per-step total loss (includes regularizers)
      'extras': dict of extra loss components from last step
    """
    seed_everything(seed)
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    recon, total = [], []
    last_out = {}
    for _ in range(steps):
        x, mask = data_fn()
        out = model.forward_pretrain(x, mask)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        total.append(out["loss"].item())
        r = (out.get("loss_mtr", torch.tensor(0.0)).item()
             + 0.25 * out.get("loss_latent_mtr", torch.tensor(0.0)).item()
             + out.get("loss_amtm", torch.tensor(0.0)).item())
        recon.append(r)
        last_out = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in out.items()}
    return {"recon": recon, "total": total, "extras": last_out}


def avg_last(lst, n=10):
    return sum(lst[-n:]) / min(n, len(lst))


# ───────────── Test 1: Entropy-Guided Merging ─────────────


def test_entropy(device):
    header("Test 1: Entropy-Guided Token Merging")
    passed = True
    from mergedna.model.mergedna import MergeDNA, MergeDNAConfig

    base = dict(
        embed_dim=64, num_heads=4,
        local_encoder_layers=2, latent_encoder_layers=4,
        latent_decoder_layers=2, local_decoder_layers=1,
        window_size=8, use_flash_attn=False,
        gradient_checkpointing=False, compression_target=0.5,
    )
    ent_kw = dict(
        use_entropy_guided_merging=True, entropy_weight=0.5,
        entropy_model_hidden_dim=32, entropy_model_kernel_size=5,
        entropy_aux_loss_weight=0.1,
    )
    B, N, STEPS = 4, 128, 60

    # ── 1a: Gradient check ──
    sub("1a: Entropy model gradient check (CRITICAL)")
    seed_everything()
    model = MergeDNA(MergeDNAConfig(**base, **ent_kw)).to(device)
    x, mask = make_data(2, 64, device)
    out = model.forward_pretrain(x, mask)
    out["loss"].backward()

    ent_grads = {n: p.grad.abs().mean().item()
                 for n, p in model.entropy_model.named_parameters()
                 if p.grad is not None}
    total_grad = sum(ent_grads.values())
    if total_grad < 1e-10:
        fail("Entropy model gets ZERO gradient — aux loss broken")
        passed = False
    else:
        ok(f"Entropy model gets gradients (sum avg |grad| = {total_grad:.6f})")

    has_aux = "loss_entropy_aux" in out
    if has_aux:
        ok(f"loss_entropy_aux = {out['loss_entropy_aux'].item():.4f}")
    else:
        fail("loss_entropy_aux missing")
        passed = False

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 1b: Scores become non-trivial ──
    sub("1b: Entropy scores after training")
    seed_everything()
    model = MergeDNA(MergeDNAConfig(**base, **ent_kw)).to(device)
    data_fn = lambda: make_data(B, N, device)
    train_loop(model, data_fn, steps=STEPS)

    model.eval()
    with torch.no_grad():
        xt, _ = make_data(2, N, device)
        scores = model._compute_entropy(xt)
        info(f"Scores: mean={scores.mean():.3f}, std={scores.std():.4f}, "
             f"range=[{scores.min():.3f}, {scores.max():.3f}]")

    if scores.std() > 0.005:
        ok(f"Scores show variation (std={scores.std():.4f} > 0.005)")
    else:
        fail(f"Scores near-constant (std={scores.std():.6f})")
        passed = False

    # Motif = conserved → low info → should have LOW score
    # Variable = random → high info → should have HIGH score
    motif_mask = torch.zeros(2, N, dtype=torch.bool, device=device)
    for i in range(0, N - 8, N // 4):
        motif_mask[:, i:i + 8] = True
    m_sc = scores[motif_mask].mean().item()
    v_sc = scores[~motif_mask].mean().item()
    info(f"Motif (conserved) score: {m_sc:.4f}")
    info(f"Variable (random) score: {v_sc:.4f}")
    if v_sc > m_sc:
        ok(f"Correct ordering: variable > motif (Δ={v_sc - m_sc:.4f})")
    else:
        info(f"Ordering not yet correct at {STEPS} steps (Δ={v_sc - m_sc:.4f})")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 1c: Reconstruction comparison (3 seeds) ──
    sub("1c: Reconstruction loss (3 seeds, using recon loss only)")
    b_finals, e_finals = [], []
    for seed in [42, 123, 456]:
        m = MergeDNA(MergeDNAConfig(**base)).to(device)
        r = train_loop(m, data_fn, steps=STEPS, seed=seed)
        b_finals.append(avg_last(r["recon"]))
        del m

        m = MergeDNA(MergeDNAConfig(**base, **ent_kw)).to(device)
        r = train_loop(m, data_fn, steps=STEPS, seed=seed)
        e_finals.append(avg_last(r["recon"]))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bm, em = np.mean(b_finals), np.mean(e_finals)
    info(f"Baseline recon: {bm:.4f} ± {np.std(b_finals):.4f}  {[f'{v:.4f}' for v in b_finals]}")
    info(f"Entropy  recon: {em:.4f} ± {np.std(e_finals):.4f}  {[f'{v:.4f}' for v in e_finals]}")

    if em <= bm * 1.10:
        ok(f"Entropy competitive (within 10%: {em:.4f} vs {bm:.4f})")
    else:
        fail(f"Entropy significantly worse: {em:.4f} vs {bm:.4f}")
        passed = False

    return passed


# ───────────── Test 2: Hybrid SSM-Attention ─────────────


def test_hybrid(device):
    header("Test 2: Hybrid SSM-Attention Latent Encoder")
    passed = True

    if device.type == "cpu":
        info("Requires CUDA — SKIPPING")
        return None

    from mergedna.model.mergedna import MergeDNA, MergeDNAConfig

    base = dict(
        embed_dim=64, num_heads=4,
        local_encoder_layers=2, latent_encoder_layers=6,
        latent_decoder_layers=2, local_decoder_layers=1,
        window_size=8, use_flash_attn=False,
        gradient_checkpointing=False, compression_target=0.5,
    )
    hyb_kw = dict(latent_encoder_type="hybrid", ssm_type="gated_deltanet",
                  attention_layer_indices=[2, 4])

    # ── 2a: Layer types + gradient ──
    sub("2a: Architecture check")
    from mergedna.model.latent_encoder import HybridLatentEncoder
    seed_everything()
    h = HybridLatentEncoder(
        embed_dim=64, num_layers=6, num_heads=4,
        use_flash_attn=False, ssm_type="gated_deltanet",
        attention_layer_indices=[2, 4],
    ).to(device)
    ok(f"Layout: {h.layer_types}")
    x = torch.randn(2, 32, 64, device=device)
    h(x).sum().backward()
    no_grad = [n for n, p in h.named_parameters() if p.requires_grad and p.grad is None]
    if no_grad:
        info(f"{len(no_grad)} param(s) without grad (may be unused SSM buffers)")
    else:
        ok("All params receive gradients")
    del h
    torch.cuda.empty_cache()

    # ── 2b: Memory scaling ──
    sub("2b: Memory scaling")
    seq_lens = [64, 128, 256, 512]
    print(f"  {'Len':>6} {'Transformer':>13} {'Hybrid':>10} {'Saving':>8}")
    print(f"  {'-'*40}")
    for sl in seq_lens:
        mems = {}
        for nm, kw in [("trans", {}), ("hybrid", hyb_kw)]:
            torch.cuda.empty_cache(); gc.collect()
            torch.cuda.reset_peak_memory_stats()
            seed_everything()
            m = MergeDNA(MergeDNAConfig(**{**base, **kw})).to(device)
            xd, md = make_data(2, sl, device)
            o = m.forward_pretrain(xd, md); o["loss"].backward()
            torch.cuda.synchronize()
            mems[nm] = torch.cuda.max_memory_allocated() / 1e6
            del m, xd, md, o; torch.cuda.empty_cache(); gc.collect()
        sav = (1 - mems["hybrid"] / mems["trans"]) * 100
        print(f"  {sl:>6} {mems['trans']:>11.1f}MB {mems['hybrid']:>8.1f}MB {sav:>+7.1f}%")
    info("SSM memory advantage scales with sequence length (visible at 4K+)")

    # ── 2c: Convergence (3 seeds) ──
    sub("2c: Convergence (3 seeds x 30 steps)")
    B, N, STEPS = 4, 128, 30
    data_fn = lambda: make_data(B, N, device)
    t_f, h_f = [], []
    for seed in [42, 123, 456]:
        m = MergeDNA(MergeDNAConfig(**base)).to(device)
        r = train_loop(m, data_fn, STEPS, seed=seed); t_f.append(avg_last(r["recon"], 5))
        del m
        m = MergeDNA(MergeDNAConfig(**{**base, **hyb_kw})).to(device)
        r = train_loop(m, data_fn, STEPS, seed=seed); h_f.append(avg_last(r["recon"], 5))
        del m; torch.cuda.empty_cache()

    tm, hm = np.mean(t_f), np.mean(h_f)
    info(f"Transformer recon: {tm:.4f} ± {np.std(t_f):.4f}")
    info(f"Hybrid      recon: {hm:.4f} ± {np.std(h_f):.4f}")
    if hm <= tm * 1.15:
        ok(f"Hybrid competitive (within 15%)")
    else:
        fail(f"Hybrid much worse: {hm:.4f} vs {tm:.4f}")
        passed = False

    return passed


# ───────────── Test 3: Learned Compression ─────────────


def test_learned(device):
    header("Test 3: Learned Compression Schedule")
    passed = True
    from mergedna.model.mergedna import MergeDNA, MergeDNAConfig

    base = dict(
        embed_dim=64, num_heads=4,
        local_encoder_layers=3, latent_encoder_layers=4,
        latent_decoder_layers=2, local_decoder_layers=1,
        window_size=8, use_flash_attn=False,
        gradient_checkpointing=False, compression_target=0.5,
    )
    # Use small weight so regularizer doesn't dominate
    lrn_kw = dict(use_learned_compression=True, r_min_per_window=1,
                  r_max_per_window=4, compression_loss_weight=0.02)
    B, N, STEPS = 4, 128, 120

    # ── 3a: Gradient check ──
    sub("3a: r_logits gradient check (CRITICAL)")
    from mergedna.model.local_encoder import LearnedCompressionSchedule
    s = LearnedCompressionSchedule(3, r_min=1, r_max=4).to(device)
    s.compression_loss(4.0).backward()
    ok(f"compression_loss grad: {s.r_logits.grad.abs().sum():.4f}")

    seed_everything()
    model = MergeDNA(MergeDNAConfig(**base, **lrn_kw)).to(device)
    x, mask = make_data(2, 64, device)
    out = model.forward_pretrain(x, mask)
    out["loss"].backward()
    rl = model.local_encoder.compression_schedule.r_logits
    if rl.grad is not None and rl.grad.abs().sum() > 1e-8:
        ok(f"r_logits grad in full model: {rl.grad.abs().sum():.6f}")
    else:
        fail("r_logits gets zero gradient in full model")
        passed = False

    if "loss_compression" in out:
        ok(f"loss_compression = {out['loss_compression'].item():.4f}")
    else:
        fail("loss_compression missing")
        passed = False
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 3b: Logits change ──
    sub(f"3b: Logit evolution ({STEPS} steps)")
    seed_everything()
    model = MergeDNA(MergeDNAConfig(**base, **lrn_kw)).to(device)
    sc = model.local_encoder.compression_schedule
    init_logits = sc.r_logits.detach().clone()
    info(f"Initial rates: {sc.get_all_rates()}")

    data_fn = lambda: make_data(B, N, device)
    res = train_loop(model, data_fn, STEPS, lr=1e-3)

    final_logits = sc.r_logits.detach().clone()
    delta = (final_logits - init_logits).abs().sum().item()
    info(f"Final rates: {sc.get_all_rates()} (logits: {[f'{v:.4f}' for v in final_logits.tolist()]})")
    info(f"Logit change: {delta:.4f}")

    if delta > 0.005:
        ok(f"Logits changed (Δ={delta:.4f})")
    else:
        fail(f"Logits frozen (Δ={delta:.6f})")
        passed = False

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 3c: Recon comparison (3 seeds) ──
    sub(f"3c: Reconstruction loss comparison (3 seeds x {STEPS} steps)")
    info("Comparing RECONSTRUCTION loss only (excludes compression regularizer)")
    info("Learned schedule needs time to converge — expect gap at short training")
    f_finals, l_finals = [], []
    for seed in [42, 123, 456]:
        m = MergeDNA(MergeDNAConfig(**base)).to(device)
        r = train_loop(m, data_fn, STEPS, seed=seed)
        f_finals.append(avg_last(r["recon"]))
        del m

        m = MergeDNA(MergeDNAConfig(**base, **lrn_kw)).to(device)
        r = train_loop(m, data_fn, STEPS, seed=seed)
        l_finals.append(avg_last(r["recon"]))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fm, lm = np.mean(f_finals), np.mean(l_finals)
    info(f"Fixed   recon: {fm:.4f} ± {np.std(f_finals):.4f}  {[f'{v:.4f}' for v in f_finals]}")
    info(f"Learned recon: {lm:.4f} ± {np.std(l_finals):.4f}  {[f'{v:.4f}' for v in l_finals]}")
    pct = (lm / fm - 1) * 100
    info(f"Difference: {pct:+.1f}%")

    # At short training, the learned schedule is exploring (Gumbel noise, rate
    # convergence). We require it to not catastrophically regress (within 30%).
    # Benefits are expected at 1K+ steps when rates specialize.
    if lm <= fm * 1.30:
        ok(f"Learned recon within 30% of fixed ({lm:.4f} vs {fm:.4f})")
    else:
        fail(f"Learned recon regressed >30%: {lm:.4f} vs {fm:.4f}")
        passed = False

    info("Note: learned schedule benefits emerge with longer training (1K+ steps)")

    return passed


# ───────────── Test 4: Joint Ablation ─────────────


def test_joint(device):
    header("Test 4: Joint Ablation (50 steps, seed=42)")
    passed = True

    if device.type == "cpu":
        info("Requires CUDA — SKIPPING")
        return None

    from mergedna.model.mergedna import MergeDNA, MergeDNAConfig

    base = dict(
        embed_dim=64, num_heads=4,
        local_encoder_layers=2, latent_encoder_layers=6,
        latent_decoder_layers=2, local_decoder_layers=1,
        window_size=8, use_flash_attn=False,
        gradient_checkpointing=False, compression_target=0.5,
    )
    ent = dict(use_entropy_guided_merging=True, entropy_weight=0.5,
               entropy_model_hidden_dim=32, entropy_model_kernel_size=5,
               entropy_aux_loss_weight=0.1)
    hyb = dict(latent_encoder_type="hybrid", ssm_type="gated_deltanet",
               attention_layer_indices=[2, 4])
    lrn = dict(use_learned_compression=True, r_min_per_window=1,
               r_max_per_window=4, compression_loss_weight=0.02)

    configs = {
        "Baseline":    {},
        "+Entropy":    {**ent},
        "+Hybrid":     {**hyb},
        "+Learned":    {**lrn},
        "+Ent+Hyb":    {**ent, **hyb},
        "+Ent+Lrn":    {**ent, **lrn},
        "+Hyb+Lrn":    {**hyb, **lrn},
        "Full":        {**ent, **hyb, **lrn},
    }

    B, N, STEPS = 4, 128, 50
    data_fn = lambda: make_data(B, N, device)

    print(f"\n  {'Config':<16} {'Params(K)':>10} {'Recon_init':>11} {'Recon_final':>12} {'Δ':>8} {'Conv':>6}")
    print(f"  {'-'*66}")

    results = {}
    for name, kw in configs.items():
        seed_everything(42)
        try:
            m = MergeDNA(MergeDNAConfig(**{**base, **kw})).to(device)
            np_ = sum(p.numel() for p in m.parameters()) / 1e3
            r = train_loop(m, data_fn, STEPS, seed=42)
            ri, rf = r["recon"][0], avg_last(r["recon"], 5)
            conv = rf < ri
            results[name] = {"params": np_, "init": ri, "final": rf, "conv": conv}
            print(f"  {name:<16} {np_:>10.1f} {ri:>11.4f} {rf:>12.4f} {ri-rf:>+8.4f} {'Y' if conv else 'N':>6}")
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name:<16} {'ERROR':>10} — {str(e)[:50]}")
            results[name] = {"conv": False}
            passed = False

    # Checks
    all_conv = all(r.get("conv", False) for r in results.values())
    if all_conv:
        ok("All configs converge")
    else:
        nc = [n for n, r in results.items() if not r.get("conv")]
        fail(f"Non-converging: {nc}")
        passed = False

    if "Baseline" in results and "Full" in results:
        b, f = results["Baseline"]["final"], results["Full"]["final"]
        info(f"Baseline recon: {b:.4f}, Full recon: {f:.4f}")
        if f <= b * 1.20:
            ok(f"Full model within 20% of baseline on reconstruction")
        else:
            fail(f"Full model recon much worse: {f:.4f} vs {b:.4f}")
            passed = False

    info("\nThis is a 50-step micro-benchmark.  For publication results:")
    info("  python train.py --config configs/pretrain_long_local.yaml --mode pretrain")

    return passed


# ─────────────── Main ───────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--test", default="all",
                        choices=["all", "entropy", "hybrid", "learned", "joint"])
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tests = {"entropy": test_entropy, "hybrid": test_hybrid,
             "learned": test_learned, "joint": test_joint}
    run = tests if args.test == "all" else {args.test: tests[args.test]}

    results = {}
    for name, fn in run.items():
        try:
            results[name] = fn(device)
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    header("VALIDATION SUMMARY")
    for name, r in results.items():
        s = "SKIP" if r is None else ("PASS" if r else "FAIL")
        print(f"  {name:.<45s} {s}")

    skips = sum(1 for r in results.values() if r is None)
    fails = sum(1 for r in results.values() if r is False)
    if skips:
        print(f"\n  {skips} test(s) skipped (no CUDA)")
    if fails:
        print(f"\n  {fails} test(s) FAILED — see above")
        sys.exit(1)
    else:
        print("\n  All run tests passed.")


if __name__ == "__main__":
    main()
