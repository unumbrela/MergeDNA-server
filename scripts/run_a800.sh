#!/bin/bash
# =============================================================
#  MergeDNA Full Reproduction on A800 80GB (Single GPU)
#
#  This script runs the complete pipeline:
#    Step 1: Pre-training (100K steps, ~4.5 days)
#    Step 2: Fine-tuning on Genomic Benchmark (8 tasks)
#    Step 3: Fine-tuning on NT Benchmark (18 tasks)
#    Step 4: Fine-tuning on GUE Benchmark (24+ tasks)
#
#  Usage:
#    bash scripts/run_a800.sh           # Run all steps
#    bash scripts/run_a800.sh pretrain  # Only pre-training
#    bash scripts/run_a800.sh finetune  # Only fine-tuning (all 3 benchmarks)
#    bash scripts/run_a800.sh gb        # Only Genomic Benchmark
#    bash scripts/run_a800.sh nt        # Only NT Benchmark
#    bash scripts/run_a800.sh gue       # Only GUE Benchmark
# =============================================================

set -e
cd "$(dirname "$0")/.."

STEP="${1:-all}"
CKPT="./outputs/pretrain_a800/checkpoint-100000.pt"
CONFIG_FT="configs/finetune_a800.yaml"

echo "============================================"
echo "  MergeDNA Full Reproduction (A800 80GB)"
echo "============================================"
echo ""

# Check environment
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
gpu = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {gpu} ({mem:.0f} GB)')
assert mem >= 40, f'Need >=40GB VRAM, got {mem:.0f}GB'
"

# ============ Step 1: Pre-training ============
if [ "$STEP" = "all" ] || [ "$STEP" = "pretrain" ]; then
    echo ""
    echo "========== Step 1/4: Pre-training (100K steps) =========="
    echo ""

    # Check data
    if [ ! -d "./data/pretrain/multi_species_genomes" ]; then
        echo "ERROR: Pretrain data not found."
        echo "Copy data/pretrain/multi_species_genomes/ from local machine."
        exit 1
    fi

    python train.py \
        --config configs/pretrain_a800.yaml \
        --mode pretrain

    echo ""
    echo "Pre-training complete! Checkpoint: $CKPT"
fi

# ============ Step 2: Genomic Benchmark ============
if [ "$STEP" = "all" ] || [ "$STEP" = "finetune" ] || [ "$STEP" = "gb" ]; then
    echo ""
    echo "========== Step 2/4: Genomic Benchmark (8 tasks) =========="
    echo ""

    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Pretrain checkpoint not found: $CKPT"
        echo "Run pre-training first."
        exit 1
    fi

    python train.py \
        --config "$CONFIG_FT" \
        --mode finetune_all_gb \
        --pretrain_ckpt "$CKPT" \
        --output_dir ./outputs/finetune_a800/gb
fi

# ============ Step 3: NT Benchmark ============
if [ "$STEP" = "all" ] || [ "$STEP" = "finetune" ] || [ "$STEP" = "nt" ]; then
    echo ""
    echo "========== Step 3/4: NT Benchmark (18 tasks) =========="
    echo ""

    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Pretrain checkpoint not found: $CKPT"
        exit 1
    fi

    python train.py \
        --config "$CONFIG_FT" \
        --mode finetune_all_nt \
        --pretrain_ckpt "$CKPT" \
        --output_dir ./outputs/finetune_a800/nt
fi

# ============ Step 4: GUE Benchmark ============
if [ "$STEP" = "all" ] || [ "$STEP" = "finetune" ] || [ "$STEP" = "gue" ]; then
    echo ""
    echo "========== Step 4/4: GUE Benchmark (24+ tasks) =========="
    echo ""

    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Pretrain checkpoint not found: $CKPT"
        exit 1
    fi

    if [ ! -d "./data/gue_benchmark/GUE" ]; then
        echo "ERROR: GUE data not found at ./data/gue_benchmark/GUE"
        exit 1
    fi

    python train.py \
        --config "$CONFIG_FT" \
        --mode finetune_all_gue \
        --pretrain_ckpt "$CKPT" \
        --gue_data_dir ./data/gue_benchmark/GUE \
        --output_dir ./outputs/finetune_a800/gue
fi

echo ""
echo "============================================"
echo "  All done! Check outputs/ for results."
echo "============================================"
