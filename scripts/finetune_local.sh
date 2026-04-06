#!/bin/bash
# Fine-tune MergeDNA on single GPU (RTX 5070 Ti 16GB)
# Usage:
#   bash scripts/finetune_local.sh
#   bash scripts/finetune_local.sh --genomic_benchmark_data_dir ~/.genomic_benchmarks
#   bash scripts/finetune_local.sh --task_name H3 --mode finetune  # Single NT task

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "  MergeDNA Fine-tuning (Local GPU)"
echo "========================================"

MODE="${1:-finetune_all_gb}"

# Treat leading option flags as overrides for the default GB sweep.
if [ "$1" = "" ] || [ "$MODE" = "finetune_all_gb" ] || [[ "$1" == --* ]]; then
    echo "Fine-tuning on all Genomic Benchmark tasks..."
    EXTRA_ARGS=("$@")
    if [ "$MODE" = "finetune_all_gb" ]; then
        EXTRA_ARGS=("${@:2}")
    fi
    python train.py \
        --config configs/finetune_local.yaml \
        --mode finetune_all_gb \
        "${EXTRA_ARGS[@]}"
else
    echo "Fine-tuning with custom args..."
    python train.py \
        --config configs/finetune_local.yaml \
        "$@"
fi
