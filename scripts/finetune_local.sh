#!/bin/bash
# Fine-tune MergeDNA on single GPU (RTX 5070 Ti 16GB)
# Usage:
#   bash scripts/finetune_local.sh                           # All GB tasks
#   bash scripts/finetune_local.sh --task_name H3 --mode finetune  # Single NT task

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "  MergeDNA Fine-tuning (Local GPU)"
echo "========================================"

MODE="${1:-finetune_all_gb}"

if [ "$MODE" = "finetune_all_gb" ] || [ "$1" = "" ]; then
    echo "Fine-tuning on all Genomic Benchmark tasks..."
    python train.py \
        --config configs/finetune_local.yaml \
        --mode finetune_all_gb \
        "${@:2}"
else
    echo "Fine-tuning with custom args..."
    python train.py \
        --config configs/finetune_local.yaml \
        "$@"
fi
