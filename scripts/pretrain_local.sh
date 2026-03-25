#!/bin/bash
# Pre-train MergeDNA on single GPU (RTX 5070 Ti 16GB)
#
# Quick test:  bash scripts/pretrain_local.sh --config configs/pretrain_quicktest.yaml  (~8 min)
# Full local:  bash scripts/pretrain_local.sh                                          (~11 hours)

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "  MergeDNA Pre-training (Local GPU)"
echo "========================================"
echo ""

# Check CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Check data
DATA_DIR="./data/pretrain/multi_species_genomes"
if [ ! -d "$DATA_DIR" ] && [ ! -f "$DATA_DIR" ]; then
    echo ""
    echo "ERROR: Pretrain data not found at $DATA_DIR"
    echo "Download and extract the Multi-Species Genomes dataset first."
    exit 1
fi

CONFIG="${1:-configs/pretrain_local.yaml}"
shift 2>/dev/null || true

echo ""
echo "Config: $CONFIG"
echo "Starting pre-training..."
echo ""

python train.py \
    --config "$CONFIG" \
    --mode pretrain \
    "$@"
