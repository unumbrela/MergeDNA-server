#!/bin/bash
set -e
cd "$(dirname "$0")/.."

python scripts/run_all_experiments.py \
  --config configs/finetune_a800.yaml \
  --pretrain-config configs/pretrain_a800.yaml \
  --skip-existing \
  "$@"
