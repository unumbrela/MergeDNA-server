#!/bin/bash
set -e
cd "$(dirname "$0")/.."

python scripts/run_all_experiments.py \
  --config configs/finetune_local.yaml \
  --pretrain-config configs/pretrain_local.yaml \
  --skip-existing \
  "$@"
