#!/bin/bash
# Fine-tune MergeDNA on all 8 Genomic Benchmark tasks
# Reports top-1 accuracy

python train.py \
    --config configs/finetune_genomic_benchmark.yaml \
    --mode finetune_all_gb \
    --pretrain_ckpt ./outputs/pretrain/checkpoint-100000.pt \
    --output_dir ./outputs/finetune/genomic_benchmark
