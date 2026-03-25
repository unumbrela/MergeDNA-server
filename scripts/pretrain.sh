#!/bin/bash
# MergeDNA Pre-training Script
# Paper setting: 4x A100-80G GPUs, 100K iterations

# Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# Single GPU
# python train.py --config configs/pretrain.yaml --data_path /path/to/data

# Multi-GPU with DDP (4 GPUs)
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    train.py \
    --config configs/pretrain.yaml \
    --data_path /path/to/multi_species_genomes \
    --output_dir ./outputs/pretrain
