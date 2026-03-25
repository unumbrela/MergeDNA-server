#!/bin/bash
# Fine-tune MergeDNA on Nucleotide Transformer benchmark (18 tasks)
# Reports MCC or F1

NT_TASKS=(
    "H3" "H3K4me1" "H3K4me2" "H3K4me3" "H3K9ac"
    "H3K14ac" "H3K36me3" "H3K79me3" "H4" "H4ac"
    "enhancers" "enhancers_types" "promoter_all"
    "promoter_no_tata" "promoter_tata"
    "splice_sites_all" "splice_sites_acceptors" "splice_sites_donors"
)

PRETRAIN_CKPT="./outputs/pretrain/checkpoint-100000.pt"

for TASK in "${NT_TASKS[@]}"; do
    echo "===== Fine-tuning on $TASK ====="
    python train.py \
        --config configs/finetune_nt_benchmark.yaml \
        --mode finetune \
        --task_name "$TASK" \
        --pretrain_ckpt "$PRETRAIN_CKPT" \
        --output_dir "./outputs/finetune/nt_benchmark/$TASK"
done
