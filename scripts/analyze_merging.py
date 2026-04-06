#!/usr/bin/env python3
"""Analyze merge patterns of MergeDNA / EfficientMergeDNA.

Produces:
1. Token length distribution across genomic contexts (like Fig. 3)
2. Merge boundary statistics
3. Compression ratio analysis
4. Teacher vs student pattern comparison (if both checkpoints provided)

Usage:
    python scripts/analyze_merging.py \
        --checkpoint ./outputs/efficient_distill/checkpoint-50000.pt \
        --data_path ./data/pretrain/multi_species_genomes \
        --output_dir ./outputs/merge_analysis

    # With teacher comparison:
    python scripts/analyze_merging.py \
        --checkpoint ./outputs/efficient_distill/checkpoint-50000.pt \
        --teacher_checkpoint ./outputs/pretrain_a800/checkpoint-100000.pt \
        --data_path ./data/pretrain/multi_species_genomes \
        --output_dir ./outputs/merge_analysis
"""

import os
import sys
import json
import argparse
import logging

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mergedna.data.dataset import MultiSpeciesGenomeDataset
from mergedna.data.tokenizer import DNACharTokenizer
from mergedna.data.collator import PretrainCollator
from mergedna.analysis.interpret_merging import MergePatternAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze MergeDNA merge patterns")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/merge_analysis")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Import here to avoid circular issues
    from scripts.train_sae import build_model_from_checkpoint

    # Load model
    model = build_model_from_checkpoint(args.checkpoint, args.device)
    analyzer = MergePatternAnalyzer(model, args.device)

    # Build dataloader
    tokenizer = DNACharTokenizer(max_length=args.max_seq_length)
    dataset = MultiSpeciesGenomeDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        split="train",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=PretrainCollator(pad_token_id=0),
    )

    # Collect statistics
    all_stats = []
    all_token_dists = []

    logger.info(f"Analyzing merge patterns over {args.num_batches} batches...")
    for i, batch in enumerate(dataloader):
        if i >= args.num_batches:
            break

        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)

        stats = analyzer.compute_merge_stats(input_ids, attention_mask)
        all_stats.append(stats)

        dist = analyzer.compute_token_length_distribution(input_ids, attention_mask)
        all_token_dists.append(dist)

        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i+1}/{args.num_batches} batches")

    # Aggregate stats
    avg_stats = {}
    for key in all_stats[0]:
        vals = [s[key] for s in all_stats]
        avg_stats[key] = sum(vals) / len(vals)

    logger.info(f"Average merge statistics:")
    for k, v in avg_stats.items():
        logger.info(f"  {k}: {v:.4f}")

    # Aggregate token length distribution
    agg_dist = {}
    for dist in all_token_dists:
        for k, v in dist.items():
            agg_dist[k] = agg_dist.get(k, 0) + v
    total = sum(agg_dist.values())
    agg_dist = {k: v / total for k, v in sorted(agg_dist.items())}

    logger.info(f"Token length distribution:")
    for k, v in agg_dist.items():
        bar = "#" * int(v * 100)
        logger.info(f"  len={k:2d}: {v:.4f} {bar}")

    # Save results
    results = {
        "avg_stats": avg_stats,
        "token_length_distribution": {str(k): v for k, v in agg_dist.items()},
    }

    # Teacher-student comparison
    if args.teacher_checkpoint:
        logger.info("Loading teacher model for comparison...")
        teacher_model = build_model_from_checkpoint(
            args.teacher_checkpoint, args.device
        )

        comparisons = []
        for i, batch in enumerate(dataloader):
            if i >= min(args.num_batches, 50):
                break
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            comp = analyzer.compare_teacher_student_patterns(
                teacher_model, model, input_ids, attention_mask
            )
            comparisons.append(comp)

        avg_iou = sum(c["boundary_iou"] for c in comparisons) / len(comparisons)
        results["teacher_student_comparison"] = {
            "avg_boundary_iou": avg_iou,
        }
        logger.info(f"Teacher-Student boundary IoU: {avg_iou:.4f}")

    output_path = os.path.join(args.output_dir, "merge_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
