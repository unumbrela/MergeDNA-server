"""MergeDNA Evaluation Script.

Evaluate pre-trained or fine-tuned MergeDNA on various benchmarks:
- Genomic Benchmark (8 tasks, top-1 accuracy)
- NT Benchmark (18 tasks, MCC or F1)
- GUE Benchmark (24 tasks, MCC or F1)

Usage:
    python evaluate.py --config configs/finetune_genomic_benchmark.yaml \
        --checkpoint ./outputs/finetune/best_model.pt \
        --benchmark genomic_benchmark --task_name human_enhancers_cohn
"""

import argparse
import logging
import yaml
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    matthews_corrcoef, f1_score, accuracy_score, roc_auc_score
)

from mergedna.model.mergedna import (
    MergeDNAConfig,
    MergeDNAForSequenceClassification,
    MergeDNAForTokenClassification,
)
from mergedna.data.tokenizer import DNACharTokenizer
from mergedna.data.dataset import GenomicBenchmarkDataset, NTBenchmarkDataset
from mergedna.data.collator import FineTuneCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MergeDNA-Eval")


@torch.no_grad()
def evaluate_classification(
    model,
    dataloader,
    device,
    num_classes: int = 2,
):
    """Evaluate sequence classification model."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)

        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(batch["labels"].cpu().numpy().tolist())

    preds = np.array(all_preds)
    probs = np.array(all_probs)
    labels = np.array(all_labels)

    results = {
        "accuracy": accuracy_score(labels, preds),
        "mcc": matthews_corrcoef(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

    # AUROC for binary classification
    if num_classes == 2:
        try:
            results["auroc"] = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            results["auroc"] = 0.0

    return results


def run_evaluation(config: dict, checkpoint: str, task_name: str):
    """Run evaluation on a single task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DNACharTokenizer(max_length=config.get("max_seq_length", 4096))
    gb_data_dir = (
        config.get("genomic_benchmark_data_dir")
        or config.get("data_path")
    )
    nt_data_dir = (
        config.get("nt_benchmark_data_dir")
        or config.get("data_path")
    )

    # Load dataset
    if task_name in GenomicBenchmarkDataset.TASK_NAMES:
        test_ds = GenomicBenchmarkDataset(
            task_name=task_name, tokenizer=tokenizer, split="test",
            max_length=config.get("max_seq_length", 4096),
            data_path=gb_data_dir,
        )
    else:
        test_ds = NTBenchmarkDataset(
            task_name=task_name, tokenizer=tokenizer, split="test",
            max_length=config.get("max_seq_length", 4096),
            data_path=nt_data_dir,
        )

    test_loader = DataLoader(
        test_ds, batch_size=config.get("batch_size", 32),
        shuffle=False, num_workers=4, collate_fn=FineTuneCollator(),
    )

    # Build model
    model_config = MergeDNAConfig(
        vocab_size=config.get("vocab_size", 10),
        embed_dim=config.get("embed_dim", 1024),
        num_heads=config.get("num_heads", 16),
        local_encoder_layers=config.get("local_encoder_layers", 4),
        latent_encoder_layers=config.get("latent_encoder_layers", 20),
        latent_decoder_layers=config.get("latent_decoder_layers", 4),
        local_decoder_layers=config.get("local_decoder_layers", 2),
        window_size=config.get("window_size", 16),
        use_flash_attn=config.get("use_flash_attn", True),
    )

    model = MergeDNAForSequenceClassification(
        model_config, num_classes=test_ds.num_classes
    ).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)

    # Evaluate
    results = evaluate_classification(
        model, test_loader, device, test_ds.num_classes
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="MergeDNA Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="genomic_benchmark",
                        choices=["genomic_benchmark", "nt_benchmark"])
    parser.add_argument("--task_name", type=str, default=None,
                        help="Specific task. If None, runs all tasks.")
    parser.add_argument("--genomic_benchmark_data_dir", type=str, default=None)
    parser.add_argument("--nt_benchmark_data_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.genomic_benchmark_data_dir:
        config["genomic_benchmark_data_dir"] = args.genomic_benchmark_data_dir
    if args.nt_benchmark_data_dir:
        config["nt_benchmark_data_dir"] = args.nt_benchmark_data_dir

    if args.benchmark == "genomic_benchmark":
        tasks = [args.task_name] if args.task_name else GenomicBenchmarkDataset.TASK_NAMES
    else:
        tasks = [args.task_name] if args.task_name else [
            "H3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac",
            "H3K14ac", "H3K36me3", "H3K79me3", "H4", "H4ac",
            "enhancers", "enhancers_types", "promoter_all",
            "promoter_no_tata", "promoter_tata",
            "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
        ]

    all_results = {}
    for task in tasks:
        logger.info(f"Evaluating: {task}")
        ckpt = args.checkpoint
        task_ckpt = os.path.join(os.path.dirname(ckpt), task, "best_model.pt")
        if os.path.exists(task_ckpt):
            ckpt = task_ckpt

        try:
            results = run_evaluation(config, ckpt, task)
            all_results[task] = results
            logger.info(f"  {task}: {results}")
        except Exception as e:
            logger.error(f"  {task}: FAILED - {e}")

    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print(f"{'Task':<40} {'Accuracy':>10} {'MCC':>10} {'F1':>10}")
        print("-" * 70)
        accs, mccs, f1s = [], [], []
        for task, res in all_results.items():
            acc = res.get("accuracy", 0)
            mcc = res.get("mcc", 0)
            f1 = res.get("f1_macro", 0)
            accs.append(acc)
            mccs.append(mcc)
            f1s.append(f1)
            print(f"{task:<40} {acc:>10.4f} {mcc:>10.4f} {f1:>10.4f}")
        print("-" * 70)
        print(f"{'Average':<40} {np.mean(accs):>10.4f} {np.mean(mccs):>10.4f} {np.mean(f1s):>10.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
