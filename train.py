"""MergeDNA Training Entry Point.

Usage:
    # Pre-training (single GPU)
    python train.py --config configs/pretrain_a800.yaml --mode pretrain

    # Fine-tuning single task
    python train.py --config configs/finetune_a800.yaml --mode finetune --task_name H3

    # Fine-tuning all Genomic Benchmark (8 tasks)
    python train.py --config configs/finetune_a800.yaml --mode finetune_all_gb

    # Fine-tuning all NT Benchmark (18 tasks)
    python train.py --config configs/finetune_a800.yaml --mode finetune_all_nt

    # Fine-tuning all GUE Benchmark (24+ tasks)
    python train.py --config configs/finetune_a800.yaml --mode finetune_all_gue \
        --gue_data_dir ./data/gue_benchmark/GUE
"""

import os
import argparse
import logging
import yaml
import json
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MergeDNA")


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v
    return config


def configure_torch_runtime(config: dict) -> None:
    """Enable low-risk CUDA runtime optimizations."""
    if not torch.cuda.is_available():
        return

    if config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _summary_path(name: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_results.json")


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _task_result_path(task_output_dir: str) -> str:
    return os.path.join(task_output_dir, "results.json")


def _load_existing_task_result(
    task_name: str,
    task_output_dir: str,
    summary_name: str,
    root_output_dir: str,
):
    result_path = _task_result_path(task_output_dir)
    if os.path.exists(result_path):
        payload = _load_json(result_path)
        metrics = payload.get("metrics")
        if metrics is not None:
            return metrics
        return payload

    summary_path = _summary_path(summary_name, root_output_dir)
    if os.path.exists(summary_path):
        payload = _load_json(summary_path)
        if task_name in payload and os.path.exists(os.path.join(task_output_dir, "best_model.pt")):
            return payload[task_name]
    return None


def setup_distributed():
    """Setup distributed training if launched with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    return 0, 1


def run_pretrain(config: dict):
    """Run pre-training."""
    from mergedna.training.pretrain import PretrainRunner

    local_rank, world_size = setup_distributed()
    config["local_rank"] = local_rank
    config["world_size"] = world_size
    config["device"] = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    runner = PretrainRunner(config)
    runner.train()


def run_distill(config: dict):
    """Run knowledge distillation training for EfficientMergeDNA."""
    from mergedna.training.distill import DistillRunner

    local_rank, world_size = setup_distributed()
    config["local_rank"] = local_rank
    config["world_size"] = world_size
    config["device"] = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    runner = DistillRunner(config)
    runner.train()


def run_finetune(config: dict, task_name: str = None):
    """Run fine-tuning on a single task."""
    from mergedna.training.finetune import FineTuneRunner
    from mergedna.data.tokenizer import DNACharTokenizer
    from mergedna.data.dataset import (
        GenomicBenchmarkDataset, NTBenchmarkDataset, GUEBenchmarkDataset,
    )
    from mergedna.data.collator import FineTuneCollator

    if task_name:
        config["task_name"] = task_name

    runner = FineTuneRunner(config)
    tokenizer = DNACharTokenizer(max_length=config.get("max_seq_length", 4096))
    task = config["task_name"]
    max_len = config.get("max_seq_length", 4096)
    gb_data_dir = (
        config.get("genomic_benchmark_data_dir")
        or config.get("data_path")
    )
    nt_data_dir = (
        config.get("nt_benchmark_data_dir")
        or config.get("data_path")
    )

    # Determine dataset class
    gue_data_dir = config.get("gue_data_dir")
    if gue_data_dir and os.path.isdir(os.path.join(gue_data_dir, task.split("/")[-1] if "/" not in task else task)):
        # GUE task: task_name is like "EMP/H3" or just the task path
        task_path = os.path.join(gue_data_dir, task)
        train_ds = GUEBenchmarkDataset(task_path, tokenizer, "train", max_len)
        # GUE uses dev.csv for evaluation (test.csv may not have labels)
        eval_split = "test" if os.path.exists(os.path.join(task_path, "test.csv")) else "dev"
        test_ds = GUEBenchmarkDataset(task_path, tokenizer, eval_split, max_len)
    elif task in GenomicBenchmarkDataset.TASK_NAMES:
        train_ds = GenomicBenchmarkDataset(
            task, tokenizer, "train", max_len, data_path=gb_data_dir
        )
        test_ds = GenomicBenchmarkDataset(
            task, tokenizer, "test", max_len, data_path=gb_data_dir
        )
    else:
        train_ds = NTBenchmarkDataset(
            task, tokenizer, "train", max_len, data_path=nt_data_dir
        )
        test_ds = NTBenchmarkDataset(
            task, tokenizer, "test", max_len, data_path=nt_data_dir
        )

    collator = FineTuneCollator()
    nw = config.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=config.get("batch_size", 32),
        shuffle=True, num_workers=nw, collate_fn=collator,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.get("batch_size", 32),
        shuffle=False, num_workers=nw, collate_fn=collator,
    )

    num_classes = train_ds.num_classes
    model = runner.build_model(
        num_classes=num_classes,
        task_type=config.get("task_type", "sequence_classification"),
        pretrain_ckpt=config.get("pretrain_ckpt"),
    )

    results = runner.train(model, train_loader, test_loader)
    logger.info(f"Task {task} | Best results: {results}")
    return results


def run_finetune_all_gb(config: dict):
    """Run fine-tuning on all 8 Genomic Benchmark tasks."""
    from mergedna.data.dataset import GenomicBenchmarkDataset

    all_results = {}
    for task_name in GenomicBenchmarkDataset.TASK_NAMES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fine-tuning on: {task_name}")
        logger.info(f"{'='*60}")

        task_config = config.copy()
        task_config["output_dir"] = os.path.join(
            config.get("output_dir", "./outputs"), "gb", task_name
        )
        if config.get("skip_existing"):
            existing = _load_existing_task_result(
                task_name,
                task_config["output_dir"],
                "GENOMIC BENCHMARK",
                config.get("output_dir", "./outputs"),
            )
            if existing is not None:
                logger.info(f"Skipping completed GB task: {task_name}")
                all_results[task_name] = existing
                if not os.path.exists(_task_result_path(task_config["output_dir"])):
                    _save_json(
                        _task_result_path(task_config["output_dir"]),
                        {"task_name": task_name, "metrics": existing},
                    )
                continue
        started_at = time.time()
        try:
            results = run_finetune(task_config, task_name=task_name)
            all_results[task_name] = results
            _save_json(
                _task_result_path(task_config["output_dir"]),
                {
                    "task_name": task_name,
                    "metrics": results,
                    "duration_seconds": time.time() - started_at,
                },
            )
        except Exception as e:
            logger.error(f"Failed on {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}

    _print_summary("GENOMIC BENCHMARK", all_results, config)
    return all_results


def run_finetune_all_nt(config: dict):
    """Run fine-tuning on all 18 NT Benchmark tasks."""
    nt_tasks = [
        "H3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac",
        "H3K14ac", "H3K36me3", "H3K79me3", "H4", "H4ac",
        "enhancers", "enhancers_types", "promoter_all",
        "promoter_no_tata", "promoter_tata",
        "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
    ]

    all_results = {}
    for task_name in nt_tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fine-tuning on NT: {task_name}")
        logger.info(f"{'='*60}")

        task_config = config.copy()
        task_config["output_dir"] = os.path.join(
            config.get("output_dir", "./outputs"), "nt", task_name
        )
        if config.get("skip_existing"):
            existing = _load_existing_task_result(
                task_name,
                task_config["output_dir"],
                "NT BENCHMARK",
                config.get("output_dir", "./outputs"),
            )
            if existing is not None:
                logger.info(f"Skipping completed NT task: {task_name}")
                all_results[task_name] = existing
                if not os.path.exists(_task_result_path(task_config["output_dir"])):
                    _save_json(
                        _task_result_path(task_config["output_dir"]),
                        {"task_name": task_name, "metrics": existing},
                    )
                continue
        started_at = time.time()
        try:
            results = run_finetune(task_config, task_name=task_name)
            all_results[task_name] = results
            _save_json(
                _task_result_path(task_config["output_dir"]),
                {
                    "task_name": task_name,
                    "metrics": results,
                    "duration_seconds": time.time() - started_at,
                },
            )
        except Exception as e:
            logger.error(f"Failed on {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}

    _print_summary("NT BENCHMARK", all_results, config)
    return all_results


def run_finetune_all_gue(config: dict):
    """Run fine-tuning on all GUE Benchmark tasks."""
    gue_dir = config.get("gue_data_dir", "./data/gue_benchmark/GUE")
    if not os.path.isdir(gue_dir):
        raise FileNotFoundError(f"GUE data dir not found: {gue_dir}")

    # Discover all tasks: GUE/{category}/{task}/
    gue_tasks = []
    for category in sorted(os.listdir(gue_dir)):
        cat_dir = os.path.join(gue_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for task in sorted(os.listdir(cat_dir)):
            task_dir = os.path.join(cat_dir, task)
            if os.path.isdir(task_dir) and os.path.exists(os.path.join(task_dir, "train.csv")):
                gue_tasks.append(f"{category}/{task}")

    logger.info(f"Found {len(gue_tasks)} GUE tasks")

    all_results = {}
    for task_rel in gue_tasks:
        task_path = os.path.join(gue_dir, task_rel)
        logger.info(f"\n{'='*60}")
        logger.info(f"Fine-tuning on GUE: {task_rel}")
        logger.info(f"{'='*60}")

        task_config = config.copy()
        task_config["task_name"] = task_rel
        task_config["gue_data_dir"] = gue_dir
        task_config["output_dir"] = os.path.join(
            config.get("output_dir", "./outputs"), "gue", task_rel.replace("/", "_")
        )
        if config.get("skip_existing"):
            existing = _load_existing_task_result(
                task_rel,
                task_config["output_dir"],
                "GUE BENCHMARK",
                config.get("output_dir", "./outputs"),
            )
            if existing is not None:
                logger.info(f"Skipping completed GUE task: {task_rel}")
                all_results[task_rel] = existing
                if not os.path.exists(_task_result_path(task_config["output_dir"])):
                    _save_json(
                        _task_result_path(task_config["output_dir"]),
                        {"task_name": task_rel, "metrics": existing},
                    )
                continue
        started_at = time.time()
        try:
            results = run_finetune(task_config, task_name=task_rel)
            all_results[task_rel] = results
            _save_json(
                _task_result_path(task_config["output_dir"]),
                {
                    "task_name": task_rel,
                    "metrics": results,
                    "duration_seconds": time.time() - started_at,
                },
            )
        except Exception as e:
            logger.error(f"Failed on {task_rel}: {e}")
            all_results[task_rel] = {"error": str(e)}

    _print_summary("GUE BENCHMARK", all_results, config)
    return all_results


def _print_summary(name: str, all_results: dict, config: dict):
    """Print and save results summary."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} RESULTS SUMMARY")
    logger.info(f"{'='*60}")

    accs, mccs = [], []
    for task, res in sorted(all_results.items()):
        if "error" in res:
            logger.info(f"  {task}: FAILED - {res['error'][:60]}")
        else:
            acc = res.get("accuracy", 0)
            mcc = res.get("mcc", 0)
            f1 = res.get("f1", 0)
            accs.append(acc)
            mccs.append(mcc)
            logger.info(f"  {task}: acc={acc:.4f} mcc={mcc:.4f} f1={f1:.4f}")

    if accs:
        logger.info(f"\n  Average Accuracy: {sum(accs)/len(accs):.4f}")
        logger.info(f"  Average MCC:      {sum(mccs)/len(mccs):.4f}")

    # Save to JSON
    out_dir = config.get("output_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = _summary_path(name, out_dir)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\n  Results saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="MergeDNA Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--mode", type=str, default="pretrain",
                        choices=["pretrain", "distill", "finetune",
                                 "finetune_all_gb", "finetune_all_nt", "finetune_all_gue"],
                        help="Training mode")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--gue_data_dir", type=str, default=None)
    parser.add_argument("--genomic_benchmark_data_dir", type=str, default=None)
    parser.add_argument("--nt_benchmark_data_dir", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()
    config = load_config(
        args.config,
        overrides={
            "data_path": args.data_path,
            "output_dir": args.output_dir,
            "pretrain_ckpt": args.pretrain_ckpt,
            "gue_data_dir": args.gue_data_dir,
            "genomic_benchmark_data_dir": args.genomic_benchmark_data_dir,
            "nt_benchmark_data_dir": args.nt_benchmark_data_dir,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "skip_existing": args.skip_existing,
        },
    )
    configure_torch_runtime(config)

    if args.mode == "pretrain":
        run_pretrain(config)
    elif args.mode == "distill":
        run_distill(config)
    elif args.mode == "finetune":
        run_finetune(config, task_name=args.task_name)
    elif args.mode == "finetune_all_gb":
        run_finetune_all_gb(config)
    elif args.mode == "finetune_all_nt":
        run_finetune_all_nt(config)
    elif args.mode == "finetune_all_gue":
        run_finetune_all_gue(config)


if __name__ == "__main__":
    main()
