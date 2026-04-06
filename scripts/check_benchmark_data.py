"""Validate local benchmark datasets needed for downstream evaluation."""

import sys
from pathlib import Path

from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mergedna.data.dataset import GenomicBenchmarkDataset


def check_genomic_benchmark() -> None:
    root = Path("data/genomic_benchmark_sequences")
    print("[Genomic Benchmark]")
    if not root.is_dir():
        print(f"  MISSING: {root}")
        return
    for task in GenomicBenchmarkDataset.TASK_NAMES:
        train_dir = root / task / "train"
        test_dir = root / task / "test"
        train_classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        test_classes = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
        print(f"  {task}: train={train_classes} test={test_classes}")


def check_nt_benchmark() -> None:
    root = Path("data/nt_benchmark")
    print("[NT Benchmark]")
    arrow_paths = sorted(
        root.glob(
            "InstaDeepAI___nucleotide_transformer_downstream_tasks/**/"
            "nucleotide_transformer_downstream_tasks-train.arrow"
        )
    )
    if not arrow_paths:
        print(f"  MISSING aggregated cache under {root}")
        return
    ds = Dataset.from_file(str(arrow_paths[0]))
    tasks = sorted(ds.unique("task"))
    print(f"  tasks={len(tasks)}")
    print(f"  names={tasks}")


def check_gue() -> None:
    root = Path("data/gue_benchmark/GUE")
    print("[GUE]")
    if not root.is_dir():
        print(f"  MISSING: {root}")
        return
    tasks = []
    for category in sorted(root.iterdir()):
        if not category.is_dir():
            continue
        for task in sorted(category.iterdir()):
            if task.is_dir() and (task / "train.csv").exists():
                tasks.append(f"{category.name}/{task.name}")
    print(f"  tasks={len(tasks)}")
    print(f"  names={tasks}")


if __name__ == "__main__":
    check_genomic_benchmark()
    check_nt_benchmark()
    check_gue()
