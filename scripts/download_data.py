"""Download all datasets needed for MergeDNA pre-training and evaluation.

Datasets:
1. Multi-Species Genomes (pre-training) - from HuggingFace
2. Genomic Benchmark (8 tasks) - from HuggingFace
3. Nucleotide Transformer Benchmark (18 tasks) - from HuggingFace
4. GUE Benchmark (24 tasks) - from Google Drive (manual)
5. SpliceAI dataset - from HuggingFace
"""

import os
import sys
import argparse

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def download_multi_species_genomes(save_dir):
    """Download Multi-Species Genomes corpus for pre-training."""
    from datasets import load_dataset
    print("=" * 60)
    print("Downloading Multi-Species Genomes (pre-training corpus)...")
    print("Source: InstaDeepAI/multi_species_genomes")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset("InstaDeepAI/multi_species_genomes", split="train",
                       cache_dir=save_dir)
    print(f"  Downloaded {len(ds)} sequences")
    print(f"  Saved to: {save_dir}")
    return ds


def download_genomic_benchmark(save_dir):
    """Download Genomic Benchmark (8 tasks)."""
    from datasets import load_dataset
    print("=" * 60)
    print("Downloading Genomic Benchmark (8 tasks)...")
    print("Source: InstaDeepAI/genomic-benchmarks")
    print("=" * 60)

    tasks = [
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
        "demo_coding_vs_intergenomic_seqs",
        "demo_human_or_worm",
        "dummy_mouse_enhancers_ensembl",
        "human_ensembl_regulatory",
        "human_nontata_promoters",
        "human_ocr_ensembl",
    ]

    os.makedirs(save_dir, exist_ok=True)
    for task in tasks:
        print(f"\n  Downloading: {task}...")
        try:
            ds = load_dataset(
                "InstaDeepAI/genomic-benchmarks", task,
                cache_dir=save_dir
            )
            for split in ds:
                print(f"    {split}: {len(ds[split])} samples")
        except Exception as e:
            print(f"    FAILED with InstaDeepAI source: {e}")
            print(f"    Trying katielink/genomic-benchmarks...")
            try:
                ds = load_dataset(
                    "katielink/genomic-benchmarks", task,
                    cache_dir=save_dir
                )
                for split in ds:
                    print(f"    {split}: {len(ds[split])} samples")
            except Exception as e2:
                print(f"    FAILED: {e2}")


def download_nt_benchmark(save_dir):
    """Download Nucleotide Transformer Benchmark (18 tasks)."""
    from datasets import load_dataset
    print("=" * 60)
    print("Downloading NT Benchmark (18 tasks)...")
    print("Source: InstaDeepAI/nucleotide_transformer_downstream_tasks")
    print("=" * 60)

    tasks = [
        "H3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac",
        "H3K14ac", "H3K36me3", "H3K79me3", "H4", "H4ac",
        "enhancers", "enhancers_types", "promoter_all",
        "promoter_no_tata", "promoter_tata",
        "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
    ]

    os.makedirs(save_dir, exist_ok=True)
    for task in tasks:
        print(f"\n  Downloading: {task}...")
        try:
            ds = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks",
                task,
                cache_dir=save_dir
            )
            for split in ds:
                print(f"    {split}: {len(ds[split])} samples")
        except Exception as e:
            print(f"    FAILED: {e}")


def download_nt_benchmark_revised(save_dir):
    """Download revised NT Benchmark."""
    from datasets import load_dataset
    print("=" * 60)
    print("Downloading NT Benchmark Revised...")
    print("Source: InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    try:
        ds = load_dataset(
            "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
            cache_dir=save_dir
        )
        print(f"  Downloaded successfully")
        for split in ds:
            print(f"    {split}: {len(ds[split])} samples")
    except Exception as e:
        print(f"  FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download MergeDNA datasets")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                        help="Root directory for data")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "pretrain", "genomic_benchmark",
                                 "nt_benchmark", "nt_revised"],
                        help="Which dataset to download")
    args = parser.parse_args()

    if args.dataset in ("all", "genomic_benchmark"):
        download_genomic_benchmark(
            os.path.join(args.data_root, "genomic_benchmark")
        )

    if args.dataset in ("all", "nt_benchmark"):
        download_nt_benchmark(
            os.path.join(args.data_root, "nt_benchmark")
        )

    if args.dataset in ("all", "pretrain"):
        download_multi_species_genomes(
            os.path.join(args.data_root, "pretrain")
        )

    if args.dataset == "nt_revised":
        download_nt_benchmark_revised(
            os.path.join(args.data_root, "nt_benchmark_revised")
        )

    print("\n" + "=" * 60)
    print("GUE Benchmark (24 tasks) - MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("Download from Google Drive:")
    print("  https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view?usp=sharing")
    print(f"Extract to: {os.path.join(args.data_root, 'gue_benchmark')}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
