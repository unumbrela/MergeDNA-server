"""Download datasets and reference repos needed by MergeDNA experiments."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

H38_REFERENCE_GENOME_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
H19_REFERENCE_GENOME_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz"
GENCODE_V24_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_24/"
    "gencode.v24.annotation.gtf.gz"
)
LRB_HF_BASE = "https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark/resolve/main"
PROTEINGYM_BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3"
PROTEINGYM_SUBS_ZIP = "DMS_ProteinGym_substitutions.zip"

PROTEIN_CDS_SOURCES = {
    "BLAT_ECOLX_Firnberg_2014": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=J01749.1&rettype=fasta_cds_na&retmode=text",
    "BRCA1_HUMAN_Findlay_2018": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NM_007294.4&rettype=fasta_cds_na&retmode=text",
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_url(url: str, dest: Path, force: bool = False) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and not force:
        print(f"  Skipping existing: {dest}")
        return dest
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to: {dest}")
    return dest


def extract_gzip(src: Path, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"  Skipping existing extract: {dest}")
        return dest
    ensure_dir(dest.parent)
    with gzip.open(src, "rb") as file_in, open(dest, "wb") as file_out:
        shutil.copyfileobj(file_in, file_out)
    print(f"  Extracted to: {dest}")
    return dest


def clone_repo(repo_url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and (dest / ".git").exists() and not force:
        print(f"  Repo already exists: {dest}")
        return dest
    if dest.exists() and force:
        shutil.rmtree(dest)
    ensure_dir(dest.parent)
    run(["git", "clone", "--depth", "1", repo_url, str(dest)])
    return dest


def maybe_git_lfs_pull(repo_dir: Path) -> None:
    try:
        run(["git", "lfs", "pull"], cwd=repo_dir)
    except Exception as exc:
        print(f"  WARNING: git lfs pull failed in {repo_dir}: {exc}")


def download_multi_species_genomes(save_dir: Path):
    """Download Multi-Species Genomes corpus for pre-training."""
    from datasets import load_dataset

    print("=" * 60)
    print("Downloading Multi-Species Genomes (pre-training corpus)...")
    print("Source: InstaDeepAI/multi_species_genomes")
    print("=" * 60)

    save_dir = ensure_dir(save_dir)
    cache_dir = ensure_dir(save_dir / ".cache" / "huggingface")
    dataset_dir = ensure_dir(save_dir / "multi_species_genomes")

    ds = load_dataset(
        "InstaDeepAI/multi_species_genomes",
        cache_dir=str(cache_dir),
    )
    if not isinstance(ds, dict):
        ds = {"train": ds}

    split_name_map = {
        "validation": "dev",
        "valid": "dev",
    }
    saved = {}
    for split_name, split_ds in ds.items():
        target_name = split_name_map.get(split_name, split_name)
        output_path = dataset_dir / f"{target_name}.txt"
        text_column = "sequence" if "sequence" in split_ds.column_names else "text"
        with open(output_path, "w") as f:
            for row in split_ds:
                sequence = str(row[text_column]).strip().upper()
                if sequence:
                    f.write(sequence + "\n")
        saved[target_name] = len(split_ds)
        print(f"  Wrote {len(split_ds)} sequences to: {output_path}")

    if "train" not in saved:
        raise RuntimeError(
            "Expected a train split for multi_species_genomes, but none was found."
        )

    print(f"  Training data ready at: {dataset_dir}")
    return dataset_dir


def download_genomic_benchmark(save_dir: Path):
    """Download Genomic Benchmark (8 tasks) in full-sequence format."""
    print("=" * 60)
    print("Downloading Genomic Benchmark (8 tasks, full sequences)...")
    print("Source: genomic-benchmarks Python package")
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

    try:
        from genomic_benchmarks.loc2seq import download_dataset
    except ImportError:
        print("  FAILED: `genomic_benchmarks` is not installed.")
        print("  Install it with `pip install genomic-benchmarks` or")
        print("  `pip install -r requirements.txt`, then rerun this command.")
        return

    ensure_dir(save_dir)
    for task in tasks:
        print(f"\n  Downloading: {task}...")
        try:
            out_dir = download_dataset(task, version=0)
            print(f"    Saved to: {out_dir}")
        except Exception as exc:
            print(f"    FAILED: {exc}")


def download_nt_benchmark(save_dir: Path):
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

    ensure_dir(save_dir)
    for task in tasks:
        print(f"\n  Downloading: {task}...")
        try:
            ds = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks",
                task,
                cache_dir=str(save_dir),
            )
            for split in ds:
                print(f"    {split}: {len(ds[split])} samples")
        except Exception as exc:
            print(f"    FAILED: {exc}")


def download_nt_benchmark_revised(save_dir: Path):
    """Download revised NT Benchmark."""
    from datasets import load_dataset

    print("=" * 60)
    print("Downloading NT Benchmark Revised...")
    print("Source: InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    print("=" * 60)

    ensure_dir(save_dir)
    ds = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
        cache_dir=str(save_dir),
    )
    print("  Downloaded successfully")
    for split in ds:
        print(f"    {split}: {len(ds[split])} samples")


def download_lrb(data_root: Path, refs_root: Path, force: bool = False):
    """Download long-range benchmark metadata and genomes."""
    print("=" * 60)
    print("Downloading LRB resources...")
    print("=" * 60)

    ref_repo = clone_repo(
        "https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark",
        refs_root / "genomics-long-range-benchmark",
        force=force,
    )
    maybe_git_lfs_pull(ref_repo)

    raw_dir = ensure_dir(data_root / "raw")
    extract_gzip(
        download_url(H38_REFERENCE_GENOME_URL, raw_dir / "hg38.fa.gz", force=force),
        raw_dir / "hg38.fa",
        force=force,
    )
    extract_gzip(
        download_url(H19_REFERENCE_GENOME_URL, raw_dir / "hg19.fa.gz", force=force),
        raw_dir / "hg19.fa",
        force=force,
    )

    # Download required CSV files directly from the HF dataset repo. This avoids
    # depending on local git-lfs checkout state.
    download_url(
        f"{LRB_HF_BASE}/variant_effect_causal_eqtl/All_Tissues.csv",
        data_root / "variant_effect_causal_eqtl" / "All_Tissues.csv",
        force=force,
    )
    download_url(
        f"{LRB_HF_BASE}/bulk_rna_expression/gene_coordinates.csv",
        data_root / "bulk_rna_expression" / "gene_coordinates.csv",
        force=force,
    )
    download_url(
        f"{LRB_HF_BASE}/bulk_rna_expression/rna_expression_values.csv",
        data_root / "bulk_rna_expression" / "rna_expression_values.csv",
        force=force,
    )
    download_url(
        f"{LRB_HF_BASE}/bulk_rna_expression/label_mapping.csv",
        data_root / "bulk_rna_expression" / "label_mapping.csv",
        force=force,
    )


def download_spliceai(data_root: Path, refs_root: Path, force: bool = False):
    """Download SpliceAI reference repo, genome, and GENCODE annotation."""
    print("=" * 60)
    print("Downloading SpliceAI resources...")
    print("=" * 60)

    clone_repo(
        "https://github.com/Illumina/SpliceAI.git",
        refs_root / "SpliceAI",
        force=force,
    )

    raw_dir = ensure_dir(data_root / "raw")
    extract_gzip(
        download_url(H38_REFERENCE_GENOME_URL, raw_dir / "hg38.fa.gz", force=force),
        raw_dir / "hg38.fa",
        force=force,
    )
    extract_gzip(
        download_url(
            GENCODE_V24_GTF_URL,
            raw_dir / "gencode.v24.annotation.gtf.gz",
            force=force,
        ),
        raw_dir / "gencode.v24.annotation.gtf",
        force=force,
    )


def _extract_selected_zip_members(zip_path: Path, dest_dir: Path, suffixes: list[str], force: bool = False):
    ensure_dir(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not any(member.endswith(suffix) for suffix in suffixes):
                continue
            out_path = dest_dir / Path(member).name
            if out_path.exists() and not force:
                print(f"  Skipping existing extracted file: {out_path}")
                continue
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  Extracted: {out_path}")


def download_protein_fitness(data_root: Path, refs_root: Path, force: bool = False):
    """Download ProteinGym reference repo, two assay CSVs, and wild-type CDS files."""
    print("=" * 60)
    print("Downloading Protein Fitness resources...")
    print("=" * 60)

    clone_repo(
        "https://github.com/OATML-Markslab/ProteinGym.git",
        refs_root / "ProteinGym",
        force=force,
    )

    assays_dir = ensure_dir(data_root / "DMS_assays" / "substitutions")
    zip_path = download_url(
        f"{PROTEINGYM_BASE}/{PROTEINGYM_SUBS_ZIP}",
        data_root / PROTEINGYM_SUBS_ZIP,
        force=force,
    )
    _extract_selected_zip_members(
        zip_path,
        assays_dir,
        ["BLAT_ECOLX_Firnberg_2014.csv", "BRCA1_HUMAN_Findlay_2018.csv"],
        force=force,
    )

    wt_dir = ensure_dir(data_root / "wildtype_cds")
    for assay_id, url in PROTEIN_CDS_SOURCES.items():
        download_url(url, wt_dir / f"{assay_id}.fasta", force=force)


def main():
    parser = argparse.ArgumentParser(description="Download MergeDNA datasets")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT), help="Root directory for data")
    parser.add_argument(
        "--refs_root",
        type=str,
        default=str(DATA_ROOT / "external_refs"),
        help="Root directory for cloned reference repos",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "all",
            "all_experiments",
            "references",
            "pretrain",
            "genomic_benchmark",
            "nt_benchmark",
            "nt_revised",
            "lrb",
            "spliceai",
            "protein_fitness",
        ],
        help="Which dataset bundle to download",
    )
    parser.add_argument("--force", action="store_true", help="Redownload or reclone existing resources")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    refs_root = Path(args.refs_root).expanduser()

    if args.dataset in ("all", "genomic_benchmark", "all_experiments"):
        download_genomic_benchmark(data_root / "genomic_benchmark_sequences")

    if args.dataset in ("all", "nt_benchmark", "all_experiments"):
        download_nt_benchmark(data_root / "nt_benchmark")

    if args.dataset in ("all", "pretrain", "all_experiments"):
        download_multi_species_genomes(data_root / "pretrain")

    if args.dataset == "nt_revised":
        download_nt_benchmark_revised(data_root / "nt_benchmark_revised")

    if args.dataset in ("all_experiments", "references"):
        clone_repo(
            "https://github.com/Illumina/SpliceAI.git",
            refs_root / "SpliceAI",
            force=args.force,
        )
        clone_repo(
            "https://github.com/OATML-Markslab/ProteinGym.git",
            refs_root / "ProteinGym",
            force=args.force,
        )
        clone_repo(
            "https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark",
            refs_root / "genomics-long-range-benchmark",
            force=args.force,
        )

    if args.dataset in ("lrb", "all_experiments"):
        download_lrb(data_root / "lrb", refs_root, force=args.force)

    if args.dataset in ("spliceai", "all_experiments"):
        download_spliceai(data_root / "spliceai", refs_root, force=args.force)

    if args.dataset in ("protein_fitness", "all_experiments"):
        download_protein_fitness(data_root / "protein_fitness", refs_root, force=args.force)

    print("\n" + "=" * 60)
    print("GUE Benchmark (24 tasks) - MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("Download from Google Drive:")
    print("  https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view?usp=sharing")
    print(f"Extract to: {data_root / 'gue_benchmark'}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
