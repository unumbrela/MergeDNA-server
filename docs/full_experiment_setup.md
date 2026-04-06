# MergeDNA Full Experiment Setup

## What Is Added

- Unified downstream experiment runner:
  - `scripts/run_all_experiments.py`
  - `scripts/run_all_local.sh`
  - `scripts/run_all_a800.sh`
- New experiment runners:
  - `mergedna/experiments/spliceai.py`
  - `mergedna/experiments/lrb.py`
  - `mergedna/experiments/protein_fitness.py`
  - `mergedna/experiments/ablation.py`
- Extended data download entry:
  - `scripts/download_data.py`
- Existing GB / NT / GUE loops now support:
  - `--skip_existing`
  - per-task `results.json`
  - task-level runtime capture

## Data Layout

- Genomic Benchmark:
  - `data/genomic_benchmark_sequences/`
  - or upstream package cache `~/.genomic_benchmarks/`
- NT Benchmark:
  - `data/nt_benchmark/`
- GUE:
  - `data/gue_benchmark/GUE/`
- LRB:
  - `data/lrb/raw/hg19.fa`
  - `data/lrb/raw/hg38.fa`
  - `data/lrb/variant_effect_causal_eqtl/All_Tissues.csv`
  - `data/lrb/bulk_rna_expression/gene_coordinates.csv`
  - `data/lrb/bulk_rna_expression/rna_expression_values.csv`
- SpliceAI:
  - `data/spliceai/raw/hg38.fa`
  - `data/spliceai/raw/gencode.v24.annotation.gtf`
  - processed CSVs will be generated under `data/spliceai/processed/`
- Protein Fitness:
  - `data/protein_fitness/DMS_assays/substitutions/BLAT_ECOLX_Firnberg_2014.csv`
  - `data/protein_fitness/DMS_assays/substitutions/BRCA1_HUMAN_Findlay_2018.csv`
  - `data/protein_fitness/wildtype_cds/*.fasta`
- External references:
  - `data/external_refs/SpliceAI`
  - `data/external_refs/ProteinGym`
  - `data/external_refs/genomics-long-range-benchmark`

## Download Commands

Clone / download all non-GUE resources:

```bash
python scripts/download_data.py --dataset all_experiments --data_root ./data --refs_root ./data/external_refs
```

Only clone reference repos:

```bash
python scripts/download_data.py --dataset references --data_root ./data --refs_root ./data/external_refs
```

Only pull one missing bundle:

```bash
python scripts/download_data.py --dataset spliceai
python scripts/download_data.py --dataset lrb
python scripts/download_data.py --dataset protein_fitness
```

GUE still needs manual download:

```bash
bash scripts/download_gdrive.sh
```

## Dry Run And Prepare

Check whether all experiments are ready:

```bash
bash scripts/run_all_local.sh --groups all --dry-run
```

Only build derived data such as SpliceAI processed CSVs:

```bash
bash scripts/run_all_local.sh --groups spliceai --prepare-only
```

## Run Commands

Run all local downstream experiments and skip finished ones:

```bash
bash scripts/run_all_local.sh --groups all
```

Run a subset:

```bash
bash scripts/run_all_local.sh --groups gb,nt,gue
bash scripts/run_all_local.sh --groups spliceai,lrb,protein
bash scripts/run_all_local.sh --groups ablation
```

Run the A800 version:

```bash
bash scripts/run_all_a800.sh --groups all
```

## Result Files

- Group summary:
  - `outputs/.../all_experiments_summary.json`
- GB / NT / GUE task results:
  - `outputs/.../<group>/<task>/results.json`
- SpliceAI:
  - `outputs/.../spliceai/donor/results.json`
  - `outputs/.../spliceai/acceptor/results.json`
- LRB:
  - `outputs/.../lrb/causal_eqtl/results.json`
  - `outputs/.../lrb/bulk_rna/results.json`
- Protein Fitness:
  - `outputs/.../protein_fitness/bacteria/results.json`
  - `outputs/.../protein_fitness/human/results.json`
- Ablation:
  - `outputs/.../ablation/<variant>/results.json`

All of these include runtime fields, and the top-level summary also records total runtime.

## Important Assumptions

- SpliceAI is implemented as long-context frozen embedding extraction plus logistic regression over donor / acceptor binary tasks.
- The SpliceAI processed dataset is generated from `hg38 + GENCODE v24` using chromosome splits:
  - train: `chr1` to `chr19`
  - dev: `chr20`
  - test: `chr21`, `chr22`
- LRB follows the paper appendix more closely:
  - frozen embeddings
  - logistic regression for causal eQTL
  - ridge regression for bulk RNA
- Protein Fitness has one unavoidable approximation:
  - if the assay CSV does not provide exact mutated DNA sequences, the current code maps amino-acid mutations back to codons using a minimal-Hamming-distance codon choice relative to the wild-type CDS.
  - this makes the pipeline runnable, but it is not guaranteed to exactly match the paper’s codon-level zero-shot evaluation unless assay-specific nucleotide variants are available.
- Ablation runs require a pretraining config because each variant first produces its own pretrain checkpoint and then runs GB fine-tuning.

## Suggested Workflow

1. Download / clone all required resources.
2. Run `--dry-run` and confirm no missing prerequisites except optional ones you intentionally skipped.
3. Run `--prepare-only` for `spliceai` once.
4. Run `gb,nt,gue` first to verify the standard benchmark path.
5. Run `spliceai,lrb,protein`.
6. Run `ablation` last because it is the most expensive.
