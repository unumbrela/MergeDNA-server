# MergeDNA Server Pretrain Setup

This guide is for running the exact command below on a server:

```bash
python train.py --config configs/pretrain_a800.yaml --mode pretrain
```

## 1. What Must Be Present On The Server

Required:

- Repository root with the code, configs, and scripts.
- `data/pretrain/multi_species_genomes/train.txt`
- Optional but recommended: `data/pretrain/multi_species_genomes/dev.txt`

Not required for this pretrain command:

- `outputs/`
- `logs/`
- `data/genomic_benchmark_sequences/`
- `data/nt_benchmark/`
- `data/gue_benchmark/`
- `data/lrb/`
- `data/spliceai/`
- `data/protein_fitness/`
- `reference_repos/`

## 2. Environment Setup

Create a clean Python environment first. Python 3.12 is recommended.

```bash
conda create -n mergedna python=3.12 -y
conda activate mergedna
```

Install a CUDA-enabled PyTorch build that matches the server's CUDA runtime.
Do this before installing the Python requirements in this repo.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

Optional but recommended on A800:

```bash
pip install flash-attn --no-build-isolation
```

The code will still run without `flash-attn`, but it will be slower.

## 3. Data Setup

You have two practical options.

### Option A: Copy The Existing Pretrain Data From Local

From your local machine, sync only the required pretraining corpus:

```bash
rsync -avP data/pretrain/multi_species_genomes/ <server>:/path/to/MergeDNA/data/pretrain/multi_species_genomes/
```

### Option B: Download On The Server

After cloning the repo and setting up the environment:

```bash
python scripts/download_data.py --dataset pretrain --data_root ./data
```

This will materialize the dataset under:

```text
data/pretrain/multi_species_genomes/
```

## 4. Sanity Check

Before launching the full run, confirm the required files exist:

```bash
ls data/pretrain/multi_species_genomes
python -c "import yaml, einops; import train; from mergedna.training.pretrain import PretrainRunner; print('pretrain_import_ok')"
```

## 5. Launch

Run from the repository root:

```bash
python train.py --config configs/pretrain_a800.yaml --mode pretrain
```

Important notes:

- The command must be run from the repo root because the config uses relative paths.
- `configs/pretrain_a800.yaml` writes checkpoints to `./outputs/pretrain_a800`.
- The current config expects a single A800-class GPU and uses:
  - `max_seq_length: 4096`
  - `batch_size: 8`
  - `gradient_accumulation: 4`
  - `use_flash_attn: true`
- If the server is a restricted container and dataloader workers fail to start,
  lower `num_workers` in `configs/pretrain_a800.yaml` from `8` to `0` or `1`.

## 6. Minimal Transfer Checklist

If you want the smallest server-side payload, transfer only:

- the GitHub repo contents
- `data/pretrain/multi_species_genomes/`

Everything else can stay local for this command.
