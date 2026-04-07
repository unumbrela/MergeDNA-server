"""Dataset classes for MergeDNA pre-training and fine-tuning."""

import os
import csv
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List

import torch
from torch.utils.data import Dataset

from .tokenizer import DNACharTokenizer

logger = logging.getLogger(__name__)


class MultiSpeciesGenomeDataset(Dataset):
    """Pre-training dataset for Multi-Species Genomes corpus.

    Supports two modes:
    1. Fixed-record mode: For DNABERT-2 pretrain data where every line is
       exactly the same length (1000bp). Uses O(1) file seeking, zero RAM.
    2. In-memory mode: For small files or variable-length sequences.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: DNACharTokenizer,
        max_length: int = 1024,
        split: str = "train",
        max_samples: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.max_samples = max_samples

        # Will be set by _load_data
        self._file_path = None
        self._record_len = 0  # bytes per line (including newline)
        self._num_lines = 0
        self._sequences: Optional[List[str]] = None  # fallback in-memory
        self._file_handle = None  # lazy-opened in fixed-record mode per worker process

        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Detect data format and load accordingly."""
        # Find the text file
        if os.path.isdir(data_path):
            txt_file = os.path.join(data_path, f"{self.split}.txt")
            if not os.path.exists(txt_file):
                # Try any .txt file
                for f in sorted(os.listdir(data_path)):
                    if f.endswith((".txt", ".fa", ".fasta")):
                        txt_file = os.path.join(data_path, f)
                        break
            if os.path.exists(txt_file):
                data_path = txt_file
            else:
                raise FileNotFoundError(f"No text files found in {data_path}")

        if os.path.isfile(data_path):
            # Check if fixed-record format (all lines same length)
            with open(data_path, "rb") as f:
                first_line = f.readline()
                second_line = f.readline()
            if len(first_line) == len(second_line) and len(first_line) > 100:
                # Fixed-record mode
                self._file_path = data_path
                self._record_len = len(first_line)
                file_size = os.path.getsize(data_path)
                self._num_lines = file_size // self._record_len
                seq_len = self._record_len - 1  # minus newline
                if self.max_samples > 0:
                    self._num_lines = min(self._num_lines, self.max_samples)
                logger.info(
                    f"Fixed-record dataset: {self._num_lines} sequences, "
                    f"{seq_len}bp each, O(1) random access"
                )
                return

            # Variable-length: load into memory
            self._load_into_memory(data_path)
        else:
            # Try HuggingFace
            try:
                from datasets import load_dataset
                ds = load_dataset(data_path, split=self.split)
                col = "sequence" if "sequence" in ds.column_names else "text"
                self._sequences = ds[col]
            except Exception as e:
                raise ValueError(f"Cannot load data from {data_path}: {e}")

    def _load_into_memory(self, fpath: str):
        """Fallback: load all sequences into memory."""
        self._sequences = []
        with open(fpath, "r") as f:
            current_seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_seq:
                        self._sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    self._sequences.append(line.upper()) if not current_seq else current_seq.append(line.upper())
            if current_seq:
                self._sequences.append("".join(current_seq))
        if self.max_samples > 0:
            self._sequences = self._sequences[:self.max_samples]
        logger.info(f"In-memory dataset: {len(self._sequences)} sequences")

    def __len__(self) -> int:
        if self._file_path:
            return self._num_lines
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._file_path:
            # Fixed-record: O(1) seek. Keep one handle per worker process to
            # avoid paying open/close overhead on every sample.
            offset = idx * self._record_len
            if self._file_handle is None:
                self._file_handle = open(self._file_path, "rb")
            self._file_handle.seek(offset)
            seq = self._file_handle.read(self._record_len).decode("ascii").strip()
        else:
            seq = self._sequences[idx]

        # Random crop if longer than max_length
        if len(seq) > self.max_length:
            start = random.randint(0, len(seq) - self.max_length)
            seq = seq[start : start + self.max_length]

        encoded = self.tokenizer(
            [seq],
            max_length=self.max_length,
            padding=False,
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __del__(self):
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass


class GUEBenchmarkDataset(Dataset):
    """GUE Benchmark dataset (DNABERT-2 format, CSV files).

    Supports all 24+ GUE tasks organized as:
        GUE/{category}/{task}/train.csv, dev.csv, test.csv
    """

    def __init__(
        self,
        task_path: str,
        tokenizer: DNACharTokenizer,
        split: str = "train",
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        self.labels = []

        csv_file = os.path.join(task_path, f"{split}.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"GUE CSV not found: {csv_file}")

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sequences.append(row["sequence"])
                self.labels.append(int(row["label"]))

        logger.info(f"GUE {os.path.basename(task_path)} [{split}]: {len(self.sequences)} samples, {self.num_classes} classes")

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            [seq],
            max_length=self.max_length,
            padding=False,
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class GenomicBenchmarkDataset(Dataset):
    """Genomic Benchmark dataset (8 tasks).

    Supports two valid sources:
    1. Local full-sequence directories produced by the `genomic-benchmarks`
       package: `<root>/<task>/<split>/<class_name>/*.txt`
    2. HuggingFace datasets that already expose sequence and label columns.

    Some mirrors only expose the raw coordinate CSV files from
    `katielink/genomic-benchmarks`. Those files are not training-ready for this
    project and are rejected with a clear error.
    """

    TASK_NAMES = [
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
        "demo_coding_vs_intergenomic_seqs",
        "demo_human_or_worm",
        "dummy_mouse_enhancers_ensembl",
        "human_ensembl_regulatory",
        "human_nontata_promoters",
        "human_ocr_ensembl",
    ]

    def __init__(
        self,
        task_name: str,
        tokenizer: DNACharTokenizer,
        split: str = "train",
        max_length: int = 1024,
        data_path: Optional[str] = None,
    ):
        assert task_name in self.TASK_NAMES, (
            f"Unknown task: {task_name}. Available: {self.TASK_NAMES}"
        )
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.sequences = []
        self.labels = []

        self._load_data(data_path)

    def _load_data(self, data_path: Optional[str]):
        """Load from local full-sequence data or a compatible HF dataset."""
        task_dir = self._find_local_task_dir(data_path)
        if task_dir is not None:
            self._load_local(task_dir)
            return

        errors = []
        for dataset_name in (
            "InstaDeepAI/genomic-benchmarks",
            "katielink/genomic-benchmarks",
        ):
            try:
                self._load_huggingface(dataset_name)
                return
            except Exception as e:
                errors.append(f"{dataset_name}: {e}")

        details = "; ".join(errors) if errors else "no usable data source found"
        raise ValueError(
            f"Cannot load genomic benchmark {self.task_name}: {details}. "
            "Expected full-sequence data with sequence/label columns, or a "
            "local directory like `<root>/<task>/<split>/<class>/*.txt`. "
            "The `katielink/genomic-benchmarks` mirror currently resolves to "
            "raw coordinate CSVs rather than training-ready sequences. "
            "Install `genomic-benchmarks` and run "
            "`python scripts/download_data.py --dataset genomic_benchmark`, "
            "then rerun with `--genomic_benchmark_data_dir ~/.genomic_benchmarks` "
            "if needed."
        )

    def _candidate_roots(self, data_path: Optional[str]) -> List[Path]:
        roots = []
        for raw_path in (
            data_path,
            os.getenv("GENOMIC_BENCHMARK_DATA_DIR"),
            str(Path(__file__).resolve().parents[2] / "data" / "genomic_benchmark"),
            str(Path.home() / ".genomic_benchmarks"),
        ):
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if path not in roots:
                roots.append(path)
        return roots

    def _find_local_task_dir(self, data_path: Optional[str]) -> Optional[Path]:
        for root in self._candidate_roots(data_path):
            for candidate in (root, root / self.task_name):
                split_dir = candidate / self.split
                if not split_dir.is_dir():
                    continue
                if any(path.is_dir() for path in split_dir.iterdir()):
                    return candidate
        return None

    def _load_local(self, task_dir: Path):
        split_dir = task_dir / self.split
        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if not class_dirs:
            raise ValueError(f"No class directories found in {split_dir}")

        class_to_label = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}
        for class_dir in class_dirs:
            label = class_to_label[class_dir.name]
            for seq_file in sorted(class_dir.iterdir()):
                if not seq_file.is_file():
                    continue
                seq = self._read_sequence_file(seq_file)
                if not seq:
                    continue
                self.sequences.append(seq)
                self.labels.append(label)

        if not self.sequences:
            raise ValueError(f"No sequences found in {split_dir}")

        logger.info(
            "Genomic Benchmark %s [%s]: %d samples, %d classes (local: %s)",
            self.task_name,
            self.split,
            len(self.sequences),
            self.num_classes,
            task_dir,
        )

    def _load_huggingface(self, dataset_name: str):
        from datasets import load_dataset

        ds = load_dataset(
            dataset_name,
            self.task_name,
            split="train" if self.split == "train" else "test",
        )

        seq_col = next(
            (col for col in ("sequence", "seq", "text") if col in ds.column_names),
            None,
        )
        label_col = next(
            (col for col in ("label", "labels", "target") if col in ds.column_names),
            None,
        )
        if seq_col is None or label_col is None:
            raise ValueError(
                f"unsupported columns {list(ds.column_names)}"
            )

        self.sequences = [str(seq).upper() for seq in ds[seq_col]]
        self.labels = [int(label) for label in ds[label_col]]
        logger.info(
            "Genomic Benchmark %s [%s]: %d samples, %d classes (HF: %s)",
            self.task_name,
            self.split,
            len(self.sequences),
            self.num_classes,
            dataset_name,
        )

    @staticmethod
    def _read_sequence_file(path: Path) -> str:
        chunks = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(">"):
                    continue
                chunks.append(line)
        return "".join(chunks).upper()

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            [seq],
            max_length=self.max_length,
            padding=False,
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class NTBenchmarkDataset(Dataset):
    """Nucleotide Transformer benchmark dataset (18 tasks)."""

    def __init__(
        self,
        task_name: str,
        tokenizer: DNACharTokenizer,
        split: str = "train",
        max_length: int = 1024,
        data_path: Optional[str] = None,
    ):
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.sequences = []
        self.labels = []

        self._load_data(data_path)

    def _load_data(self, data_path: Optional[str]):
        """Load NT benchmark data from HuggingFace."""
        local_errors = []
        for root in self._candidate_roots(data_path):
            try:
                self._load_local(root)
                return
            except Exception as local_e:
                local_errors.append(f"{root}: {local_e}")

        try:
            from datasets import load_dataset
            ds = load_dataset(
                "InstaDeepAI/nucleotide_transformer_downstream_tasks",
                self.task_name,
                split="train" if self.split == "train" else "test",
            )
            self.sequences = ds["sequence"]
            self.labels = ds["label"]
        except Exception as e:
            detail = "; ".join(local_errors) if local_errors else "no local cache found"
            raise ValueError(
                f"Cannot load NT benchmark {self.task_name}: HF={e}; local={detail}"
            )

    def _candidate_roots(self, data_path: Optional[str]) -> List[Path]:
        roots = []
        for raw_path in (
            data_path,
            os.getenv("NT_BENCHMARK_DATA_DIR"),
            str(Path(__file__).resolve().parents[2] / "data" / "nt_benchmark"),
        ):
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if path not in roots:
                roots.append(path)
        return roots

    def _load_local(self, data_path: Path):
        """Load from local CSV files or the aggregated HF arrow cache."""
        import pandas as pd
        from datasets import Dataset

        csv_path = data_path / self.task_name / f"{self.split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.sequences = df["sequence"].tolist()
            self.labels = df["label"].tolist()
            return

        arrow_split = "train" if self.split == "train" else "test"
        arrow_paths = sorted(
            data_path.glob(
                f"InstaDeepAI___nucleotide_transformer_downstream_tasks/**/"
                f"nucleotide_transformer_downstream_tasks-{arrow_split}.arrow"
            )
        )
        if not arrow_paths:
            raise FileNotFoundError("aggregated NT arrow cache not found")

        ds = Dataset.from_file(str(arrow_paths[0]))
        ds = ds.filter(lambda x: x["task"] == self.task_name)
        if len(ds) == 0:
            raise ValueError(f"task {self.task_name} not found in {arrow_paths[0]}")

        self.sequences = ds["sequence"]
        self.labels = ds["label"]

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            [seq],
            max_length=self.max_length,
            padding=False,
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
