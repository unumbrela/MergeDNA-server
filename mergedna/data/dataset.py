"""Dataset classes for MergeDNA pre-training and fine-tuning."""

import os
import csv
import random
import logging
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
            # Fixed-record: O(1) seek
            offset = idx * self._record_len
            with open(self._file_path, "rb") as f:
                f.seek(offset)
                seq = f.read(self._record_len).decode("ascii").strip()
        else:
            seq = self._sequences[idx]

        # Random crop if longer than max_length
        if len(seq) > self.max_length:
            start = random.randint(0, len(seq) - self.max_length)
            seq = seq[start : start + self.max_length]

        encoded = self.tokenizer(
            [seq],
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


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
            padding=True,
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class GenomicBenchmarkDataset(Dataset):
    """Genomic Benchmark dataset (8 tasks) from HuggingFace."""

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
        """Load from HuggingFace or local cache."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "katielink/genomic-benchmarks",
                self.task_name,
                split="train" if self.split == "train" else "test",
            )
            self.sequences = ds["seq"]
            self.labels = ds["label"]
        except Exception as e:
            raise ValueError(
                f"Cannot load genomic benchmark {self.task_name}: {e}"
            )

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
            padding=True,
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
            if data_path and os.path.isdir(data_path):
                self._load_local(data_path)
            else:
                raise ValueError(f"Cannot load NT benchmark {self.task_name}: {e}")

    def _load_local(self, data_path: str):
        """Load from local CSV files."""
        import pandas as pd
        fpath = os.path.join(data_path, self.task_name, f"{self.split}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            self.sequences = df["sequence"].tolist()
            self.labels = df["label"].tolist()

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
            padding=True,
            truncation=True,
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
