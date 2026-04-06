"""SpliceAI-style long-context splice site benchmark."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .common import LongSequenceEmbedder, Timer, load_result, save_json, save_result

logger = logging.getLogger(__name__)


TRAIN_CHROMS = {f"chr{i}" for i in range(1, 20)}
VAL_CHROMS = {"chr20"}
TEST_CHROMS = {"chr21", "chr22"}


def resolve_spliceai_resources(config: dict) -> Dict[str, Path]:
    data_dir = Path(config.get("spliceai_data_dir", "./data/spliceai")).expanduser()
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    refs_dir = Path(config.get("external_spliceai_ref_dir", "./data/external_refs/SpliceAI"))
    return {
        "data_dir": data_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "gencode_gtf": raw_dir / "gencode.v24.annotation.gtf",
        "hg38_fasta": raw_dir / "hg38.fa",
        "spliceai_repo": refs_dir,
    }


def check_spliceai_prerequisites(config: dict) -> Dict[str, bool]:
    resources = resolve_spliceai_resources(config)
    return {
        "gencode_gtf": resources["gencode_gtf"].exists(),
        "hg38_fasta": resources["hg38_fasta"].exists(),
    }


def _parse_attributes(value: str) -> Dict[str, str]:
    attrs = {}
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if " " not in chunk:
            continue
        key, raw = chunk.split(" ", 1)
        attrs[key] = raw.strip().strip('"')
    return attrs


def _iter_exons(gtf_path: Path) -> Iterable[Tuple[str, str, int, int, str]]:
    with open(gtf_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = row
            if feature != "exon":
                continue
            if chrom not in TRAIN_CHROMS | VAL_CHROMS | TEST_CHROMS:
                continue
            parsed = _parse_attributes(attrs)
            transcript_id = parsed.get("transcript_id")
            if not transcript_id:
                continue
            yield chrom, transcript_id, int(start), int(end), strand


def _reverse_complement(seq: str) -> str:
    table = str.maketrans("ATCGNatcgn", "TAGCNtagcn")
    return seq.translate(table)[::-1]


def _standardize(seq: str) -> str:
    return "".join(base if base in "ATCG" else "N" for base in seq.upper())


def _fetch_centered_window(
    genome: Fasta,
    chrom: str,
    center_1based: int,
    window_len: int,
    strand: str,
) -> str | None:
    start = center_1based - 1 - window_len // 2
    end = start + window_len
    if chrom not in genome:
        return None
    chromosome = genome[chrom]
    if start < 0 or end >= len(chromosome):
        return None
    seq = chromosome[start:end].seq
    if strand == "-":
        seq = _reverse_complement(seq)
    return _standardize(seq)


def prepare_spliceai_dataset(config: dict, force: bool = False) -> Dict[str, str]:
    resources = resolve_spliceai_resources(config)
    processed_dir = resources["processed_dir"]
    donor_train = processed_dir / "donor" / "train.csv"
    if donor_train.exists() and not force:
        return {
            "donor": str(processed_dir / "donor"),
            "acceptor": str(processed_dir / "acceptor"),
        }

    if not resources["gencode_gtf"].exists():
        raise FileNotFoundError(f"Missing GENCODE GTF: {resources['gencode_gtf']}")
    if not resources["hg38_fasta"].exists():
        raise FileNotFoundError(f"Missing hg38 FASTA: {resources['hg38_fasta']}")

    genome = Fasta(str(resources["hg38_fasta"]), one_based_attributes=False)
    window_len = int(config.get("spliceai_sequence_length", 10000))
    neg_per_pos = int(config.get("spliceai_negatives_per_positive", 1))
    rng = np.random.default_rng(int(config.get("spliceai_seed", 42)))

    transcripts = defaultdict(list)
    transcript_meta = {}
    for chrom, transcript_id, start, end, strand in _iter_exons(resources["gencode_gtf"]):
        transcripts[transcript_id].append((start, end))
        transcript_meta[transcript_id] = (chrom, strand)

    splits = {
        "train": TRAIN_CHROMS,
        "dev": VAL_CHROMS,
        "test": TEST_CHROMS,
    }
    site_rows = {
        "donor": {"train": [], "dev": [], "test": []},
        "acceptor": {"train": [], "dev": [], "test": []},
    }

    for transcript_id, exons in transcripts.items():
        chrom, strand = transcript_meta[transcript_id]
        if chrom in TRAIN_CHROMS:
            split = "train"
        elif chrom in VAL_CHROMS:
            split = "dev"
        elif chrom in TEST_CHROMS:
            split = "test"
        else:
            continue

        exons = sorted(exons, key=lambda item: item[0])
        if len(exons) < 2:
            continue

        positives = {"donor": [], "acceptor": []}
        if strand == "+":
            positives["donor"] = [end for _, end in exons[:-1]]
            positives["acceptor"] = [start for start, _ in exons[1:]]
        else:
            positives["donor"] = [start for start, _ in exons[1:]]
            positives["acceptor"] = [end for _, end in exons[:-1]]

        blocked = set()
        for positions in positives.values():
            for pos in positions:
                blocked.update(range(max(1, pos - 50), pos + 51))

        tx_start = exons[0][0]
        tx_end = exons[-1][1]
        candidate_negatives = [
            pos for pos in range(tx_start + 50, tx_end - 50)
            if pos not in blocked
        ]
        if not candidate_negatives:
            continue

        for site_type, positions in positives.items():
            negative_count = min(len(candidate_negatives), len(positions) * neg_per_pos)
            negatives = rng.choice(candidate_negatives, size=negative_count, replace=False)

            for label, sampled_positions in ((1, positions), (0, negatives.tolist())):
                for pos in sampled_positions:
                    sequence = _fetch_centered_window(
                        genome, chrom, int(pos), window_len, strand
                    )
                    if sequence is None:
                        continue
                    site_rows[site_type][split].append(
                        {
                            "sequence": sequence,
                            "label": int(label),
                            "chromosome": chrom,
                            "position": int(pos),
                            "strand": strand,
                            "transcript_id": transcript_id,
                        }
                    )

    for site_type in ("donor", "acceptor"):
        task_dir = processed_dir / site_type
        task_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in site_rows[site_type].items():
            df = pd.DataFrame(rows)
            df.to_csv(task_dir / f"{split}.csv", index=False)

    save_json(
        processed_dir / "metadata.json",
        {
            "window_length": window_len,
            "negatives_per_positive": neg_per_pos,
        },
    )
    return {
        "donor": str(processed_dir / "donor"),
        "acceptor": str(processed_dir / "acceptor"),
    }


def _load_split(task_dir: Path, split: str) -> pd.DataFrame:
    path = task_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing SpliceAI split file: {path}")
    return pd.read_csv(path)


def _build_features(
    df: pd.DataFrame,
    embedder: LongSequenceEmbedder,
    max_rows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).sort_index()
    features = []
    labels = []
    for _, row in df.iterrows():
        features.append(embedder.embed_sequence(str(row["sequence"])))
        labels.append(int(row["label"]))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def run_spliceai_task(config: dict, output_dir: str, site_type: str) -> dict:
    if site_type not in {"donor", "acceptor"}:
        raise ValueError(f"unsupported splice site type: {site_type}")

    output_dir = Path(output_dir)
    result_path = output_dir / "results.json"
    existing = load_result(result_path)
    if existing and config.get("skip_existing", False):
        logger.info("Skipping existing SpliceAI result: %s", result_path)
        return existing

    prepared = prepare_spliceai_dataset(config, force=False)
    task_dir = Path(prepared[site_type])
    train_df = _load_split(task_dir, "train")
    dev_df = _load_split(task_dir, "dev")
    test_df = _load_split(task_dir, "test")

    embedder = LongSequenceEmbedder(config)
    max_train = config.get("spliceai_max_train_samples")
    max_dev = config.get("spliceai_max_dev_samples")
    max_test = config.get("spliceai_max_test_samples")

    with Timer() as timer:
        X_train, y_train = _build_features(train_df, embedder, max_train)
        X_dev, y_dev = _build_features(dev_df, embedder, max_dev)
        X_test, y_test = _build_features(test_df, embedder, max_test)

        best_C = None
        best_auc = -1.0
        for C in config.get("spliceai_c_grid", [0.01, 0.1, 1.0, 10.0]):
            clf = LogisticRegression(
                C=float(C),
                max_iter=int(config.get("spliceai_max_iter", 2000)),
                class_weight="balanced",
                solver="lbfgs",
            )
            clf.fit(X_train, y_train)
            dev_probs = clf.predict_proba(X_dev)[:, 1]
            auc = float(roc_auc_score(y_dev, dev_probs))
            if auc > best_auc:
                best_auc = auc
                best_C = float(C)

        final_clf = LogisticRegression(
            C=best_C,
            max_iter=int(config.get("spliceai_max_iter", 2000)),
            class_weight="balanced",
            solver="lbfgs",
        )
        final_clf.fit(
            np.concatenate([X_train, X_dev], axis=0),
            np.concatenate([y_train, y_dev], axis=0),
        )
        test_probs = final_clf.predict_proba(X_test)[:, 1]
        metrics = {
            "auroc": float(roc_auc_score(y_test, test_probs)),
            "val_auroc": float(best_auc),
            "best_C": float(best_C),
            "num_train": int(len(y_train)),
            "num_dev": int(len(y_dev)),
            "num_test": int(len(y_test)),
            "feature_dim": int(X_train.shape[1]) if len(X_train) else 0,
        }

    return save_result(
        result_path,
        f"spliceai/{site_type}",
        metrics,
        timer.started_at,
        timer.finished_at,
    )
