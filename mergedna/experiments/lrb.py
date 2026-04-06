"""Long-range benchmark runners."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

from .common import LongSequenceEmbedder, Timer, load_result, save_json, save_result

logger = logging.getLogger(__name__)


def _candidate_paths(*paths: str | Path) -> List[Path]:
    out = []
    for path in paths:
        if path:
            out.append(Path(path).expanduser())
    return out


def resolve_lrb_resources(config: dict) -> Dict[str, Path]:
    lrb_data_dir = Path(config.get("lrb_data_dir", "./data/lrb")).expanduser()
    ref_dir = Path(
        config.get(
            "external_lrb_ref_dir",
            "./data/external_refs/genomics-long-range-benchmark",
        )
    ).expanduser()

    def pick(*candidates: Path) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    return {
        "eqtl_csv": pick(
            lrb_data_dir / "variant_effect_causal_eqtl" / "All_Tissues.csv",
            ref_dir / "variant_effect_causal_eqtl" / "All_Tissues.csv",
        ),
        "bulk_coordinates_csv": pick(
            lrb_data_dir / "bulk_rna_expression" / "gene_coordinates.csv",
            ref_dir / "bulk_rna_expression" / "gene_coordinates.csv",
        ),
        "bulk_labels_csv": pick(
            lrb_data_dir / "bulk_rna_expression" / "rna_expression_values.csv",
            ref_dir / "bulk_rna_expression" / "rna_expression_values.csv",
        ),
        "bulk_label_mapping_csv": pick(
            lrb_data_dir / "bulk_rna_expression" / "label_mapping.csv",
            ref_dir / "bulk_rna_expression" / "label_mapping.csv",
        ),
        # pyfaidx does not support plain gzip FASTA files; require extracted .fa files.
        "hg38_fasta": lrb_data_dir / "raw" / "hg38.fa",
        "hg19_fasta": lrb_data_dir / "raw" / "hg19.fa",
    }


def check_lrb_prerequisites(config: dict) -> Dict[str, bool]:
    paths = resolve_lrb_resources(config)
    status = {}
    for name, path in paths.items():
        if name.endswith("_fasta"):
            ok = path.exists() and path.stat().st_size > 0
            if ok:
                try:
                    with open(path, "r") as handle:
                        ok = handle.readline().startswith(">")
                except OSError:
                    ok = False
            status[name] = ok
        else:
            status[name] = path.exists()
    return status


def _standardize_sequence(sequence: str) -> str:
    return "".join(base if base in "ATCG" else "N" for base in sequence.upper())


def _fetch_centered_sequence(
    genome: Fasta,
    chrom: str,
    center_pos_1based: int,
    length: int,
    negative_strand: bool = False,
) -> str | None:
    start = center_pos_1based - 1 - length // 2
    end = start + length
    if chrom not in genome:
        return None
    chromosome = genome[chrom]
    if start < 0 or end >= len(chromosome):
        return None
    seq = chromosome[start:end].seq
    if negative_strand:
        seq = chromosome[start:end].reverse.complement.seq
    return _standardize_sequence(seq)


def _load_bulk_labels(labels_csv: Path) -> np.ndarray:
    df = pd.read_csv(labels_csv)
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] == 0:
        raise ValueError(f"no numeric expression columns found in {labels_csv}")
    return numeric_df.to_numpy(dtype=np.float32)


def _load_eqtl_dataframe(eqtl_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(eqtl_csv)
    required = {
        "split", "CHROM", "POS", "ALT", "label", "tissue", "distance_to_nearest_TSS",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing eqtl columns {sorted(missing)} in {eqtl_csv}")
    return df


def _limit_rows(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit and limit > 0 and len(df) > limit:
        return df.sample(n=limit, random_state=42).sort_index()
    return df


def _build_eqtl_features(
    df: pd.DataFrame,
    genome: Fasta,
    embedder: LongSequenceEmbedder,
    seq_len: int,
    max_rows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    df = _limit_rows(df, max_rows)
    tissues = sorted(df["tissue"].astype(str).unique().tolist())
    tissue_to_idx = {name: idx for idx, name in enumerate(tissues)}

    features = []
    labels = []
    for _, row in df.iterrows():
        chrom = str(row["CHROM"])
        chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
        pos = int(row["POS"])
        ref_seq = _fetch_centered_sequence(genome, chrom, pos, seq_len)
        if ref_seq is None:
            continue
        alt_seq = list(ref_seq)
        alt_seq[seq_len // 2] = str(row["ALT"]).upper()
        alt_seq = _standardize_sequence("".join(alt_seq))

        ref_emb = embedder.embed_sequence(ref_seq)
        alt_emb = embedder.embed_sequence(alt_seq)
        tissue_one_hot = np.zeros(len(tissues), dtype=np.float32)
        tissue_one_hot[tissue_to_idx[str(row["tissue"])]] = 1.0
        distance = np.array(
            [float(row["distance_to_nearest_TSS"]) / max(1.0, seq_len)],
            dtype=np.float32,
        )
        feature = np.concatenate(
            [ref_emb, alt_emb, alt_emb - ref_emb, tissue_one_hot, distance]
        )
        features.append(feature)
        labels.append(int(row["label"]))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _build_bulk_features(
    coordinates_df: pd.DataFrame,
    labels: np.ndarray,
    genome: Fasta,
    embedder: LongSequenceEmbedder,
    seq_len: int,
    max_rows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(labels) != len(coordinates_df):
        raise ValueError("bulk RNA coordinates and labels row counts do not match")
    indices = list(coordinates_df.index)
    if max_rows and max_rows > 0 and len(indices) > max_rows:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(indices, size=max_rows, replace=False).tolist())

    features = []
    targets = []
    for idx in indices:
        row = coordinates_df.loc[idx]
        chrom = str(row["chrom"])
        pos = int(row["CAGE_representative_TSS"])
        seq = _fetch_centered_sequence(
            genome,
            chrom,
            pos,
            seq_len,
            negative_strand=str(row["strand"]) == "-",
        )
        if seq is None:
            continue
        features.append(embedder.embed_sequence(seq))
        targets.append(labels[idx])
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def run_lrb_eqtl(config: dict, output_dir: str) -> dict:
    output_dir = Path(output_dir)
    result_path = output_dir / "results.json"
    existing = load_result(result_path)
    if existing and config.get("skip_existing", False):
        logger.info("Skipping existing LRB causal eQTL result: %s", result_path)
        return existing

    resources = resolve_lrb_resources(config)
    if not resources["eqtl_csv"].exists():
        raise FileNotFoundError(f"Missing LRB eQTL CSV: {resources['eqtl_csv']}")
    if not resources["hg38_fasta"].exists():
        raise FileNotFoundError(f"Missing hg38 FASTA: {resources['hg38_fasta']}")

    ensure_payload = {"resources": {k: str(v) for k, v in resources.items()}}
    save_json(output_dir / "metadata.json", ensure_payload)

    df = _load_eqtl_dataframe(resources["eqtl_csv"])
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    genome = Fasta(str(resources["hg38_fasta"]), one_based_attributes=False)
    embedder = LongSequenceEmbedder(config)
    seq_len = int(config.get("lrb_eqtl_sequence_length", 20000))
    max_train = config.get("lrb_eqtl_max_train_samples")
    max_test = config.get("lrb_eqtl_max_test_samples")

    with Timer() as timer:
        X_train, y_train = _build_eqtl_features(train_df, genome, embedder, seq_len, max_train)
        X_test, y_test = _build_eqtl_features(test_df, genome, embedder, seq_len, max_test)

        clf = LogisticRegression(
            max_iter=int(config.get("lrb_eqtl_max_iter", 2000)),
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=None,
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        metrics = {
            "auroc": float(roc_auc_score(y_test, probs)),
            "num_train": int(len(y_train)),
            "num_test": int(len(y_test)),
            "feature_dim": int(X_train.shape[1]) if len(X_train) else 0,
        }

    return save_result(
        result_path,
        "lrb/causal_eqtl",
        metrics,
        timer.started_at,
        timer.finished_at,
    )


def run_lrb_bulk_rna(config: dict, output_dir: str) -> dict:
    output_dir = Path(output_dir)
    result_path = output_dir / "results.json"
    existing = load_result(result_path)
    if existing and config.get("skip_existing", False):
        logger.info("Skipping existing LRB bulk RNA result: %s", result_path)
        return existing

    resources = resolve_lrb_resources(config)
    if not resources["bulk_coordinates_csv"].exists():
        raise FileNotFoundError(
            f"Missing LRB bulk RNA coordinates CSV: {resources['bulk_coordinates_csv']}"
        )
    if not resources["bulk_labels_csv"].exists():
        raise FileNotFoundError(
            f"Missing LRB bulk RNA labels CSV: {resources['bulk_labels_csv']}"
        )
    if not resources["hg19_fasta"].exists():
        raise FileNotFoundError(f"Missing hg19 FASTA: {resources['hg19_fasta']}")

    coords = pd.read_csv(resources["bulk_coordinates_csv"])
    labels = _load_bulk_labels(resources["bulk_labels_csv"])
    train_coords = coords[coords["split"] == "train"].copy()
    test_coords = coords[coords["split"] == "test"].copy()
    train_labels = labels[train_coords.index]
    test_labels = labels[test_coords.index]
    genome = Fasta(str(resources["hg19_fasta"]), one_based_attributes=False)
    embedder = LongSequenceEmbedder(config)
    seq_len = int(config.get("lrb_bulk_rna_sequence_length", 40000))
    max_train = config.get("lrb_bulk_rna_max_train_samples")
    max_test = config.get("lrb_bulk_rna_max_test_samples")

    with Timer() as timer:
        X_train, y_train = _build_bulk_features(
            train_coords, train_labels, genome, embedder, seq_len, max_train
        )
        X_test, y_test = _build_bulk_features(
            test_coords, test_labels, genome, embedder, seq_len, max_test
        )
        reg = Ridge(alpha=float(config.get("lrb_bulk_rna_alpha", 1.0)))
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, preds, multioutput="uniform_average")),
            "num_train": int(len(y_train)),
            "num_test": int(len(y_test)),
            "num_outputs": int(y_train.shape[1]) if len(y_train) else 0,
            "feature_dim": int(X_train.shape[1]) if len(X_train) else 0,
        }

    return save_result(
        result_path,
        "lrb/bulk_rna",
        metrics,
        timer.started_at,
        timer.finished_at,
    )
