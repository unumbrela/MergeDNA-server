"""Protein fitness zero-shot scoring utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import (
    LongSequenceEmbedder,
    Timer,
    apply_aa_mutations_to_cds,
    load_result,
    save_result,
)

logger = logging.getLogger(__name__)


DEFAULT_PROTEIN_ASSAYS = {
    "bacteria": "BLAT_ECOLX_Firnberg_2014",
    "human": "BRCA1_HUMAN_Findlay_2018",
}


def resolve_protein_resources(config: dict) -> Dict[str, Path]:
    data_dir = Path(
        config.get("protein_fitness_data_dir", "./data/protein_fitness")
    ).expanduser()
    refs_dir = Path(
        config.get("external_proteingym_ref_dir", "./data/external_refs/ProteinGym")
    ).expanduser()
    return {
        "assays_dir": data_dir / "DMS_assays" / "substitutions",
        "wildtype_dir": data_dir / "wildtype_cds",
        "reference_csv": refs_dir / "reference_files" / "DMS_substitutions.csv",
    }


def check_protein_fitness_prerequisites(config: dict) -> Dict[str, bool]:
    resources = resolve_protein_resources(config)
    assay_map = get_protein_assay_map(config)
    status = {
        "reference_csv": resources["reference_csv"].exists(),
        "assays_dir": resources["assays_dir"].exists(),
        "wildtype_dir": resources["wildtype_dir"].exists(),
    }
    for alias, assay in assay_map.items():
        status[f"assay_csv:{alias}"] = (
            resources["assays_dir"] / f"{assay}.csv"
        ).exists()
        status[f"wildtype_cds:{alias}"] = (
            resources["wildtype_dir"] / f"{assay}.fasta"
        ).exists() or (
            resources["wildtype_dir"] / f"{assay}.fa"
        ).exists() or (
            resources["wildtype_dir"] / f"{assay}.txt"
        ).exists()
    return status


def get_protein_assay_map(config: dict) -> Dict[str, str]:
    assay_map = dict(DEFAULT_PROTEIN_ASSAYS)
    assay_map.update(config.get("protein_fitness_assays", {}))
    return assay_map


def _load_reference_row(reference_csv: Path, assay_id: str) -> pd.Series:
    df = pd.read_csv(reference_csv)
    matched = df[df["DMS_id"] == assay_id]
    if matched.empty:
        raise ValueError(f"ProteinGym assay not found in reference CSV: {assay_id}")
    return matched.iloc[0]


STANDARD_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records = []
    header = None
    chunks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and chunks:
                    records.append((header, "".join(chunks).upper()))
                header = line[1:]
                chunks = []
                continue
            chunks.append(line)
    if header is not None and chunks:
        records.append((header, "".join(chunks).upper()))
    return records


def _translate_cds(sequence: str) -> str:
    amino_acids = []
    for start in range(0, len(sequence) - 2, 3):
        amino_acids.append(STANDARD_CODON_TABLE.get(sequence[start : start + 3], "X"))
    if amino_acids and amino_acids[-1] == "*":
        amino_acids.pop()
    return "".join(amino_acids)


def _read_fasta_sequence(path: Path, reference_row: pd.Series) -> str:
    records = _read_fasta_records(path)
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    if len(records) == 1:
        return records[0][1]

    target_seq = str(reference_row.get("target_seq", "")).strip().upper()
    seq_len = reference_row.get("seq_len")
    candidates = records

    if pd.notna(seq_len):
        desired_len = int(seq_len)
        length_filtered = [
            record
            for record in records
            if len(record[1]) in {desired_len * 3, (desired_len + 1) * 3}
        ]
        if len(length_filtered) == 1:
            return length_filtered[0][1]
        if length_filtered:
            candidates = length_filtered

    if target_seq:
        translated_matches = [
            record for record in candidates if _translate_cds(record[1]) == target_seq
        ]
        if len(translated_matches) == 1:
            return translated_matches[0][1]

    raise ValueError(
        f"Ambiguous multi-record FASTA for {path}; expected a single matching CDS record"
    )


def _find_wildtype_cds(resources: Dict[str, Path], assay_id: str) -> Path:
    for suffix in (".fasta", ".fa", ".txt"):
        path = resources["wildtype_dir"] / f"{assay_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing wild-type CDS for {assay_id} under {resources['wildtype_dir']}"
    )


def _select_mutants(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if "mutant" not in df.columns or "DMS_score" not in df.columns:
        raise ValueError("Protein fitness CSV must contain mutant and DMS_score columns")
    max_mutations = int(config.get("protein_fitness_max_mutations", 2))
    df = df.copy()
    df["num_mutations"] = df["mutant"].astype(str).apply(lambda value: value.count(":") + 1)
    df = df[df["num_mutations"] <= max_mutations]
    limit = config.get("protein_fitness_max_samples")
    if limit and limit > 0 and len(df) > limit:
        df = df.sample(n=limit, random_state=42).sort_index()
    return df


def _score_dataframe(
    df: pd.DataFrame,
    wt_cds: str,
    embedder: LongSequenceEmbedder,
) -> np.ndarray:
    scores = []
    for _, row in df.iterrows():
        if "mutated_dna_sequence" in row and isinstance(row["mutated_dna_sequence"], str):
            mutated_cds = row["mutated_dna_sequence"].upper()
            mutated_positions = [
                idx for idx, (a, b) in enumerate(zip(wt_cds, mutated_cds)) if a != b
            ]
        else:
            mutated_cds, mutated_positions = apply_aa_mutations_to_cds(
                wt_cds, str(row["mutant"])
            )
        log_probs = embedder.masked_base_log_probs(mutated_cds, mutated_positions)
        scores.append(float(log_probs.sum()))
    return np.asarray(scores, dtype=np.float32)


def run_protein_fitness_task(config: dict, output_dir: str, alias: str) -> dict:
    assay_map = get_protein_assay_map(config)
    if alias not in assay_map:
        raise ValueError(f"unknown protein fitness alias: {alias}")

    assay_id = assay_map[alias]
    output_dir = Path(output_dir)
    result_path = output_dir / "results.json"
    existing = load_result(result_path)
    if existing and config.get("skip_existing", False):
        logger.info("Skipping existing protein fitness result: %s", result_path)
        return existing

    resources = resolve_protein_resources(config)
    reference_row = _load_reference_row(resources["reference_csv"], assay_id)
    assay_csv = resources["assays_dir"] / f"{assay_id}.csv"
    if not assay_csv.exists():
        raise FileNotFoundError(f"Missing ProteinGym assay CSV: {assay_csv}")
    wt_cds_path = _find_wildtype_cds(resources, assay_id)
    wt_cds = _read_fasta_sequence(wt_cds_path, reference_row)

    df = pd.read_csv(assay_csv)
    df = _select_mutants(df, config)
    embedder = LongSequenceEmbedder(config)

    with Timer() as timer:
        model_scores = _score_dataframe(df, wt_cds, embedder)
        corr = spearmanr(model_scores, df["DMS_score"].to_numpy(dtype=np.float32)).statistic
        metrics = {
            "srcc": float(corr),
            "num_mutants": int(len(df)),
            "assay_id": assay_id,
            "approximate_codon_mapping": "mutated_dna_sequence"
            not in df.columns,
            "selection_assay": str(reference_row.get("selection_assay", "")),
        }

    return save_result(
        result_path,
        f"protein_fitness/{alias}",
        metrics,
        timer.started_at,
        timer.finished_at,
        extra={
            "assay_id": assay_id,
            "wildtype_cds_path": str(wt_cds_path),
        },
    )
