#!/usr/bin/env python
"""Run MergeDNA downstream experiments with resume/skip support."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import (
    load_config,
    run_finetune_all_gb,
    run_finetune_all_gue,
    run_finetune_all_nt,
)
from mergedna.experiments.ablation import ABLATION_VARIANTS, run_ablation_variant
from mergedna.experiments.common import Timer
from mergedna.experiments.lrb import (
    check_lrb_prerequisites,
    run_lrb_bulk_rna,
    run_lrb_eqtl,
)
from mergedna.experiments.protein_fitness import (
    check_protein_fitness_prerequisites,
    get_protein_assay_map,
    run_protein_fitness_task,
)
from mergedna.experiments.spliceai import (
    check_spliceai_prerequisites,
    prepare_spliceai_dataset,
    run_spliceai_task,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MergeDNA-All")


ALL_GROUPS = ["gb", "nt", "gue", "spliceai", "lrb", "protein", "ablation"]


def _mean_metric(results: dict, key: str) -> float:
    values = [item.get(key, 0.0) for item in results.values() if "error" not in item]
    return float(sum(values) / len(values)) if values else 0.0


def _save_summary(output_dir: str, payload: dict) -> None:
    path = Path(output_dir) / "all_experiments_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved run summary to %s", path)


def _check_group_prereqs(group: str, config: dict) -> dict:
    if group == "spliceai":
        return check_spliceai_prerequisites(config)
    if group == "lrb":
        return check_lrb_prerequisites(config)
    if group == "protein":
        return check_protein_fitness_prerequisites(config)
    return {"ready": True}


def main():
    parser = argparse.ArgumentParser(description="Run all MergeDNA experiments")
    parser.add_argument("--config", required=True, help="Downstream config YAML")
    parser.add_argument(
        "--pretrain-config",
        default=None,
        help="Pretrain config YAML, required for ablation runs",
    )
    parser.add_argument(
        "--groups",
        default="all",
        help="Comma-separated subset of groups: gb,nt,gue,spliceai,lrb,protein,ablation",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    downstream_config = load_config(args.config)
    if args.output_dir:
        downstream_config["output_dir"] = args.output_dir
    downstream_config["skip_existing"] = args.skip_existing

    if args.groups == "all":
        groups = list(ALL_GROUPS)
    else:
        groups = [item.strip() for item in args.groups.split(",") if item.strip()]
        unknown = sorted(set(groups) - set(ALL_GROUPS))
        if unknown:
            raise ValueError(f"unknown groups: {unknown}")

    pretrain_config = None
    if "ablation" in groups:
        if not args.pretrain_config:
            raise ValueError("--pretrain-config is required when groups include ablation")
        pretrain_config = load_config(args.pretrain_config)
        if args.output_dir:
            pretrain_config["output_dir"] = args.output_dir
        pretrain_config["skip_existing"] = args.skip_existing

    if args.prepare_only or args.dry_run:
        payload = {"groups": groups, "checks": {}}
        for group in groups:
            payload["checks"][group] = _check_group_prereqs(group, downstream_config)
            if group == "spliceai":
                try:
                    prepared = prepare_spliceai_dataset(
                        downstream_config, force=args.force_prepare
                    )
                    payload["checks"][group]["prepared"] = prepared
                except Exception as exc:
                    payload["checks"][group]["prepare_error"] = str(exc)
        _save_summary(downstream_config.get("output_dir", "./outputs"), payload)
        return

    summary = {"groups": {}, "output_dir": downstream_config.get("output_dir", "./outputs")}

    for group in groups:
        logger.info("========== Running group: %s ==========", group)
        if group == "gb":
            with Timer() as timer:
                results = run_finetune_all_gb(downstream_config.copy())
            summary["groups"]["gb"] = {
                "average_accuracy": _mean_metric(results, "accuracy"),
                "results": results,
                "duration_seconds": timer.duration_seconds,
            }
        elif group == "nt":
            with Timer() as timer:
                results = run_finetune_all_nt(downstream_config.copy())
            summary["groups"]["nt"] = {
                "average_accuracy": _mean_metric(results, "accuracy"),
                "average_mcc": _mean_metric(results, "mcc"),
                "results": results,
                "duration_seconds": timer.duration_seconds,
            }
        elif group == "gue":
            with Timer() as timer:
                results = run_finetune_all_gue(downstream_config.copy())
            summary["groups"]["gue"] = {
                "average_accuracy": _mean_metric(results, "accuracy"),
                "average_mcc": _mean_metric(results, "mcc"),
                "results": results,
                "duration_seconds": timer.duration_seconds,
            }
        elif group == "spliceai":
            with Timer() as timer:
                prepare_spliceai_dataset(downstream_config, force=args.force_prepare)
                donor = run_spliceai_task(
                    downstream_config.copy(),
                    str(Path(summary["output_dir"]) / "spliceai" / "donor"),
                    "donor",
                )
                acceptor = run_spliceai_task(
                    downstream_config.copy(),
                    str(Path(summary["output_dir"]) / "spliceai" / "acceptor"),
                    "acceptor",
                )
            summary["groups"]["spliceai"] = {
                "donor": donor,
                "acceptor": acceptor,
                "mean_auroc": (
                    donor["metrics"]["auroc"] + acceptor["metrics"]["auroc"]
                ) / 2.0,
                "duration_seconds": timer.duration_seconds,
            }
        elif group == "lrb":
            with Timer() as timer:
                eqtl = run_lrb_eqtl(
                    downstream_config.copy(),
                    str(Path(summary["output_dir"]) / "lrb" / "causal_eqtl"),
                )
                bulk = run_lrb_bulk_rna(
                    downstream_config.copy(),
                    str(Path(summary["output_dir"]) / "lrb" / "bulk_rna"),
                )
            summary["groups"]["lrb"] = {
                "causal_eqtl": eqtl,
                "bulk_rna": bulk,
                "duration_seconds": timer.duration_seconds,
            }
        elif group == "protein":
            assays = get_protein_assay_map(downstream_config)
            group_results = {}
            with Timer() as timer:
                for alias in ("bacteria", "human"):
                    if alias not in assays:
                        continue
                    group_results[alias] = run_protein_fitness_task(
                        downstream_config.copy(),
                        str(Path(summary["output_dir"]) / "protein_fitness" / alias),
                        alias,
                    )
            group_results["duration_seconds"] = timer.duration_seconds
            summary["groups"]["protein"] = group_results
        elif group == "ablation":
            assert pretrain_config is not None
            ablation_base = downstream_config.copy()
            ablation_base.update(pretrain_config)
            ablation_base["skip_existing"] = args.skip_existing
            ablation_root = Path(summary["output_dir"]) / "ablation"
            ablation_results = {}
            with Timer() as timer:
                for variant_name in ABLATION_VARIANTS:
                    ablation_results[variant_name] = run_ablation_variant(
                        ablation_base,
                        str(ablation_root),
                        variant_name,
                    )
            byte_acc = ablation_results["byte_mtm"]["metrics"]["avg_accuracy"]
            for variant_name, result in ablation_results.items():
                result["metrics"]["delta_vs_byte"] = (
                    result["metrics"]["avg_accuracy"] - byte_acc
                )
            ablation_results["duration_seconds"] = timer.duration_seconds
            summary["groups"]["ablation"] = ablation_results

    summary["total_duration_seconds"] = float(
        sum(
            group_payload.get("duration_seconds", 0.0)
            for group_payload in summary["groups"].values()
            if isinstance(group_payload, dict)
        )
    )
    _save_summary(summary["output_dir"], summary)


if __name__ == "__main__":
    main()
