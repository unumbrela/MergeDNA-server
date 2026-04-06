"""Ablation runners for Table 7-style experiments."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Dict

from mergedna.training.pretrain import PretrainRunner

from .common import Timer, load_result, save_result

logger = logging.getLogger(__name__)


ABLATION_VARIANTS = {
    "byte_mtm": {
        "local_encoder_layers": 0,
        "latent_encoder_layers": 24,
        "local_decoder_layers": 2,
        "use_mtr": False,
        "use_latent_mtr": False,
        "use_amtm": True,
        "amtm_masking_strategy": "random",
    },
    "local_mtr_mtm": {
        "local_encoder_layers": 4,
        "latent_encoder_layers": 20,
        "local_decoder_layers": 2,
        "use_mtr": True,
        "use_latent_mtr": False,
        "use_amtm": True,
        "amtm_masking_strategy": "random",
    },
    "local_full_lambda1": {
        "local_encoder_layers": 4,
        "latent_encoder_layers": 20,
        "local_decoder_layers": 2,
        "use_mtr": True,
        "use_latent_mtr": True,
        "use_amtm": True,
        "amtm_masking_strategy": "adaptive",
        "lambda_latent": 1.0,
    },
    "local_full_selected": {
        "local_encoder_layers": 4,
        "latent_encoder_layers": 20,
        "local_decoder_layers": 2,
        "use_mtr": True,
        "use_latent_mtr": True,
        "use_amtm": True,
        "amtm_masking_strategy": "adaptive",
        "lambda_latent": 0.25,
    },
    "local2_full_selected": {
        "local_encoder_layers": 2,
        "latent_encoder_layers": 20,
        "local_decoder_layers": 4,
        "use_mtr": True,
        "use_latent_mtr": True,
        "use_amtm": True,
        "amtm_masking_strategy": "adaptive",
        "lambda_latent": 0.25,
    },
}


def _latest_checkpoint(output_dir: str) -> str | None:
    path = Path(output_dir)
    if not path.exists():
        return None
    ckpts = [p for p in path.iterdir() if p.name.startswith("checkpoint-") and p.suffix == ".pt"]
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda p: int(p.stem.split("-")[1])))


def _mean_accuracy(results: Dict[str, dict]) -> float:
    values = [value.get("accuracy", 0.0) for value in results.values() if "error" not in value]
    return float(sum(values) / len(values)) if values else 0.0


def run_ablation_variant(base_config: dict, output_root: str, variant_name: str) -> dict:
    if variant_name not in ABLATION_VARIANTS:
        raise ValueError(f"unknown ablation variant: {variant_name}")

    from train import run_finetune_all_gb

    variant_dir = Path(output_root) / variant_name
    result_path = variant_dir / "results.json"
    existing = load_result(result_path)
    if existing and base_config.get("skip_existing", False):
        logger.info("Skipping existing ablation variant: %s", variant_name)
        return existing

    overrides = ABLATION_VARIANTS[variant_name]
    pretrain_config = copy.deepcopy(base_config)
    pretrain_config.update(overrides)
    pretrain_config["output_dir"] = str(variant_dir / "pretrain")
    pretrain_config["max_steps"] = int(
        base_config.get("ablation_pretrain_steps", base_config.get("max_steps", 0) or 0)
    )
    if pretrain_config["max_steps"] <= 0:
        raise ValueError("ablation_pretrain_steps or max_steps must be set for ablations")

    ckpt = _latest_checkpoint(pretrain_config["output_dir"])
    with Timer() as timer:
        if ckpt is None:
            PretrainRunner(pretrain_config).train()
            ckpt = _latest_checkpoint(pretrain_config["output_dir"])
        if ckpt is None:
            raise RuntimeError(f"failed to produce checkpoint for ablation {variant_name}")

        finetune_config = copy.deepcopy(base_config)
        finetune_config.update(overrides)
        finetune_config["pretrain_ckpt"] = ckpt
        finetune_config["output_dir"] = str(variant_dir / "finetune")
        finetune_config["skip_existing"] = base_config.get("skip_existing", False)
        gb_results = run_finetune_all_gb(finetune_config)
        avg_acc = _mean_accuracy(gb_results)
        metrics = {
            "avg_accuracy": avg_acc,
            "gb_results_path": os.path.join(
                finetune_config["output_dir"],
                "genomic_benchmark_results.json",
            ),
            "pretrain_ckpt": ckpt,
        }

    return save_result(
        result_path,
        f"ablation/{variant_name}",
        metrics,
        timer.started_at,
        timer.finished_at,
        extra={"variant_overrides": overrides},
    )
