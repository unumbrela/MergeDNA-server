"""Knowledge distillation runner for EfficientMergeDNA.

Distils a pretrained MergeDNA teacher (380M) into a lightweight student
(~50M) via three complementary signals:
1. Merge pattern distillation (source matrix alignment)
2. Latent representation distillation (projected feature matching)
3. Output distillation (soft-label KD)

The student simultaneously learns from its own pre-training objectives
(MTR + latent MTR + AMTM) and from the teacher's knowledge.

Usage:
    python train.py --config configs/efficient/distill.yaml --mode distill
"""

import os
import math
import time
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast

from ..model.mergedna import MergeDNA, MergeDNAConfig, EfficientMergeDNAConfig
from ..data.dataset import MultiSpeciesGenomeDataset
from ..data.tokenizer import DNACharTokenizer
from ..data.collator import PretrainCollator
from .distill_losses import (
    MergePatternDistillLoss,
    LatentRepresentationDistillLoss,
    OutputDistillLoss,
)

logger = logging.getLogger(__name__)


def _config_from_dict(d: Dict[str, Any], prefix: str = "") -> MergeDNAConfig:
    """Build a MergeDNAConfig from a flat dict, optionally with a key prefix."""
    def _get(key, default=None):
        return d.get(f"{prefix}{key}", d.get(key, default))

    return MergeDNAConfig(
        vocab_size=_get("vocab_size", 10),
        embed_dim=_get("embed_dim", 1024),
        num_heads=_get("num_heads", 16),
        local_encoder_layers=_get("local_encoder_layers", 4),
        latent_encoder_layers=_get("latent_encoder_layers", 20),
        latent_decoder_layers=_get("latent_decoder_layers", 4),
        local_decoder_layers=_get("local_decoder_layers", 2),
        window_size=_get("window_size", 16),
        dropout=_get("dropout", 0.0),
        use_flash_attn=_get("use_flash_attn", True),
        max_seq_length=_get("max_seq_length", 4096),
        compression_target=_get("compression_target", 0.5),
        compression_variance=_get("compression_variance", 0.1),
        lambda_latent=_get("lambda_latent", 0.25),
        K_ratio=_get("K_ratio", 0.5),
        gradient_checkpointing=_get("gradient_checkpointing", False),
        use_mtr=_get("use_mtr", True),
        use_latent_mtr=_get("use_latent_mtr", True),
        use_amtm=_get("use_amtm", True),
        amtm_masking_strategy=_get("amtm_masking_strategy", "adaptive"),
        random_mask_ratio=_get("random_mask_ratio", 0.15),
        use_entropy_guided_merging=_get("use_entropy_guided_merging", False),
        entropy_weight=_get("entropy_weight", 0.5),
        entropy_model_hidden_dim=_get("entropy_model_hidden_dim", 128),
        entropy_model_kernel_size=_get("entropy_model_kernel_size", 9),
        entropy_aux_loss_weight=_get("entropy_aux_loss_weight", 0.1),
        use_learned_compression=_get("use_learned_compression", False),
        r_min_per_window=_get("r_min_per_window", 1),
        r_max_per_window=_get("r_max_per_window", 8),
        compression_loss_weight=_get("compression_loss_weight", 0.1),
        latent_encoder_type=_get("latent_encoder_type", "transformer"),
        ssm_type=_get("ssm_type", "gated_deltanet"),
        attention_layer_indices=_get("attention_layer_indices", [5, 11, 17]),
    )


class DistillRunner:
    """Knowledge distillation runner for EfficientMergeDNA."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.local_rank = config.get("local_rank", 0)
        self.world_size = config.get("world_size", 1)
        self.distributed = self.world_size > 1

        # Training hyperparameters
        self.max_steps = config.get("max_steps", 50000)
        self.batch_size = config.get("batch_size", 16)
        self.gradient_accumulation = config.get("gradient_accumulation", 2)
        self.learning_rate = config.get("learning_rate", 2e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 2000)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_amp = config.get("use_amp", True)
        self.log_interval = config.get("log_interval", 100)
        self.save_interval = config.get("save_interval", 5000)
        self.output_dir = config.get("output_dir", "./outputs/efficient_distill")

        # Distillation weights
        self.distill_task_weight = config.get("distill_task_weight", 1.0)
        self.distill_merge_weight = config.get("distill_merge_weight", 1.0)
        self.distill_latent_weight = config.get("distill_latent_weight", 2.0)
        self.distill_output_weight = config.get("distill_output_weight", 1.0)
        self.distill_temperature = config.get("distill_temperature", 4.0)

        # Build components
        self._build_teacher()
        self._build_student()
        self._build_distill_losses()
        self._build_data()
        self._build_optimizer()

    def _build_teacher(self):
        """Build and freeze the teacher model.

        Supports two modes:
        1. Internal MergeDNA teacher: teacher_type="mergedna" (default)
           - Loads from teacher_ckpt checkpoint file
           - Full merge pattern + latent + output distillation
        2. External HuggingFace teacher: teacher_type="external"
           - Loads from teacher_model_name (e.g. "zhihan1996/DNABERT-2-117M")
           - Only latent representation + output distillation (no merge patterns)
        """
        teacher_type = self.config.get("teacher_type", "mergedna")
        self.teacher_type = teacher_type

        if teacher_type == "external":
            from .external_teacher import ExternalTeacherWrapper
            model_name = self.config["teacher_model_name"]
            self.teacher = ExternalTeacherWrapper(
                model_name=model_name,
                device=str(self.device),
                max_length=self.config.get("max_seq_length", 4096),
            ).to(self.device)
            self.teacher_hidden_dim = self.teacher.get_hidden_dim()
            # External teachers don't have merge patterns
            self.distill_merge_weight = 0.0
            logger.info(
                f"External teacher: {model_name} "
                f"(hidden_dim={self.teacher_hidden_dim}, merge distill disabled)"
            )
        else:
            # Internal MergeDNA teacher
            teacher_config = _config_from_dict(self.config, prefix="teacher_")
            self.teacher = MergeDNA(teacher_config).to(self.device)

            ckpt_path = self.config["teacher_ckpt"]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Teacher checkpoint not found: {ckpt_path}. "
                    "Please pretrain MergeDNA first or provide a valid path."
                )
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.teacher.load_state_dict(state_dict, strict=True)
            self.teacher_hidden_dim = self.config.get("teacher_embed_dim", 1024)

            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

            n_params = self.teacher.get_num_params()
            logger.info(f"MergeDNA teacher: {n_params / 1e6:.1f}M parameters (frozen)")

    def _build_student(self):
        """Build the student model from config."""
        student_config = _config_from_dict(self.config)
        self.student = MergeDNA(student_config).to(self.device)

        n_params = self.student.get_num_params()
        logger.info(f"Student: {n_params / 1e6:.1f}M parameters (trainable)")

        if self.distributed:
            self.student = nn.parallel.DistributedDataParallel(
                self.student,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )

    def _build_distill_losses(self):
        """Build distillation loss modules."""
        student_dim = self.config.get("embed_dim", 512)
        teacher_dim = self.teacher_hidden_dim

        self.merge_distill = MergePatternDistillLoss(
            temperature=1.0
        ).to(self.device)

        self.latent_distill = LatentRepresentationDistillLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
        ).to(self.device)

        self.output_distill = OutputDistillLoss(
            temperature=self.distill_temperature,
        ).to(self.device)

    def _build_data(self):
        """Build dataset and dataloader."""
        tokenizer = DNACharTokenizer(
            max_length=self.config.get("max_seq_length", 4096)
        )
        dataset = MultiSpeciesGenomeDataset(
            data_path=self.config["data_path"],
            tokenizer=tokenizer,
            max_length=self.config.get("max_seq_length", 4096),
            split="train",
            max_samples=self.config.get("max_samples", 0),
        )

        sampler = None
        if self.distributed:
            sampler = DistributedSampler(dataset, shuffle=True)

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.get("num_workers", 4),
            collate_fn=PretrainCollator(pad_token_id=0),
            pin_memory=True,
            drop_last=True,
        )
        self.data_iter = iter(self.dataloader)

    def _build_optimizer(self):
        """Build optimizer for student + distillation projection layers."""
        all_params = []

        # Student parameters
        decay_params, no_decay_params = [], []
        student_model = self.student.module if self.distributed else self.student
        for name, param in student_model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        all_params.append({"params": decay_params, "weight_decay": self.weight_decay})
        all_params.append({"params": no_decay_params, "weight_decay": 0.0})

        # Distillation projection layer parameters (latent_distill.proj)
        all_params.append({
            "params": list(self.latent_distill.parameters()),
            "weight_decay": 0.0,
            "lr": self.learning_rate,
        })

        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Resume support
        self.start_step = 0
        self._auto_resume()

    def _auto_resume(self):
        """Auto-detect and resume from latest checkpoint."""
        if not os.path.isdir(self.output_dir):
            return
        ckpts = [
            f for f in os.listdir(self.output_dir)
            if f.startswith("checkpoint-") and f.endswith(".pt")
        ]
        if not ckpts:
            return
        latest = max(ckpts, key=lambda x: int(x.split("-")[1].split(".")[0]))
        ckpt_path = os.path.join(self.output_dir, latest)
        logger.info(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        student_model = self.student.module if self.distributed else self.student
        student_model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "latent_distill_state_dict" in ckpt:
            self.latent_distill.load_state_dict(ckpt["latent_distill_state_dict"])
        self.start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from step {self.start_step}")

    def _get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with warmup."""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        return self.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch, cycling through the dataloader."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return {k: v.to(self.device) for k, v in batch.items()}

    def _save_checkpoint(self, step: int):
        """Save student checkpoint."""
        if self.local_rank != 0:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}.pt")
        student_model = self.student.module if self.distributed else self.student
        torch.save(
            {
                "step": step,
                "model_state_dict": student_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "latent_distill_state_dict": self.latent_distill.state_dict(),
                "config": self.config,
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")

    @torch.no_grad()
    def _teacher_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run teacher forward pass (frozen, no grad).

        For external teachers, adapts the output format to match what
        _compute_distill_loss expects.
        """
        if self.teacher_type == "external":
            out = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            # Wrap into the same format as MergeDNA intermediates
            return {
                "z_L_prime": out["hidden_states"],  # [B, N, D_teacher]
                "logits": out.get("logits"),  # [B, N, V] or None
                "source": None,  # external teachers don't have source matrices
                "mask_L": batch["attention_mask"],  # use input mask
            }
        else:
            return self.teacher.forward_with_intermediates(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

    def _student_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Run student forward: intermediates + pretrain losses."""
        student_model = self.student.module if self.distributed else self.student

        # Get intermediates
        intermediates = student_model.forward_with_intermediates(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # Also get pretrain losses (student learns its own objectives too)
        pretrain_losses = student_model.forward_pretrain(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        intermediates["pretrain_losses"] = pretrain_losses
        return intermediates

    def _compute_distill_loss(
        self,
        student_out: Dict[str, torch.Tensor],
        teacher_out: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all distillation losses.

        Gracefully handles external teachers that may not provide source
        matrices or logits.
        """
        losses = {}

        # 1. Merge pattern distillation (only for MergeDNA teachers)
        if (
            self.distill_merge_weight > 0
            and teacher_out.get("source") is not None
        ):
            losses["distill_merge"] = self.merge_distill(
                student_source=student_out["source"],
                teacher_source=teacher_out["source"],
                attention_mask=attention_mask,
            )

        # 2. Latent representation distillation
        if self.distill_latent_weight > 0:
            losses["distill_latent"] = self.latent_distill(
                student_z=student_out["z_L_prime"],
                teacher_z=teacher_out["z_L_prime"],
                student_mask=student_out["mask_L"],
                teacher_mask=teacher_out.get("mask_L"),
            )

        # 3. Output distillation (only if teacher provides logits)
        if (
            self.distill_output_weight > 0
            and teacher_out.get("logits") is not None
        ):
            losses["distill_output"] = self.output_distill(
                student_logits=student_out["logits"],
                teacher_logits=teacher_out["logits"],
                attention_mask=attention_mask,
            )

        return losses

    def train(self):
        """Main distillation training loop."""
        self.student.train()
        self.teacher.eval()

        global_step = self.start_step
        # Accumulators for logging
        acc = {
            "total": 0.0, "task": 0.0,
            "d_merge": 0.0, "d_latent": 0.0, "d_output": 0.0,
        }
        start_time = time.time()

        logger.info(f"Starting distillation for {self.max_steps} steps (from {global_step})")
        logger.info(
            f"Batch: {self.batch_size} x {self.gradient_accumulation} accumulation"
        )
        logger.info(
            f"Distill weights: task={self.distill_task_weight}, "
            f"merge={self.distill_merge_weight}, "
            f"latent={self.distill_latent_weight}, "
            f"output={self.distill_output_weight}"
        )

        while global_step < self.max_steps:
            self.optimizer.zero_grad()

            for _ in range(self.gradient_accumulation):
                batch = self._get_batch()

                with autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.use_amp,
                ):
                    # Teacher forward (no grad, bf16)
                    teacher_out = self._teacher_forward(batch)

                    # Student forward (with grad)
                    student_out = self._student_forward(batch)

                    # Distillation losses
                    distill_losses = self._compute_distill_loss(
                        student_out, teacher_out, batch["attention_mask"]
                    )

                    # Combine losses
                    pretrain_loss = student_out["pretrain_losses"]["loss"]
                    total_loss = self.distill_task_weight * pretrain_loss

                    if "distill_merge" in distill_losses:
                        total_loss = total_loss + self.distill_merge_weight * distill_losses["distill_merge"]
                    if "distill_latent" in distill_losses:
                        total_loss = total_loss + self.distill_latent_weight * distill_losses["distill_latent"]
                    if "distill_output" in distill_losses:
                        total_loss = total_loss + self.distill_output_weight * distill_losses["distill_output"]

                    scaled_loss = total_loss / self.gradient_accumulation

                self.scaler.scale(scaled_loss).backward()

                # Accumulate for logging
                ga = self.gradient_accumulation
                acc["total"] += total_loss.item() / ga
                acc["task"] += pretrain_loss.item() / ga
                acc["d_merge"] += distill_losses.get(
                    "distill_merge", torch.tensor(0.0)
                ).item() / ga
                acc["d_latent"] += distill_losses.get(
                    "distill_latent", torch.tensor(0.0)
                ).item() / ga
                acc["d_output"] += distill_losses.get(
                    "distill_output", torch.tensor(0.0)
                ).item() / ga

            # LR schedule
            lr = self._get_lr(global_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient clipping and step
            self.scaler.unscale_(self.optimizer)
            all_params = list(self.student.parameters()) + list(
                self.latent_distill.parameters()
            )
            torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            global_step += 1

            # Logging
            if global_step % self.log_interval == 0 and self.local_rank == 0:
                elapsed = time.time() - start_time
                n = self.log_interval
                logger.info(
                    f"Step {global_step}/{self.max_steps} | "
                    f"Loss: {acc['total']/n:.4f} "
                    f"(task: {acc['task']/n:.4f}, "
                    f"merge: {acc['d_merge']/n:.4f}, "
                    f"latent: {acc['d_latent']/n:.4f}, "
                    f"output: {acc['d_output']/n:.4f}) | "
                    f"LR: {lr:.6f} | Time: {elapsed:.1f}s"
                )
                acc = {k: 0.0 for k in acc}
                start_time = time.time()

            # Save checkpoint
            if global_step % self.save_interval == 0:
                self._save_checkpoint(global_step)

        self._save_checkpoint(global_step)
        logger.info("Distillation training complete!")
