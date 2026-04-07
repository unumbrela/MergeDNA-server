"""Pre-training runner for MergeDNA.

Implements the pre-training pipeline:
- Multi-Species Genomes corpus
- AdamW optimizer with cosine schedule
- 100K iterations, max sequence length 4096
- Compression ratio sampling
- Distributed training support via PyTorch DDP
- Gradient accumulation, mixed precision, checkpointing
"""

import os
import math
import time
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast

from ..model.mergedna import MergeDNA, MergeDNAConfig
from ..data.dataset import MultiSpeciesGenomeDataset
from ..data.tokenizer import DNACharTokenizer
from ..data.collator import PretrainCollator

logger = logging.getLogger(__name__)


class PretrainRunner:
    """Pre-training runner for MergeDNA."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.local_rank = config.get("local_rank", 0)
        self.world_size = config.get("world_size", 1)
        self.distributed = self.world_size > 1

        # Training hyperparameters
        self.max_steps = config.get("max_steps", 100000)
        self.batch_size = config.get("batch_size", 8)
        self.gradient_accumulation = config.get("gradient_accumulation", 4)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 2000)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_amp = config.get("use_amp", True)
        self.log_interval = config.get("log_interval", 100)
        self.save_interval = config.get("save_interval", 5000)
        self.output_dir = config.get("output_dir", "./outputs/pretrain")

        # Build components
        self._build_model()
        self._build_data()
        self._build_optimizer()

    def _build_model(self):
        """Build MergeDNA model."""
        model_config = MergeDNAConfig(
            vocab_size=self.config.get("vocab_size", 10),
            embed_dim=self.config.get("embed_dim", 1024),
            num_heads=self.config.get("num_heads", 16),
            local_encoder_layers=self.config.get("local_encoder_layers", 4),
            latent_encoder_layers=self.config.get("latent_encoder_layers", 20),
            latent_decoder_layers=self.config.get("latent_decoder_layers", 4),
            local_decoder_layers=self.config.get("local_decoder_layers", 2),
            window_size=self.config.get("window_size", 16),
            dropout=self.config.get("dropout", 0.0),
            use_flash_attn=self.config.get("use_flash_attn", True),
            max_seq_length=self.config.get("max_seq_length", 4096),
            lambda_latent=self.config.get("lambda_latent", 0.25),
            gradient_checkpointing=self.config.get("gradient_checkpointing", False),
            use_mtr=self.config.get("use_mtr", True),
            use_latent_mtr=self.config.get("use_latent_mtr", True),
            use_amtm=self.config.get("use_amtm", True),
            amtm_masking_strategy=self.config.get("amtm_masking_strategy", "adaptive"),
            random_mask_ratio=self.config.get("random_mask_ratio", 0.15),
            # MergeDNA-Long extensions
            use_entropy_guided_merging=self.config.get("use_entropy_guided_merging", False),
            entropy_weight=self.config.get("entropy_weight", 0.5),
            entropy_model_hidden_dim=self.config.get("entropy_model_hidden_dim", 128),
            entropy_model_kernel_size=self.config.get("entropy_model_kernel_size", 9),
            entropy_aux_loss_weight=self.config.get("entropy_aux_loss_weight", 0.1),
            use_learned_compression=self.config.get("use_learned_compression", False),
            r_min_per_window=self.config.get("r_min_per_window", 1),
            r_max_per_window=self.config.get("r_max_per_window", 8),
            compression_loss_weight=self.config.get("compression_loss_weight", 0.1),
            latent_encoder_type=self.config.get("latent_encoder_type", "transformer"),
            ssm_type=self.config.get("ssm_type", "gated_deltanet"),
            attention_layer_indices=self.config.get("attention_layer_indices", [5, 11, 17]),
        )
        self.model = MergeDNA(model_config).to(self.device)

        n_params = self.model.get_num_params()
        logger.info(f"MergeDNA model: {n_params / 1e6:.1f}M parameters")
        if model_config.gradient_checkpointing:
            logger.info("Gradient checkpointing enabled")

        # torch.compile for speedup (PyTorch 2.0+)
        if self.config.get("compile", False):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )

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
        """Build optimizer and scheduler."""
        # Separate weight decay params
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.95),
        )

        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Resume from checkpoint if available
        self.start_step = 0
        resume_ckpt = self.config.get("resume_from")
        if resume_ckpt and os.path.exists(resume_ckpt):
            self._load_checkpoint(resume_ckpt)
        else:
            # Auto-detect latest checkpoint in output_dir
            self._auto_resume()

    def _load_checkpoint(self, ckpt_path: str):
        """Load checkpoint for resumption."""
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_to_load = self.model.module if self.distributed else self.model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from step {self.start_step}")

    def _auto_resume(self):
        """Auto-detect latest checkpoint in output_dir."""
        if not os.path.isdir(self.output_dir):
            return
        ckpts = [f for f in os.listdir(self.output_dir) if f.startswith("checkpoint-") and f.endswith(".pt")]
        if not ckpts:
            return
        # Find latest by step number
        latest = max(ckpts, key=lambda x: int(x.split("-")[1].split(".")[0]))
        self._load_checkpoint(os.path.join(self.output_dir, latest))

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
        """Save model checkpoint."""
        if self.local_rank != 0:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        ckpt_path = os.path.join(self.output_dir, f"checkpoint-{step}.pt")

        model_to_save = (
            self.model.module if self.distributed else self.model
        )
        torch.save(
            {
                "step": step,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")

    def train(self):
        """Main training loop."""
        self.model.train()
        global_step = self.start_step
        total_loss = 0.0
        loss_mtr_sum = 0.0
        loss_latent_sum = 0.0
        loss_amtm_sum = 0.0
        start_time = time.time()

        logger.info(f"Starting pre-training for {self.max_steps} steps (from step {global_step})")
        logger.info(f"Batch size: {self.batch_size} x {self.gradient_accumulation} accumulation")
        logger.info(f"Device: {self.device}")

        while global_step < self.max_steps:
            self.optimizer.zero_grad()

            for accum_step in range(self.gradient_accumulation):
                batch = self._get_batch()

                with autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.use_amp,
                ):
                    model_fn = (
                        self.model.module if self.distributed else self.model
                    )
                    losses = model_fn.forward_pretrain(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = losses["loss"] / self.gradient_accumulation

                self.scaler.scale(loss).backward()

                total_loss += losses["loss"].item() / self.gradient_accumulation
                loss_mtr_sum += losses.get("loss_mtr", torch.tensor(0.0)).item() / self.gradient_accumulation
                loss_latent_sum += losses.get("loss_latent_mtr", torch.tensor(0.0)).item() / self.gradient_accumulation
                loss_amtm_sum += losses.get("loss_amtm", torch.tensor(0.0)).item() / self.gradient_accumulation

            # Update LR
            lr = self._get_lr(global_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient clipping and step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            global_step += 1

            # Logging
            if global_step % self.log_interval == 0 and self.local_rank == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / self.log_interval
                avg_mtr = loss_mtr_sum / self.log_interval
                avg_latent = loss_latent_sum / self.log_interval
                avg_amtm = loss_amtm_sum / self.log_interval

                logger.info(
                    f"Step {global_step}/{self.max_steps} | "
                    f"Loss: {avg_loss:.4f} (MTR: {avg_mtr:.4f}, "
                    f"Latent: {avg_latent:.4f}, AMTM: {avg_amtm:.4f}) | "
                    f"LR: {lr:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

                # Log to wandb if available
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "loss": avg_loss,
                            "loss_mtr": avg_mtr,
                            "loss_latent_mtr": avg_latent,
                            "loss_amtm": avg_amtm,
                            "lr": lr,
                            "step": global_step,
                        })
                except ImportError:
                    pass

                total_loss = 0.0
                loss_mtr_sum = 0.0
                loss_latent_sum = 0.0
                loss_amtm_sum = 0.0
                start_time = time.time()

            # Save checkpoint
            if global_step % self.save_interval == 0:
                self._save_checkpoint(global_step)

        # Final save
        self._save_checkpoint(global_step)
        logger.info("Pre-training complete!")
