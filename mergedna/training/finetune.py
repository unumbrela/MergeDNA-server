"""Fine-tuning runner for MergeDNA.

Supports:
- Sequence classification (Genomic Benchmark, NT Benchmark, GUE)
- Token classification (splice site prediction)
- LoRA fine-tuning for parameter efficiency
"""

import os
import logging
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
import numpy as np

from ..model.mergedna import (
    MergeDNA,
    MergeDNAConfig,
    MergeDNAForSequenceClassification,
    MergeDNAForTokenClassification,
)
from ..data.tokenizer import DNACharTokenizer
from ..data.collator import FineTuneCollator

logger = logging.getLogger(__name__)


class FineTuneRunner:
    """Fine-tuning runner for downstream tasks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Training params
        self.num_epochs = config.get("num_epochs", 10)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 5e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_ratio = config.get("warmup_ratio", 0.1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_amp = config.get("use_amp", True)
        self.use_lora = config.get("use_lora", True)
        self.lora_rank = config.get("lora_rank", 8)
        self.lora_alpha = config.get("lora_alpha", 16)
        self.output_dir = config.get("output_dir", "./outputs/finetune")

    def build_model(
        self,
        num_classes: int,
        task_type: str = "sequence_classification",
        pretrain_ckpt: Optional[str] = None,
    ) -> nn.Module:
        """Build and load model for fine-tuning."""
        model_config = MergeDNAConfig(
            vocab_size=self.config.get("vocab_size", 10),
            embed_dim=self.config.get("embed_dim", 1024),
            num_heads=self.config.get("num_heads", 16),
            local_encoder_layers=self.config.get("local_encoder_layers", 4),
            latent_encoder_layers=self.config.get("latent_encoder_layers", 20),
            latent_decoder_layers=self.config.get("latent_decoder_layers", 4),
            local_decoder_layers=self.config.get("local_decoder_layers", 2),
            window_size=self.config.get("window_size", 16),
            use_flash_attn=self.config.get("use_flash_attn", True),
            gradient_checkpointing=self.config.get("gradient_checkpointing", False),
        )

        if task_type == "sequence_classification":
            model = MergeDNAForSequenceClassification(model_config, num_classes)
        elif task_type == "token_classification":
            model = MergeDNAForTokenClassification(model_config, num_classes)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Load pre-trained weights
        if pretrain_ckpt and os.path.exists(pretrain_ckpt):
            ckpt = torch.load(pretrain_ckpt, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            # Load into the mergedna backbone
            missing, unexpected = model.mergedna.load_state_dict(
                state_dict, strict=False
            )
            logger.info(
                f"Loaded pretrained weights. Missing: {len(missing)}, "
                f"Unexpected: {len(unexpected)}"
            )

        # Apply LoRA if requested
        if self.use_lora and task_type == "sequence_classification":
            model.apply_lora(rank=self.lora_rank, alpha=self.lora_alpha)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(
                f"LoRA applied. Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M "
                f"({100 * trainable / total:.1f}%)"
            )

        return model.to(self.device)

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Run fine-tuning training loop."""
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scaler = GradScaler("cuda", enabled=self.use_amp)

        best_metric = -float("inf")
        best_results = {}

        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                global_step = epoch * len(train_loader) + step

                # Cosine schedule with warmup
                if global_step < warmup_steps:
                    lr = self.learning_rate * global_step / max(1, warmup_steps)
                else:
                    progress = (global_step - warmup_steps) / max(
                        1, total_steps - warmup_steps
                    )
                    lr = self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                with autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.use_amp,
                ):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} | Loss: {avg_loss:.4f}")

            # Evaluate
            if val_loader is not None:
                results = self.evaluate(model, val_loader)
                metric = results.get("accuracy", results.get("mcc", 0))
                logger.info(
                    f"Epoch {epoch + 1} Eval: {results}"
                )

                if metric > best_metric:
                    best_metric = metric
                    best_results = results
                    # Save best model
                    os.makedirs(self.output_dir, exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.output_dir, "best_model.pt"),
                    )

        return best_results

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        model.eval()
        all_preds = []
        all_labels = []

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        return {
            "accuracy": accuracy,
            "mcc": mcc,
            "f1": f1,
        }
