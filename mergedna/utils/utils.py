"""Utility functions for MergeDNA."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        breakdown[name] = n

    return {
        "total": total,
        "trainable": trainable,
        "breakdown": breakdown,
    }


def print_model_summary(model: torch.nn.Module):
    """Print model parameter summary."""
    info = count_parameters(model)
    print(f"\nModel Parameter Summary:")
    print(f"  Total: {info['total'] / 1e6:.2f}M")
    print(f"  Trainable: {info['trainable'] / 1e6:.2f}M")
    print(f"\n  Breakdown:")
    for name, count in info["breakdown"].items():
        print(f"    {name}: {count / 1e6:.2f}M")
    print()
