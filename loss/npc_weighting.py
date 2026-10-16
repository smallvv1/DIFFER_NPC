"""Explicit weighted CE reductions for controlled NPC ablation.

weight_sum reproduces the historical DIFFER+NPC implementation.
batch_mean follows Eq. (9) of Zhang et al., AAAI 2024.
"""
import torch


def reduce_weighted_ce(values: torch.Tensor, weights: torch.Tensor, reduction: str = "weight_sum") -> torch.Tensor:
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("Expected aligned 1D per-sample CE and weights")
    values = values.float()
    weights = weights.to(device=values.device, dtype=values.dtype)
    if reduction == "weight_sum":
        return (values * weights).sum() / weights.sum().clamp_min(1e-12)
    if reduction == "batch_mean":
        return (values * weights).mean()
    raise ValueError(f"Unknown NPC CE reduction: {reduction}")
