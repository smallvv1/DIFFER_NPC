"""Safely align loss values calculated on valid-reference subbatches.

A masked subbatch's reference losses must be keyed using the matching masked
anchor indices, not the first N indices from the full training batch.
"""
from __future__ import annotations

from collections.abc import Sequence
import torch


def align_valid_reference_losses(
    full_anchor_indices: torch.Tensor,
    valid_positions: torch.Tensor,
    post_identity: Sequence[float],
    post_consistency: Sequence[float],
) -> dict[int, dict[str, float]]:
    full = torch.as_tensor(full_anchor_indices, dtype=torch.long).reshape(-1)
    positions = torch.as_tensor(valid_positions, dtype=torch.long).reshape(-1)
    if (positions < 0).any() or (positions >= len(full)).any():
        raise ValueError("Valid positions exceed full-batch index bounds")
    ids = full.index_select(0, positions.to(full.device)).detach().cpu().tolist()
    if len(ids) != len(post_identity) or len(ids) != len(post_consistency):
        raise ValueError("Reference loss cardinality mismatched valid anchors")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample indices in batch require occurrence-based matching")
    return {
        int(idx): {"post_identity": float(p), "post_reference": float(q)}
        for idx, p, q in zip(ids, post_identity, post_consistency)
    }
