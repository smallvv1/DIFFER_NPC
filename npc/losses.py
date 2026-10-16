"""NPC-ReID 的逐样本信号、记忆监督和权重公式。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def extract_classifier_logits(score: Any) -> torch.Tensor:
    if isinstance(score, dict):
        score = score["cls_score"]
    if isinstance(score, (list, tuple)):
        score = score[0]
    if not torch.is_tensor(score) or score.ndim != 2:
        raise TypeError("无法从 DIFFER 输出中提取二维身份分类 logits")
    return score


def extract_reid_features(feat: Any) -> torch.Tensor:
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    if not torch.is_tensor(feat) or feat.ndim != 2:
        raise TypeError("无法从 DIFFER 输出中提取二维 ReID 特征")
    return feat


def per_sample_identity_ce(
    score: Any,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """返回与原始身份 CE 同定义的逐样本值，不做批次平均。"""
    logits = extract_classifier_logits(score)
    target = target.to(device=logits.device, dtype=torch.long)
    if label_smoothing <= 0:
        return F.cross_entropy(logits, target, reduction="none")
    log_probs = F.log_softmax(logits, dim=1)
    one_hot = torch.zeros_like(log_probs).scatter_(1, target[:, None], 1.0)
    smoothed = (1.0 - label_smoothing) * one_hot + label_smoothing / logits.shape[1]
    return (-smoothed * log_probs).sum(dim=1)


def per_sample_reference_consistency(
    anchor_features: torch.Tensor,
    reference_features: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """同身份可靠参考的一致性损失，值为 ``1-cosine``。"""
    anchor = F.normalize(anchor_features.float(), dim=-1, eps=eps)
    reference = F.normalize(reference_features.float(), dim=-1, eps=eps)
    return (1.0 - (anchor * reference).sum(dim=-1)).clamp_min(0.0)


def calculate_npc_weights(
    pre_identity: torch.Tensor,
    post_identity: torch.Tensor,
    pre_reference: torch.Tensor,
    post_reference: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """实现 NPC 比值与权重公式，并显式支持两项信号变化不一致的情况。"""
    tensors = [pre_identity, post_identity, pre_reference, post_reference]
    tensors = [torch.nan_to_num(x.detach().float(), nan=1.0, posinf=1.0, neginf=1.0) for x in tensors]
    p_before, p_after, q_before, q_after = tensors
    p_ratio = p_before.clamp_min(eps) / p_after.clamp_min(eps)
    q_ratio = q_before.clamp_min(eps) / q_after.clamp_min(eps)
    ratio = 0.5 * (p_ratio + q_ratio)
    ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=0.0)
    weights = torch.where(ratio < 1.0, torch.tanh(ratio), torch.ones_like(ratio))
    weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return ratio, weights, torch.stack((p_ratio, q_ratio), dim=-1)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    values = values.float()
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def weight_statistics(weights: torch.Tensor) -> dict[str, float]:
    values = weights.detach().float()
    return {
        "均值": float(values.mean().item()),
        "最小值": float(values.min().item()),
        "最大值": float(values.max().item()),
        "低权重比例(<0.5)": float((values < 0.5).float().mean().item()),
        "低权重比例(<0.9)": float((values < 0.9).float().mean().item()),
    }
