"""NDP 局部身份参考与局部间隔保护的独立实现。

设计边界：
1. 只接受 train split 的样本与特征；调用方负责证明没有传入 query/gallery。
2. 正样本严格要求同 PID、不同 clothes_id，且索引不能等于 anchor。
3. 困难负样本严格要求不同 PID，按初始特征相似度从高到低选 K 个。
4. ``local_margin_protection_loss`` 使用冻结的参考特征，仅对当前 anchor
   特征产生梯度。这是本轮诊断探针，不是已经冻结的正式 NDP 训练算法。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _as_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def build_train_only_reference_library(
    items: list[dict[str, Any]],
    features: torch.Tensor | np.ndarray,
    hard_k: int = 10,
    negative_block_size: int = 512,
) -> dict[str, Any]:
    """构建 train-only 的同身份正样本和 K 个困难异身份邻居。

    ``features`` 必须与 ``items`` 一一对应，并且已经由同一 train-only
    checkpoint 提取。每个 anchor 的正样本选择为跨服装候选中相似度最高者；
    该规则不读取测试集，也不使用任何额外的置信度阈值。
    """

    if isinstance(features, np.ndarray):
        feature_tensor = torch.from_numpy(features)
    else:
        feature_tensor = features.detach().cpu()
    feature_tensor = F.normalize(feature_tensor.float(), dim=1)
    if feature_tensor.ndim != 2 or feature_tensor.shape[0] != len(items):
        raise ValueError("features 必须是 [len(items), dim]，且与 train items 对齐")
    if hard_k < 1:
        raise ValueError("hard_k 必须为正数")

    pids = np.asarray([_as_int(item["pid"]) for item in items], dtype=np.int64)
    clothes = np.asarray([_as_int(item["clothes_id"]) for item in items], dtype=np.int64)
    by_pid: dict[int, list[int]] = defaultdict(list)
    for index, pid in enumerate(pids.tolist()):
        by_pid[int(pid)].append(index)

    positive_index = np.full(len(items), -1, dtype=np.int64)
    positive_similarity = np.full(len(items), np.nan, dtype=np.float32)
    for anchor in range(len(items)):
        candidates = [
            index
            for index in by_pid[int(pids[anchor])]
            if index != anchor and clothes[index] != clothes[anchor]
        ]
        if candidates:
            candidate_tensor = feature_tensor[candidates]
            similarities = candidate_tensor @ feature_tensor[anchor]
            # 最高相似度优先；索引作为稳定的第二排序键，避免并列随机化。
            order = sorted(
                range(len(candidates)),
                key=lambda position: (-float(similarities[position]), int(candidates[position])),
            )
            chosen = candidates[order[0]]
            positive_index[anchor] = int(chosen)
            positive_similarity[anchor] = float(similarities[order[0]])

    negative_indices = np.full((len(items), hard_k), -1, dtype=np.int64)
    negative_similarity = np.full((len(items), hard_k), np.nan, dtype=np.float32)
    all_indices = np.arange(len(items), dtype=np.int64)
    block_size = max(1, int(negative_block_size))
    for start in range(0, len(items), block_size):
        stop = min(len(items), start + block_size)
        block_similarity = feature_tensor[start:stop] @ feature_tensor.T
        for row, anchor in enumerate(range(start, stop)):
            valid = all_indices[pids != pids[anchor]]
            if len(valid) == 0:
                continue
            values = block_similarity[row, valid]
            take = min(hard_k, len(valid))
            # 稳定排序同时保证不同 PID；self 已经因 PID 过滤而排除。
            order = np.argsort(-values.numpy(), kind="stable")[:take]
            chosen = valid[order]
            negative_indices[anchor, :take] = chosen
            negative_similarity[anchor, :take] = values[order].numpy().astype(np.float32)

    records = []
    for anchor, item in enumerate(items):
        pos = int(positive_index[anchor])
        negs = [int(index) for index in negative_indices[anchor].tolist() if int(index) >= 0]
        records.append(
            {
                "index": int(anchor),
                "image_path": str(item["image_path"]),
                "pid": int(pids[anchor]),
                "clothes_id": int(clothes[anchor]),
                "positive_index": pos,
                "positive_image_path": str(items[pos]["image_path"]) if pos >= 0 else None,
                "positive_pid": int(pids[pos]) if pos >= 0 else None,
                "positive_clothes_id": int(clothes[pos]) if pos >= 0 else None,
                "positive_similarity": float(positive_similarity[anchor]) if pos >= 0 else None,
                "negative_indices": negs,
                "negative_image_paths": [str(items[index]["image_path"]) for index in negs],
                "negative_pids": [int(pids[index]) for index in negs],
                "negative_similarities": [float(value) for value in negative_similarity[anchor] if np.isfinite(value)],
                "valid_positive": bool(
                    pos >= 0
                    and pos != anchor
                    and int(pids[pos]) == int(pids[anchor])
                    and int(clothes[pos]) != int(clothes[anchor])
                ),
                "valid_negative_count": len(negs),
            }
        )

    valid_records = [row for row in records if row["valid_positive"] and row["valid_negative_count"] >= hard_k]
    return {
        "hard_k": int(hard_k),
        "items": records,
        "valid_anchor_indices": [int(row["index"]) for row in valid_records],
        "train_image_count": len(items),
        "positive_coverage": float(sum(row["valid_positive"] for row in records) / max(len(records), 1)),
        "full_reference_coverage": float(len(valid_records) / max(len(records), 1)),
        "self_reference_count": int(
            sum(1 for row in records if row["positive_index"] >= 0 and row["positive_index"] == row["index"])
        ),
        "positive_relation_contract": "same PID, different clothes_id, index != anchor",
        "negative_relation_contract": "different PID, top-K cosine neighbors from train-only features",
        "train_only": True,
    }


def local_margin_protection_loss(
    anchor_features: torch.Tensor,
    reference_features: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """计算独立 NDP 局部间隔保护损失。

    对每个有效 anchor：
    ``relu(margin + max_k cos(anchor, neg_k) - cos(anchor, pos))``。

    参考特征应来自冻结的初始 train-only 参考库；函数不对它们 detach，
    由调用方显式传入 detached 张量，以便测试梯度方向和计算对象。
    """

    if anchor_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("anchor_features/reference_features 必须是二维张量")
    if anchor_features.shape[0] != positive_indices.shape[0] or positive_indices.shape[0] != negative_indices.shape[0]:
        raise ValueError("anchor 与参考索引的 batch 维度不一致")
    if margin < 0:
        raise ValueError("margin 不应为负数")

    valid_positive = positive_indices >= 0
    valid_negative = (negative_indices >= 0).any(dim=1)
    valid = valid_positive & valid_negative
    if not bool(valid.any()):
        zero = anchor_features.sum() * 0.0
        return zero, {
            "active_anchor_count": 0,
            "valid_anchor_count": 0,
            "total_anchor_count": int(anchor_features.shape[0]),
            "mean_hinge": 0.0,
            "positive_similarity": None,
            "hard_negative_similarity": None,
            "margin": float(margin),
        }

    anchor = F.normalize(anchor_features[valid].float(), dim=1)
    pos_idx = positive_indices[valid].long()
    pos = F.normalize(reference_features[pos_idx].float(), dim=1)
    pos_sim = (anchor * pos).sum(dim=1)

    neg_idx = negative_indices[valid].long()
    neg_valid = neg_idx >= 0
    safe_neg_idx = neg_idx.clamp_min(0)
    neg = F.normalize(reference_features[safe_neg_idx].float(), dim=2)
    neg_sim = torch.einsum("bd,bkd->bk", anchor, neg)
    neg_sim = neg_sim.masked_fill(~neg_valid, -1.0)
    hard_neg_sim = neg_sim.max(dim=1).values
    hinge = F.relu(float(margin) + hard_neg_sim - pos_sim)
    loss = hinge.mean()
    return loss, {
        "active_anchor_count": int((hinge > 0).sum().detach().cpu()),
        "valid_anchor_count": int(valid.sum().detach().cpu()),
        "total_anchor_count": int(anchor_features.shape[0]),
        "mean_hinge": float(hinge.detach().mean().cpu()),
        "positive_similarity": float(pos_sim.detach().mean().cpu()),
        "hard_negative_similarity": float(hard_neg_sim.detach().mean().cpu()),
        "margin": float(margin),
    }


def summarize_local_geometry(
    anchor_features: torch.Tensor | np.ndarray,
    positive_features: torch.Tensor | np.ndarray,
    negative_features: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """计算一组固定 reference rows 的局部排序和间隔统计。"""

    def tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        return F.normalize(value.detach().float().cpu(), dim=-1)

    anchor = tensor(anchor_features)
    positive = tensor(positive_features)
    negative = tensor(negative_features)
    if anchor.ndim != 2 or positive.shape != anchor.shape or negative.ndim != 3 or negative.shape[0] != anchor.shape[0]:
        raise ValueError("局部几何张量形状不正确")
    positive_similarity = (anchor * positive).sum(dim=1)
    negative_similarity = torch.einsum("bd,bkd->bk", anchor, negative)
    max_negative = negative_similarity.max(dim=1).values
    mean_negative = negative_similarity.mean(dim=1)
    margin = positive_similarity - max_negative
    # 1 表示正样本最高；并列时使用 conservative 的排名。
    rank = 1 + (negative_similarity >= positive_similarity.unsqueeze(1)).sum(dim=1)
    return {
        "count": int(anchor.shape[0]),
        "cross_clothes_positive_similarity_mean": float(positive_similarity.mean()),
        "cross_clothes_positive_similarity_std": float(positive_similarity.std(unbiased=False)),
        "hard_negative_similarity_mean": float(max_negative.mean()),
        "hard_negative_similarity_std": float(max_negative.std(unbiased=False)),
        "mean_negative_similarity_mean": float(mean_negative.mean()),
        "local_margin_mean": float(margin.mean()),
        "local_margin_std": float(margin.std(unbiased=False)),
        "positive_rank1_rate": float((rank == 1).float().mean()),
        "impostor_intrusion_rate": float((max_negative >= positive_similarity).float().mean()),
        "rank_mean": float(rank.float().mean()),
        "rank_max": int(rank.max()),
        "rank_values": rank.tolist(),
        "positive_similarity_values": positive_similarity.tolist(),
        "hard_negative_similarity_values": max_negative.tolist(),
        "local_margin_values": margin.tolist(),
    }
