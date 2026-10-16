"""仅由训练集建立的身份/服装可追踪记忆库。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class MemoryBank:
    reference_indices: np.ndarray
    records: list[dict]
    reference_metadata: list[dict]
    valid_reference_mask: np.ndarray
    features: torch.Tensor | None = None
    version: int = 0

    def __post_init__(self) -> None:
        """在记忆库边界再次拒绝自引用、越界和身份错配。"""
        total = len(self.records)
        if self.reference_indices.shape != (total,):
            raise ValueError("reference_indices 形状与训练集记录不一致")
        if self.valid_reference_mask.shape != (total,):
            raise ValueError("valid_reference_mask 形状与训练集记录不一致")
        if len(self.reference_metadata) != total:
            raise ValueError("reference_metadata 长度与训练集记录不一致")
        for source_index, is_valid in enumerate(self.valid_reference_mask):
            if not bool(is_valid):
                continue
            reference_index = int(self.reference_indices[source_index])
            if reference_index < 0 or reference_index >= total:
                raise ValueError("有效记忆参考索引越界")
            if reference_index == source_index:
                raise ValueError("记忆库禁止当前样本引用自身")
            if self.records[source_index].get("pid") != self.records[reference_index].get("pid"):
                raise ValueError("记忆库参考 PID 与源样本不一致")

    def update_features(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[0] != len(self.records):
            raise ValueError("记忆特征形状与训练集记录不一致")
        self.features = F.normalize(features.detach().float().cpu(), dim=-1)
        self.version += 1

    def lookup_with_validity(
        self,
        indices: torch.Tensor | Sequence[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回参考特征和有效性；无效参考用零向量占位但不会被调用方使用。"""
        if self.features is None:
            raise RuntimeError("记忆库尚未写入特征")
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().long().tolist()
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        valid = torch.as_tensor(self.valid_reference_mask[idx], dtype=torch.bool)
        result = torch.zeros((len(idx), self.features.shape[1]), dtype=self.features.dtype)
        if bool(valid.any()):
            refs = torch.as_tensor(self.reference_indices[idx[valid]], dtype=torch.long)
            result[valid] = self.features.index_select(0, refs)
        return result.to(device), valid.to(device)

    def lookup(self, indices: torch.Tensor | Sequence[int], device: torch.device) -> torch.Tensor:
        """仅允许全部参考有效；混入无效参考时显式报错，禁止静默自引用。"""
        result, valid = self.lookup_with_validity(indices, device)
        if not bool(valid.all()):
            raise RuntimeError("记忆库包含无效参考；请使用 lookup_with_validity 并屏蔽这些样本")
        return result

    def coverage(self) -> dict[str, float | int]:
        cross = 0
        same = 0
        fallback = 0
        valid_count = 0
        reliable_pids: set[str] = set()
        for ref in self.reference_metadata:
            if not ref.get("valid_reference", False):
                continue
            valid_count += 1
            reliable_pids.add(str(ref.get("pid")))
            if ref.get("cross_clothes"):
                cross += 1
            else:
                same += 1
            if ref.get("fallback"):
                fallback += 1
        total = len(self.records)
        return {
            "样本总数": total,
            "有效参考数": valid_count,
            "无效参考数": total - valid_count,
            "有效参考比例": valid_count / max(1, total),
            "跨服装参考数": cross,
            "同服装参考数": same,
            "跨服装参考比例": cross / max(1, valid_count),
            "全样本跨服装参考比例": cross / max(1, total),
            "身份回退数": fallback,
            "身份回退比例": fallback / max(1, valid_count),
            "身份覆盖数": len({str(x.get("pid")) for x in self.records}),
            "可靠参考身份覆盖数": len(reliable_pids),
            "可靠参考身份覆盖率": len(reliable_pids) / max(1, len({str(x.get("pid")) for x in self.records})),
        }


def build_memory_references(
    records: Sequence[dict],
    clean_mask: Sequence[bool] | np.ndarray,
    features: torch.Tensor | None = None,
) -> MemoryBank:
    """为每个训练样本分配可靠同身份参考，跨服装优先，同服装回退。

    可靠参考必须是另一个、GMM 选中的同 PID 训练样本。没有这样的样本时，
    使用 -1 明确表示无参考；绝不把当前样本自身或未选中的样本静默当作参考。
    """
    clean = np.asarray(clean_mask, dtype=bool)
    if len(records) != len(clean):
        raise ValueError("clean_mask 长度与训练集不一致")
    by_pid: dict[object, list[int]] = {}
    for index, record in enumerate(records):
        by_pid.setdefault(record.get("pid"), []).append(index)

    reference_indices = np.full(len(records), -1, dtype=np.int64)
    valid_reference_mask = np.zeros(len(records), dtype=bool)
    reference_metadata: list[dict] = []
    normalized_features = None
    if features is not None:
        if features.ndim != 2 or features.shape[0] != len(records):
            raise ValueError("参考特征形状与训练集记录不一致")
        normalized_features = F.normalize(features.detach().float(), dim=-1).cpu()

    for index, record in enumerate(records):
        candidates = [
            i for i in by_pid.get(record.get("pid"), [])
            if i != index and clean[i]
        ]
        if not candidates:
            reference_metadata.append({
                "source_index": index,
                "reference_index": -1,
                "pid": record.get("pid"),
                "reference_pid": None,
                "clothes_id": record.get("clothes_id"),
                "reference_clothes_id": None,
                "source_path": record.get("image_path"),
                "reference_path": None,
                "cross_clothes": False,
                "fallback": False,
                "valid_reference": False,
                "invalid_reason": "no_reliable_same_pid_reference",
            })
            continue
        cross_cloth = [i for i in candidates if records[i].get("clothes_id") != record.get("clothes_id")]
        preferred = cross_cloth or candidates
        if normalized_features is not None and len(preferred) > 1:
            sims = normalized_features[preferred] @ normalized_features[index]
            ref_index = preferred[int(torch.argmax(sims).item())]
        else:
            ref_index = preferred[0]
        reference_indices[index] = ref_index
        ref_record = records[ref_index]
        valid_reference_mask[index] = True
        reference_metadata.append({
            "source_index": index,
            "reference_index": int(ref_index),
            "pid": record.get("pid"),
            "reference_pid": ref_record.get("pid"),
            "clothes_id": record.get("clothes_id"),
            "reference_clothes_id": ref_record.get("clothes_id"),
            "source_path": record.get("image_path"),
            "reference_path": ref_record.get("image_path"),
            "cross_clothes": bool(ref_record.get("clothes_id") != record.get("clothes_id")),
            "fallback": bool(not cross_cloth),
            "valid_reference": True,
            "invalid_reason": None,
        })

    return MemoryBank(
        reference_indices=reference_indices,
        records=[dict(x) for x in records],
        reference_metadata=reference_metadata,
        valid_reference_mask=valid_reference_mask,
    )
