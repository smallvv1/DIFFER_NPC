"""训练集逐样本损失的 GMM 可靠性估计。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMResult:
    probabilities: np.ndarray
    selected: np.ndarray
    means: list[float]
    weights: list[float]
    variances: list[float]
    clean_component: int
    threshold: float
    fallback: bool

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["probabilities"] = self.probabilities.tolist()
        result["selected"] = self.selected.astype(bool).tolist()
        return result


def fit_clean_probabilities(
    losses: Sequence[float] | np.ndarray,
    threshold: float = 0.99,
    min_samples: int = 20,
    random_state: int = 0,
) -> GMMResult:
    """拟合双成分 GMM，并返回低均值成分的后验概率。

    训练样本不足、损失退化或 GMM 拟合失败时使用保守的有限回退，且把
    回退原因写入结果，避免把“全部选中”误报为成功的 GMM 筛选。
    """
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("GMM 输入损失为空")
    finite = np.isfinite(values)
    safe_values = values.copy()
    if not finite.all():
        finite_values = safe_values[finite]
        replacement = float(np.max(finite_values)) if finite_values.size else 0.0
        safe_values[~finite] = replacement

    fallback = False
    try:
        if safe_values.size < max(2, int(min_samples)) or np.ptp(safe_values) <= 1e-12:
            raise ValueError("样本数量不足或损失分布退化")
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            max_iter=100,
            tol=1e-2,
            reg_covar=5e-4,
            random_state=random_state,
        )
        gmm.fit(safe_values.reshape(-1, 1))
        posterior = gmm.predict_proba(safe_values.reshape(-1, 1))
        clean_component = int(np.argmin(gmm.means_.reshape(-1)))
        probabilities = posterior[:, clean_component]
        means = gmm.means_.reshape(-1).astype(float).tolist()
        weights = gmm.weights_.reshape(-1).astype(float).tolist()
        variances = gmm.covariances_.reshape(-1).astype(float).tolist()
    except Exception:
        # 不能凭空把困难样本判成噪声；退化时用有限概率，并让调用方在日志中
        # 看到 fallback=True。阈值仍然生效，后续记忆库会按身份做回退。
        fallback = True
        probabilities = np.ones_like(safe_values, dtype=np.float64)
        clean_component = 0
        means = [float(np.mean(safe_values)), float(np.mean(safe_values))]
        weights = [1.0, 0.0]
        variances = [float(np.var(safe_values)), float(np.var(safe_values))]

    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities[~finite] = 0.0
    selected = probabilities >= float(threshold)
    return GMMResult(
        probabilities=probabilities,
        selected=selected,
        means=means,
        weights=weights,
        variances=variances,
        clean_component=clean_component,
        threshold=float(threshold),
        fallback=fallback,
    )


def reliability_coverage(records: Sequence[dict], result: GMMResult) -> dict:
    """统计 GMM 选择对身份与服装状态的覆盖，供中文诊断报告使用。"""
    selected_indices = np.flatnonzero(result.selected)
    selected_records = [records[int(i)] for i in selected_indices]
    pids = {str(item.get("pid")) for item in selected_records}
    clothes = {str(item.get("clothes_id")) for item in selected_records}
    total_pids = {str(item.get("pid")) for item in records}
    total_clothes = {str(item.get("clothes_id")) for item in records}
    return {
        "样本总数": len(records),
        "GMM选中数量": int(result.selected.sum()),
        "GMM选中比例": float(result.selected.mean()),
        "身份覆盖数": len(pids),
        "身份总数": len(total_pids),
        "身份覆盖率": float(len(pids) / max(1, len(total_pids))),
        "服装覆盖数": len(clothes),
        "服装总数": len(total_clothes),
        "服装覆盖率": float(len(clothes) / max(1, len(total_clothes))),
        "GMM回退": bool(result.fallback),
    }


def identity_diagnostics(
    records: Sequence[dict],
    losses: Sequence[float] | np.ndarray,
    result: GMMResult,
    reference_metadata: Sequence[dict] | None = None,
) -> list[dict]:
    """按身份展开 GMM 选择、跨服装和可靠参考覆盖情况。

    该函数只读取 GMM 已产生的后验和选择结果，不改变拟合公式或阈值。
    """
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    by_pid: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_pid.setdefault(str(record.get("pid")), []).append(index)

    diagnostics: list[dict] = []
    for pid in sorted(by_pid):
        indices = by_pid[pid]
        pid_clothes = {str(records[index].get("clothes_id")) for index in indices}
        selected_indices = [index for index in indices if bool(result.selected[index])]
        selected_cross = [
            index for index in selected_indices
            if any(records[other].get("clothes_id") != records[index].get("clothes_id") for other in indices)
        ]
        pid_losses = values[indices]
        selected_losses = values[selected_indices] if selected_indices else np.asarray([], dtype=np.float64)
        entry = {
            "pid": pid,
            "sample_count": len(indices),
            "clothes_count": len(pid_clothes),
            "identity_ce_mean": float(np.mean(pid_losses)),
            "identity_ce_std": float(np.std(pid_losses)),
            "identity_ce_median": float(np.median(pid_losses)),
            "identity_ce_p25": float(np.percentile(pid_losses, 25)),
            "identity_ce_p75": float(np.percentile(pid_losses, 75)),
            "identity_ce_min": float(np.min(pid_losses)),
            "identity_ce_max": float(np.max(pid_losses)),
            "gmm_selected_count": len(selected_indices),
            "gmm_selected_ratio": float(len(selected_indices) / max(1, len(indices))),
            "selected_ce_mean": float(np.mean(selected_losses)) if selected_indices else None,
            "selected_cross_clothes_count": len(selected_cross),
            "selected_cross_clothes_ratio": float(len(selected_cross) / max(1, len(selected_indices))),
            "selected_clothes_count": len({str(records[index].get("clothes_id")) for index in selected_indices}),
        }
        if reference_metadata is not None:
            refs = [reference_metadata[index] for index in indices]
            valid_refs = [item for item in refs if item.get("valid_reference", False)]
            entry.update({
                "reliable_reference_count": len(valid_refs),
                "reliable_reference_coverage": float(len(valid_refs) / max(1, len(indices))),
                "reliable_cross_clothes_count": sum(bool(item.get("cross_clothes")) for item in valid_refs),
                "reliable_cross_clothes_ratio": float(
                    sum(bool(item.get("cross_clothes")) for item in valid_refs) / max(1, len(valid_refs))
                ),
                "self_reference_count": sum(
                    item.get("valid_reference", False)
                    and item.get("source_index") == item.get("reference_index")
                    for item in refs
                ),
            })
        diagnostics.append(entry)
    return diagnostics
