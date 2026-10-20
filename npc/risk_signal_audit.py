"""Read-only paired audit of NPC's two-channel risk decision.

The labels below are training-reference *proxy losses*, not retrieval accuracy
or evidence that changing the NPC loss improves LTCC.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

MODES = ("TEMP_COPY_OPT_STATE_FALSE", "TEMP_COPY_OPT_STATE_TRUE")
RULES = ("original_mean_ratio", "identity_only", "consistency_only",
         "either_channel", "both_channels")


def _finite(row: Mapping, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} for anchor {row.get('sample_index')}")
    return value


def _safe_rate(a: int, b: int) -> float | None:
    return a / b if b else None


def _classify(row: Mapping) -> dict[str, bool]:
    # The original NPC score is 0.5*(pre_p/post_p + pre_q/post_q).
    # Independent channels are read-only counterfactual diagnostics.
    ip = _finite(row, "predicted_identity_delta") > 0.0
    cq = _finite(row, "predicted_reference_delta") > 0.0
    return {
        "original_mean_ratio": _finite(row, "risk_ratio") < 1.0,
        "identity_only": ip,
        "consistency_only": cq,
        "either_channel": ip or cq,
        "both_channels": ip and cq,
    }


def _confusion(predictions: Sequence[bool], labels: Sequence[bool]) -> dict:
    tp = sum(p and a for p, a in zip(predictions, labels))
    fp = sum(p and not a for p, a in zip(predictions, labels))
    fn = sum((not p) and a for p, a in zip(predictions, labels))
    tn = sum((not p) and (not a) for p, a in zip(predictions, labels))
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": _safe_rate(tp, tp + fp),
        "recall": _safe_rate(tp, tp + fn),
        "false_positive_rate": _safe_rate(fp, fp + tn),
    }


def summarize_risk_signals(
    payload: Mapping,
    thresholds: Sequence[float] = (0.0, 1e-5, 1e-4),
) -> dict:
    batches = payload["paired_results"]
    if not batches:
        raise ValueError("No paired batch records")
    if any((not math.isfinite(float(t)) or float(t) < 0) for t in thresholds):
        raise ValueError("Thresholds must be finite and >= 0")
    result = {
        "schema_version": 1,
        "source_checkpoint": payload.get("checkpoint"),
        "source_checkpoint_sha256": payload.get("checkpoint_sha256"),
        "batch_count": len(batches),
        "thresholds": [float(t) for t in thresholds],
        "modes": {},
        "interpretation": (
            "Proxy harm means the formal reference identity CE OR "
            "cross-clothes consistency loss delta exceeds the threshold; "
            "this is not measured ReID damage. Alternative decisions are "
            "shadow audits only, and do not change NPC training."
        ),
    }
    for mode in MODES:
        rows = []
        for batch in batches:
            if mode not in batch:
                raise ValueError(f"Missing {mode} in paired batch")
            current = batch[mode]["paired_reference_rows"]
            if not current:
                raise ValueError("Comparison batch has no valid reference")
            rows.extend(current)
        decisions = [_classify(row) for row in rows]
        ids = [
            (int(row["sample_index"]), int(row["reference_index"]))
            for row in rows
        ]
        predicted_ip = [d["identity_only"] for d in decisions]
        predicted_cq = [d["consistency_only"] for d in decisions]
        original = [d["original_mean_ratio"] for d in decisions]
        masked_by_average = [
            i for i, d in enumerate(decisions)
            if d["either_channel"] and not d["original_mean_ratio"]
        ]
        active_weight = [
            _finite(row, "weight") < 0.9 for row in rows
        ]
        pred_identity_deltas = [_finite(row, "predicted_identity_delta") for row in rows]
        pred_reference_deltas = [_finite(row, "predicted_reference_delta") for row in rows]
        actual_identity = [_finite(row, "actual_identity_delta") for row in rows]
        actual_reference = [_finite(row, "actual_reference_delta") for row in rows]
        out = {
            "observations": len(rows),
            "unique_anchor_reference_pairs": len(set(ids)),
            "unique_identity_count": len({int(row["pid"]) for row in rows}),
            "original_risk_positive_count": sum(original),
            "identity_risk_positive_count": sum(predicted_ip),
            "consistency_risk_positive_count": sum(predicted_cq),
            "either_channel_risk_count": sum(d["either_channel"] for d in decisions),
            "both_channels_risk_count": sum(d["both_channels"] for d in decisions),
            "opposing_direction_count": sum(a != b for a, b in zip(predicted_ip, predicted_cq)),
            "either_channel_masked_by_average_count": len(masked_by_average),
            "masked_identity_only_count": sum(predicted_ip[i] and not predicted_cq[i] for i in masked_by_average),
            "masked_consistency_only_count": sum(predicted_cq[i] and not predicted_ip[i] for i in masked_by_average),
            "actual_weight_lt_0_9_count": sum(active_weight),
            "threshold_comparisons": {},
            "masking_examples": [
                {
                    "sample_index": ids[i][0],
                    "reference_index": ids[i][1],
                    "pid": int(rows[i]["pid"]),
                    "predicted_identity_delta": pred_identity_deltas[i],
                    "predicted_reference_delta": pred_reference_deltas[i],
                    "actual_identity_delta": actual_identity[i],
                    "actual_reference_delta": actual_reference[i],
                    "ratio": _finite(rows[i], "risk_ratio"),
                    "weight": _finite(rows[i], "weight"),
                }
                for i in masked_by_average[:10]
            ],
        }
        for threshold in thresholds:
            target = [
                dp > threshold or dq > threshold
                for dp, dq in zip(actual_identity, actual_reference)
            ]
            out["threshold_comparisons"][str(float(threshold))] = {
                "proxy_harmful_count": sum(target),
                "rules": {
                    name: _confusion([d[name] for d in decisions], target)
                    for name in RULES
                },
                "wrong_downweight_lt_0_9": sum(w and not h for w, h in zip(active_weight, target)),
                "missed_actual_harm_by_original": sum(
                    (not d["original_mean_ratio"]) and h
                    for d, h in zip(decisions, target)
                ),
                "masked_by_average_and_actually_harmful": sum(target[i] for i in masked_by_average),
            }
        result["modes"][mode] = out
    left = result["modes"][MODES[0]]
    right = result["modes"][MODES[1]]
    if (left["observations"], left["unique_anchor_reference_pairs"]) != (
        right["observations"], right["unique_anchor_reference_pairs"]
    ):
        raise ValueError("Mode observations are not paired")
    return result
