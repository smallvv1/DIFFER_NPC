"""Aggregate temporary-vs-formal reference-risk prediction across paired batches.

"harm" here is only a proxy: positive change in a *reliable reference*
identity or cross-clothes consistency loss, not necessarily a retrieval error.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


MODES = ("TEMP_COPY_OPT_STATE_FALSE", "TEMP_COPY_OPT_STATE_TRUE")


def _mean(values):
    return sum(values) / len(values) if values else None


def _mode_summary(batches: Sequence[Mapping], mode: str, thresholds: Sequence[float]):
    if not batches:
        raise ValueError("At least one paired batch is required")
    cosines = []
    points = []
    batch_errors = []
    for result in batches:
        data = result[mode]
        rows = data["paired_reference_rows"]
        if not rows:
            raise ValueError("Batch without reliable-reference pairs")
        value = data["temporary_vs_formal_update"]["cosine"]
        if value is None or not math.isfinite(float(value)):
            raise ValueError("Non-finite temporary/formal update cosine")
        cosines.append(float(value))
        ref_errors = [
            abs(float(row["predicted_reference_delta"]) - float(row["actual_reference_delta"]))
            for row in rows
        ]
        batch_errors.append(_mean(ref_errors))
        points.extend(rows)
    if not points:
        raise ValueError("No valid reference pairs")
    for row in points:
        for key in ("actual_identity_delta", "actual_reference_delta",
                    "predicted_identity_delta", "predicted_reference_delta",
                    "weight", "risk_ratio"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Non-finite risk field: {key}")
    result = {
        "batch_count": len(batches),
        "reference_observation_count": len(points),
        "unique_anchor_reference_pairs": len({
            (int(row["sample_index"]), int(row["reference_index"])) for row in points
        }),
        "unique_pids": len({int(row["pid"]) for row in points}),
        "mean_update_cosine_over_batches": _mean(cosines),
        "reference_absolute_error_mean_over_observations": _mean([
            abs(float(row["predicted_reference_delta"]) - float(row["actual_reference_delta"]))
            for row in points
        ]),
        "identity_absolute_error_mean_over_observations": _mean([
            abs(float(row["predicted_identity_delta"]) - float(row["actual_identity_delta"]))
            for row in points
        ]),
        "reference_absolute_error_mean_over_batches": _mean(batch_errors),
        "low_weight_lt_0_9": sum(float(row["weight"]) < 0.9 for row in points),
        "proxy_risk_thresholds": {},
    }
    for threshold in thresholds:
        if threshold < 0:
            raise ValueError("Threshold must not be negative")
        tp = fp = fn = tn = 0
        false_downweights = 0
        for row in points:
            # Operational proxy: either reliable reference loss got worse.
            actually_harmful = (
                float(row["actual_identity_delta"]) > threshold
                or float(row["actual_reference_delta"]) > threshold
            )
            predicted_harmful = float(row["risk_ratio"]) < 1.0
            if predicted_harmful and actually_harmful:
                tp += 1
            elif predicted_harmful:
                fp += 1
            elif actually_harmful:
                fn += 1
            else:
                tn += 1
            if not actually_harmful and float(row["weight"]) < 0.9:
                false_downweights += 1
        result["proxy_risk_thresholds"][str(threshold)] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_downweights_w_lt_0_9_on_nonharmful": false_downweights,
            "actual_harmful_proxy_count": tp + fn,
        }
    return result


def aggregate_paired_risk(
    batches: Sequence[Mapping],
    thresholds: Sequence[float] = (0.0, 1e-5, 1e-4),
):
    for item in batches:
        for mode in MODES:
            if mode not in item:
                raise ValueError(f"Unpaired comparison batch lacks mode {mode}")
        left = item[MODES[0]]["paired_reference_rows"]
        right = item[MODES[1]]["paired_reference_rows"]
        if [(r["sample_index"], r["reference_index"]) for r in left] != [
            (r["sample_index"], r["reference_index"]) for r in right
        ]:
            raise ValueError("False/True predictions not aligned to the same samples")
        for a, b in zip(left, right):
            for key in ("actual_identity_delta", "actual_reference_delta"):
                if abs(float(a[key]) - float(b[key])) > 1e-4:
                    raise ValueError("Ground truth formal update differs across modes")
    return {
        "label_semantics": "proxy loss harm: formal identity OR reference consistency delta greater than threshold",
        "paired_batches": len(batches),
        "aggregation_warning": "per-sample observations may share PID/anchor, and are not independent significance tests",
        "modes": {
            mode: _mode_summary(batches, mode, thresholds)
            for mode in MODES
        },
    }
