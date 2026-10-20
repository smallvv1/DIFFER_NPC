import pytest

from npc.risk_signal_audit import summarize_risk_signals


def row(sample, ref, predicted_p, predicted_q, actual_p, actual_q, ratio, weight=1.0):
    return {
        "sample_index": sample, "reference_index": ref, "pid": sample,
        "predicted_identity_delta": predicted_p,
        "predicted_reference_delta": predicted_q,
        "actual_identity_delta": actual_p,
        "actual_reference_delta": actual_q,
        "risk_ratio": ratio,
        "weight": weight,
    }


def payload(a, b):
    return {
        "checkpoint": "model_best.pth", "checkpoint_sha256": "abcd",
        "paired_results": [{
            "TEMP_COPY_OPT_STATE_FALSE": {"paired_reference_rows": a},
            "TEMP_COPY_OPT_STATE_TRUE": {"paired_reference_rows": b},
        }],
    }


def test_independent_signals_catch_average_cancellation():
    # Identity gets worse while consistency improves even more; mean ratio
    # happens to be >=1, thereby suppressing the original harm flag.
    r = row(10, 15, 0.1, -0.3, 0.2, -0.4, 1.1)
    summary = summarize_risk_signals(payload([r], [r]))
    a = summary["modes"]["TEMP_COPY_OPT_STATE_FALSE"]
    assert a["opposing_direction_count"] == 1
    assert a["either_channel_masked_by_average_count"] == 1
    assert a["masked_identity_only_count"] == 1
    obj = a["threshold_comparisons"]["0.0"]["rules"]
    assert obj["original_mean_ratio"]["FN"] == 1
    assert obj["either_channel"]["TP"] == 1
    assert obj["both_channels"]["FN"] == 1


def test_strict_threshold_and_false_downweight_are_reported():
    harmful = row(1, 2, 0.00002, 0, 0.000009, 0, 0.999, 0.761)
    safe = row(3, 4, 0, -0.0001, 0, -0.001, 1.001, 1.0)
    out = summarize_risk_signals(payload([harmful, safe], [harmful, safe]))
    report = out["modes"]["TEMP_COPY_OPT_STATE_TRUE"]["threshold_comparisons"]["1e-05"]
    assert report["proxy_harmful_count"] == 0
    assert report["rules"]["original_mean_ratio"]["FP"] == 1
    assert report["wrong_downweight_lt_0_9"] == 1


def test_reject_empty_or_incomplete_modes():
    with pytest.raises(ValueError, match="No paired"):
        summarize_risk_signals({"paired_results": []})
    with pytest.raises(ValueError, match="Missing TEMP_COPY"):
        summarize_risk_signals({"paired_results": [{"TEMP_COPY_OPT_STATE_FALSE": {}}]})


def test_reject_nan_values_and_invalid_threshold():
    bad = row(10, 15, 0, 0, 0, 0, float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        summarize_risk_signals(payload([bad], [bad]))
    good = row(10, 15, 0, 0, 0, 0, 1.0)
    with pytest.raises(ValueError, match="Thresholds"):
        summarize_risk_signals(payload([good], [good]), [-0.1])
