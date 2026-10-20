import pytest

from npc.risk_validation import aggregate_paired_risk


def sample_row(sample_index, reference_index, actual_q, predicted_q, ratio, weight,
               actual_p=0.0, predicted_p=0.0):
    return {
        "sample_index": sample_index, "reference_index": reference_index,
        "pid": sample_index // 10, "clothes_id": 0, "reference_clothes_id": 1,
        "actual_identity_delta": actual_p, "predicted_identity_delta": predicted_p,
        "actual_reference_delta": actual_q, "predicted_reference_delta": predicted_q,
        "weight": weight, "risk_ratio": ratio,
    }


def packet(rows, cosine):
    return {
        "paired_reference_rows": rows,
        "temporary_vs_formal_update": {"cosine": cosine},
    }


def paired(a_rows, b_rows):
    return {
        "batch_index": 0,
        "TEMP_COPY_OPT_STATE_FALSE": packet(a_rows, 0.45),
        "TEMP_COPY_OPT_STATE_TRUE": packet(b_rows, 0.95),
    }


def test_error_and_harm_proxy_are_paired_and_transparent():
    # Both modes predict against exactly the SAME counterfactual formal update.
    actual = [-0.0002, 0.0003, -0.0001]
    left = [sample_row(10 + i, 20 + i, y, 0.0006, 0.99, 0.76) for i, y in enumerate(actual)]
    right = [
        sample_row(10 + i, 20 + i, y, y + 0.00001, 0.99 if y > 0 else 1.01,
                   0.76 if y > 0 else 1.0)
        for i, y in enumerate(actual)
    ]
    summary = aggregate_paired_risk([paired(left, right)])
    a = summary["modes"]["TEMP_COPY_OPT_STATE_FALSE"]
    b = summary["modes"]["TEMP_COPY_OPT_STATE_TRUE"]
    assert a["reference_observation_count"] == 3
    assert b["unique_anchor_reference_pairs"] == 3
    assert b["reference_absolute_error_mean_over_observations"] < a["reference_absolute_error_mean_over_observations"]
    assert b["mean_update_cosine_over_batches"] > a["mean_update_cosine_over_batches"]
    confusion = b["proxy_risk_thresholds"]["0.0001"]
    assert confusion["TP"] == 1 and confusion["TN"] == 2
    assert confusion["FP"] == confusion["FN"] == 0


def test_refuses_unpaired_sample_reference_ids():
    a = [sample_row(1, 3, -0.1, -0.01, 1.01, 1)]
    b = [sample_row(1, 4, -0.1, -0.01, 1.01, 1)]
    with pytest.raises(ValueError, match="not aligned"):
        aggregate_paired_risk([paired(a, b)])


def test_refuses_different_actual_formal_updates():
    a = [sample_row(1, 3, -0.1, -0.01, 1.01, 1)]
    b = [sample_row(1, 3, 0.1, -0.01, 1.01, 1)]
    with pytest.raises(ValueError, match="differs"):
        aggregate_paired_risk([paired(a, b)])


def test_rejects_nonfinite_risk_and_empty_batch():
    a = [sample_row(1, 3, 0.0, 0.0, float("nan"), 1)]
    with pytest.raises(ValueError, match="Non-finite"):
        aggregate_paired_risk([paired(a, a)])
    with pytest.raises(ValueError, match="without reliable"):
        aggregate_paired_risk([paired([], [])])
