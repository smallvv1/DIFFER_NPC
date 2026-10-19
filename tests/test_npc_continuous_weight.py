"""Independent numeric ablation tests for NPC's discontinuous weight rule.

These are CPU math tests, not a replacement for the PPU smoke test.
"""
import math

import pytest
import torch

from npc.losses import calculate_npc_weights


def _score_for_ratio(ratio: float, **kwargs):
    # Construct identical p/q ratios: (pre/post) = ratio, post=1.
    before = torch.tensor([ratio, ratio], dtype=torch.float64)
    after = torch.ones_like(before)
    r, w, direction = calculate_npc_weights(
        before, after, before, after, **kwargs
    )
    return r, w, direction


def test_original_is_exact_legacy_formula():
    before = torch.tensor([0.0, 0.25, 0.999, 1.0, 1.1], dtype=torch.float32)
    after = torch.ones_like(before)
    r, w, directions = calculate_npc_weights(before, after, before, after)
    expected_r = before.clamp_min(1e-8)
    expected_w = torch.where(expected_r < 1, expected_r.tanh(), torch.ones_like(expected_r))
    torch.testing.assert_close(r, expected_r)
    torch.testing.assert_close(w, expected_w)
    torch.testing.assert_close(directions[:, 0], expected_r)
    torch.testing.assert_close(directions[:, 1], expected_r)


def test_continuous_at_neutral_threshold():
    _, w_below, _ = _score_for_ratio(0.999999, policy="continuous", smooth_beta=1.0)
    _, w_exact, _ = _score_for_ratio(1.0, policy="continuous", smooth_beta=1.0)
    _, w_above, _ = _score_for_ratio(1.000001, policy="continuous", smooth_beta=1.0)
    assert torch.all(w_exact == 1)
    assert torch.all(w_above == 1)
    assert (w_exact - w_below).abs().max() < 2e-6

    _, w_original, _ = _score_for_ratio(0.999999, policy="original")
    assert float(w_original[0]) < 0.77


def test_smooth_policy_downweights_monotonically():
    weights = []
    for ratio in [0.0, 0.2, 0.5, 0.8, 0.99, 1.0, 1.2]:
        _, w, _ = _score_for_ratio(ratio, policy="continuous", smooth_beta=1.0)
        weights.append(float(w[0]))
    assert all(a <= b for a, b in zip(weights, weights[1:]))
    assert weights[-1] == weights[-2] == 1.0


def test_beta_control_and_bounds():
    for beta in [0.5, 1.0, 2.0, 4.0]:
        _, w, _ = _score_for_ratio(0.8, policy="continuous", smooth_beta=beta)
        assert torch.all((w >= 0) & (w <= 1))
        torch.testing.assert_close(w, torch.full_like(w, math.exp(-beta * 0.2)))


@pytest.mark.parametrize("beta", [-1.0, 0.0, float("nan"), float("inf")])
def test_bad_beta_is_rejected(beta):
    with pytest.raises(ValueError):
        _score_for_ratio(0.5, policy="continuous", smooth_beta=beta)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        _score_for_ratio(0.5, policy="unknown")


def test_weight_calculation_does_not_backpropagate_through_risk():
    before = torch.tensor([0.8, 1.2], requires_grad=True)
    after = torch.ones_like(before)
    _, weight, _ = calculate_npc_weights(
        before, after, before, after, policy="continuous", smooth_beta=1.0
    )
    assert not weight.requires_grad
