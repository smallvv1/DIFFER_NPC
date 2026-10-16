import pytest
import torch

from loss.npc_weighting import reduce_weighted_ce


def test_legacy_weight_sum_matches_original():
    values = torch.tensor([1., 2., 4.], requires_grad=True)
    weights = torch.tensor([1., .6, .8])
    old = (values * weights).sum() / weights.sum()
    new = reduce_weighted_ce(values, weights, "weight_sum")
    torch.testing.assert_close(new, old)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
def test_legacy_weight_sum_preserves_loss_and_gradient(dtype):
    torch.manual_seed(1234)
    old_values = torch.randn(17, dtype=dtype, requires_grad=True)
    new_values = old_values.detach().clone().requires_grad_(True)
    weights = torch.rand(17, dtype=dtype)

    old = (old_values * weights).sum() / weights.sum().clamp_min(1e-12)
    new = reduce_weighted_ce(new_values, weights, "weight_sum")
    old.backward()
    new.backward()

    torch.testing.assert_close(new, old)
    torch.testing.assert_close(new_values.grad, old_values.grad)


def test_batch_mean_matches_npc_equation_nine():
    values = torch.tensor([1., 2., 4.], requires_grad=True)
    weights = torch.tensor([1., .6, .8])
    expected = (values * weights).sum() / 3
    actual = reduce_weighted_ce(values, weights, "batch_mean")
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(values.grad, weights / 3)


def test_unit_weights_match_both_reductions():
    values = torch.tensor([1., 2., 4.])
    unit = torch.ones_like(values)
    for reduction in ("weight_sum", "batch_mean"):
        torch.testing.assert_close(reduce_weighted_ce(values, unit, reduction), values.mean())


def test_invalid_reduction_and_shapes_rejected():
    with pytest.raises(ValueError):
        reduce_weighted_ce(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        reduce_weighted_ce(torch.ones(2), torch.ones(2), "unknown")
