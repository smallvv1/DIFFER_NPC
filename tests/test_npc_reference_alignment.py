import pytest
import torch

from npc.audit_utils import align_valid_reference_losses


def test_non_prefix_valid_subbatch_aligns_to_correct_indices():
    # Full-batch PID sampler order 10,20,30,40. Only slots 1,3 have references.
    out = align_valid_reference_losses(
        torch.tensor([10, 20, 30, 40]),
        torch.tensor([1, 3]),
        [0.15, 0.25],
        [0.01, 0.02],
    )
    assert sorted(out) == [20, 40]
    assert out[20] == {"post_identity": 0.15, "post_reference": 0.01}
    assert out[40] == {"post_identity": 0.25, "post_reference": 0.02}


def test_all_valid_matches_identity_order():
    out = align_valid_reference_losses(
        torch.tensor([7, 3, 5]), torch.tensor([0, 1, 2]),
        [0.5, 0.6, 0.7], [0.05, 0.06, 0.07],
    )
    assert list(out) == [7, 3, 5]


@pytest.mark.parametrize("positions,p,q", [
    ([0, 2], [1.0], [0.1, 0.2]),
    ([0, 2], [1.0, 2.0], [0.1]),
    ([0, 4], [1.0, 2.0], [0.1, 0.2]),
])
def test_mismatched_or_out_of_range_input_fails(positions, p, q):
    with pytest.raises(ValueError):
        align_valid_reference_losses(torch.tensor([3, 4, 5]), torch.tensor(positions), p, q)


def test_duplicate_anchor_index_fails_instead_of_silent_overwrite():
    with pytest.raises(ValueError, match="Duplicate"):
        align_valid_reference_losses(
            torch.tensor([1, 1, 2]), torch.tensor([0, 1]),
            [0.3, 0.4], [0.01, 0.02],
        )
