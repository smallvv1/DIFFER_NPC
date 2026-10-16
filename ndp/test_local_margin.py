"""NDP 独立组件的最小正确性测试。"""

import torch

from ndp.local_margin import build_train_only_reference_library, local_margin_protection_loss


def test_reference_library_has_no_self_reference():
    items = [
        {"image_path": f"train/{i}.png", "pid": i // 2, "clothes_id": i % 2}
        for i in range(8)
    ]
    features = torch.eye(8, dtype=torch.float32)
    bank = build_train_only_reference_library(items, features, hard_k=2)
    assert bank["self_reference_count"] == 0
    for row in bank["items"]:
        if row["valid_positive"]:
            assert row["positive_index"] != row["index"]
            assert row["pid"] == row["positive_pid"]
            assert row["clothes_id"] != row["positive_clothes_id"]


def test_local_margin_loss_propagates_anchor_gradient():
    anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
    reference = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.99, 0.14]], dtype=torch.float32)
    positive = torch.tensor([1])
    negative = torch.tensor([[2]])
    loss, info = local_margin_protection_loss(anchor, reference, positive, negative, margin=0.2)
    loss.backward()
    assert info["valid_anchor_count"] == 1
    assert info["active_anchor_count"] == 1
    assert torch.isfinite(loss)
    assert anchor.grad is not None
    assert torch.isfinite(anchor.grad).all()
    assert float(anchor.grad.abs().sum()) > 0.0
