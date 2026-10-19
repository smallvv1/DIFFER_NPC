"""Temporarily copying SGD momentum must be opt-in, exact, and isolated.

CPU unit tests only; full PPU LTCC diagnostics are still required before
any claim that this improves NPC risk prediction or retrieval accuracy.
"""
import copy
from types import SimpleNamespace

import pytest
import torch

from npc.engine import NPCController


def make_controller(tmp_path, model, optimizer, copy_state=False):
    npc = SimpleNamespace(
        ENABLED=False, GMM_ENABLED=False,
        DYNAMIC_WEIGHT_ENABLED=False, MEMORY_LOSS_ENABLED=False,
        TEMP_COPY_OPT_STATE=copy_state, TEMP_LR_SCALE=1.0,
    )
    cfg = SimpleNamespace(MODEL=SimpleNamespace(NPC=npc))
    controller = NPCController(
        cfg, SimpleNamespace(train=[]), optimizer, tmp_path, torch.device("cpu"),
    )
    controller._ensure_temp_model(model)
    controller.temp_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return controller


def warmup(model, optimizer):
    x = torch.tensor([[1.5, -0.5], [0.7, 0.9]], dtype=torch.float32)
    optimizer.zero_grad()
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    assert all("momentum_buffer" in optimizer.state[p] for p in model.parameters())


def test_temp_optimizer_state_copy_is_exact_but_not_aliased(tmp_path):
    torch.manual_seed(123)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    warmup(model, optimizer)
    official_history = {
        p: optimizer.state[p]["momentum_buffer"].detach().clone()
        for p in model.parameters()
    }
    controller = make_controller(tmp_path, model, optimizer, copy_state=True)
    temp_optimizer = controller._make_temporary_optimizer(model)
    for p_main, p_temp in zip(model.parameters(), controller.temp_model.parameters()):
        a = optimizer.state[p_main]["momentum_buffer"]
        b = temp_optimizer.state[p_temp]["momentum_buffer"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.data_ptr() != b.data_ptr()
    for p_temp in controller.temp_model.parameters():
        p_temp.grad = torch.ones_like(p_temp)
    temp_optimizer.step()
    for p_main in model.parameters():
        torch.testing.assert_close(
            optimizer.state[p_main]["momentum_buffer"],
            official_history[p_main],
            rtol=0, atol=0,
        )
    controller.close()


def test_copied_sgd_state_matches_formal_step_with_identical_gradients(tmp_path):
    torch.manual_seed(125)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    warmup(model, optimizer)
    controller = make_controller(tmp_path, model, optimizer, copy_state=True)
    temp_optimizer = controller._make_temporary_optimizer(model)
    for p_main, p_temp in zip(model.parameters(), controller.temp_model.parameters()):
        grad = torch.arange(p_main.numel(), dtype=p_main.dtype).view_as(p_main) + 0.2
        p_main.grad = grad.clone()
        p_temp.grad = grad.clone()
    optimizer.step()
    temp_optimizer.step()
    for p_main, p_temp in zip(model.parameters(), controller.temp_model.parameters()):
        torch.testing.assert_close(p_main, p_temp, rtol=0, atol=1e-7)
    controller.close()


def test_default_temp_optimizer_resets_momentum(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    warmup(model, optimizer)
    controller = make_controller(tmp_path, model, optimizer, copy_state=False)
    temp_optimizer = controller._make_temporary_optimizer(model)
    assert all(p not in temp_optimizer.state for p in controller.temp_model.parameters())
    controller.close()


def test_copied_optimizer_state_is_sgd_only(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    controller = make_controller(tmp_path, model, optimizer, copy_state=True)
    with pytest.raises(RuntimeError, match="仅支持 SGD"):
        controller._make_temporary_optimizer(model)
    controller.close()
