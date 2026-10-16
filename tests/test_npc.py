import copy
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from yacs.config import CfgNode as CN

from npc.engine import NPCController
from npc.gmm import fit_clean_probabilities
from npc.losses import calculate_npc_weights, per_sample_identity_ce
from npc.memory import build_memory_references
from data.samplers import DistributedRandomIdentitySampler
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC


class TinyReID(torch.nn.Module):
    def __init__(self, input_dim=4, feature_dim=4, classes=2):
        super().__init__()
        self.embed = torch.nn.Linear(input_dim, feature_dim, bias=False)
        self.head = torch.nn.Linear(feature_dim, classes, bias=False)

    def forward(self, x, cam_label=None):
        feat = self.embed(x)
        score = {"cls_score": self.head(feat)}
        return score, [feat]


class TinyWriter:
    def add_scalar(self, *args, **kwargs):
        pass


class TinyReferenceDataset:
    """与 dataset.train 同序的训练集参考条目，模拟真实 ImageDataset。"""

    def __init__(self, records, batch):
        self.rows = []
        for index, record in enumerate(records):
            row = dict(record)
            row.update({
                "image": batch["image"][index],
                "pid": int(batch["pid"][index]),
                "camid": int(batch["camid"][index]),
                "clothes_id": int(batch["clothes_id"][index]),
            })
            self.rows.append(row)

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


def tiny_loss(score, feat, target, target_cam, caption_feature, clothes_ids, train_writer, step, npc_weights=None):
    per = F.cross_entropy(score["cls_score"], target, reduction="none")
    if npc_weights is None:
        return per.mean()
    return (per * npc_weights).sum() / npc_weights.sum().clamp_min(1e-12)


def make_cfg(tmp_path, dynamic=True, memory=True):
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.NPC = CN()
    cfg.MODEL.NPC.ENABLED = True
    cfg.MODEL.NPC.GMM_ENABLED = True
    cfg.MODEL.NPC.DYNAMIC_WEIGHT_ENABLED = dynamic
    cfg.MODEL.NPC.MEMORY_LOSS_ENABLED = memory
    cfg.MODEL.NPC.GMM_THRESHOLD = 0.90
    cfg.MODEL.NPC.GMM_MIN_SAMPLES = 2
    cfg.MODEL.NPC.MEMORY_UPDATE_FREQ = 1
    cfg.MODEL.NPC.TEMP_OPT_STEPS = 1
    cfg.MODEL.NPC.MEMORY_LOSS_WEIGHT = 0.1
    cfg.MODEL.NPC.DEBUG = True
    cfg.MODEL.IF_LABELSMOOTH = "off"
    cfg.SOLVER = CN()
    cfg.SOLVER.OPTIMIZER_NAME = "SGD"
    cfg.SOLVER.BASE_LR = 0.05
    cfg.SOLVER.BIAS_LR_FACTOR = 1.0
    cfg.SOLVER.WEIGHT_DECAY = 0.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0
    cfg.SOLVER.MOMENTUM = 0.0
    return cfg


def make_data():
    images = torch.tensor([
        [2.0, 0.0, 0.1, 0.0],
        [2.0, 0.1, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.1],
        [0.0, 2.0, 0.1, 0.0],
    ])
    records = [
        {"image_path": f"train/{i}.png", "pid": i // 2, "clothes_id": i % 2}
        for i in range(4)
    ]
    data = {
        "image": images,
        "pid": torch.tensor([0, 0, 1, 1]),
        "camid": torch.zeros(4, dtype=torch.long),
        "clothes_id": torch.tensor([0, 1, 0, 1]),
        "image_path": [x["image_path"] for x in records],
    }
    return records, data


def test_gmm_and_coverage_are_finite():
    result = fit_clean_probabilities([0.05, 0.07, 2.0, 2.2], threshold=0.9, min_samples=2)
    assert result.probabilities.shape == (4,)
    assert np.isfinite(result.probabilities).all()
    assert result.selected.sum() >= 1


def test_reference_bank_prefers_cross_clothing_and_never_changes_pid():
    records, _ = make_data()
    bank = build_memory_references(records, np.ones(4, dtype=bool))
    for i, ref in enumerate(bank.reference_metadata):
        assert ref["pid"] == ref["reference_pid"]
        assert ref["valid_reference"] is True
        assert bank.reference_indices[i] != i
    assert bank.coverage()["跨服装参考比例"] == 1.0
    assert int((bank.reference_indices == np.arange(len(records))).sum()) == 0


def test_reference_bank_marks_missing_reference_without_self_fallback():
    records = [{"image_path": "train/only.png", "pid": 0, "clothes_id": 0}]
    bank = build_memory_references(records, np.ones(1, dtype=bool))
    assert bank.reference_indices.tolist() == [-1]
    assert bank.valid_reference_mask.tolist() == [False]
    assert bank.reference_metadata[0]["valid_reference"] is False
    assert bank.reference_metadata[0]["invalid_reason"] == "no_reliable_same_pid_reference"
    assert bank.coverage()["有效参考数"] == 0
    assert bank.coverage()["无效参考数"] == 1


def test_reference_bank_does_not_use_unreliable_same_pid_sample():
    records = [
        {"image_path": "train/a.png", "pid": 0, "clothes_id": 0},
        {"image_path": "train/b.png", "pid": 0, "clothes_id": 1},
    ]
    bank = build_memory_references(records, np.array([False, False]))
    assert np.all(bank.reference_indices == -1)
    assert not bank.valid_reference_mask.any()


def test_formula_handles_two_directions_independently():
    # 样本0：两项均恶化，权重应低于1；样本1：一项改善、一项恶化，
    # 不能把“p_ratio<1”简单等同于最终权重低于1。
    ratio, weights, direction = calculate_npc_weights(
        torch.tensor([1.0, 1.0]),
        torch.tensor([2.0, 0.5]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([2.0, 2.0]),
    )
    assert torch.allclose(direction[:, 0], torch.tensor([0.5, 2.0]))
    assert torch.allclose(direction[:, 1], torch.tensor([0.5, 0.5]))
    assert weights[0] < 1.0
    assert torch.isfinite(ratio).all()
    assert torch.isfinite(weights).all()


def test_controller_temp_update_does_not_touch_base_or_optimizer(tmp_path):
    records, batch = make_data()
    model = TinyReID()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    cfg = make_cfg(tmp_path, dynamic=True, memory=True)
    cfg.MODEL.NPC.GMM_THRESHOLD = 0.0
    controller = NPCController(
        cfg,
        SimpleNamespace(train=records),
        optimizer,
        tmp_path,
        torch.device("cpu"),
        reference_dataset=TinyReferenceDataset(records, batch),
    )
    full_loader = [batch]
    controller.initialize_memory(model, full_loader, tiny_loss, epoch=1)
    model.train()
    score, feat = model(batch["image"], cam_label=batch["camid"])
    model_state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer_state_before = copy.deepcopy(optimizer.state_dict())
    weights, stats = controller.preaware_weights(
        model, tiny_loss, batch, score, feat, batch["pid"], batch["camid"], None,
        batch["clothes_id"], epoch=1, iteration=1,
    )
    assert stats["finite"] is True
    assert torch.isfinite(weights).all()
    for key, value in model.state_dict().items():
        assert torch.equal(value, model_state_before[key])
    assert optimizer.state_dict() == optimizer_state_before
    assert controller.temp_model is not None
    assert any(
        not torch.equal(value, model_state_before[key])
        for key, value in controller.temp_model.state_dict().items()
    )
    controller.close()


def test_preaware_loss_is_measured_on_valid_reference_entry(tmp_path):
    """NPC 的 p/q 信号必须是可靠参考条目的 pre/post 前向损失。"""
    records, batch = make_data()
    model = TinyReID()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    cfg = make_cfg(tmp_path, dynamic=True, memory=True)
    cfg.MODEL.NPC.GMM_THRESHOLD = 0.0
    reference_dataset = TinyReferenceDataset(records, batch)
    controller = NPCController(
        cfg,
        SimpleNamespace(train=records),
        optimizer,
        tmp_path,
        torch.device("cpu"),
        reference_dataset=reference_dataset,
    )
    controller.initialize_memory(model, [batch], tiny_loss, epoch=1)
    model.train()
    score, feat = model(batch["image"], cam_label=batch["camid"])
    weights, stats = controller.preaware_weights(
        model, tiny_loss, batch, score, feat, batch["pid"], batch["camid"], None,
        batch["clothes_id"], epoch=1, iteration=1,
    )
    assert stats["preaware_loss_object"] == "valid_reference_entries"
    assert stats["valid_reference_count"] == len(records)
    assert torch.isfinite(weights).all()
    model.eval()
    with torch.no_grad():
        ref_score, _ = model(batch["image"][1:2], cam_label=batch["camid"][1:2])
        expected = F.cross_entropy(ref_score["cls_score"], batch["pid"][1:2]).item()
    model.train()
    with open(tmp_path / "npc_diagnostics" / "npc_weights.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row0 = next(row for row in rows if int(row["sample_index"]) == 0)
    assert float(row0["reference_index"]) == 1
    assert float(row0["pre_identity"]) == pytest.approx(expected, rel=1e-5, abs=1e-6)
    controller.close()


def test_n1_does_not_compute_memory_loss_and_n2_does_not_update_temp(tmp_path):
    records, batch = make_data()
    model = TinyReID()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    cfg_n1 = make_cfg(tmp_path / "n1", dynamic=True, memory=False)
    c1 = NPCController(cfg_n1, SimpleNamespace(train=records), optimizer, tmp_path / "n1", torch.device("cpu"))
    c1.initialize_memory(model, [batch], tiny_loss, 1)
    _, feat = model(batch["image"], batch["camid"])
    assert c1.memory_supervision(feat, batch)[0].item() == 0.0
    c1.close()

    cfg_n2 = make_cfg(tmp_path / "n2", dynamic=False, memory=True)
    c2 = NPCController(cfg_n2, SimpleNamespace(train=records), optimizer, tmp_path / "n2", torch.device("cpu"))
    c2.initialize_memory(model, [batch], tiny_loss, 1)
    _, feat = model(batch["image"], batch["camid"])
    mem_loss, per = c2.memory_supervision(feat, batch)
    assert torch.isfinite(mem_loss)
    assert c2.temp_model is None
    c2.close()


def test_invalid_references_default_to_one_and_skip_npc_signal(tmp_path):
    records, batch = make_data()
    model = TinyReID()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    cfg = make_cfg(tmp_path, dynamic=True, memory=True)
    cfg.MODEL.NPC.GMM_THRESHOLD = 1.1
    controller = NPCController(cfg, SimpleNamespace(train=records), optimizer, tmp_path, torch.device("cpu"))
    controller.initialize_memory(model, [batch], tiny_loss, epoch=1)
    score, feat = model(batch["image"], batch["camid"])
    weights, stats = controller.preaware_weights(
        model, tiny_loss, batch, score, feat, batch["pid"], batch["camid"], None,
        batch["clothes_id"], epoch=1, iteration=1,
    )
    assert torch.equal(weights, torch.ones_like(weights))
    assert stats["valid_reference_count"] == 0
    assert controller.temp_model is None
    memory_loss, per_sample = controller.memory_supervision(feat, batch)
    assert memory_loss.item() == 0.0
    assert torch.equal(per_sample, torch.zeros_like(per_sample))
    controller.close()


def test_ltcc_general_and_clothes_changing_use_different_filters():
    features = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    ])
    pids = torch.tensor([1, 2, 1, 1, 2, 2])
    camids = torch.tensor([0, 0, 1, 2, 1, 2])
    clothes = torch.tensor([0, 0, 0, 1, 0, 1])
    paths = [f"eval/{i}.png" for i in range(len(features))]

    general = R1_mAP_eval(num_query=2, max_rank=4, feat_norm=True)
    general.reset()
    general.update((features, pids, camids, clothes, paths))
    general_cmc, general_map, *_ = general.compute()

    changed = R1_mAP_eval_LTCC(num_query=2, max_rank=4, feat_norm=True)
    changed.reset()
    changed.update((features, pids, camids, clothes, paths))
    changed_cmc, changed_map, *_ = changed.compute()

    assert general_cmc[0] == pytest.approx(1.0)
    assert changed_cmc[0] < general_cmc[0]
    assert changed_map < general_map


def test_identity_sampler_keeps_multiple_pids_in_every_triplet_batch():
    records = [
        {"pid": pid, "image_path": f"train/{pid}_{image}.png"}
        for pid in range(8)
        for image in range(4)
    ]
    sampler = DistributedRandomIdentitySampler(
        records,
        num_instances=2,
        batch_size=8,
        num_replicas=1,
        rank=0,
        seed=1234,
    )
    sampler.set_epoch(1)
    indices = list(iter(sampler))
    assert len(indices) == len(sampler)
    assert len(indices) % 8 == 0
    for start in range(0, len(indices), 8):
        pids = [records[index]["pid"] for index in indices[start:start + 8]]
        assert len(set(pids)) == 4
        assert sorted(pids.count(pid) for pid in set(pids)) == [2, 2, 2, 2]
