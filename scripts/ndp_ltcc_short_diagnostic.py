#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIFFER + GPC 上的 NDP LTCC 小规模 train-only 诊断。

本脚本是独立的诊断入口：

* 读取已冻结的 DIFFER+GPC E60 checkpoint；
* 只从 LTCC train split 提取特征并建立 NDP 局部参考库；
* 使用严格匹配的 GPC 更新序列，与 GPC+NDP 局部间隔保护探针比较；
* 只做 16/32/64 步短程实验，不启动正式 60 epoch，不改写 checkpoint。

NDP 的风险预测和任何动态权重均不进入优化图。本脚本也不修改 GPC
selector、pair budget、GPC loss 或 GPC lambda。
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
import sys
import subprocess
import time
import types
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import cfg as GLOBAL_CFG  # noqa: E402
from data import build_dataset, build_img_transforms  # noqa: E402
from data.dataset_loader import ImageDataset  # noqa: E402
from loss import make_loss  # noqa: E402
from model import build_model  # noqa: E402
from ndp.local_margin import (  # noqa: E402
    build_train_only_reference_library,
    local_margin_protection_loss,
    summarize_local_geometry,
)
# 当前工作区没有正式测试阶段才会使用的 processor 包。历史 GPC 入口在
# import 阶段无条件导入它；这里仅为复用 train-only 的 sampler/selector/
# forward_loss 提供一个不可调用的测试占位，不会进入正式 evaluator。
if "processor.processor" not in sys.modules:
    _processor_package = types.ModuleType("processor")
    _processor_module = types.ModuleType("processor.processor")
    _processor_module.test = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("NDP 诊断禁止调用正式 query/gallery evaluator")
    )
    _processor_package.processor = _processor_module
    sys.modules["processor"] = _processor_package
    sys.modules["processor.processor"] = _processor_module
from scripts.train_final_gpc import (  # noqa: E402
    FinalBatchSampler,
    MetricCapture,
    build_train_loader,
    build_train_only_pair_bank,
    batch_entries,
    cfg_for,
    forward_loss,
    make_optimizer,
    set_seed as gpc_set_seed,
    select_gpc_pairs,
    state_hash,
    zero_relation,
)


CHECKPOINT = ROOT / "outputs/final_confirmatory_gpc/ltcc/seed1234/gpc/model_final.pth"
EXPECTED_CHECKPOINT_SHA256 = "51e94a312d09d4e172afb8c04d8988e6cf9ad3b8cf4c82bbdf003562a6f6da26"
HARD_K = 10
NDP_MARGIN = 0.05
NDP_LAMBDA_PROBE = 0.1
SHORT_STEPS = (16, 32, 64)
REPEATS = 3
EVAL_ANCHORS = 96
SHORT_BATCH_SIZE = 8
SHORT_STEPS_PER_EPOCH = max(SHORT_STEPS)
DEVICE_NAME = "PPU-ZW810E"


def now_bj() -> dt.datetime:
    return dt.datetime.now(ZoneInfo("Asia/Shanghai"))


def timestamp() -> str:
    return now_bj().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe(row.get(key, "")) for key in fields})


class ChineseLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8")

    def info(self, message: str) -> None:
        line = f"[{now_bj().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def runtime_gate() -> tuple[torch.device, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("PPU 运行拒绝：torch.cuda.is_available() 为 False")
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if not names or any(name != DEVICE_NAME for name in names):
        raise RuntimeError(f"发现的设备不是预期 PPU-ZW810E：{names}")
    torch.cuda.set_device(0)
    return torch.device("cuda:0"), {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_count": count,
        "device_names": names,
        "selected_device": "cuda:0",
        "cpu_model_fallback": False,
        "note": "本轮为单卡小规模诊断；当前环境只暴露一张 PPU。",
    }


def cfg_diagnostic() -> object:
    cfg = cfg_for("ltcc", ROOT / "outputs" / "_ndp_diagnostic_placeholder")
    cfg.defrost()
    # 当前工作区保留了 train.npz，但没有历史正式运行所需的
    # train_caption_summary_biometric.json。基础 LTCC 配置的 false 会直接
    # 使用同一 train.npz 逐图 caption；这只保证本轮短程对照可执行，且在
    # manifest 中明确记录为复现限制，不宣称等同于正式 GPC 重训。
    cfg.DATA.SUMMERY_TEXT = False
    cfg.DATA.NUM_WORKERS = 0
    cfg.DATA.BATCH_SIZE = SHORT_BATCH_SIZE
    cfg.SOLVER.MAX_EPOCHS = 60
    cfg.freeze()
    return cfg


def load_dataset_and_model(cfg, device: torch.device, state: dict[str, torch.Tensor]):
    dataset = build_dataset(cfg)
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return dataset, model


def copy_items(items: list[dict]) -> list[dict]:
    result = []
    for item in items:
        row = dict(item)
        if isinstance(row.get("caption_feature"), np.ndarray):
            row["caption_feature"] = row["caption_feature"].copy()
        result.append(row)
    return result


def extract_features(model, items: list[dict], transform, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    dataset = ImageDataset(copy_items(items), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    was_training = model.training
    model.eval()
    features = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=False)
            camids = batch["camid"].to(device, non_blocking=False)
            with amp.autocast(enabled=True):
                current = model(images, cam_label=camids)
            if isinstance(current, (list, tuple)):
                current = current[0]
            features.append(current.detach().float().cpu())
    if was_training:
        model.train()
    if not features:
        raise RuntimeError("特征提取为空")
    return torch.cat(features, dim=0)


def extract_index_features(model, all_items: list[dict], indices: list[int], transform, device: torch.device) -> torch.Tensor:
    return extract_features(model, [all_items[index] for index in indices], transform, device)


def collect_fixed_training_batches(cfg, dataset, seed: int, log: ChineseLog) -> list[dict]:
    """用同一个固定 CPU batch 序列给 GPC 和 GPC+NDP 两个 arm。"""
    gpc_set_seed(seed)
    sampler = FinalBatchSampler(
        dataset.train,
        SHORT_BATCH_SIZE,
        int(cfg.DATA.NUM_INSTANCES),
        SHORT_STEPS_PER_EPOCH,
        seed,
        0,
    )
    sampler.prepare(1)
    loader = build_train_loader(cfg, dataset, sampler)
    batches = []
    iterator = iter(loader)
    for _ in range(SHORT_STEPS_PER_EPOCH):
        batch = next(iterator)
        kept = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                kept[key] = value.clone()
            elif isinstance(value, list):
                kept[key] = list(value)
            else:
                kept[key] = copy.deepcopy(value)
        batches.append(kept)
    if hasattr(loader, "shutdown"):
        loader.shutdown()
    log.info(f"固定训练 batch 序列已生成：seed={seed}，步数={len(batches)}，batch_size={SHORT_BATCH_SIZE}。")
    return batches


def batch_reference_indices(batch: dict, library: dict, path_to_index: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    positives = []
    negatives = []
    for path in batch["image_path"]:
        index = int(path_to_index[str(path)])
        row = library["items"][index]
        positives.append(int(row["positive_index"]) if row["valid_positive"] else -1)
        values = list(row["negative_indices"])
        values = values[:HARD_K] + [-1] * max(0, HARD_K - len(values))
        negatives.append(values[:HARD_K])
    return torch.tensor(positives, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)


def diagnostic_gpc_forward_loss(model, loss_fn, batch, device, gpc_bank, step):
    """与历史 GPC ``forward_loss`` 相同的计算顺序，但关闭 AMP。

    历史入口把 ``amp.autocast(enabled=True)`` 写死，导致本轮已经观察到
    PPU 第 3 步非有限梯度。为了验证机制本身，短程恢复只在这里使用
    FP32；GPC selector、base loss、pair reduction 和 lambda 均直接复用
    同一实现/常量。该函数不替换历史训练入口。
    """
    writer = MetricCapture()
    images = batch["image"].to(device, non_blocking=False)
    pids = batch["pid"].to(device, non_blocking=False)
    camids = batch["camid"].to(device, non_blocking=False)
    clothes = batch["clothes_id"].to(device, non_blocking=False)
    captions = batch["caption_feature"].to(device, non_blocking=False)
    score, feat = model(images, cam_label=camids)
    base_total = loss_fn(
        score,
        feat,
        pids,
        camids,
        captions,
        clothes_ids=clothes,
        train_writer=writer,
        step=step,
    )
    representation = F.normalize(feat[0].float(), dim=-1)
    entries = batch_entries(batch)
    selected, relation_info = select_gpc_pairs(entries, gpc_bank, 1, 0, step, 1234)
    if selected:
        relation = torch.stack([
            1.0 - torch.sum(representation[left] * representation[right])
            for left, right, _image, _state in selected
        ]).mean()
    else:
        relation = zero_relation(representation)
    total = base_total + 0.1 * relation
    components = getattr(loss_fn, "last_components", {})
    cls_score = score["cls_score"] if isinstance(score, dict) else score
    return {
        "total": total,
        "base_total": base_total,
        "ce": components.get("ce", total * 0.0),
        "triplet": components.get("triplet", total * 0.0),
        "bio": components.get("bio", total * 0.0),
        "gpc": relation,
        "relation_info": relation_info,
        "writer": writer,
        "forward_device": str(feat[0].device),
        "gpc_device": str(relation.device),
        "id_acc": (cls_score.argmax(dim=-1) == pids).float().mean(),
    }


def gradient_l2(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        total += float(torch.sum(value * value).detach().cpu())
    return math.sqrt(max(total, 0.0))


def loss_gradient_l2(loss: torch.Tensor, parameters) -> float:
    grads = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    total = 0.0
    for grad in grads:
        if grad is not None:
            value = grad.detach().float()
            total += float(torch.sum(value * value).detach().cpu())
    return math.sqrt(max(total, 0.0))


def metric_rows_for_indices(model, all_items, anchor_indices, library, transform, device):
    eval_indices = []
    for anchor in anchor_indices:
        row = library["items"][anchor]
        eval_indices.extend([int(anchor), int(row["positive_index"]), *[int(v) for v in row["negative_indices"]]])
    eval_indices = sorted(set(index for index in eval_indices if index >= 0))
    feature_matrix = extract_index_features(model, all_items, eval_indices, transform, device)
    position = {index: pos for pos, index in enumerate(eval_indices)}
    anchors = torch.stack([feature_matrix[position[int(index)]] for index in anchor_indices])
    positives = torch.stack([feature_matrix[position[int(library["items"][index]["positive_index"])] ] for index in anchor_indices])
    negatives = torch.stack(
        [torch.stack([feature_matrix[position[int(negative)]] for negative in library["items"][index]["negative_indices"]]) for index in anchor_indices]
    )
    return summarize_local_geometry(anchors, positives, negatives)


def initial_geometry(features: torch.Tensor, anchor_indices: list[int], library: dict) -> dict:
    anchors = torch.stack([features[index] for index in anchor_indices])
    positives = torch.stack([features[int(library["items"][index]["positive_index"])] for index in anchor_indices])
    negatives = torch.stack([
        torch.stack([features[int(negative)] for negative in library["items"][index]["negative_indices"]])
        for index in anchor_indices
    ])
    return summarize_local_geometry(anchors, positives, negatives)


def snapshot_geometry(model, all_items, anchor_indices, library, transform, device, initial: dict, step: int, arm: str, repeat: int, output_dir: Path):
    current = initial_geometry(library["feature_tensor"], anchor_indices, library) if step == 0 else metric_rows_for_indices(model, all_items, anchor_indices, library, transform, device)
    current = dict(current)
    current.update({"step": int(step), "arm": arm, "repeat": int(repeat)})
    if step > 0:
        current["delta_local_margin_mean"] = current["local_margin_mean"] - initial["local_margin_mean"]
        current["delta_cross_clothes_positive_similarity_mean"] = current["cross_clothes_positive_similarity_mean"] - initial["cross_clothes_positive_similarity_mean"]
        current["delta_positive_rank1_rate"] = current["positive_rank1_rate"] - initial["positive_rank1_rate"]
        current["delta_impostor_intrusion_rate"] = current["impostor_intrusion_rate"] - initial["impostor_intrusion_rate"]
    write_json(output_dir / "geometry" / f"{arm}_repeat{repeat}_step{step:03d}.json", current)
    return current


def run_one_arm(
    cfg,
    state,
    dataset,
    batches,
    library,
    path_to_index,
    anchor_indices,
    initial,
    transform_test,
    device,
    arm: str,
    repeat: int,
    output_dir: Path,
    log: ChineseLog,
):
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device).train()
    loss_fn, center = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, _ = make_optimizer(cfg, model, center)
    gpc_bank = build_train_only_pair_bank(dataset, "ltcc")
    reference_device = library["feature_tensor"].to(device)
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rows = []
    snapshots = []
    total_skips = 0
    total_nonfinite = 0
    total_optimizer_updates = 0
    ndp_active = 0
    ndp_valid = 0
    gpc_active = 0
    component_probe_done = False
    log.info(f"开始短程 arm：{arm}，repeat={repeat}；GPC 源码路径保持不变。")
    for step, batch in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        result = diagnostic_gpc_forward_loss(model, loss_fn, batch, device, gpc_bank, step)
        positive_indices, negative_indices = batch_reference_indices(batch, library, path_to_index)
        positive_indices = positive_indices.to(device)
        negative_indices = negative_indices.to(device)
        # 从同一 batch 重新取得 NDP anchor 特征。这样不改动现有
        # forward_loss 的 GPC 代码，GPC selector、pair-budget 和 loss 图保持原样。
        with amp.autocast(enabled=True):
            _, ndp_feat = model(batch["image"].to(device), cam_label=batch["camid"].to(device))
        ndp_loss, ndp_info = local_margin_protection_loss(
            ndp_feat[0],
            reference_device,
            positive_indices,
            negative_indices,
            margin=NDP_MARGIN,
        )
        total = result["total"] if arm == "gpc" else result["total"] + NDP_LAMBDA_PROBE * ndp_loss
        finite_loss = bool(torch.isfinite(total.detach()).item())
        if not finite_loss:
            total_nonfinite += 1
            raise RuntimeError(f"{arm} repeat={repeat} step={step} 总损失非有限")
        gpc_active += int(bool(result["relation_info"]["active"]))
        ndp_active += int(ndp_info["active_anchor_count"])
        ndp_valid += int(ndp_info["valid_anchor_count"])
        gpc_gradient = None
        ndp_gradient = None
        if not component_probe_done or step in SHORT_STEPS:
            gpc_gradient = loss_gradient_l2(result["gpc"], model_parameters)
            ndp_gradient = loss_gradient_l2(ndp_loss, model_parameters)
            component_probe_done = True
        total.backward()
        gradient_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad.detach()).all().item())
            for parameter in model_parameters
        )
        if not gradient_finite:
            total_nonfinite += 1
            raise RuntimeError(f"{arm} repeat={repeat} step={step} 梯度非有限")
        grad_norm = gradient_l2(model_parameters)
        before_hash = None
        if step == 1:
            before_hash = sha256_bytes(torch.cat([parameter.detach().float().reshape(-1)[:16].cpu() for parameter in model_parameters[:8]]).numpy().tobytes())
        optimizer.step()
        inf = 0.0
        total_optimizer_updates += 1
        rows.append(
            {
                "arm": arm,
                "repeat": repeat,
                "step": step,
                "gpc_total_loss": float(result["total"].detach().cpu()),
                "gpc_base_loss": float(result["base_total"].detach().cpu()),
                "gpc_relation_loss": float(result["gpc"].detach().cpu()),
                "ndp_loss": float(ndp_loss.detach().cpu()),
                "ndp_applied_loss": float((NDP_LAMBDA_PROBE * ndp_loss).detach().cpu()),
                "effective_total_loss": float(total.detach().cpu()),
                "ndp_valid_anchor_count": ndp_info["valid_anchor_count"],
                "ndp_active_anchor_count": ndp_info["active_anchor_count"],
                "ndp_positive_similarity": ndp_info["positive_similarity"],
                "ndp_hard_negative_similarity": ndp_info["hard_negative_similarity"],
                "gpc_active": bool(result["relation_info"]["active"]),
                "gpc_selected_count": int(result["relation_info"]["selected_count"]),
                "gradient_l2": grad_norm,
                "gpc_gradient_l2_probe": gpc_gradient,
                "ndp_gradient_l2_probe": ndp_gradient,
                "amp_scale": 1.0,
                "found_inf": inf,
                "optimizer_update": bool(inf <= 0),
                "finite_loss": finite_loss,
                "finite_gradient": gradient_finite,
                "forward_device": result["forward_device"],
                "gpc_device": result["gpc_device"],
                "ndp_device": str(ndp_loss.device),
                "before_parameter_probe_hash": before_hash,
            }
        )
        if step in SHORT_STEPS:
            snapshot = snapshot_geometry(model, dataset.train, anchor_indices, library, transform_test, device, initial, step, arm, repeat, output_dir)
            snapshots.append(snapshot)
            log.info(
                f"{arm} repeat={repeat} step={step}：NDP有效={ndp_valid}/{step * len(batch['pid'])}，"
                f"本步NDP激活={ndp_info['active_anchor_count']}，局部间隔={snapshot['local_margin_mean']:.6f}，"
                f"正样本Rank-1={snapshot['positive_rank1_rate']:.4f}，侵入率={snapshot['impostor_intrusion_rate']:.4f}，"
                f"optimizer更新={total_optimizer_updates}，跳步={total_skips}。"
            )
    write_csv(output_dir / "step_metrics" / f"{arm}_repeat{repeat}.csv", rows)
    summary = {
        "arm": arm,
        "repeat": repeat,
        "steps": SHORT_STEPS_PER_EPOCH,
        "amp_skipped_steps": total_skips,
        "nonfinite_events": total_nonfinite,
        "optimizer_updates": total_optimizer_updates,
        "gpc_active_batches": gpc_active,
        "ndp_valid_anchor_observations": ndp_valid,
        "ndp_active_anchor_observations": ndp_active,
        "ndp_activation_rate_over_valid": float(ndp_active / max(ndp_valid, 1)),
        "snapshots": snapshots,
        "all_finite": total_nonfinite == 0,
        "all_on_ppu": all(row["ndp_device"].startswith("cuda") and row["forward_device"].startswith("cuda") for row in rows),
    }
    write_json(output_dir / "summaries" / f"{arm}_repeat{repeat}.json", summary)
    del model, loss_fn, center, optimizer, reference_device
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if np.std(xa) == 0 or np.std(ya) == 0:
        return None
    rx = np.argsort(np.argsort(xa))
    ry = np.argsort(np.argsort(ya))
    return float(np.corrcoef(rx, ry)[0, 1])


def auc_binary(scores: list[float], labels: list[int]) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    comparisons = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(comparisons / (len(pos) * len(neg)))


def analyze_risk(results: list[dict], initial: dict, output_dir: Path, log: ChineseLog) -> dict:
    """只做离线风险诊断，不把预测分数送回训练权重。"""
    rows = []
    for result in results:
        if result["arm"] != "gpc":
            continue
        for snapshot in result["snapshots"]:
            margin_values = snapshot.get("local_margin_values", [])
            initial_values = initial.get("local_margin_values", [])
            if len(margin_values) != len(initial_values):
                continue
            delta = np.asarray(margin_values) - np.asarray(initial_values)
            for index, value in enumerate(delta.tolist()):
                rows.append({
                    "arm": result["arm"],
                    "repeat": result["repeat"],
                    "step": snapshot["step"],
                    "anchor_position": index,
                    "pre_margin": float(initial_values[index]),
                    "post_margin_delta": float(value),
                    "deteriorated": int(value < 0),
                })
    if not rows:
        result = {"status": "INCONCLUSIVE", "reason": "没有足够的固定 anchor 结构变化记录"}
    else:
        scores = [-row["pre_margin"] for row in rows]
        labels = [row["deteriorated"] for row in rows]
        result = {
            "status": "DIAGNOSTIC_ONLY",
            "rows": len(rows),
            "pre_margin_to_deterioration_spearman": spearman(scores, labels),
            "pre_margin_risk_auc": auc_binary(scores, labels),
            "risk_weighting_enabled": False,
            "interpretation": "仅检查更新前局部间隔能否预测后续恶化；未将预测结果用于任何优化权重。",
        }
    write_csv(output_dir / "risk" / "pre_update_risk_rows.csv", rows)
    write_json(output_dir / "risk" / "risk_diagnostic.json", result)
    log.info(f"更新前风险预测诊断完成：状态={result['status']}；风险引导权重启用=False。")
    return result


def aggregate(results: list[dict], initial: dict, output_dir: Path, log: ChineseLog) -> dict:
    rows = []
    for result in results:
        for snapshot in result["snapshots"]:
            row = {
                "arm": result["arm"],
                "repeat": result["repeat"],
                "step": snapshot["step"],
                "local_margin_mean": snapshot["local_margin_mean"],
                "delta_local_margin_mean": snapshot.get("delta_local_margin_mean", 0.0),
                "cross_clothes_positive_similarity_mean": snapshot["cross_clothes_positive_similarity_mean"],
                "delta_cross_clothes_positive_similarity_mean": snapshot.get("delta_cross_clothes_positive_similarity_mean", 0.0),
                "positive_rank1_rate": snapshot["positive_rank1_rate"],
                "delta_positive_rank1_rate": snapshot.get("delta_positive_rank1_rate", 0.0),
                "impostor_intrusion_rate": snapshot["impostor_intrusion_rate"],
                "delta_impostor_intrusion_rate": snapshot.get("delta_impostor_intrusion_rate", 0.0),
                "mean_rank": snapshot["rank_mean"],
            }
            rows.append(row)
    write_csv(output_dir / "geometry" / "all_snapshot_metrics.csv", rows)
    summary = {}
    for arm in sorted({row["arm"] for row in rows}):
        summary[arm] = {}
        for step in SHORT_STEPS:
            selected = [row for row in rows if row["arm"] == arm and row["step"] == step]
            if not selected:
                continue
            summary[arm][str(step)] = {
                "count": len(selected),
                "delta_margin_mean": float(np.mean([row["delta_local_margin_mean"] for row in selected])),
                "delta_margin_std": float(np.std([row["delta_local_margin_mean"] for row in selected])),
                "delta_cross_positive_mean": float(np.mean([row["delta_cross_clothes_positive_similarity_mean"] for row in selected])),
                "delta_rank1_mean": float(np.mean([row["delta_positive_rank1_rate"] for row in selected])),
                "delta_intrusion_mean": float(np.mean([row["delta_impostor_intrusion_rate"] for row in selected])),
                "margin_signal_to_repeat_noise": float(abs(np.mean([row["delta_local_margin_mean"] for row in selected])) / max(np.std([row["delta_local_margin_mean"] for row in selected]), 1e-12)),
            }
    write_json(output_dir / "geometry" / "aggregate_summary.json", summary)
    log.info(f"局部几何聚合完成：已保存 {len(rows)} 条 16/32/64 步快照记录。")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="DIFFER+GPC 上的 NDP LTCC 短程诊断")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    run_time = timestamp()
    output_dir = args.output_root or ROOT / "outputs" / f"NDP_LTCC_DIAGNOSTIC_{run_time}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log = ChineseLog(output_dir / f"NDP_LTCC短程诊断_{run_time}.log")
    start = time.time()
    exit_code = 1
    try:
        device, runtime = runtime_gate()
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(CHECKPOINT)
        checkpoint_sha = sha256_file(CHECKPOINT)
        if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(f"GPC checkpoint 哈希与已冻结 manifest 不一致：{checkpoint_sha}")
        log.info(f"输出目录（北京时间）：{output_dir}")
        log.info(f"PPU 运行时：{runtime}")
        log.info(f"GPC E60 checkpoint 已冻结且只读：{CHECKPOINT}；SHA256={checkpoint_sha}")
        log.info("本轮不读取 query/gallery 特征，不启动正式 60 epoch，不启动风险引导权重。")

        cfg = cfg_diagnostic()
        cfg_text = cfg.dump()
        (output_dir / "配置").mkdir(parents=True, exist_ok=True)
        (output_dir / "配置" / "ltcc_gpc_frozen.yml").write_text(cfg_text, encoding="utf-8")
        source_paths = [
            ROOT / "scripts/train_final_gpc.py",
            ROOT / "loss/make_loss.py",
            ROOT / "model/build.py",
            ROOT / "model/eva_meta_bio.py",
            ROOT / "data/__init__.py",
            ROOT / "data/dataset_loader.py",
            ROOT / "data/datasets/ltcc.py",
            ROOT / "ndp/local_margin.py",
            ROOT / "scripts/ndp_ltcc_short_diagnostic.py",
            ROOT / "configs/ltcc/eva02_l_bio.yml",
        ]
        source_hashes = {str(path): sha256_file(path) for path in source_paths if path.is_file()}
        state = torch.load(CHECKPOINT, map_location="cpu")
        state_tensor_hash = state_hash(state)
        write_json(output_dir / "代码与检查点冻结.json", {
            "created_beijing": now_bj().isoformat(),
            "project_root": str(ROOT),
            "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_sha, "state_hash": state_tensor_hash, "epoch": 60, "read_only": True},
            "source_hashes": source_hashes,
            "gpc_contract": "GPC selector/pair-budget/loss/lambda=0.1 unchanged; NDP only added outside forward_loss.",
            "formal_training_started": False,
        })

        log.info("开始加载 LTCC；后续所有参考特征只取 dataset.train。")
        dataset, model = load_dataset_and_model(cfg, device, state)
        transform_train, transform_test = build_img_transforms(cfg)
        train_items = list(dataset.train)
        train_paths = {str(item["image_path"]) for item in train_items}
        non_train_paths = {str(item["image_path"]) for item in list(dataset.query) + list(dataset.gallery)}
        if train_paths & non_train_paths:
            raise RuntimeError("LTCC train 与 query/gallery 路径发生重叠")
        log.info(f"LTCC train-only 样本：{len(train_items)}；train PID：{dataset.num_train_pids}；query/gallery 未进入参考库。")

        log.info("在 PPU 上提取完整 LTCC train-only checkpoint 特征，建立 NDP 参考库。")
        library_features = extract_features(model, train_items, transform_test, device, batch_size=64)
        library = build_train_only_reference_library(train_items, library_features, hard_k=HARD_K)
        library["feature_tensor"] = library_features
        library_json = dict(library)
        library_json.pop("feature_tensor", None)
        write_json(output_dir / "参考库" / "ltcc_train_only_reference_library.json", library_json)
        np.save(output_dir / "参考库" / "ltcc_train_only_features.npy", library_features.numpy())
        write_json(output_dir / "参考库" / "参考库审计.json", {
            "train_count": len(train_items),
            "query_count_not_used": len(dataset.query),
            "gallery_count_not_used": len(dataset.gallery),
            "train_only": True,
            "positive_coverage": library["positive_coverage"],
            "full_reference_coverage": library["full_reference_coverage"],
            "self_reference_count": library["self_reference_count"],
            "hard_k": HARD_K,
            "positive_rule": library["positive_relation_contract"],
            "negative_rule": library["negative_relation_contract"],
        })
        log.info(f"参考库完成：正样本覆盖率={library['positive_coverage']:.4f}，完整 K={HARD_K} 覆盖率={library['full_reference_coverage']:.4f}，自引用={library['self_reference_count']}。")
        if library["self_reference_count"] != 0:
            raise RuntimeError("参考库自引用不为 0，拒绝继续")
        valid_indices = list(library["valid_anchor_indices"])
        if len(valid_indices) < EVAL_ANCHORS:
            eval_anchor_indices = valid_indices
        else:
            rng = np.random.default_rng(20261016)
            eval_anchor_indices = sorted(rng.choice(valid_indices, size=EVAL_ANCHORS, replace=False).tolist())
        write_json(output_dir / "参考库" / "固定诊断 anchors.json", {
            "seed": 20261016,
            "count": len(eval_anchor_indices),
            "indices": eval_anchor_indices,
            "train_only": True,
        })
        initial = initial_geometry(library_features, eval_anchor_indices, library)
        write_json(output_dir / "geometry" / "initial_geometry.json", initial)
        log.info(f"固定局部几何诊断集：{len(eval_anchor_indices)} 个 anchor；初始局部间隔={initial['local_margin_mean']:.6f}，正样本 Rank-1={initial['positive_rank1_rate']:.4f}，侵入率={initial['impostor_intrusion_rate']:.4f}。")

        gpc_bank = build_train_only_pair_bank(dataset, "ltcc")
        write_json(output_dir / "参考库" / "gpc_train_only_bank_audit.json", {
            "source": gpc_bank["source"],
            "source_path": gpc_bank["source_path"],
            "cross_key_count": gpc_bank["cross_key_count"],
            "train_only": gpc_bank["all_keys_train_only"],
            "gpc_lambda": 0.1,
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()

        path_to_index = {str(item["image_path"]): index for index, item in enumerate(train_items)}
        all_results = []
        for repeat in range(1, REPEATS + 1):
            batch_seed = 1234 + repeat * 100
            batches = collect_fixed_training_batches(cfg, dataset, batch_seed, log)
            for arm in ("gpc", "gpc_ndp"):
                result = run_one_arm(
                    cfg,
                    state,
                    dataset,
                    batches,
                    library,
                    path_to_index,
                    eval_anchor_indices,
                    initial,
                    transform_test,
                    device,
                    arm,
                    repeat,
                    output_dir,
                    log,
                )
                all_results.append(result)
        write_json(output_dir / "短程结果.json", all_results)
        aggregate_result = aggregate(all_results, initial, output_dir, log)
        risk_result = analyze_risk(all_results, initial, output_dir, log)

        manifest = {
            "created_beijing": now_bj().isoformat(),
            "run_time": run_time,
            "output_dir": str(output_dir),
            "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_sha, "epoch": 60, "read_only": True},
            "runtime": runtime,
            "dataset": "LTCC",
            "train_only": True,
            "formal_60_epoch_started": False,
            "gpc_unchanged": True,
            "ndp": {
                "hard_k": HARD_K,
                "margin_probe": NDP_MARGIN,
                "lambda_probe": NDP_LAMBDA_PROBE,
                "reference_positive": "same PID + different clothes_id + not self; highest initial cosine among valid train candidates",
                "reference_negative": "top-10 initial cosine neighbors with different PID",
                "loss": "relu(margin + max_negative_cosine - positive_cosine)",
                "reference_features_frozen_for_loss": True,
                "risk_guided_weighting_enabled": False,
            },
            "repeats": REPEATS,
            "steps": list(SHORT_STEPS),
            "reference_library": {
                "train_images": len(train_items),
                "positive_coverage": library["positive_coverage"],
                "full_reference_coverage": library["full_reference_coverage"],
                "self_reference_count": library["self_reference_count"],
                "diagnostic_anchor_count": len(eval_anchor_indices),
            },
            "geometry": aggregate_result,
            "risk": risk_result,
            "source_hashes": source_hashes,
            "elapsed_seconds": time.time() - start,
        }
        write_json(output_dir / "manifest.json", manifest)
        log.info("全部 NDP 短程诊断完成；未覆盖任何历史 checkpoint。")
        exit_code = 0
    except Exception as error:
        log.info(f"诊断失败：{type(error).__name__}: {error}")
        write_json(output_dir / "失败信息.json", {"created_beijing": now_bj().isoformat(), "error_type": type(error).__name__, "error": str(error)})
    finally:
        log.close()
        if os.environ.get("NDP_DISABLE_GITHUB_SYNC", "0") == "1":
            print("NDP_DISABLE_GITHUB_SYNC=1：跳过自动 GitHub 同步。", flush=True)
        else:
            sync_command = [
                str(ROOT / "scripts/auto_sync_github.sh"),
                str(output_dir),
                f"NDP LTCC 短程诊断 {run_time}",
            ]
            report_path = ROOT / "reports" / f"NDP_LTCC_DIAGNOSTIC_REPORT_{run_time}.md"
            if report_path.is_file():
                sync_command.append(str(report_path))
            sync_result = subprocess.run(sync_command, cwd=ROOT, check=False)
            if exit_code == 0 and sync_result.returncode != 0:
                exit_code = sync_result.returncode
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
