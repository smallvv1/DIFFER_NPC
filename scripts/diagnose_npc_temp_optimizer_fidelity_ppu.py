#!/usr/bin/env python3
"""在官方 PPU 上验证 NPC 临时 SGD 是否应继承正式动量状态。

本脚本只做短诊断：从已有 N3 模型开始，先用固定的 train-only 批次做少量
正式 SGD 更新以生成 ``momentum_buffer``，随后在同一个模型/批次/随机状态/学习率
上分别测试 ``TEMP_COPY_OPT_STATE=False/True``。所有模型和优化器状态都只存在于
该诊断进程和输出目录，不覆盖已有检查点，也不启动正式 60 轮训练。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml
from torch.cuda import amp


PROJECT = Path(__file__).resolve().parents[1]
MAIN_TREE = Path("/home/DIFFER_NPC")
PROCESSOR = Path("/home/processor")
for path in (str(MAIN_TREE), str(PROCESSOR), str(PROJECT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import cfg
from data import build_dataloader
from loss import make_loss
from model import build_model
from npc.engine import NPCController
from npc.losses import (
    extract_reid_features,
    per_sample_identity_ce,
    per_sample_reference_consistency,
)
from npc.runner import apply_variant, ensure_single_process_group, make_full_train_loader, set_seed
from solver import make_optimizer


CONFIG_PATH = PROJECT / "outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/config.yml"
DEFAULT_CHECKPOINT = MAIN_TREE / "outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_best.pth"


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("NPC临时SGD动量保真度PPU诊断")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def overlay_saved_config(node, values: dict, prefix: str = "") -> None:
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in node:
            raise KeyError(f"历史配置包含当前代码未定义字段：{full_key}")
        current = node[key]
        if isinstance(value, dict):
            overlay_saved_config(current, value, full_key)
        else:
            node[key] = value


def clone_batch(data: dict) -> dict:
    result = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, list):
            result[key] = list(value)
        elif isinstance(value, tuple):
            result[key] = tuple(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def capture_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def model_state_difference(before: dict[str, torch.Tensor], model: torch.nn.Module) -> dict:
    changed = 0
    max_abs = 0.0
    for key, original in before.items():
        current = model.state_dict()[key].detach().cpu()
        delta = (current.float() - original.float()).abs()
        max_abs = max(max_abs, float(delta.max().item()) if delta.numel() else 0.0)
        if not torch.equal(current, original):
            changed += 1
    return {"changed_tensor_count": changed, "max_abs_change": max_abs}


def capture_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    snapshot = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not state:
            continue
        snapshot[name] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                snapshot[name][key] = value.detach().cpu().clone()
            else:
                snapshot[name][key] = copy.deepcopy(value)
    return snapshot


def restore_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, snapshot: dict) -> None:
    optimizer.state.clear()
    parameters = dict(model.named_parameters())
    for name, state in snapshot.items():
        parameter = parameters[name]
        restored = {}
        for key, value in state.items():
            restored[key] = value.to(parameter.device).clone() if torch.is_tensor(value) else copy.deepcopy(value)
        optimizer.state[parameter] = restored


def optimizer_state_difference(model: torch.nn.Module, optimizer: torch.optim.Optimizer, snapshot: dict) -> dict:
    parameters = dict(model.named_parameters())
    changed = 0
    max_abs = 0.0
    missing = 0
    for name, expected in snapshot.items():
        state = optimizer.state.get(parameters[name])
        if state is None:
            missing += 1
            continue
        for key, expected_value in expected.items():
            actual = state.get(key)
            if torch.is_tensor(expected_value):
                if actual is None:
                    changed += 1
                    missing += 1
                    continue
                delta = (actual.detach().float().cpu() - expected_value.float()).abs()
                max_abs = max(max_abs, float(delta.max().item()) if delta.numel() else 0.0)
                if not torch.equal(actual.detach().cpu(), expected_value):
                    changed += 1
            elif actual != expected_value:
                changed += 1
    return {"changed_state_entries": changed, "missing_state_entries": missing, "max_abs_change": max_abs}


def optimizer_summary(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    count = 0
    numel = 0
    squared = 0.0
    max_abs = 0.0
    for parameter in model.parameters():
        state = optimizer.state.get(parameter, {})
        buffer = state.get("momentum_buffer")
        if buffer is None:
            continue
        count += 1
        numel += int(buffer.numel())
        value = buffer.detach().float()
        squared += float(torch.sum(value * value).item())
        max_abs = max(max_abs, float(value.abs().max().item()))
    return {
        "optimizer_class": type(optimizer).__name__,
        "momentum_buffer_count": count,
        "momentum_buffer_numel": numel,
        "momentum_buffer_l2": squared ** 0.5,
        "momentum_buffer_max_abs": max_abs,
        "lr": [float(group["lr"]) for group in optimizer.param_groups],
        "momentum": [float(group.get("momentum", 0.0)) for group in optimizer.param_groups],
    }


def parameter_delta_metrics(before: torch.nn.Module, after: torch.nn.Module) -> dict:
    dot = 0.0
    before_sq = 0.0
    after_sq = 0.0
    max_abs = 0.0
    changed = 0
    after_parameters = dict(after.named_parameters())
    for name, parameter in before.named_parameters():
        delta = after_parameters[name].detach().float() - parameter.detach().float()
        value = delta.detach()
        before_sq += float(torch.sum(value * value).item())
        max_abs = max(max_abs, float(value.abs().max().item()))
        if bool(torch.any(value != 0).item()):
            changed += 1
    return {
        "l2": before_sq ** 0.5,
        "max_abs": max_abs,
        "changed_parameter_count": changed,
    }


def cross_delta_metrics(
    base_state: dict[str, torch.Tensor],
    candidate: torch.nn.Module,
    other: torch.nn.Module,
) -> dict:
    """比较 candidate-base 和 other-base 的方向与范数。"""
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    left_max = 0.0
    right_max = 0.0
    candidate_parameters = dict(candidate.named_parameters())
    other_parameters = dict(other.named_parameters())
    for name in candidate_parameters:
        base = base_state[name].float()
        left = candidate_parameters[name].detach().float().cpu() - base
        right = other_parameters[name].detach().float().cpu() - base
        dot += float(torch.sum(left * right).item())
        left_sq += float(torch.sum(left * left).item())
        right_sq += float(torch.sum(right * right).item())
        left_max = max(left_max, float(left.abs().max().item()))
        right_max = max(right_max, float(right.abs().max().item()))
    left_norm = left_sq ** 0.5
    right_norm = right_sq ** 0.5
    return {
        "temporary_l2": left_norm,
        "formal_l2": right_norm,
        "temporary_max_abs": left_max,
        "formal_max_abs": right_max,
        "temporary_formal_cosine": dot / max(left_norm * right_norm, 1e-30),
    }


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def finite_float(value: str) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def read_mode_rows(controller: NPCController, iteration: int) -> list[dict[str, str]]:
    path = controller.output_dir / "npc_diagnostics" / "npc_weights.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if int(row["iteration"]) == iteration and row["reference_valid"] == "True"]


def reference_post_metrics(
    controller: NPCController,
    model: torch.nn.Module,
    data: dict,
) -> dict[int, dict[str, float]]:
    indices = controller._batch_indices(data)
    valid_mask = torch.as_tensor(controller.memory_bank.valid_reference_mask[indices.detach().cpu().numpy()], device=indices.device)
    valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if not bool(valid_positions.numel()):
        return {}
    ref_indices = torch.as_tensor(
        controller.memory_bank.reference_indices[
            indices.detach().cpu().numpy()[valid_mask.detach().cpu().numpy()]
        ],
        dtype=torch.long,
    )
    reference_batch = controller._load_reference_batch(ref_indices)
    original_training = model.training
    model.train()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
        images = data["image"].index_select(0, valid_positions.cpu()).to(controller.device)
        camids = data["camid"].index_select(0, valid_positions.cpu()).to(controller.device)
        _, anchor_feat = model(images, cam_label=camids)
        reference_score, reference_feat = model(
            reference_batch["image"], cam_label=reference_batch["camid"]
        )
        identity = per_sample_identity_ce(reference_score, reference_batch["pid"]).float().cpu().numpy()
        consistency = per_sample_reference_consistency(
            extract_reid_features(anchor_feat), extract_reid_features(reference_feat)
        ).float().cpu().numpy()
    model.train(original_training)
    result = {}
    for position, sample_index, identity_value, consistency_value in zip(
        valid_positions.detach().cpu().tolist(),
        indices.detach().cpu().tolist(),
        identity.tolist(),
        consistency.tolist(),
    ):
        result[int(sample_index)] = {
            "post_identity": float(identity_value),
            "post_reference": float(consistency_value),
        }
    return result


def batch_triplet_info(controller: NPCController, data: dict) -> dict:
    indices = controller._batch_indices(data)
    valid_mask = torch.as_tensor(controller.memory_bank.valid_reference_mask[indices.detach().cpu().numpy()], device=indices.device)
    valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
    targets = data["pid"].index_select(0, valid_positions.cpu())
    unique, counts = torch.unique(targets, return_counts=True)
    compatible = bool(unique.numel() >= 2 and torch.all(counts == counts[0]).item())
    return {
        "full_batch_size": int(len(data["pid"])),
        "valid_reference_count": int(valid_positions.numel()),
        "valid_reference_ratio": float(valid_positions.numel() / max(1, len(data["pid"]))),
        "valid_subbatch_unique_pid_count": int(unique.numel()),
        "valid_subbatch_pid_counts": [int(value) for value in counts.cpu().tolist()],
        "temporary_triplet_compatible": compatible,
        "temporary_loss_path": "CE+Triplet" if compatible else "CE fallback (valid subbatch)",
        "formal_loss_path": "CE+Triplet+memory on full batch",
    }


def mode_result(
    mode: bool,
    cfg,
    controller: NPCController,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    data: dict,
    output: Path,
    logger: logging.Logger,
    base_model_state: dict[str, torch.Tensor],
    base_optimizer_state: dict,
    iteration: int,
) -> dict:
    restore_model_state(model, base_model_state)
    restore_optimizer_state(model, optimizer, base_optimizer_state)
    model.train()
    cfg.defrost()
    cfg.MODEL.NPC.TEMP_COPY_OPT_STATE = mode
    cfg.freeze()
    state_before_preaware = capture_model_state(model)
    optimizer_before_preaware = capture_optimizer_state(model, optimizer)
    batch_info = batch_triplet_info(controller, data)
    indices = controller._batch_indices(data)
    images = data["image"].to(controller.device, non_blocking=True)
    targets = data["pid"].to(controller.device, non_blocking=True)
    camids = data["camid"].to(controller.device, non_blocking=True)
    clothes = data.get("clothes_id")
    if torch.is_tensor(clothes):
        clothes = clothes.to(controller.device, non_blocking=True)
    captions = data.get("caption_feature")
    if torch.is_tensor(captions):
        captions = captions.to(controller.device, non_blocking=True)
    scaler = amp.GradScaler(init_scale=1024)
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast(enabled=True):
        score, feat = model(images, cam_label=camids)
    weights, stats = controller.preaware_weights(
        model, loss_fn, data, score, feat, targets, camids, captions, clothes,
        epoch=17, iteration=iteration,
    )
    model_pollution = model_state_difference(state_before_preaware, model)
    optimizer_pollution = optimizer_state_difference(model, optimizer, optimizer_before_preaware)
    temp_model = controller.temp_model
    if temp_model is None:
        raise RuntimeError("preaware_weights 未建立临时模型")
    with amp.autocast(enabled=True):
        base_loss = loss_fn(
            score, feat, targets, camids, captions, clothes_ids=clothes,
            train_writer=NullWriter(), step=iteration, npc_weights=weights,
        )
        memory_loss, _ = controller.memory_supervision(feat, data)
        total_loss = base_loss + float(cfg.MODEL.NPC.MEMORY_LOSS_WEIGHT) * memory_loss
    if not torch.isfinite(total_loss.detach()):
        raise RuntimeError(f"正式短更新损失非有限：mode={mode}")
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    direction = cross_delta_metrics(base_model_state, temp_model, model)
    formal_reference = reference_post_metrics(controller, model, data)
    rows = read_mode_rows(controller, iteration)
    predicted_identity = []
    actual_identity = []
    predicted_reference = []
    actual_reference = []
    improvement_identity = []
    improvement_reference = []
    for row in rows:
        sample_index = int(row["sample_index"])
        actual = formal_reference.get(sample_index)
        if actual is None:
            continue
        pre_identity = finite_float(row["pre_identity"])
        post_identity = finite_float(row["post_identity"])
        pre_reference = finite_float(row["pre_reference"])
        post_reference = finite_float(row["post_reference"])
        if None in (pre_identity, post_identity, pre_reference, post_reference):
            continue
        predicted_identity.append(post_identity - pre_identity)
        actual_identity.append(actual["post_identity"] - pre_identity)
        predicted_reference.append(post_reference - pre_reference)
        actual_reference.append(actual["post_reference"] - pre_reference)
        improvement_identity.append(float((post_identity - pre_identity < 0) == (actual["post_identity"] - pre_identity < 0)))
        improvement_reference.append(float((post_reference - pre_reference < 0) == (actual["post_reference"] - pre_reference < 0)))

    weight_values = [float(row["weight"]) for row in rows]
    ratio_values = [float(row["ratio"]) for row in rows]
    p_values = [float(row["p_ratio"]) for row in rows]
    q_values = [float(row["q_ratio"]) for row in rows]
    return {
        "TEMP_COPY_OPT_STATE": mode,
        "configuration": {
            "TEMP_COPY_OPT_STATE": mode,
            "TEMP_LR_SCALE": float(cfg.MODEL.NPC.TEMP_LR_SCALE),
            "TEMP_OPT_STEPS": int(cfg.MODEL.NPC.TEMP_OPT_STEPS),
            "PREAWARE_FP32_ANCHOR": bool(cfg.MODEL.NPC.PREAWARE_FP32_ANCHOR),
            "formal_lr": [float(group["lr"]) for group in optimizer.param_groups],
        },
        "batch": batch_info,
        "preaware_stats": stats,
        "risk_ratio": summarize(ratio_values),
        "p_ratio": summarize(p_values),
        "q_ratio": summarize(q_values),
        "dynamic_weight": {
            **summarize(weight_values),
            "lt_0_99_ratio": float(np.mean(np.asarray(weight_values) < 0.99)) if weight_values else 0.0,
            "lt_0_90_ratio": float(np.mean(np.asarray(weight_values) < 0.90)) if weight_values else 0.0,
        },
        "temporary_vs_formal_update": direction,
        "predicted_vs_actual_formal_reference_change": {
            "identity_predicted_delta": summarize(predicted_identity),
            "identity_actual_delta": summarize(actual_identity),
            "identity_abs_gap_mean": float(np.mean(np.abs(np.asarray(predicted_identity) - np.asarray(actual_identity)))) if predicted_identity else None,
            "identity_improvement_sign_agreement": float(np.mean(improvement_identity)) if improvement_identity else None,
            "reference_predicted_delta": summarize(predicted_reference),
            "reference_actual_delta": summarize(actual_reference),
            "reference_abs_gap_mean": float(np.mean(np.abs(np.asarray(predicted_reference) - np.asarray(actual_reference)))) if predicted_reference else None,
            "reference_improvement_sign_agreement": float(np.mean(improvement_reference)) if improvement_reference else None,
        },
        "pollution_check_after_preaware": {
            "formal_model": model_pollution,
            "formal_optimizer": optimizer_pollution,
        },
        "formal_update": {
            "total_loss": float(total_loss.detach().float().item()),
            "base_loss": float(base_loss.detach().float().item()),
            "memory_loss": float(memory_loss.detach().float().item()),
            "amp_scale_after": float(scaler.get_scale()),
        },
        "optimizer_after_formal_update": optimizer_summary(model, optimizer),
        "csv_valid_row_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-epoch", type=int, default=17)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.checkpoint_epoch < 1:
        parser.error("warmup-steps 和 checkpoint-epoch 必须为正数")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"检查点不存在：{checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("该诊断必须在官方 PPU/CUDA 上运行，禁止 CPU 回退")

    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = args.output or PROJECT / "outputs" / f"PHASE_NPC_TEMP_OPTIMIZER_FIDELITY_PPU_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    logger = configure_logger(output / f"temp_optimizer_fidelity_{timestamp}.log")
    logger.info("开始真实 PPU 临时 SGD 动量诊断；输出=%s", output)
    logger.info("设备=%s；PyTorch=%s；代码提交=%s", torch.cuda.get_device_name(0), torch.__version__, git_head())
    logger.info("实际检查点=%s；SHA256=%s", checkpoint, sha256(checkpoint))

    cfg.defrost()
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        overlay_saved_config(cfg, yaml.safe_load(handle))
    apply_variant(cfg, "N3")
    cfg.defrost()
    cfg.DATA.ROOT = "/home/data"
    cfg.DATA.CAPTION_DIR = "/home/data/DIFFER_TextCaption/LTCC_ReID"
    cfg.DATA.TEXT_MODEL_VERSION = f"{cfg.DATA.TEXT_MODEL}-{ {512: 'B-16', 768: 'L-14', 1024: 'bigE-14'}[cfg.MODEL.CLIP_DIM] }"
    cfg.SOLVER.SEED = 1234
    cfg.TEST.WEIGHT = ""
    cfg.MODEL.NPC.TEMP_COPY_OPT_STATE = False
    cfg.MODEL.NPC.TEMP_LR_SCALE = 1.0
    cfg.MODEL.NPC.PREAWARE_FP32_ANCHOR = True
    cfg.OUTPUT_DIR = str(output)
    cfg.freeze()

    set_seed(int(cfg.SOLVER.SEED))
    ensure_single_process_group(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    train_loader, _, _, dataset, _, _, _ = build_dataloader(cfg)
    full_train_loader = make_full_train_loader(cfg, dataset)
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"检查点装载不完整：missing={incompatible.missing_keys[:5]}, unexpected={incompatible.unexpected_keys[:5]}")
    model.to(device)
    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    controller_false = NPCController(
        cfg, dataset, optimizer, output / "TEMP_COPY_OPT_STATE_FALSE", device, logger,
        reference_dataset=full_train_loader.dataset,
    )

    def progress_loader():
        for batch in full_train_loader:
            yield batch

    memory = controller_false.initialize_memory(model, progress_loader(), loss_fn, epoch=args.checkpoint_epoch)
    if controller_false.memory_bank is None:
        raise RuntimeError("记忆库初始化失败")
    batches = []
    selected = None
    for batch in full_train_loader:
        frozen = clone_batch(batch)
        batches.append(frozen)
        indices = controller_false._batch_indices(frozen)
        valid = int(controller_false.memory_bank.valid_reference_mask[indices.detach().cpu().numpy()].sum())
        if len(batches) >= args.warmup_steps + 1 and valid >= 2 and selected is None:
            selected = frozen
            break
    if selected is None:
        raise RuntimeError("未找到含至少 2 个可靠参考样本的固定比较批次")
    logger.info("固定比较批次：预热批次=%d，batch_size=%d，可靠参考=%d", args.warmup_steps, len(selected["pid"]), batch_triplet_info(controller_false, selected)["valid_reference_count"])

    warmup_scaler = amp.GradScaler(init_scale=1024)
    warmup_rows = []
    model.train()
    for warmup_index, batch in enumerate(batches[:args.warmup_steps], start=1):
        images = batch["image"].to(device)
        targets = batch["pid"].to(device)
        camids = batch["camid"].to(device)
        clothes = batch.get("clothes_id")
        if torch.is_tensor(clothes):
            clothes = clothes.to(device)
        captions = batch.get("caption_feature")
        if torch.is_tensor(captions):
            captions = captions.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=True):
            score, feat = model(images, cam_label=camids)
            base_loss = loss_fn(
                score, feat, targets, camids, captions, clothes_ids=clothes,
                train_writer=NullWriter(), step=warmup_index, npc_weights=None,
            )
            memory_loss, _ = controller_false.memory_supervision(feat, batch)
            total_loss = base_loss + float(cfg.MODEL.NPC.MEMORY_LOSS_WEIGHT) * memory_loss
        warmup_scaler.scale(total_loss).backward()
        warmup_scaler.unscale_(optimizer)
        warmup_scaler.step(optimizer)
        warmup_scaler.update()
        warmup_rows.append({
            "step": warmup_index,
            "loss": float(total_loss.detach().float().item()),
            "base_loss": float(base_loss.detach().float().item()),
            "memory_loss": float(memory_loss.detach().float().item()),
            "batch_size": len(batch["pid"]),
        })
        logger.info("正式 SGD 预热 %d/%d：loss=%.6f，momentum_buffer=%d", warmup_index, args.warmup_steps, warmup_rows[-1]["loss"], optimizer_summary(model, optimizer)["momentum_buffer_count"])

    base_model_state = capture_model_state(model)
    base_optimizer_state = capture_optimizer_state(model, optimizer)
    momentum_summary = optimizer_summary(model, optimizer)
    if momentum_summary["momentum_buffer_count"] == 0:
        raise RuntimeError("预热后没有形成正式 SGD momentum_buffer")

    false_result = mode_result(
        False, cfg, controller_false, model, optimizer, loss_fn, selected, output,
        logger, base_model_state, base_optimizer_state, iteration=9001,
    )
    controller_false.close()

    cfg.defrost()
    cfg.MODEL.NPC.TEMP_COPY_OPT_STATE = True
    cfg.freeze()
    controller_true = NPCController(
        cfg, dataset, optimizer, output / "TEMP_COPY_OPT_STATE_TRUE", device, logger,
        reference_dataset=full_train_loader.dataset,
    )
    controller_true.memory_bank = controller_false.memory_bank
    controller_true.clean_probabilities = controller_false.clean_probabilities
    controller_true.clean_mask = controller_false.clean_mask
    true_result = mode_result(
        True, cfg, controller_true, model, optimizer, loss_fn, selected, output,
        logger, base_model_state, base_optimizer_state, iteration=9002,
    )
    controller_true.close()

    payload = {
        "schema_version": 1,
        "timestamp": timestamp,
        "code_head": git_head(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dataset": "LTCC train-only",
        "no_formal_60_epoch_training": True,
        "no_gmm_weight_lambda_changes": True,
        "config": {
            "variant": "N3",
            "seed": 1234,
            "lr": [float(group["lr"]) for group in optimizer.param_groups],
            "momentum": [float(group.get("momentum", 0.0)) for group in optimizer.param_groups],
            "TEMP_LR_SCALE": 1.0,
            "TEMP_OPT_STEPS": int(cfg.MODEL.NPC.TEMP_OPT_STEPS),
            "PREAWARE_FP32_ANCHOR": True,
        },
        "memory_initialization": memory,
        "warmup": {
            "steps": args.warmup_steps,
            "rows": warmup_rows,
            "formal_optimizer_after_warmup": momentum_summary,
        },
        "modes": {
            "TEMP_COPY_OPT_STATE_FALSE": false_result,
            "TEMP_COPY_OPT_STATE_TRUE": true_result,
        },
        "interpretation": {
            "temporary_vs_formal_update_direction": "temporary_l2/formal_l2/cosine are computed from the exact same pre-comparison model state and batch",
            "risk_prediction_target": "temporary pre/post reliable-reference identity and consistency loss deltas versus actual formal post-update deltas",
            "causes_kept_separate": [
                "formal SGD momentum_buffer inheritance",
                "reliable-reference filtered subbatch size and target multiplicity",
                "temporary CE fallback versus formal full-batch CE+Triplet+memory loss",
                "formal AMP path versus fixed FP32 preaware anchor calibration",
            ],
        },
    }
    with (output / "temp_optimizer_fidelity_ppu.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info("诊断完成；结果=%s", output / "temp_optimizer_fidelity_ppu.json")


if __name__ == "__main__":
    main()
