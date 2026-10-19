#!/usr/bin/env python3
"""在真实 LTCC train split 上审计 NPC 零临时更新风险信号。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
MAIN_TREE = Path("/home/DIFFER_NPC")
PROCESSOR = Path("/home/processor")
for path in (str(MAIN_TREE), str(PROCESSOR), str(PROJECT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import cfg
from loss import make_loss
from model import build_model
from npc.engine import NPCController
from npc.losses import (
    calculate_npc_weights,
    extract_reid_features,
    per_sample_identity_ce,
    per_sample_reference_consistency,
)
from npc.runner import (
    apply_variant,
    ensure_single_process_group,
    make_full_train_loader,
    set_seed,
)
from data import build_dataloader
from solver import make_optimizer


CONFIG_PATH = PROJECT / "outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/config.yml"
CHECKPOINT_PATH = MAIN_TREE / "outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234/N3/checkpoints/model_best.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("NPC零更新风险审计")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def overlay_saved_config(node, values: dict, prefix: str = "") -> None:
    """按历史正式配置逐字段覆盖，避免 yacs 将引号数字字符串误解码成整数。"""
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in node:
            raise KeyError(f"历史配置包含当前代码未定义字段：{full_key}")
        current = node[key]
        if isinstance(value, dict):
            if not hasattr(current, "items"):
                raise TypeError(f"历史配置字段类型冲突：{full_key}")
            overlay_saved_config(current, value, full_key)
        else:
            node[key] = value


def snapshot_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def state_delta(before: dict[str, torch.Tensor], module: torch.nn.Module) -> dict:
    after = module.state_dict()
    changed = []
    max_abs = 0.0
    for key, original in before.items():
        current = after[key].detach().cpu()
        delta = (current.float() - original.float()).abs()
        item = float(delta.max().item()) if delta.numel() else 0.0
        max_abs = max(max_abs, item)
        if not torch.equal(current, original):
            changed.append(key)
    return {"changed_tensor_count": len(changed), "changed_tensor_names": changed[:30], "max_abs_change": max_abs}


def tensor_delta(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference.detach().float() - candidate.detach().float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "rms": float(torch.sqrt(torch.mean(delta.square())).item()),
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    x = np.asarray(values, dtype=np.float64)
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std()),
        "min": float(x.min()),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def json_fallback(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"不能序列化诊断字段：{type(value).__name__}")


def move_batch(data: dict, device: torch.device) -> tuple:
    image = data["image"].to(device, non_blocking=True)
    target = data["pid"].to(device, non_blocking=True)
    camid = data["camid"].to(device, non_blocking=True)
    clothes = data.get("clothes_id")
    if torch.is_tensor(clothes):
        clothes = clothes.to(device, non_blocking=True)
    caption = data.get("caption_feature")
    if torch.is_tensor(caption):
        caption = caption.to(device, non_blocking=True)
    return image, target, camid, clothes, caption


def metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trials-per-mode", type=int, default=4)
    args = parser.parse_args()

    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = args.output or PROJECT / "outputs" / f"NPC零更新风险审计_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / f"零更新风险审计日志_{timestamp}.log"
    logger = configure_logger(log_path)

    if not torch.cuda.is_available():
        raise RuntimeError("零更新审计要求使用 PPU/CUDA；未允许 CPU 回退")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    logger.info("开始真实 LTCC 零更新审计；输出目录：%s", output)
    logger.info("设备：%s；设备数：%d；PyTorch：%s", torch.cuda.get_device_name(0), torch.cuda.device_count(), torch.__version__)
    logger.info("代码分支：experiment/npc-continuous-weight-20261019；工作树：%s", PROJECT)

    # yacs 会把历史 YAML 中带引号的数值字符串（如 BIO_INDEX="0"、
    # DEVICE_ID="0"）再次解码为整数，与运行时 schema 冲突。逐字段覆盖
    # 保留正式配置原始类型和值；未出现在历史文件中的新字段沿用当前默认值。
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
    cfg.MODEL.NPC.TEMP_LR_SCALE = 0.0
    cfg.MODEL.NPC.WEIGHT_POLICY = "continuous"
    cfg.MODEL.NPC.WEIGHT_SMOOTH_BETA = 1.0
    cfg.OUTPUT_DIR = str(output)
    cfg.freeze()

    set_seed(int(cfg.SOLVER.SEED))
    ensure_single_process_group(0)
    train_loader, _, _, dataset, _, _, _ = build_dataloader(cfg)
    full_train_loader = make_full_train_loader(cfg, dataset)
    logger.info("LTCC train-only 数据：%d 条记录、%d 个身份；完整库 batch=%d", len(dataset.train), dataset.num_train_pids, cfg.DATA.TEST_BATCH)

    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "已有 N3 检查点未能完整装载："
            f"missing={incompatible.missing_keys[:10]}, unexpected={incompatible.unexpected_keys[:10]}"
        )
    del state
    model.to(device)
    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    controller = NPCController(
        cfg,
        dataset,
        optimizer,
        output / "NPC诊断明细",
        device,
        logger,
        reference_dataset=full_train_loader.dataset,
    )

    def progress_loader():
        total = math.ceil(len(dataset.train) / int(cfg.DATA.TEST_BATCH))
        for index, batch in enumerate(full_train_loader, start=1):
            if index == 1 or index % 40 == 0 or index == total:
                logger.info("正在用 LTCC train-only 建库：%d/%d batches", index, total)
            yield batch

    logger.info("加载已完成训练的 N3 最优检查点（用于诊断，不作为正式训练初始化）：%s", CHECKPOINT_PATH)
    memory_init = controller.initialize_memory(model, progress_loader(), loss_fn, epoch=17)
    bank = controller.memory_bank
    if bank is None:
        raise RuntimeError("记忆库未成功建立")
    valid_indices = np.flatnonzero(bank.valid_reference_mask)
    self_refs = sum(int(bank.reference_indices[i]) == int(i) for i in valid_indices)
    identity_mismatches = sum(
        dataset.train[i]["pid"] != dataset.train[int(bank.reference_indices[i])]["pid"] for i in valid_indices
    )
    cross_clothes = sum(
        dataset.train[i].get("clothes_id") != dataset.train[int(bank.reference_indices[i])].get("clothes_id")
        for i in valid_indices
    )
    memory_coverage = bank.coverage()
    logger.info(
        "记忆库核验：有效 %d/%d（%.2f%%），有效参考自引用=%d，身份错配=%d，跨服装=%d/%d（%.2f%%）",
        len(valid_indices), len(dataset.train), 100 * len(valid_indices) / max(1, len(dataset.train)),
        self_refs, identity_mismatches, cross_clothes, len(valid_indices),
        100 * cross_clothes / max(1, len(valid_indices)),
    )
    if not len(valid_indices) or self_refs or identity_mismatches:
        raise RuntimeError("记忆库有效参考为空、包含自引用或身份错配，停止审计")

    # 固定 8 个真实 train-loader batch；同一图像增强重复多次，以估计前向噪声而不混入换批差异。
    batches = []
    iterator = iter(train_loader)
    for _ in range(8):
        try:
            batches.append(next(iterator))
        except StopIteration:
            break
    if not batches:
        raise RuntimeError("LTCC train loader 未产生批次")

    base_model = model
    base_model.train()
    model_before = snapshot_state(base_model)

    # 训练态连续前向：记录模型自身是否存在 RNG/随机层波动；再比较 AMP 与 FP32。
    fixed_image, fixed_target, fixed_camid, _, _ = move_batch(batches[0], device)
    train_fp32_runs = []
    train_amp_runs = []
    with torch.no_grad():
        for trial in range(5):
            torch.manual_seed(1000 + trial)
            torch.cuda.manual_seed_all(1000 + trial)
            with torch.amp.autocast("cuda", enabled=False):
                score, feat = base_model(fixed_image, cam_label=fixed_camid)
            train_fp32_runs.append((score["cls_score"].detach().float().cpu(), extract_reid_features(feat).detach().float().cpu()))
        for trial in range(5):
            torch.manual_seed(2000 + trial)
            torch.cuda.manual_seed_all(2000 + trial)
            with torch.amp.autocast("cuda", enabled=True):
                score, feat = base_model(fixed_image, cam_label=fixed_camid)
            train_amp_runs.append((score["cls_score"].detach().float().cpu(), extract_reid_features(feat).detach().float().cpu()))

    def run_delta(runs: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
        return {
            "logits": tensor_delta(runs[0][0], runs[-1][0]),
            "features": tensor_delta(runs[0][1], runs[-1][1]),
            "different_rng_seeds": len(runs),
        }

    forward_repeat = {"train_fp32": run_delta(train_fp32_runs), "train_amp": run_delta(train_amp_runs)}
    logger.info(
        "训练态不同 RNG 前向差异：FP32 feat max|Δ|=%.8g，AMP feat max|Δ|=%.8g",
        forward_repeat["train_fp32"]["features"]["max_abs"],
        forward_repeat["train_amp"]["features"]["max_abs"],
    )

    # 直接量化：当前锚点 AMP/FP32 的差异如何改变 q=1-cos(anchor, valid reference)。
    amp_impact = {"sample_count": 0, "q_amp": [], "q_fp32": [], "q_ratio_amp_over_fp32": [], "feature_amp_vs_fp32": []}
    for raw in batches:
        image, target, camid, _, _ = move_batch(raw, device)
        sample_indices = controller._batch_indices(raw)
        _, valid = bank.lookup_with_validity(sample_indices, device)
        if not bool(valid.any()):
            continue
        ref_idx_np = bank.reference_indices[sample_indices.detach().cpu().numpy()[valid.detach().cpu().numpy()]]
        ref_idx = torch.as_tensor(ref_idx_np, dtype=torch.long)
        ref_batch = controller._load_reference_batch(ref_idx)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                _, feat_amp = base_model(image, cam_label=camid)
            with torch.amp.autocast("cuda", enabled=False):
                _, feat_fp32 = base_model(image, cam_label=camid)
                _, ref_feat = base_model(ref_batch["image"], cam_label=ref_batch["camid"])
            q_amp = per_sample_reference_consistency(
                extract_reid_features(feat_amp)[valid], extract_reid_features(ref_feat)
            )
            q_fp32 = per_sample_reference_consistency(
                extract_reid_features(feat_fp32)[valid], extract_reid_features(ref_feat)
            )
        amp_impact["sample_count"] += int(valid.sum().item())
        amp_impact["q_amp"].extend(q_amp.detach().cpu().tolist())
        amp_impact["q_fp32"].extend(q_fp32.detach().cpu().tolist())
        amp_impact["q_ratio_amp_over_fp32"].extend((q_amp / q_fp32.clamp_min(1e-8)).detach().cpu().tolist())
        amp_impact["feature_amp_vs_fp32"].append(tensor_delta(extract_reid_features(feat_amp), extract_reid_features(feat_fp32)))
    amp_impact_summary = {
        "sample_count": amp_impact["sample_count"],
        "q_amp": summarize(amp_impact["q_amp"]),
        "q_fp32": summarize(amp_impact["q_fp32"]),
        "q_ratio_amp_over_fp32": summarize(amp_impact["q_ratio_amp_over_fp32"]),
        "feature_amp_vs_fp32_max_abs_max": max((row["max_abs"] for row in amp_impact["feature_amp_vs_fp32"]), default=float("nan")),
        "feature_amp_vs_fp32_mean_abs_mean": float(np.mean([row["mean_abs"] for row in amp_impact["feature_amp_vs_fp32"]])) if amp_impact["feature_amp_vs_fp32"] else float("nan"),
    }
    logger.info(
        "AMP 对 q 的影响：q(AMP)均值=%.8f，q(FP32)均值=%.8f，q比值均值=%.8f，特征最大绝对差=%.8g",
        amp_impact_summary["q_amp"]["mean"], amp_impact_summary["q_fp32"]["mean"],
        amp_impact_summary["q_ratio_amp_over_fp32"]["mean"], amp_impact_summary["feature_amp_vs_fp32_max_abs_max"],
    )

    # 用实际 NPCController 计算零步风险；分 AMP（正式训练路径）和 FP32 anchor 对照。
    trial_call_count = {"amp": 0, "fp32": 0}
    temp_state_deltas = []
    weights_by_mode = {"amp": [], "fp32": []}
    for mode in ("amp", "fp32"):
        for repeat in range(args.trials_per_mode):
            for batch_index, raw in enumerate(batches):
                image, target, camid, clothes, caption = move_batch(raw, device)
                base_model.train()
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=(mode == "amp")):
                        score, feat = base_model(image, cam_label=camid)
                before_call = snapshot_state(base_model)
                weights, stats = controller.preaware_weights(
                    base_model, loss_fn, raw, score, feat, target, camid, caption, clothes,
                    epoch=17, iteration=(1000 if mode == "amp" else 2000) + repeat * 100 + batch_index,
                )
                trial_call_count[mode] += 1
                weights_by_mode[mode].extend(weights.detach().float().cpu().tolist())
                if controller.temp_model is not None:
                    temp_state_deltas.append(state_delta(before_call, controller.temp_model))
                base_change = state_delta(before_call, base_model)
                if base_change["changed_tensor_count"]:
                    raise RuntimeError(f"NPC 零更新调用改动了正式模型状态：{base_change}")
                if not stats["finite"]:
                    raise RuntimeError(f"NPC 实际诊断输出非有限：mode={mode}, stats={stats}")
                logger.info(
                    "%s 第%d/%d轮第%d批：有效参考=%d/%d，均值r=%.8f，均值w=%.8f，w<0.99比例=%.2f%%",
                    "AMP" if mode == "amp" else "FP32",
                    repeat + 1, args.trials_per_mode, batch_index + 1,
                    stats["valid_reference_count"], len(target), stats["ratio_mean_valid"], stats["均值"],
                    100 * (weights < 0.99).float().mean().item(),
                )

    if controller.weights_file is not None:
        controller.weights_file.flush()
        controller.weights_file.close()
        controller.weights_file = None

    all_rows = metric_rows(output / "NPC诊断明细/npc_diagnostics/npc_weights.csv")
    per_mode = {}
    for mode, first_iteration, last_iteration in (("amp", 1000, 1999), ("fp32", 2000, 2999)):
        rows = [row for row in all_rows if row.get("iteration") and first_iteration <= int(row["iteration"]) <= last_iteration and row.get("reference_valid") == "True"]
        ratios = [float(row["ratio"]) for row in rows if row.get("ratio")]
        weights = [float(row["weight"]) for row in rows if row.get("weight")]
        p_ratios = [float(row["p_ratio"]) for row in rows if row.get("p_ratio")]
        q_ratios = [float(row["q_ratio"]) for row in rows if row.get("q_ratio")]
        per_mode[mode] = {
            "valid_entry_count": len(rows),
            "risk_ratio": summarize(ratios),
            "weight": summarize(weights),
            "p_ratio": summarize(p_ratios),
            "q_ratio": summarize(q_ratios),
            "strict_false_positive_ratio_r_lt_1": float(np.mean(np.asarray(ratios) < 1.0)) if ratios else float("nan"),
            "false_positive_ratio_r_lt_0_999": float(np.mean(np.asarray(ratios) < 0.999)) if ratios else float("nan"),
            "operational_false_positive_weight_lt_0_99": float(np.mean(np.asarray(weights) < 0.99)) if weights else float("nan"),
            "weight_lt_0_9_ratio": float(np.mean(np.asarray(weights) < 0.9)) if weights else float("nan"),
        }
        logger.info(
            "%s 汇总：有效条目=%d，r均值=%.8f，r<1严格假阳性=%.2f%%，r<0.999=%.2f%%，w均值=%.8f，w<0.99=%.2f%%",
            mode, per_mode[mode]["valid_entry_count"], per_mode[mode]["risk_ratio"]["mean"],
            100 * per_mode[mode]["strict_false_positive_ratio_r_lt_1"],
            100 * per_mode[mode]["false_positive_ratio_r_lt_0_999"],
            per_mode[mode]["weight"]["mean"],
            100 * per_mode[mode]["operational_false_positive_weight_lt_0_99"],
        )

    # 权重即根据 ratio 从同一真实有效条目计算；明确报告公式输出的零更新理论基线。
    zero_baseline = {}
    for ratio_value in (1.0, 0.999, 0.99, 0.95, 0.9):
        ratio_tensor = torch.tensor([ratio_value], dtype=torch.float32)
        _, weight_tensor, _ = calculate_npc_weights(
            ratio_tensor, torch.ones_like(ratio_tensor), ratio_tensor,
            torch.ones_like(ratio_tensor), policy="continuous", smooth_beta=1.0,
        )
        zero_baseline[str(ratio_value)] = float(weight_tensor.item())

    post_model_state = state_delta(model_before, base_model)
    temp_delta_summary = {
        "comparison_count": len(temp_state_deltas),
        "any_changed_tensor_count": sum(row["changed_tensor_count"] > 0 for row in temp_state_deltas),
        "max_abs_change": max((row["max_abs_change"] for row in temp_state_deltas), default=0.0),
        "changed_examples": [name for row in temp_state_deltas for name in row["changed_tensor_names"][:3]][:12],
    }
    logger.info(
        "零更新状态核验：正式模型状态差异张量=%d；临时模型比较=%d次，有差异=%d次，最大绝对变化=%.8g",
        post_model_state["changed_tensor_count"], temp_delta_summary["comparison_count"],
        temp_delta_summary["any_changed_tensor_count"], temp_delta_summary["max_abs_change"],
    )

    report_data = {
        "timestamp_beijing": timestamp,
        "branch": "experiment/npc-continuous-weight-20261019",
        "code_head": os.popen(f"git -C {PROJECT} rev-parse HEAD").read().strip(),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
        "checkpoint_semantics": "已完成训练的 N3 epoch17 best checkpoint；仅用于风险诊断，不用于正式训练续训",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "configuration": {
            "dataset": "LTCC train split only",
            "seed": int(cfg.SOLVER.SEED),
            "weight_policy": str(cfg.MODEL.NPC.WEIGHT_POLICY),
            "weight_smooth_beta": float(cfg.MODEL.NPC.WEIGHT_SMOOTH_BETA),
            "temporary_lr_scale": float(cfg.MODEL.NPC.TEMP_LR_SCALE),
            "temporary_optimizer_steps": int(cfg.MODEL.NPC.TEMP_OPT_STEPS),
            "gmm_threshold": float(cfg.MODEL.NPC.GMM_THRESHOLD),
            "amp_enabled_for_anchor": True,
        },
        "memory_coverage": memory_coverage,
        "memory_validation": {
            "valid_reference_count": int(len(valid_indices)),
            "self_reference_count": int(self_refs),
            "identity_mismatch_count": int(identity_mismatches),
            "cross_clothes_reference_count": int(cross_clothes),
            "cross_clothes_ratio_among_valid": float(cross_clothes / max(1, len(valid_indices))),
            "gmm": memory_init,
        },
        "actual_preaware_trials": trial_call_count,
        "risk_and_weight_by_mode": per_mode,
        "forward_repeatability": forward_repeat,
        "amp_impact_on_consistency": amp_impact_summary,
        "formal_model_state_delta": post_model_state,
        "temporary_model_zero_lr_state_delta": temp_delta_summary,
        "continuous_weight_reference_values": zero_baseline,
        "decision_gate": {
            "strict_fp_definition": "真实无参数更新时 r<1；另报告 r<0.999 和连续权重 w<0.99 的实用阈值",
            "formal_training_started": False,
        },
    }
    with (output / "零更新风险审计结果.json").open("w", encoding="utf-8") as handle:
        json.dump(report_data, handle, ensure_ascii=False, indent=2, default=json_fallback)
    logger.info("零更新风险审计完成；原始指标及配置已写入：%s", output / "零更新风险审计结果.json")
    logger.info("本脚本未启动 60 轮训练，未修改 NPC GMM、记忆库、临时更新或损失算法")


if __name__ == "__main__":
    main()
