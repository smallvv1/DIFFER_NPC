"""独立 DIFFER+NPC 训练入口。

用法示例（数据恢复后）：
``python npc/runner.py --config_file configs/ltcc/eva02_l_bio.yml --variant N3```

B0/N1/N2/N3 共用 DIFFER 的 60 epoch 配置和官方评估入口；B0 在 NPC 关闭
时不创建任何 NPC 状态，作为原始 DIFFER 的同一训练循环对照。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOME_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HOME_ROOT))  # 当前环境的原始 processor 位于 /home/processor

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import amp
from torch.utils.data import DataLoader

from config import cfg
from data import build_dataloader, build_dataset, build_img_transforms
from data.dataset_loader import ImageDataset
from loss import make_loss
from model import build_model
from npc.engine import NPCController, unwrap_model
from npc.losses import per_sample_identity_ce
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval, R1_mAP_eval_LTCC


VARIANTS = {
    "B0": {"enabled": False, "gmm": False, "dynamic": False, "memory": False},
    "N1": {"enabled": True, "gmm": True, "dynamic": True, "memory": False},
    "N2": {"enabled": True, "gmm": True, "dynamic": False, "memory": True},
    "N3": {"enabled": True, "gmm": True, "dynamic": True, "memory": True},
}

TEXT_MODEL_VERSION_INFO = {
    "EVA02-CLIP": {
        512: "B-16",
        768: "L-14",
        1024: "bigE-14",
    },
}


class MetricWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        self.scalars.setdefault(str(tag), []).append(float(value))


def grad_l2_norm(parameters) -> float:
    """只读计算当前梯度范数；不修改梯度或优化器状态。"""
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float()
            squared += float(torch.sum(value * value).item())
    return float(squared ** 0.5)


def detached_grad_l2_norm(gradients) -> float:
    """只读计算 autograd.grad 返回值的范数。"""
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            value = gradient.detach().float()
            squared += float(torch.sum(value * value).item())
    return float(squared ** 0.5)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_single_process_group(local_rank: int) -> None:
    if dist.is_available() and dist.is_initialized():
        return
    # 现有 DIFFER 的 Distributed sampler/DataLoaderX 即使在单进程也要求
    # process group；这里仅初始化 world_size=1，不改变模型或评估协议。
    init_file = f"/tmp/differ_npc_dist_{os.getpid()}"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=1,
        rank=0,
    )


def apply_variant(config, variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"未知消融设置 {variant}，可选 {sorted(VARIANTS)}")
    flags = VARIANTS[variant]
    config.MODEL.NPC.ENABLED = flags["enabled"]
    config.MODEL.NPC.GMM_ENABLED = flags["gmm"]
    config.MODEL.NPC.DYNAMIC_WEIGHT_ENABLED = flags["dynamic"]
    config.MODEL.NPC.MEMORY_LOSS_ENABLED = flags["memory"]


def move_batch(data: dict, device: torch.device) -> tuple:
    images = data["image"].to(device, non_blocking=True)
    targets = data["pid"].to(device, non_blocking=True)
    camids = data["camid"].to(device, non_blocking=True)
    clothes = data.get("clothes_id")
    if torch.is_tensor(clothes):
        clothes = clothes.to(device, non_blocking=True)
    captions = data.get("caption_feature")
    if torch.is_tensor(captions):
        captions = captions.to(device, non_blocking=True)
    return images, targets, camids, clothes, captions


def make_full_train_loader(config, dataset):
    _, transform_test = build_img_transforms(config)
    return DataLoader(
        dataset=ImageDataset(dataset.train, transform=transform_test),
        batch_size=config.DATA.TEST_BATCH,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )


def evaluate_one(config, model, dataset, val_loader, epoch, changed: bool, logger) -> dict:
    if config.DATA.DATASET == "ltcc":
        # LTCC clothes-changing 过滤同时排除同身份同相机和同服装样本；general
        # 使用项目已有的普通 Market-1501 评估器，只排除同身份同相机样本。
        evaluator = (
            R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=config.TEST.FEAT_NORM)
            if changed else
            R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=config.TEST.FEAT_NORM)
        )
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=config.TEST.FEAT_NORM)
    evaluator.reset()
    model.eval()
    for data in val_loader:
        images = data["image"].to("cuda", non_blocking=True)
        camids = data["camid"].to("cuda", non_blocking=True)
        with torch.no_grad():
            features = model(images, cam_label=camids)
        evaluator.update((features, data["pid"], data["camid"], data["clothes_id"], data["image_path"]))
    cmc, mAP, *_ = evaluator.compute()
    result = {
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": float(mAP),
        "setting": "clothes-changing" if changed else "general",
        "epoch": int(epoch),
    }
    logger.info(
        "%s epoch=%d：Rank-1 %.4f，Rank-5 %.4f，Rank-10 %.4f，mAP %.4f",
        result["setting"], epoch, result["rank1"], result["rank5"], result["rank10"], result["mAP"],
    )
    return result


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def train(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("正式 DIFFER+NPC 要求使用当前 PPU/CUDA 设备，未允许 CPU 回退")
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    apply_variant(cfg, args.variant)
    # 与原始 train.py 相同：caption 特征目录由 CLIP_DIM 决定，不能沿用
    # defaults.py 的 bigE-14 默认值覆盖 LTCC/PRCC 的正式协议。
    cfg.defrost()
    cfg.DATA.TEXT_MODEL_VERSION = (
        f"{cfg.DATA.TEXT_MODEL}-{TEXT_MODEL_VERSION_INFO[cfg.DATA.TEXT_MODEL][cfg.MODEL.CLIP_DIM]}"
    )
    cfg.freeze()
    if args.max_epochs is not None:
        cfg.defrost()
        cfg.SOLVER.MAX_EPOCHS = int(args.max_epochs)
        cfg.freeze()

    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or (PROJECT_ROOT / "outputs" / f"DIFFER_NPC_{timestamp}"))
    run_dir = output_root / cfg.DATA.DATASET / f"seed{cfg.SOLVER.SEED}" / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.defrost()
    cfg.OUTPUT_DIR = str(run_dir)
    cfg.freeze()
    set_seed(int(cfg.SOLVER.SEED))
    ensure_single_process_group(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    logger = setup_logger("DIFFER_NPC", str(run_dir), if_train=True)
    logger.info("开始实验：数据集=%s，设置=%s，种子=%d，设备=%s", cfg.DATA.DATASET, args.variant, cfg.SOLVER.SEED, torch.cuda.get_device_name(args.local_rank))
    logger.info("NPC 配置：\n%s", cfg.MODEL.NPC)
    with open(run_dir / "config.yml", "w", encoding="utf-8") as handle:
        handle.write(cfg.dump())
    save_json(run_dir / "source_snapshot" / "runtime.json", {
        "python": sys.version,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(args.local_rank),
        "device_count": torch.cuda.device_count(),
        "config_file": args.config_file,
        "variant": args.variant,
        "seed": int(cfg.SOLVER.SEED),
    })

    if cfg.DATA.DATASET == "prcc":
        train_loader, _, _, _, dataset, train_sampler, val_loader, val_loader_same = build_dataloader(cfg)
    else:
        train_loader, _, _, dataset, train_sampler, val_loader, _ = build_dataloader(cfg)
        val_loader_same = None
    model = build_model(cfg, dataset.num_train_pids, dataset.num_camera)
    if cfg.TEST.WEIGHT:
        state = torch.load(cfg.TEST.WEIGHT, map_location="cpu")
        state = {key.replace("module.", ""): value for key, value in state.items()}
        model.load_state_dict(state, strict=False)
        logger.info("加载检查点：%s", cfg.TEST.WEIGHT)
    model.to(device)

    loss_fn, center_criterion = make_loss(cfg, num_classes=dataset.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    full_train_loader = make_full_train_loader(cfg, dataset)
    # 预感知的 memory-entry pre/post loss 需要独立重新前向可靠参考条目；
    # 该数据集与 dataset.train 同源，仅使用训练集，不引入 query/gallery。
    npc = NPCController(
        cfg,
        dataset,
        optimizer,
        run_dir,
        device,
        logger,
        reference_dataset=full_train_loader.dataset,
    )
    scaler_init = 512 if cfg.DATA.DATASET == "prcc" else 1024
    scaler = amp.GradScaler(init_scale=scaler_init)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics" / "metrics_per_epoch.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "epoch", "train_loss", "raw_ce_loss", "ce_loss", "triplet_loss", "bio_loss",
        "reverse_hair_loss", "reverse_cloth_loss", "memory_loss", "memory_grad_norm",
        "total_grad_norm", "memory_grad_ratio", "lr", "updates", "amp_scale",
        "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP", "general_rank1", "general_rank5",
        "general_rank10", "general_mAP", "gmm_selected_ratio", "gmm_identity_coverage",
        "memory_valid_reference_ratio", "memory_cross_clothes_count", "memory_cross_clothes_ratio",
        "memory_invalid_ratio", "npc_weight_mean", "npc_weight_median", "npc_weight_std",
        "npc_weight_min", "npc_weight_max", "npc_weight_eq1_ratio", "npc_weight_lt09_ratio",
        "npc_weight_lt05_ratio", "npc_valid_weight_mean", "npc_valid_weight_median",
        "npc_valid_weight_std", "npc_valid_weight_lt09_ratio",
    ]
    best_rank1 = -float("inf")
    best_epoch = 0
    start_time = time.time()
    with open(metrics_path, "w", newline="", encoding="utf-8") as metrics_handle:
        metrics_writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
        metrics_writer.writeheader()
        try:
            for epoch in range(int(cfg.TRAIN.START_EPOCH), int(cfg.SOLVER.MAX_EPOCHS) + 1):
                if hasattr(train_sampler, "set_epoch"):
                    train_sampler.set_epoch(epoch)
                memory_init_stats = {}
                if npc.active and epoch >= int(cfg.MODEL.NPC.START_EPOCH) and (epoch - int(cfg.MODEL.NPC.START_EPOCH)) % int(cfg.MODEL.NPC.MEMORY_UPDATE_FREQ) == 0:
                    memory_init_stats = npc.initialize_memory(model, full_train_loader, loss_fn, epoch)
                model.train()
                scheduler.step(epoch)
                train_loss_total = 0.0
                train_raw_ce = 0.0
                train_ce = 0.0
                train_triplet = 0.0
                train_bio = 0.0
                train_reverse_hair = 0.0
                train_reverse_cloth = 0.0
                train_memory = 0.0
                train_memory_grad = 0.0
                train_total_grad = 0.0
                train_memory_grad_ratio = 0.0
                count = 0
                updates = 0
                weight_values = []
                valid_weight_values = []
                for iteration, data in enumerate(train_loader, start=1):
                    if args.max_iterations and iteration > args.max_iterations:
                        break
                    images, targets, camids, clothes, captions = move_batch(data, device)
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_center.zero_grad(set_to_none=True)
                    with amp.autocast(enabled=True):
                        score, feat = model(images, cam_label=camids)
                    npc_weights = None
                    npc_stats = {}
                    if npc.use_dynamic:
                        npc_weights, npc_stats = npc.preaware_weights(
                            model, loss_fn, data, score, feat, targets, camids, captions, clothes,
                            epoch, iteration,
                        )
                        weight_values.extend(npc_weights.detach().float().cpu().tolist())
                    with amp.autocast(enabled=True):
                        base_loss = loss_fn(
                            score, feat, targets, camids, captions, clothes_ids=clothes,
                            train_writer=MetricWriter(), step=(epoch - 1) * len(train_loader) + iteration,
                            npc_weights=npc_weights,
                        )
                        memory_loss, _ = npc.memory_supervision(feat, data)
                        total_loss = base_loss + float(cfg.MODEL.NPC.MEMORY_LOSS_WEIGHT) * memory_loss
                    raw_ce = per_sample_identity_ce(
                        score,
                        targets,
                        label_smoothing=0.1 if cfg.MODEL.IF_LABELSMOOTH == "on" else 0.0,
                    ).mean().detach().float()
                    memory_grad_norm = 0.0
                    if memory_loss.requires_grad and memory_loss.grad_fn is not None:
                        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                        memory_grad_norm = detached_grad_l2_norm(
                            torch.autograd.grad(
                                memory_loss,
                                trainable_parameters,
                                retain_graph=True,
                                allow_unused=True,
                            )
                        )
                    if not torch.isfinite(total_loss.detach()):
                        logger.error("非有限正式损失：epoch=%d iteration=%d，跳过该更新", epoch, iteration)
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    total_grad_norm = grad_l2_norm(model.parameters())
                    memory_grad_ratio = memory_grad_norm / max(total_grad_norm, 1e-12)
                    scaler.step(optimizer)
                    if "center" in cfg.MODEL.LOSS_TYPE:
                        for parameter in center_criterion.parameters():
                            if parameter.grad is not None:
                                parameter.grad.data *= (1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                        scaler.step(optimizer_center)
                    scaler.update()
                    updates += 1
                    train_loss_total += float(total_loss.detach().item())
                    train_raw_ce += float(raw_ce.item())
                    train_memory += float(memory_loss.detach().item())
                    train_memory_grad += memory_grad_norm
                    train_total_grad += total_grad_norm
                    train_memory_grad_ratio += memory_grad_ratio
                    components = getattr(loss_fn, "last_components", {})
                    if "ce" in components:
                        train_ce += float(components["ce"].detach().item())
                    if "triplet" in components:
                        train_triplet += float(components["triplet"].detach().item())
                    if "bio" in components:
                        train_bio += float(components["bio"].detach().item())
                    if "reverse_hair" in components:
                        train_reverse_hair += float(components["reverse_hair"].detach().item())
                    if "reverse_clothing" in components:
                        train_reverse_cloth += float(components["reverse_clothing"].detach().item())
                    if npc.use_dynamic:
                        valid_count = int(npc_stats.get("valid_reference_count", 0))
                        batch_weights = npc_weights.detach().float().cpu().numpy()
                        if valid_count:
                            valid_weight_values.extend(batch_weights[:0].tolist())
                            indices = npc._batch_indices(data).detach().cpu().numpy().tolist()
                            valid_weight_values.extend(
                                [float(batch_weights[row]) for row, sample_index in enumerate(indices)
                                 if bool(npc.memory_bank.valid_reference_mask[int(sample_index)])]
                            )
                    count += 1
                if args.max_iterations:
                    logger.info("调试模式达到 max_iterations=%d，停止本 epoch 后续批次", args.max_iterations)
                cc = evaluate_one(cfg, model, dataset, val_loader, epoch, True, logger)
                general = {}
                if cfg.DATA.DATASET == "prcc":
                    general = evaluate_one(cfg, model, dataset, val_loader_same, epoch, False, logger)
                else:
                    general = evaluate_one(cfg, model, dataset, val_loader, epoch, False, logger)
                if cc["rank1"] > best_rank1:
                    best_rank1 = cc["rank1"]
                    best_epoch = epoch
                    torch.save(unwrap_model(model).state_dict(), checkpoint_dir / "model_best.pth")
                memory_coverage = npc.memory_bank.coverage() if npc.memory_bank is not None else {}
                gmm_coverage = memory_init_stats.get("coverage", {}) if memory_init_stats else {}
                all_weights = np.asarray(weight_values, dtype=np.float64)
                valid_weights = np.asarray(valid_weight_values, dtype=np.float64)
                def _stat(values, operation, default="NA"):
                    return float(operation(values)) if values.size else default
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss_total / max(1, count),
                    "raw_ce_loss": train_raw_ce / max(1, count),
                    "ce_loss": train_ce / max(1, count),
                    "triplet_loss": train_triplet / max(1, count),
                    "bio_loss": train_bio / max(1, count),
                    "reverse_hair_loss": train_reverse_hair / max(1, count),
                    "reverse_cloth_loss": train_reverse_cloth / max(1, count),
                    "memory_loss": train_memory / max(1, count),
                    "memory_grad_norm": train_memory_grad / max(1, count),
                    "total_grad_norm": train_total_grad / max(1, count),
                    "memory_grad_ratio": train_memory_grad_ratio / max(1, count),
                    "lr": optimizer.param_groups[0]["lr"],
                    "updates": updates,
                    "amp_scale": scaler.get_scale(),
                    "cc_rank1": cc["rank1"], "cc_rank5": cc["rank5"], "cc_rank10": cc["rank10"], "cc_mAP": cc["mAP"],
                    "general_rank1": general.get("rank1", "NA"),
                    "general_rank5": general.get("rank5", "NA"),
                    "general_rank10": general.get("rank10", "NA"),
                    "general_mAP": general.get("mAP", "NA"),
                    "gmm_selected_ratio": gmm_coverage.get("GMM选中比例", "NA"),
                    "gmm_identity_coverage": gmm_coverage.get("身份覆盖率", "NA"),
                    "memory_valid_reference_ratio": memory_coverage.get("有效参考比例", "NA"),
                    "memory_cross_clothes_count": memory_coverage.get("跨服装参考数", "NA"),
                    "memory_cross_clothes_ratio": memory_coverage.get("跨服装参考比例", "NA"),
                    "memory_invalid_ratio": 1.0 - float(memory_coverage["有效参考比例"]) if memory_coverage else "NA",
                    "npc_weight_mean": _stat(all_weights, np.mean),
                    "npc_weight_median": _stat(all_weights, np.median),
                    "npc_weight_std": _stat(all_weights, np.std),
                    "npc_weight_min": _stat(all_weights, np.min),
                    "npc_weight_max": _stat(all_weights, np.max),
                    "npc_weight_eq1_ratio": _stat(all_weights, lambda values: np.mean(values == 1.0)),
                    "npc_weight_lt09_ratio": _stat(all_weights, lambda values: np.mean(values < 0.9)),
                    "npc_weight_lt05_ratio": _stat(all_weights, lambda values: np.mean(values < 0.5)),
                    "npc_valid_weight_mean": _stat(valid_weights, np.mean),
                    "npc_valid_weight_median": _stat(valid_weights, np.median),
                    "npc_valid_weight_std": _stat(valid_weights, np.std),
                    "npc_valid_weight_lt09_ratio": _stat(valid_weights, lambda values: np.mean(values < 0.9)),
                }
                metrics_writer.writerow(row)
                metrics_handle.flush()
                logger.info("epoch=%d 完成：正式更新=%d，平均损失=%.6f，耗时 %.1fs", epoch, updates, row["train_loss"], time.time() - start_time)
                if args.max_iterations:
                    break
            final_path = checkpoint_dir / "model_final.pth"
            torch.save(unwrap_model(model).state_dict(), final_path)
            save_json(run_dir / "metrics" / "run_summary.json", {
                "status": "completed",
                "dataset": cfg.DATA.DATASET,
                "variant": args.variant,
                "seed": int(cfg.SOLVER.SEED),
                "best_rank1": best_rank1,
                "best_epoch": best_epoch,
                "final_checkpoint": str(final_path),
                "elapsed_seconds": time.time() - start_time,
            })
        finally:
            npc.close()
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="DIFFER + NPC（NPC-ReID适配版）")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="N3")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=0, help="仅用于真实数据上的短迭代验证；0表示完整训练")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    run_directory = train(parse_args())
    print(f"实验完成：{run_directory}")
