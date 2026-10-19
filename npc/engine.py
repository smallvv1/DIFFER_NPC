"""DIFFER 训练循环使用的 NPC-ReID 适配控制器。

该文件只处理 NPC 的额外状态和计算。正式模型、正式优化器、GradScaler 和
原有身份/Triplet/Bio/Reverse 损失仍由外部训练循环管理；NPC 的临时模型与
临时优化器从不写回正式模型或正式优化器状态。
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .gmm import GMMResult, fit_clean_probabilities, identity_diagnostics, reliability_coverage
from .losses import (
    calculate_npc_weights,
    extract_reid_features,
    per_sample_identity_ce,
    per_sample_reference_consistency,
    weight_statistics,
)
from .memory import MemoryBank, build_memory_references


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None


def _disabled_autocast():
    # 临时模型使用独立 FP32 图；使用新版 API 避免 PyTorch 2.4 的弃用警告。
    return torch.amp.autocast(device_type="cuda", enabled=False)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _as_path_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)):
        return [str(x) for x in value]
    return [str(value)]


def _select_batch_value(value: Any, positions: torch.Tensor) -> Any:
    """按临时更新的有效参考位置裁剪 batch，同时保留非 Tensor 元数据。"""
    if torch.is_tensor(value):
        return value.index_select(0, positions.to(value.device))
    if isinstance(value, (list, tuple)):
        selected = [value[int(index)] for index in positions.detach().cpu().tolist()]
        return type(value)(selected) if isinstance(value, tuple) else selected
    return value


class NPCController:
    """管理 GMM、训练集记忆库、临时更新和诊断输出。"""

    def __init__(
        self,
        cfg: Any,
        dataset: Any,
        optimizer: torch.optim.Optimizer,
        output_dir: str | os.PathLike[str],
        device: torch.device,
        logger: logging.Logger | None = None,
        reference_dataset: Any | None = None,
    ) -> None:
        npc_cfg = cfg.MODEL.NPC
        self.cfg = cfg
        self.npc_cfg = npc_cfg
        self.records = [dict(item) for item in dataset.train]
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        self.diag_dir = self.output_dir / "npc_diagnostics"
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.logger = logger or logging.getLogger("DIFFER.NPC")
        # 预感知需要对记忆条目本身重新前向。该数据集必须与 dataset.train
        # 保持相同索引和路径顺序，且仍然只能来自训练集。
        self.reference_dataset = reference_dataset
        self.enabled = bool(npc_cfg.ENABLED)
        self.use_gmm = self.enabled and bool(npc_cfg.GMM_ENABLED)
        self.use_dynamic = self.enabled and bool(npc_cfg.DYNAMIC_WEIGHT_ENABLED)
        self.use_memory_loss = self.enabled and bool(npc_cfg.MEMORY_LOSS_ENABLED)
        self.memory_bank: MemoryBank | None = None
        self.clean_probabilities: np.ndarray | None = None
        self.clean_mask: np.ndarray | None = None
        self.temp_model: torch.nn.Module | None = None
        self.path_to_index = {
            str(item.get("image_path")): index for index, item in enumerate(self.records)
        }
        self.weights_file = None
        self.weights_writer = None
        if self.enabled:
            self.weights_file = open(self.diag_dir / "npc_weights.csv", "w", newline="", encoding="utf-8")
            self.weights_writer = csv.DictWriter(self.weights_file, fieldnames=[
                "epoch", "iteration", "sample_index", "pid", "clothes_id", "clean_probability",
                "gmm_selected", "reference_valid", "invalid_reference_reason", "reference_index", "reference_clothes_id", "pre_identity",
                "post_identity", "pre_reference", "post_reference", "ratio", "p_ratio",
                "q_ratio", "weight", "reference_cross_clothes", "anchor_pre_identity", "anchor_post_identity",
            ])
            self.weights_writer.writeheader()
            self.weights_file.flush()

        self.logger.info(
            "NPC 配置：enabled=%s, GMM=%s, dynamic=%s, memory_loss=%s, threshold=%.4f, "
            "temp_steps=%d, lambda_MB=%.6f",
            self.enabled,
            self.use_gmm,
            self.use_dynamic,
            self.use_memory_loss,
            float(npc_cfg.GMM_THRESHOLD),
            int(npc_cfg.TEMP_OPT_STEPS),
            float(npc_cfg.MEMORY_LOSS_WEIGHT),
        )

    @property
    def active(self) -> bool:
        return self.use_dynamic or self.use_memory_loss

    def _batch_indices(self, data: dict) -> torch.Tensor:
        paths = _as_path_list(data.get("image_path", []))
        indices = []
        for path in paths:
            if path not in self.path_to_index:
                raise KeyError(f"NPC 记忆库找不到训练样本路径：{path}")
            indices.append(self.path_to_index[path])
        return torch.as_tensor(indices, dtype=torch.long, device=self.device)

    def _load_reference_batch(self, indices: torch.Tensor) -> dict[str, Any]:
        """按记忆索引加载训练集参考条目，禁止回到当前 batch 的锚点数据。

        ``reference_dataset`` 由 runner 从 ``dataset.train`` 建立的
        ``ImageDataset`` 提供，因此这里不会读取 query/gallery 或其他训练集外数据。
        没有该数据集时直接报错，避免静默退化成只在当前样本上计算预感知量。
        """
        if self.reference_dataset is None:
            raise RuntimeError(
                "NPC 预感知存在有效参考，但未提供 reference_dataset；"
                "禁止退化为仅对当前训练样本计算"
            )
        rows = [self.reference_dataset[int(index)] for index in indices.detach().cpu().tolist()]
        if not rows or any("image" not in row for row in rows):
            raise RuntimeError("NPC reference_dataset 未返回已变换的训练图像")
        paths = [str(row.get("image_path")) for row in rows]
        expected = [str(self.records[int(index)].get("image_path")) for index in indices.detach().cpu().tolist()]
        if paths != expected:
            raise RuntimeError("NPC 参考数据集索引/路径与 dataset.train 不一致")
        images = torch.stack([row["image"] for row in rows], dim=0)
        targets = torch.as_tensor([row.get("pid") for row in rows], dtype=torch.long, device=self.device)
        camids = torch.as_tensor([row.get("camid") for row in rows], dtype=torch.long, device=self.device)
        clothes = torch.as_tensor([row.get("clothes_id") for row in rows], dtype=torch.long, device=self.device)
        return {
            "image": images.to(self.device, non_blocking=True),
            "pid": targets,
            "camid": camids,
            "clothes_id": clothes,
            "image_path": paths,
        }

    def _write_json(self, name: str, payload: dict) -> None:
        with open(self.diag_dir / name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def initialize_memory(
        self,
        model: torch.nn.Module,
        full_train_loader: Iterable[dict],
        loss_fn: Any,
        epoch: int,
    ) -> dict:
        """只使用 dataset.train 的完整遍历建立/刷新 GMM 与记忆库。"""
        if not self.active:
            return {"enabled": False, "reason": "NPC未启用或没有启用动态权重/记忆监督"}

        base_model = unwrap_model(model)
        original_training = base_model.training
        base_model.train()  # DIFFER 在 train() 下才同时返回 logits 与 feat
        losses = np.zeros(len(self.records), dtype=np.float64)
        features: torch.Tensor | None = None
        feature_rows: list[tuple[int, torch.Tensor]] = []
        null_writer = _NullWriter()
        with torch.no_grad():
            for data in full_train_loader:
                images = data["image"].to(self.device, non_blocking=True)
                targets = data["pid"].to(self.device, non_blocking=True)
                camids = data["camid"].to(self.device, non_blocking=True)
                captions = data.get("caption_feature")
                if torch.is_tensor(captions):
                    captions = captions.to(self.device, non_blocking=True)
                clothes = data.get("clothes_id")
                if torch.is_tensor(clothes):
                    clothes = clothes.to(self.device, non_blocking=True)
                indices = self._batch_indices(data)
                score, feat = base_model(images, cam_label=camids)
                per_sample = per_sample_identity_ce(
                    score,
                    targets,
                    label_smoothing=0.1 if cfg_value(self.cfg, "MODEL.IF_LABELSMOOTH", "off") == "on" else 0.0,
                )
                losses[indices.detach().cpu().numpy()] = per_sample.detach().float().cpu().numpy()
                batch_features = extract_reid_features(feat).detach().float().cpu()
                feature_rows.extend(zip(indices.detach().cpu().tolist(), batch_features))

        if original_training is False:
            base_model.eval()
        feature_dim = int(feature_rows[0][1].numel()) if feature_rows else 0
        features = torch.zeros((len(self.records), feature_dim), dtype=torch.float32)
        for index, feature in feature_rows:
            features[index] = feature

        if self.use_gmm:
            gmm_result = fit_clean_probabilities(
                losses,
                threshold=float(self.npc_cfg.GMM_THRESHOLD),
                min_samples=int(self.npc_cfg.GMM_MIN_SAMPLES),
            )
            clean_probabilities = gmm_result.probabilities
            clean_mask = gmm_result.selected
            gmm_payload = {
                "epoch": int(epoch),
                "means": gmm_result.means,
                "weights": gmm_result.weights,
                "variances": gmm_result.variances,
                "clean_component": gmm_result.clean_component,
                "threshold": gmm_result.threshold,
                "fallback": gmm_result.fallback,
                "coverage": reliability_coverage(self.records, gmm_result),
            }
        else:
            clean_probabilities = np.ones(len(self.records), dtype=np.float64)
            clean_mask = np.ones(len(self.records), dtype=bool)
            gmm_result = GMMResult(
                probabilities=clean_probabilities,
                selected=clean_mask,
                means=[],
                weights=[],
                variances=[],
                clean_component=0,
                threshold=1.0,
                fallback=True,
            )
            gmm_payload = {
                "epoch": int(epoch),
                "fallback": True,
                "reason": "消融设置关闭GMM；记忆库候选使用全部训练样本",
                "coverage": {
                    "样本总数": len(self.records),
                    "GMM选中数量": len(self.records),
                    "GMM选中比例": 1.0,
                },
            }

        self.clean_probabilities = clean_probabilities
        self.clean_mask = clean_mask
        self.memory_bank = build_memory_references(self.records, clean_mask, features)
        self.memory_bank.update_features(features)
        gmm_payload["identity_diagnostics"] = identity_diagnostics(
            self.records, losses, gmm_result, self.memory_bank.reference_metadata,
        )
        self._write_json(f"gmm_epoch_{epoch:03d}.json", gmm_payload)
        self._write_json(
            f"memory_epoch_{epoch:03d}.json",
            {
                "epoch": int(epoch),
                "source": "dataset.train only",
                "coverage": self.memory_bank.coverage(),
                "reference_examples": self.memory_bank.reference_metadata[:20],
            },
        )
        with open(self.diag_dir / f"reliability_epoch_{epoch:03d}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_index", "pid", "clothes_id", "image_path", "identity_ce", "clean_probability", "gmm_selected"])
            writer.writeheader()
            for index, record in enumerate(self.records):
                writer.writerow({
                    "sample_index": index,
                    "pid": record.get("pid"),
                    "clothes_id": record.get("clothes_id"),
                    "image_path": record.get("image_path"),
                    "identity_ce": float(losses[index]),
                    "clean_probability": float(clean_probabilities[index]),
                    "gmm_selected": bool(clean_mask[index]),
                })
        memory_coverage = self.memory_bank.coverage()
        self.logger.info(
            "NPC 记忆库 epoch=%d：GMM选中 %d/%d（%.2f%%），身份覆盖 %.2f%%，有效参考 %.2f%%，"
            "有效参考中的跨服装 %.2f%%，同服装回退 %.2f%%，无效参考 %.2f%%",
            epoch,
            int(clean_mask.sum()),
            len(clean_mask),
            100.0 * float(clean_mask.mean()),
            100.0 * float(gmm_payload["coverage"].get("身份覆盖率", 0.0)),
            100.0 * float(memory_coverage["有效参考比例"]),
            100.0 * float(memory_coverage["跨服装参考比例"]),
            100.0 * float(memory_coverage["身份回退比例"]),
            100.0 * float(1.0 - memory_coverage["有效参考比例"]),
        )
        return {"enabled": True, **gmm_payload, "memory_coverage": memory_coverage}

    def _make_temporary_optimizer(self, base_model: torch.nn.Module) -> torch.optim.Optimizer:
        """按正式优化器的参数组超参创建独立临时优化器。"""
        temp_parameters = dict(self.temp_model.named_parameters())
        base_names = {id(parameter): name for name, parameter in base_model.named_parameters()}
        groups = []
        for group in self.optimizer.param_groups:
            params = [temp_parameters[base_names[id(parameter)]] for parameter in group["params"] if id(parameter) in base_names]
            options = {key: value for key, value in group.items() if key != "params" and key != "initial_lr"}
            if "lr" in options:
                options["lr"] = float(options["lr"]) * float(cfg_value(self.cfg, "MODEL.NPC.TEMP_LR_SCALE", 1.0))
            groups.append({"params": params, **options})
        return type(self.optimizer)(groups)

    def _ensure_temp_model(self, model: torch.nn.Module) -> None:
        if self.temp_model is None:
            self.temp_model = copy.deepcopy(unwrap_model(model)).to(self.device)
            self.temp_model.requires_grad_(True)

    def preaware_weights(
        self,
        model: torch.nn.Module,
        loss_fn: Any,
        data: dict,
        score: Any,
        feat: Any,
        targets: torch.Tensor,
        camids: torch.Tensor,
        captions: Any,
        clothes: Any,
        epoch: int,
        iteration: int,
    ) -> tuple[torch.Tensor, dict]:
        """对当前批次做独立临时更新，并返回逐样本 NPC 权重。"""
        if not self.use_dynamic:
            raise RuntimeError("当前消融设置未启用 NPC 动态权重")
        if self.memory_bank is None:
            raise RuntimeError("NPC 动态权重使用前记忆库尚未初始化")
        indices = self._batch_indices(data)
        reference_features, reference_valid = self.memory_bank.lookup_with_validity(indices, self.device)
        current_features = extract_reid_features(feat)
        # 锚点损失只作诊断；NPC 比值中的 p 必须来自有效 memory entry，
        # 与原 NPC 的“memory-bank loss before/after temporary update”对象一致。
        # 正式前向可能处于 AMP 半精度；NPC 比值和 CSV 诊断统一使用 FP32，
        # 以便与临时 FP32 参考条目前向安全合并。
        anchor_pre_identity = per_sample_identity_ce(score, targets).float()
        pre_identity = torch.full_like(anchor_pre_identity, float("nan"))
        pre_reference = torch.zeros_like(pre_identity)
        reference_batch = None
        reference_indices = None
        if bool(reference_valid.any()):
            reference_indices = torch.as_tensor(
                self.memory_bank.reference_indices[indices.detach().cpu().numpy()[reference_valid.detach().cpu().numpy()]],
                dtype=torch.long,
            )
            reference_batch = self._load_reference_batch(reference_indices)
            base_model = unwrap_model(model)
            base_training = base_model.training
            # DIFFER 的 forward 在 eval() 时只返回推理特征；NPC 需要训练态
            # 的 (score, feat) 接口，但这里仍然完全关闭梯度。
            base_model.train()
            with torch.no_grad(), _disabled_autocast():
                pre_reference_score, pre_reference_feat = base_model(
                    reference_batch["image"], cam_label=reference_batch["camid"]
                )
                pre_identity_valid = per_sample_identity_ce(
                    pre_reference_score, reference_batch["pid"]
                )
                pre_reference_features = extract_reid_features(pre_reference_feat)
            base_model.train(base_training)
            pre_identity[reference_valid] = pre_identity_valid
            pre_reference[reference_valid] = per_sample_reference_consistency(
                current_features[reference_valid], pre_reference_features
            )

        # 无可靠参考的样本不参与记忆监督或负面影响估计，权重明确为 1。
        weights = torch.ones_like(pre_identity)
        ratio = torch.ones_like(pre_identity)
        direction_ratios = torch.ones((pre_identity.shape[0], 2), device=pre_identity.device, dtype=pre_identity.dtype)
        post_identity = torch.full_like(pre_identity, float("nan"))
        post_reference = torch.full_like(pre_reference, float("nan"))
        finite = True
        if bool(reference_valid.any()):
            base_model = unwrap_model(model)
            self._ensure_temp_model(base_model)
            self.temp_model.load_state_dict(copy.deepcopy(base_model.state_dict()), strict=True)
            self.temp_model.train()
            temp_optimizer = self._make_temporary_optimizer(base_model)
            valid_positions = torch.nonzero(reference_valid, as_tuple=False).flatten()
            image_positions = valid_positions.to(data["image"].device)
            target_positions = valid_positions.to(targets.device)
            temp_images = data["image"].index_select(0, image_positions).to(self.device)
            temp_targets = targets.index_select(0, target_positions)
            temp_camids = camids.index_select(0, valid_positions.to(camids.device))
            temp_captions = _select_batch_value(captions, valid_positions)
            temp_clothes = _select_batch_value(clothes, valid_positions)

            for _ in range(max(1, int(self.npc_cfg.TEMP_OPT_STEPS))):
                temp_optimizer.zero_grad(set_to_none=True)
                # 临时优化只使用有可靠参考的当前样本；正式模型和正式优化器不参与此更新。
                with _disabled_autocast():
                    temp_score, temp_feat = self.temp_model(temp_images, cam_label=temp_camids)
                    _, target_counts = torch.unique(temp_targets, return_counts=True)
                    triplet_compatible = bool(
                        target_counts.numel() >= 2
                        and torch.all(target_counts == target_counts[0]).item()
                    )
                    if not triplet_compatible:
                        # DIFFER 原 Triplet hard mining 要求至少两个身份且每个
                        # 身份样本数相等；可靠参考过滤后不满足时仅用身份 CE，
                        # 不影响正式损失。
                        temp_loss = per_sample_identity_ce(temp_score, temp_targets).mean()
                    else:
                        temp_loss = loss_fn(
                            temp_score,
                            temp_feat,
                            temp_targets,
                            temp_camids,
                            temp_captions,
                            clothes_ids=temp_clothes,
                            train_writer=_NullWriter(),
                            step=iteration,
                            npc_weights=None,
                        )
                if not torch.isfinite(temp_loss.detach()):
                    finite = False
                    self.logger.warning("NPC 临时更新遇到非有限损失，当前批次权重回退为 1")
                    temp_optimizer.zero_grad(set_to_none=True)
                    break
                temp_loss.backward()
                temp_optimizer.step()
                if not all(torch.isfinite(parameter).all() for parameter in self.temp_model.parameters()):
                    finite = False
                    self.logger.warning("NPC 临时更新后参数出现非有限值，当前批次权重回退为 1")
                    break

            if finite:
                # 与正式 DIFFER 输出接口一致：eval() 只返回特征，不能用于
                # 同时取得身份 logits 和 ReID 特征。
                self.temp_model.train()
                with torch.no_grad(), _disabled_autocast():
                    post_score, post_feat = self.temp_model(temp_images, cam_label=temp_camids)
                    post_score_reference, post_feat_reference = self.temp_model(
                        reference_batch["image"], cam_label=reference_batch["camid"]
                    )
                    post_identity_valid = per_sample_identity_ce(
                        post_score_reference, reference_batch["pid"]
                    )
                    post_reference_valid = per_sample_reference_consistency(
                        extract_reid_features(post_feat), extract_reid_features(post_feat_reference)
                    )
                    anchor_post_identity_valid = per_sample_identity_ce(post_score, temp_targets)
                post_identity[reference_valid] = post_identity_valid
                post_reference[reference_valid] = post_reference_valid
                anchor_post_identity = torch.full_like(anchor_pre_identity, float("nan"))
                anchor_post_identity[reference_valid] = anchor_post_identity_valid
                self.temp_model.eval()
                ratio_valid, weights_valid, directions_valid = calculate_npc_weights(
                    pre_identity[reference_valid],
                    post_identity[reference_valid],
                    pre_reference[reference_valid],
                    post_reference[reference_valid],
                    policy=str(getattr(self.npc_cfg, "WEIGHT_POLICY", "original")),
                    smooth_beta=float(getattr(self.npc_cfg, "WEIGHT_SMOOTH_BETA", 1.0)),
                )
                ratio[reference_valid] = ratio_valid
                weights[reference_valid] = weights_valid
                direction_ratios[reference_valid] = directions_valid

        stats = {
            "finite": finite,
            **weight_statistics(weights),
            "valid_reference_count": int(reference_valid.sum().item()),
            "valid_reference_ratio": float(reference_valid.float().mean().item()),
            "ratio_mean_valid": float(ratio[reference_valid].mean().item()) if bool(reference_valid.any()) else float("nan"),
            "pre_identity_mean_valid": float(pre_identity[reference_valid].mean().item()) if bool(reference_valid.any()) else float("nan"),
            "post_identity_mean_valid": float(post_identity[reference_valid].mean().item()) if finite and bool(reference_valid.any()) else float("nan"),
            "pre_reference_mean_valid": float(pre_reference[reference_valid].mean().item()) if bool(reference_valid.any()) else float("nan"),
            "post_reference_mean_valid": float(post_reference[reference_valid].mean().item()) if finite and bool(reference_valid.any()) else float("nan"),
            "anchor_pre_identity_mean_valid": float(anchor_pre_identity[reference_valid].mean().item()) if bool(reference_valid.any()) else float("nan"),
            "anchor_post_identity_mean_valid": float(anchor_post_identity[reference_valid].mean().item()) if finite and bool(reference_valid.any()) else float("nan"),
            "preaware_loss_object": "valid_reference_entries",
        }
        if not bool(reference_valid.any()):
            anchor_post_identity = torch.full_like(anchor_pre_identity, float("nan"))
        if self.weights_writer is not None:
            refs = self.memory_bank.reference_indices[indices.detach().cpu().numpy()]
            for row_index, sample_index in enumerate(indices.detach().cpu().tolist()):
                ref_index = int(refs[row_index])
                ref_meta = self.memory_bank.reference_metadata[sample_index]
                is_valid = bool(reference_valid[row_index].item())
                self.weights_writer.writerow({
                    "epoch": epoch,
                    "iteration": iteration,
                    "sample_index": sample_index,
                    "pid": self.records[sample_index].get("pid"),
                    "clothes_id": self.records[sample_index].get("clothes_id"),
                    "clean_probability": float(self.clean_probabilities[sample_index]) if self.clean_probabilities is not None else 1.0,
                    "gmm_selected": bool(self.clean_mask[sample_index]) if self.clean_mask is not None else True,
                    "reference_valid": is_valid,
                    "invalid_reference_reason": ref_meta.get("invalid_reason") if not is_valid else "",
                    "reference_index": ref_index if is_valid else "",
                    "reference_clothes_id": self.records[ref_index].get("clothes_id") if is_valid else "",
                    "pre_identity": float(pre_identity[row_index].item()) if is_valid else "",
                    "post_identity": float(post_identity[row_index].item()) if finite and is_valid else "",
                    "pre_reference": float(pre_reference[row_index].item()) if is_valid else "",
                    "post_reference": float(post_reference[row_index].item()) if finite and is_valid else "",
                    "ratio": float(ratio[row_index].item()) if finite and is_valid else "",
                    "p_ratio": float(direction_ratios[row_index, 0].item()) if finite and is_valid else "",
                    "q_ratio": float(direction_ratios[row_index, 1].item()) if finite and is_valid else "",
                    "weight": float(weights[row_index].item()),
                    "reference_cross_clothes": bool(ref_meta.get("cross_clothes")),
                    "anchor_pre_identity": float(anchor_pre_identity[row_index].item()) if is_valid else "",
                    "anchor_post_identity": float(anchor_post_identity[row_index].item()) if finite and is_valid else "",
                })
            self.weights_file.flush()
        return weights, stats

    def memory_supervision(self, feat: Any, data: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """返回逐样本可靠参考一致性及其均值；该项只在 N2/N3 启用。"""
        if not self.use_memory_loss:
            features = extract_reid_features(feat)
            zero = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
            return zero.mean(), zero
        if self.memory_bank is None:
            raise RuntimeError("记忆监督使用前记忆库尚未初始化")
        indices = self._batch_indices(data)
        reference_features, reference_valid = self.memory_bank.lookup_with_validity(indices, self.device)
        features = extract_reid_features(feat)
        per_sample = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
        if bool(reference_valid.any()):
            per_sample[reference_valid] = per_sample_reference_consistency(
                features[reference_valid], reference_features[reference_valid]
            )
            return per_sample[reference_valid].mean(), per_sample
        return per_sample.mean(), per_sample

    def close(self) -> None:
        if self.weights_file is not None:
            self.weights_file.close()
            self.weights_file = None


def cfg_value(cfg: Any, path: str, default: Any) -> Any:
    current = cfg
    try:
        for part in path.split("."):
            current = getattr(current, part)
        return current
    except (AttributeError, KeyError):
        return default
