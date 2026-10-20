#!/usr/bin/env python3
"""Verify two completed paired LTCC N3 ablations and write a Chinese report.

Reads results only. Never runs training, modifies checkpoints or chooses
different epochs for best Rank-1 and its corresponding mAP.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "outputs/NPC_CONFIRMATORY_20261014_140116/ltcc/seed1234"


def read_epochs(path: Path, expected_epochs: int = 60) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = list(range(1, expected_epochs + 1))
    actual = [int(x["epoch"]) for x in rows]
    if actual != expected:
        raise ValueError(f"{path}: expected epochs {expected_epochs}, got {actual}")
    for row in rows:
        for key in ("cc_rank1", "cc_mAP", "cc_rank5", "cc_rank10"):
            value = float(row[key])
            if not (math.isfinite(value) and 0 <= value <= 1):
                raise ValueError(f"Invalid {key} at epoch {row['epoch']}: {value}")
        if int(row["updates"]) < 1:
            raise ValueError(f"Invalid zero updates at epoch {row['epoch']}")
    return rows


def best_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Empty metrics")
    best = max(rows, key=lambda x: float(x["cc_rank1"]))
    last = rows[-1]
    return {
        "epoch": int(best["epoch"]),
        "best_rank1": 100 * float(best["cc_rank1"]),
        "same_checkpoint_map": 100 * float(best["cc_mAP"]),
        "final_rank1": 100 * float(last["cc_rank1"]),
        "final_map": 100 * float(last["cc_mAP"]),
        "updates": [int(row["updates"]) for row in rows],
        "best_mAP_epoch": int(max(rows, key=lambda x: float(x["cc_mAP"]))["epoch"]),
    }


def normalized_config(path: Path, momentum: bool, seed: int, epochs: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data["MODEL"]["NPC"]["PREAWARE_FP32_ANCHOR"] is not True:
        raise ValueError(f"{path}: FP32-anchor not enabled")
    if data["MODEL"]["NPC"]["TEMP_COPY_OPT_STATE"] is not momentum:
        raise ValueError(f"{path}: momentum flag mismatch")
    if int(data["SOLVER"]["SEED"]) != seed or int(data["SOLVER"]["MAX_EPOCHS"]) != epochs:
        raise ValueError(f"{path}: seed/epoch mismatch")
    if data["TEST"]["WEIGHT"]:
        raise ValueError(f"{path}: TEST.WEIGHT is set, cannot claim common EVA02 initialization")
    if data["MODEL"]["NPC"]["DYNAMIC_WEIGHT_ENABLED"] is not True:
        raise ValueError("Must enable N3 dynamic weights")
    if data["MODEL"]["NPC"]["MEMORY_LOSS_ENABLED"] is not True:
        raise ValueError("Must enable N3 memory supervision")
    # Only the existing model switch and paths may differ between paired arms.
    data.pop("OUTPUT_DIR", None)
    data["MODEL"]["NPC"].pop("TEMP_COPY_OPT_STATE", None)
    return data


def verify_pair(run_root: Path, history_root: Path, seed: int, epochs: int) -> dict:
    manifest = run_root / "experiment_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    if plan["mode"] != "formal" or plan["seed"] != seed or plan["epochs"] != epochs:
        raise ValueError("Not a matching completed 60-epoch formal ablation")
    names = ("N3_FP32_ONLY", "N3_FP32_MOMENTUM")
    modes = (False, True)
    if [x["arm"] for x in plan["arms"]] != list(names):
        raise ValueError("Expected two independent and ordered ablation arms")
    if any(x["status"] != "completed" or x.get("returncode") != 0 for x in plan["arms"]):
        raise ValueError("Incomplete or failed training arm")
    configs = []
    result = {"seed": seed, "epochs": epochs, "source_commit": plan["code_git_head"], "arms": {}}
    for name, mode in zip(names, modes):
        run_dir = run_root / name / "ltcc" / f"seed{seed}" / "N3"
        config = normalized_config(run_dir / "config.yml", mode, seed, epochs)
        configs.append(config)
        rows = read_epochs(run_dir / "metrics/metrics_per_epoch.csv", epochs)
        for ckpt in ("model_best.pth", "model_final.pth"):
            if not (run_dir / "checkpoints" / ckpt).is_file():
                raise FileNotFoundError(run_dir / "checkpoints" / ckpt)
        result["arms"][name] = best_metrics(rows)
    if configs[0] != configs[1]:
        raise ValueError("Formal ablation config changed outside momentum flag/OUTPUT_DIR")
    if result["arms"][names[0]]["updates"] != result["arms"][names[1]]["updates"]:
        raise ValueError("Epoch update counters differ across paired arms")
    for name in ("N2", "N3"):
        result["arms"]["HISTORICAL_" + name] = best_metrics(
            read_epochs(history_root / name / "metrics/metrics_per_epoch.csv", epochs)
        )
    a, b = (result["arms"][k] for k in names)
    result["delta_true_minus_fp32_only"] = {
        "rank1_pp": b["best_rank1"] - a["best_rank1"],
        "map_pp": b["same_checkpoint_map"] - a["same_checkpoint_map"],
        "final_rank1_pp": b["final_rank1"] - a["final_rank1"],
        "final_map_pp": b["final_map"] - a["final_map"],
    }
    # Updates are logged iterations; AMP skipped steps might be included in
    # the existing counter and cannot be proven away from the CSV.
    result["limits"] = [
        "One seed; no statistical generalization",
        "Same-epoch mAP accompanies best-Rank1 checkpoint",
        "Logged updates count may include AMP-skipped optimizer.step",
        "The historical N3 differs from both new arms in FP32 anchor, so it is not a clean control for momentum",
    ]
    return result


def markdown_report(result: dict) -> str:
    names = ("HISTORICAL_N2", "HISTORICAL_N3", "N3_FP32_ONLY", "N3_FP32_MOMENTUM")
    label = {
        "HISTORICAL_N2": "历史 N2：只加可靠记忆监督",
        "HISTORICAL_N3": "历史 N3：AMP 锚点，未继承动量",
        "N3_FP32_ONLY": "N3-FP32：精度校准，不继承动量",
        "N3_FP32_MOMENTUM": "N3-FP32+Momentum：精度校准并继承动量",
    }
    lines = [
        "# PHASE NPC-FP32-MOMENTUM-CONFIRMATORY — LTCC 严格配对确认结果",
        "",
        f"源代码提交：`{result['source_commit']}`；Seed={result['seed']}；epoch={result['epochs']}。",
        "两组新训练仅 TEMP_COPY_OPT_STATE 不同；PREAWARE_FP32_ANCHOR 同为 True。",
        "历史 N2/N3 仅作背景参照；不混淆历史 N3 与新两组的单变量差异。",
        "",
        "| 实验 | 最优R1轮次 | Rank-1 | 同轮 mAP | E60 Rank-1 | E60 mAP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        x = result["arms"][name]
        lines.append(
            f"| {label[name]} | {x['epoch']} | {x['best_rank1']:.3f}% | "
            f"{x['same_checkpoint_map']:.3f}% | {x['final_rank1']:.3f}% | {x['final_map']:.3f}% |"
        )
    diffs = result["delta_true_minus_fp32_only"]
    lines += [
        "",
        "**唯一新增动量状态继承的单种子差值：**",
        f"best Rank-1：{diffs['rank1_pp']:+.3f}pp；同轮 mAP：{diffs['map_pp']:+.3f}pp；"
        f"E60 Rank-1：{diffs['final_rank1_pp']:+.3f}pp；E60 mAP：{diffs['final_map_pp']:+.3f}pp。",
        "",
        "## 证据范围",
        "",
        "- 不把 best mAP 和 best Rank-1 不同 checkpoint 的结果拼接。",
        "- 单种子收益或下降只能作为初步证据；后续需第二种子和 PRCC 验证。",
        "- 需要结合每轮有效参考覆盖率、GMM PID 覆盖、原始动态权重和有效子集权重分布。",
        "- 若 FP32-only 相比历史 N3 有变化，那是 FP32 校准的独立作用，不能归因于动量。",
        "- 请检查历史 runner 的 AMP GradScaler 跳步统计；当前 updates 列可能只是循环计数。",
        "- 如 True 未超过 FP32-only，先分析权重干预方向，不要继续堆叠风险阈值或新 GMM。",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--history-root", type=Path, default=HISTORY)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()
    root = args.experiment_root.expanduser().resolve()
    outcome = verify_pair(root, args.history_root, args.seed, args.epochs)
    (root / "comparison_summary.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = root / "PHASE_NPC_FP32_MOMENTUM_CONFIRMATORY_REPORT.md"
    report.write_text(markdown_report(outcome), encoding="utf-8")
    print(f"配对分析已完成：{report}")


if __name__ == "__main__":
    main()
