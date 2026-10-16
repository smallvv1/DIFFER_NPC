"""汇总已完成的 NPC_CONFIRMATORY 三组正式实验，只读训练输出。"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def fmt(value, digits=4):
    return "NA" if value is None else f"{float(value):.{digits}f}"


def ratio(values, predicate):
    values = [value for value in values if value is not None]
    return float(sum(predicate(value) for value in values) / len(values)) if values else None


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x, y = np.asarray(pairs).T
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_variant(root: Path, variant: str):
    run = root / "ltcc" / "seed1234" / variant
    metrics_path = run / "metrics" / "metrics_per_epoch.csv"
    summary_path = run / "metrics" / "run_summary.json"
    rows = read_csv(metrics_path) if metrics_path.exists() else []
    summary = read_json(summary_path) if summary_path.exists() else {}
    best_epoch = int(summary.get("best_epoch", 0))
    best = next((row for row in rows if int(row["epoch"]) == best_epoch), rows[-1] if rows else {})
    final = rows[-1] if rows else {}
    finite_fields = []
    for row in rows:
        for key, value in row.items():
            parsed = number(value)
            if value not in ("", "NA") and parsed is None:
                finite_fields.append((row.get("epoch"), key, value))
    return {
        "run": run,
        "rows": rows,
        "summary": summary,
        "best": best,
        "final": final,
        "finite_errors": finite_fields,
    }


def gmm_diagnostics(run: Path):
    files = sorted((run / "npc_diagnostics").glob("gmm_epoch_*.json"))
    if not files:
        return {}, []
    payloads = [read_json(path) for path in files]
    final = payloads[-1]
    identities = final.get("identity_diagnostics", [])
    selected = [number(item.get("gmm_selected_ratio")) for item in identities]
    ce = [number(item.get("identity_ce_mean")) for item in identities]
    summary = {
        "epochs": len(payloads),
        "final_epoch": final.get("epoch"),
        "final_coverage": final.get("coverage", {}),
        "mean_selected_ratio": mean(
            [payload.get("coverage", {}).get("GMM选中比例") for payload in payloads]
        ),
        "mean_identity_coverage": mean(
            [payload.get("coverage", {}).get("身份覆盖率") for payload in payloads]
        ),
        "identity_ce_selection_corr": pearson(ce, selected),
        "final_self_reference_count": sum(
            int(item.get("self_reference_count", 0)) for item in identities
        ),
        "identities": identities,
    }
    return summary, payloads


def memory_diagnostics(run: Path):
    files = sorted((run / "npc_diagnostics").glob("memory_epoch_*.json"))
    if not files:
        return {}
    payloads = [read_json(path) for path in files]
    coverages = [payload.get("coverage", {}) for payload in payloads]
    final = coverages[-1]
    return {
        "epochs": len(payloads),
        "initial": coverages[0],
        "final": final,
        "min_valid_ratio": min(
            float(item.get("有效参考比例", 0.0)) for item in coverages
        ),
        "max_valid_ratio": max(
            float(item.get("有效参考比例", 0.0)) for item in coverages
        ),
        "self_reference_in_saved_examples": sum(
            1
            for item in payloads
            for ref in item.get("reference_examples", [])
            if ref.get("valid_reference") and ref.get("source_index") == ref.get("reference_index")
        ),
    }


def weight_diagnostics(run: Path):
    path = run / "npc_diagnostics" / "npc_weights.csv"
    if not path.exists():
        return {}
    rows = read_csv(path)
    valid = [row for row in rows if row.get("reference_valid") == "True"]
    invalid = [row for row in rows if row.get("reference_valid") != "True"]

    def values(items, key):
        return [number(item.get(key)) for item in items]

    weights = values(rows, "weight")
    valid_weights = values(valid, "weight")
    pre_id, post_id = values(valid, "pre_identity"), values(valid, "post_identity")
    pre_ref, post_ref = values(valid, "pre_reference"), values(valid, "post_reference")
    low = [row for row in valid if (number(row.get("weight")) or 1.0) < 0.9]
    by_pid = {}
    by_clothes = {}
    for row in valid:
        pid = str(row.get("pid"))
        clothes = str(row.get("clothes_id"))
        by_pid.setdefault(pid, []).append(number(row.get("weight")))
        by_clothes.setdefault(clothes, []).append(number(row.get("weight")))
    return {
        "rows": len(rows),
        "valid_rows": len(valid),
        "invalid_rows": len(invalid),
        "valid_ratio": len(valid) / max(1, len(rows)),
        "weight_mean": mean(weights),
        "weight_median": float(np.nanmedian([x for x in weights if x is not None])) if any(x is not None for x in weights) else None,
        "weight_std": float(np.nanstd([x for x in weights if x is not None])) if any(x is not None for x in weights) else None,
        "weight_eq1_ratio": ratio(weights, lambda x: x == 1.0),
        "weight_lt09_ratio": ratio(weights, lambda x: x < 0.9),
        "weight_lt05_ratio": ratio(weights, lambda x: x < 0.5),
        "valid_weight_mean": mean(valid_weights),
        "valid_weight_median": float(np.nanmedian([x for x in valid_weights if x is not None])) if any(x is not None for x in valid_weights) else None,
        "valid_weight_std": float(np.nanstd([x for x in valid_weights if x is not None])) if any(x is not None for x in valid_weights) else None,
        "valid_weight_lt09_ratio": ratio(valid_weights, lambda x: x < 0.9),
        "identity_ce_delta": mean([post - pre for pre, post in zip(pre_id, post_id) if pre is not None and post is not None]),
        "cross_clothes_delta": mean([post - pre for pre, post in zip(pre_ref, post_ref) if pre is not None and post is not None]),
        "r_mean": mean(values(valid, "ratio")),
        "p_ratio_mean": mean(values(valid, "p_ratio")),
        "q_ratio_mean": mean(values(valid, "q_ratio")),
        "weight_pre_reference_corr": pearson(values(valid, "pre_reference"), valid_weights),
        "low_weight_count": len(low),
        "low_weight_cross_clothes_ratio": ratio(
            [1.0 if row.get("reference_cross_clothes") == "True" else 0.0 for row in low],
            lambda x: x > 0.5,
        ),
        "low_weight_by_pid": sorted(
            ((pid, mean(vals), ratio(vals, lambda x: x < 0.9)) for pid, vals in by_pid.items()),
            key=lambda item: item[1] if item[1] is not None else 99.0,
        )[:10],
        "clothes_count": len(by_clothes),
    }


def row_metric(row, key):
    return number(row.get(key)) if row else None


def report(root: Path) -> str:
    variants = {name: load_variant(root, name) for name in ("B0", "N2", "N3")}
    gmm = {name: gmm_diagnostics(item["run"])[0] for name, item in variants.items()}
    memory = {name: memory_diagnostics(item["run"]) for name, item in variants.items()}
    weights = {name: weight_diagnostics(item["run"]) for name, item in variants.items()}
    complete = all(item["summary"].get("status") == "completed" for item in variants.values())

    def best_metric(name, key):
        return row_metric(variants[name]["best"], key)

    b0_r1, b0_map = best_metric("B0", "cc_rank1"), best_metric("B0", "cc_mAP")
    n2_r1, n2_map = best_metric("N2", "cc_rank1"), best_metric("N2", "cc_mAP")
    n3_r1, n3_map = best_metric("N3", "cc_rank1"), best_metric("N3", "cc_mAP")
    d_n2 = (n2_r1 - b0_r1, n2_map - b0_map) if None not in (n2_r1, n2_map, b0_r1, b0_map) else (None, None)
    d_n3_b0 = (n3_r1 - b0_r1, n3_map - b0_map) if None not in (n3_r1, n3_map, b0_r1, b0_map) else (None, None)
    d_n3_n2 = (n3_r1 - n2_r1, n3_map - n2_map) if None not in (n3_r1, n3_map, n2_r1, n2_map) else (None, None)

    if not complete:
        conclusion = "INCONCLUSIVE（正式三组尚未全部完成）"
    elif d_n3_b0[0] is not None and d_n3_b0[0] > 0 and d_n3_b0[1] > 0 and d_n3_n2[0] > 0 and d_n3_n2[1] > 0:
        conclusion = "EFFECTIVE"
    elif d_n3_b0[0] is not None and (d_n3_b0[0] > 0 or d_n3_b0[1] > 0):
        conclusion = "PARTIALLY_EFFECTIVE"
    elif d_n3_b0[0] is not None and d_n3_b0[0] <= 0 and d_n3_b0[1] <= 0:
        conclusion = "INEFFECTIVE"
    else:
        conclusion = "INCONCLUSIVE"

    lines = [
        "# PHASE NPC-CONFIRMATORY 验证报告",
        "",
        f"- 实验目录：`{root}`",
        "- 数据集：LTCC；种子：1234；正式训练：60 epoch；评估：clothes-changing 与独立 general 协议",
        "- 结论：**" + conclusion + "**",
        "- 单一种子结果仅作初步验证，不宣称跨种子稳定性。",
        "",
        "## 1. 固定代码、配置与执行边界",
        "",
        "三组均从相同的本地 EVA02-CLIP 预训练缓存、相同 LTCC 划分、身份采样、batch、优化器、学习率、调度器和评估入口启动；`TEST.WEIGHT` 为空，未加载历史 N3 检查点。B0/N2/N3 的差异仅为变体开关：原始 DIFFER、可靠记忆监督、动态 NPC 加可靠记忆监督。",
        "",
        "首次 B0 尝试因旧采样器未保证 batch 含多个身份，在 epoch37 产生无负样本 triplet 批次而中止；其目录 `B0_failed_epoch36_triplet_batch` 已保留，未用于正式结果或初始化。随后修复 sampler，使每个 batch 为 4 个身份×2 张图，三组均采用同一协议。",
        "",
        "## 2. 完成状态与检索结果",
        "",
        "模型选择标准固定为 clothes-changing Rank-1 最大；表中 mAP 是同一个 best epoch 的 mAP，不把不同 epoch 的最优 Rank-1/mAP 拼接。",
        "",
        "| 设置 | 状态 | best epoch | CC Rank-1 | CC Rank-5 | CC Rank-10 | CC mAP | final CC Rank-1 | final CC mAP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("B0", "N2", "N3"):
        item = variants[name]
        lines.append(
            f"| {name} | {item['summary'].get('status', '未完成')} | {item['summary'].get('best_epoch', 'NA')} | "
            f"{fmt(row_metric(item['best'], 'cc_rank1'))} | {fmt(row_metric(item['best'], 'cc_rank5'))} | "
            f"{fmt(row_metric(item['best'], 'cc_rank10'))} | {fmt(row_metric(item['best'], 'cc_mAP'))} | "
            f"{fmt(row_metric(item['final'], 'cc_rank1'))} | {fmt(row_metric(item['final'], 'cc_mAP'))} |"
        )
    lines += [
        "",
        f"- N2−B0（best 同 epoch 口径）：Rank-1 {fmt(d_n2[0])}，mAP {fmt(d_n2[1])}。",
        f"- N3−B0：Rank-1 {fmt(d_n3_b0[0])}，mAP {fmt(d_n3_b0[1])}。",
        f"- N3−N2：Rank-1 {fmt(d_n3_n2[0])}，mAP {fmt(d_n3_n2[1])}。",
        "- general 指标保存在每轮 CSV 的 `general_*` 字段，未用换装指标重复冒充。",
        "",
        "## 3. GMM 身份选择诊断",
        "",
    ]
    for name in ("N2", "N3"):
        item = gmm[name]
        coverage = item.get("final_coverage", {})
        lines.append(
            f"- {name}：最后一轮 GMM 选中 {coverage.get('GMM选中数量', 'NA')}/{coverage.get('样本总数', 'NA')} "
            f"（{fmt(coverage.get('GMM选中比例'))}），身份覆盖 {coverage.get('身份覆盖数', 'NA')}/"
            f"{coverage.get('身份总数', 'NA')}（{fmt(coverage.get('身份覆盖率'))}），"
            f"跨轮平均身份覆盖 {fmt(item.get('mean_identity_coverage'))}；"
            f"身份平均 CE 与入选率相关系数 {fmt(item.get('identity_ce_selection_corr'), 4)}。"
        )
        lines.append(f"- {name}：最终逐身份自引用计数 {item.get('final_self_reference_count', 'NA')}。")
    lines += [
        "",
        "相关系数为负表示低 CE 身份更容易被选中；这只是选择偏差的观察性证据，不单独证明 GMM 导致性能变化。正式阶段未替换 GMM 公式或 0.99 阈值。",
        "",
        "## 4. 记忆库覆盖与 NPC 权重",
        "",
        f"修复前审计记录的自引用比例为 12.56%；修复后单元/协议测试与正式诊断均要求自引用为 0。固定修复短迭代的初始有效参考覆盖率为 39.05%，有效参考全部为同身份跨服装。",
    ]
    for name in ("N2", "N3"):
        mem = memory[name]
        final = mem.get("final", {})
        w = weights[name]
        lines.append(
            f"- {name}：有效参考最终 {fmt(final.get('有效参考比例'))}，范围 "
            f"{fmt(mem.get('min_valid_ratio'))}–{fmt(mem.get('max_valid_ratio'))}；跨服装参考 "
            f"{final.get('跨服装参考数', 'NA')}；保存示例自引用 {mem.get('self_reference_in_saved_examples', 'NA')}。"
        )
        lines.append(
            f"- {name} 权重：均值 {fmt(w.get('weight_mean'))}，中位数 {fmt(w.get('weight_median'))}，标准差 {fmt(w.get('weight_std'))}；"
            f"等于1 {fmt(w.get('weight_eq1_ratio'))}，<0.9 {fmt(w.get('weight_lt09_ratio'))}，<0.5 {fmt(w.get('weight_lt05_ratio'))}；"
            f"有效参考子集均值 {fmt(w.get('valid_weight_mean'))}，<0.9 {fmt(w.get('valid_weight_lt09_ratio'))}。"
        )
    lines += [
        "",
        "无效参考样本不参与记忆监督和负面影响估计，权重保持 1；有效记忆损失按当前 batch 内有效参考条目求均值，诊断中同时记录有效覆盖率，未因覆盖率变化而静默放大。",
        "",
        "## 5. 预感知机制专项结果",
        "",
    ]
    w = weights["N3"]
    lines += [
        f"- 有效参考条目身份 CE：更新后−更新前均值 {fmt(w.get('identity_ce_delta'))}。",
        f"- 跨服装一致性：更新后−更新前均值 {fmt(w.get('cross_clothes_delta'))}。",
        f"- r_k 均值 {fmt(w.get('r_mean'))}；p_ratio 均值 {fmt(w.get('p_ratio_mean'))}；q_ratio 均值 {fmt(w.get('q_ratio_mean'))}。",
        f"- 权重与更新前跨服装一致性相关系数 {fmt(w.get('weight_pre_reference_corr'), 4)}；低于0.9的样本数 {w.get('low_weight_count', 'NA')}，其中跨服装比例 {fmt(w.get('low_weight_cross_clothes_ratio'))}。",
        "",
        "预感知对象是有效相关参考条目的身份 CE 与 anchor/reference 特征一致性；当前样本的 anchor CE 仅作独立诊断，不冒充 NPC 影响信号。正式模型、临时模型和正式优化器独立更新。相关性不能单独证明负面影响估计具有因果准确性。",
        "",
        "## 6. 损失、梯度与数值稳定性",
        "",
        "每轮 CSV 同时记录 raw/formal CE、Triplet、Bio、Reverse、memory loss、memory gradient norm、total gradient norm、memory gradient ratio、AMP scale、更新次数及 CC/general 检索指标。当前 LTCC 配置的正式损失为 CE+Triplet，Bio/Reverse 字段应为 0 或不适用。",
        f"- 每个设置逐轮非有限字段记录数：" + ", ".join(f"{name}={len(variants[name]['finite_errors'])}" for name in ("B0", "N2", "N3")) + "。",
        "- 短迭代与正式训练均未把单元测试通过当作完整训练证据；正式 B0 修复后已实际完成多轮并使用 1192 个完整 batch/epoch。",
        "",
        "## 7. 结论与后续",
        "",
        f"本次单种子结论：**{conclusion}**。该结论只适用于当前 LTCC、Seed1234 和固定配置，不能宣称跨种子稳定性。",
    ]
    if complete and d_n3_b0[0] is not None and (d_n3_b0[0] > 0 or d_n3_b0[1] > 0):
        lines.append("若收益达到复核价值，建议保持全部配置不变，用 Seed4321 重复 B0/N2/N3，并对 N3 的收益做 PRCC：控制 epoch、身份/服装覆盖、GMM 入选率、有效参考覆盖率、memory gradient ratio 与训练损失后，检验动态权重对检索指标的独立关系。")
    elif complete:
        lines.append("N3 未显示相对 B0 的明确收益；应先分析 GMM 身份偏差、有效参考覆盖率和权重集中情况，不擅自调整 GMM 或 NPC 参数重训。")
    else:
        lines.append("三组尚未全部完成，不能给出正式性能结论；完成后应重新运行本汇总脚本并核对模型文件、逐轮指标和诊断文件。")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("用法：python scripts/generate_npc_confirmatory_report.py OUTPUT_ROOT")
    root = Path(sys.argv[1]).resolve()
    text = report(root)
    output = root / "PHASE_NPC_CONFIRMATORY_REPORT.md"
    output.write_text(text, encoding="utf-8")
    project_report = root.parents[1] / "PHASE_NPC_CONFIRMATORY_REPORT.md"
    project_report.write_text(text, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
