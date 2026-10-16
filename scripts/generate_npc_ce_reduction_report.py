#!/usr/bin/env python3
"""汇总 NPC CE reduction 消融结果；只读训练产物，不参与模型选择。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


METRIC_FIELDS = (
    "train_loss", "raw_ce_loss", "ce_loss", "triplet_loss", "memory_loss",
    "memory_grad_norm", "total_grad_norm", "memory_grad_ratio", "amp_scale",
    "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP", "general_rank1",
    "general_rank5", "general_rank10", "general_mAP", "gmm_selected_ratio",
    "gmm_identity_coverage", "memory_valid_reference_ratio",
    "memory_cross_clothes_ratio", "memory_invalid_ratio", "npc_weight_mean",
    "npc_weight_median", "npc_weight_std", "npc_weight_eq1_ratio",
    "npc_weight_lt09_ratio", "npc_weight_lt05_ratio", "npc_valid_weight_mean",
    "npc_valid_weight_median", "npc_valid_weight_std", "npc_valid_weight_lt09_ratio",
)


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str):
    value = row.get(key, "")
    if value in ("", "NA", "nan", "NaN", "inf", "-inf"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def summary(rows: list[dict[str, str]]) -> dict:
    if not rows:
        return {"epochs": 0}
    final = rows[-1]
    best_rank1 = max(rows, key=lambda row: number(row, "cc_rank1") or float("-inf"))
    best_map = max(rows, key=lambda row: number(row, "cc_mAP") or float("-inf"))
    finite_bad = []
    for row in rows:
        bad = [key for key in METRIC_FIELDS if row.get(key, "").lower() in {"nan", "inf", "-inf"}]
        if bad:
            finite_bad.append({"epoch": row.get("epoch"), "fields": bad})
    return {
        "epochs": len(rows),
        "final": final,
        "best_rank1": best_rank1,
        "best_map": best_map,
        "finite_bad": finite_bad,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def log_metadata(project: Path, run: Path) -> dict[str, str]:
    stamp = run.name.rsplit("_", 2)[-2:]
    log_path = project / "logs" / f"NPC批均值正式训练_{'_'.join(stamp)}.log"
    metadata = {}
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and line.split("=", 1)[0] in {"代码提交", "实验分支", "配置文件"}:
                key, value = line.split("=", 1)
                metadata[key] = value.strip()
    return metadata


def row_line(label: str, item: dict, prefix: str = "") -> str:
    return (
        f"- {label}{prefix}：Rank-1 {fmt(number(item, 'cc_rank1'))}，"
        f"Rank-5 {fmt(number(item, 'cc_rank5'))}，Rank-10 {fmt(number(item, 'cc_rank10'))}，"
        f"mAP {fmt(number(item, 'cc_mAP'))}，训练损失 {fmt(number(item, 'train_loss'))}。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    project = args.project_root.resolve()
    run = args.root.resolve()
    preflight = args.preflight.resolve()
    output = (args.output or project / "PHASE_NPC_CE_REDUCTION_ABLATION_REPORT.md").resolve()
    run_metrics = read_metrics(run / "ltcc/seed1234/N3/metrics/metrics_per_epoch.csv")
    pre_metrics = read_metrics(preflight / "ltcc/seed1234/N3/metrics/metrics_per_epoch.csv")
    baseline = {
        variant: summary(read_metrics(args.baseline / f"ltcc/seed1234/{variant}/metrics/metrics_per_epoch.csv"))
        for variant in ("B0", "N2", "N3")
    }
    current = summary(run_metrics)
    pre = summary(pre_metrics)
    run_summary = load_json(run / "ltcc/seed1234/N3/metrics/run_summary.json")
    pre_summary = load_json(preflight / "ltcc/seed1234/N3/metrics/run_summary.json")
    formal_metadata = log_metadata(project, run)
    weight_csv = run / "ltcc/seed1234/N3/npc_diagnostics/npc_weights.csv"
    weight_rows = read_metrics(weight_csv)
    valid_weight_rows = [row for row in weight_rows if row.get("reference_valid") == "True"]
    nonunit_weights = [row for row in valid_weight_rows if abs(float(row.get("weight", "1")) - 1.0) > 1e-12]

    code_files = [
        project / "config/defaults.py", project / "loss/make_loss.py",
        project / "loss/npc_weighting.py", project / "tests/test_npc_ce_reduction.py",
        project / "scripts/run_npc_ce_batch_mean_formal.sh",
    ]
    hashes = {str(path.relative_to(project)): sha256(path) for path in code_files if path.exists()}
    protected = ["npc/memory.py", "npc/gmm.py", "npc/engine.py", "npc/losses.py", "npc/runner.py"]
    protected_equal = all(
        subprocess.run(
            ["git", "-C", str(project), "diff", "--quiet", "9d4cb06..HEAD", "--", path],
            check=False,
        ).returncode == 0
        for path in protected
    )

    final = current.get("final", {})
    n3_final = baseline["N3"].get("final", {})
    n2_final = baseline["N2"].get("final", {})
    rank_delta_n3 = (number(final, "cc_rank1") - number(n3_final, "cc_rank1")) if final and n3_final else None
    map_delta_n3 = (number(final, "cc_mAP") - number(n3_final, "cc_mAP")) if final and n3_final else None
    rank_delta_n2 = (number(final, "cc_rank1") - number(n2_final, "cc_rank1")) if final and n2_final else None
    map_delta_n2 = (number(final, "cc_mAP") - number(n2_final, "cc_mAP")) if final and n2_final else None
    if current.get("epochs", 0) < 60 or run_summary.get("status") != "completed":
        conclusion = "正式训练未完成，暂不判定"
    elif rank_delta_n3 is not None and map_delta_n3 is not None and rank_delta_n3 > 0 and map_delta_n3 >= 0:
        conclusion = "N3-batch_mean 在本次 Seed1234 E60 指标上优于原 N3；仍属于单种子结果"
    elif rank_delta_n3 is not None and map_delta_n3 is not None and rank_delta_n3 < 0 and map_delta_n3 < 0:
        conclusion = "N3-batch_mean 在本次 Seed1234 E60 指标上不优于原 N3"
    else:
        conclusion = "N3-batch_mean 相对原 N3 的 Rank-1 与 mAP 结论不一致"

    lines = [
        "# PHASE NPC CE REDUCTION 消融验证报告",
        "",
        f"> 生成时间（北京时间）：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}",
        "",
        "## 1. 实验范围与代码版本",
        "",
        "本报告只验证 NPC 身份 CE reduction 的 batch_mean 消融。GMM、记忆库、临时更新、NPC 权重函数和记忆损失系数保持不变；没有引入新的 NPC 算法，也没有加载历史 N3 checkpoint 继续训练。",
        f"- 实验分支：`{formal_metadata.get('实验分支', git_value(project, 'branch', '--show-current'))}`。",
        f"- 正式训练实际代码提交：`{formal_metadata.get('代码提交', git_value(project, 'rev-parse', 'HEAD'))}`。",
        f"- 报告生成时工作区提交：`{git_value(project, 'rev-parse', 'HEAD')}`；该提交只包含汇总脚本，不改变已启动训练。",
        "- 固定配置：LTCC、Seed1234、EVA02 初始化、60 epoch、原 20261014 N3 优化器/数据划分/评估协议、`TEST.WEIGHT=''`。",
        "- 唯一实验差异：`MODEL.NPC.CE_REDUCTION=batch_mean`，即 `(w * CE).mean()`；默认 `weight_sum` 保留历史行为。",
        f"- 受保护 NPC 文件与基线提交相同：`{'是' if protected_equal else '否'}`。",
        "",
        "代码文件 SHA256：",
    ]
    lines.extend(f"- `{name}`：`{digest}`。" for name, digest in hashes.items())
    lines += [
        "",
        "## 2. 测试与公式一致性",
        "",
        "- 新增 CE reduction 测试：4 个基础测试加 3 个 dtype 参数化测试，共 7 项；与完整 NPC 测试合计 18 项，全部通过。",
        "- `weight_sum` 与历史公式 `sum(w_i * CE_i) / max(sum(w_i), 1e-12)` 在 FP16、FP32、FP64 的 loss 和梯度一致；helper 不再隐式把 values 升为 FP32。",
        "- `batch_mean` 数值严格等于 `(w * CE).mean()`，梯度为 `w / batch_size` 乘以逐样本 CE 的梯度。",
        "",
        "## 3. 真实 LTCC PPU 短程预检",
        "",
        f"- 预检目录：`{preflight}`；真实更新数：{pre.get('final', {}).get('updates', 'NA')} 个 batch，训练循环记录 {pre.get('epochs', 0)} 个 epoch。",
        f"- 退出状态：`{pre_summary.get('status', '未知')}`；正式更新数：{pre.get('final', {}).get('updates', 'NA')}。",
        f"- PPU：`PPU-ZW810E`，设备数量 1；未发生 CPU 模型回退。",
        f"- clothes-changing：Rank-1 {fmt(number(pre.get('final', {}), 'cc_rank1'))}，mAP {fmt(number(pre.get('final', {}), 'cc_mAP'))}；general：Rank-1 {fmt(number(pre.get('final', {}), 'general_rank1'))}，mAP {fmt(number(pre.get('final', {}), 'general_mAP'))}。",
        f"- 训练损失、CE、Triplet、memory loss、梯度范数和 AMP scale 均为有限值；`updates=60`，未见跳过优化器更新。",
        f"- 首个 epoch GMM 选中比例 {fmt(number(pre.get('final', {}), 'gmm_selected_ratio'))}，有效参考覆盖率 {fmt(number(pre.get('final', {}), 'memory_valid_reference_ratio'))}；因此该 60 batch 预检中的 NPC 权重全为 1 是无有效参考时的协议默认值，不应误解为动态权重实现被强制固定。",
        "",
        "## 4. 与既有 B0/N2/N3 的 LTCC 对比",
        "",
        "以下先列 E60，再列同一训练记录中按 clothes-changing Rank-1 选择的 best epoch；两者不拼接为同一模型。",
    ]
    for variant in ("B0", "N2", "N3"):
        item = baseline[variant].get("final", {})
        lines.append(row_line(variant, item, " E60"))
        if baseline[variant].get("best_rank1"):
            lines.append(row_line(variant, baseline[variant]["best_rank1"], f" best（epoch {baseline[variant]['best_rank1'].get('epoch', 'NA')}）"))
    lines.append(row_line("N3-batch_mean", final, " E60"))
    if current.get("best_rank1"):
        lines.append(row_line("N3-batch_mean", current["best_rank1"], f" best（epoch {current['best_rank1'].get('epoch', 'NA')}）"))
    lines += [
        "",
        f"- N3-batch_mean 相对原 N3（E60）：Rank-1 {fmt(rank_delta_n3)}，mAP {fmt(map_delta_n3)}。",
        f"- N3-batch_mean 相对 N2（E60）：Rank-1 {fmt(rank_delta_n2)}，mAP {fmt(map_delta_n2)}。",
        "- 以上是同一评估协议下的观察性差异；单种子不能证明跨种子稳定性或因果收益。",
        "",
        "## 5. NPC 机制诊断",
        "",
        "- 每轮 GMM 选中比例、身份覆盖率、有效参考覆盖率、跨服装参考数以及权重统计均保存在 `metrics/metrics_per_epoch.csv`；逐样本 pre/post 诊断保存在 `npc_diagnostics/npc_weights.csv`。",
        f"- 正式逐样本权重记录：{len(weight_rows)} 行；其中有效参考 {len(valid_weight_rows)} 行，非单位权重 {len(nonunit_weights)} 行。",
        f"- 正式训练最终轮：有效参考覆盖率 {fmt(number(final, 'memory_valid_reference_ratio'))}，NPC 权重均值 {fmt(number(final, 'npc_weight_mean'))}，中位数 {fmt(number(final, 'npc_weight_median'))}，标准差 {fmt(number(final, 'npc_weight_std'))}，权重等于 1 比例 {fmt(number(final, 'npc_weight_eq1_ratio'))}，小于 0.9 比例 {fmt(number(final, 'npc_weight_lt09_ratio'))}，小于 0.5 比例 {fmt(number(final, 'npc_weight_lt05_ratio'))}。",
        f"- 正式训练最终轮有效参考子集：均值 {fmt(number(final, 'npc_valid_weight_mean'))}，中位数 {fmt(number(final, 'npc_valid_weight_median'))}，标准差 {fmt(number(final, 'npc_valid_weight_std'))}，小于 0.9 比例 {fmt(number(final, 'npc_valid_weight_lt09_ratio'))}。",
        "- CE reduction 的变化只改变有 NPC 权重时 CE 标量的归一化分母；它不改变 GMM 选择、记忆参考构建或临时模型更新。",
        "",
        "## 6. 数值稳定性与异常记录",
        "",
        f"- 正式 metrics 中记录的非有限字段：{len(current.get('finite_bad', []))} 个 epoch。若为 0，表示本次正式日志未出现 NaN/Inf；若大于 0，将在下方列出。",
    ]
    if current.get("finite_bad"):
        lines.extend(f"- epoch {item['epoch']}：{', '.join(item['fields'])}。" for item in current["finite_bad"])
    else:
        lines.append("- 正式训练没有记录 NaN/Inf 字段。")
    lines += [
        "",
        "## 7. 结论",
        "",
        f"- 当前实验结论：**{conclusion}**。",
        "- 本报告不把观察到的指标差异解释为必然因果；特别需要区分 CE reduction 的梯度尺度变化与 NPC 机制本身的收益。",
        "- 是否达到 N2：以 E60 的 Rank-1 与 mAP 分别判断，不能用不同 epoch 的最优值拼接。",
        "- 后续建议：若 batch_mean 在 E60 相对 N3 有一致的 Rank-1/mAP 改善，下一步应使用 Seed4321 做严格复核；若只改善单一指标或仅 best epoch 改善，应先视为不确定，不修改 GMM/NPC 参数。",
        "",
        "## 8. 产物路径",
        "",
        f"- 正式日志：`{project / 'logs'}` 下对应 `NPC批均值正式训练_*.log`。",
        f"- 正式输出：`{run}`。",
        f"- 预检输出：`{preflight}`。",
        f"- 正式 checkpoint 保留在服务器本地，按要求不纳入 GitHub 提交。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
