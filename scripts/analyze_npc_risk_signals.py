#!/usr/bin/env python3
"""Inspect already completed LTCC NPC multi-batch PPU risk logs, no model execution."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from npc.risk_signal_audit import MODES, RULES, summarize_risk_signals


def _display_pct(ratio):
    return "N/A" if ratio is None else f"{100 * ratio:.1f}%"


def render_markdown(entries):
    lines = [
        "# PHASE NPC-RISK-SIGNAL-AUDIT：双通道风险抵消与误降权审计",
        "",
        "本报告仅对已有 LTCC train-only 多批次结果进行离线分析；",
        "没有修改模型、GMM、记忆库、动态权重、损失或 checkpoint。",
        "所谓有害是正式更新后的可靠参考身份CE或跨服装一致性损失恶化代理，",
        "**并不等同于异身份邻域被破坏，也不代表最终 Rank-1/mAP 性能损失**。",
        "",
        "比较 original_mean_ratio、identity_only、consistency_only、",
        "either_channel、both_channels 五种**只读决策规则**，",
        "后四种只作为研究诊断，不是自动采用的新 NPC 方法。",
        "",
    ]
    for source, summary in entries:
        lines += [
            f"## 来源：`{source}`",
            "",
            f"检查点：`{summary['source_checkpoint']}`",
            f"SHA256：`{summary['source_checkpoint_sha256']}`",
            f"真实配对批次数：{summary['batch_count']}",
            "",
            "| 动量继承 | 有效观测 | 独立身份 | p/q相反方向 | 均值抵消风险 | 真正降权<0.9 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for mode in MODES:
            a = summary["modes"][mode]
            lines.append(
                f"| {mode.endswith('TRUE')} | {a['observations']} | "
                f"{a['unique_identity_count']} | {a['opposing_direction_count']} | "
                f"{a['either_channel_masked_by_average_count']} | "
                f"{a['actual_weight_lt_0_9_count']} |"
            )
        lines += ["", "各个阈值均为预先固定的 0、1e-5、1e-4，不依据该日志选择权重超参。", ""]
        for mode in MODES:
            vals = summary["modes"][mode]["threshold_comparisons"]
            lines += [
                f"### {mode}",
                "",
                "| 损失恶化阈值 | 风险规则 | TP/FP/FN/TN | Precision | Recall |",
                "|---:|---|---|---:|---:|",
            ]
            for threshold, obj in vals.items():
                for rule in RULES:
                    x = obj["rules"][rule]
                    lines.append(
                        f"| {threshold} | {rule} | "
                        f"{x['TP']}/{x['FP']}/{x['FN']}/{x['TN']} | "
                        f"{_display_pct(x['precision'])} | {_display_pct(x['recall'])} |"
                    )
            lines += [
                "",
                "注意：规则间的混淆矩阵是假设性诊断，不等于实际训练后的变化；",
                "不同阈值对应不同代理标签，不能把某个阈值下的Precision/Recall与另一个阈值直接混为同一任务。",
                "",
            ]
    lines += [
        "## 下一步决策",
        "",
        "1. 报告 p、q 预测方向相反及均值抵消风险的数量，检查是否导致真实有害参考被遗漏。",
        "2. 检查继承动量后新增降权是否真正对准损害严重的样本，还是主要落在极小损失变化附近。",
        "3. 本报告不允许直接把 either_channel 或新阈值引入正式 N3；若确有潜力，需另外开分支做独立预注册消融。",
        "4. 当前证据只覆盖少量身份，不可宣称 NPC 已超过 N2，也不可当成 NDP 邻域异身份风险结论。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Existing PPU temp_optimizer_multibatch_ppu.json; pass twice for Best/Final")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    folder = args.output_dir or Path("outputs") / f"NPC_RISK_SIGNAL_AUDIT_{now}"
    folder.mkdir(parents=True, exist_ok=False)
    entries = []
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        entries.append((str(path.resolve()), summarize_risk_signals(doc)))
    (folder / "risk_signal_audit_summary.json").write_text(
        json.dumps({"sources": [{"path": p, "summary": o} for p, o in entries]},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "PHASE_NPC_RISK_SIGNAL_AUDIT_REPORT.md").write_text(
        render_markdown(entries), encoding="utf-8"
    )
    print(f"只读分析完成：{folder.resolve()}")


if __name__ == "__main__":
    main()
