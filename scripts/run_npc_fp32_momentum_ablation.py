#!/usr/bin/env python3
"""Prepare (dry-run by default) two paired LTCC N3 60-epoch ablation arms.

Only variable between arms: MODEL.NPC.TEMP_COPY_OPT_STATE.
Both arms enable the already zero-step validated FP32 anchor. Historical N3
(no FP32) is a separate, non-single-variable reference, never the main control.

Examples:
  python scripts/run_npc_fp32_momentum_ablation.py
  python scripts/run_npc_fp32_momentum_ablation.py --execute --smoke-iterations 30
  python scripts/run_npc_fp32_momentum_ablation.py --execute

No model, GPU or datasets are accessed in dry-run mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ltcc/eva02_l_bio.yml"
ARM_FLAGS = (("N3_FP32_ONLY", False), ("N3_FP32_MOMENTUM", True))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def git_value(*args: str) -> str:
    output = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True, capture_output=True, text=True,
    )
    return output.stdout.strip()


def arm_command(
    python: str, output_root: Path, *,
    seed: int, epochs: int, carry_momentum: bool, smoke_iterations: int,
    data_root: Path, caption_dir: Path,
) -> list[str]:
    if seed < 0 or epochs < 1 or smoke_iterations < 0:
        raise ValueError("seed>=0, epochs>=1 and smoke_iterations>=0 required")
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    command = [
        python, str(ROOT / "npc/runner.py"),
        "--config_file", str(CONFIG), "--variant", "N3",
        "--output-root", str(output_root),
        "--max-epochs", str(2 if smoke_iterations else epochs),
    ]
    if smoke_iterations:
        command.extend(["--max-iterations", str(smoke_iterations)])
    # Identical options, except the final boolean. Original NPC weight_sum
    # reduction, GMM, memory loss, sampler, seed and pretraining remain fixed.
    command.extend([
        "DATA.ROOT", str(data_root),
        "DATA.CAPTION_DIR", str(caption_dir),
        "SOLVER.SEED", str(seed),
        "MODEL.NPC.PREAWARE_FP32_ANCHOR", "True",
        "MODEL.NPC.TEMP_COPY_OPT_STATE", str(carry_momentum),
        "MODEL.NPC.GMM_THRESHOLD", "0.99",
        "MODEL.NPC.MEMORY_LOSS_WEIGHT", "0.1",
        "MODEL.NPC.TEMP_LR_SCALE", "1.0",
        "MODEL.NPC.TEMP_OPT_STEPS", "1",
        "TEST.WEIGHT", "''",
    ])
    return command


def build_plan(
    output_root: Path, *, python: str, seed: int, epochs: int,
    smoke_iterations: int, data_root: Path, caption_dir: Path,
    code_sha: str,
) -> dict:
    arms = []
    for name, carry in ARM_FLAGS:
        root = output_root / name
        command = arm_command(
            python, root, seed=seed, epochs=epochs,
            carry_momentum=carry, smoke_iterations=smoke_iterations,
            data_root=data_root, caption_dir=caption_dir,
        )
        arms.append({
            "arm": name,
            "temp_copy_opt_state": carry,
            "preaware_fp32_anchor": True,
            "output_root": str(root),
            "expected_metrics": str(root / "ltcc" / f"seed{seed}" / "N3" / "metrics" / "metrics_per_epoch.csv"),
            "command": command,
            "shell_command": shlex.join(command),
            "status": "not_started",
        })
    left, right = (list(x["command"]) for x in arms)
    assert left[6] == right[6] == "--output-root"
    key = "MODEL.NPC.TEMP_COPY_OPT_STATE"
    li, ri = left.index(key), right.index(key)
    assert li == ri
    assert left[li + 1] == "False" and right[ri + 1] == "True"
    # Only two differences are legitimate: an isolated output folder and
    # the explicit momentum state flag. Normalize both before comparing.
    left[7] = right[7] = "<arm-output>"
    left[li + 1] = right[ri + 1] = "<momentum-flag>"
    assert left == right, "Unexpected difference beyond momentum flag/output path"
    return {
        "protocol": "LTCC / N3 / FP32-only versus FP32+SGD momentum; one variable",
        "version": 1,
        "code_git_head": code_sha,
        "base_config": str(CONFIG),
        "base_config_sha256": sha256_file(CONFIG),
        "seed": seed,
        "epochs": epochs,
        "mode": "smoke" if smoke_iterations else "formal",
        "smoke_iterations": smoke_iterations,
        "data_root": str(data_root),
        "caption_dir": str(caption_dir),
        "python": python,
        "arms": arms,
    }


def write_plan(path: Path, plan: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare paired N3 FP32/SGD state ablation")
    p.add_argument("--execute", action="store_true", help="Actually use PPU; default prints only")
    p.add_argument("--smoke-iterations", type=int, default=0,
                   help="With --execute, run only N iterations (1 epoch) per arm")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--data-root", type=Path, default=Path("/home/data"))
    p.add_argument("--caption-dir", type=Path, default=Path("/home/data/DIFFER_TextCaption/LTCC_ReID"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    prefix = "NPC_FP32_MOMENTUM_SMOKE" if args.smoke_iterations else "NPC_FP32_MOMENTUM_CONFIRMATORY"
    root = (args.output_root or ROOT / "outputs" / f"{prefix}_{now}").expanduser().resolve()
    plan = build_plan(
        root, python=args.python, seed=args.seed, epochs=args.epochs,
        smoke_iterations=args.smoke_iterations,
        data_root=args.data_root.resolve(), caption_dir=args.caption_dir.resolve(),
        code_sha=git_value("rev-parse", "HEAD"),
    )
    print("Code commit:", plan["code_git_head"])
    print("Base config SHA256:", plan["base_config_sha256"])
    print("Mode:", plan["mode"], "Output:", root)
    for arm in plan["arms"]:
        print(arm["arm"], arm["shell_command"])
    if not args.execute:
        print("DRY RUN: no GPU use, no training, no output files created.")
        return

    if root.exists():
        raise FileExistsError(f"拒绝覆盖已经存在的实验目录：{root}")
    changed_tracked = git_value("status", "--porcelain", "--untracked-files=no")
    if changed_tracked:
        raise RuntimeError(
            "检测到未提交的代码或配置修改，拒绝正式训练：\n" + changed_tracked
        )
    if not args.data_root.is_dir() or not args.caption_dir.is_dir():
        raise FileNotFoundError("服务器数据根目录或 CAPTION_DIR 不存在；终止训练")
    # A git worktree sits below /home/DIFFER_NPC_worktrees/, while the
    # original processor package remains at /home/processor. The original
    # runner adds PROJECT_ROOT.parent (worktrees), not /home; fix module
    # search paths for subprocesses without mixing model source branches.
    external_parent = next((p for p in ROOT.parents if (p / "processor").is_dir()), None)
    if external_parent is None:
        raise FileNotFoundError("无法定位旧版共享 processor 包所在上级目录")
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(external_parent), child_env.get("PYTHONPATH", "")]
    )
    if plan["mode"] == "formal" and (args.seed != 1234 or args.epochs != 60):
        raise ValueError("本次正式确认实验预注册为 Seed1234/60 epoch，不能静默更改")
    root.mkdir(parents=True, exist_ok=False)
    plan["git_status_before"] = git_value("status", "--porcelain")
    manifest = root / "experiment_manifest.json"
    write_plan(manifest, plan)

    for arm in plan["arms"]:
        if git_value("rev-parse", "HEAD") != plan["code_git_head"]:
            raise RuntimeError("两个实验臂间Git HEAD发生变化，拒绝不公平配对")
        if git_value("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("两个实验臂间存在未提交代码修改，拒绝不公平配对")
        run_log = root / f"{arm['arm']}.launcher.log"
        arm["status"] = "running"
        arm["started_at_bj"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        write_plan(manifest, plan)
        with run_log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                arm["command"], cwd=str(ROOT), stdout=handle,
                stderr=subprocess.STDOUT, check=False, env=child_env,
            )
        arm["returncode"] = result.returncode
        arm["finished_at_bj"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        arm["status"] = "completed" if result.returncode == 0 else "failed"
        write_plan(manifest, plan)
        if result.returncode:
            raise RuntimeError(f"{arm['arm']} failed (exit {result.returncode}); retained {run_log}")
        # A zero exit code alone does not prove 60 epochs of valid evaluation.
        # scripts/compare_npc_fp32_momentum_ablation.py performs that check.
    print("Arms exited with code 0. Validate epoch counts and configs in the comparison script.")


if __name__ == "__main__":
    main()
