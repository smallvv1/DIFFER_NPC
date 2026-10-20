"""CPU-only tests of the dry-run plan and formal result validity gates."""
import csv
import json
from pathlib import Path

import pytest
import yaml

from scripts import run_npc_fp32_momentum_ablation as launcher
from scripts.compare_npc_fp32_momentum_ablation import (
    best_metrics, read_epochs, verify_pair,
)


def mock_cfg_file(path, momentum):
    conf = {
        "MODEL": {"NPC": {
            "PREAWARE_FP32_ANCHOR": True,
            "TEMP_COPY_OPT_STATE": momentum,
            "DYNAMIC_WEIGHT_ENABLED": True,
            "MEMORY_LOSS_ENABLED": True,
            "GMM_THRESHOLD": 0.99,
            "MEMORY_LOSS_WEIGHT": 0.1,
        }},
        "SOLVER": {"SEED": 1234, "MAX_EPOCHS": 60},
        "TEST": {"WEIGHT": ""},
        "OUTPUT_DIR": "/output/different",
    }
    path.write_text(yaml.safe_dump(conf), encoding="utf-8")


def mock_epochs(path, rank1_max=0.45, epochs=60):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "updates", "cc_rank1", "cc_rank5", "cc_rank10", "cc_mAP"]
        )
        writer.writeheader()
        for e in range(1, epochs + 1):
            writer.writerow({
                "epoch": e, "updates": 1192,
                "cc_rank1": rank1_max if e == 17 else 0.39,
                "cc_rank5": 0.6, "cc_rank10": 0.7,
                "cc_mAP": rank1_max / 2 if e == 17 else 0.20,
            })


def test_dry_run_arms_only_differ_by_momentum_and_output(tmp_path, monkeypatch):
    conf = tmp_path / "base.yml"
    conf.write_text("MODEL: {}", encoding="utf-8")
    monkeypatch.setattr(launcher, "CONFIG", conf)
    plan = launcher.build_plan(
        tmp_path / "run", python="/usr/local/bin/python3", seed=1234,
        epochs=60, smoke_iterations=0, data_root=tmp_path, caption_dir=tmp_path,
        code_sha="git-sha",
    )
    left, right = [x["command"] for x in plan["arms"]]
    assert left[7] != right[7]
    a = left.index("MODEL.NPC.TEMP_COPY_OPT_STATE")
    b = right.index("MODEL.NPC.TEMP_COPY_OPT_STATE")
    assert left[a + 1] == "False" and right[b + 1] == "True"
    assert "--max-iterations" not in left
    assert not (tmp_path / "run").exists()


def test_smoke_checks_epoch_two_npc_path(tmp_path, monkeypatch):
    conf = tmp_path / "base.yml"
    conf.write_text("MODEL: {}", encoding="utf-8")
    monkeypatch.setattr(launcher, "CONFIG", conf)
    cmd = launcher.arm_command(
        "python", tmp_path / "out", seed=1234, epochs=60,
        smoke_iterations=30, carry_momentum=False,
        data_root=tmp_path, caption_dir=tmp_path,
    )
    assert cmd[cmd.index("--max-epochs") + 1] == "2"
    assert cmd[cmd.index("--max-iterations") + 1] == "30"


def test_best_rank1_and_same_checkpoint_map(tmp_path):
    csv_path = tmp_path / "m.csv"
    mock_epochs(csv_path, epochs=60)
    best = best_metrics(read_epochs(csv_path))
    assert best["epoch"] == 17
    assert best["best_rank1"] == pytest.approx(45.0)
    assert best["same_checkpoint_map"] == pytest.approx(22.5)


def test_validate_completed_pair_and_reject_config_drift(tmp_path):
    root, history = tmp_path / "new", tmp_path / "old"
    root.mkdir()
    arms = []
    for name, momentum, rank in [
        ("N3_FP32_ONLY", False, 0.45),
        ("N3_FP32_MOMENTUM", True, 0.50),
    ]:
        run = root / name / "ltcc/seed1234/N3"
        (run / "metrics").mkdir(parents=True)
        (run / "checkpoints").mkdir()
        mock_cfg_file(run / "config.yml", momentum)
        mock_epochs(run / "metrics/metrics_per_epoch.csv", rank)
        for ckpt in ("model_best.pth", "model_final.pth"):
            (run / "checkpoints" / ckpt).write_bytes(b"checkpoint stub")
        arms.append({"arm": name, "status": "completed", "returncode": 0})
    for name, rank in [("N2", 0.56), ("N3", 0.50)]:
        mock_epochs(history / name / "metrics/metrics_per_epoch.csv", rank)
    (root / "experiment_manifest.json").write_text(json.dumps({
        "mode": "formal", "seed": 1234, "epochs": 60,
        "code_git_head": "SHA1", "arms": arms,
    }), encoding="utf-8")
    result = verify_pair(root, history, 1234, 60)
    assert result["delta_true_minus_fp32_only"]["rank1_pp"] == pytest.approx(5)
    assert result["delta_true_minus_fp32_only"]["map_pp"] == pytest.approx(2.5)

    cfgpath = root / "N3_FP32_MOMENTUM/ltcc/seed1234/N3/config.yml"
    conf = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
    conf["MODEL"]["NPC"]["GMM_THRESHOLD"] = 0.95
    cfgpath.write_text(yaml.safe_dump(conf), encoding="utf-8")
    with pytest.raises(ValueError, match="config changed"):
        verify_pair(root, history, 1234, 60)


def test_reject_incomplete_60_epoch_csv(tmp_path):
    path = tmp_path / "incomplete.csv"
    mock_epochs(path, epochs=59)
    with pytest.raises(ValueError, match="expected epochs 60"):
        read_epochs(path)
