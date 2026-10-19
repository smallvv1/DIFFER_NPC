"""Static regression guard: zero-step audit must load the requested checkpoint.

This test intentionally requires only Python stdlib; no PPU, dataset or torch needed.
"""
from __future__ import annotations

import ast
from pathlib import Path


AUDIT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_npc_zero_update_fp32.py"


def test_actual_torch_load_uses_cli_resolved_checkpoint():
    root = ast.parse(AUDIT_PATH.read_text(encoding="utf-8"))
    call_args = []
    for node in ast.walk(root):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "load":
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "torch":
            continue
        call_args.append(node.args[0])
    assert len(call_args) == 1, "audit expects exactly one model torch.load"
    assert isinstance(call_args[0], ast.Name)
    assert call_args[0].id == "checkpoint_path", (
        "torch.load must use --checkpoint path, not hard-coded CHECKPOINT_PATH"
    )


def test_report_hash_and_model_load_have_same_source():
    source = AUDIT_PATH.read_text(encoding="utf-8")
    assert 'checkpoint_path = args.checkpoint.expanduser().resolve()' in source
    assert 'state = torch.load(checkpoint_path,' in source
    assert '"checkpoint_sha256": sha256(checkpoint_path)' in source
    assert '"checkpoint": str(checkpoint_path)' in source
