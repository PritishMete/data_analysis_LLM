from __future__ import annotations

import json
from pathlib import Path

from scripts import kaggle_runner


def test_available_commands_include_bnb_cycle():
    commands = kaggle_runner.available_commands()
    assert "diagnose" in commands
    assert "bnb-compat-cycle" in commands
    assert "torch-compat-cycle" in commands


def test_diagnose_writes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "_safe_subprocess", lambda *_, **__: type("P", (), {"returncode": 0, "stdout": "kaggle 1.0"})())
    snapshot = kaggle_runner.diagnose(stage_root=tmp_path)
    assert snapshot["repo"]["head"] == "abc123"


def test_bnb_cycle_dispatch_can_be_stubbed(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "preflight", lambda spec, stage_root: {"ready": True, "spec": spec.to_dict(), "stage_root": str(stage_root)})
    monkeypatch.setattr(kaggle_runner, "run", lambda spec, stage_root, run_id=None: {"status": "complete", "workflow_mode": spec.workflow_mode})
    monkeypatch.setattr(kaggle_runner, "outputs", lambda spec, stage_root: {"downloaded_safe_artifacts": []})
    report = kaggle_runner.bnb_compat_cycle(stage_root=tmp_path)
    assert report["preflight"]["ready"] is True

