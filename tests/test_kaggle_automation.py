from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import kaggle_runner


def _completed(stdout: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=["kaggle"], returncode=returncode, stdout=stdout, stderr="")


def test_preflight_detects_auth_dataset_and_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "_kaggle", lambda *args, **kwargs: _completed("file1\nfile2\n"))

    result = kaggle_runner.preflight(stage_root=tmp_path / "stage")

    assert result["auth"]["available"] is True
    assert result["dataset_accessible"] is True
    assert result["kernel_ref"] == "jiban/data-analysis-llm-semantic-extractor"
    assert result["ready"] is True


def test_preflight_reports_missing_auth_without_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(False, None, None, None, "missing_kaggle_credentials"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))

    result = kaggle_runner.preflight(stage_root=tmp_path / "stage")

    assert result["auth"]["available"] is False
    assert "kaggle.json" in result["one_time_action"]
    assert "token" not in json.dumps(result).lower()


def test_push_run_status_and_outputs_use_official_cli(monkeypatch, tmp_path):
    calls = []

    def fake_kaggle(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("kernels", "status"):
            return _completed("complete")
        if args[:2] == ("kernels", "output"):
            download_dir = Path(args[args.index("-p") + 1])
            download_dir.mkdir(parents=True, exist_ok=True)
            (download_dir / "final_report.json").write_text("{}", encoding="utf-8")
            (download_dir / "semantic_metrics.json").write_text("{}", encoding="utf-8")
            (download_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            (download_dir / "semantic_extractor_artifacts.zip").write_bytes(b"zip")
            return _completed("downloaded")
        return _completed("pushed")

    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "_kaggle", fake_kaggle)
    monkeypatch.setattr(kaggle_runner, "_run_command", lambda *args, **kwargs: _completed("pytest ok"))

    spec = kaggle_runner.KaggleNotebookSpec()
    push_result = kaggle_runner.push(spec, stage_root=tmp_path / "stage")
    run_result = kaggle_runner.run(spec, stage_root=tmp_path / "stage", poll_seconds=0)
    output_result = kaggle_runner.outputs(spec, stage_root=tmp_path / "stage")

    assert push_result["notebook_ref"] == "jiban/data-analysis-llm-semantic-extractor"
    assert run_result["status"] == "complete"
    assert sorted(output_result["downloaded_safe_artifacts"]) == ["artifact_manifest.json", "final_report.json", "semantic_extractor_artifacts.zip", "semantic_metrics.json"]
    assert any(call[:2] == ("kernels", "push") for call in calls)


def test_full_cycle_refuses_when_local_tests_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "_run_command", lambda *args, **kwargs: _completed("pytest failed", returncode=1))

    try:
        kaggle_runner.full_cycle(stage_root=tmp_path / "stage", tests_command=["pytest"])
    except kaggle_runner.KaggleAutomationError as exc:
        assert str(exc) == "local_tests_failed"
    else:
        raise AssertionError("full_cycle should refuse when tests fail")


def test_smoke_cycle_skips_local_tests_and_runs_kaggle(monkeypatch, tmp_path):
    calls = []

    def fake_kaggle(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("kernels", "status"):
            return _completed("complete")
        if args[:2] == ("kernels", "output"):
            download_dir = Path(args[args.index("-p") + 1])
            download_dir.mkdir(parents=True, exist_ok=True)
            (download_dir / "final_report.json").write_text("{}", encoding="utf-8")
            (download_dir / "semantic_metrics.json").write_text("{}", encoding="utf-8")
            (download_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            (download_dir / "semantic_extractor_artifacts.zip").write_bytes(b"zip")
            return _completed("downloaded")
        return _completed("pushed")

    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "_kaggle", fake_kaggle)
    monkeypatch.setattr(kaggle_runner, "_run_command", lambda *args, **kwargs: _completed("should not run tests"))

    result = kaggle_runner.smoke_cycle(stage_root=tmp_path / "stage")

    assert "preflight" in result
    assert result["run"]["status"] == "complete"
    assert any(call[:2] == ("kernels", "push") for call in calls)
    assert any(call[:2] == ("kernels", "output") for call in calls)


def test_build_kernel_metadata_and_safe_outputs(monkeypatch):
    auth = kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json")
    spec = kaggle_runner.KaggleNotebookSpec()
    metadata = kaggle_runner.build_kernel_metadata(auth=auth, spec=spec)

    assert metadata["id"] == "jiban/data-analysis-llm-semantic-extractor"
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == [kaggle_runner.DEFAULT_DATASET_REF]
    assert kaggle_runner.SAFE_OUTPUT_NAMES == {
        "final_report.json",
        "semantic_metrics.json",
        "artifact_manifest.json",
        "semantic_extractor_artifacts.zip",
    }
