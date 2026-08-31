from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import kaggle_runner
from kaggle.run_context import generate_run_id, ensure_run_root


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


def test_run_writes_preflight_heartbeat_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    def fake_kaggle(*args, **kwargs):
        if args[:2] == ("kernels", "status"):
            return _completed("complete")
        if args[:2] == ("kernels", "output"):
            download_dir = Path(kwargs.get("cwd", tmp_path / "stage"))
            download_dir.mkdir(parents=True, exist_ok=True)
            return _completed("downloaded")
        return _completed("pushed")

    monkeypatch.setattr(kaggle_runner, "_kaggle", fake_kaggle)
    monkeypatch.setattr(kaggle_runner, "_run_command", lambda *args, **kwargs: _completed("ok"))

    result = kaggle_runner.run(stage_root=tmp_path / "stage", poll_seconds=0, timeout_seconds=1)

    heartbeat_path = tmp_path / "stage" / "runner_heartbeat.json"
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert payload["phase"] in {"outputs_complete", "runner_complete"}
    assert payload["expected_commit"] == "abc123"


def test_torch_compat_cycle_available_in_preflight_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "_kaggle", lambda *args, **kwargs: _completed("data-analysis-llm"))

    payload = kaggle_runner.preflight(stage_root=tmp_path / "stage")

    assert "torch-compat-cycle" in payload["available_commands"]


def test_torch_compat_cycle_sets_workflow_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    observed = {}

    monkeypatch.setattr(kaggle_runner, "preflight", lambda spec, stage_root: {"ready": True, "spec": spec.to_dict(), "stage_root": str(stage_root)})
    monkeypatch.setattr(kaggle_runner, "run", lambda spec, stage_root, run_id=None: observed.setdefault("workflow_mode", spec.workflow_mode) or {"status": "complete"})
    monkeypatch.setattr(kaggle_runner, "outputs", lambda spec, stage_root: {"downloaded_safe_artifacts": []})

    result = kaggle_runner.torch_compat_cycle(stage_root=tmp_path / "stage")

    assert observed["workflow_mode"] == "torch_compat"
    assert result["preflight"]["ready"] is True


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
        "smoke_training_report.json",
        "smoke_heartbeat.json",
        "smoke_breadcrumbs.jsonl",
        "smoke_failure.json",
        "dependency_preflight.json",
        "dependency_install_result.json",
    }


def test_status_error_triggers_postmortem_download(monkeypatch, tmp_path):
    def fake_kaggle(*args, **kwargs):
        if args[:2] == ("kernels", "status"):
            return _completed("error")
        if args[:2] == ("kernels", "output"):
            download_dir = Path(args[args.index("-p") + 1])
            (download_dir / "reports").mkdir(parents=True, exist_ok=True)
            (download_dir / "reports" / "smoke_breadcrumbs.jsonl").write_text(
                json.dumps({"stage": "model_loaded", "success": True, "safe_message": "ok"}) + "\n",
                encoding="utf-8",
            )
            (download_dir / "reports" / "smoke_failure.json").write_text(
                json.dumps({"stage": "model_load_started", "exception_type": "RuntimeError", "sanitized_exception_message": "boom"}),
                encoding="utf-8",
            )
            (download_dir / "kaggle.log").write_text("line1\nline2\n", encoding="utf-8")
            return _completed("downloaded")
        if args[:2] == ("kernels", "logs"):
            return _completed("line1\nline2")
        return _completed("pushed")

    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "_kaggle", fake_kaggle)

    result = kaggle_runner.status(stage_root=tmp_path / "stage")

    assert "error" in result["status"].lower()
    assert result["postmortem"]["last_breadcrumb_stage"] == "model_loaded"
    assert result["postmortem"]["smoke_failure"]["stage"] == "model_load_started"
    assert "line1" in result["postmortem"]["last_safe_log_lines"]
    assert result["postmortem"]["live_log_tail"] == "line1\nline2"


def test_run_refuses_duplicate_active_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "_status_payload", lambda spec, stage_root: {"status": "KernelWorkerStatus.RUNNING"})
    monkeypatch.setattr(kaggle_runner, "push", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("push should not run")))

    try:
        kaggle_runner.run(stage_root=tmp_path / "stage")
    except kaggle_runner.KaggleAutomationError as exc:
        assert str(exc) == "kernel_already_running"
    else:
        raise AssertionError("run should refuse to launch a duplicate session")


def test_run_command_uses_utf8_safe_subprocess(monkeypatch):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return _completed("ok")

    monkeypatch.setattr(kaggle_runner.subprocess, "run", fake_run)
    result = kaggle_runner._run_command(["echo", "hi"])

    assert result.stdout == "ok"
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"
    assert seen["env"]["PYTHONUTF8"] == "1"
    assert seen["env"]["PYTHONIOENCODING"] == "utf-8"


def test_kaggle_module_preferred_over_system_cli(monkeypatch):
    calls = []

    monkeypatch.setattr(kaggle_runner, "_kaggle_python_available", lambda: True)
    monkeypatch.setattr(kaggle_runner.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(kaggle_runner, "_run_command", lambda args, **kwargs: calls.append(args) or _completed("ok"))

    result = kaggle_runner._kaggle("kernels", "status", "user/notebook", timeout=5)

    assert result.stdout == "ok"
    assert calls[0][:3] == [kaggle_runner.sys.executable, "-m", "kaggle"]


def test_sdk_kernel_status_is_used_for_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "access_token"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "_sdk_kernel_status", lambda auth, spec: "RUNNING")
    monkeypatch.setattr(kaggle_runner, "_kaggle_checked", lambda *args, **kwargs: _completed("file1\nfile2\n"))

    result = kaggle_runner.preflight(stage_root=tmp_path / "stage")

    assert result["kernel_status"] == "RUNNING"
    assert result["kernel_exists"] is True
    assert result["ready"] is True


def test_normalize_status_text_handles_sdk_enums():
    class FakeStatus:
        value = "KernelWorkerStatus.ERROR"

    assert kaggle_runner._normalize_status_text(FakeStatus()) == "KernelWorkerStatus.ERROR"
    assert kaggle_runner._normalize_status_text("RUNNING") == "RUNNING"
    assert kaggle_runner._normalize_status_text(None) is None


def test_runner_heartbeat_and_failure_files_are_written(tmp_path):
    heartbeat = kaggle_runner._write_runner_heartbeat(
        phase="runner_started",
        kernel_ref="user/notebook",
        expected_commit="abc123",
        elapsed_seconds=1.25,
        last_status="RUNNING",
        safe_message="start",
        stage_root=tmp_path,
    )
    failure = kaggle_runner._write_runner_failure(
        phase="poll_started",
        command="kaggle kernels status",
        exc=RuntimeError("boom"),
        timeout_seconds=30,
        kernel_ref="user/notebook",
        expected_commit="abc123",
        elapsed_seconds=2.5,
        stdout="line1\nline2",
        stderr="err1\nerr2",
        last_status="RUNNING",
        stage_root=tmp_path,
    )

    heartbeat_payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    failure_payload = json.loads(failure.read_text(encoding="utf-8"))

    assert heartbeat_payload["phase"] == "runner_started"
    assert heartbeat_payload["kernel_ref"] == "user/notebook"
    assert failure_payload["phase"] == "poll_started"
    assert failure_payload["timeout_seconds"] == 30
    assert "line2" in failure_payload["safe_stdout_tail"]
    assert "err2" in failure_payload["safe_stderr_tail"]


def test_generate_run_id_has_commit_timestamp_and_suffix():
    run_id = generate_run_id(git_commit="c1fa220eaa3b22ad109ee812d6caf88aa6b4cb87", timestamp="20260830T083500Z")

    assert run_id.startswith("c1fa220-20260830T083500Z-")
    assert len(run_id.split("-")[-1]) == 4


def test_run_scoped_root_and_report_command_are_isolated(tmp_path, monkeypatch):
    run_id = "abc123-20260830T083500Z-wxyz"
    run_root = ensure_run_root(run_id, base_root=tmp_path / "runtime" / "kaggle_runner")
    (run_root / "runner_metadata.json").write_text(json.dumps({"run_id": run_id, "expected_git_commit": "abc123"}), encoding="utf-8")
    (run_root / "notebook_started.json").write_text(json.dumps({"run_id": run_id, "expected_git_commit": "abc123", "executed_git_commit": "abc123"}), encoding="utf-8")
    (run_root / "smoke_breadcrumbs.jsonl").write_text(json.dumps({"run_id": run_id, "stage": "notebook_started", "success": True}) + "\n", encoding="utf-8")
    (run_root / "smoke_heartbeat.json").write_text(json.dumps({"run_id": run_id, "stage": "notebook_started", "expected_git_commit": "abc123", "executed_git_commit": "abc123"}), encoding="utf-8")

    result = kaggle_runner.report(run_id=run_id, stage_root=tmp_path / "runtime" / "kaggle_runner")

    assert result["run_id"] == run_id
    assert result["heartbeat"]["run_id"] == run_id
    assert result["breadcrumbs"][0]["run_id"] == run_id
    assert result["remote_identity"] is None


def test_postmortem_rejects_historical_logs_without_current_run_evidence(tmp_path):
    stage_root = tmp_path / "runtime" / "kaggle_runner" / "run-a"
    stage = kaggle_runner.ensure_stage_paths(stage_root)
    log = stage.download_dir / "old.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("historical log only\n", encoding="utf-8")
    (stage.download_dir / "reports").mkdir(parents=True, exist_ok=True)
    (stage.download_dir / "reports" / "smoke_heartbeat.json").write_text(json.dumps({"run_id": "run-a", "stage": "poll_iteration", "executed_git_commit": "abc123"}), encoding="utf-8")

    payload = kaggle_runner._kaggle_postmortem(kaggle_runner.KaggleNotebookSpec(), stage_root=stage_root)

    assert payload["historical_log_detected"] is True
    assert payload["log_evidence_current_run"] is False


def test_bounded_kaggle_call_writes_failure_on_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "_kaggle", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=args, timeout=30)))

    try:
        kaggle_runner._kaggle_checked(
            "kernels",
            "status",
            "user/notebook",
            timeout=30,
            phase="poll_started",
            kernel_ref="user/notebook",
            expected_commit="abc123",
            stage_root=tmp_path,
        )
    except kaggle_runner.KaggleAutomationError as exc:
        assert str(exc) == "poll_started_timeout"
    else:
        raise AssertionError("timeout should be wrapped as KaggleAutomationError")

    failure_path = tmp_path / "runner_failure.json"
    assert failure_path.exists()
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "poll_started"
    assert payload["timeout_seconds"] == 30


def test_diagnose_reports_heartbeat_and_postmortem(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "access_token"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "_status_payload", lambda spec, stage_root: {"status": "RUNNING"})
    heartbeat_path = kaggle_runner._write_runner_heartbeat(
        phase="poll_iteration",
        kernel_ref="jiban/data-analysis-llm-semantic-extractor",
        expected_commit="abc123",
        elapsed_seconds=3.0,
        last_status="RUNNING",
        safe_message="poll",
        stage_root=tmp_path,
    )
    assert heartbeat_path.exists()

    result = kaggle_runner.diagnose(stage_root=tmp_path)

    assert result["auth"]["available"] is True
    assert result["expected_commit"] == "abc123"
    assert result["runner_heartbeat"]["phase"] == "poll_iteration"


def test_training_package_import_is_lazy_for_benchmark(monkeypatch):
    import importlib
    import sys

    sys.modules.pop("src.training", None)
    sys.modules.pop("src.training.benchmark", None)

    training_pkg = importlib.import_module("src.training")

    assert "src.training.benchmark" not in sys.modules
    assert hasattr(training_pkg, "run_planner_benchmark")
    assert "src.training.benchmark" in sys.modules


def test_torch_compat_bootstrap_is_idempotent_and_preserves_real_skip_code(monkeypatch):
    import types
    import sys
    from src.training.torch_compat import ensure_torch_dynamo_compatibility

    fake_eval_frame = types.SimpleNamespace(skip_code=lambda *args, **kwargs: "real")
    fake_torch = types.SimpleNamespace(_C=types.SimpleNamespace(_dynamo=types.SimpleNamespace(eval_frame=fake_eval_frame)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    first = ensure_torch_dynamo_compatibility()
    second = ensure_torch_dynamo_compatibility()

    assert first.skip_code_present_before is True
    assert first.skip_code_patch_applied is False
    assert second.skip_code_present_before is True
    assert second.skip_code_patch_applied is False
    assert fake_torch._C._dynamo.eval_frame.skip_code() == "real"


def test_torch_compat_bootstrap_mirrors_skip_code_to_python_eval_frame(monkeypatch):
    import types
    import sys
    from src.training.torch_compat import ensure_torch_dynamo_compatibility

    c_eval_frame = types.SimpleNamespace()
    python_eval_frame = types.SimpleNamespace()
    fake_torch = types.SimpleNamespace(
        _C=types.SimpleNamespace(_dynamo=types.SimpleNamespace(eval_frame=c_eval_frame)),
        _dynamo=types.SimpleNamespace(eval_frame=python_eval_frame),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    report = ensure_torch_dynamo_compatibility()

    assert report.skip_code_patch_applied is True
    assert callable(fake_torch._C._dynamo.eval_frame.skip_code)
    assert callable(fake_torch._dynamo.eval_frame.skip_code)
    assert fake_torch._C._dynamo.eval_frame.skip_code is fake_torch._dynamo.eval_frame.skip_code


def test_torch_compat_bootstrap_registers_skip_code_module_alias(monkeypatch):
    import types
    import sys
    from src.training.torch_compat import ensure_torch_dynamo_compatibility

    sys.modules.pop("torch._C._dynamo.eval_frame", None)
    c_eval_frame = types.SimpleNamespace()
    fake_torch = types.SimpleNamespace(_C=types.SimpleNamespace(_dynamo=types.SimpleNamespace(eval_frame=c_eval_frame)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    report = ensure_torch_dynamo_compatibility()

    assert report.skip_code_patch_applied is True
    assert "torch._C._dynamo.eval_frame" in sys.modules
    assert callable(sys.modules["torch._C._dynamo.eval_frame"].skip_code)
