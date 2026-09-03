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
        if args[:2] == ("kernels", "logs"):
            return _completed('RUN_IDENTITY_JSON={"run_id":"abc123-20260830T000000Z-test","expected_commit":"abc123","executed_commit":"abc123","started_at":1}')
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
    run_id = "abc123-20260830T000000Z-test"
    push_result = kaggle_runner.push(spec, stage_root=tmp_path / "stage")
    run_result = kaggle_runner.run(spec, stage_root=tmp_path / "stage", poll_seconds=0, run_id=run_id, expected_commit="abc123")
    output_result = kaggle_runner.outputs(spec, stage_root=tmp_path / "stage")

    assert push_result["notebook_ref"] == "jiban/data-analysis-llm-semantic-extractor"
    assert run_result["status"] == "complete"
    assert sorted(output_result["downloaded_safe_artifacts"]) == ["artifact_manifest.json", "final_report.json", "semantic_extractor_artifacts.zip", "semantic_metrics.json"]
    assert any(call[:2] == ("kernels", "push") for call in calls)


def test_prepare_only_creates_manifest_without_kaggle_submission(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/safe/auth", "access_token"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="a" * 40, dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "generate_run_id", lambda **_: "a" * 7 + "-20260902T000000Z-test")
    monkeypatch.setattr(kaggle_runner, "_kaggle", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prepare-only submitted to Kaggle")))

    result = kaggle_runner.prepare_only(stage_root=tmp_path / "stage")
    manifest_path = Path(result["submission_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["remote_submission_performed"] is False
    assert manifest["run_id"] == result["run_id"]
    assert manifest["expected_commit"] == "a" * 40
    assert manifest["current_run_id_embedded"] is True
    assert manifest["current_commit_embedded"] is True
    assert manifest["startup_marker_position_verified"] is True
    assert manifest["generated_entrypoint_sha256"] == kaggle_runner._sha256_file(Path(manifest["generated_entrypoint"]))
    assert manifest["metadata_sha256"] == kaggle_runner._sha256_file(Path(manifest["metadata_file"]))
    prepared_notebook = json.loads(Path(manifest["generated_entrypoint"]).read_text(encoding="utf-8"))
    assert prepared_notebook["cells"][0]["cell_type"] == "code"
    first_cell = "".join(prepared_notebook["cells"][0]["source"])
    assert "RUN_IDENTITY_JSON=" in first_cell
    assert "RUN_IDENTITY.json" in first_cell
    assert "flush()" in first_cell
    assert "__RUN_ID__" not in first_cell
    assert "__EXPECTED_GIT_COMMIT__" not in first_cell


def test_prepared_submission_rejects_notebook_without_first_identity_cell(tmp_path, monkeypatch):
    entrypoint = tmp_path / "semantic_extractor_training.ipynb"
    metadata = tmp_path / "kernel-metadata.json"
    entrypoint.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": ["setup"]}]}), encoding="utf-8")
    metadata.write_text(json.dumps({"code_file": entrypoint.name, "id": "jiban/data-analysis-llm-semantic-extractor"}), encoding="utf-8")
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/safe/auth", "access_token"))
    try:
        kaggle_runner._validate_prepared_submission(notebook_dir=tmp_path, run_id="new-run", expected_commit="a" * 40, spec=kaggle_runner.KaggleNotebookSpec())
    except kaggle_runner.KaggleAutomationError as exc:
        assert str(exc) == "prepared_submission_startup_identity_not_first_cell"
    else:
        raise AssertionError("notebook without identity cell was accepted")


def test_dependency_handoff_uses_one_run_dir_and_initializes_report_before_work():
    bootstrap = Path("kaggle/bootstrap_environment.py").read_text(encoding="utf-8")
    notebook = Path("kaggle/semantic_extractor_training.ipynb").read_text(encoding="utf-8")
    execute = Path("kaggle/execute_smoke_training.py").read_text(encoding="utf-8")
    assert 'dependency_report_path = report_root / "dependency_install_result.json"' in bootstrap
    assert '"status": "STARTED"' in bootstrap
    assert '"status": "SUCCESS"' in bootstrap
    assert '"status": "FAILED"' in bootstrap
    assert "KAGGLE_RUN_DIR" in bootstrap
    assert "KAGGLE_RUN_DIR" in execute
    assert "os.environ['KAGGLE_RUN_DIR'] = str(RUN_DIR)" in notebook
    assert "dependency_report_path = RUN_DIR / 'dependency_install_result.json'" in notebook
    assert "DEPENDENCY_REPORT_HANDOFF_FAILED" in notebook


def test_dependency_report_consumer_requires_success_before_model_load():
    source = Path("kaggle/semantic_extractor_training.ipynb").read_text(encoding="utf-8")
    report_guard = "if bootstrap_report.get('status') != 'SUCCESS' or not bootstrap_report.get('install_success') or not bootstrap_report.get('stack_verified'):"
    assert report_guard in source
    assert source.index(report_guard) < source.index("write_heartbeat('model_load_begin'")


def test_prepared_submission_rejects_historical_identity(tmp_path):
    entrypoint = tmp_path / "semantic_extractor_training.ipynb"
    metadata = tmp_path / "kernel-metadata.json"
    entrypoint.write_text("1bcfd66-20260901T175300Z-t6xm", encoding="utf-8")
    metadata.write_text(json.dumps({"code_file": entrypoint.name, "id": "jiban/data-analysis-llm-semantic-extractor"}), encoding="utf-8")
    spec = kaggle_runner.KaggleNotebookSpec()
    try:
        kaggle_runner._validate_prepared_submission(notebook_dir=tmp_path, run_id="new-run", expected_commit="a" * 40, spec=spec)
    except kaggle_runner.KaggleAutomationError as exc:
        assert str(exc) == "prepared_submission_identity_mismatch"
    else:
        raise AssertionError("historical source was accepted")


def test_outputs_rejects_json_from_another_run(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/safe/auth", "access_token"))
    def fake_kaggle(*args, **kwargs):
        output = Path(args[args.index("-p") + 1])
        (output / "final_report.json").write_text(json.dumps({"run_id": "old-run"}), encoding="utf-8")
        return _completed("downloaded")
    monkeypatch.setattr(kaggle_runner, "_kaggle", fake_kaggle)
    result = kaggle_runner.outputs(stage_root=tmp_path / "stage", run_id="new-run")
    assert result["downloaded_safe_artifacts"] == []


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
        if args[:2] == ("kernels", "logs"):
            return _completed('RUN_IDENTITY_JSON={"run_id":"abc123-20260830T000000Z-test","expected_commit":"abc123","executed_commit":"abc123","started_at":1}')
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
    monkeypatch.setattr(kaggle_runner, "generate_run_id", lambda **_: "abc123-20260830T000000Z-test")

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
        if args[:2] == ("kernels", "logs"):
            return _completed('RUN_IDENTITY_JSON={"run_id":"abc123-20260830T000000Z-test","expected_commit":"abc123","executed_commit":"abc123","started_at":1}')
        if args[:2] == ("kernels", "output"):
            download_dir = Path(kwargs.get("cwd", tmp_path / "stage"))
            download_dir.mkdir(parents=True, exist_ok=True)
            return _completed("downloaded")
        return _completed("pushed")

    monkeypatch.setattr(kaggle_runner, "_kaggle", fake_kaggle)
    monkeypatch.setattr(kaggle_runner, "_run_command", lambda *args, **kwargs: _completed("ok"))
    monkeypatch.setattr(kaggle_runner, "generate_run_id", lambda **_: "abc123-20260830T000000Z-test")

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
    assert "bnb-compat-cycle" in payload["available_commands"]


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


def test_bnb_compat_cycle_sets_workflow_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jiban", "/tmp/kaggle.json", "kaggle.json"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    observed = {}

    monkeypatch.setattr(kaggle_runner, "preflight", lambda spec, stage_root: {"ready": True, "spec": spec.to_dict(), "stage_root": str(stage_root)})
    monkeypatch.setattr(kaggle_runner, "run", lambda spec, stage_root, run_id=None: observed.setdefault("workflow_mode", spec.workflow_mode) or {"status": "complete"})
    monkeypatch.setattr(kaggle_runner, "outputs", lambda spec, stage_root: {"downloaded_safe_artifacts": []})

    result = kaggle_runner.bnb_compat_cycle(stage_root=tmp_path / "stage")

    assert observed["workflow_mode"] == "bnb_compat"
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
        "probe_torch_preinstall.json",
        "probe_torch_install.json",
        "probe_torch_runtime.json",
        "probe_torch_import_runtime.json",
        "probe_torch_cuda_runtime.json",
        "probe_torch_runtime_post_bnb.json",
        "bnb_compat_report.json",
        "probe_bnb_precheck.json",
        "probe_bnb_install.json",
        "probe_bnb_import.json",
            "probe_bnb_cuda.json",
            "probe_nf4.json",
            "bnb_native_diagnostic_report.json",
        "probe_bnb_native_load.json",
        "probe_bnb_cuda_dependency.json",
        "cuda_dependency_inspection.json",
        "TORCH_PREINSTALL_INSPECTION_JSON",
        "TORCH_INSTALL_RESULT_JSON",
        "TORCH_POSTINSTALL_RUNTIME_JSON",
        "TORCH_POSTINSTALL_CUDA_JSON",
        "bnb_terminal_summary.json",
        "bnb_internal_state.json",
        "bnb_native_symbols.json",
        "bnb_real_cuda_operation.json",
        "model_dependency_result.json",
        "tokenizer_result.json",
        "model_load_result.json",
        "model_device_result.json",
        "model_memory_result.json",
            "model_forward_result.json",
            "qwen_nf4_load_report.json",
            "peft_dependency_result.json",
            "kbit_preparation_result.json",
            "lora_attachment_result.json",
            "lora_parameter_result.json",
            "qlora_forward_result.json",
            "qlora_backward_result.json",
            "qlora_optimizer_result.json",
            "qlora_memory_result.json",
            "qlora_backward_report.json",
            "learning_experiment_dataset_result.json",
            "learning_experiment_privacy_result.json",
            "semantic_contract_audit.json",
            "learning_experiment_peft_result.json",
            "learning_experiment_lora_result.json",
            "learning_experiment_tokenization_result.json",
            "learning_experiment_validation_metrics.json",
            "learning_experiment_report.json",
            "learning_experiment_final_result.json",
            "semantic_corpus_audit.json",
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


def test_resolve_kaggle_executable_prefers_repo_venv_over_path(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    venv_exe = repo_root / ".venv" / "Scripts" / "kaggle.exe"
    path_exe = tmp_path / "path" / "kaggle.exe"
    python_exe = repo_root / "Python313" / "python.exe"
    venv_exe.parent.mkdir(parents=True, exist_ok=True)
    path_exe.parent.mkdir(parents=True, exist_ok=True)
    python_exe.parent.mkdir(parents=True, exist_ok=True)
    venv_exe.write_text("venv", encoding="utf-8")
    path_exe.write_text("path", encoding="utf-8")
    python_exe.write_text("python", encoding="utf-8")

    monkeypatch.setattr(kaggle_runner, "REPO_ROOT", repo_root)
    monkeypatch.setattr(kaggle_runner.sys, "executable", str(python_exe))
    monkeypatch.setattr(kaggle_runner.shutil, "which", lambda name: str(path_exe))

    assert kaggle_runner._resolve_kaggle_executable() == str(venv_exe)


def test_kaggle_auth_uses_access_token_and_resolves_username(monkeypatch, tmp_path):
    token_dir = tmp_path / ".kaggle"
    token = token_dir / "access_token.txt"
    token_dir.mkdir(parents=True, exist_ok=True)
    token.write_text("secret-token", encoding="utf-8")
    monkeypatch.setattr(kaggle_runner.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kaggle_runner, "_discover_username_from_cli", lambda: "jaistudio")

    auth = kaggle_runner.discover_kaggle_auth()

    assert auth.available is True
    assert auth.source == "access_token"
    assert auth.username == "jaistudio"
    assert "secret-token" not in json.dumps(auth.to_dict()).lower()


def test_kaggle_command_uses_resolved_executable(monkeypatch, tmp_path):
    calls = []
    exe = tmp_path / "Scripts" / "kaggle.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("kaggle", encoding="utf-8")
    monkeypatch.setattr(kaggle_runner, "_resolve_kaggle_executable", lambda: str(exe))
    monkeypatch.setattr(kaggle_runner, "_run_command", lambda args, **kwargs: calls.append(args) or _completed("ok"))

    result = kaggle_runner._kaggle("kernels", "status", "user/notebook", timeout=5)

    assert result.stdout == "ok"
    assert calls[0][0] == str(exe)


def test_sdk_kernel_status_is_used_for_preflight(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "kaggle_cli_available", lambda: True)
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jaistudio", "/tmp/kaggle.json", "access_token"))
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
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jaistudio", "/tmp/kaggle.json", "access_token"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "_status_payload", lambda spec, stage_root: {"status": "RUNNING"})
    monkeypatch.setattr(kaggle_runner, "resolve_kaggle_runtime", lambda: {"python_executable": "python.exe", "kaggle_package_available": True, "kaggle_executable": "kaggle.exe", "kaggle_cli_available": True, "auth_available": True, "auth_source": "access_token", "auth_username_resolved": True, "kernel_ref": "jaistudio/data-analysis-llm-semantic-extractor", "kernel_ref_resolved": True})
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
    assert result["kernel_ref"] == "jaistudio/data-analysis-llm-semantic-extractor"
    assert result["kernel_ref_resolved"] is True
    assert result["runner_heartbeat"]["phase"] == "poll_iteration"


def test_preflight_and_diagnose_report_resolved_kaggle_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_runner, "discover_kaggle_auth", lambda: kaggle_runner.KaggleAuthState(True, "jaistudio", "/tmp/kaggle/access_token.txt", "access_token"))
    monkeypatch.setattr(kaggle_runner, "get_repo_state", lambda: kaggle_runner.KaggleRepoState(head="abc123", dirty=False, branch="main"))
    monkeypatch.setattr(kaggle_runner, "resolve_kaggle_runtime", lambda: {
        "python_executable": "C:\\repo\\.venv\\Scripts\\python.exe",
        "kaggle_package_available": True,
        "kaggle_executable": "C:\\repo\\.venv\\Scripts\\kaggle.exe",
        "kaggle_cli_available": True,
        "auth_available": True,
        "auth_source": "access_token",
        "auth_username_resolved": True,
        "kernel_ref": "jaistudio/data-analysis-llm-semantic-extractor",
        "kernel_ref_resolved": True,
    })
    monkeypatch.setattr(kaggle_runner, "_sdk_kernel_status", lambda auth, spec: "RUNNING")
    monkeypatch.setattr(kaggle_runner, "_kaggle_checked", lambda *args, **kwargs: _completed("ok"))

    preflight_result = kaggle_runner.preflight(stage_root=tmp_path / "stage")
    diagnose_result = kaggle_runner.diagnose(stage_root=tmp_path / "stage")

    assert preflight_result["python_executable"].endswith("python.exe")
    assert preflight_result["kaggle_executable"].endswith("kaggle.exe")
    assert preflight_result["kernel_ref"] == "jaistudio/data-analysis-llm-semantic-extractor"
    assert preflight_result["kernel_ref_resolved"] is True
    assert diagnose_result["kernel_ref"] == "jaistudio/data-analysis-llm-semantic-extractor"
    assert diagnose_result["kaggle_cli_available"] is True


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
