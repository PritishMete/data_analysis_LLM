from __future__ import annotations

import importlib.metadata as metadata
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
except Exception:  # pragma: no cover - optional in local test env
    KaggleClient = None  # type: ignore[assignment]
    KaggleEnv = None  # type: ignore[assignment]
    ApiGetKernelSessionStatusRequest = None  # type: ignore[assignment]

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from kaggle.run_context import ensure_run_root, generate_run_id, read_json, resolve_current_run_id, run_root_for, write_json  # type: ignore[no-redef]
else:
    from kaggle.run_context import ensure_run_root, generate_run_id, read_json, resolve_current_run_id, run_root_for, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = REPO_ROOT / "kaggle" / "semantic_extractor_training.ipynb"
DEFAULT_DATASET_REF = "jaistudio/data-analysis-llm"
DEFAULT_NOTEBOOK_SLUG = "data-analysis-llm-semantic-extractor"
DEFAULT_NOTEBOOK_TITLE = "Data Analysis LLM Semantic Extractor"
DEFAULT_STAGE_ROOT = REPO_ROOT / "runtime" / "kaggle_runner"
RUNNER_HEARTBEAT_PATH = DEFAULT_STAGE_ROOT / "runner_heartbeat.json"
RUNNER_FAILURE_PATH = DEFAULT_STAGE_ROOT / "runner_failure.json"
RUNNER_METADATA_NAME = "runner_metadata.json"
SUBMISSION_ATTEMPT_NAME = "submission_attempt.json"
REMOTE_IDENTITY_NAME = "remote_identity.json"
RETRIEVAL_REPORT_NAME = "retrieval_report.json"
DEFAULT_OUTPUTS_TIMEOUT_SECONDS = 600
DEFAULT_OUTPUTS_RETRY_COUNT = 1
SAFE_OUTPUT_NAMES = {
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


def _current_commit() -> str | None:
    return get_repo_state().head


def _safe_subprocess(command: list[str], *, timeout: int = 120, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

WINDOWS_EXE_NAME = "kaggle.exe" if os.name == "nt" else "kaggle"


class KaggleAutomationError(RuntimeError):
    pass


@dataclass(slots=True)
class KaggleCommandResult:
    command_safe: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float

    @property
    def returncode(self) -> int | None:
        return self.exit_code


def _safe_command(args: list[str]) -> list[str]:
    redacted = []
    for arg in args:
        value = str(arg)
        if any(marker in value.lower() for marker in ("token", "secret", "password", "api_key")):
            redacted.append("[REDACTED]")
        else:
            redacted.append(value)
    return redacted


@dataclass(slots=True)
class KaggleAuthState:
    available: bool
    username: str | None
    config_path: str | None
    source: str | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KaggleRepoState:
    head: str | None
    dirty: bool
    branch: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KaggleNotebookSpec:
    dataset_ref: str = DEFAULT_DATASET_REF
    notebook_slug: str = DEFAULT_NOTEBOOK_SLUG
    title: str = DEFAULT_NOTEBOOK_TITLE
    language: str = "python"
    kernel_type: str = "notebook"
    enable_gpu: bool = True
    enable_internet: bool = True
    code_file: str = NOTEBOOK_SOURCE.name
    output_dir: str = "output"
    workflow_mode: str = "smoke"

    def notebook_ref(self, username: str | None) -> str:
        owner = username or "unknown"
        return f"{owner}/{self.notebook_slug}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KagglePaths:
    stage_root: Path
    notebook_dir: Path
    download_dir: Path
    logs_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _resolve_stage_root(stage_root: Path = DEFAULT_STAGE_ROOT, run_id: str | None = None) -> Path:
    if run_id:
        return run_root_for(run_id, base_root=stage_root)
    if stage_root == DEFAULT_STAGE_ROOT and (stage_root / "smoke_runs").exists():
        runs_root = stage_root / "smoke_runs"
        candidates = sorted(path for path in runs_root.iterdir() if path.is_dir())
        if candidates:
            return candidates[-1]
    if stage_root.name == "kaggle_runner":
        return stage_root
    return stage_root


def _run_id_from_stage_root(stage_root: Path) -> str | None:
    if stage_root.name and stage_root.parent.name == "smoke_runs":
        return stage_root.name
    return None


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_command(args: list[str], *, check: bool = False, timeout: int | None = None, cwd: str | None = None) -> KaggleCommandResult:
    del check  # Results are always returned, including nonzero exits.
    started = time.perf_counter()
    command_safe = _safe_command(args)
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout if timeout is not None else 120,
            cwd=cwd,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        return KaggleCommandResult(command_safe, completed.returncode, completed.stdout or "", completed.stderr or "", False, round(time.perf_counter() - started, 3))
    except subprocess.TimeoutExpired as exc:
        return KaggleCommandResult(command_safe, None, str(exc.stdout or ""), str(exc.stderr or ""), True, round(time.perf_counter() - started, 3))
    except OSError as exc:
        return KaggleCommandResult(command_safe, None, "", str(exc), False, round(time.perf_counter() - started, 3))


def _safe_subprocess(args: list[str], *, check: bool = False, timeout: int | None = None, cwd: str | None = None) -> KaggleCommandResult:
    return _run_command(args, check=check, timeout=timeout, cwd=cwd)


def _safe_tail(text: str | None, *, lines: int = 40) -> str | None:
    if not text:
        return None
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def _safe_cli_output(result: Any) -> str:
    text = getattr(result, "stdout", "") or ""
    return text.strip()


def _command_result_payload(result: Any, command: list[str]) -> dict[str, Any]:
    return {
        "command_safe": getattr(result, "command_safe", _safe_command(command)),
        "exit_code": getattr(result, "exit_code", getattr(result, "returncode", None)),
        "stdout": getattr(result, "stdout", "") or "",
        "stderr": getattr(result, "stderr", "") or "",
        "timed_out": bool(getattr(result, "timed_out", False)),
        "duration_seconds": getattr(result, "duration_seconds", None),
    }


def parse_run_identity_marker(text: str | None, *, run_id: str, expected_commit: str | None) -> dict[str, Any] | None:
    """Return a matching remote identity marker, never accepting stale output."""
    if not text:
        return None
    for line in text.splitlines():
        if not line.startswith("RUN_IDENTITY_JSON="):
            continue
        try:
            payload = json.loads(line.split("=", 1)[1])
        except (TypeError, json.JSONDecodeError):
            continue
        if payload.get("run_id") != run_id:
            continue
        if expected_commit and payload.get("expected_commit") != expected_commit:
            continue
        if expected_commit and payload.get("executed_commit") != expected_commit:
            continue
        if not payload.get("started_at"):
            continue
        return payload
    return None


def fresh_run_proven(text: str | None, *, run_id: str, expected_commit: str | None) -> bool:
    return parse_run_identity_marker(text, run_id=run_id, expected_commit=expected_commit) is not None


def _normalize_status_text(status: Any | None) -> str | None:
    if status is None:
        return None
    value = getattr(status, "value", status)
    return str(value)


def _current_python_executable() -> str:
    return str(Path(sys.executable).resolve())


def _resolve_kaggle_executable() -> str | None:
    candidates: list[Path] = []
    python_path = Path(sys.executable).resolve()
    candidates.append(python_path.parent / WINDOWS_EXE_NAME)
    candidates.append(python_path.parent / "Scripts" / WINDOWS_EXE_NAME)
    candidates.append(REPO_ROOT / ".venv" / "Scripts" / WINDOWS_EXE_NAME)
    candidates.append(REPO_ROOT / "venv" / "Scripts" / WINDOWS_EXE_NAME)
    which_path = shutil.which("kaggle")
    if which_path:
        candidates.append(Path(which_path))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return str(candidate)
    return None


def _kaggle_module_available() -> bool:
    try:
        metadata.distribution("kaggle")
        return True
    except Exception:
        return False


def _kaggle_command_available() -> bool:
    return _resolve_kaggle_executable() is not None


def _sdk_kernel_status(auth: KaggleAuthState, spec: KaggleNotebookSpec) -> str | None:
    if not auth.available or not auth.username or KaggleClient is None or KaggleEnv is None or ApiGetKernelSessionStatusRequest is None:
        return None
    client = KaggleClient(KaggleEnv.PROD, username=auth.username)
    request = ApiGetKernelSessionStatusRequest()
    request._user_name = auth.username
    request._kernel_slug = spec.notebook_slug
    try:
        response = client.kernels.kernels_api_client.get_kernel_session_status(request)
        status = getattr(response, "status", None)
        return _normalize_status_text(status)
    except Exception:
        return None


def kaggle_cli_available() -> bool:
    return _kaggle_command_available()


def available_commands() -> list[str]:
    return [
        "preflight",
        "push",
        "run",
        "status",
        "outputs",
        "full-cycle",
        "smoke-cycle",
        "torch-compat-cycle",
        "bnb-compat-cycle",
        "qwen-nf4-load-cycle",
        "bnb-native-diagnose",
        "diagnose",
        "report",
    ]


def _kaggle_python_available() -> bool:
    try:
        import importlib.metadata as metadata

        return metadata.version("kaggle") is not None
    except Exception:
        return False


def discover_kaggle_auth() -> KaggleAuthState:
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    candidates = []
    if config_dir:
        candidates.append(Path(config_dir) / "kaggle.json")
        candidates.append(Path(config_dir) / "access_token")
        candidates.append(Path(config_dir) / "access_token.txt")
    candidates.append(Path.home() / ".kaggle" / "kaggle.json")
    candidates.append(Path.home() / ".kaggle" / "access_token")
    candidates.append(Path.home() / ".kaggle" / "access_token.txt")
    for path in candidates:
        if path.exists():
            try:
                if path.name == "kaggle.json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    username = data.get("username") or os.environ.get("KAGGLE_USERNAME")
                    return KaggleAuthState(True, username=username, config_path=str(path), source="kaggle.json")
                if path.name.startswith("access_token"):
                    username = os.environ.get("KAGGLE_USERNAME") or _discover_username_from_cli()
                    return KaggleAuthState(True, username=username, config_path=str(path), source="access_token")
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    username = os.environ.get("KAGGLE_USERNAME") or _discover_username_from_cli()
                    return KaggleAuthState(True, username=username, config_path=str(path), source=path.name)
                return KaggleAuthState(False, username=os.environ.get("KAGGLE_USERNAME"), config_path=str(path), source=path.name, reason="empty_access_token")
            except Exception:
                return KaggleAuthState(False, username=os.environ.get("KAGGLE_USERNAME"), config_path=str(path), source=path.name, reason="invalid_kaggle_credentials")
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    api_token = os.environ.get("KAGGLE_API_TOKEN")
    if username and key:
        return KaggleAuthState(True, username=username, config_path=None, source="environment")
    if api_token:
        return KaggleAuthState(True, username=username or _discover_username_from_cli(), config_path=None, source="environment_api_token")
    return KaggleAuthState(False, username=username, config_path=None, source=None, reason="missing_kaggle_credentials")


def _discover_username_from_cli() -> str | None:
    kaggle_exe = _resolve_kaggle_executable()
    if kaggle_exe is None:
        return None
    try:
        result = _run_command([kaggle_exe, "config", "view"], timeout=30, cwd=str(Path.home()))
        text = result.stdout or ""
        for line in text.splitlines():
            lower = line.lower()
            if "username" in lower and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value and "token" not in lower:
                    return value
    except Exception:
        return None
    return None


def get_repo_state() -> KaggleRepoState:
    head = None
    branch = None
    dirty = False
    try:
        head = _run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
        branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        status = _run_command(["git", "status", "--short"]).stdout.strip()
        dirty = bool(status)
    except Exception:
        pass
    return KaggleRepoState(head=head or None, dirty=dirty, branch=branch or None)


def ensure_stage_paths(stage_root: Path = DEFAULT_STAGE_ROOT) -> KagglePaths:
    stage_root = Path(stage_root).resolve()
    notebook_dir = stage_root / "notebook"
    download_dir = stage_root / "downloads"
    logs_dir = stage_root / "logs"
    for path in (notebook_dir, download_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return KagglePaths(stage_root=stage_root, notebook_dir=notebook_dir, download_dir=download_dir, logs_dir=logs_dir)


def _validate_kernel_directory(notebook_dir: Path, code_file: str) -> Path:
    resolved = Path(notebook_dir).resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise KaggleAutomationError("KAGGLE_LOCAL_KERNEL_PATH_INVALID: notebook directory")
    if not (resolved / "kernel-metadata.json").is_file():
        raise KaggleAutomationError("KAGGLE_LOCAL_KERNEL_PATH_INVALID: kernel-metadata.json")
    if not (resolved / code_file).is_file():
        raise KaggleAutomationError(f"KAGGLE_LOCAL_KERNEL_PATH_INVALID: {code_file}")
    return resolved


def _write_runner_heartbeat(
    *,
    phase: str,
    kernel_ref: str | None,
    expected_commit: str | None,
    run_id: str | None = None,
    elapsed_seconds: float | None,
    last_status: str | None,
    safe_message: str,
    stage_root: Path = DEFAULT_STAGE_ROOT,
) -> Path:
    stage = ensure_stage_paths(stage_root)
    payload = {
        "phase": phase,
        "timestamp": time.time(),
        "kernel_ref": kernel_ref,
        "expected_commit": expected_commit,
        "run_id": run_id or _run_id_from_stage_root(stage.stage_root),
        "elapsed_seconds": elapsed_seconds,
        "last_status": last_status,
        "safe_message": safe_message,
    }
    path = stage.stage_root / "runner_heartbeat.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_runner_failure(
    *,
    phase: str,
    command: str | None,
    exc: BaseException,
    timeout_seconds: int | None,
    kernel_ref: str | None,
    expected_commit: str | None,
    run_id: str | None = None,
    elapsed_seconds: float | None,
    stdout: str | None = None,
    stderr: str | None = None,
    last_status: str | None = None,
    stage_root: Path = DEFAULT_STAGE_ROOT,
) -> Path:
    stage = ensure_stage_paths(stage_root)
    payload = {
        "phase": phase,
        "command": command,
        "exception_type": type(exc).__name__,
        "sanitized_message": str(exc).replace("\n", " ").strip()[:1000],
        "timeout_seconds": timeout_seconds,
        "kernel_ref": kernel_ref,
        "expected_commit": expected_commit,
        "run_id": run_id or _run_id_from_stage_root(stage.stage_root),
        "elapsed_seconds": elapsed_seconds,
        "last_known_kernel_status": last_status,
        "safe_stdout_tail": _safe_tail(stdout),
        "safe_stderr_tail": _safe_tail(stderr),
    }
    path = stage.stage_root / "runner_failure.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_runner_metadata(
    *,
    stage_root: Path,
    run_id: str,
    expected_commit: str | None,
    notebook_ref: str | None,
    dataset_ref: str | None,
    started_at: float | None = None,
) -> Path:
    stage = ensure_stage_paths(stage_root)
    payload = {
        "run_id": run_id,
        "expected_git_commit": expected_commit,
        "notebook_ref": notebook_ref,
        "dataset_ref": dataset_ref,
        "started_at": started_at or time.time(),
    }
    return write_json(stage.stage_root / RUNNER_METADATA_NAME, payload)


def build_kernel_metadata(*, auth: KaggleAuthState, spec: KaggleNotebookSpec) -> dict[str, Any]:
    return {
        "id": spec.notebook_ref(auth.username),
        "title": spec.title,
        "code_file": spec.code_file,
        "language": spec.language,
        "kernel_type": spec.kernel_type,
        "enable_gpu": spec.enable_gpu,
        "enable_internet": spec.enable_internet,
        "dataset_sources": [spec.dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }


def sync_notebook_to_stage(
    stage: KagglePaths,
    spec: KaggleNotebookSpec,
    auth: KaggleAuthState,
    *,
    run_id: str | None = None,
    expected_commit: str | None = None,
) -> Path:
    if not NOTEBOOK_SOURCE.exists():
        raise KaggleAutomationError(f"missing_notebook_source:{NOTEBOOK_SOURCE}")
    notebook_dir = stage.notebook_dir
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_target = notebook_dir / spec.code_file
    notebook_text = NOTEBOOK_SOURCE.read_text(encoding="utf-8")
    if run_id is not None:
        notebook_text = notebook_text.replace("__RUN_ID__", run_id)
    if expected_commit is not None:
        notebook_text = notebook_text.replace("__EXPECTED_GIT_COMMIT__", expected_commit)
    notebook_text = notebook_text.replace("__WORKFLOW_MODE__", spec.workflow_mode)
    notebook_target.write_text(notebook_text, encoding="utf-8")
    scripts_dir = notebook_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for relative in (Path("scripts") / "__init__.py", Path("scripts") / "kaggle_runner.py"):
        source = REPO_ROOT / relative
        if source.exists():
            shutil.copy2(source, scripts_dir / source.name)
    metadata = build_kernel_metadata(auth=auth, spec=spec)
    (notebook_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_json = json.dumps(
        {
            "run_id": run_id,
            "expected_git_commit": expected_commit,
            "notebook_ref": kaggle_kernel_ref(auth, spec),
            "dataset_ref": spec.dataset_ref,
            "timestamp": time.time(),
        },
        indent=2,
        sort_keys=True,
    )
    (notebook_dir / RUNNER_METADATA_NAME).write_text(write_json, encoding="utf-8")
    return notebook_dir


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_prepared_submission(*, notebook_dir: Path, run_id: str, expected_commit: str, spec: KaggleNotebookSpec) -> dict[str, Any]:
    notebook_dir = Path(notebook_dir).resolve()
    entrypoint = notebook_dir / spec.code_file
    metadata_path = notebook_dir / "kernel-metadata.json"
    if not entrypoint.exists() or not metadata_path.exists():
        raise KaggleAutomationError("prepared_submission_missing_files")
    source = entrypoint.read_text(encoding="utf-8")
    try:
        notebook_payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise KaggleAutomationError("prepared_submission_identity_mismatch") from exc
    cells = notebook_payload.get("cells")
    if not isinstance(cells, list) or not cells or cells[0].get("cell_type") != "code":
        raise KaggleAutomationError("prepared_submission_startup_identity_not_first_cell")
    first_source = "".join(cells[0].get("source", []))
    forbidden_before_identity = ("torch", "transformers", "peft", "bitsandbytes", "datasets", "pip install")
    if "RUN_IDENTITY_JSON" not in first_source or "RUN_IDENTITY.json" not in first_source or "flush()" not in first_source:
        raise KaggleAutomationError("prepared_submission_startup_identity_incomplete")
    if any(value in first_source.lower() for value in forbidden_before_identity):
        raise KaggleAutomationError("prepared_submission_startup_identity_not_identity_only")
    historical_ids = ("1bcfd66-20260901T175300Z-t6xm", "db9a3f1-20260901T173537Z-flh7", "7b632bf-20260902T033839Z-kv6d")
    stale_ids = [value for value in historical_ids if value in source]
    if run_id not in source or expected_commit not in source or stale_ids:
        raise KaggleAutomationError("prepared_submission_identity_mismatch")
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata_payload.get("code_file") != spec.code_file:
        raise KaggleAutomationError("prepared_submission_code_file_mismatch")
    if metadata_payload.get("id") != spec.notebook_ref(discover_kaggle_auth().username):
        raise KaggleAutomationError("prepared_submission_kernel_ref_mismatch")
    return {
        "source_file": str(NOTEBOOK_SOURCE),
        "source_file_sha256": _sha256_file(NOTEBOOK_SOURCE),
        "generated_entrypoint": str(entrypoint),
        "generated_entrypoint_sha256": _sha256_file(entrypoint),
        "metadata_file": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "metadata": metadata_payload,
        "current_run_id_embedded": True,
        "current_commit_embedded": True,
        "historical_active_run_ids": stale_ids,
        "startup_marker_position_verified": True,
        "startup_identity_cell_index": 0,
        "first_executable_cell_verified": True,
        "run_scoped_identity_path": "/kaggle/working/<run_id>/RUN_IDENTITY.json",
        "run_scoped_log_path": "/kaggle/working/<run_id>/remote.log",
        "failure_path": "/kaggle/working/<run_id>/failure.json",
    }


def prepare_only(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="qwen_semantic_generation_diagnostic")
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    expected_commit = get_repo_state().head
    if not expected_commit:
        raise KaggleAutomationError("git_commit_unavailable")
    resolved_run_id = run_id or generate_run_id(git_commit=expected_commit)
    stage = ensure_stage_paths(run_root_for(resolved_run_id, base_root=stage_root))
    notebook_dir = sync_notebook_to_stage(stage, spec, auth, run_id=resolved_run_id, expected_commit=expected_commit)
    prepared = _validate_prepared_submission(notebook_dir=notebook_dir, run_id=resolved_run_id, expected_commit=expected_commit, spec=spec)
    manifest = {
        "run_id": resolved_run_id,
        "expected_commit": expected_commit,
        "git_head": expected_commit,
        "kernel_slug": spec.notebook_slug,
        "kernel_ref": kaggle_kernel_ref(auth, spec),
        "source_entrypoint": str(NOTEBOOK_SOURCE),
        "submission_timestamp": time.time(),
        "remote_submission_performed": False,
        **prepared,
    }
    manifest_path = stage.stage_root / "submission_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "run_id": resolved_run_id,
        "expected_commit": expected_commit,
        "kernel_ref": kaggle_kernel_ref(auth, spec),
        "notebook_dir": str(notebook_dir),
        "submission_manifest": str(manifest_path),
        "manifest": manifest,
        "safe_push_command": ["kaggle", "kernels", "push", "-p", str(notebook_dir)],
        "remote_submission_performed": False,
    }


def kaggle_kernel_ref(auth: KaggleAuthState, spec: KaggleNotebookSpec) -> str:
    return spec.notebook_ref(auth.username)


def _kaggle(*args: str, timeout: int | None = None) -> KaggleCommandResult:
    kaggle_exe = _resolve_kaggle_executable()
    if kaggle_exe is not None:
        return _run_command([kaggle_exe, *args], timeout=timeout, cwd=str(Path.home()))
    return KaggleCommandResult(_safe_command(["kaggle", *args]), None, "", "kaggle_cli_missing", False, 0.0)


def resolve_kaggle_runtime() -> dict[str, Any]:
    auth = discover_kaggle_auth()
    executable = _resolve_kaggle_executable()
    notebook_ref = kaggle_kernel_ref(auth, KaggleNotebookSpec())
    return {
        "python_executable": _current_python_executable(),
        "kaggle_package_available": _kaggle_module_available(),
        "kaggle_executable": executable,
        "kaggle_cli_available": executable is not None,
        "auth_available": auth.available,
        "auth_source": auth.source,
        "auth_username_resolved": bool(auth.username),
        "kernel_ref": notebook_ref,
        "kernel_ref_resolved": bool(auth.username),
    }


def _kaggle_checked(
    *args: str,
    timeout: int | None,
    phase: str,
    kernel_ref: str | None = None,
    expected_commit: str | None = None,
    run_id: str | None = None,
    start_time: float | None = None,
    last_status: str | None = None,
    stage_root: Path = DEFAULT_STAGE_ROOT,
) -> KaggleCommandResult:
    try:
        result = _kaggle(*args, timeout=timeout)
        result_payload = _command_result_payload(result, ["kaggle", *args])
        if result_payload["timed_out"] or result_payload["exit_code"] not in (0, None):
            exc = KaggleAutomationError("kaggle_command_failed")
            _write_runner_failure(
                phase=phase,
                command="kaggle " + " ".join(args),
                exc=exc,
                timeout_seconds=timeout,
                kernel_ref=kernel_ref,
                expected_commit=expected_commit,
                run_id=run_id,
                elapsed_seconds=(time.perf_counter() - start_time) if start_time is not None else None,
                stdout=result_payload["stdout"],
                stderr=result_payload["stderr"],
                last_status=last_status,
                stage_root=stage_root,
            )
        return result
    except subprocess.TimeoutExpired as exc:  # compatibility with mocked subprocess implementations
        _write_runner_failure(
            phase=phase,
            command="kaggle " + " ".join(args),
            exc=exc,
            timeout_seconds=timeout,
            kernel_ref=kernel_ref,
            expected_commit=expected_commit,
            run_id=run_id,
            elapsed_seconds=(time.perf_counter() - start_time) if start_time is not None else None,
            stdout=getattr(exc, "stdout", None),
            stderr=getattr(exc, "stderr", None),
            last_status=last_status,
            stage_root=stage_root,
        )
        return KaggleCommandResult(_safe_command(["kaggle", *args]), None, str(getattr(exc, "stdout", "") or ""), str(getattr(exc, "stderr", "") or ""), True, 0.0)
    except Exception as exc:
        _write_runner_failure(
            phase=phase,
            command="kaggle " + " ".join(args),
            exc=exc,
            timeout_seconds=timeout,
            kernel_ref=kernel_ref,
            expected_commit=expected_commit,
            run_id=run_id,
            elapsed_seconds=(time.perf_counter() - start_time) if start_time is not None else None,
            last_status=last_status,
            stage_root=stage_root,
        )
        return KaggleCommandResult(_safe_command(["kaggle", *args]), None, "", str(exc), False, 0.0)


def preflight(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    repo = get_repo_state()
    stage = ensure_stage_paths(stage_root)
    runtime = resolve_kaggle_runtime()
    notebook_ref = kaggle_kernel_ref(auth, spec)
    notebook_exists = None
    notebook_error = None
    notebook_status = None
    if auth.available and auth.username:
        _write_runner_heartbeat(
            phase="auth_check_started",
            kernel_ref=notebook_ref,
            expected_commit=repo.head,
            elapsed_seconds=None,
            last_status=None,
            safe_message="auth notebook status check started",
            stage_root=stage_root,
        )
        try:
            notebook_result = _kaggle_checked(
                "kernels", "status", notebook_ref, timeout=60,
                phase="auth_check_complete", kernel_ref=notebook_ref,
                expected_commit=repo.head, stage_root=stage_root,
            )
            notebook_status = _safe_cli_output(notebook_result)
            notebook_exists = getattr(notebook_result, "exit_code", getattr(notebook_result, "returncode", None)) == 0
            notebook_error = _safe_tail(notebook_result.stderr)
        except Exception as exc:
            notebook_error = str(exc)
    dataset_ok = False
    dataset_error = None
    if auth.available:
        _write_runner_heartbeat(
            phase="dataset_check_started",
            kernel_ref=notebook_ref,
            expected_commit=repo.head,
            elapsed_seconds=None,
            last_status=None,
            safe_message=f"dataset check started for {spec.dataset_ref}",
            stage_root=stage_root,
        )
        try:
            result = _kaggle_checked("datasets", "files", "-d", spec.dataset_ref, timeout=45, phase="auth_check_complete", kernel_ref=notebook_ref, stage_root=stage_root)
            dataset_ok = result.returncode == 0 and bool(_safe_cli_output(result))
        except subprocess.CalledProcessError as exc:
            dataset_error = _safe_cli_output(exc)
        except Exception as exc:
            dataset_error = str(exc)
    preflight_payload = {
        "python_executable": runtime["python_executable"],
        "kaggle_package_available": runtime["kaggle_package_available"],
        "kaggle_executable": runtime["kaggle_executable"],
        "cli_available": runtime["kaggle_cli_available"],
        "auth": auth.to_dict(),
        "repo": repo.to_dict(),
        "spec": spec.to_dict(),
        "paths": stage.to_dict(),
        "kernel_ref": notebook_ref,
        "kernel_ref_resolved": runtime["kernel_ref_resolved"],
        "kernel_status": notebook_status,
        "kernel_exists": notebook_exists,
        "kernel_error": notebook_error,
        "dataset_ref": spec.dataset_ref,
        "dataset_accessible": dataset_ok,
        "dataset_error": dataset_error,
        "gpu_metadata": {
            "enable_gpu": spec.enable_gpu,
            "enable_internet": spec.enable_internet,
        },
        "internet_required": spec.enable_internet,
        "available_commands": available_commands(),
        "ready": bool(kaggle_cli_available() and auth.available and dataset_ok and repo.head and auth.username),
        "one_time_action": None if auth.available else "configure Kaggle CLI credentials in ~/.kaggle/kaggle.json or KAGGLE_CONFIG_DIR",
    }
    return preflight_payload


def push(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT, run_id: str | None = None, expected_commit: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    stage = ensure_stage_paths(stage_root)
    attempt_path = stage.stage_root / SUBMISSION_ATTEMPT_NAME
    if attempt_path.exists():
        try:
            prior_attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_attempt = {}
        if run_id is None or prior_attempt.get("run_id") == run_id:
            raise KaggleAutomationError("duplicate_submission_attempt_blocked")
    expected_commit = expected_commit or get_repo_state().head
    if not expected_commit:
        raise KaggleAutomationError("git_commit_unavailable")
    notebook_dir = sync_notebook_to_stage(stage, spec, auth, run_id=run_id, expected_commit=expected_commit)
    notebook_dir = _validate_kernel_directory(notebook_dir, spec.code_file)
    prepared = _validate_prepared_submission(notebook_dir=notebook_dir, run_id=run_id or "", expected_commit=expected_commit, spec=spec)
    submission_manifest = {
        "run_id": run_id,
        "expected_commit": expected_commit,
        "git_head": get_repo_state().head,
        "kernel_slug": spec.notebook_slug,
        "kernel_ref": kaggle_kernel_ref(auth, spec),
        "source_entrypoint": str(NOTEBOOK_SOURCE),
        "submission_timestamp": time.time(),
        "remote_submission_performed": False,
        **prepared,
    }
    write_json(stage.stage_root / "submission_manifest.json", submission_manifest)
    result = _kaggle_checked("kernels", "push", "-p", str(notebook_dir), timeout=120, phase="push_complete", kernel_ref=kaggle_kernel_ref(auth, spec), expected_commit=expected_commit, run_id=run_id, stage_root=stage_root)
    result_payload = _command_result_payload(result, ["kaggle", "kernels", "push", "-p", str(notebook_dir)])
    attempt = {
        "run_id": run_id,
        "expected_commit": expected_commit,
        "timestamp": time.time(),
        "command_safe": result_payload["command_safe"],
        "exit_code": result_payload["exit_code"],
        "stdout_safe_tail": _safe_tail(result_payload["stdout"]),
        "stderr_safe_tail": _safe_tail(result_payload["stderr"]),
        "timed_out": result_payload["timed_out"],
        "duration_seconds": result_payload["duration_seconds"],
    }
    write_json(attempt_path, attempt)
    if result_payload["timed_out"]:
        raise KaggleAutomationError("push_timeout")
    if result_payload["exit_code"] != 0:
        raise KaggleAutomationError(json.dumps({"phase": "push", **{key: result_payload[key] for key in ("command_safe", "exit_code")}, "stdout_safe_tail": _safe_tail(result_payload["stdout"]), "stderr_safe_tail": _safe_tail(result_payload["stderr"])}, sort_keys=True))
    _write_runner_metadata(
        stage_root=stage_root,
        run_id=run_id or _run_id_from_stage_root(stage_root) or "unknown",
        expected_commit=expected_commit,
        notebook_ref=kaggle_kernel_ref(auth, spec),
        dataset_ref=spec.dataset_ref,
        started_at=time.time(),
    )
    return {
        "notebook_ref": kaggle_kernel_ref(auth, spec),
        "dataset_ref": spec.dataset_ref,
        "stdout": _safe_cli_output(result),
        "command_result": result_payload,
        "stage_dir": str(notebook_dir),
        "submission_manifest": str(stage.stage_root / "submission_manifest.json"),
    }


def run(
    spec: KaggleNotebookSpec | None = None,
    *,
    stage_root: Path = DEFAULT_STAGE_ROOT,
    poll_seconds: int = 30,
    timeout_seconds: int = 3600,
    run_id: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    kernel_ref = kaggle_kernel_ref(auth, spec)
    resolved_run_id = run_id or _run_id_from_stage_root(stage_root) or generate_run_id(git_commit=get_repo_state().head)
    _write_runner_metadata(
        stage_root=stage_root,
        run_id=resolved_run_id,
        expected_commit=expected_commit or get_repo_state().head,
        notebook_ref=kernel_ref,
        dataset_ref=spec.dataset_ref,
        started_at=time.time(),
    )
    start_time = time.perf_counter()
    _write_runner_heartbeat(
        phase="runner_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=0.0,
        last_status=None,
        safe_message="runner start",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="preflight_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=None,
        safe_message="preflight start",
        stage_root=stage_root,
    )
    current = _status_payload(spec, stage_root=stage_root)
    _write_runner_heartbeat(
        phase="preflight_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="preflight complete",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="auth_check_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="status checked",
        stage_root=stage_root,
    )
    if "running" in current.get("status", "").lower():
        raise KaggleAutomationError("kernel_already_running")
    _write_runner_heartbeat(
        phase="push_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="push start",
        stage_root=stage_root,
    )
    push_result = push(spec, stage_root=stage_root, run_id=resolved_run_id, expected_commit=expected_commit or get_repo_state().head)
    _write_runner_heartbeat(
        phase="push_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="push complete",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="execute_request_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="execute request acknowledged",
        stage_root=stage_root,
    )
    deadline = time.time() + timeout_seconds
    polls = 0
    status_text = None
    identity_marker: dict[str, Any] | None = None
    _write_runner_heartbeat(
        phase="execute_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="execute request accepted",
        stage_root=stage_root,
    )
    while time.time() < deadline:
        polls += 1
        _write_runner_heartbeat(
            phase="poll_iteration",
            kernel_ref=kernel_ref,
            expected_commit=get_repo_state().head,
            run_id=resolved_run_id,
            elapsed_seconds=round(time.perf_counter() - start_time, 2),
            last_status=status_text or current.get("status"),
            safe_message=f"poll={polls}",
            stage_root=stage_root,
        )
        result = _kaggle_checked("kernels", "status", kernel_ref, timeout=min(60, poll_seconds + 15), phase="poll_started", kernel_ref=kernel_ref, expected_commit=get_repo_state().head, start_time=start_time, last_status=status_text or current.get("status"), stage_root=stage_root)
        status_text = _safe_cli_output(result)
        if any(token in status_text.lower() for token in ("complete", "error", "failed", "killed", "cancelled")):
            log_result = _kaggle_checked(
                "kernels", "logs", kernel_ref,
                timeout=60,
                phase="logs_started",
                kernel_ref=kernel_ref,
                expected_commit=expected_commit or get_repo_state().head,
                run_id=resolved_run_id,
                start_time=start_time,
                last_status=status_text,
                stage_root=stage_root,
            )
            log_text = _safe_cli_output(log_result)
            ensure_stage_paths(stage_root).logs_dir.joinpath("remote.log").write_text(log_text, encoding="utf-8")
            identity_marker = parse_run_identity_marker(
                log_text,
                run_id=resolved_run_id,
                expected_commit=expected_commit or get_repo_state().head,
            )
            if identity_marker is None:
                exc = KaggleAutomationError("ORCHESTRATION_FAILURE:fresh_run_unproven")
                _write_runner_failure(
                    phase="terminal_status_received",
                    command="kaggle kernels logs",
                    exc=exc,
                    timeout_seconds=60,
                    kernel_ref=kernel_ref,
                    expected_commit=expected_commit or get_repo_state().head,
                    run_id=resolved_run_id,
                    elapsed_seconds=round(time.perf_counter() - start_time, 2),
                    stdout=log_text,
                    last_status=status_text,
                    stage_root=stage_root,
                )
                raise exc
            _write_runner_heartbeat(
                phase="terminal_status_received",
                kernel_ref=kernel_ref,
                expected_commit=get_repo_state().head,
                run_id=resolved_run_id,
                elapsed_seconds=round(time.perf_counter() - start_time, 2),
                last_status=status_text,
                safe_message="fresh run identity verified",
                stage_root=stage_root,
            )
            break
        time.sleep(poll_seconds)
    else:
        _write_runner_failure(
            phase="poll_started",
            command="kaggle kernels status",
            exc=KaggleAutomationError("timeout"),
            timeout_seconds=timeout_seconds,
            kernel_ref=kernel_ref,
            expected_commit=get_repo_state().head,
            elapsed_seconds=round(time.perf_counter() - start_time, 2),
            last_status=status_text or current.get("status"),
            stage_root=stage_root,
        )
        raise KaggleAutomationError("timeout")
    _write_runner_heartbeat(
            phase="logs_started",
            kernel_ref=kernel_ref,
            expected_commit=get_repo_state().head,
            run_id=resolved_run_id,
            elapsed_seconds=round(time.perf_counter() - start_time, 2),
            last_status=status_text,
            safe_message="logs requested",
            stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="logs_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=status_text,
        safe_message="logs handled",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="outputs_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=status_text,
        safe_message="outputs requested",
        stage_root=stage_root,
    )
    output_result = outputs(spec, stage_root=stage_root)
    _write_runner_heartbeat(
        phase="outputs_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=status_text,
        safe_message="outputs complete",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="runner_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        run_id=resolved_run_id,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=status_text,
        safe_message="runner complete",
        stage_root=stage_root,
    )
    return {
        "notebook_ref": kernel_ref,
        "push": push_result,
        "status": status_text,
        "polls": polls,
        "fresh_execution_verified": identity_marker is not None,
        "run_identity": identity_marker,
    }


def status(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    return _status_payload(spec, stage_root=_resolve_stage_root(stage_root, run_id))


def outputs(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    stage_root = _resolve_stage_root(stage_root, run_id)
    stage = ensure_stage_paths(stage_root)
    kernel_ref = kaggle_kernel_ref(auth, spec)
    timeout_seconds = int(os.environ.get("KAGGLE_OUTPUTS_TIMEOUT_SECONDS", str(DEFAULT_OUTPUTS_TIMEOUT_SECONDS)))
    retry_count = int(os.environ.get("KAGGLE_OUTPUTS_RETRY_COUNT", str(DEFAULT_OUTPUTS_RETRY_COUNT)))
    if stage.download_dir.exists():
        shutil.rmtree(stage.download_dir)
    stage.download_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    last_error: dict[str, Any] | None = None
    for attempt_index in range(max(1, retry_count + 1)):
        try:
            result = _kaggle_checked(
                "kernels",
                "output",
                kernel_ref,
                "-p",
                str(stage.download_dir),
                timeout=timeout_seconds,
                phase="outputs_started",
                kernel_ref=kernel_ref,
                stage_root=stage_root,
            )
            result_payload = _command_result_payload(result, ["kaggle", "kernels", "output", kernel_ref])
            if result_payload["timed_out"]:
                raise KaggleAutomationError("outputs_timeout")
            if result_payload["exit_code"] != 0:
                raise KaggleAutomationError(json.dumps({"exit_code": result_payload["exit_code"], "stdout_safe_tail": _safe_tail(result_payload["stdout"]), "stderr_safe_tail": _safe_tail(result_payload["stderr"])}, sort_keys=True))
            downloaded = []
            for path in stage.download_dir.rglob("*"):
                if not path.is_file() or path.name not in SAFE_OUTPUT_NAMES:
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
                except (OSError, json.JSONDecodeError):
                    payload = None
                if run_id and isinstance(payload, dict) and payload.get("run_id") not in {None, run_id}:
                    continue
                downloaded.append(path.name)
            return {
                "notebook_ref": kernel_ref,
                "download_dir": str(stage.download_dir),
                "downloaded_safe_artifacts": sorted(downloaded),
                "stdout": _safe_cli_output(result),
                "timeout_seconds": timeout_seconds,
                "attempts": attempts + [{"attempt": attempt_index + 1, "status": "success"}],
            }
        except subprocess.TimeoutExpired as exc:
            last_error = {
                "attempt": attempt_index + 1,
                "status": "timeout",
                "timeout_seconds": timeout_seconds,
                "sanitized_message": str(exc),
            }
            attempts.append(last_error)
            if attempt_index < retry_count:
                continue
            break
    raise KaggleAutomationError(json.dumps({"phase": "outputs_started", "timeout_seconds": timeout_seconds, "attempts": attempts, "error": last_error}, sort_keys=True))


def _read_tail(path: Path, *, lines: int = 100) -> str | None:
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(content[-lines:])
    except Exception:
        return None


def _read_heartbeat(stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any] | None:
    stage = ensure_stage_paths(stage_root)
    candidates = [
        stage.stage_root / "smoke_heartbeat.json",
        stage.stage_root / "reports" / "smoke_heartbeat.json",
        stage.download_dir / "reports" / "smoke_heartbeat.json",
        *stage.download_dir.rglob("smoke_heartbeat.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _read_breadcrumbs(stage_root: Path = DEFAULT_STAGE_ROOT) -> list[dict[str, Any]]:
    stage = ensure_stage_paths(stage_root)
    candidates = [
        stage.stage_root / "smoke_breadcrumbs.jsonl",
        stage.stage_root / "reports" / "smoke_breadcrumbs.jsonl",
        stage.download_dir / "reports" / "smoke_breadcrumbs.jsonl",
        *stage.download_dir.rglob("smoke_breadcrumbs.jsonl"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            continue
    return []


def _read_runner_heartbeat(stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any] | None:
    stage = ensure_stage_paths(stage_root)
    path = stage.stage_root / "runner_heartbeat.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_runner_failure(stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any] | None:
    stage = ensure_stage_paths(stage_root)
    path = stage.stage_root / "runner_failure.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _kaggle_postmortem(spec: KaggleNotebookSpec, *, stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any]:
    stage = ensure_stage_paths(stage_root)
    reports_dir = stage.download_dir / "reports"
    failure_path = reports_dir / "smoke_failure.json"
    log_paths = list(stage.download_dir.rglob("*.log"))
    payload: dict[str, Any] = {
        "last_breadcrumb_stage": None,
        "smoke_failure": None,
        "last_safe_log_lines": None,
        "last_progress_timestamp": None,
        "heartbeat": None,
        "live_log_tail": None,
        "log_evidence_current_run": False,
        "historical_log_detected": False,
        "timeout_phase": None,
    }
    records = _read_breadcrumbs(stage_root)
    if records:
        payload["last_breadcrumb_stage"] = records[-1].get("stage")
        payload["last_progress_timestamp"] = records[-1].get("timestamp")
    heartbeat = _read_heartbeat(stage_root)
    if heartbeat:
        payload["heartbeat"] = heartbeat
    if failure_path.exists():
        try:
            payload["smoke_failure"] = json.loads(failure_path.read_text(encoding="utf-8"))
        except Exception:
            payload["smoke_failure"] = None
    if log_paths:
        payload["last_safe_log_lines"] = _read_tail(log_paths[0], lines=100)
        log_text = payload["last_safe_log_lines"] or ""
        heartbeat = heartbeat or {}
        run_id = heartbeat.get("run_id")
        executed_commit = heartbeat.get("git_commit") or (records[-1].get("run_id") if records else None)
        if run_id or executed_commit:
            if (run_id and run_id in log_text) or (executed_commit and executed_commit in log_text):
                payload["log_evidence_current_run"] = True
        payload["historical_log_detected"] = not payload["log_evidence_current_run"] and bool(log_text)
    try:
        auth = discover_kaggle_auth()
        if auth.available:
            kernel_ref = kaggle_kernel_ref(auth, spec)
            live_result = _kaggle_checked(
                "kernels",
                "logs",
                kernel_ref,
                timeout=60,
                phase="logs_started",
                kernel_ref=kernel_ref,
                stage_root=stage_root,
            )
            live_text = _safe_cli_output(live_result)
            if live_text:
                payload["live_log_tail"] = "\n".join(live_text.splitlines()[-100:])
    except Exception:
        pass
    if payload["smoke_failure"]:
        payload["timeout_phase"] = payload["smoke_failure"].get("stage") if isinstance(payload["smoke_failure"], dict) else None
    return payload


def _status_payload(
    spec: KaggleNotebookSpec,
    *,
    stage_root: Path = DEFAULT_STAGE_ROOT,
    include_error_artifacts: bool = True,
) -> dict[str, Any]:
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    kernel_ref = kaggle_kernel_ref(auth, spec)
    start_time = time.perf_counter()
    result = _kaggle_checked("kernels", "status", kernel_ref, timeout=60, phase="poll_started", kernel_ref=kernel_ref, start_time=start_time, stage_root=stage_root)
    if getattr(result, "timed_out", False):
        status_text = "KAGGLE_API_TIMEOUT"
    elif getattr(result, "exit_code", getattr(result, "returncode", None)) != 0:
        status_text = _safe_cli_output(result) or "KAGGLE_STATUS_FAILED"
    else:
        status_text = _safe_cli_output(result)
    breadcrumbs = _read_breadcrumbs(stage_root)
    heartbeat = _read_heartbeat(stage_root)
    stall_threshold = int(os.environ.get("KAGGLE_SMOKE_NO_PROGRESS_THRESHOLD_SECONDS", "1800"))
    stall_detection = None
    if breadcrumbs:
        last = breadcrumbs[-1]
        stall_detection = {
            "last_stage": last.get("stage"),
            "last_progress_timestamp": last.get("timestamp"),
            "no_progress_threshold_seconds": stall_threshold,
        }
    payload: dict[str, Any] = {
        "notebook_ref": kernel_ref,
        "status": status_text,
        "heartbeat": heartbeat,
        "breadcrumbs": breadcrumbs,
        "stall_detection": stall_detection,
    }
    if "error" in status_text.lower() and include_error_artifacts:
        try:
            payload["outputs"] = outputs(spec, stage_root=stage_root)
        except Exception as exc:
            payload["outputs_error"] = str(exc)
        payload["postmortem"] = _kaggle_postmortem(spec, stage_root=stage_root)
    return payload


def full_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, tests_command: list[str] | None = None, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    tests_command = tests_command or [sys.executable, "-m", "pytest", "-q"]
    test_run = _run_command(tests_command, check=False)
    if test_run.returncode != 0:
        raise KaggleAutomationError("local_tests_failed")
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=run_id)
    output_result = outputs(spec, stage_root=stage_root)
    return {
        "tests": {"returncode": test_run.returncode, "stdout": test_run.stdout, "stderr": test_run.stderr},
        "preflight": preflight_result,
        "run": run_result,
        "outputs": output_result,
    }


def smoke_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root)
    return {
        "run_id": resolved_run_id,
        "preflight": preflight_result,
        "run": run_result,
        "outputs": output_result,
    }


def torch_compat_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="torch_compat")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root)
    return {
        "run_id": resolved_run_id,
        "preflight": preflight_result,
        "run": run_result,
        "outputs": output_result,
    }


def bnb_compat_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="bnb_compat")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root)
    return {
        "run_id": resolved_run_id,
        "preflight": preflight_result,
        "run": run_result,
        "outputs": output_result,
    }


def bnb_native_diagnose(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="bnb_native_diagnose")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root, run_id=resolved_run_id)
    return {"run_id": resolved_run_id, "preflight": preflight_result, "run": run_result, "outputs": output_result}


def qwen_nf4_load_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="qwen_nf4_load")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root, run_id=resolved_run_id)
    return {"run_id": resolved_run_id, "preflight": preflight_result, "run": run_result, "outputs": output_result}


def qwen_qlora_backward_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="qwen_qlora_backward")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root, run_id=resolved_run_id)
    return {"run_id": resolved_run_id, "preflight": preflight_result, "run": run_result, "outputs": output_result}


def qwen_qlora_training_smoke_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="qwen_qlora_training_smoke")
    return qwen_qlora_backward_cycle(stage_root=stage_root, spec=spec, run_id=run_id)


def qwen_qlora_learning_experiment_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="qwen_qlora_learning_experiment")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root, run_id=resolved_run_id)
    return {"run_id": resolved_run_id, "preflight": preflight_result, "run": run_result, "outputs": output_result}


def qwen_semantic_memorization_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="qwen_semantic_memorization")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root, run_id=resolved_run_id)
    return {"run_id": resolved_run_id, "preflight": preflight_result, "run": run_result, "outputs": output_result}


def qwen_semantic_generation_diagnostic_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="qwen_semantic_generation_diagnostic")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root, run_id=resolved_run_id)
    return {"run_id": resolved_run_id, "preflight": preflight_result, "run": run_result, "outputs": output_result}


def semantic_corpus_audit_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec(workflow_mode="semantic_corpus_audit")
    resolved_run_id = run_id or generate_run_id(git_commit=get_repo_state().head)
    stage_root = _resolve_stage_root(stage_root, resolved_run_id)
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root, run_id=resolved_run_id)
    output_result = outputs(spec, stage_root=stage_root, run_id=resolved_run_id)
    return {"run_id": resolved_run_id, "preflight": preflight_result, "run": run_result, "outputs": output_result}


def run_bnb_compat_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None, runtime_dir: Path | None = None) -> dict[str, Any]:
    return bnb_compat_cycle(stage_root=stage_root, spec=spec, run_id=run_id, runtime_dir=runtime_dir)


def diagnose(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None, run_id: str | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    stage_root = _resolve_stage_root(stage_root, run_id)
    auth = discover_kaggle_auth()
    repo = get_repo_state()
    stage = ensure_stage_paths(stage_root)
    runtime = resolve_kaggle_runtime()
    kernel_ref = kaggle_kernel_ref(auth, spec)
    status_result: dict[str, Any] | None = None
    status_error: str | None = None
    try:
        status_result = _status_payload(spec, stage_root=stage_root, include_error_artifacts=False)
    except Exception as exc:
        status_error = str(exc)
    return {
        "auth": auth.to_dict(),
        "python_executable": runtime["python_executable"],
        "kaggle_package_available": runtime["kaggle_package_available"],
        "kaggle_executable": runtime["kaggle_executable"],
        "kaggle_cli_available": runtime["kaggle_cli_available"],
        "repo": repo.to_dict(),
        "kernel_ref": kernel_ref,
        "kernel_ref_resolved": runtime["kernel_ref_resolved"],
        "status_result": status_result,
        "status_error": status_error,
        "runner_heartbeat": _read_runner_heartbeat(stage_root),
        "runner_failure": _read_runner_failure(stage_root),
        "heartbeat": _read_heartbeat(stage_root),
        "breadcrumbs": _read_breadcrumbs(stage_root),
        "postmortem": _kaggle_postmortem(spec, stage_root=stage_root),
        "stage_root": stage.to_dict(),
        "expected_commit": repo.head,
        "dataset_ref": spec.dataset_ref,
        "run_id": run_id or _run_id_from_stage_root(stage_root),
    }


def report(*, run_id: str, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    stage_root = _resolve_stage_root(stage_root, run_id)
    stage = ensure_stage_paths(stage_root)
    remote_identity = read_json(stage.stage_root / REMOTE_IDENTITY_NAME)
    retrieval_report = read_json(stage.stage_root / RETRIEVAL_REPORT_NAME)
    heartbeat = _read_heartbeat(stage_root)
    breadcrumbs = _read_breadcrumbs(stage_root)
    failure = read_json(stage.stage_root / "smoke_failure.json") or _read_runner_failure(stage_root)
    outputs_summary = {
        "download_dir": str(stage.download_dir),
        "safe_files": sorted(
            path.name
            for path in stage.download_dir.rglob("*")
            if path.is_file() and path.name in SAFE_OUTPUT_NAMES
        ),
    }
    return {
        "run_id": run_id,
        "dataset_ref": spec.dataset_ref,
        "stage_root": stage.to_dict(),
        "remote_identity": remote_identity,
        "retrieval_report": retrieval_report,
        "heartbeat": heartbeat,
        "breadcrumbs": breadcrumbs,
        "failure": failure,
        "outputs": outputs_summary,
        "runner_heartbeat": _read_runner_heartbeat(stage_root),
        "runner_failure": _read_runner_failure(stage_root),
        "postmortem": _kaggle_postmortem(spec, stage_root=stage_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle notebook deployment and execution workflow")
    parser.add_argument("--dataset-ref", default=DEFAULT_DATASET_REF)
    parser.add_argument("--notebook-slug", default=DEFAULT_NOTEBOOK_SLUG)
    parser.add_argument("--title", default=DEFAULT_NOTEBOOK_TITLE)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--run-id", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("push")
    sub.add_parser("run")
    sub.add_parser("status")
    sub.add_parser("outputs")
    sub.add_parser("full-cycle")
    sub.add_parser("smoke-cycle")
    sub.add_parser("torch-compat-cycle")
    sub.add_parser("bnb-compat-cycle")
    sub.add_parser("qwen-nf4-load-cycle")
    sub.add_parser("qwen-qlora-backward-cycle")
    sub.add_parser("qwen-qlora-training-smoke-cycle")
    sub.add_parser("qwen-qlora-learning-experiment-cycle")
    sub.add_parser("qwen-semantic-memorization-cycle")
    sub.add_parser("qwen-semantic-generation-diagnostic-cycle")
    sub.add_parser("semantic-corpus-audit-cycle")
    sub.add_parser("bnb-native-diagnose")
    sub.add_parser("diagnose")
    sub.add_parser("prepare-only")
    report = sub.add_parser("report")
    report.add_argument("--run-id", required=True)
    return parser


def _spec_from_args(args: argparse.Namespace) -> KaggleNotebookSpec:
    return KaggleNotebookSpec(
        dataset_ref=args.dataset_ref,
        notebook_slug=args.notebook_slug,
        title=args.title,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = _spec_from_args(args)
    try:
        if args.command == "preflight":
            _emit_json(preflight(spec, stage_root=args.stage_root))
        elif args.command == "push":
            _emit_json(push(spec, stage_root=args.stage_root, run_id=args.run_id))
        elif args.command == "run":
            _emit_json(run(spec, stage_root=args.stage_root, run_id=args.run_id))
        elif args.command == "status":
            _emit_json(status(spec, stage_root=args.stage_root, run_id=args.run_id))
        elif args.command == "outputs":
            _emit_json(outputs(spec, stage_root=args.stage_root, run_id=args.run_id))
        elif args.command == "full-cycle":
            _emit_json(full_cycle(stage_root=args.stage_root, spec=spec, run_id=args.run_id))
        elif args.command == "smoke-cycle":
            _emit_json(smoke_cycle(stage_root=args.stage_root, spec=spec, run_id=args.run_id))
        elif args.command == "torch-compat-cycle":
            _emit_json(torch_compat_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "torch_compat"}), run_id=args.run_id))
        elif args.command == "bnb-compat-cycle":
            _emit_json(bnb_compat_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "bnb_compat"}), run_id=args.run_id))
        elif args.command == "qwen-nf4-load-cycle":
            _emit_json(qwen_nf4_load_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "qwen_nf4_load"}), run_id=args.run_id))
        elif args.command == "qwen-qlora-backward-cycle":
            _emit_json(qwen_qlora_backward_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "qwen_qlora_backward"}), run_id=args.run_id))
        elif args.command == "qwen-qlora-training-smoke-cycle":
            _emit_json(qwen_qlora_training_smoke_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "qwen_qlora_training_smoke"}), run_id=args.run_id))
        elif args.command == "qwen-qlora-learning-experiment-cycle":
            _emit_json(qwen_qlora_learning_experiment_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "qwen_qlora_learning_experiment"}), run_id=args.run_id))
        elif args.command == "qwen-semantic-memorization-cycle":
            _emit_json(qwen_semantic_memorization_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "qwen_semantic_memorization"}), run_id=args.run_id))
        elif args.command == "qwen-semantic-generation-diagnostic-cycle":
            _emit_json(qwen_semantic_generation_diagnostic_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "qwen_semantic_generation_diagnostic"}), run_id=args.run_id))
        elif args.command == "semantic-corpus-audit-cycle":
            _emit_json(semantic_corpus_audit_cycle(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "semantic_corpus_audit"}), run_id=args.run_id))
        elif args.command == "bnb-native-diagnose":
            _emit_json(bnb_native_diagnose(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "bnb_native_diagnose"}), run_id=args.run_id))
        elif args.command == "diagnose":
            _emit_json(diagnose(stage_root=args.stage_root, spec=spec, run_id=args.run_id))
        elif args.command == "prepare-only":
            _emit_json(prepare_only(stage_root=args.stage_root, spec=KaggleNotebookSpec(**{**spec.to_dict(), "workflow_mode": "qwen_semantic_generation_diagnostic"}), run_id=args.run_id))
        elif args.command == "report":
            _emit_json(report(run_id=args.run_id, stage_root=args.stage_root, spec=spec))
        else:
            raise KaggleAutomationError(f"unsupported_command:{args.command}")
        return 0
    except KaggleAutomationError as exc:
        runtime = resolve_kaggle_runtime()
        _emit_json({"ok": False, "error": str(exc), "command": args.command, "dataset_ref": spec.dataset_ref, "kernel_ref": runtime["kernel_ref"], "kernel_ref_resolved": runtime["kernel_ref_resolved"]})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
