from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = REPO_ROOT / "kaggle" / "semantic_extractor_training.ipynb"
DEFAULT_DATASET_REF = "jaistudio/data-analysis-llm"
DEFAULT_NOTEBOOK_SLUG = "data-analysis-llm-semantic-extractor"
DEFAULT_NOTEBOOK_TITLE = "Data Analysis LLM Semantic Extractor"
DEFAULT_STAGE_ROOT = REPO_ROOT / "runtime" / "kaggle_runner"
RUNNER_HEARTBEAT_PATH = DEFAULT_STAGE_ROOT / "runner_heartbeat.json"
RUNNER_FAILURE_PATH = DEFAULT_STAGE_ROOT / "runner_failure.json"
SAFE_OUTPUT_NAMES = {
    "final_report.json",
    "semantic_metrics.json",
    "artifact_manifest.json",
    "semantic_extractor_artifacts.zip",
    "smoke_training_report.json",
    "smoke_breadcrumbs.jsonl",
    "smoke_failure.json",
}


class KaggleAutomationError(RuntimeError):
    pass


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


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_command(args: list[str], *, check: bool = True, timeout: int | None = None, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        timeout=timeout,
        cwd=cwd,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )


def _safe_tail(text: str | None, *, lines: int = 40) -> str | None:
    if not text:
        return None
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def _safe_cli_output(result: subprocess.CompletedProcess[str]) -> str:
    text = result.stdout or ""
    return text.strip()


def _normalize_status_text(status: Any | None) -> str | None:
    if status is None:
        return None
    value = getattr(status, "value", status)
    return str(value)


def _sdk_kernel_status(auth: KaggleAuthState, spec: KaggleNotebookSpec) -> str | None:
    if not auth.available or not auth.username:
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
    return shutil.which("kaggle") is not None or _kaggle_python_available()


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
    if not _kaggle_python_available() and shutil.which("kaggle") is None:
        return None
    try:
        result = _kaggle("config", "view", timeout=30)
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
    notebook_dir = stage_root / "notebook"
    download_dir = stage_root / "downloads"
    logs_dir = stage_root / "logs"
    for path in (notebook_dir, download_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return KagglePaths(stage_root=stage_root, notebook_dir=notebook_dir, download_dir=download_dir, logs_dir=logs_dir)


def _write_runner_heartbeat(
    *,
    phase: str,
    kernel_ref: str | None,
    expected_commit: str | None,
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
        "elapsed_seconds": elapsed_seconds,
        "last_known_kernel_status": last_status,
        "safe_stdout_tail": _safe_tail(stdout),
        "safe_stderr_tail": _safe_tail(stderr),
    }
    path = stage.stage_root / "runner_failure.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


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


def sync_notebook_to_stage(stage: KagglePaths, spec: KaggleNotebookSpec, auth: KaggleAuthState) -> Path:
    if not NOTEBOOK_SOURCE.exists():
        raise KaggleAutomationError(f"missing_notebook_source:{NOTEBOOK_SOURCE}")
    notebook_dir = stage.notebook_dir
    notebook_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK_SOURCE, notebook_dir / spec.code_file)
    scripts_dir = notebook_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for relative in (Path("scripts") / "__init__.py", Path("scripts") / "kaggle_runner.py"):
        source = REPO_ROOT / relative
        if source.exists():
            shutil.copy2(source, scripts_dir / source.name)
    metadata = build_kernel_metadata(auth=auth, spec=spec)
    (notebook_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return notebook_dir


def kaggle_kernel_ref(auth: KaggleAuthState, spec: KaggleNotebookSpec) -> str:
    return spec.notebook_ref(auth.username)


def _kaggle(*args: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    if _kaggle_python_available():
        return _run_command([sys.executable, "-m", "kaggle", *args], timeout=timeout, cwd=str(Path.home()))
    if shutil.which("kaggle") is not None:
        return _run_command(["kaggle", *args], timeout=timeout)
    raise KaggleAutomationError("kaggle_cli_missing")


def _kaggle_checked(
    *args: str,
    timeout: int | None,
    phase: str,
    kernel_ref: str | None = None,
    expected_commit: str | None = None,
    start_time: float | None = None,
    last_status: str | None = None,
    stage_root: Path = DEFAULT_STAGE_ROOT,
) -> subprocess.CompletedProcess[str]:
    try:
        return _kaggle(*args, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _write_runner_failure(
            phase=phase,
            command="kaggle " + " ".join(args),
            exc=exc,
            timeout_seconds=timeout,
            kernel_ref=kernel_ref,
            expected_commit=expected_commit,
            elapsed_seconds=(time.perf_counter() - start_time) if start_time is not None else None,
            stdout=getattr(exc, "stdout", None),
            stderr=getattr(exc, "stderr", None),
            last_status=last_status,
            stage_root=stage_root,
        )
        raise KaggleAutomationError(f"{phase}_timeout") from exc
    except Exception as exc:
        _write_runner_failure(
            phase=phase,
            command="kaggle " + " ".join(args),
            exc=exc,
            timeout_seconds=timeout,
            kernel_ref=kernel_ref,
            expected_commit=expected_commit,
            elapsed_seconds=(time.perf_counter() - start_time) if start_time is not None else None,
            last_status=last_status,
            stage_root=stage_root,
        )
        raise


def preflight(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    repo = get_repo_state()
    stage = ensure_stage_paths(stage_root)
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
            notebook_status = _sdk_kernel_status(auth, spec)
            notebook_exists = notebook_status is not None
        except subprocess.CalledProcessError as exc:
            notebook_error = _safe_cli_output(exc)
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
        "cli_available": kaggle_cli_available(),
        "auth": auth.to_dict(),
        "repo": repo.to_dict(),
        "spec": spec.to_dict(),
        "paths": stage.to_dict(),
        "kernel_ref": notebook_ref,
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
        "available_commands": ["preflight", "push", "run", "status", "outputs", "full-cycle", "smoke-cycle"],
        "ready": bool(kaggle_cli_available() and auth.available and dataset_ok and repo.head),
        "one_time_action": None if auth.available else "configure Kaggle CLI credentials in ~/.kaggle/kaggle.json or KAGGLE_CONFIG_DIR",
    }
    return preflight_payload


def push(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    stage = ensure_stage_paths(stage_root)
    notebook_dir = sync_notebook_to_stage(stage, spec, auth)
    result = _kaggle_checked("kernels", "push", "-p", str(notebook_dir), timeout=120, phase="push_complete", kernel_ref=kaggle_kernel_ref(auth, spec), stage_root=stage_root)
    return {
        "notebook_ref": kaggle_kernel_ref(auth, spec),
        "dataset_ref": spec.dataset_ref,
        "stdout": _safe_cli_output(result),
        "stage_dir": str(notebook_dir),
    }


def run(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT, poll_seconds: int = 30, timeout_seconds: int = 3600) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    kernel_ref = kaggle_kernel_ref(auth, spec)
    start_time = time.perf_counter()
    _write_runner_heartbeat(
        phase="runner_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        elapsed_seconds=0.0,
        last_status=None,
        safe_message="runner start",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="preflight_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
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
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="preflight complete",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="auth_check_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
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
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="push start",
        stage_root=stage_root,
    )
    push_result = push(spec, stage_root=stage_root)
    _write_runner_heartbeat(
        phase="push_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="push complete",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="execute_request_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=current.get("status"),
        safe_message="execute request acknowledged",
        stage_root=stage_root,
    )
    deadline = time.time() + timeout_seconds
    polls = 0
    status_text = None
    _write_runner_heartbeat(
        phase="execute_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
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
            elapsed_seconds=round(time.perf_counter() - start_time, 2),
            last_status=status_text or current.get("status"),
            safe_message=f"poll={polls}",
            stage_root=stage_root,
        )
        result = _kaggle_checked("kernels", "status", kernel_ref, timeout=min(60, poll_seconds + 15), phase="poll_started", kernel_ref=kernel_ref, expected_commit=get_repo_state().head, start_time=start_time, last_status=status_text or current.get("status"), stage_root=stage_root)
        status_text = _safe_cli_output(result)
        if any(token in status_text.lower() for token in ("complete", "error", "failed", "killed", "cancelled", "cancelled")):
            _write_runner_heartbeat(
                phase="terminal_status_received",
                kernel_ref=kernel_ref,
                expected_commit=get_repo_state().head,
                elapsed_seconds=round(time.perf_counter() - start_time, 2),
                last_status=status_text,
                safe_message=status_text,
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
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=status_text,
        safe_message="logs requested",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="logs_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=status_text,
        safe_message="logs handled",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="outputs_started",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
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
        elapsed_seconds=round(time.perf_counter() - start_time, 2),
        last_status=status_text,
        safe_message="outputs complete",
        stage_root=stage_root,
    )
    _write_runner_heartbeat(
        phase="runner_complete",
        kernel_ref=kernel_ref,
        expected_commit=get_repo_state().head,
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
    }


def status(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    return _status_payload(spec, stage_root=stage_root)


def outputs(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    stage = ensure_stage_paths(stage_root)
    kernel_ref = kaggle_kernel_ref(auth, spec)
    result = _kaggle_checked("kernels", "output", kernel_ref, "-p", str(stage.download_dir), timeout=180, phase="outputs_started", kernel_ref=kernel_ref, stage_root=stage_root)
    downloaded = [path.name for path in stage.download_dir.iterdir() if path.is_file() and path.name in SAFE_OUTPUT_NAMES]
    return {
        "notebook_ref": kernel_ref,
        "download_dir": str(stage.download_dir),
        "downloaded_safe_artifacts": sorted(downloaded),
        "stdout": _safe_cli_output(result),
    }


def _read_tail(path: Path, *, lines: int = 100) -> str | None:
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(content[-lines:])
    except Exception:
        return None


def _read_heartbeat(stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any] | None:
    path = ensure_stage_paths(stage_root).download_dir / "reports" / "smoke_heartbeat.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_breadcrumbs(stage_root: Path = DEFAULT_STAGE_ROOT) -> list[dict[str, Any]]:
    path = ensure_stage_paths(stage_root).download_dir / "reports" / "smoke_breadcrumbs.jsonl"
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _read_runner_heartbeat(stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any] | None:
    path = ensure_stage_paths(stage_root).stage_root / "runner_heartbeat.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_runner_failure(stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any] | None:
    path = ensure_stage_paths(stage_root).stage_root / "runner_failure.json"
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
    log_paths = list(stage.download_dir.glob("*.log"))
    payload: dict[str, Any] = {
        "last_breadcrumb_stage": None,
        "smoke_failure": None,
        "last_safe_log_lines": None,
        "last_progress_timestamp": None,
        "heartbeat": None,
        "live_log_tail": None,
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
    status_text = _sdk_kernel_status(auth, spec)
    if status_text is None:
        result = _kaggle_checked("kernels", "status", kernel_ref, timeout=60, phase="poll_started", kernel_ref=kernel_ref, start_time=start_time, stage_root=stage_root)
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


def full_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, tests_command: list[str] | None = None, spec: KaggleNotebookSpec | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    tests_command = tests_command or [sys.executable, "-m", "pytest", "-q"]
    test_run = _run_command(tests_command, check=False)
    if test_run.returncode != 0:
        raise KaggleAutomationError("local_tests_failed")
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root)
    output_result = outputs(spec, stage_root=stage_root)
    return {
        "tests": {"returncode": test_run.returncode, "stdout": test_run.stdout, "stderr": test_run.stderr},
        "preflight": preflight_result,
        "run": run_result,
        "outputs": output_result,
    }


def smoke_cycle(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    preflight_result = preflight(spec, stage_root=stage_root)
    if not preflight_result["ready"]:
        raise KaggleAutomationError(preflight_result.get("one_time_action") or "preflight_failed")
    run_result = run(spec, stage_root=stage_root)
    output_result = outputs(spec, stage_root=stage_root)
    return {
        "preflight": preflight_result,
        "run": run_result,
        "outputs": output_result,
    }


def diagnose(*, stage_root: Path = DEFAULT_STAGE_ROOT, spec: KaggleNotebookSpec | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    repo = get_repo_state()
    stage = ensure_stage_paths(stage_root)
    kernel_ref = kaggle_kernel_ref(auth, spec)
    status_result: dict[str, Any] | None = None
    status_error: str | None = None
    try:
        status_result = _status_payload(spec, stage_root=stage_root, include_error_artifacts=False)
    except Exception as exc:
        status_error = str(exc)
    return {
        "auth": auth.to_dict(),
        "repo": repo.to_dict(),
        "kernel_ref": kernel_ref,
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
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle notebook deployment and execution workflow")
    parser.add_argument("--dataset-ref", default=DEFAULT_DATASET_REF)
    parser.add_argument("--notebook-slug", default=DEFAULT_NOTEBOOK_SLUG)
    parser.add_argument("--title", default=DEFAULT_NOTEBOOK_TITLE)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("push")
    sub.add_parser("run")
    sub.add_parser("status")
    sub.add_parser("outputs")
    sub.add_parser("full-cycle")
    sub.add_parser("smoke-cycle")
    sub.add_parser("diagnose")
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
            _emit_json(push(spec, stage_root=args.stage_root))
        elif args.command == "run":
            _emit_json(run(spec, stage_root=args.stage_root))
        elif args.command == "status":
            _emit_json(status(spec))
        elif args.command == "outputs":
            _emit_json(outputs(spec, stage_root=args.stage_root))
        elif args.command == "full-cycle":
            _emit_json(full_cycle(stage_root=args.stage_root, spec=spec))
        elif args.command == "smoke-cycle":
            _emit_json(smoke_cycle(stage_root=args.stage_root, spec=spec))
        elif args.command == "diagnose":
            _emit_json(diagnose(stage_root=args.stage_root, spec=spec))
        else:
            raise KaggleAutomationError(f"unsupported_command:{args.command}")
        return 0
    except KaggleAutomationError as exc:
        _emit_json({"ok": False, "error": str(exc), "command": args.command, "dataset_ref": spec.dataset_ref, "kernel_ref": spec.notebook_ref(discover_kaggle_auth().username)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
