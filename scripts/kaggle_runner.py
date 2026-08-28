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


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = REPO_ROOT / "kaggle" / "semantic_extractor_training.ipynb"
DEFAULT_DATASET_REF = "jaistudio/data-analysis-llm"
DEFAULT_NOTEBOOK_SLUG = "data-analysis-llm-semantic-extractor"
DEFAULT_NOTEBOOK_TITLE = "Data Analysis LLM Semantic Extractor"
DEFAULT_STAGE_ROOT = REPO_ROOT / "runtime" / "kaggle_runner"
SAFE_OUTPUT_NAMES = {
    "final_report.json",
    "semantic_metrics.json",
    "artifact_manifest.json",
    "semantic_extractor_artifacts.zip",
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
        check=check,
        timeout=timeout,
        cwd=cwd,
    )


def _safe_cli_output(result: subprocess.CompletedProcess[str]) -> str:
    text = result.stdout or ""
    return text.strip()


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
    if shutil.which("kaggle") is not None:
        return _run_command(["kaggle", *args], timeout=timeout)
    if _kaggle_python_available():
        return _run_command([sys.executable, "-m", "kaggle", *args], timeout=timeout, cwd=str(Path.home()))
    raise KaggleAutomationError("kaggle_cli_missing")


def preflight(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    repo = get_repo_state()
    stage = ensure_stage_paths(stage_root)
    notebook_ref = kaggle_kernel_ref(auth, spec)
    notebook_exists = None
    notebook_error = None
    if auth.available and auth.username:
        try:
            result = _kaggle("kernels", "status", notebook_ref)
            notebook_exists = result.returncode == 0
        except subprocess.CalledProcessError as exc:
            notebook_error = _safe_cli_output(exc)
            if "404" in (exc.stderr or "") or "not found" in (exc.stderr or "").lower():
                notebook_exists = False
            else:
                notebook_exists = None
        except Exception as exc:
            notebook_error = str(exc)
    dataset_ok = False
    dataset_error = None
    if auth.available:
        try:
            result = _kaggle("datasets", "files", "-d", spec.dataset_ref)
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
    result = _kaggle("kernels", "push", "-p", str(notebook_dir))
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
    push_result = push(spec, stage_root=stage_root)
    deadline = time.time() + timeout_seconds
    polls = 0
    status_text = None
    while time.time() < deadline:
        polls += 1
        result = _kaggle("kernels", "status", kernel_ref)
        status_text = _safe_cli_output(result)
        if any(token in status_text.lower() for token in ("complete", "error", "failed", "killed", "cancelled", "cancelled")):
            break
        time.sleep(poll_seconds)
    else:
        raise KaggleAutomationError("timeout")
    return {
        "notebook_ref": kernel_ref,
        "push": push_result,
        "status": status_text,
        "polls": polls,
    }


def status(spec: KaggleNotebookSpec | None = None) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    kernel_ref = kaggle_kernel_ref(auth, spec)
    result = _kaggle("kernels", "status", kernel_ref)
    return {
        "notebook_ref": kernel_ref,
        "status": _safe_cli_output(result),
    }


def outputs(spec: KaggleNotebookSpec | None = None, *, stage_root: Path = DEFAULT_STAGE_ROOT) -> dict[str, Any]:
    spec = spec or KaggleNotebookSpec()
    auth = discover_kaggle_auth()
    if not auth.available:
        raise KaggleAutomationError("authentication_missing")
    stage = ensure_stage_paths(stage_root)
    kernel_ref = kaggle_kernel_ref(auth, spec)
    result = _kaggle("kernels", "output", kernel_ref, "-p", str(stage.download_dir))
    downloaded = [path.name for path in stage.download_dir.iterdir() if path.is_file() and path.name in SAFE_OUTPUT_NAMES]
    return {
        "notebook_ref": kernel_ref,
        "download_dir": str(stage.download_dir),
        "downloaded_safe_artifacts": sorted(downloaded),
        "stdout": _safe_cli_output(result),
    }


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
        else:
            raise KaggleAutomationError(f"unsupported_command:{args.command}")
        return 0
    except KaggleAutomationError as exc:
        _emit_json({"ok": False, "error": str(exc), "command": args.command, "dataset_ref": spec.dataset_ref, "kernel_ref": spec.notebook_ref(discover_kaggle_auth().username)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
