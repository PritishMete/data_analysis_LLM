from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kaggle.bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths, inspect_kaggle_gpu_identity  # type: ignore[no-redef]
    from kaggle.import_trace import write_import_trace  # type: ignore[no-redef]
    from kaggle.run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity  # type: ignore[no-redef]
else:
    from .bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths, inspect_kaggle_gpu_identity
    from .import_trace import write_import_trace
    from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


CU118_INDEX_URL = "https://download.pytorch.org/whl/cu118"
TORCH_CU118_PACKAGE = "torch==2.5.1"
TORCH_CU118_PACKAGES = [TORCH_CU118_PACKAGE]
WORKFLOW_MODE = "torch_compat"


@dataclass(slots=True)
class TorchCompatCycleReport:
    run_id: str
    expected_git_commit: str | None
    executed_git_commit: str | None
    workflow_mode: str
    gpu: dict[str, Any]
    installer: dict[str, Any]
    runtime_probe: dict[str, Any]
    basic_cuda: dict[str, Any]
    artifact_paths: dict[str, str]
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:
            pass


def _safe_tail(text: str | None, *, lines: int = 40) -> str | None:
    if not text:
        return None
    return "\n".join(text.splitlines()[-lines:])


def _write_heartbeat(report_root: Path, *, stage: str, run_id: str, expected_git_commit: str | None, executed_git_commit: str | None, safe_message: str) -> None:
    _write_json(
        report_root / "smoke_heartbeat.json",
        {
            "stage": stage,
            "timestamp": time.time(),
            "run_id": run_id,
            "expected_git_commit": expected_git_commit,
            "executed_git_commit": executed_git_commit,
            "workflow_mode": WORKFLOW_MODE,
            "smoke_mode": False,
            "safe_message": safe_message,
        },
    )


def _emit_breadcrumb(path: Path, *, stage: str, success: bool, safe_message: str, run_id: str) -> None:
    _append_jsonl(
        path,
        {
            "run_id": run_id,
            "stage": stage,
            "success": success,
            "safe_message": safe_message,
            "timestamp": time.time(),
        },
    )


def _write_failure(report_root: Path, *, stage: str, exc: BaseException, run_id: str, expected_git_commit: str | None, executed_git_commit: str | None, stdout: str | None = None, stderr: str | None = None) -> Path:
    payload = {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "sanitized_message": str(exc).replace("\n", " ").strip()[:1000],
        "traceback_tail": None,
        "package_versions": {},
        "gpu": inspect_kaggle_gpu_identity(),
        "python_version": sys.version,
        "torch_version": None,
        "transformers_version": None,
        "peft_version": None,
        "bitsandbytes_version": None,
        "cuda_state": {
            "available": False,
            "device_name": None,
            "capability": None,
            "arch_list": None,
        },
        "run_id": run_id,
        "expected_git_commit": expected_git_commit,
        "executed_git_commit": executed_git_commit,
        "safe_stdout_tail": _safe_tail(stdout),
        "safe_stderr_tail": _safe_tail(stderr),
    }
    try:
        payload["traceback_tail"] = "\n".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-25:])
    except Exception:
        payload["traceback_tail"] = None
    for name in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            from importlib.metadata import version

            payload["package_versions"][name] = version(name)
        except Exception:
            payload["package_versions"][name] = None
    try:
        import torch

        payload["torch_version"] = torch.__version__
        payload["cuda_state"] = {
            "available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "arch_list": list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else None,
        }
    except Exception:
        pass
    path = report_root / "smoke_failure.json"
    return _write_json(path, payload)


def _run_command(command: list[str], *, timeout: int, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )


def _installer_snippet() -> str:
    return f"""
import subprocess, sys
cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir", "--index-url", "{CU118_INDEX_URL}", "{TORCH_CU118_PACKAGE}"]
result = subprocess.run(cmd, check=False)
raise SystemExit(result.returncode)
"""


def _probe_snippet() -> str:
    return """
import json
import torch
import torch._dynamo
from torch._C._dynamo.eval_frame import skip_code

payload = {
    "torch_version": torch.__version__,
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "torch_dynamo_import_passed": True,
    "skip_code_available": callable(skip_code),
    "gpu_available": bool(torch.cuda.is_available()),
    "gpu_name": None,
    "compute_capability": None,
    "arch_list": None,
}
if payload["gpu_available"]:
    payload["gpu_name"] = torch.cuda.get_device_name(0)
    payload["compute_capability"] = list(torch.cuda.get_device_capability(0))
    try:
        payload["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception:
        payload["arch_list"] = None
print(json.dumps(payload))
"""


def _validate_runtime_snippet() -> str:
    return """
import json
import torch

payload = {
    "cuda_available": bool(torch.cuda.is_available()),
    "device_name": None,
    "capability": None,
    "arch_list": list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else None,
    "basic_cuda_tensor_test": False,
    "synchronize": False,
}
if payload["cuda_available"]:
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["capability"] = list(torch.cuda.get_device_capability(0))
    x = torch.tensor([1.0], device="cuda")
    y = torch.tensor([2.0], device="cuda")
    z = x @ y.reshape(1, 1)
    torch.cuda.synchronize()
    payload["basic_cuda_tensor_test"] = bool(z.item() == 2.0)
    payload["synchronize"] = True
print(json.dumps(payload))
"""


def run_torch_compat_cycle(
    *,
    output_root: Path,
    run_id: str,
    expected_git_commit: str | None,
    source_root: Path,
    bootstrap_pid: int | None = None,
) -> dict[str, Any]:
    report_root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    paths = ensure_kaggle_paths(report_root)
    breadcrumbs_path = report_root / "smoke_breadcrumbs.jsonl"
    _write_heartbeat(report_root, stage="notebook_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=None, safe_message="torch compat start")
    _emit_breadcrumb(breadcrumbs_path, stage="notebook_started", success=True, safe_message="notebook started", run_id=run_id)
    _emit_breadcrumb(breadcrumbs_path, stage="bootstrap_started", success=True, safe_message="bootstrap start", run_id=run_id)
    _write_json(report_root / "runner_metadata.json", {"run_id": run_id, "expected_git_commit": expected_git_commit, "bootstrap_pid": bootstrap_pid, "workflow_mode": WORKFLOW_MODE, "timestamp": time.time()})
    repo_root = Path(source_root)
    resolved = resolve_executed_source_commit(run_root=report_root, repo_root=repo_root, expected_git_commit=expected_git_commit)
    executed_git_commit = resolved.get("executed_source_commit")
    write_source_identity(
        report_root,
        run_id=run_id,
        expected_git_commit=expected_git_commit,
        executed_source_commit=executed_git_commit,
        source_identity_method=str(resolved.get("source_identity_method") or "unknown"),
        source_identity_verified=bool(resolved.get("source_identity_verified")),
    )
    _write_heartbeat(report_root, stage="source_identity_verified", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="source identity verified")
    if expected_git_commit and executed_git_commit != expected_git_commit:
        exc = RuntimeError("stale_kaggle_checkout")
        _write_failure(report_root, stage="source_identity_verified", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit)
        raise exc

    _emit_breadcrumb(breadcrumbs_path, stage="repo_checkout_started", success=True, safe_message="repo checkout start", run_id=run_id)
    _emit_breadcrumb(breadcrumbs_path, stage="repo_checkout_complete", success=True, safe_message="repo checkout complete", run_id=run_id)
    _write_heartbeat(report_root, stage="repo_checkout_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="repo checkout complete")

    _emit_breadcrumb(breadcrumbs_path, stage="dependency_preflight_started", success=True, safe_message="dependency preflight start", run_id=run_id)
    _write_heartbeat(report_root, stage="dependency_preflight_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="dependency preflight start")

    installer = _run_command(
        [sys.executable, "-c", _installer_snippet()],
        timeout=900,
        cwd=repo_root,
    )
    installer_payload = {
        "ok": installer.returncode == 0,
        "returncode": installer.returncode,
        "stdout_tail": _safe_tail(installer.stdout),
        "stderr_tail": _safe_tail(installer.stderr),
        "torch_distribution": None,
    }
    try:
        from importlib.metadata import version

        installer_payload["torch_distribution"] = version("torch")
    except Exception:
        installer_payload["torch_distribution"] = None
    _write_json(report_root / "probe_torch_install.json", installer_payload)
    _emit_breadcrumb(
        breadcrumbs_path,
        stage="dependency_preflight_complete",
        success=bool(installer.returncode == 0),
        safe_message="torch install complete",
        run_id=run_id,
    )
    _write_heartbeat(report_root, stage="dependency_preflight_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="torch install complete")

    runtime_probe = _run_command([sys.executable, "-c", _probe_snippet()], timeout=120, cwd=repo_root)
    runtime_payload = {
        "ok": runtime_probe.returncode == 0,
        "returncode": runtime_probe.returncode,
        "stdout_tail": _safe_tail(runtime_probe.stdout),
        "stderr_tail": _safe_tail(runtime_probe.stderr),
        "json": None,
    }
    if runtime_probe.stdout:
        try:
            runtime_payload["json"] = json.loads(runtime_probe.stdout.splitlines()[-1])
        except Exception:
            runtime_payload["json"] = None
    _write_json(report_root / "probe_torch_runtime.json", runtime_payload)
    if not runtime_payload["ok"]:
        exc = RuntimeError("torch_runtime_probe_failed")
        _write_failure(report_root, stage="torch_runtime_probe", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=runtime_probe.stdout, stderr=runtime_probe.stderr)
        raise exc

    basic_cuda = _run_command([sys.executable, "-c", _validate_runtime_snippet()], timeout=120, cwd=repo_root)
    basic_payload = {
        "ok": basic_cuda.returncode == 0,
        "returncode": basic_cuda.returncode,
        "stdout_tail": _safe_tail(basic_cuda.stdout),
        "stderr_tail": _safe_tail(basic_cuda.stderr),
        "json": None,
    }
    if basic_cuda.stdout:
        try:
            basic_payload["json"] = json.loads(basic_cuda.stdout.splitlines()[-1])
        except Exception:
            basic_payload["json"] = None
    _write_json(report_root / "probe_torch_cuda_runtime.json", basic_payload)
    if not basic_payload["ok"]:
        exc = RuntimeError("basic_cuda_probe_failed")
        _write_failure(report_root, stage="torch_cuda_probe", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=basic_cuda.stdout, stderr=basic_cuda.stderr)
        raise exc

    probe_json = runtime_payload.get("json") or {}
    gpu_name = probe_json.get("gpu_name")
    capability = tuple(probe_json.get("compute_capability") or [])
    arch_list = list(probe_json.get("arch_list") or [])
    sm60_supported = capability == (6, 0) and "sm_60" in arch_list
    verdict = "P100_QLORA_RUNTIME_COMPATIBLE" if sm60_supported and basic_payload.get("ok") else "PYTORCH_SM60_UNSUPPORTED"
    final_report = TorchCompatCycleReport(
        run_id=run_id,
        expected_git_commit=expected_git_commit,
        executed_git_commit=executed_git_commit,
        workflow_mode=WORKFLOW_MODE,
        gpu=inspect_kaggle_gpu_identity(),
        installer=installer_payload,
        runtime_probe=runtime_payload,
        basic_cuda=basic_payload,
        artifact_paths={
            "probe_torch_install.json": str(report_root / "probe_torch_install.json"),
            "probe_torch_runtime.json": str(report_root / "probe_torch_runtime.json"),
            "probe_torch_cuda_runtime.json": str(report_root / "probe_torch_cuda_runtime.json"),
        },
        verdict=verdict,
    ).to_dict()
    _write_json(report_root / "torch_compat_report.json", final_report)
    _emit_breadcrumb(breadcrumbs_path, stage="torch_compat_complete", success=True, safe_message=verdict, run_id=run_id)
    _write_heartbeat(report_root, stage="torch_compat_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message=verdict)
    return final_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(KAGGLE_WORKING_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expected-git-commit", default=None)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--bootstrap-pid", type=int, default=None)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    resolved_run_id = args.run_id or resolve_current_run_id(base_root=output_root / "smoke_runs") or generate_run_id()
    report = run_torch_compat_cycle(
        output_root=output_root,
        run_id=resolved_run_id,
        expected_git_commit=args.expected_git_commit,
        source_root=Path(args.source_root) if args.source_root else (output_root / "data_analysis_LLM"),
        bootstrap_pid=args.bootstrap_pid,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("verdict") == "P100_QLORA_RUNTIME_COMPATIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
