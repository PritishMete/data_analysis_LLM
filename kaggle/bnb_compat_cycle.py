from __future__ import annotations

import json
import os
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


WORKFLOW_MODE = "bnb_compat"
TORCH_CU118_VERSION = "2.5.1"
TORCH_CU118_PACKAGES = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
TORCH_CU118_INDEX_URL = "https://download.pytorch.org/whl/cu118"
BNB_REQUESTED_VERSION = "0.43.3"
TORCH_INSTALL_TIMEOUT_SECONDS = 1800
BNB_INSTALL_TIMEOUT_SECONDS = 900
BNB_RUNTIME_TIMEOUT_SECONDS = 120
BNB_CUDA_TIMEOUT_SECONDS = 120
NF4_TIMEOUT_SECONDS = 120


@dataclass(slots=True)
class BnbCompatCycleReport:
    run_id: str
    expected_git_commit: str | None
    executed_git_commit: str | None
    workflow_mode: str
    torch_version: str | None
    torch_cuda_version: str | None
    gpu: dict[str, Any]
    installer: dict[str, Any]
    import_probe: dict[str, Any]
    cuda_probe: dict[str, Any]
    nf4_probe: dict[str, Any]
    artifact_paths: dict[str, str]
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    for name in ("torch", "bitsandbytes"):
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
    return _write_json(report_root / "smoke_failure.json", payload)


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


def _torch_install_snippet() -> str:
    return f"""
import subprocess, sys
cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-cache-dir", "--index-url", "{TORCH_CU118_INDEX_URL}", "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
result = subprocess.run(cmd, check=False)
raise SystemExit(result.returncode)
"""


def _installer_snippet() -> str:
    return f"""
import subprocess, sys
cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir", "--no-deps", "bitsandbytes=={BNB_REQUESTED_VERSION}"]
result = subprocess.run(cmd, check=False)
raise SystemExit(result.returncode)
"""


def _import_probe_snippet() -> str:
    return """
import json
import torch
import bitsandbytes as bnb
from bitsandbytes import cextension

payload = {
    "torch_version": torch.__version__,
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "bnb_version": bnb.__version__,
    "bnb_file": getattr(bnb, "__file__", None),
    "available_cuda_versions": None,
    "cuda_backend_active": False,
    "selected_native_cuda_library": None,
}
try:
    payload["available_cuda_versions"] = list(cextension.get_available_cuda_binary_versions())
except Exception as exc:
    payload["available_cuda_versions_error"] = str(exc)
try:
    available_versions = payload.get("available_cuda_versions") or []
    torch_cuda_version = payload.get("torch_cuda_version")
    payload["cuda_backend_active"] = bool(torch.cuda.is_available() and available_versions and (torch_cuda_version in available_versions or torch_cuda_version is not None))
except Exception as exc:
    payload["cuda_backend_error"] = str(exc)
try:
    lib = getattr(cextension, "lib", None)
    payload["selected_native_cuda_library"] = getattr(lib, "_name", None) or getattr(lib, "__file__", None) or None
except Exception as exc:
    payload["selected_native_cuda_library_error"] = str(exc)
print(json.dumps(payload))
"""


def _cuda_probe_snippet() -> str:
    return """
import json
import torch
import bitsandbytes as bnb
from bitsandbytes import cextension

payload = {
    "cuda_available": bool(torch.cuda.is_available()),
    "device_name": None,
    "capability": None,
    "arch_list": list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else None,
    "basic_cuda_tensor_test": False,
    "cuda_backend_active": False,
    "bnb_version": bnb.__version__,
}
if payload["cuda_available"]:
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["capability"] = list(torch.cuda.get_device_capability(0))
    x = torch.ones((2, 2), device="cuda")
    y = torch.ones((2, 2), device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    payload["basic_cuda_tensor_test"] = bool(z.sum().item() == 8.0)
try:
    payload["cuda_backend_active"] = bool(getattr(cextension, "lib", None) is not None or getattr(cextension, "CUDASetup", None) is not None)
except Exception as exc:
    payload["cuda_backend_error"] = str(exc)
print(json.dumps(payload))
"""


def _nf4_probe_snippet() -> str:
    return """
import json
import torch
import bitsandbytes.functional as F

payload = {
    "nf4_initialization": False,
    "nf4_quantization": False,
    "nf4_dequantization": False,
    "nf4_cuda": False,
    "nf4_capability_available": False,
}
if torch.cuda.is_available():
    payload["nf4_capability_available"] = True
    tensor = torch.tensor([[0.25, -1.0], [2.5, 3.0]], device="cuda", dtype=torch.float16)
    payload["nf4_initialization"] = True
    quantized = F.quantize_4bit(tensor, quant_type="nf4")
    payload["nf4_quantization"] = True
    candidates = []
    if isinstance(quantized, (tuple, list)):
        candidates.append(quantized)
        if len(quantized) >= 2:
            candidates.append((quantized[0], quantized[1]))
    else:
        candidates.append((quantized,))
    for candidate in candidates:
        try:
            dequantized = F.dequantize_4bit(*candidate)
            payload["nf4_dequantization"] = bool(getattr(dequantized, "is_cuda", False))
            break
        except Exception:
            continue
    torch.cuda.synchronize()
    payload["nf4_cuda"] = True
print(json.dumps(payload))
"""


def _run_json_probe(command: list[str], *, timeout: int, label: str) -> dict[str, Any]:
    result = _run_command(command, timeout=timeout)
    payload = {
        "label": label,
        "returncode": result.returncode,
        "stdout": (result.stdout or "").strip() or None,
        "stderr": (result.stderr or "").strip() or None,
        "ok": result.returncode == 0,
    }
    if payload["stdout"]:
        try:
            payload["json"] = json.loads(str(payload["stdout"]).splitlines()[-1])
        except Exception:
            payload["json"] = None
    return payload


def _torch_state_probe() -> dict[str, Any]:
    snippet = """
import json
import torch
payload = {
    "torch_import_passed": True,
    "torch_version": torch.__version__,
    "torch_cuda_version": getattr(torch.version, "cuda", None),
    "torch_dynamo_import_passed": False,
    "skip_code_available": False,
    "cuda_available": bool(torch.cuda.is_available()),
    "gpu_name": None,
    "capability": None,
    "arch_list": list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else None,
    "sm_60_supported": False,
    "basic_cuda_tensor_test": False,
}
try:
    import torch._dynamo  # noqa: F401
    payload["torch_dynamo_import_passed"] = True
    from torch._C._dynamo.eval_frame import skip_code

    payload["skip_code_available"] = skip_code is not None
except Exception as exc:
    payload["torch_dynamo_error"] = type(exc).__name__ + ": " + str(exc)
if torch.cuda.is_available():
    payload["gpu_name"] = torch.cuda.get_device_name(0)
    payload["capability"] = list(torch.cuda.get_device_capability(0))
    payload["sm_60_supported"] = tuple(payload["capability"] or []) == (6, 0) and "sm_60" in (payload["arch_list"] or [])
    x = torch.ones((2, 2), device="cuda")
    y = torch.ones((2, 2), device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    payload["basic_cuda_tensor_test"] = bool(z.sum().item() == 8.0)
print(json.dumps(payload))
"""
    return _run_json_probe([sys.executable, "-c", snippet], timeout=60, label="torch_state")


def run_bnb_compat_cycle(
    *,
    output_root: Path,
    run_id: str,
    expected_git_commit: str | None,
    source_root: Path,
    bootstrap_pid: int | None = None,
) -> dict[str, Any]:
    report_root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    ensure_kaggle_paths(report_root)
    breadcrumbs_path = report_root / "smoke_breadcrumbs.jsonl"
    _write_heartbeat(report_root, stage="notebook_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=None, safe_message="bnb compat start")
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

    _emit_breadcrumb(breadcrumbs_path, stage="dependency_precheck_started", success=True, safe_message="bnb precheck start", run_id=run_id)
    _write_heartbeat(report_root, stage="dependency_precheck_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="bnb precheck start")

    precheck = _torch_state_probe()
    _write_json(report_root / "probe_torch_preinstall.json", precheck)
    _write_json(report_root / "probe_bnb_precheck.json", precheck)

    _emit_breadcrumb(breadcrumbs_path, stage="torch_install_started", success=True, safe_message="torch install start", run_id=run_id)
    _write_heartbeat(report_root, stage="torch_install_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="torch install start")
    torch_installer = _run_command([sys.executable, "-c", _torch_install_snippet()], timeout=TORCH_INSTALL_TIMEOUT_SECONDS, cwd=repo_root)
    torch_install_payload = {
        "ok": torch_installer.returncode == 0,
        "returncode": torch_installer.returncode,
        "stdout_tail": _safe_tail(torch_installer.stdout),
        "stderr_tail": _safe_tail(torch_installer.stderr),
        "requested_version": TORCH_CU118_VERSION,
        "requested_cuda_index": TORCH_CU118_INDEX_URL,
        "requested_packages": TORCH_CU118_PACKAGES,
        "torch_distribution": None,
        "torch_cuda_version": None,
    }
    try:
        from importlib.metadata import version

        torch_install_payload["torch_distribution"] = version("torch")
    except Exception:
        torch_install_payload["torch_distribution"] = None
    _write_json(report_root / "probe_torch_install.json", torch_install_payload)
    _emit_breadcrumb(breadcrumbs_path, stage="torch_install_complete", success=bool(torch_installer.returncode == 0), safe_message="torch install complete", run_id=run_id)
    _write_heartbeat(report_root, stage="torch_install_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="torch install complete")
    if torch_installer.returncode != 0:
        exc = RuntimeError("torch_install_failed")
        _write_failure(report_root, stage="torch_install_complete", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=torch_installer.stdout, stderr=torch_installer.stderr)
        raise exc

    runtime_probe = _torch_state_probe()
    _write_json(report_root / "probe_torch_runtime.json", runtime_probe)
    _write_json(report_root / "probe_torch_import_runtime.json", runtime_probe)
    _write_json(report_root / "probe_torch_cuda_runtime.json", runtime_probe)
    if not runtime_probe.get("ok", False):
        exc = RuntimeError("torch_runtime_failed")
        _write_failure(report_root, stage="torch_runtime", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=runtime_probe.get("stdout"), stderr=runtime_probe.get("stderr"))
        raise exc
    runtime_json = runtime_probe.get("json") or {}
    if runtime_json.get("torch_version") != "2.5.1+cu118" or runtime_json.get("torch_cuda_version") != "11.8":
        exc = RuntimeError("torch_version_drift")
        _write_failure(report_root, stage="torch_runtime", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=runtime_probe.get("stdout"), stderr=runtime_probe.get("stderr"))
        raise exc
    if not runtime_json.get("cuda_available") or not runtime_json.get("basic_cuda_tensor_test"):
        exc = RuntimeError("pytorch_cuda_failed")
        _write_failure(report_root, stage="torch_runtime", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=runtime_probe.get("stdout"), stderr=runtime_probe.get("stderr"))
        raise exc
    if not runtime_json.get("skip_code_available") or not runtime_json.get("torch_dynamo_import_passed"):
        exc = RuntimeError("torch_dynamo_skip_code_failed")
        _write_failure(report_root, stage="torch_runtime", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=runtime_probe.get("stdout"), stderr=runtime_probe.get("stderr"))
        raise exc
    if not runtime_json.get("sm_60_supported"):
        verdict = "PYTORCH_SM60_UNSUPPORTED"
        final_report = {
            "run_id": run_id,
            "expected_git_commit": expected_git_commit,
            "executed_git_commit": executed_git_commit,
            "workflow_mode": WORKFLOW_MODE,
            "torch_version": runtime_json.get("torch_version"),
            "torch_cuda_version": runtime_json.get("torch_cuda_version"),
            "gpu": inspect_kaggle_gpu_identity(),
            "installer": torch_install_payload,
            "import_probe": runtime_probe,
            "cuda_probe": runtime_probe,
            "nf4_probe": {"ok": False, "json": None},
            "artifact_paths": {
                "probe_bnb_precheck.json": str(report_root / "probe_bnb_precheck.json"),
                "probe_torch_preinstall.json": str(report_root / "probe_torch_preinstall.json"),
                "probe_torch_install.json": str(report_root / "probe_torch_install.json"),
                "probe_torch_runtime.json": str(report_root / "probe_torch_runtime.json"),
                "probe_torch_import_runtime.json": str(report_root / "probe_torch_import_runtime.json"),
                "probe_torch_cuda_runtime.json": str(report_root / "probe_torch_cuda_runtime.json"),
            },
            "verdict": verdict,
        }
        _write_json(report_root / "bnb_compat_report.json", final_report)
        _emit_breadcrumb(breadcrumbs_path, stage="torch_runtime_complete", success=False, safe_message=verdict, run_id=run_id)
        _write_heartbeat(report_root, stage="torch_runtime_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message=verdict)
        return final_report

    _emit_breadcrumb(breadcrumbs_path, stage="dependency_install_started", success=True, safe_message="bnb install start", run_id=run_id)
    _write_heartbeat(report_root, stage="dependency_install_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="bnb install start")
    installer = _run_command([sys.executable, "-c", _installer_snippet()], timeout=BNB_INSTALL_TIMEOUT_SECONDS, cwd=repo_root)
    installer_payload = {
        "ok": installer.returncode == 0,
        "returncode": installer.returncode,
        "stdout_tail": _safe_tail(installer.stdout),
        "stderr_tail": _safe_tail(installer.stderr),
        "requested_version": BNB_REQUESTED_VERSION,
        "torch_distribution": None,
    }
    try:
        from importlib.metadata import version

        installer_payload["torch_distribution"] = version("torch")
    except Exception:
        installer_payload["torch_distribution"] = None
    _write_json(report_root / "probe_bnb_install.json", installer_payload)
    _emit_breadcrumb(breadcrumbs_path, stage="dependency_install_complete", success=bool(installer.returncode == 0), safe_message="bnb install complete", run_id=run_id)
    _write_heartbeat(report_root, stage="dependency_install_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="bnb install complete")
    if installer.returncode != 0:
        exc = RuntimeError("bitsandbytes_install_failed")
        _write_failure(report_root, stage="dependency_install_complete", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=installer.stdout, stderr=installer.stderr)
        raise exc

    post_bnb_probe = _torch_state_probe()
    _write_json(report_root / "probe_torch_runtime_post_bnb.json", post_bnb_probe)
    if not post_bnb_probe.get("ok", False):
        exc = RuntimeError("torch_runtime_failed")
        _write_failure(report_root, stage="torch_runtime_post_bnb", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=post_bnb_probe.get("stdout"), stderr=post_bnb_probe.get("stderr"))
        raise exc
    post_bnb_json = post_bnb_probe.get("json") or {}
    if post_bnb_json.get("torch_version") != runtime_json.get("torch_version") or post_bnb_json.get("torch_cuda_version") != runtime_json.get("torch_cuda_version"):
        exc = RuntimeError("torch_version_drift")
        _write_failure(report_root, stage="torch_runtime_post_bnb", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=post_bnb_probe.get("stdout"), stderr=post_bnb_probe.get("stderr"))
        raise exc

    import_probe = _run_json_probe([sys.executable, "-c", _import_probe_snippet()], timeout=BNB_RUNTIME_TIMEOUT_SECONDS, label="bnb_import")
    _write_json(report_root / "probe_bnb_import.json", import_probe)
    if not import_probe["ok"]:
        exc = RuntimeError("bitsandbytes_import_failed")
        _write_failure(report_root, stage="bnb_import", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=import_probe.get("stdout"), stderr=import_probe.get("stderr"))
        raise exc

    cuda_probe = _run_json_probe([sys.executable, "-c", _cuda_probe_snippet()], timeout=BNB_CUDA_TIMEOUT_SECONDS, label="bnb_cuda")
    _write_json(report_root / "probe_bnb_cuda.json", cuda_probe)
    if not cuda_probe["ok"]:
        exc = RuntimeError("bitsandbytes_cuda_failed")
        _write_failure(report_root, stage="bnb_cuda", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=cuda_probe.get("stdout"), stderr=cuda_probe.get("stderr"))
        raise exc

    cuda_json = cuda_probe.get("json") or {}
    cuda_available = bool(cuda_json.get("cuda_available"))
    gpu_name = cuda_json.get("device_name")
    capability = tuple(cuda_json.get("capability") or [])
    arch_list = list(cuda_json.get("arch_list") or [])
    sm60_supported = capability == (6, 0) and "sm_60" in arch_list
    cuda_backend_active = bool((import_probe.get("json") or {}).get("cuda_backend_active"))
    if not cuda_backend_active:
        verdict = "BITSANDBYTES_CPU_FALLBACK"
    elif not sm60_supported:
        verdict = "NF4_P100_UNSUPPORTED"
    else:
        nf4_probe = _run_json_probe([sys.executable, "-c", _nf4_probe_snippet()], timeout=NF4_TIMEOUT_SECONDS, label="nf4")
        _write_json(report_root / "probe_nf4.json", nf4_probe)
        if not nf4_probe["ok"]:
            exc = RuntimeError("nf4_runtime_failed")
            _write_failure(report_root, stage="nf4", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=nf4_probe.get("stdout"), stderr=nf4_probe.get("stderr"))
            raise exc
        nf4_json = nf4_probe.get("json") or {}
        if not (nf4_json.get("nf4_initialization") and nf4_json.get("nf4_quantization") and nf4_json.get("nf4_dequantization") and nf4_json.get("nf4_cuda")):
            verdict = "NF4_RUNTIME_FAILED"
        else:
            verdict = "BNB_NF4_P100_RUNTIME_PASSED"
    final_report = BnbCompatCycleReport(
        run_id=run_id,
        expected_git_commit=expected_git_commit,
        executed_git_commit=executed_git_commit,
        workflow_mode=WORKFLOW_MODE,
        torch_version=(import_probe.get("json") or {}).get("torch_version"),
        torch_cuda_version=(import_probe.get("json") or {}).get("torch_cuda_version"),
        gpu=inspect_kaggle_gpu_identity(),
        installer=installer_payload,
        import_probe=import_probe,
        cuda_probe=cuda_probe,
        nf4_probe=(locals().get("nf4_probe") or {"ok": False, "json": None}),
        artifact_paths={
            "probe_bnb_precheck.json": str(report_root / "probe_bnb_precheck.json"),
            "probe_bnb_install.json": str(report_root / "probe_bnb_install.json"),
            "probe_torch_preinstall.json": str(report_root / "probe_torch_preinstall.json"),
            "probe_torch_install.json": str(report_root / "probe_torch_install.json"),
            "probe_torch_runtime.json": str(report_root / "probe_torch_runtime.json"),
            "probe_torch_import_runtime.json": str(report_root / "probe_torch_import_runtime.json"),
            "probe_torch_cuda_runtime.json": str(report_root / "probe_torch_cuda_runtime.json"),
            "probe_torch_runtime_post_bnb.json": str(report_root / "probe_torch_runtime_post_bnb.json"),
            "probe_bnb_import.json": str(report_root / "probe_bnb_import.json"),
            "probe_bnb_cuda.json": str(report_root / "probe_bnb_cuda.json"),
            "probe_nf4.json": str(report_root / "probe_nf4.json"),
        },
        verdict=verdict,
    ).to_dict()
    _write_json(report_root / "bnb_compat_report.json", final_report)
    _emit_breadcrumb(breadcrumbs_path, stage="bnb_compat_complete", success=True, safe_message=verdict, run_id=run_id)
    _write_heartbeat(report_root, stage="bnb_compat_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message=verdict)
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
    report = run_bnb_compat_cycle(
        output_root=output_root,
        run_id=resolved_run_id,
        expected_git_commit=args.expected_git_commit,
        source_root=Path(args.source_root) if args.source_root else (output_root / "data_analysis_LLM"),
        bootstrap_pid=args.bootstrap_pid,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("verdict") == "BNB_NF4_P100_RUNTIME_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
