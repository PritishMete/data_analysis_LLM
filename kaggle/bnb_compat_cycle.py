from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kaggle.bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths, inspect_kaggle_gpu_identity  # type: ignore[no-redef]
    from kaggle.import_trace import write_import_trace  # type: ignore[no-redef]
    from kaggle.p100_torch_runtime import run_shared_p100_torch_bootstrap, run_shared_p100_torch_validation  # type: ignore[no-redef]
    from kaggle.run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity  # type: ignore[no-redef]
else:
    from .bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths, inspect_kaggle_gpu_identity
    from .import_trace import write_import_trace
    from .p100_torch_runtime import run_shared_p100_torch_bootstrap, run_shared_p100_torch_validation
    from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


WORKFLOW_MODE = "bnb_compat"
BNB_REQUESTED_VERSION = "0.43.3"
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
    shared_torch_bootstrap: dict[str, Any]
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


def _write_failure(
    report_root: Path,
    *,
    stage: str,
    exc: BaseException,
    run_id: str,
    expected_git_commit: str | None,
    executed_git_commit: str | None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> Path:
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
        import traceback

        payload["traceback_tail"] = "\n".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-25:])
    except Exception:
        payload["traceback_tail"] = None
    for name in ("torch", "bitsandbytes", "transformers", "peft"):
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


def _installer_snippet() -> str:
    return f"""
import subprocess, sys
cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir", "--no-deps", "bitsandbytes=={BNB_REQUESTED_VERSION}"]
result = subprocess.run(cmd, check=False)
raise SystemExit(result.returncode)
"""


def _bnb_import_snippet() -> str:
    return """
import ctypes
import json
import os
from pathlib import Path
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
    "compiled_with_cuda": getattr(cextension, "COMPILED_WITH_CUDA", None),
    "lib_type": type(getattr(cextension, "lib", None)).__name__,
    "native_library_loaded": False,
    "required_cuda_symbols": {},
    "diagnostic_first_error": None,
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
    selected = payload["selected_native_cuda_library"]
    if selected and Path(str(selected)).exists():
        native = ctypes.CDLL(str(selected), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        payload["native_library_loaded"] = True
        symbols = ("cadam32bit_grad_fp32", "cget_col_row_stats", "cquantize_blockwise_fp16_nf4", "cdequantize_blockwise_fp16_nf4")
        payload["required_cuda_symbols"] = {name: bool(getattr(native, name, None)) for name in symbols}
except Exception as exc:
    payload["selected_native_cuda_library_error"] = str(exc)
try:
    diagnostic = getattr(getattr(cextension, "CUDASetup", None), "get_instance", lambda: None)()
    if diagnostic is not None and hasattr(diagnostic, "generate_instructions"):
        diagnostic.generate_instructions()
except Exception as exc:
    payload["diagnostic_first_error"] = str(exc)[:500]
print(json.dumps(payload))
"""


def _bnb_cuda_snippet() -> str:
    return """
import json
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F

payload = {
    "cuda_available": bool(torch.cuda.is_available()),
    "device_name": None,
    "capability": None,
    "arch_list": list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else None,
    "basic_cuda_tensor_test": False,
    "real_bnb_cuda_operation": False,
    "real_bnb_cuda_device": None,
    "real_bnb_cuda_error": None,
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
    if payload["cuda_available"]:
        tensor = torch.tensor([[0.25, -1.0], [2.5, 3.0]], device="cuda", dtype=torch.float16)
        quantized = F.quantize_4bit(tensor, quant_type="nf4")
        if isinstance(quantized, (tuple, list)):
            dequantized = F.dequantize_4bit(*quantized)
        else:
            dequantized = F.dequantize_4bit(quantized)
        torch.cuda.synchronize()
        payload["real_bnb_cuda_operation"] = bool(getattr(dequantized, "is_cuda", False))
        payload["real_bnb_cuda_device"] = str(dequantized.device)
except Exception as exc:
    payload["real_bnb_cuda_error"] = str(exc)[:1000]
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


def _emit_probe_result(marker: str, payload: dict[str, Any]) -> None:
    print(f"{marker}={json.dumps(payload, sort_keys=True)}", flush=True)


def _write_terminal_summary(
    report_root: Path,
    *,
    run_id: str,
    executed_commit: str | None,
    completed_stages: list[str],
    first_failed_stage: str | None,
    classification: str | None,
    final_verdict: str | None,
) -> Path:
    return _write_json(
        report_root / "bnb_terminal_summary.json",
        {
            "run_id": run_id,
            "executed_commit": executed_commit,
            "completed_stages": completed_stages,
            "first_failed_stage": first_failed_stage,
            "classification": classification,
            "final_verdict": final_verdict,
        },
    )


def _run_torch_preflight(report_root: Path, *, run_id: str, expected_git_commit: str | None, executed_git_commit: str | None, breadcrumbs_path: Path, repo_root: Path) -> dict[str, Any]:
    torch_result = run_shared_p100_torch_bootstrap(report_root=report_root, repo_root=repo_root, phase_prefix="bnb_compat", write_markers=True)
    _emit_breadcrumb(breadcrumbs_path, stage="dependency_precheck_complete", success=bool(torch_result.get("verdict") == "P100_TORCH_RUNTIME_PASSED"), safe_message=str(torch_result.get("verdict") or "unknown"), run_id=run_id)
    _write_heartbeat(report_root, stage="dependency_precheck_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message=str(torch_result.get("verdict") or "unknown"))
    if not torch_result.get("sm_60_supported") or not torch_result.get("basic_cuda_tensor_test"):
        raise RuntimeError("PYTORCH_SM60_UNSUPPORTED")
    return torch_result


def _run_bnb_compat_cycle_impl(
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

    torch_result = _run_torch_preflight(report_root, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, breadcrumbs_path=breadcrumbs_path, repo_root=repo_root)

    _emit_breadcrumb(breadcrumbs_path, stage="dependency_install_started", success=True, safe_message="bnb install start", run_id=run_id)
    _write_heartbeat(report_root, stage="dependency_install_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="bnb install start")
    installer = _run_command([sys.executable, "-c", _installer_snippet()], timeout=BNB_INSTALL_TIMEOUT_SECONDS, cwd=repo_root)
    installer_payload = {
        "ok": installer.returncode == 0,
        "returncode": installer.returncode,
        "stdout_tail": _safe_tail(installer.stdout),
        "stderr_tail": _safe_tail(installer.stderr),
        "requested_version": BNB_REQUESTED_VERSION,
        "torch_distribution_before": torch_result.get("torch_version"),
        "torch_distribution_after": None,
    }
    try:
        from importlib.metadata import version

        installer_payload["torch_distribution_after"] = version("torch")
    except Exception:
        installer_payload["torch_distribution_after"] = None
    _write_json(report_root / "probe_bnb_install.json", installer_payload)
    _emit_probe_result("BNB_INSTALL_RESULT_JSON", installer_payload)
    _emit_breadcrumb(breadcrumbs_path, stage="dependency_install_complete", success=bool(installer.returncode == 0), safe_message="bnb install complete", run_id=run_id)
    _write_heartbeat(report_root, stage="dependency_install_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="bnb install complete")
    if installer.returncode != 0:
        exc = RuntimeError("bitsandbytes_install_failed")
        _write_failure(report_root, stage="dependency_install_complete", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=installer.stdout, stderr=installer.stderr)
        raise exc

    post_install_torch = run_shared_p100_torch_validation(report_root=report_root, phase_prefix="bnb_compat", write_markers=True)
    _emit_probe_result("TORCH_POSTINSTALL_RESULT_JSON", post_install_torch)
    if post_install_torch.get("torch_version") != torch_result.get("torch_version"):
        exc = RuntimeError("TORCH_VERSION_DRIFT")
        _write_failure(report_root, stage="torch_version_drift", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit)
        raise exc

    import_probe = _run_json_probe([sys.executable, "-c", _bnb_import_snippet()], timeout=BNB_RUNTIME_TIMEOUT_SECONDS, label="bnb_import")
    _write_json(report_root / "probe_bnb_import.json", import_probe)
    _emit_probe_result("BNB_IMPORT_RESULT_JSON", import_probe)
    _emit_probe_result("BNB_INTERNAL_STATE_JSON", import_probe.get("json") or {})
    _emit_probe_result("BNB_NATIVE_SYMBOLS_JSON", {
        "selected_native_cuda_library": (import_probe.get("json") or {}).get("selected_native_cuda_library"),
        "native_library_loaded": (import_probe.get("json") or {}).get("native_library_loaded"),
        "required_cuda_symbols": (import_probe.get("json") or {}).get("required_cuda_symbols", {}),
    })
    if not import_probe["ok"]:
        exc = RuntimeError("bitsandbytes_import_failed")
        _write_failure(report_root, stage="bnb_import", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=import_probe.get("stdout"), stderr=import_probe.get("stderr"))
        raise exc

    cuda_probe = _run_json_probe([sys.executable, "-c", _bnb_cuda_snippet()], timeout=BNB_CUDA_TIMEOUT_SECONDS, label="bnb_cuda")
    _write_json(report_root / "probe_bnb_cuda.json", cuda_probe)
    _emit_probe_result("BNB_CUDA_RESULT_JSON", cuda_probe)
    _emit_probe_result("BNB_REAL_CUDA_OPERATION_JSON", cuda_probe.get("json") or {})
    if not cuda_probe["ok"]:
        exc = RuntimeError("bitsandbytes_cuda_failed")
        _write_failure(report_root, stage="bnb_cuda", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=cuda_probe.get("stdout"), stderr=cuda_probe.get("stderr"))
        raise exc

    cuda_json = cuda_probe.get("json") or {}
    real_bnb_cuda_operation = bool((cuda_json or {}).get("real_bnb_cuda_operation"))
    if not real_bnb_cuda_operation:
        verdict = "BITSANDBYTES_CPU_FALLBACK"
        nf4_probe = {"ok": False, "json": None}
    else:
        nf4_probe = _run_json_probe([sys.executable, "-c", _nf4_probe_snippet()], timeout=NF4_TIMEOUT_SECONDS, label="nf4")
        _write_json(report_root / "probe_nf4.json", nf4_probe)
        _emit_probe_result("NF4_RESULT_JSON", nf4_probe)
        if not nf4_probe["ok"]:
            exc = RuntimeError("nf4_runtime_failed")
            _write_failure(report_root, stage="nf4", exc=exc, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, stdout=nf4_probe.get("stdout"), stderr=nf4_probe.get("stderr"))
            raise exc
        nf4_json = nf4_probe.get("json") or {}
        if not (nf4_json.get("nf4_initialization") and nf4_json.get("nf4_quantization") and nf4_json.get("nf4_dequantization") and nf4_json.get("nf4_cuda")):
            verdict = "NF4_RUNTIME_FAILED"
        else:
            verdict = "BNB_NF4_P100_RUNTIME_PASSED"
    _emit_probe_result("NF4_RESULT_JSON", nf4_probe)

    final_report = BnbCompatCycleReport(
        run_id=run_id,
        expected_git_commit=expected_git_commit,
        executed_git_commit=executed_git_commit,
        workflow_mode=WORKFLOW_MODE,
        shared_torch_bootstrap=torch_result,
        gpu=inspect_kaggle_gpu_identity(),
        installer=installer_payload,
        import_probe=import_probe,
        cuda_probe=cuda_probe,
        nf4_probe=nf4_probe,
        artifact_paths={
            "probe_torch_preinstall.json": str(report_root / "probe_torch_preinstall.json"),
            "probe_torch_install.json": str(report_root / "probe_torch_install.json"),
            "probe_torch_runtime.json": str(report_root / "probe_torch_runtime.json"),
            "probe_torch_cuda_runtime.json": str(report_root / "probe_torch_cuda_runtime.json"),
            "shared_torch_bootstrap_result.json": str(report_root / "shared_torch_bootstrap_result.json"),
            "shared_torch_runtime_result.json": str(report_root / "shared_torch_runtime_result.json"),
            "probe_bnb_install.json": str(report_root / "probe_bnb_install.json"),
            "probe_bnb_import.json": str(report_root / "probe_bnb_import.json"),
            "probe_bnb_cuda.json": str(report_root / "probe_bnb_cuda.json"),
            "probe_nf4.json": str(report_root / "probe_nf4.json"),
        },
        verdict=verdict,
    ).to_dict()
    _write_json(report_root / "bnb_compat_report.json", final_report)
    _write_json(report_root / "bnb_internal_state.json", import_probe.get("json") or {})
    _write_json(report_root / "bnb_native_symbols.json", {
        "selected_native_cuda_library": (import_probe.get("json") or {}).get("selected_native_cuda_library"),
        "native_library_loaded": (import_probe.get("json") or {}).get("native_library_loaded"),
        "required_cuda_symbols": (import_probe.get("json") or {}).get("required_cuda_symbols", {}),
    })
    _write_json(report_root / "bnb_real_cuda_operation.json", cuda_json)
    _emit_probe_result("BNB_FINAL_RESULT_JSON", final_report)
    _write_terminal_summary(
        report_root,
        run_id=run_id,
        executed_commit=executed_git_commit,
        completed_stages=["preinstall_inspection", "torch_install", "cuda_runtime", "torch_postinstall", "bnb_install", "bnb_import", "bnb_cuda", "nf4", "terminal_summary"],
        first_failed_stage=None,
        classification=None,
        final_verdict=verdict,
    )
    _emit_breadcrumb(breadcrumbs_path, stage="bnb_compat_complete", success=True, safe_message=verdict, run_id=run_id)
    _write_heartbeat(report_root, stage="bnb_compat_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message=verdict)
    return final_report


def run_bnb_compat_cycle(
    *,
    output_root: Path,
    run_id: str,
    expected_git_commit: str | None,
    source_root: Path,
    bootstrap_pid: int | None = None,
) -> dict[str, Any]:
    report_root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    try:
        return _run_bnb_compat_cycle_impl(
            output_root=output_root,
            run_id=run_id,
            expected_git_commit=expected_git_commit,
            source_root=source_root,
            bootstrap_pid=bootstrap_pid,
        )
    except Exception as exc:
        _write_terminal_summary(
            report_root,
            run_id=run_id,
            executed_commit=None,
            completed_stages=[],
            first_failed_stage=str(exc),
            classification=str(exc),
            final_verdict=str(exc),
        )
        raise


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
