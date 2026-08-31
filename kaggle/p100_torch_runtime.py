from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TORCH_REQUESTED_VERSION = "2.5.1+cu118"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"
TORCH_PACKAGES = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
TORCH_INSTALL_TIMEOUT_SECONDS = 900
TORCH_RUNTIME_TIMEOUT_SECONDS = 120
TORCH_CUDA_TIMEOUT_SECONDS = 120
SHARED_TORCH_BOOTSTRAP_RESULT_JSON = "shared_torch_bootstrap_result.json"
SHARED_TORCH_RUNTIME_RESULT_JSON = "shared_torch_runtime_result.json"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _safe_tail(text: str | None, *, lines: int = 40) -> str | None:
    if not text:
        return None
    return "\n".join(text.splitlines()[-lines:])


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
cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--force-reinstall", "--no-cache-dir", "--index-url", "{TORCH_INDEX_URL}", *{TORCH_PACKAGES!r}]
result = subprocess.run(cmd, check=False)
raise SystemExit(result.returncode)
"""


def _torch_probe_snippet() -> str:
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


def _torch_cuda_validation_snippet() -> str:
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


def _run_json_probe(command: list[str], *, timeout: int, label: str) -> dict[str, Any]:
    result = _run_command(command, timeout=timeout)
    payload: dict[str, Any] = {
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


def _classify_torch_probe_failure(runtime_probe: dict[str, Any]) -> str:
    stderr = str(runtime_probe.get("stderr") or "").lower()
    stdout = str(runtime_probe.get("stdout") or "").lower()
    text = f"{stderr}\n{stdout}"
    if "skip_code" in text:
        return "TORCH_DYNAMO_BINARY_MISMATCH"
    if "cuda capability" in text or "sm_60" in text:
        return "PYTORCH_SM60_UNSUPPORTED"
    return "TORCH_RUNTIME_FAILED"


def run_shared_p100_torch_bootstrap(
    *,
    report_root: Path,
    repo_root: Path,
    phase_prefix: str,
    write_markers: bool = True,
) -> dict[str, Any]:
    bootstrap_result = {
        "phase_prefix": phase_prefix,
        "requested_version": TORCH_REQUESTED_VERSION,
        "torch_install_timeout_seconds": TORCH_INSTALL_TIMEOUT_SECONDS,
        "torch_runtime_timeout_seconds": TORCH_RUNTIME_TIMEOUT_SECONDS,
        "torch_cuda_timeout_seconds": TORCH_CUDA_TIMEOUT_SECONDS,
    }

    preinstall = _run_json_probe([sys.executable, "-c", _torch_probe_snippet()], timeout=TORCH_RUNTIME_TIMEOUT_SECONDS, label=f"{phase_prefix}_preinstall")
    _write_json(report_root / "probe_torch_preinstall.json", preinstall)
    bootstrap_result["preinstall"] = preinstall
    if not preinstall.get("ok"):
        raise RuntimeError(_classify_torch_probe_failure(preinstall))

    installer = _run_command([sys.executable, "-c", _torch_install_snippet()], timeout=TORCH_INSTALL_TIMEOUT_SECONDS, cwd=repo_root)
    install_payload = {
        "ok": installer.returncode == 0,
        "returncode": installer.returncode,
        "stdout_tail": _safe_tail(installer.stdout),
        "stderr_tail": _safe_tail(installer.stderr),
        "requested_version": TORCH_REQUESTED_VERSION,
    }
    try:
        from importlib.metadata import version

        install_payload["torch_distribution"] = version("torch")
    except Exception:
        install_payload["torch_distribution"] = None
    _write_json(report_root / "probe_torch_install.json", install_payload)
    bootstrap_result["install"] = install_payload
    if installer.returncode != 0:
        raise RuntimeError("TORCH_INSTALL_FAILED")

    runtime_probe = _run_json_probe([sys.executable, "-c", _torch_probe_snippet()], timeout=TORCH_RUNTIME_TIMEOUT_SECONDS, label=f"{phase_prefix}_runtime")
    _write_json(report_root / "probe_torch_runtime.json", runtime_probe)
    bootstrap_result["runtime"] = runtime_probe
    if not runtime_probe.get("ok"):
        raise RuntimeError(_classify_torch_probe_failure(runtime_probe))

    cuda_probe = _run_json_probe([sys.executable, "-c", _torch_cuda_validation_snippet()], timeout=TORCH_CUDA_TIMEOUT_SECONDS, label=f"{phase_prefix}_cuda")
    _write_json(report_root / "probe_torch_cuda_runtime.json", cuda_probe)
    bootstrap_result["cuda"] = cuda_probe
    if not cuda_probe.get("ok"):
        raise RuntimeError("PYTORCH_CUDA_FAILED")

    runtime_json = runtime_probe.get("json") or {}
    cuda_json = cuda_probe.get("json") or {}
    sm60_supported = tuple(runtime_json.get("compute_capability") or []) == (6, 0) and "sm_60" in (runtime_json.get("arch_list") or [])
    bootstrap_result["verdict"] = "P100_TORCH_RUNTIME_PASSED" if sm60_supported and cuda_json.get("basic_cuda_tensor_test") else "PYTORCH_SM60_UNSUPPORTED"
    bootstrap_result["torch_version"] = runtime_json.get("torch_version")
    bootstrap_result["torch_cuda_version"] = runtime_json.get("torch_cuda_version")
    bootstrap_result["gpu_name"] = runtime_json.get("gpu_name")
    bootstrap_result["compute_capability"] = runtime_json.get("compute_capability")
    bootstrap_result["arch_list"] = runtime_json.get("arch_list")
    bootstrap_result["sm_60_supported"] = sm60_supported
    bootstrap_result["basic_cuda_tensor_test"] = bool(cuda_json.get("basic_cuda_tensor_test"))
    bootstrap_result["skip_code_available"] = bool(runtime_json.get("skip_code_available"))
    if write_markers:
        _write_json(report_root / SHARED_TORCH_BOOTSTRAP_RESULT_JSON, bootstrap_result)
        _write_json(
            report_root / SHARED_TORCH_RUNTIME_RESULT_JSON,
            {
                "torch_version": bootstrap_result["torch_version"],
                "torch_cuda_version": bootstrap_result["torch_cuda_version"],
                "gpu_name": bootstrap_result["gpu_name"],
                "compute_capability": bootstrap_result["compute_capability"],
                "arch_list": bootstrap_result["arch_list"],
                "skip_code_available": bootstrap_result["skip_code_available"],
                "sm_60_supported": bootstrap_result["sm_60_supported"],
                "basic_cuda_tensor_test": bootstrap_result["basic_cuda_tensor_test"],
                "timestamp": time.time(),
            },
        )
    return bootstrap_result


def run_shared_p100_torch_validation(
    *,
    report_root: Path,
    phase_prefix: str,
    write_markers: bool = True,
) -> dict[str, Any]:
    validation = _run_json_probe([sys.executable, "-c", _torch_probe_snippet()], timeout=TORCH_RUNTIME_TIMEOUT_SECONDS, label=f"{phase_prefix}_runtime")
    _write_json(report_root / "probe_torch_runtime.json", validation)
    if not validation.get("ok"):
        raise RuntimeError(_classify_torch_probe_failure(validation))

    cuda_probe = _run_json_probe([sys.executable, "-c", _torch_cuda_validation_snippet()], timeout=TORCH_CUDA_TIMEOUT_SECONDS, label=f"{phase_prefix}_cuda")
    _write_json(report_root / "probe_torch_cuda_runtime.json", cuda_probe)
    if not cuda_probe.get("ok"):
        raise RuntimeError("PYTORCH_CUDA_FAILED")

    runtime_json = validation.get("json") or {}
    cuda_json = cuda_probe.get("json") or {}
    sm60_supported = tuple(runtime_json.get("compute_capability") or []) == (6, 0) and "sm_60" in (runtime_json.get("arch_list") or [])
    payload = {
        "phase_prefix": phase_prefix,
        "torch_version": runtime_json.get("torch_version"),
        "torch_cuda_version": runtime_json.get("torch_cuda_version"),
        "gpu_name": runtime_json.get("gpu_name"),
        "compute_capability": runtime_json.get("compute_capability"),
        "arch_list": runtime_json.get("arch_list"),
        "skip_code_available": bool(runtime_json.get("skip_code_available")),
        "sm_60_supported": sm60_supported,
        "basic_cuda_tensor_test": bool(cuda_json.get("basic_cuda_tensor_test")),
        "verdict": "P100_TORCH_RUNTIME_PASSED" if sm60_supported and cuda_json.get("basic_cuda_tensor_test") else "PYTORCH_SM60_UNSUPPORTED",
        "runtime": validation,
        "cuda": cuda_probe,
    }
    if write_markers:
        _write_json(report_root / SHARED_TORCH_RUNTIME_RESULT_JSON, payload)
    return payload
