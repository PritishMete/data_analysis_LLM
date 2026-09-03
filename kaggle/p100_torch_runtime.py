from __future__ import annotations

import json
import ctypes
import importlib.metadata as metadata
import os
import subprocess
import sys
import time
from packaging.requirements import InvalidRequirement, Requirement
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
TORCH_PREINSTALL_INSPECTION_JSON = "TORCH_PREINSTALL_INSPECTION_JSON"
TORCH_INSTALL_RESULT_JSON = "TORCH_INSTALL_RESULT_JSON"
TORCH_POSTINSTALL_RUNTIME_JSON = "TORCH_POSTINSTALL_RUNTIME_JSON"
TORCH_POSTINSTALL_CUDA_JSON = "TORCH_POSTINSTALL_CUDA_JSON"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _safe_tail(text: str | None, *, lines: int = 40) -> str | None:
    if not text:
        return None
    return "\n".join(text.splitlines()[-lines:])


def _torch_cuda_requirement_records() -> list[dict[str, Any]]:
    """Parse applicable CUDA 11 requirements from Torch metadata."""
    try:
        requirements = getattr(metadata.distribution("torch"), "requires", None) or []
    except metadata.PackageNotFoundError:
        return []
    result: list[dict[str, Any]] = []
    for requirement in requirements:
        try:
            parsed = Requirement(str(requirement))
        except InvalidRequirement as exc:
            if str(requirement).lower().startswith("nvidia-") and "-cu11" in str(requirement).lower():
                raise RuntimeError("CUDA_DEPENDENCY_PARSE_FAILED") from exc
            continue
        if not parsed.name.lower().startswith("nvidia-") or "-cu11" not in parsed.name.lower():
            continue
        if parsed.marker is not None and not parsed.marker.evaluate():
            continue
        record = {
            "name": parsed.name,
            "specifier": str(parsed.specifier),
            "requirement": parsed.name + str(parsed.specifier),
            "marker": str(parsed.marker) if parsed.marker is not None else None,
        }
        if record not in result:
            result.append(record)
    return sorted(result, key=lambda item: item["name"].lower())


def _torch_cuda_requirements() -> list[str]:
    """Return normalized, marker-filtered install requirements for CUDA 11."""
    return [record["requirement"] for record in _torch_cuda_requirement_records()]


def _cuda_library_paths() -> dict[str, list[str]]:
    names = {"libcudart": "libcudart.so", "libcublas": "libcublas.so", "libcusparse": "libcusparse.so"}
    found = {key: [] for key in names}
    records = _torch_cuda_requirement_records()
    if not records:
        records = []
        for requirement in _torch_cuda_requirements():
            parsed = Requirement(requirement)
            records.append({"name": parsed.name, "specifier": str(parsed.specifier), "requirement": requirement})
    for record in records:
        package_name = record["name"]
        try:
            distribution = metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            continue
        files = getattr(distribution, "files", None) or []
        candidates = [Path(distribution.locate_file(file)) for file in files if str(file).lower().endswith(".so") or ".so." in str(file).lower()]
        if not candidates:
            root = Path(distribution.locate_file(""))
            candidates = list(root.rglob("*.so*"))
        for path in candidates:
            for key, prefix in names.items():
                if path.name.startswith(prefix) and str(path) not in found[key]:
                    found[key].append(str(path))
    return found


def _prepare_cuda_runtime(report_root: Path) -> dict[str, Any]:
    requirement_records = _torch_cuda_requirement_records()
    requirements = [record["requirement"] for record in requirement_records]
    before = None
    try:
        before = metadata.version("torch")
    except metadata.PackageNotFoundError:
        pass
    installed_before = {record["name"]: _version_or_none(record["name"]) for record in requirement_records}
    missing = [record["requirement"] for record in requirement_records if not _installed_satisfies(record)]
    install_result: dict[str, Any] = {"attempted": bool(missing), "requirements": missing, "returncode": 0}
    if missing:
        result = _run_command(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir", "--no-deps", *missing],
            timeout=TORCH_INSTALL_TIMEOUT_SECONDS,
        )
        install_result.update({"returncode": result.returncode, "stdout_tail": _safe_tail(result.stdout), "stderr_tail": _safe_tail(result.stderr)})
        if result.returncode != 0:
            install_result["classification"] = "CUDA_RUNTIME_DEPENDENCY_INSTALL_FAILED"
    after = None
    try:
        after = metadata.version("torch")
    except metadata.PackageNotFoundError:
        pass
    if before and after != before:
        raise RuntimeError("TORCH_VERSION_DRIFT")
    paths = _cuda_library_paths()
    directories = sorted({str(Path(path).parent) for values in paths.values() for path in values})
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in directories if entry]
    if existing:
        entries.extend(part for part in existing.split(os.pathsep) if part and part not in entries)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(entries)
    loads: dict[str, Any] = {}
    for key, values in paths.items():
        loads[key] = {"path": values[0] if values else None, "passed": False, "error": None}
        if values:
            try:
                ctypes.CDLL(values[0])
                loads[key]["passed"] = True
            except OSError as exc:
                loads[key]["error"] = str(exc)[:500]
    if install_result.get("returncode") != 0:
        classification = "CUDA_RUNTIME_DEPENDENCY_INSTALL_FAILED"
    elif any(not installed_before.get(record["name"]) and not _version_or_none(record["name"]) for record in requirement_records):
        classification = "CUDA_PACKAGE_MISSING"
    elif any(not item["path"] for item in loads.values()):
        classification = "CUDA_LIBRARY_NOT_FOUND"
    elif any(not item["passed"] for item in loads.values()):
        classification = "CUDA_RUNTIME_LIBRARY_LOAD_FAILED"
    else:
        classification = "CUDA_RUNTIME_READY"
    payload = {
        "torch_version_before": before,
        "torch_version_after": after,
        "torch_cuda_requirements": requirement_records,
        "installed_nvidia_packages": {
            record["name"]: {
                "required_specifier": record["specifier"],
                "installed_version": _version_or_none(record["name"]),
                "location": _distribution_location(record["name"]),
            }
            for record in requirement_records
        },
        "missing_before_install": missing,
        "install": install_result,
        "library_paths": paths,
        "ld_library_path_entries": entries,
        "library_loads": loads,
        "classification": classification,
    }
    _write_json(report_root / "cuda_dependency_inspection.json", payload)
    _marker("CUDA_DEPENDENCY_INSPECTION_JSON", payload)
    _marker("CUDA_DEPENDENCY_INSTALL_JSON", install_result)
    _marker("CUDA_LIBRARY_PATH_JSON", {"entries": entries})
    _marker("CUDA_LIBRARY_LOAD_JSON", loads)
    return payload


def _version_or_none(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _distribution_location(name: str) -> str | None:
    try:
        return str(metadata.distribution(name).locate_file(""))
    except metadata.PackageNotFoundError:
        return None


def _installed_satisfies(record: dict[str, Any]) -> bool:
    installed = _version_or_none(record["name"])
    if installed is None:
        return False
    try:
        return Requirement(record["name"] + record["specifier"]).specifier.contains(installed, prereleases=True)
    except InvalidRequirement:
        return False


def _marker(name: str, payload: Any) -> None:
    print(f"{name}={json.dumps(payload, sort_keys=True)}", flush=True)


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


def _torch_preinstall_inspection_snippet() -> str:
    return """
import json
import importlib.metadata as metadata

payload = {
    "torch_distribution": None,
    "torch_module_version": None,
    "torch_cuda_version": None,
    "torch_location": None,
    "default_torch_appears_p100_incompatible": None,
    "inspect_only": True,
}
try:
    import torch
    payload["torch_module_version"] = getattr(torch, "__version__", None)
    payload["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    payload["torch_location"] = getattr(torch, "__file__", None)
    try:
        payload["default_torch_appears_p100_incompatible"] = not bool(torch.cuda.is_available()) or tuple(torch.cuda.get_device_capability(0)) != (6, 0) if torch.cuda.is_available() else True
    except Exception:
        payload["default_torch_appears_p100_incompatible"] = True
except Exception as exc:
    payload["torch_import_error"] = str(exc)
try:
    payload["torch_distribution"] = metadata.version("torch")
except Exception as exc:
    payload["torch_distribution_error"] = str(exc)
print(json.dumps(payload))
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


def _classify_preinstall_inspection(preinstall_probe: dict[str, Any]) -> str | None:
    if not preinstall_probe.get("ok"):
        return "TORCH_PREINSTALL_INSPECTION_FAILED"
    return None


def _torch_profile_failure(runtime: dict[str, Any], cuda: dict[str, Any]) -> str | None:
    """Classify the pinned runtime only after it was freshly imported."""
    payload = runtime.get("json") or {}
    if payload.get("torch_version") != TORCH_REQUESTED_VERSION:
        return "TORCH_VERSION_MISMATCH"
    if payload.get("torch_cuda_version") != "11.8":
        return "TORCH_CUDA_VERSION_MISMATCH"
    if tuple(payload.get("compute_capability") or ()) != (6, 0) or "sm_60" not in (payload.get("arch_list") or []):
        return "TORCH_P100_ARCH_UNSUPPORTED"
    if not payload.get("skip_code_available"):
        return "TORCH_DYNAMO_BINARY_MISMATCH"
    if not bool((cuda.get("json") or {}).get("basic_cuda_tensor_test")):
        return "PYTORCH_CUDA_FAILED"
    return None


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

    preinstall = _run_json_probe([sys.executable, "-c", _torch_preinstall_inspection_snippet()], timeout=TORCH_RUNTIME_TIMEOUT_SECONDS, label=f"{phase_prefix}_preinstall")
    _write_json(report_root / "probe_torch_preinstall.json", preinstall)
    _write_json(report_root / TORCH_PREINSTALL_INSPECTION_JSON, preinstall)
    bootstrap_result["preinstall"] = preinstall
    if not preinstall.get("ok"):
        raise RuntimeError(_classify_preinstall_inspection(preinstall) or "TORCH_PREINSTALL_INSPECTION_FAILED")

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
    _write_json(report_root / TORCH_INSTALL_RESULT_JSON, install_payload)
    bootstrap_result["install"] = install_payload
    if installer.returncode != 0:
        raise RuntimeError("TORCH_INSTALL_FAILED")

    cuda_runtime = _prepare_cuda_runtime(report_root)
    bootstrap_result["cuda_runtime"] = cuda_runtime
    if cuda_runtime["classification"] != "CUDA_RUNTIME_READY":
        raise RuntimeError(cuda_runtime["classification"])

    runtime_probe = _run_json_probe([sys.executable, "-c", _torch_probe_snippet()], timeout=TORCH_RUNTIME_TIMEOUT_SECONDS, label=f"{phase_prefix}_runtime")
    _write_json(report_root / "probe_torch_runtime.json", runtime_probe)
    _write_json(report_root / TORCH_POSTINSTALL_RUNTIME_JSON, runtime_probe)
    bootstrap_result["runtime"] = runtime_probe
    if not runtime_probe.get("ok"):
        raise RuntimeError(_classify_torch_probe_failure(runtime_probe))

    cuda_probe = _run_json_probe([sys.executable, "-c", _torch_cuda_validation_snippet()], timeout=TORCH_CUDA_TIMEOUT_SECONDS, label=f"{phase_prefix}_cuda")
    _write_json(report_root / "probe_torch_cuda_runtime.json", cuda_probe)
    _write_json(report_root / TORCH_POSTINSTALL_CUDA_JSON, cuda_probe)
    bootstrap_result["cuda"] = cuda_probe
    if not cuda_probe.get("ok"):
        raise RuntimeError("PYTORCH_CUDA_FAILED")

    runtime_json = runtime_probe.get("json") or {}
    cuda_json = cuda_probe.get("json") or {}
    sm60_supported = tuple(runtime_json.get("compute_capability") or []) == (6, 0) and "sm_60" in (runtime_json.get("arch_list") or [])
    profile_failure = _torch_profile_failure(runtime_probe, cuda_probe)
    bootstrap_result["verdict"] = "P100_TORCH_RUNTIME_PASSED" if profile_failure is None else profile_failure
    bootstrap_result["runtime_profile_verified"] = profile_failure is None
    install_payload["runtime_verified"] = profile_failure is None
    install_payload["ok"] = bool(install_payload.get("ok")) and profile_failure is None
    install_payload["failure_classification"] = profile_failure
    bootstrap_result["install"] = install_payload
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
    if profile_failure is not None:
        raise RuntimeError(profile_failure)
    return bootstrap_result


def run_shared_p100_torch_validation(
    *,
    report_root: Path,
    phase_prefix: str,
    write_markers: bool = True,
) -> dict[str, Any]:
    validation = _run_json_probe([sys.executable, "-c", _torch_probe_snippet()], timeout=TORCH_RUNTIME_TIMEOUT_SECONDS, label=f"{phase_prefix}_runtime")
    _write_json(report_root / "probe_torch_runtime.json", validation)
    _write_json(report_root / TORCH_POSTINSTALL_RUNTIME_JSON, validation)
    if not validation.get("ok"):
        raise RuntimeError(_classify_torch_probe_failure(validation))

    cuda_probe = _run_json_probe([sys.executable, "-c", _torch_cuda_validation_snippet()], timeout=TORCH_CUDA_TIMEOUT_SECONDS, label=f"{phase_prefix}_cuda")
    _write_json(report_root / "probe_torch_cuda_runtime.json", cuda_probe)
    _write_json(report_root / TORCH_POSTINSTALL_CUDA_JSON, cuda_probe)
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
