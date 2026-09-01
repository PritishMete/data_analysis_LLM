from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .bnb_compat_cycle import (
    BNB_REQUESTED_VERSION,
    TORCH_CU118_INDEX_URL,
    TORCH_CU118_PACKAGES,
    TORCH_CU118_VERSION,
    _safe_tail,
    _torch_install_snippet,
    _torch_state_probe,
)
from .bootstrap import ensure_kaggle_paths, inspect_kaggle_gpu_identity
from .run_context import ensure_run_root, resolve_executed_source_commit, write_source_identity


WORKFLOW_MODE = "bnb_native_diagnose"
INSTALL_TIMEOUT_SECONDS = 1800
DIAGNOSTIC_TIMEOUT_SECONDS = 60


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _marker(name: str, payload: dict[str, Any]) -> None:
    print(f"{name}={json.dumps(payload, sort_keys=True)}", flush=True)


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cuda_tag(cuda_version: str | None) -> str | None:
    if not cuda_version:
        return None
    parts = cuda_version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return f"cuda{parts[0]}{parts[1]}"


def _native_libraries(package_dir: Path) -> list[str]:
    return sorted(path.name for path in package_dir.glob("libbitsandbytes_*") if path.is_file())


def _dependency_resolution(path: Path) -> dict[str, Any]:
    if os.name == "nt":
        return {"method": "windows_loader", "resolved": None, "stderr_tail": None}
    try:
        result = _run(["ldd", str(path)], timeout=DIAGNOSTIC_TIMEOUT_SECONDS)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"method": "ldd", "resolved": False, "error": type(exc).__name__}
    lines = result.stdout.splitlines()
    unresolved = [line.strip() for line in lines if "not found" in line]
    return {
        "method": "ldd",
        "resolved": result.returncode == 0 and not unresolved,
        "unresolved": unresolved,
        "stdout_tail": _safe_tail(result.stdout, lines=80),
        "stderr_tail": _safe_tail(result.stderr),
    }


def _classify(*, expected_exists: bool, native_load: dict[str, Any], dependency: dict[str, Any], selected: str | None, backend_active: bool) -> str:
    if not expected_exists:
        return "BITSANDBYTES_CUDA_LIBRARY_MISSING"
    if not native_load.get("passed"):
        return "BITSANDBYTES_NATIVE_LIBRARY_LOAD_FAILED"
    if dependency.get("resolved") is False:
        return "BITSANDBYTES_CUDA_DEPENDENCY_MISSING"
    if not backend_active:
        return "BITSANDBYTES_CPU_FALLBACK_UNEXPLAINED"
    if not selected:
        return "BITSANDBYTES_CPU_FALLBACK_UNEXPLAINED"
    return "BITSANDBYTES_CUDA_BACKEND_READY"


def run_bnb_native_diagnose(*, output_root: Path, run_id: str, expected_git_commit: str | None, source_root: Path) -> dict[str, Any]:
    run_root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    ensure_kaggle_paths(run_root)
    resolved = resolve_executed_source_commit(run_root=run_root, repo_root=source_root, expected_git_commit=expected_git_commit)
    executed_commit = resolved.get("executed_source_commit")
    write_source_identity(run_root, run_id=run_id, expected_git_commit=expected_git_commit, executed_source_commit=executed_commit, source_identity_method=str(resolved.get("source_identity_method") or "unknown"), source_identity_verified=bool(resolved.get("source_identity_verified")))
    if expected_git_commit and executed_commit != expected_git_commit:
        raise RuntimeError("stale_kaggle_checkout")

    preinstall = _torch_state_probe()
    _write_json(run_root / "probe_torch_preinstall.json", preinstall)
    install = _run([sys.executable, "-c", _torch_install_snippet()], timeout=INSTALL_TIMEOUT_SECONDS)
    install_payload = {
        "requested_version": TORCH_CU118_VERSION,
        "requested_cuda_index": TORCH_CU118_INDEX_URL,
        "requested_packages": TORCH_CU118_PACKAGES,
        "returncode": install.returncode,
        "ok": install.returncode == 0,
        "stdout_tail": _safe_tail(install.stdout),
        "stderr_tail": _safe_tail(install.stderr),
        "torch_distribution": _version("torch"),
    }
    _write_json(run_root / "probe_torch_install.json", install_payload)
    _marker("TORCH_INSTALL_RESULT_JSON", install_payload)
    if install.returncode != 0:
        raise RuntimeError("torch_install_failed")

    runtime = _torch_state_probe()
    _write_json(run_root / "probe_torch_runtime.json", runtime)
    runtime_json = runtime.get("json") or {}
    _marker("TORCH_RUNTIME_RESULT_JSON", runtime_json)
    if not runtime.get("ok") or runtime_json.get("torch_version") != "2.5.1+cu118" or not runtime_json.get("sm_60_supported") or not runtime_json.get("basic_cuda_tensor_test"):
        raise RuntimeError("p100_torch_runtime_failed")

    install_bnb = _run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir", "--no-deps", f"bitsandbytes=={BNB_REQUESTED_VERSION}"], timeout=INSTALL_TIMEOUT_SECONDS)
    bnb_install = {
        "requested_version": BNB_REQUESTED_VERSION,
        "returncode": install_bnb.returncode,
        "ok": install_bnb.returncode == 0,
        "stdout_tail": _safe_tail(install_bnb.stdout),
        "stderr_tail": _safe_tail(install_bnb.stderr),
        "torch_distribution_after": _version("torch"),
        "bitsandbytes_distribution": _version("bitsandbytes"),
        "no_deps": True,
    }
    _write_json(run_root / "probe_bnb_install.json", bnb_install)
    _marker("BNB_INSTALL_RESULT_JSON", bnb_install)
    if install_bnb.returncode != 0:
        raise RuntimeError("bitsandbytes_install_failed")

    try:
        import bitsandbytes as bnb
        from bitsandbytes import cextension
        package_dir = Path(bnb.__file__).resolve().parent
        libraries = _native_libraries(package_dir)
        torch_cuda = runtime_json.get("torch_cuda_version")
        tag = _cuda_tag(torch_cuda)
        expected_name = f"libbitsandbytes_{tag}.so" if tag else None
        expected_path = package_dir / expected_name if expected_name else None
        selected_obj = getattr(cextension, "lib", None)
        selected = getattr(selected_obj, "_name", None) if selected_obj is not None else None
        import_payload = {
            "version": getattr(bnb, "__version__", None),
            "file": str(getattr(bnb, "__file__", "")),
            "package_directory": str(package_dir),
            "available_native_libraries": libraries,
            "torch_cuda_version": torch_cuda,
            "expected_cuda_library": expected_name,
            "expected_cuda_library_exists": bool(expected_path and expected_path.exists()),
            "selected_native_library": selected,
            "cuda_backend_active": bool(getattr(cextension, "lib", None) is not None and getattr(cextension, "COMPILED_WITH_CUDA", True)),
        }
    except Exception as exc:
        import_payload = {"ok": False, "exception_type": type(exc).__name__, "error": str(exc)[:500]}
        _write_json(run_root / "probe_bnb_import.json", import_payload)
        _marker("BNB_PACKAGE_RESULT_JSON", import_payload)
        raise RuntimeError("bitsandbytes_import_failed") from exc
    _write_json(run_root / "probe_bnb_import.json", import_payload)
    _marker("BNB_PACKAGE_RESULT_JSON", import_payload)
    _marker("BNB_IMPORT_RESULT_JSON", import_payload)

    expected_path = package_dir / str(import_payload.get("expected_cuda_library")) if import_payload.get("expected_cuda_library") else None
    native_load = {"passed": False, "path": str(expected_path) if expected_path else None, "error": None}
    if expected_path and expected_path.exists():
        try:
            ctypes.CDLL(str(expected_path))
            native_load["passed"] = True
        except OSError as exc:
            native_load["error"] = str(exc)[:1000]
    else:
        native_load["error"] = "expected_cuda_library_missing"
    _write_json(run_root / "probe_bnb_native_load.json", native_load)
    _marker("BNB_NATIVE_LOAD_JSON", native_load)
    dependency = _dependency_resolution(expected_path) if expected_path and expected_path.exists() else {"resolved": False, "reason": "expected_library_missing"}
    _write_json(run_root / "probe_bnb_cuda_dependency.json", dependency)
    _marker("BNB_CUDA_DEPENDENCY_JSON", dependency)
    _marker("BNB_CUDA_RESULT_JSON", {"native_load": native_load, "dependency": dependency, "backend_active": import_payload.get("cuda_backend_active")})
    classification = _classify(expected_exists=bool(import_payload.get("expected_cuda_library_exists")), native_load=native_load, dependency=dependency, selected=import_payload.get("selected_native_library"), backend_active=bool(import_payload.get("cuda_backend_active")))
    final = {
        "run_id": run_id,
        "expected_git_commit": expected_git_commit,
        "executed_git_commit": executed_commit,
        "torch": runtime_json,
        "bnb": import_payload,
        "bnb_install": bnb_install,
        "native_load": native_load,
        "cuda_dependency": dependency,
        "environment": {"BNB_CUDA_VERSION": os.environ.get("BNB_CUDA_VERSION"), "CUDA_HOME": os.environ.get("CUDA_HOME"), "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"), "python": platform.python_version()},
        "gpu": inspect_kaggle_gpu_identity(),
        "qwen_loaded": False,
        "peft_imported": False,
        "nf4_tested": False,
        "dataset_used": False,
        "classification": classification,
        "verdict": classification,
    }
    _write_json(run_root / "bnb_native_diagnostic_report.json", final)
    _marker("BNB_DIAGNOSTIC_RESULT_JSON", final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/kaggle/working")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-git-commit", default=None)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_bnb_native_diagnose(output_root=Path(args.output_root), run_id=args.run_id, expected_git_commit=args.expected_git_commit, source_root=Path(args.source_root))
        return 0 if report["verdict"] == "BITSANDBYTES_CUDA_BACKEND_READY" else 1
    except Exception as exc:
        print(json.dumps({"stage": "bnb_native_diagnose", "exception_type": type(exc).__name__, "sanitized_message": str(exc)[:500]}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
