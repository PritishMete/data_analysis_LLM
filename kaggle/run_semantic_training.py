from __future__ import annotations

import json
import os
import time
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from .import_trace import write_import_trace
from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_json, write_source_identity
from .bootstrap import (
    KAGGLE_WORKING_ROOT,
    build_kaggle_dependency_plan,
    build_artifact_manifest,
    build_semantic_kaggle_report,
    create_final_zip,
    detect_resume_checkpoint,
    inspect_kaggle_gpu_identity,
    discover_semantic_dataset,
    ensure_kaggle_paths,
    inspect_kaggle_environment,
    resolve_canonical_dataset_root,
    load_semantic_config,
    semantic_verdict,
    verify_attached_dataset,
    write_dependency_preflight_report,
    build_semantic_dataset_from_canonical,
)

COMPATIBILITY_REPORT: Any | None = None
RUNNER_METADATA_NAME = "runner_metadata.json"
PROBE_REPORT_DIRNAME = "reports"


def _load_semantic_rows(semantic_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        path = semantic_root / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _smoke_split_targets(targets: list[dict[str, Any]], train_limit: int = 100, validation_min: int = 5) -> dict[str, list[dict[str, Any]]]:
    usable = list(targets[: train_limit + validation_min])
    if len(usable) < validation_min:
        raise ValueError("smoke_validation_too_small")
    validation_size = max(validation_min, min(10, len(usable) // 10))
    if len(usable) - validation_size < 1:
        validation_size = max(validation_min, len(usable) - 1)
    validation = usable[:validation_size]
    train = usable[validation_size: min(len(usable), validation_size + train_limit)]
    if not train:
        raise ValueError("smoke_train_too_small")
    return {"train": train, "validation": validation, "test": []}


def _target_to_smoke_text(target: dict[str, Any]) -> str:
    payload = {
        "input": target.get("input") or {},
        "output": target.get("output") or {},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _runtime_memory_state(torch_module: Any | None) -> dict[str, float | None]:
    state = {"cuda_allocated_mb": None, "cuda_reserved_mb": None}
    if torch_module is None:
        return state
    try:
        if torch_module.cuda.is_available():
            state["cuda_allocated_mb"] = round(float(torch_module.cuda.memory_allocated() / (1024 * 1024)), 2)
            state["cuda_reserved_mb"] = round(float(torch_module.cuda.memory_reserved() / (1024 * 1024)), 2)
    except Exception:
        return state
    return state


def _write_jsonl_breadcrumb(path: Path, *, stage: str, success: bool, safe_message: str, torch_module: Any | None = None, run_id: str | None = None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = run_id or os.environ.get("KAGGLE_SMOKE_RUN_ID")
    payload = {
        "timestamp": time.time(),
        "stage": stage,
        "success": bool(success),
        "safe_message": safe_message,
        "run_id": run_id,
        **_runtime_memory_state(torch_module),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:
            pass
    return payload


def _sanitize_exception_message(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:1000]


def _safe_traceback_tail(exc: BaseException, *, limit: int = 25) -> str:
    tb = traceback.TracebackException.from_exception(exc)
    lines = list(tb.format())
    safe_lines = lines[-limit:]
    return "".join(safe_lines)


def _safe_tail(text: str | None, *, lines: int = 40) -> str | None:
    if not text:
        return None
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def _safe_commit_hash(repo_root: Path | None = None) -> str | None:
    explicit = os.environ.get("KAGGLE_EXECUTED_SOURCE_COMMIT") or os.environ.get("KAGGLE_SOURCE_COMMIT")
    if explicit and len(explicit.strip()) == 40:
        return explicit.strip()
    if repo_root is not None and (repo_root / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            )
            commit = (result.stdout or "").strip()
            return commit or None
        except Exception:
            return None
    return None


def _validate_archive_root(repo_root: Path) -> None:
    required_paths = [
        repo_root / "kaggle" / "bootstrap_environment.py",
        repo_root / "kaggle" / "execute_smoke_training.py",
        repo_root / "kaggle" / "run_semantic_training.py",
        repo_root / "src",
    ]
    if not repo_root.exists() or not repo_root.is_dir() or any(not path.exists() for path in required_paths):
        raise RuntimeError("ARCHIVE_ROOT_INVALID")


def _resolve_source_identity(*, run_root: Path, repo_root: Path, expected_git_commit: str | None) -> dict[str, Any]:
    resolved = resolve_executed_source_commit(run_root=run_root, repo_root=repo_root, expected_git_commit=expected_git_commit)
    executed = resolved.get("executed_source_commit")
    if not executed:
        raise RuntimeError("SOURCE_IDENTITY_MISSING")
    expected = expected_git_commit or _expected_commit_hash()
    if expected and executed != expected:
        raise RuntimeError("STALE_SOURCE_SNAPSHOT")
    resolved["source_identity_verified"] = True
    return resolved


def _run_command(args: list[str], *, timeout: int | None = None, check: bool = True, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
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


def _load_torch_compatibility_report() -> Any | None:
    try:
        from src.training.torch_compat import ensure_torch_dynamo_compatibility

        return ensure_torch_dynamo_compatibility()
    except Exception:
        return None


def _patch_torch_dynamo_compatibility(torch_module: Any | None) -> bool:
    if torch_module is None:
        return False
    try:
        dynamo_eval_frame = getattr(getattr(torch_module, "_C", None), "_dynamo", None)
        if dynamo_eval_frame is None:
            return False
        eval_frame = getattr(dynamo_eval_frame, "eval_frame", None)
        if eval_frame is None:
            return False
        if hasattr(eval_frame, "skip_code"):
            return False

        def _skip_code(*args: Any, **kwargs: Any) -> None:
            return None

        setattr(eval_frame, "skip_code", _skip_code)
        try:
            python_eval_frame = getattr(getattr(torch_module, "_dynamo", None), "eval_frame", None)
            if python_eval_frame is not None and not hasattr(python_eval_frame, "skip_code"):
                setattr(python_eval_frame, "skip_code", _skip_code)
            module_name = "torch._C._dynamo.eval_frame"
            module = sys.modules.get(module_name)
            if module is None:
                import types

                module = types.ModuleType(module_name)
                sys.modules[module_name] = module
            if not hasattr(module, "skip_code"):
                setattr(module, "skip_code", _skip_code)
        except Exception:
            pass
        return True
    except Exception:
        return False


def _dependency_probe_snippets() -> dict[str, str]:
    compat_snippet = """
import json
import pathlib
import sys
repo_root = pathlib.Path.cwd()
if not (repo_root / "src").exists() and (repo_root.parent / "src").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from src.training.torch_compat import ensure_torch_dynamo_compatibility
before = False
after = False
try:
    import torch
    dynamo_eval_frame = getattr(getattr(torch, "_C", None), "_dynamo", None)
    eval_frame = getattr(dynamo_eval_frame, "eval_frame", None) if dynamo_eval_frame is not None else None
    before = bool(eval_frame is not None and hasattr(eval_frame, "skip_code"))
except Exception:
    before = False
report = ensure_torch_dynamo_compatibility()
try:
    import torch
    dynamo_eval_frame = getattr(getattr(torch, "_C", None), "_dynamo", None)
    eval_frame = getattr(dynamo_eval_frame, "eval_frame", None) if dynamo_eval_frame is not None else None
    after = bool(eval_frame is not None and hasattr(eval_frame, "skip_code"))
except Exception:
    after = False
payload = {
    "torch_imported": report.torch_imported,
    "skip_code_present_before": report.skip_code_present_before if report.skip_code_present_before is not None else before,
    "skip_code_patch_applied": report.skip_code_patch_applied,
    "skip_code_present_after": after,
}
print(json.dumps(payload))
"""
    torch_import_snippet = """
import json
import pathlib
import sys
repo_root = pathlib.Path.cwd()
if not (repo_root / "src").exists() and (repo_root.parent / "src").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from src.training.torch_compat import ensure_torch_dynamo_compatibility
ensure_torch_dynamo_compatibility()
import torch
payload = {
    "version": torch.__version__,
    "cuda": getattr(torch.version, "cuda", None),
    "available": bool(torch.cuda.is_available()),
}
if payload["available"]:
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["capability"] = list(torch.cuda.get_device_capability(0))
print(json.dumps(payload))
"""
    torch_cuda_snippet = """
import json
import pathlib
import sys
repo_root = pathlib.Path.cwd()
if not (repo_root / "src").exists() and (repo_root.parent / "src").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from src.training.torch_compat import ensure_torch_dynamo_compatibility
ensure_torch_dynamo_compatibility()
import torch
payload = {
    "available": bool(torch.cuda.is_available()),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "arch_list": torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else None,
}
if payload["available"]:
    try:
        a = torch.ones((2, 2), device="cuda")
        b = torch.ones((2, 2), device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        payload["basic_cuda_tensor_test"] = bool(c.sum().item() == 8.0)
    except Exception as exc:
        payload["basic_cuda_tensor_test"] = False
        payload["tensor_error"] = str(exc)
print(json.dumps(payload))
"""
    bitsandbytes_snippet = """
import json
import pathlib
import sys
repo_root = pathlib.Path.cwd()
if not (repo_root / "src").exists() and (repo_root.parent / "src").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from src.training.torch_compat import ensure_torch_dynamo_compatibility
ensure_torch_dynamo_compatibility()
from bitsandbytes import cextension
import bitsandbytes as bnb
payload = {
    "version": bnb.__version__,
    "available_cuda_versions": None,
    "available_cuda_versions_status": "unsupported_by_version",
}
if hasattr(cextension, "get_available_cuda_binary_versions"):
    try:
        payload["available_cuda_versions"] = list(cextension.get_available_cuda_binary_versions())
        payload["available_cuda_versions_status"] = "reported"
    except Exception as exc:
        payload["available_cuda_versions_status"] = "probe_error"
        payload["available_cuda_versions_error"] = str(exc)[:500]
try:
    specs = cextension.get_cuda_specs()
    payload["cuda_specs"] = {
        "highest_compute_capability": list(specs.highest_compute_capability) if specs.highest_compute_capability is not None else None,
        "cuda_version_string": specs.cuda_version_string,
        "cuda_version_tuple": list(specs.cuda_version_tuple) if specs.cuda_version_tuple is not None else None,
    }
except Exception as exc:
    payload["cuda_specs_error"] = str(exc)
print(json.dumps(payload))
"""
    nf4_snippet = """
import json
import pathlib
import sys
repo_root = pathlib.Path.cwd()
if not (repo_root / "src").exists() and (repo_root.parent / "src").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from src.training.torch_compat import ensure_torch_dynamo_compatibility
ensure_torch_dynamo_compatibility()
from bitsandbytes import cextension
payload = {"nf4_capability_available": False}
try:
    specs = cextension.get_cuda_specs()
    highest = list(specs.highest_compute_capability) if specs.highest_compute_capability is not None else None
    payload["nf4_capability_available"] = bool(highest and tuple(highest) >= (6, 0))
    payload["cuda_specs"] = {
        "highest_compute_capability": highest,
        "cuda_version_string": specs.cuda_version_string,
        "cuda_version_tuple": list(specs.cuda_version_tuple) if specs.cuda_version_tuple is not None else None,
    }
except Exception as exc:
    payload["error"] = str(exc)
print(json.dumps(payload))
"""
    return {
        "compat": compat_snippet,
        "torch_import": torch_import_snippet,
        "torch_cuda": torch_cuda_snippet,
        "bitsandbytes": bitsandbytes_snippet,
        "nf4": nf4_snippet,
    }


def _run_python_probe(snippet: str, *, timeout: int, phase: str, label: str) -> dict[str, Any]:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [sys.executable, "-c", snippet],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path.cwd()),
        env=env,
    )
    timed_out = False
    stdout = ""
    stderr = ""
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    payload: dict[str, Any] = {
        "label": label,
        "phase": phase,
        "parent_pid": os.getpid(),
        "child_pid": proc.pid,
        "returncode": proc.returncode,
        "signal": None,
        "timed_out": timed_out,
        "stdout": (stdout or "").strip() or None,
        "stderr": (stderr or "").strip() or None,
        "ok": not timed_out and proc.returncode == 0,
    }
    if payload["stdout"]:
        try:
            payload["json"] = json.loads(str(payload["stdout"]))
        except Exception:
            payload["json"] = None
    return payload


def _classify_probe_result(*, label: str, probe: dict[str, Any]) -> str:
    if probe.get("ok"):
        return "OK"
    stderr = str(probe.get("stderr") or probe.get("stdout") or "").lower()
    if label == "compat":
        if "skip_code" in stderr:
            return "IMPORT_ORDER"
        return "BOOTSTRAP"
    if label in {"torch_import", "torch_cuda"}:
        return "PYTORCH_CUDA"
    if label == "bitsandbytes":
        if "ops.cu" in stderr:
            return "BITSANDBYTES"
        return "DEPENDENCY"
    if label == "nf4":
        return "NF4"
    return "DEPENDENCY"


def _write_probe_artifact(report_root: Path, *, probe_name: str, probe: dict[str, Any], classification: str) -> Path:
    report_dir = report_root / PROBE_REPORT_DIRNAME
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe_name": probe_name,
        "success": bool(probe.get("ok")),
        "parent_pid": probe.get("parent_pid"),
        "child_pid": probe.get("child_pid"),
        "return_code": probe.get("returncode"),
        "signal": probe.get("signal"),
        "timed_out": bool(probe.get("timed_out")),
        "stdout_tail": _safe_tail(probe.get("stdout"), lines=40),
        "stderr_tail": _safe_tail(probe.get("stderr"), lines=40),
        "classification": classification,
    }
    if probe.get("json") is not None:
        payload["json"] = probe["json"]
    path = report_dir / f"{probe_name}.json"
    return _write_json(path, payload)


def _run_dependency_compatibility_preflight(*, report_root: Path, breadcrumbs_path: Path) -> dict[str, Any]:
    _write_smoke_heartbeat(report_root, stage="dependency_compatibility_preflight")
    _emit_smoke_stage(breadcrumbs_path, stage="dependency_compatibility_preflight_started", success=True, safe_message="preflight start")
    gpu_identity = inspect_kaggle_gpu_identity()
    probes = _dependency_probe_snippets()
    compat_probe = _run_python_probe(probes["compat"], timeout=60, phase="dependency_compatibility_preflight", label="compat")
    torch_import_probe = _run_python_probe(probes["torch_import"], timeout=60, phase="dependency_compatibility_preflight", label="torch_import")
    torch_cuda_probe = _run_python_probe(probes["torch_cuda"], timeout=60, phase="dependency_compatibility_preflight", label="torch_cuda")
    bnb_probe = _run_python_probe(probes["bitsandbytes"], timeout=60, phase="dependency_compatibility_preflight", label="bitsandbytes")
    nf4_probe = _run_python_probe(probes["nf4"], timeout=60, phase="dependency_compatibility_preflight", label="nf4")
    _write_probe_artifact(report_root, probe_name="probe_compat_shim", probe=compat_probe, classification=_classify_probe_result(label="compat", probe=compat_probe))
    _write_probe_artifact(report_root, probe_name="probe_torch_import_runtime", probe=torch_import_probe, classification=_classify_probe_result(label="torch_import", probe=torch_import_probe))
    _write_probe_artifact(report_root, probe_name="probe_torch_cuda_runtime", probe=torch_cuda_probe, classification=_classify_probe_result(label="torch_cuda", probe=torch_cuda_probe))
    _write_probe_artifact(report_root, probe_name="probe_bitsandbytes_runtime", probe=bnb_probe, classification=_classify_probe_result(label="bitsandbytes", probe=bnb_probe))
    _write_probe_artifact(report_root, probe_name="probe_nf4_runtime", probe=nf4_probe, classification=_classify_probe_result(label="nf4", probe=nf4_probe))
    preflight = build_kaggle_dependency_plan(gpu_identity=gpu_identity, torch_probe=torch_cuda_probe, bitsandbytes_probe=bnb_probe)
    write_dependency_preflight_report(report_root, preflight)
    _emit_smoke_stage(
        breadcrumbs_path,
        stage="dependency_compatibility_preflight_complete",
        success=bool(preflight.compatibility_passed or preflight.install_plan.get("pip_groups")),
        safe_message=preflight.reason or "dependency plan ready",
    )
    return {
        "gpu_identity": gpu_identity,
        "compat_probe": compat_probe,
        "torch_import_probe": torch_import_probe,
        "torch_cuda_probe": torch_cuda_probe,
        "bitsandbytes_probe": bnb_probe,
        "nf4_probe": nf4_probe,
        "preflight": preflight.to_dict(),
    }


def _safe_probe_result(result: subprocess.CompletedProcess[str], *, label: str) -> dict[str, Any]:
    payload = {
        "label": label,
        "returncode": result.returncode,
        "stdout": (result.stdout or "").strip() or None,
        "stderr": (result.stderr or "").strip() or None,
        "ok": result.returncode == 0,
    }
    if payload["stdout"]:
        try:
            payload["json"] = json.loads(str(payload["stdout"]))
        except Exception:
            payload["json"] = None
    return payload


def _expected_commit_hash() -> str | None:
    return os.environ.get("KAGGLE_EXPECTED_GIT_COMMIT") or os.environ.get("EXPECTED_GIT_COMMIT")


def _write_smoke_heartbeat(report_root: Path, *, stage: str, smoke_mode: bool = True, run_id: str | None = None, expected_git_commit: str | None = None, executed_git_commit: str | None = None, extra: dict[str, Any] | None = None) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    run_id = run_id or os.environ.get("KAGGLE_SMOKE_RUN_ID")
    if executed_git_commit is None:
        executed_git_commit = (
            os.environ.get("KAGGLE_EXECUTED_SOURCE_COMMIT")
            or os.environ.get("KAGGLE_SOURCE_COMMIT")
            or expected_git_commit
            or _safe_commit_hash()
        )
    heartbeat_commit = executed_git_commit or expected_git_commit or os.environ.get("KAGGLE_EXPECTED_GIT_COMMIT") or _safe_commit_hash()
    payload = {
        "stage": stage,
        "timestamp": time.time(),
        "git_commit": heartbeat_commit,
        "smoke_mode": smoke_mode,
        "run_id": run_id,
        "expected_git_commit": expected_git_commit or _expected_commit_hash(),
        "executed_git_commit": executed_git_commit or heartbeat_commit,
    }
    if extra:
        payload.update(extra)
    path = report_root / "smoke_heartbeat.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_import_preflight(
    report_root: Path,
    *,
    torch_imported: bool,
    compatibility_bootstrap_ran: bool,
    skip_code_present_before: bool | None,
    skip_code_patch_applied: bool,
    transformers_import_attempted: bool,
    transformers_import_succeeded: bool,
    git_commit: str | None = None,
    error: BaseException | None = None,
) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "git_commit": git_commit or _safe_commit_hash(),
        "torch_imported": torch_imported,
        "compatibility_bootstrap_ran": compatibility_bootstrap_ran,
        "skip_code_present_before": skip_code_present_before,
        "skip_code_patch_applied": skip_code_patch_applied,
        "transformers_import_attempted": transformers_import_attempted,
        "transformers_import_succeeded": transformers_import_succeeded,
    }
    if error is not None:
        payload["sanitized_exception"] = _sanitize_exception_message(error)
        payload["exception_type"] = type(error).__name__
    path = report_root / "import_preflight.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_smoke_failure(
    *,
    report_root: Path,
    stage: str,
    exc: BaseException,
    torch_module: Any | None = None,
    run_id: str | None = None,
) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    run_id = run_id or os.environ.get("KAGGLE_SMOKE_RUN_ID")
    failure_path = report_root / "smoke_failure.json"
    try:
        import importlib.metadata as metadata

        package_versions = {
            "torch": metadata.version("torch") if torch_module is not None else None,
            "transformers": metadata.version("transformers"),
            "peft": metadata.version("peft"),
            "bitsandbytes": metadata.version("bitsandbytes"),
        }
    except Exception:
        package_versions = {"torch": None, "transformers": None, "peft": None, "bitsandbytes": None}
    payload = {
        "stage": stage,
        "run_id": run_id,
        "exception_type": type(exc).__name__,
        "sanitized_exception_message": _sanitize_exception_message(exc),
        "traceback_tail": _safe_traceback_tail(exc),
        "cuda_memory_state": _runtime_memory_state(torch_module),
        "package_versions": package_versions,
        "gpu_name": None,
        "python_version": sys.version,
        "torch_version": getattr(torch_module, "__version__", None) if torch_module is not None else None,
        "transformers_version": package_versions.get("transformers"),
        "peft_version": package_versions.get("peft"),
        "bitsandbytes_version": package_versions.get("bitsandbytes"),
    }
    try:
        if torch_module is not None and torch_module.cuda.is_available():
            payload["gpu_name"] = torch_module.cuda.get_device_name(0)
    except Exception:
        payload["gpu_name"] = None
    failure_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        with failure_path.open("a", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        pass
    return failure_path


def _build_smoke_corpus(semantic_root: Path, output_root: Path) -> dict[str, Any]:
    targets = _load_semantic_rows(semantic_root)
    split_data = _smoke_split_targets(targets)
    smoke_root = output_root / "smoke_training"
    smoke_root.mkdir(parents=True, exist_ok=True)
    for split, rows in split_data.items():
        path = smoke_root / f"{split}.jsonl"
        lines = [json.dumps({"text": _target_to_smoke_text(row)}, sort_keys=True, separators=(",", ":")) for row in rows]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    report = {
        "smoke_dataset_root": str(smoke_root),
        "train_count": len(split_data["train"]),
        "validation_count": len(split_data["validation"]),
        "test_count": len(split_data["test"]),
        "train_limit": 100,
        "validation_min": 5,
    }
    (smoke_root / "smoke_dataset_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"root": smoke_root, "report": report, "splits": split_data}


def _stage_guard(
    *,
    stage: str,
    report_root: Path,
    breadcrumbs_path: Path,
    torch_module: Any | None = None,
    smoke_mode: bool = True,
    success: bool = True,
    safe_message: str = "",
    run_id: str | None = None,
    expected_git_commit: str | None = None,
    executed_git_commit: str | None = None,
) -> None:
    _write_smoke_heartbeat(report_root, stage=stage, smoke_mode=smoke_mode, run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit)
    _emit_smoke_stage(breadcrumbs_path, stage=stage, success=success, safe_message=safe_message, torch_module=torch_module, run_id=run_id)


def _ensure_runtime_packages(*, preflight: dict[str, Any] | None = None) -> dict[str, str | None]:
    installed: dict[str, str | None] = {}
    if os.environ.get("KAGGLE_SKIP_DEP_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
        try:
            from importlib.metadata import version

            installed["torch"] = version("torch")
            installed["bitsandbytes"] = version("bitsandbytes")
            installed["transformers"] = version("transformers")
            installed["accelerate"] = version("accelerate")
            installed["peft"] = version("peft")
        except Exception:
            pass
        return installed
    preflight = preflight or {}
    installed_plan = (preflight.get("preflight") or {}).get("install_plan") or {}
    pip_groups = installed_plan.get("pip_groups") or []
    if pip_groups:
        for group in pip_groups:
            packages = group.get("packages") or []
            if not packages:
                continue
            install_args = [sys.executable, "-m", "pip", "install"]
            if group.get("upgrade"):
                install_args.append("--upgrade")
            if group.get("find_links"):
                for link in group.get("find_links") or []:
                    install_args.extend(["--find-links", str(link)])
            if group.get("index_url"):
                install_args.extend(["--index-url", str(group.get("index_url"))])
            if group.get("extra_index_url"):
                for url in group.get("extra_index_url") or []:
                    install_args.extend(["--extra-index-url", str(url)])
            install_args.extend(packages)
            subprocess.run(install_args, check=True, timeout=300)
    try:
        from importlib.metadata import version

        installed["torch"] = version("torch")
        installed["bitsandbytes"] = version("bitsandbytes")
        installed["transformers"] = version("transformers")
        installed["accelerate"] = version("accelerate")
        installed["peft"] = version("peft")
    except Exception:
        pass
    return installed


def _emit_smoke_stage(
    breadcrumbs_path: Path,
    *,
    stage: str,
    success: bool,
    safe_message: str,
    torch_module: Any | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return _write_jsonl_breadcrumb(breadcrumbs_path, stage=stage, success=success, safe_message=safe_message, torch_module=torch_module, run_id=run_id)


def _run_real_smoke_training(
    *,
    base_model: str,
    smoke_root: Path,
    output_root: Path,
    breadcrumbs_path: Path,
    run_id: str | None = None,
    expected_git_commit: str | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    report_root = output_root
    compatibility_report = _load_torch_compatibility_report()
    trace_path = report_root / "import_trace.jsonl"
    write_import_trace(trace_path, module="kaggle.run_semantic_training", event="before_torch_import")

    def _run_with_timeout(cmd: list[str], *, timeout: int, stage: str) -> None:
        try:
            subprocess.run(cmd, check=True, timeout=timeout)
        except Exception as exc:
            _write_smoke_failure(report_root=report_root, stage=stage, exc=exc, torch_module=None, run_id=run_id)
            raise

    def _ensure_runtime_packages(*, preflight: dict[str, Any] | None = None) -> dict[str, str | None]:
        installed: dict[str, str | None] = {}
        if os.environ.get("KAGGLE_SKIP_DEP_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
            try:
                from importlib.metadata import version

                installed["torch"] = version("torch")
                installed["bitsandbytes"] = version("bitsandbytes")
                installed["transformers"] = version("transformers")
                installed["accelerate"] = version("accelerate")
                installed["peft"] = version("peft")
            except Exception:
                pass
            return installed
        preflight = preflight or {}
        installed_plan = (preflight.get("preflight") or {}).get("install_plan") or {}
        pip_groups = installed_plan.get("pip_groups") or []
        if pip_groups:
            for group in pip_groups:
                packages = list(group.get("packages") or [])
                if not packages:
                    continue
                command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir"]
                index_url = group.get("index_url")
                if index_url:
                    command.extend(["--index-url", str(index_url)])
                command.extend(packages)
                _run_with_timeout(command, timeout=900, stage="dependencies_started")
        try:
            import torch

            cuda_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
            installed["torch"] = torch.__version__
            installed["cuda_capability"] = f"{cuda_capability[0]}.{cuda_capability[1]}" if cuda_capability else None
            installed["torch_cuda"] = getattr(torch.version, "cuda", None)
        except Exception:
            installed["torch"] = None
            installed["cuda_capability"] = None
            installed["torch_cuda"] = None
        try:
            from importlib.metadata import version

            installed["bitsandbytes"] = version("bitsandbytes")
        except Exception:
            installed["bitsandbytes"] = None

        required_packages = []
        if installed["bitsandbytes"] is None or tuple(int(part) for part in str(installed["bitsandbytes"]).split(".")[:2] if part.isdigit()) < (0, 43):
            required_packages.append("bitsandbytes==0.43.3")
        for package in ("accelerate>=0.31", "peft>=0.11", "transformers>=4.43", "trl>=0.9", "safetensors>=0.4", "sentencepiece>=0.2.0"):
            required_packages.append(package)

        if required_packages:
            _run_with_timeout(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir", *required_packages],
                timeout=600,
                stage="dependencies_started",
            )
        return installed

    _write_smoke_heartbeat(report_root, stage="notebook_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=_safe_commit_hash())
    _stage_guard(stage="repo_checkout_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, safe_message="repo checkout start")
    _stage_guard(stage="repo_checkout_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, safe_message="repo checkout complete")
    _stage_guard(stage="dependencies_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, safe_message="starting smoke bootstrap")
    runtime_packages = _ensure_runtime_packages()
    torch = None
    try:
        _stage_guard(stage="dependencies_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, safe_message="runtime packages checked")
        import torch as torch_module

        torch = torch_module
        write_import_trace(trace_path, module="kaggle.run_semantic_training", event="after_torch_import")
        write_import_trace(trace_path, module="kaggle.run_semantic_training", event="before_compat_patch", compatibility_patch_ran=False)
        _patch_torch_dynamo_compatibility(torch)
        write_import_trace(trace_path, module="kaggle.run_semantic_training", event="after_compat_patch", compatibility_patch_ran=True)
        _stage_guard(stage="gpu_check_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message="gpu check start")
        if torch.cuda.is_available():
            _stage_guard(stage="gpu_check_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=torch.cuda.get_device_name(0))
        else:
            raise RuntimeError("cuda_unavailable")
    except Exception as exc:
        _write_smoke_failure(report_root=report_root, stage="dependencies_complete", exc=exc, torch_module=torch)
        raise

    try:
        _write_import_preflight(
            report_root,
            torch_imported=torch is not None,
            compatibility_bootstrap_ran=bool(
                getattr(compatibility_report, "skip_code_patch_applied", False)
                or getattr(compatibility_report, "skip_code_present_before", None) is not None
            ),
            skip_code_present_before=getattr(compatibility_report, "skip_code_present_before", None),
            skip_code_patch_applied=bool(getattr(compatibility_report, "skip_code_patch_applied", False)),
            transformers_import_attempted=False,
            transformers_import_succeeded=False,
        )
        write_import_trace(trace_path, module="kaggle.run_semantic_training", event="before_peft_import", compatibility_patch_ran=True)
        _patch_torch_dynamo_compatibility(torch)
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        write_import_trace(trace_path, module="kaggle.run_semantic_training", event="after_peft_import", compatibility_patch_ran=True)
        _write_import_preflight(
            report_root,
            torch_imported=torch is not None,
            compatibility_bootstrap_ran=True,
            skip_code_present_before=getattr(compatibility_report, "skip_code_present_before", None),
            skip_code_patch_applied=bool(getattr(compatibility_report, "skip_code_patch_applied", False)),
            transformers_import_attempted=True,
            transformers_import_succeeded=True,
        )
        write_import_trace(trace_path, module="kaggle.run_semantic_training", event="before_transformers_import", compatibility_patch_ran=True)
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        write_import_trace(trace_path, module="kaggle.run_semantic_training", event="after_transformers_import", compatibility_patch_ran=True)
    except Exception as exc:
        _write_import_preflight(
            report_root,
            torch_imported=torch is not None,
            compatibility_bootstrap_ran=True,
            skip_code_present_before=getattr(compatibility_report, "skip_code_present_before", None),
            skip_code_patch_applied=bool(getattr(compatibility_report, "skip_code_patch_applied", False)),
            transformers_import_attempted=True,
            transformers_import_succeeded=False,
            error=exc,
        )
        _write_smoke_failure(report_root=report_root, stage="dependencies_complete", exc=exc, torch_module=torch)
        raise

    train_path = smoke_root / "train.jsonl"
    val_path = smoke_root / "validation.jsonl"
    if not train_path.exists() or not val_path.exists():
        exc = FileNotFoundError("smoke_dataset_missing")
        _write_smoke_failure(report_root=report_root, stage="dataset_started", exc=exc, torch_module=torch)
        raise exc
    _stage_guard(stage="dataset_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message="dataset start")
    _stage_guard(stage="dataset_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=f"train={train_path.name};validation={val_path.name}")

    try:
        _stage_guard(stage="tokenizer_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        _stage_guard(stage="tokenizer_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message="tokenizer ready")
    except Exception as exc:
        _write_smoke_failure(report_root=report_root, stage="tokenizer_started", exc=exc, torch_module=torch)
        raise

    compute_dtype = torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    try:
        _stage_guard(stage="model_download_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=base_model)
        _stage_guard(stage="model_download_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=base_model)
        _stage_guard(stage="model_load_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        _stage_guard(stage="model_load_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message="model ready")
    except Exception as exc:
        _write_smoke_failure(report_root=report_root, stage="model_load_started", exc=exc, torch_module=torch)
        raise

    try:
        _stage_guard(stage="quantization_verified", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message="load_in_4bit_nf4")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        _stage_guard(stage="lora_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message="attaching peft lora")
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.train()
        model.gradient_checkpointing_enable()
        _stage_guard(stage="lora_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message="lora ready")
    except Exception as exc:
        _write_smoke_failure(report_root=report_root, stage="lora_started", exc=exc, torch_module=torch)
        raise

    def _load_rows(path: Path) -> list[str]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(str(row["text"]))
        return rows

    train_texts = _load_rows(train_path)
    val_texts = _load_rows(val_path)

    def _encode(text: str, *, max_length: int = 256) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
        )
        encoded["labels"] = encoded["input_ids"].clone()
        return encoded

    train_encoded = [_encode(text) for text in train_texts]
    val_encoded = [_encode(text) for text in val_texts]
    smoke_steps = min(3, len(train_encoded))
    output_dir = output_root / "checkpoints" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    device = next(model.parameters()).device
    start = time.perf_counter()
    train_losses: list[float] = []
    peak_vram_mb = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step in range(smoke_steps):
        step_stage = f"training_step_{step + 1}"
        _stage_guard(stage=f"{step_stage}_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=f"step={step + 1}")
        batch = train_encoded[step % len(train_encoded)]
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None:
            exc = RuntimeError("smoke_training_missing_loss")
            _write_smoke_failure(report_root=report_root, stage=step_stage, exc=exc, torch_module=torch)
            raise exc
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(float(loss.detach().cpu().item()))
        if torch.cuda.is_available():
            peak_vram_mb = max(peak_vram_mb, float(torch.cuda.max_memory_allocated() / (1024 * 1024)))
        _stage_guard(stage=f"{step_stage}_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=f"loss={train_losses[-1]:.6f}")
    duration = time.perf_counter() - start
    torch.cuda.empty_cache()
    model.eval()
    eval_losses: list[float] = []
    with torch.no_grad():
        for batch in val_encoded:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            if outputs.loss is not None:
                eval_losses.append(float(outputs.loss.detach().cpu().item()))
    eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else None
    checkpoint_dir = output_dir / f"checkpoint-{smoke_steps}"
    checkpoint_created = checkpoint_dir.exists()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _stage_guard(stage="checkpoint_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=str(checkpoint_dir))
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    checkpoint_created = any(checkpoint_dir.iterdir())

    adapter_dir = output_root / "adapters" / "smoke"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    _stage_guard(stage="adapter_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=str(adapter_dir))
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_created = any(adapter_dir.iterdir())

    smoke_training_report = {
        "model_loaded": True,
        "quantization_4bit": True,
        "lora_attached": True,
        "cuda_training_steps": int(smoke_steps),
        "checkpoint_created": bool(checkpoint_created),
        "adapter_created": bool(adapter_created),
        "training_duration_seconds": round(duration, 2),
        "train_rows": len(train_encoded),
        "validation_rows": len(val_encoded),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "metrics": {
            "train_loss": train_losses[-1] if train_losses else None,
            "eval_loss": eval_loss,
        },
        "quantization": {
            "backend": "bitsandbytes",
            "bnb_4bit_quant_type": "nf4",
            "compute_dtype": "fp16",
        },
        "runtime_packages": runtime_packages,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
    }
    _write_smoke_heartbeat(report_root, stage="smoke_complete")
    _emit_smoke_stage(breadcrumbs_path, stage="smoke_complete", success=True, safe_message="smoke training finished", torch_module=torch)
    return {
        "smoke_training_report": smoke_training_report,
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_dir": str(adapter_dir),
        "train_steps": smoke_steps,
        "train_metrics": {"train_loss": smoke_training_report["metrics"]["train_loss"]},
        "validation_metrics": {"eval_loss": smoke_training_report["metrics"]["eval_loss"]},
    }


def _failed_smoke_training_report(
    *,
    error: Exception,
    runtime_packages: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    return {
        "model_loaded": False,
        "quantization_4bit": False,
        "lora_attached": False,
        "cuda_training_steps": 0,
        "checkpoint_created": False,
        "adapter_created": False,
        "training_duration_seconds": 0.0,
        "train_rows": 0,
        "validation_rows": 0,
        "metrics": {
            "train_loss": None,
            "eval_loss": None,
        },
        "failure": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "runtime_packages": runtime_packages or {},
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_root(output_root: Path, run_id: str | None) -> Path:
    if run_id:
        return ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    resolved_run_id = resolve_current_run_id(run_id, base_root=output_root / "smoke_runs")
    if resolved_run_id:
        return ensure_run_root(resolved_run_id, base_root=output_root / "smoke_runs")
    generated = generate_run_id(git_commit=_safe_commit_hash())
    return ensure_run_root(generated, base_root=output_root / "smoke_runs")


def _safe_runtime_report(*, dataset_dir: Path, output_root: Path) -> dict[str, Any]:
    env = inspect_kaggle_environment().to_dict()
    paths = ensure_kaggle_paths(output_root)
    config = load_semantic_config()
    verification = verify_attached_dataset(dataset_dir)
    resume_checkpoint = detect_resume_checkpoint(paths.checkpoints)
    report = build_semantic_kaggle_report(dataset_dir=dataset_dir, output_root=output_root)
    return {
        "environment": env,
        "paths": paths.to_dict(),
        "config": config,
        "dataset_dir": str(dataset_dir),
        "dataset_verification": verification,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "semantic_report": report,
    }


def build_training_plan(*, dataset_dir: Path, output_root: Path = KAGGLE_WORKING_ROOT, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> dict[str, Any]:
    paths = ensure_kaggle_paths(output_root)
    config = load_semantic_config()
    return {
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "base_model": base_model,
        "config_path": "configs/qwen25_0_5b_semantic_qlora.yaml",
        "resume_checkpoint": str(detect_resume_checkpoint(paths.checkpoints)) if detect_resume_checkpoint(paths.checkpoints) else None,
        "steps": [
            "inspect_environment",
            "verify_cuda_gpu",
            "install_missing_training_dependencies",
            "discover_semantic_dataset",
            "verify_sha256_manifest",
            "validate_dataset_readiness",
            "run_untouched_base_model_semantic_benchmark",
            "run_semantic_qlora_training",
            "evaluate_validation",
            "select_best_checkpoint",
            "freeze_checkpoint_choice",
            "run_holdout_test_once",
            "run_ood_fallback_evaluation",
            "apply_semantic_success_gates",
            "export_lora_adapter",
            "generate_artifact_manifest",
            "create_final_zip",
        ],
        "config": config,
        "paths": paths.to_dict(),
    }


def run_notebook_flow(
    *,
    output_root: Path = KAGGLE_WORKING_ROOT,
    resume_from: str | None = None,
    run_id: str | None = None,
    expected_git_commit: str | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    expected_commit = expected_git_commit or _expected_commit_hash()
    resolved_run_id = run_id or os.environ.get("KAGGLE_SMOKE_RUN_ID") or generate_run_id(git_commit=expected_commit)
    os.environ["KAGGLE_SMOKE_RUN_ID"] = resolved_run_id
    if expected_commit:
        os.environ["KAGGLE_EXPECTED_GIT_COMMIT"] = expected_commit
    run_root = _run_root(output_root, resolved_run_id)
    paths = ensure_kaggle_paths(run_root)
    repo_root = Path(source_root) if source_root is not None else (output_root / "data_analysis_LLM")
    notebook_started_path = write_json(
        run_root / "notebook_started.json",
        {
            "run_id": resolved_run_id,
            "expected_git_commit": expected_commit,
            "timestamp": time.time(),
            "pid": os.getpid(),
            "python_version": sys.version,
            "smoke_mode": True,
        },
    )
    _write_json(
        run_root / "archive_extracted.json",
        {
            "run_id": resolved_run_id,
            "timestamp": time.time(),
            "repo_root": str(repo_root),
        },
    )
    _validate_archive_root(repo_root)
    resolved_source = _resolve_source_identity(run_root=run_root, repo_root=repo_root, expected_git_commit=expected_commit)
    executed_commit = str(resolved_source["executed_source_commit"])
    _write_smoke_heartbeat(run_root, stage="source_identity_resolved", run_id=resolved_run_id, expected_git_commit=expected_commit, executed_git_commit=executed_commit)
    _write_json(
        run_root / "source_identity.json",
        {
            "run_id": resolved_run_id,
            "expected_git_commit": expected_commit,
            "executed_source_commit": executed_commit,
            "source_identity_method": resolved_source["source_identity_method"],
            "source_identity_verified": bool(resolved_source["source_identity_verified"]),
            "timestamp": time.time(),
        },
    )
    write_source_identity(
        run_root,
        run_id=resolved_run_id,
        expected_git_commit=expected_commit,
        executed_source_commit=executed_commit,
        source_identity_method=str(resolved_source["source_identity_method"]),
        source_identity_verified=bool(resolved_source["source_identity_verified"]),
    )
    _write_smoke_heartbeat(run_root, stage="source_identity_verified", run_id=resolved_run_id, expected_git_commit=expected_commit, executed_git_commit=executed_commit)
    _write_json(
        run_root / RUNNER_METADATA_NAME,
        {
            "run_id": resolved_run_id,
            "expected_git_commit": expected_commit,
            "executed_git_commit": executed_commit,
            "timestamp": time.time(),
            "pid": os.getpid(),
            "notebook_started_path": str(notebook_started_path),
        },
    )
    _write_smoke_heartbeat(run_root, stage="bootstrap_script_started", run_id=resolved_run_id, expected_git_commit=expected_commit, executed_git_commit=executed_commit)
    resolved = resolve_canonical_dataset_root()
    dataset_dir = Path(resolved["root"]) if resolved.get("root") else None
    if dataset_dir is None:
        return {
            "verdict": "KAGGLE_GPU_NOT_AVAILABLE",
            "reason": resolved.get("reason") or "no_attached_dataset_found",
            "canonical_dataset_root": None,
            "paths": paths.to_dict(),
            "run_id": resolved_run_id,
        }
    canonical_verification = verify_attached_dataset(dataset_dir)
    if not canonical_verification.get("verified"):
        return {
            "verdict": "TRAINING_FAILED",
            "reason": "canonical_dataset_verification_failed",
            "canonical_dataset_root": str(dataset_dir),
            "dataset_verification": canonical_verification,
            "paths": paths.to_dict(),
            "run_id": resolved_run_id,
        }
    breadcrumbs_path = run_root / "smoke_breadcrumbs.jsonl"
    _emit_smoke_stage(breadcrumbs_path, stage="notebook_started", success=True, safe_message="notebook started", run_id=resolved_run_id)
    _emit_smoke_stage(breadcrumbs_path, stage="archive_extracted", success=True, safe_message="archive extracted", run_id=resolved_run_id)
    _emit_smoke_stage(breadcrumbs_path, stage="source_identity_resolved", success=True, safe_message="source identity resolved", run_id=resolved_run_id)
    _emit_smoke_stage(breadcrumbs_path, stage="source_identity_verified", success=True, safe_message="source identity verified", run_id=resolved_run_id)
    _emit_smoke_stage(breadcrumbs_path, stage="bootstrap_script_started", success=True, safe_message="bootstrap start", run_id=resolved_run_id)
    dependency_preflight = _run_dependency_compatibility_preflight(report_root=run_root, breadcrumbs_path=breadcrumbs_path)
    _stage_guard(stage="dependencies_started", report_root=run_root, breadcrumbs_path=breadcrumbs_path, safe_message="starting smoke bootstrap", run_id=resolved_run_id, expected_git_commit=expected_commit, executed_git_commit=executed_commit)
    runtime_packages = _ensure_runtime_packages(preflight=dependency_preflight)
    _stage_guard(stage="dependencies_complete", report_root=run_root, breadcrumbs_path=breadcrumbs_path, safe_message="runtime packages checked", run_id=resolved_run_id, expected_git_commit=expected_commit, executed_git_commit=executed_commit)
    post_install_preflight = _run_dependency_compatibility_preflight(report_root=run_root, breadcrumbs_path=breadcrumbs_path)
    if not post_install_preflight["preflight"].get("compatibility_passed"):
        failure_reason = post_install_preflight["preflight"].get("reason") or "dependency_preflight_failed"
        exc = RuntimeError(failure_reason)
        _write_smoke_failure(report_root=run_root, stage="dependency_compatibility_preflight", exc=exc, torch_module=None, run_id=resolved_run_id)
        raise exc
    semantic_data = build_semantic_dataset_from_canonical(dataset_dir, run_root / "semantic_training")
    resume_checkpoint = detect_resume_checkpoint(paths.checkpoints, resume_from=resume_from)
    runtime = _safe_runtime_report(dataset_dir=dataset_dir, output_root=run_root)
    training_plan = build_training_plan(dataset_dir=Path(semantic_data["semantic_output_root"]), output_root=run_root)
    smoke_corpus = _build_smoke_corpus(Path(semantic_data["semantic_output_root"]), run_root)
    smoke_failure: Exception | None = None
    try:
        smoke_training = _run_real_smoke_training(
            base_model=str(training_plan["base_model"]),
            smoke_root=Path(smoke_corpus["root"]),
            output_root=run_root,
            breadcrumbs_path=breadcrumbs_path,
            run_id=resolved_run_id,
            expected_git_commit=expected_commit,
            resume_from=resume_from,
        )
    except Exception as exc:  # pragma: no cover - surfaced via Kaggle notebook logs
        smoke_failure = exc
        try:
            import torch as torch_module
        except Exception:
            torch_module = None
        _write_smoke_failure(report_root=run_root, stage="smoke_notebook", exc=exc, torch_module=torch_module, run_id=resolved_run_id)
        smoke_training = {
            "smoke_training_report": _failed_smoke_training_report(error=exc),
            "checkpoint_dir": None,
            "adapter_dir": None,
            "train_steps": 0,
            "train_metrics": {"train_loss": None},
            "validation_metrics": {"eval_loss": None},
        }
    safe_report_path = run_root / "final_report.json"
    safe_metrics_path = run_root / "semantic_metrics.json"
    smoke_report_path = run_root / "smoke_training_report.json"
    artifact_manifest_path = run_root / "artifact_manifest.json"
    _write_json(
        safe_report_path,
        {
            "run_id": resolved_run_id,
            "runtime": runtime,
            "dependency_preflight": dependency_preflight,
            "dependency_post_install_preflight": post_install_preflight,
            "training_plan": training_plan,
            "canonical_dataset_root": str(dataset_dir),
            "semantic_dataset_root": semantic_data["semantic_output_root"],
            "canonical_row_counts": semantic_data["bundle_report"].get("train_count", 0) + semantic_data["bundle_report"].get("validation_count", 0) + semantic_data["bundle_report"].get("test_count", 0),
            "semantic_row_count": semantic_data["semantic_row_count"],
            "smoke_training": smoke_training,
            "smoke_training_failure": type(smoke_failure).__name__ if smoke_failure else None,
        },
    )
    _write_json(safe_metrics_path, semantic_data["readiness"])
    _write_json(smoke_report_path, smoke_training["smoke_training_report"])
    artifact_manifest = build_artifact_manifest([safe_report_path, safe_metrics_path, smoke_report_path])
    _write_json(artifact_manifest_path, artifact_manifest)
    final_zip = create_final_zip(run_root, [safe_report_path, safe_metrics_path, smoke_report_path, artifact_manifest_path, breadcrumbs_path, run_root / "smoke_failure.json"], zip_name="semantic_extractor_artifacts.zip")
    result = {
        "run_id": resolved_run_id,
        "environment": runtime["environment"],
        "paths": paths.to_dict(),
        "executed_git_commit": executed_commit,
        "expected_git_commit": expected_commit,
        "canonical_dataset_root": str(dataset_dir),
        "semantic_dataset_root": semantic_data["semantic_output_root"],
        "dependency_preflight": dependency_preflight,
        "dependency_post_install_preflight": post_install_preflight,
        "dependency_preflight_passed": bool(post_install_preflight["preflight"].get("compatibility_passed")),
        "canonical_row_counts": {
            "train": int(semantic_data["bundle_report"].get("train_count", 0)),
            "validation": int(semantic_data["bundle_report"].get("validation_count", 0)),
            "test": int(semantic_data["bundle_report"].get("test_count", 0)),
        },
        "semantic_row_counts": semantic_data["split_counts"],
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "dataset_verification": canonical_verification,
        "semantic_readiness": semantic_data["readiness"],
        "training_plan": training_plan,
        "artifact_manifest": artifact_manifest,
        "smoke_training_report": smoke_training["smoke_training_report"],
        "smoke_training_dir": smoke_corpus["root"],
        "sha_manifest_path": semantic_data["sha_manifest_path"],
        "final_zip": str(final_zip),
        "verdict": semantic_verdict(
            gate_results={
                "intent_accuracy": 0.0,
                "binding_accuracy": 0.0,
                "predicate_coverage": 0.0,
                "logical_structure_accuracy": 0.0,
                "semantic_schema_valid_rate": 0.0,
                "fallback_accuracy": 0.0,
            },
            readiness=bool(semantic_data["readiness"].get("ready")),
            fallback_rate=1.0,
        ),
    }
    if smoke_failure is not None:
        result["smoke_training_failure"] = {
            "type": type(smoke_failure).__name__,
            "message": str(smoke_failure),
        }
    return result


def main() -> int:
    result = run_notebook_flow()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
