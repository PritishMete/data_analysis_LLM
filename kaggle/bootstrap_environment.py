from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kaggle.bootstrap import (  # type: ignore[no-redef]
        KAGGLE_WORKING_ROOT,
        build_kaggle_dependency_plan,
        detect_resume_checkpoint,
        discover_semantic_dataset,
        ensure_kaggle_paths,
        inspect_kaggle_gpu_identity,
        _safe_run_json_probe,
        _probe_bitsandbytes_runtime,
        resolve_canonical_dataset_root,
        verify_attached_dataset,
        write_dependency_preflight_report,
    )
    from kaggle.run_context import ensure_run_root, generate_run_id, resolve_current_run_id, write_json as _write_json_helper  # type: ignore[no-redef]
    from kaggle.import_trace import write_import_trace  # type: ignore[no-redef]
    from kaggle.dependency_report import write_dependency_report  # type: ignore[no-redef]
    from kaggle.p100_torch_runtime import run_shared_p100_torch_bootstrap  # type: ignore[no-redef]
else:
    from .bootstrap import (
        KAGGLE_WORKING_ROOT,
        build_kaggle_dependency_plan,
        detect_resume_checkpoint,
        discover_semantic_dataset,
        ensure_kaggle_paths,
        inspect_kaggle_gpu_identity,
        _safe_run_json_probe,
        _probe_bitsandbytes_runtime,
        resolve_canonical_dataset_root,
        verify_attached_dataset,
        write_dependency_preflight_report,
    )
    from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, write_json as _write_json_helper
    from .import_trace import write_import_trace
    from .dependency_report import write_dependency_report
    from .p100_torch_runtime import run_shared_p100_torch_bootstrap


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _safe_tail(text: str | None, *, lines: int = 40) -> str | None:
    if not text:
        return None
    split = text.splitlines()
    return "\n".join(split[-lines:])


def _finalize_dependency_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Validate and durably publish the one report consumed by the notebook."""
    finalized = write_dependency_report(path, report)
    print("DEPENDENCY_REPORT_JSON=" + json.dumps(finalized, sort_keys=True), flush=True)
    remote_log = path.parent / "remote.log"
    with remote_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "DEPENDENCY_REPORT_FINALIZED "
            + json.dumps({key: finalized[key] for key in ("status", "install_success", "stack_verified")}, sort_keys=True)
            + "\n"
        )
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    return finalized


def _probe_runtime() -> dict[str, Any]:
    snippet = """
import json
import torch
payload = {
    "version": torch.__version__,
    "cuda": getattr(torch.version, "cuda", None),
    "available": bool(torch.cuda.is_available()),
}
if payload["available"]:
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["capability"] = list(torch.cuda.get_device_capability(0))
    try:
        payload["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception:
        payload["arch_list"] = None
print(json.dumps(payload))
"""
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    payload = {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout_tail": _safe_tail(result.stdout),
        "stderr_tail": _safe_tail(result.stderr),
        "json": None,
    }
    if result.stdout:
        try:
            payload["json"] = json.loads(result.stdout.splitlines()[-1])
        except Exception:
            payload["json"] = None
    return payload


def _install_packages(plan: dict[str, Any]) -> dict[str, Any]:
    requested: list[str] = []
    result_payload = {
        "status": "STARTED",
        "stage": "dependencies",
        "install_attempted": False,
        "install_success": False,
        "requested_packages": requested,
        "pip_exit_code": None,
        "installed_torch_distribution": None,
        "installed_transformers_distribution": None,
        "installed_accelerate_distribution": None,
        "installed_peft_distribution": None,
        "installed_bitsandbytes_distribution": None,
        "fresh_process_required": True,
        "stdout_tail": None,
        "stderr_tail": None,
    }
    pip_groups = (plan.get("preflight") or {}).get("install_plan", {}).get("pip_groups") or []
    if not pip_groups:
        result_payload["install_success"] = True
        result_payload["status"] = "SUCCESS"
        return result_payload
    result_payload["install_attempted"] = True
    last = None
    for group in pip_groups:
        packages = list(group.get("packages") or [])
        requested.extend(packages)
        command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir"]
        index_url = group.get("index_url")
        if index_url:
            command.extend(["--index-url", str(index_url)])
        command.extend(packages)
        last = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        result_payload["pip_exit_code"] = last.returncode
        result_payload["stdout_tail"] = _safe_tail(last.stdout)
        result_payload["stderr_tail"] = _safe_tail(last.stderr)
        if last.returncode != 0:
            result_payload["failed_dependency"] = packages[-1] if packages else group.get("name")
            result_payload["failed_command_safe"] = command[:4] + packages
            break
    if last is not None and last.returncode == 0:
        result_payload["install_success"] = True
        result_payload["status"] = "SUCCESS"
    elif last is not None:
        result_payload["status"] = "FAILED"
    try:
        result_payload["installed_torch_distribution"] = metadata.version("torch")
    except Exception:
        pass
    try:
        result_payload["installed_transformers_distribution"] = metadata.version("transformers")
    except Exception:
        pass
    try:
        result_payload["installed_accelerate_distribution"] = metadata.version("accelerate")
    except Exception:
        pass
    try:
        result_payload["installed_peft_distribution"] = metadata.version("peft")
    except Exception:
        pass
    try:
        result_payload["installed_bitsandbytes_distribution"] = metadata.version("bitsandbytes")
    except Exception:
        pass
    return result_payload


def _probe_nf4_runtime() -> dict[str, Any]:
    snippet = """
import json
import torch
import bitsandbytes.functional as functional
payload = {"initialization": False, "quantization": False, "dequantization": False, "cuda": False}
if not torch.cuda.is_available():
    print(json.dumps(payload))
    raise SystemExit(0)
tensor = torch.randn(32, device="cuda", dtype=torch.float16)
payload["initialization"] = True
quantized, state = functional.quantize_4bit(tensor, quant_type="nf4", compress_statistics=True)
payload["quantization"] = True
restored = functional.dequantize_4bit(quantized, state)
payload["dequantization"] = True
torch.cuda.synchronize()
payload["cuda"] = bool(restored.is_cuda)
print(json.dumps(payload))
"""
    return _safe_run_json_probe([sys.executable, "-c", snippet], timeout=60)


def _installed_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "tokenizers", "accelerate", "peft", "bitsandbytes", "huggingface-hub")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _stack_verified(*, versions: dict[str, str | None], torch_probe: dict[str, Any], bnb_probe: dict[str, Any], nf4_probe: dict[str, Any]) -> bool:
    expected = {
        "torch": "2.5.1+cu118",
        "transformers": "4.46.3",
        "tokenizers": "0.20.3",
        "accelerate": "1.13.0",
        "peft": "0.13.2",
        "bitsandbytes": "0.43.3",
        "huggingface-hub": "0.26.2",
    }
    runtime = torch_probe.get("json") or {}
    nf4 = nf4_probe.get("json") or {}
    return (
        all(versions.get(name) == value for name, value in expected.items())
        and runtime.get("version") == expected["torch"]
        and runtime.get("cuda") == "11.8"
        and bool(runtime.get("available"))
        and tuple(runtime.get("capability") or ()) == (6, 0)
        and "sm_60" in (runtime.get("arch_list") or [])
        and bool(bnb_probe.get("ok"))
        and bool((bnb_probe.get("json") or {}).get("real_bnb_cuda_operation"))
        and all(bool(nf4.get(key)) for key in ("initialization", "quantization", "dequantization", "cuda"))
    )


def _run_bootstrap(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(KAGGLE_WORKING_ROOT))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    resolved_run_id = args.run_id or resolve_current_run_id(base_root=output_root / "smoke_runs") or generate_run_id()
    run_root = Path(os.environ["KAGGLE_RUN_DIR"]) if os.environ.get("KAGGLE_RUN_DIR") else ensure_run_root(resolved_run_id, base_root=output_root / "smoke_runs")
    run_root.mkdir(parents=True, exist_ok=True)
    paths = ensure_kaggle_paths(run_root)
    report_root = run_root
    bootstrap_pid = os.getpid()
    write_import_trace(report_root / "import_trace.jsonl", module="kaggle.bootstrap_environment", event="bootstrap_started")
    _write_json_helper(report_root / "runner_metadata.json", {"run_id": resolved_run_id, "bootstrap_pid": bootstrap_pid, "timestamp": time.time()})
    dependency_report_path = report_root / "dependency_install_result.json"
    initial_report = {
        "schema_version": "1.0",
        "status": "STARTED",
        "run_id": resolved_run_id,
        "stage": "dependencies",
        "bootstrap_pid": bootstrap_pid,
        "install_attempted": False,
        "install_success": False,
        "stack_verified": False,
    }
    _finalize_dependency_report(dependency_report_path, initial_report)
    if str(os.environ.get("KAGGLE_WORKFLOW_MODE") or "").strip().lower() == "qwen_nf4_load":
        _finalize_dependency_report(dependency_report_path, {"schema_version": "1.0", "status": "SUCCESS", "stage": "dependencies", "run_id": resolved_run_id, "bootstrap_pid": bootstrap_pid, "install_success": True, "stack_verified": True, "model_load_bootstrap": True, "dataset_used": False})
        return 0
    if str(os.environ.get("KAGGLE_WORKFLOW_MODE") or "").strip().lower() == "bnb_native_diagnose":
        # The native diagnostic owns its isolated Torch/BNB setup and must not
        # inspect datasets or import the general training dependency graph.
        _finalize_dependency_report(dependency_report_path, {"schema_version": "1.0", "status": "SUCCESS", "stage": "dependencies", "run_id": resolved_run_id, "bootstrap_pid": bootstrap_pid, "install_success": True, "stack_verified": True, "diagnostic_bootstrap": True, "dataset_used": False})
        return 0
    repo_dataset = resolve_canonical_dataset_root()
    dataset_dir = Path(repo_dataset["root"]) if repo_dataset.get("root") else None
    if dataset_dir is None:
        _finalize_dependency_report(dependency_report_path, {"schema_version": "1.0", "status": "FAILED", "stage": "dependencies", "install_success": False, "stack_verified": False, "install_attempted": False, "fresh_process_required": True, "reason": repo_dataset.get("reason") or "dataset_not_found", "run_id": resolved_run_id})
        return 1
    verification = verify_attached_dataset(dataset_dir)
    if not verification.get("verified"):
        _finalize_dependency_report(dependency_report_path, {"schema_version": "1.0", "status": "FAILED", "stage": "dependencies", "install_success": False, "stack_verified": False, "install_attempted": False, "fresh_process_required": True, "reason": "canonical_dataset_verification_failed", "dataset_verification": verification, "run_id": resolved_run_id})
        return 1
    gpu_identity = inspect_kaggle_gpu_identity()
    try:
        shared_torch = run_shared_p100_torch_bootstrap(
            report_root=report_root,
            repo_root=Path(__file__).resolve().parents[1],
            phase_prefix="generation",
            write_markers=True,
        )
    except Exception as exc:
        _finalize_dependency_report(
            dependency_report_path,
            {
                **initial_report,
                "status": "FAILED",
                "install_attempted": True,
                "install_success": False,
                "stack_verified": False,
                "reason": str(exc),
                "classification": str(exc),
                "gpu_identity": gpu_identity,
            },
        )
        return 1
    torch_probe = {
        "ok": True,
        "json": {
            "version": shared_torch.get("torch_version"),
            "cuda": shared_torch.get("torch_cuda_version"),
            "available": bool(shared_torch.get("basic_cuda_tensor_test")),
            "device_name": shared_torch.get("gpu_name"),
            "capability": shared_torch.get("compute_capability"),
            "arch_list": shared_torch.get("arch_list"),
        },
    }
    write_import_trace(report_root / "import_trace.jsonl", module="kaggle.bootstrap_environment", event="before_bitsandbytes_import")
    bnb_probe = {"ok": True, "json": {"version": None, "available_cuda_versions": None}}
    write_import_trace(report_root / "import_trace.jsonl", module="kaggle.bootstrap_environment", event="after_bitsandbytes_import")
    preflight = build_kaggle_dependency_plan(gpu_identity=gpu_identity, torch_probe=torch_probe, bitsandbytes_probe=bnb_probe)
    preflight_payload = preflight.to_dict()
    # Torch has already been installed and verified by the shared P100 path;
    # the remaining installer may not reinstall it through a second plan.
    preflight_payload["install_plan"]["pip_groups"] = [
        group for group in preflight_payload["install_plan"].get("pip_groups", [])
        if group.get("name") != "torch_cu126"
    ]
    preflight.install_plan["pip_groups"] = preflight_payload["install_plan"]["pip_groups"]
    write_dependency_preflight_report(report_root, preflight)
    install_result = _install_packages({"preflight": preflight_payload, "gpu_identity": gpu_identity, "torch_probe": torch_probe})
    postinstall_torch = _probe_runtime()
    postinstall_bnb = _probe_bitsandbytes_runtime()
    nf4_probe = _probe_nf4_runtime() if postinstall_bnb.get("ok") else {"ok": False, "json": {}}
    versions = _installed_versions()
    stack_verified = _stack_verified(versions=versions, torch_probe=postinstall_torch, bnb_probe=postinstall_bnb, nf4_probe=nf4_probe)
    bnb_probe_json = postinstall_bnb.get("json") or {}
    nf4_probe_json = nf4_probe.get("json") or {}
    if not postinstall_bnb.get("ok"):
        probe_classification = "BNB_IMPORT_FAILED"
    elif not bnb_probe_json.get("real_bnb_cuda_operation"):
        probe_classification = "BNB_CUDA_RUNTIME_FAILED"
    elif not all(bool(nf4_probe_json.get(key)) for key in ("initialization", "quantization", "dequantization", "cuda")):
        probe_classification = "BNB_NF4_RUNTIME_FAILED"
    else:
        probe_classification = None
    install_result.update({
        "status": "SUCCESS" if install_result["install_success"] and stack_verified else "FAILED",
        "stage": "dependencies",
        "bootstrap_pid": bootstrap_pid,
        "run_id": resolved_run_id,
        "dataset_dir": str(dataset_dir),
        "gpu_identity": gpu_identity,
        "torch_probe": torch_probe,
        "shared_torch_bootstrap": shared_torch,
        "dependency_preflight_passed": bool(preflight.compatibility_passed or preflight.install_plan.get("pip_groups")),
        "planned_packages": preflight.install_plan,
        "runtime_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "versions": versions,
        "postinstall_torch": postinstall_torch,
        "postinstall_bnb": postinstall_bnb,
        "nf4_probe": nf4_probe,
        "stack_verified": stack_verified,
        "probe_classification": probe_classification,
        "resume_checkpoint": str(detect_resume_checkpoint(paths.checkpoints)) if detect_resume_checkpoint(paths.checkpoints) else None,
        "semantic_dataset_root": str(discover_semantic_dataset() or ""),
    })
    install_result["stack_verified"] = stack_verified
    install_result["schema_version"] = "1.0"
    _finalize_dependency_report(dependency_report_path, install_result)
    return 0 if install_result["status"] == "SUCCESS" else 1


def _finalize_unexpected_failure(argv: list[str] | None, exc: BaseException) -> None:
    """Publish a terminal failure when an unexpected bootstrap error escapes."""
    values = argv or sys.argv[1:]
    parsed: dict[str, str] = {}
    for index, value in enumerate(values):
        if value.startswith("--") and index + 1 < len(values):
            parsed[value[2:].replace("-", "_")] = values[index + 1]
    output_root = Path(parsed.get("output_root") or KAGGLE_WORKING_ROOT)
    run_id = parsed.get("run_id") or os.environ.get("KAGGLE_SMOKE_RUN_ID") or "unknown"
    run_root = Path(os.environ["KAGGLE_RUN_DIR"]) if os.environ.get("KAGGLE_RUN_DIR") else ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    report_path = run_root / "dependency_install_result.json"
    message = str(exc).replace("\n", " ")
    report = {
        "schema_version": "1.0",
        "status": "FAILED",
        "run_id": run_id,
        "stage": "dependencies",
        "install_success": False,
        "stack_verified": False,
        "install_attempted": True,
        "failed_dependency": None,
        "failed_command": None,
        "exit_code": None,
        "stderr_safe_tail": _safe_tail(message),
        "classification": type(exc).__name__,
        "reason": message[:1000],
        "unexpected_exception": True,
    }
    try:
        _finalize_dependency_report(report_path, report)
    except Exception as finalize_exc:
        print("DEPENDENCY_REPORT_FINALIZATION_FAILED=" + type(finalize_exc).__name__, flush=True)


def main(argv: list[str] | None = None) -> int:
    try:
        return _run_bootstrap(argv)
    except BaseException as exc:
        _finalize_unexpected_failure(argv, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
