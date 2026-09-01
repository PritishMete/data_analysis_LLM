from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .bnb_compat_cycle import run_bnb_compat_cycle
from .bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths, inspect_kaggle_gpu_identity
from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


WORKFLOW_MODE = "qwen_nf4_load"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TRANSFORMERS_MINIMUMS = ("transformers", "tokenizers", "accelerate", "huggingface_hub")
DEPENDENCY_TIMEOUT_SECONDS = 600


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _marker(name: str, payload: dict[str, Any]) -> None:
    print(f"{name}={json.dumps(payload, sort_keys=True)}", flush=True)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _safe_tail(text: str | None, lines: int = 40) -> str | None:
    return "\n".join((text or "").splitlines()[-lines:]) or None


def _install_missing_dependencies() -> dict[str, Any]:
    missing = [name for name in TRANSFORMERS_MINIMUMS if _version(name) is None]
    payload: dict[str, Any] = {
        "requested": missing,
        "no_deps": True,
        "torch_before": _version("torch"),
        "torch_after": None,
        "returncode": 0,
        "stdout_tail": None,
        "stderr_tail": None,
    }
    if missing:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", *missing],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=DEPENDENCY_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        payload.update({"returncode": result.returncode, "stdout_tail": _safe_tail(result.stdout), "stderr_tail": _safe_tail(result.stderr)})
    payload["torch_after"] = _version("torch")
    payload["versions"] = {name: _version(name) for name in TRANSFORMERS_MINIMUMS}
    payload["ok"] = payload["returncode"] == 0 and payload["torch_before"] == payload["torch_after"]
    if payload["torch_before"] != payload["torch_after"]:
        payload["classification"] = "TORCH_VERSION_DRIFT"
    elif payload["returncode"] != 0:
        payload["classification"] = "MODEL_DEPENDENCY_INSTALL_FAILED"
    return payload


def _memory(torch: Any) -> dict[str, Any]:
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2),
    }


def _failure(report_root: Path, stage: str, exc: BaseException, *, expected: str | None, executed: str | None) -> None:
    import traceback

    _write_json(report_root / "smoke_failure.json", {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "sanitized_message": str(exc).replace("\n", " ")[:1000],
        "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-12:]),
        "expected_git_commit": expected,
        "executed_git_commit": executed,
        "package_versions": {name: _version(name) for name in ("torch", "bitsandbytes", "transformers", "accelerate")},
        "gpu": inspect_kaggle_gpu_identity(),
        "cuda_state": {},
    })


def run_qwen_nf4_load_cycle(*, output_root: Path, run_id: str, expected_git_commit: str | None, source_root: Path) -> dict[str, Any]:
    report_root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    ensure_kaggle_paths(report_root)
    executed: str | None = None
    try:
        resolved = resolve_executed_source_commit(run_root=report_root, repo_root=source_root, expected_git_commit=expected_git_commit)
        executed = resolved.get("executed_source_commit")
        write_source_identity(report_root, run_id=run_id, expected_git_commit=expected_git_commit, executed_source_commit=executed, source_identity_method=str(resolved.get("source_identity_method") or "unknown"), source_identity_verified=bool(resolved.get("source_identity_verified")))
        if expected_git_commit and executed != expected_git_commit:
            raise RuntimeError("stale_kaggle_checkout")

        dependencies = _install_missing_dependencies()
        _write_json(report_root / "model_dependency_result.json", dependencies)
        _marker("MODEL_DEPENDENCY_RESULT_JSON", dependencies)
        if not dependencies["ok"]:
            raise RuntimeError(str(dependencies.get("classification") or "MODEL_DEPENDENCY_INSTALL_FAILED"))

        # This call is the single proven Torch + CUDA + BNB + NF4 gate.
        bnb_report = run_bnb_compat_cycle(output_root=output_root, run_id=run_id, expected_git_commit=expected_git_commit, source_root=source_root)
        if bnb_report.get("verdict") != "BNB_NF4_P100_RUNTIME_PASSED":
            raise RuntimeError(str(bnb_report.get("verdict") or "BNB_NF4_GATE_FAILED"))

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before_memory = _memory(torch)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer_payload = {
            "model_id": MODEL_ID,
            "tokenizer_loaded": True,
            "pad_token_valid": tokenizer.pad_token_id is not None,
            "eos_token_valid": tokenizer.eos_token_id is not None,
            "vocab_size": len(tokenizer),
        }
        _write_json(report_root / "tokenizer_result.json", tokenizer_payload)
        _marker("TOKENIZER_RESULT_JSON", tokenizer_payload)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.float16, trust_remote_code=True)
        quantized_modules = sum(1 for module in model.modules() if "4bit" in type(module).__name__.lower())
        model_payload = {
            "model_id": MODEL_ID,
            "model_class": type(model).__name__,
            "model_loaded": True,
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "float16",
            "quantized_module_count": quantized_modules,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        _write_json(report_root / "model_load_result.json", model_payload)
        _marker("MODEL_LOAD_RESULT_JSON", model_payload)
        devices = sorted({str(parameter.device) for parameter in model.parameters()})
        device_payload = {"devices": devices, "primary_device": "cuda:0", "cpu_fallback": any(device.startswith("cpu") for device in devices)}
        _write_json(report_root / "model_device_result.json", device_payload)
        _marker("MODEL_DEVICE_RESULT_JSON", device_payload)
        after_memory = _memory(torch)
        memory_payload = {"before": before_memory, "after_load": after_memory}
        _write_json(report_root / "model_memory_result.json", memory_payload)
        _marker("MODEL_MEMORY_RESULT_JSON", memory_payload)
        if not quantized_modules or device_payload["cpu_fallback"]:
            raise RuntimeError("MODEL_CPU_FALLBACK" if device_payload["cpu_fallback"] else "MODEL_4BIT_LOAD_FAILED")

        inputs = tokenizer("Classify the analytics intent: show average sales by region", return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        forward_payload = {"input_device": str(inputs["input_ids"].device), "generation_succeeded": True, "cuda_synchronize": True, "output_decoded": bool(decoded), "generated_tokens": int(generated.shape[-1])}
        _write_json(report_root / "model_forward_result.json", forward_payload)
        _marker("MODEL_FORWARD_RESULT_JSON", forward_payload)
        final = {
            "run_id": run_id,
            "expected_git_commit": expected_git_commit,
            "executed_git_commit": executed,
            "bnb_nf4_gate": True,
            "dependencies": dependencies,
            "tokenizer": tokenizer_payload,
            "model": model_payload,
            "device": device_payload,
            "memory": {"before": before_memory, "after_load": after_memory, "peak": _memory(torch)},
            "forward": forward_payload,
            "peft_imported": False,
            "lora_created": False,
            "backward_called": False,
            "train_data_used": False,
            "validation_data_used": False,
            "test_data_used": False,
            "verdict": "QWEN_0_5B_NF4_P100_RUNTIME_PASSED",
        }
        _write_json(report_root / "qwen_nf4_load_report.json", final)
        _marker("MODEL_FINAL_RESULT_JSON", final)
        return final
    except Exception as exc:
        _failure(report_root, "qwen_nf4_load", exc, expected=expected_git_commit, executed=executed)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(KAGGLE_WORKING_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expected-git-commit", default=None)
    parser.add_argument("--source-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.output_root)
    run_id = args.run_id or resolve_current_run_id(base_root=root / "smoke_runs") or generate_run_id()
    result = run_qwen_nf4_load_cycle(output_root=root, run_id=run_id, expected_git_commit=args.expected_git_commit, source_root=Path(args.source_root) if args.source_root else root / "data_analysis_LLM")
    return 0 if result["verdict"] == "QWEN_0_5B_NF4_P100_RUNTIME_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
