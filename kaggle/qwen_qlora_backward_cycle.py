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

from .bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths
from .qwen_nf4_load_cycle import MODEL_ID, _memory, _write_json, run_qwen_nf4_load_cycle
from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


WORKFLOW_MODE = "qwen_qlora_backward"
PEFT_VERSION = "0.13.2"
PEFT_SPEC = f"peft=={PEFT_VERSION}"
MAX_SMOKE_SEQUENCE_LENGTH = 128
PEFT_INSTALL_TIMEOUT_SECONDS = 600


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _marker(name: str, payload: dict[str, Any]) -> None:
    print(f"{name}={json.dumps(payload, sort_keys=True)}", flush=True)


def _safe_tail(value: str | None, lines: int = 40) -> str | None:
    return "\n".join((value or "").splitlines()[-lines:]) or None


def _install_peft() -> dict[str, Any]:
    protected = ("torch", "bitsandbytes", "transformers", "tokenizers")
    before = {name: _version(name) for name in protected + ("peft",)}
    result: dict[str, Any] = {"requested": PEFT_SPEC, "no_deps": True, "before": before, "returncode": 0}
    if before["peft"] != PEFT_VERSION:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", PEFT_SPEC],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PEFT_INSTALL_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        result.update({"returncode": completed.returncode, "stdout_tail": _safe_tail(completed.stdout), "stderr_tail": _safe_tail(completed.stderr)})
    after = {name: _version(name) for name in protected + ("peft",)}
    result["after"] = after
    result["dependency_drift"] = {name: before[name] != after[name] for name in protected}
    result["ok"] = result["returncode"] == 0 and after["peft"] == PEFT_VERSION and not any(result["dependency_drift"].values())
    if not result["ok"]:
        result["classification"] = "DEPENDENCY_VERSION_DRIFT" if any(result["dependency_drift"].values()) else "PEFT_INSTALL_FAILED"
    return result


def _memory_or_none(torch_module: Any) -> dict[str, float | None]:
    try:
        return _memory(torch_module)
    except Exception:
        return {"total_mb": None, "allocated_mb": None, "reserved_mb": None, "peak_allocated_mb": None, "peak_reserved_mb": None}


def _synthetic_example(tokenizer: Any) -> tuple[dict[str, Any], int]:
    # This string is deliberately self-contained and is not derived from any corpus.
    text = '{"intent":"filter","semantic_bindings":{"measure":"revenue","dimension":"region"},"predicate_graph":{"operator":"AND"},"aggregation":null,"ranking":null,"limit":null,"requires_fallback":false,"confidence":0.99}'
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SMOKE_SEQUENCE_LENGTH, padding=False)
    sequence_length = int(encoded["input_ids"].shape[-1])
    if sequence_length > MAX_SMOKE_SEQUENCE_LENGTH:
        raise ValueError("synthetic_sequence_too_long")
    return encoded, sequence_length


def _write_failure(report_root: Path, stage: str, exc: BaseException, run_id: str, expected: str | None, executed: str | None) -> None:
    import traceback

    versions = {name: _version(name) for name in ("torch", "bitsandbytes", "transformers", "tokenizers", "accelerate", "peft")}
    cuda: dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {"available": bool(torch.cuda.is_available())}
        if cuda["available"]:
            cuda.update({"device_name": torch.cuda.get_device_name(0), "memory": _memory_or_none(torch)})
    except Exception:
        pass
    _write_json(report_root / "smoke_failure.json", {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "sanitized_message": str(exc).replace("\n", " ")[:1000],
        "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-20:]),
        "package_versions": versions,
        "cuda_state": cuda,
        "run_id": run_id,
        "expected_git_commit": expected,
        "executed_git_commit": executed,
    })


def _run_cycle_impl(*, output_root: Path, run_id: str, expected_git_commit: str | None, source_root: Path) -> dict[str, Any]:
    report_root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    ensure_kaggle_paths(report_root)
    executed: str | None = None
    try:
        resolved = resolve_executed_source_commit(run_root=report_root, repo_root=source_root, expected_git_commit=expected_git_commit)
        executed = resolved.get("executed_source_commit")
        write_source_identity(report_root, run_id=run_id, expected_git_commit=expected_git_commit, executed_source_commit=executed, source_identity_method=str(resolved.get("source_identity_method") or "unknown"), source_identity_verified=bool(resolved.get("source_identity_verified")))
        if expected_git_commit and executed != expected_git_commit:
            raise RuntimeError("stale_kaggle_checkout")

        _marker("QWEN_RUNTIME_GATE_STARTED_JSON", {"run_id": run_id, "required": ["BNB_NF4_P100_RUNTIME_PASSED", "QWEN_0_5B_NF4_P100_RUNTIME_PASSED"]})
        qwen_gate = run_qwen_nf4_load_cycle(output_root=output_root, run_id=run_id, expected_git_commit=expected_git_commit, source_root=source_root)
        if qwen_gate.get("verdict") != "QWEN_0_5B_NF4_P100_RUNTIME_PASSED":
            raise RuntimeError("QWEN_NF4_RUNTIME_GATE_FAILED")

        dependency = _install_peft()
        _write_json(report_root / "peft_dependency_result.json", dependency)
        _marker("PEFT_DEPENDENCY_RESULT_JSON", dependency)
        if not dependency["ok"]:
            raise RuntimeError(str(dependency.get("classification") or "PEFT_INSTALL_FAILED"))

        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.float16, trust_remote_code=True)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False
        preparation = {"ok": True, "gradient_checkpointing": True, "use_cache": bool(model.config.use_cache is False)}
        _write_json(report_root / "kbit_preparation_result.json", preparation)
        _marker("KBIT_PREPARATION_RESULT_JSON", preparation)

        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        missing_targets = [name for name in targets if name not in module_names]
        if missing_targets:
            raise RuntimeError("LORA_TARGET_MODULE_NOT_FOUND")
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=targets)
        model = get_peft_model(model, lora_config)
        lora_modules = sum(1 for name, _ in model.named_modules() if "lora_" in name)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        base_trainable = [name for name in trainable_names if "lora_" not in name.lower()]
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        lora_payload = {"ok": bool(trainable_parameters and not base_trainable), "r": 16, "alpha": 32, "dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM", "target_modules": targets, "lora_module_count": lora_modules}
        _write_json(report_root / "lora_attachment_result.json", lora_payload)
        _marker("LORA_ATTACHMENT_RESULT_JSON", lora_payload)
        parameter_payload = {"total_parameters": total_parameters, "trainable_parameters": trainable_parameters, "trainable_percentage": round(trainable_parameters / total_parameters * 100, 6), "base_model_frozen": not base_trainable, "trainable_parameter_names_are_lora": not base_trainable}
        _write_json(report_root / "lora_parameter_result.json", parameter_payload)
        _marker("LORA_PARAMETER_RESULT_JSON", parameter_payload)
        if not lora_payload["ok"]:
            raise RuntimeError("NO_TRAINABLE_LORA_PARAMETERS" if not trainable_parameters else "BASE_MODEL_UNEXPECTEDLY_TRAINABLE")

        encoded, sequence_length = _synthetic_example(tokenizer)
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        encoded["labels"] = encoded["input_ids"].clone()
        model.train()
        torch.cuda.reset_peak_memory_stats()
        memory_after_lora = _memory_or_none(torch)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-5)
        optimizer.zero_grad(set_to_none=True)
        output = model(**encoded)
        loss = output.loss
        forward = {"loss": float(loss.detach().cpu()), "loss_finite": bool(torch.isfinite(loss).item()), "sequence_length": sequence_length}
        _write_json(report_root / "qlora_forward_result.json", forward)
        _marker("QLORA_FORWARD_RESULT_JSON", forward)
        if not forward["loss_finite"]:
            raise RuntimeError("QLORA_LOSS_NONFINITE")

        loss.backward()
        gradients = [parameter.grad for name, parameter in model.named_parameters() if "lora_" in name.lower() and parameter.grad is not None]
        gradient_norm = float(torch.sqrt(sum(torch.sum(gradient.detach().float() ** 2) for gradient in gradients)).item()) if gradients else 0.0
        backward = {"backward_called": True, "lora_gradient_present": bool(gradients), "lora_gradient_finite": bool(gradients) and all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients), "lora_gradient_norm": gradient_norm, "base_gradient_violation": bool(base_trainable)}
        _write_json(report_root / "qlora_backward_result.json", backward)
        _marker("QLORA_BACKWARD_RESULT_JSON", backward)
        if not backward["lora_gradient_present"]:
            raise RuntimeError("LORA_GRADIENT_MISSING")
        if not backward["lora_gradient_finite"]:
            raise RuntimeError("LORA_GRADIENT_NONFINITE")
        if gradient_norm <= 0:
            raise RuntimeError("LORA_GRADIENT_ZERO")

        before_params = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if "lora_" in name.lower()}
        optimizer.step()
        changed = any(not torch.equal(before_params[name], parameter.detach()) for name, parameter in model.named_parameters() if name in before_params)
        optimizer_payload = {"optimizer": "AdamW", "optimizer_step_count": 1, "lora_parameter_changed": changed, "base_model_changed": False}
        _write_json(report_root / "qlora_optimizer_result.json", optimizer_payload)
        _marker("QLORA_OPTIMIZER_RESULT_JSON", optimizer_payload)
        if not changed:
            raise RuntimeError("LORA_PARAMETER_UNCHANGED")

        memory = {"after_lora": memory_after_lora, "after_forward": _memory_or_none(torch), "after_backward": _memory_or_none(torch), "peak": _memory_or_none(torch)}
        _write_json(report_root / "qlora_memory_result.json", memory)
        _marker("QLORA_MEMORY_RESULT_JSON", memory)
        final = {"run_id": run_id, "expected_git_commit": expected_git_commit, "executed_git_commit": executed, "torch": qwen_gate.get("dependencies", {}), "qwen_gate": qwen_gate, "peft": dependency, "preparation": preparation, "lora": lora_payload, "parameters": parameter_payload, "synthetic_example": True, "max_sequence_length": MAX_SMOKE_SEQUENCE_LENGTH, "forward": forward, "backward": backward, "optimizer": optimizer_payload, "memory": memory, "train_data_used": False, "validation_data_used": False, "test_data_used": False, "checkpoint_saved": False, "adapter_saved": False, "verdict": "QWEN_0_5B_QLORA_BACKWARD_P100_PASSED"}
        _write_json(report_root / "qlora_backward_report.json", final)
        _marker("QLORA_FINAL_RESULT_JSON", final)
        return final
    except Exception as exc:
        _write_failure(report_root, "qwen_qlora_backward", exc, run_id, expected_git_commit, executed)
        raise


def run_qwen_qlora_backward_cycle(*, output_root: Path, run_id: str, expected_git_commit: str | None, source_root: Path) -> dict[str, Any]:
    return _run_cycle_impl(output_root=output_root, run_id=run_id, expected_git_commit=expected_git_commit, source_root=source_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(KAGGLE_WORKING_ROOT))
    parser.add_argument("--run-id")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--source-root")
    args = parser.parse_args(argv)
    root = Path(args.output_root)
    run_id = args.run_id or resolve_current_run_id(base_root=root / "smoke_runs") or generate_run_id()
    report = run_qwen_qlora_backward_cycle(output_root=root, run_id=run_id, expected_git_commit=args.expected_git_commit, source_root=Path(args.source_root) if args.source_root else root / "data_analysis_LLM")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("verdict") == "QWEN_0_5B_QLORA_BACKWARD_P100_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
