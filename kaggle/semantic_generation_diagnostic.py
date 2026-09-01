"""Generation-only diagnostics for the semantic extractor.

This workflow deliberately loads no canonical dataset and performs no
optimizer, backward, or training operation.  All prompts are synthetic.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .qwen_qlora_learning_experiment import (
    MODEL_ID,
    _extract_first_json_object,
    _generation_termination_reason,
    _parse_prediction_diagnostic,
    _prediction_is_valid,
    _prompt,
    _target_output,
    _version,
)
from .qwen_nf4_load_cycle import _memory
from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


DIAGNOSTIC_BUDGETS = (192, 256, 384)
EXPECTED_DEPENDENCY_VERSIONS = {
    "torch": "2.5.1+cu118",
    "transformers": "4.46.3",
    "tokenizers": "0.20.3",
    "accelerate": "1.13.0",
    "bitsandbytes": "0.43.3",
    "peft": "0.13.2",
}
GENERATION_DEPENDENCIES = (
    ("transformers", "transformers==4.46.3"),
    ("tokenizers", "tokenizers==0.20.3"),
    ("accelerate", "accelerate==1.13.0"),
    ("peft", "peft==0.13.2"),
    ("huggingface_hub", "huggingface_hub==0.26.2"),
)
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


def _redact_safe_text(value: str | None) -> str | None:
    text = value or ""
    for token in ("token", "password", "secret", "api_key", "access_key", "authorization"):
        import re
        text = re.sub(rf"(?i)({token})\s*([:=])\s*[^\s]+", r"\1\2[REDACTED]", text)
    return text


def _safe_tail(value: str | None, lines: int = 40) -> str | None:
    return _redact_safe_text("\n".join((value or "").splitlines()[-lines:])) or None


def _run_dependency_install(package: str, requirement: str) -> dict[str, Any]:
    command = ["<python>", "-m", "pip", "install", "--no-deps", requirement]
    try:
        result = subprocess.run(
            [os.sys.executable, "-m", "pip", "install", "--no-deps", requirement],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=DEPENDENCY_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "package": package,
            "requirement": requirement,
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": _safe_tail(result.stdout),
            "stderr_tail": _safe_tail(result.stderr),
            "timed_out": False,
            "classification": None if result.returncode == 0 else f"{package.upper()}_INSTALL_FAILED",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "package": package,
            "requirement": requirement,
            "command": command,
            "returncode": None,
            "stdout_tail": _safe_tail(exc.stdout),
            "stderr_tail": _safe_tail(exc.stderr),
            "timed_out": True,
            "timeout_seconds": DEPENDENCY_TIMEOUT_SECONDS,
            "classification": f"{package.upper()}_INSTALL_FAILED",
        }


def _install_generation_dependencies(*, import_checker: Any = None) -> dict[str, Any]:
    before = {name: _version(name) for name in EXPECTED_DEPENDENCY_VERSIONS}
    commands = []
    for package, requirement in GENERATION_DEPENDENCIES:
        if before.get(package) != EXPECTED_DEPENDENCY_VERSIONS.get(package):
            result = _run_dependency_install(package, requirement)
            commands.append(result)
            if result.get("classification"):
                after = {name: _version(name) for name in EXPECTED_DEPENDENCY_VERSIONS}
                return {
                    "ok": False,
                    "classification": result["classification"],
                    "versions_before": before,
                    "versions": after,
                    "install_commands": commands,
                    "torch_version": after.get("torch"),
                    "torch_cuda": None,
                    "requested_versions": EXPECTED_DEPENDENCY_VERSIONS,
                }
    after = {name: _version(name) for name in EXPECTED_DEPENDENCY_VERSIONS}
    drift = [name for name, expected in EXPECTED_DEPENDENCY_VERSIONS.items() if after.get(name) != expected]
    import_failures = []
    import_checker = import_checker or __import__
    for module in ("torch", "transformers", "tokenizers", "accelerate", "bitsandbytes", "peft"):
        try:
            import_checker(module)
        except Exception as exc:
            import_failures.append({"module": module, "exception_type": type(exc).__name__, "message": _redact_safe_text(str(exc))})
    classification = "DEPENDENCY_IMPORT_FAILED" if import_failures else "DEPENDENCY_VERSION_MISMATCH" if drift else None
    torch_cuda = None
    try:
        import torch
        torch_cuda = getattr(torch.version, "cuda", None)
    except Exception:
        pass
    return {
        "ok": not drift and not import_failures,
        "classification": classification,
        "versions_before": before,
        "versions": after,
        "install_commands": commands,
        "requested_versions": EXPECTED_DEPENDENCY_VERSIONS,
        "torch_version": after.get("torch"),
        "torch_cuda": torch_cuda,
        "import_failures": import_failures,
        "torch_unchanged": before.get("torch") == after.get("torch"),
        "bitsandbytes_unchanged": before.get("bitsandbytes") == after.get("bitsandbytes"),
    }


def _synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    for intent, role in (("aggregate", "numeric_metric"), ("filter", "boolean"), ("compare", "numeric_metric"), ("trend", "date")):
        rows.append({
            "input": {
                "intent": intent,
                "safe_field_aliases": [role],
                "semantic_roles": [role],
                "dtypes": ["number" if role == "numeric_metric" else "string"],
                "logical_hints": {"logical_structure": "SINGLE", "predicate_count": 0, "operators": []},
            },
            "output": {
                "intent": intent,
                "semantic_bindings": {"intent_hint": intent},
                "predicate_graph": {"logical_structure": "SINGLE", "predicate_count": 0, "operators": ["SINGLE"], "roles": [role]},
                "aggregation": {"measure_roles": [role], "required": intent == "aggregate"},
                "ranking": {"required": False, "direction": "desc"},
                "limit": None,
                "requires_fallback": False,
                "confidence": 1.0,
            },
        })
    return rows


def _repetition_stat(token_ids: list[int], text: str) -> dict[str, Any]:
    repeated_adjacent = any(left == right for left, right in zip(token_ids, token_ids[1:]))
    trigrams = [tuple(token_ids[index : index + 3]) for index in range(max(0, len(token_ids) - 2))]
    repeated_trigrams = len(trigrams) != len(set(trigrams))
    return {"repeated_adjacent_token": repeated_adjacent, "repeated_trigram": repeated_trigrams, "repetition_detected": repeated_adjacent or repeated_trigrams, "completion_characters": len(text)}


def _target_eos_audit(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    target_text = json.dumps(_target_output(row), sort_keys=True, separators=(",", ":"))
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    return {"target_tokens_without_eos": len(target_ids), "target_eos_present": eos_id is not None, "target_eos_token_id": eos_id, "target_eos_supervised": eos_id is not None}


def _template_audit(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = _prompt(row)
    target = json.dumps(_target_output(row), sort_keys=True, separators=(",", ":"))
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    full_ids = tokenizer(prompt + target, add_special_tokens=True, truncation=False)["input_ids"]
    prefix_match = full_ids[: len(prompt_ids)] == prompt_ids
    return {
        "chat_template_used": False,
        "apply_chat_template_used": False,
        "add_generation_prompt": False,
        "training_inference_prefix_match": prefix_match,
        "system_range": None,
        "user_range": [0, len(prompt_ids)],
        "assistant_boundary": len(prompt_ids),
        "target_range": [len(prompt_ids), len(full_ids)],
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(full_ids) - len(prompt_ids),
    }


def _run_budget(model: Any, tokenizer: Any, torch_module: Any, rows: list[dict[str, Any]], budget: int) -> dict[str, Any]:
    results = []
    counts = {"EOS": 0, "MAX_NEW_TOKENS_REACHED": 0, "OTHER_STOPPING_CRITERION": 0, "NO_GENERATION": 0}
    classifications: dict[str, int] = {}
    valid_json = schema_valid = 0
    for row in rows:
        encoded = tokenizer(_prompt(row), return_tensors="pt", truncation=False).to("cuda:0")
        input_tokens = int(encoded["input_ids"].shape[-1])
        with torch_module.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=budget, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        completion_ids = generated[0][input_tokens:]
        completion_list = [int(value) for value in completion_ids.detach().cpu().tolist()]
        new_tokens = len(completion_list)
        decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)
        termination = _generation_termination_reason(completion_ids, generated_tokens=new_tokens, max_new_tokens=budget, eos_token_id=tokenizer.eos_token_id)
        counts[termination] = counts.get(termination, 0) + 1
        prediction, classification = _parse_prediction_diagnostic(decoded, generated_tokens=new_tokens, max_new_tokens=budget, termination_reason=termination)
        classifications[classification] = classifications.get(classification, 0) + 1
        object_text = _extract_first_json_object(decoded)
        has_json = object_text is not None
        valid_json += int(has_json)
        schema_ok = prediction is not None and _prediction_is_valid(prediction, row)
        schema_valid += int(schema_ok)
        results.append({
            "intent": row["output"]["intent"],
            "input_tokens": input_tokens,
            "total_output_tokens": int(generated.shape[-1]),
            "new_tokens": new_tokens,
            "completion_characters": len(decoded),
            "termination_reason": termination,
            "valid_json_object": has_json,
            "schema_valid": schema_ok,
            "parse_classification": classification,
            "completion_text": decoded,
            "repetition": _repetition_stat(completion_list, decoded),
        })
    return {
        "budget": budget,
        "eos_terminations": counts.get("EOS", 0),
        "limit_terminations": counts.get("MAX_NEW_TOKENS_REACHED", 0),
        "other_terminations": counts.get("OTHER_STOPPING_CRITERION", 0),
        "valid_json_objects": valid_json,
        "schema_valid_objects": schema_valid,
        "average_new_tokens": sum(item["new_tokens"] for item in results) / len(results),
        "repetition_count": sum(int(item["repetition"]["repetition_detected"]) for item in results),
        "classification_counts": classifications,
        "outputs": results,
    }


def _failure(root: Path, stage: str, exc: BaseException, *, expected: str | None, executed: str | None, torch_module: Any = None) -> None:
    import traceback

    cuda_state = {}
    if torch_module is not None and getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available():
        cuda_state = _memory(torch_module)
    _write_json(root / "smoke_failure.json", {"stage": stage, "exception_type": type(exc).__name__, "sanitized_message": _redact_safe_text(str(exc).replace("\n", " "))[:1000], "traceback_tail": _redact_safe_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-10:])), "expected_git_commit": expected, "executed_git_commit": executed, "package_versions": {name: _version(name) for name in ("torch", "transformers", "tokenizers", "accelerate", "bitsandbytes", "peft")}, "cuda_state": cuda_state, "test_split_accessed": False})


def run_semantic_generation_diagnostic(*, output_root: Path, run_id: str, expected_git_commit: str | None, source_root: Path) -> dict[str, Any]:
    root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    executed = None
    torch_module = None
    stage = "source_identity"
    try:
        resolved = resolve_executed_source_commit(run_root=root, repo_root=source_root, expected_git_commit=expected_git_commit)
        executed = resolved.get("executed_source_commit")
        write_source_identity(root, run_id=run_id, expected_git_commit=expected_git_commit, executed_source_commit=executed, source_identity_method=str(resolved.get("source_identity_method") or "unknown"), source_identity_verified=bool(resolved.get("source_identity_verified")))
        if expected_git_commit and executed != expected_git_commit:
            raise RuntimeError("stale_kaggle_checkout")
        stage = "dependencies"
        dependencies = _install_generation_dependencies()
        _write_json(root / "generation_diagnostic_dependencies.json", dependencies)
        _marker("MODEL_DEPENDENCY_RESULT_JSON", dependencies)
        if not dependencies.get("ok"):
            raise RuntimeError(str(dependencies.get("classification") or "MODEL_DEPENDENCY_INSTALL_FAILED"))
        stage = "model_load"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        torch_module = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.float16, trust_remote_code=True)
        model.eval()
        rows = _synthetic_rows()
        template = _template_audit(tokenizer, rows[0])
        eos_audit = _target_eos_audit(tokenizer, rows[0])
        generation_config = getattr(model, "generation_config", None)
        config_audit = {"eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "generation_eos_token_id": getattr(generation_config, "eos_token_id", None), "generation_pad_token_id": getattr(generation_config, "pad_token_id", None)}
        budgets = []
        for budget in DIAGNOSTIC_BUDGETS:
            stage = f"generation_{budget}"
            budgets.append(_run_budget(model, tokenizer, torch, rows, budget))
        torch.cuda.synchronize()
        final = {
            "run_id": run_id,
            "expected_git_commit": expected_git_commit,
            "executed_git_commit": executed,
            "model": MODEL_ID,
            "dependencies": dependencies,
            "test_split_accessed": False,
            "training_performed": False,
            "backward_called": False,
            "optimizer_step_count": 0,
            "tokenizer_loaded": True,
            "model_loaded": True,
            "quantization_4bit": True,
            "quantization_type": "NF4",
            "compute_dtype": "float16",
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "template_audit": template,
            "eos_audit": eos_audit,
            "generation_config": config_audit,
            "completion_only_decoding": True,
            "balanced_json_extraction": True,
            "budgets": budgets,
            "base_vs_adapter": {"compared": False, "reason": "previous_runtime_adapter_unavailable"},
            "vram_peak": _memory(torch),
            "classification": "GENERATION_TERMINATION_WORKING" if any(item["valid_json_objects"] for item in budgets) else "GENERATION_REPETITION_FOUND" if any(item["repetition_count"] for item in budgets) else "GENERATION_BUDGET_INSUFFICIENT",
        }
        _write_json(root / "semantic_generation_diagnostic_report.json", final)
        _marker("SEMANTIC_GENERATION_DIAGNOSTIC_FINAL_RESULT_JSON", final)
        return final
    except Exception as exc:
        _failure(root, stage, exc, expected=expected_git_commit, executed=executed, torch_module=torch_module)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/kaggle/working")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expected-git-commit", default=None)
    parser.add_argument("--source-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.output_root)
    run_id = args.run_id or resolve_current_run_id(base_root=root / "smoke_runs") or generate_run_id()
    result = run_semantic_generation_diagnostic(output_root=root, run_id=run_id, expected_git_commit=args.expected_git_commit, source_root=Path(args.source_root) if args.source_root else root / "data_analysis_LLM")
    return 0 if result.get("model_loaded") else 1


if __name__ == "__main__":
    raise SystemExit(main())
