"""Controlled semantic QLoRA learning experiment.

This workflow is intentionally separate from the runtime smoke test.  It uses
only deterministic train/validation subsets and never opens the test file.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from learning.semantic_extractor_training import (
    ALLOWED_SEMANTIC_OUTPUT_KEYS,
    semantic_metrics,
    validate_semantic_target,
)

from .bootstrap import (
    KAGGLE_INPUT_ROOT,
    _load_canonical_record,
    _load_json_if_exists,
    _split_row_count,
    resolve_canonical_dataset_root,
)
from .qwen_qlora_backward_cycle import (
    MODEL_ID,
    _load_training_rows,
    _safe_target_hash,
    _target_output,
)
from .qwen_nf4_load_cycle import _memory, run_qwen_nf4_load_cycle
from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


TRAIN_EXAMPLES = 128
VALIDATION_EXAMPLES = 16
GRADIENT_ACCUMULATION = 8
OPTIMIZER_STEPS = 16
VALIDATION_STEPS = (0, 4, 8, 12, 16)
LEARNING_RATE = 1e-5
MAX_SEQUENCE_LENGTH = 768
GENERATION_MAX_NEW_TOKENS = 192
PEFT_VERSION = "0.13.2"


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _marker(name: str, payload: dict[str, Any]) -> None:
    print(f"{name}={json.dumps(payload, sort_keys=True)}", flush=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _prompt(row: dict[str, Any]) -> str:
    return "Extract the semantic analytics contract. Return JSON only with exactly these keys: intent, semantic_bindings, predicate_graph, aggregation, ranking, limit, requires_fallback, confidence. INPUT=" + json.dumps(row.get("input") or {}, sort_keys=True, separators=(",", ":")) + " OUTPUT="


def _parse_prediction(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(value, dict) or set(value) != ALLOWED_SEMANTIC_OUTPUT_KEYS:
        return None
    return value


def _prediction_is_valid(prediction: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if prediction is None:
        return False
    envelope = {
        "source_kind": "experience",
        "source_id": "0" * 64,
        "split": "validation",
        "family_fingerprint": "0" * 64,
        "input": expected.get("input") or {},
        "output": prediction,
        "metadata": {},
    }
    return validate_semantic_target(envelope)[0]


def _structure_summary(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    graph = value.get("predicate_graph") if isinstance(value.get("predicate_graph"), dict) else {}
    bindings = value.get("semantic_bindings") if isinstance(value.get("semantic_bindings"), dict) else {}
    return {
        "keys": sorted(value.keys()),
        "binding_keys": sorted(bindings.keys()),
        "logical_structure": graph.get("logical_structure"),
        "predicate_count": graph.get("predicate_count"),
        "operator_count": len(graph.get("operators") or []) if isinstance(graph.get("operators"), list) else 0,
        "role_count": len(graph.get("roles") or []) if isinstance(graph.get("roles"), list) else 0,
    }


def _load_experiment_dataset() -> dict[str, Any]:
    resolved = resolve_canonical_dataset_root(KAGGLE_INPUT_ROOT)
    if not resolved.get("root"):
        raise RuntimeError(str(resolved.get("reason") or "canonical_dataset_missing"))
    root = Path(str(resolved["root"]))
    # Deliberately inspect metadata only for test; the test JSONL is never opened.
    manifest = _load_json_if_exists(root / "dataset_manifest.json") or _load_json_if_exists(root / "manifest.json") or {}
    report = _load_json_if_exists(root / "dataset_report.json") or _load_json_if_exists(root / "report.json") or {}
    train_total = _split_row_count(root / "train.jsonl")
    validation_total = _split_row_count(root / "validation.jsonl")
    test_total = int(manifest.get("test_count", report.get("test_count", 0)) or 0)
    if (train_total, validation_total, test_total) != (407, 47, 46):
        raise RuntimeError("canonical_split_counts_unexpected")
    train_rows = sorted(_load_training_rows(root, "train"), key=_safe_target_hash)
    validation_rows = sorted(_load_training_rows(root, "validation"), key=_safe_target_hash)
    selected = {"train": train_rows[:TRAIN_EXAMPLES], "validation": validation_rows[:VALIDATION_EXAMPLES]}
    if len(selected["train"]) != TRAIN_EXAMPLES or len(selected["validation"]) != VALIDATION_EXAMPLES:
        raise RuntimeError("experiment_subset_too_small")
    for split, rows in selected.items():
        for row in rows:
            valid, reason = validate_semantic_target(row)
            safe = json.dumps({"input": row.get("input"), "output": row.get("output")}, sort_keys=True)
            metadata_payload = row.get("metadata") or {}
            if not valid or float(metadata_payload.get("quality") or 0.0) < 0.95 or bool(metadata_payload.get("ambiguity")):
                raise RuntimeError(f"{split}_eligibility_failed:{reason or 'quality'}")
            if any(token in safe.lower() for token in ("query_text", "normalized_query", "workbook", "sheet_name", "filename", "customer name", "sql", "tool_graph")):
                raise RuntimeError("training_privacy_gate_failed")
    return {
        "root": str(root),
        "train_total": train_total,
        "validation_total": validation_total,
        "test_total": test_total,
        "selected": selected,
        "selected_hashes": {split: [_safe_target_hash(row) for row in rows] for split, rows in selected.items()},
        "test_split_accessed": False,
    }


def _audit_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    for row in rows[:4]:
        target = _target_output(row)
        valid, reason = validate_semantic_target(row)
        summaries.append({
            "example_id": _safe_target_hash(row),
            "target_parse_success": valid,
            "expected_intent": target.get("intent"),
            "expected_structure": _structure_summary(target),
            "failure_reason": reason,
        })
    return {
        "target_keys": sorted(ALLOWED_SEMANTIC_OUTPUT_KEYS),
        "parser_keys": sorted(ALLOWED_SEMANTIC_OUTPUT_KEYS),
        "training_target_schema_matches_parser": True,
        "samples": summaries,
    }


def _install_peft() -> dict[str, Any]:
    protected = ("torch", "bitsandbytes", "transformers", "tokenizers")
    before = {name: _version(name) for name in protected + ("peft",)}
    result: dict[str, Any] = {"requested": f"peft=={PEFT_VERSION}", "no_deps": True, "before": before, "returncode": 0}
    if before["peft"] != PEFT_VERSION:
        completed = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", f"peft=={PEFT_VERSION}"], check=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        result.update({"returncode": completed.returncode, "stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]), "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:])})
    after = {name: _version(name) for name in protected + ("peft",)}
    result.update({"after": after, "dependency_drift": {name: before[name] != after[name] for name in protected}})
    result["ok"] = result["returncode"] == 0 and after["peft"] == PEFT_VERSION and not any(result["dependency_drift"].values())
    return result


def _tokenize_supervised(tokenizer: Any, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    prompt_ids = tokenizer(_prompt(row), add_special_tokens=True, truncation=False)["input_ids"]
    target_text = json.dumps(_target_output(row), sort_keys=True, separators=(",", ":"))
    full = tokenizer(_prompt(row) + target_text, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH, padding=False)
    input_ids = full["input_ids"]
    prompt_count = min(len(prompt_ids), int(input_ids.shape[-1]))
    labels = input_ids.clone()
    labels[:, :prompt_count] = -100
    supervised = int((labels != -100).sum().item())
    if supervised <= 0:
        raise RuntimeError("NO_SUPERVISED_TARGET_TOKENS")
    return {key: value.to("cuda:0") for key, value in full.items()} | {"labels": labels.to("cuda:0")}, {
        "input_tokens": int(input_ids.shape[-1]),
        "target_tokens": supervised,
        "masked_label_count": prompt_count,
        "supervised_label_count": supervised,
    }


def _evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], torch_module: Any, *, diagnostics: bool = False) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    parse_failures = empty_outputs = truncated = 0
    model.eval()
    for row in rows:
        encoded = tokenizer(_prompt(row), return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH).to("cuda:0")
        with torch_module.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=GENERATION_MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
        prediction = _parse_prediction(decoded)
        if not decoded.strip():
            empty_outputs += 1
        if prediction is None:
            parse_failures += 1
        if int(generated.shape[-1] - encoded["input_ids"].shape[-1]) >= GENERATION_MAX_NEW_TOKENS:
            truncated += 1
        expected_output = _target_output(row)
        predictions.append(prediction or {})
        expected.append(expected_output)
        if diagnostics and len(samples) < 4:
            reasons = []
            if prediction is None:
                reasons.append("parse_or_schema_failure")
            elif prediction.get("intent") != expected_output.get("intent"):
                reasons.append("intent_mismatch")
            elif prediction.get("semantic_bindings") != expected_output.get("semantic_bindings"):
                reasons.append("binding_mismatch")
            elif prediction.get("predicate_graph") != expected_output.get("predicate_graph"):
                reasons.append("predicate_mismatch")
            samples.append({"example_id": _safe_target_hash(row), "expected_intent": expected_output.get("intent"), "predicted_intent": (prediction or {}).get("intent"), "target_parse_success": True, "prediction_parse_success": prediction is not None, "expected_structure": _structure_summary(expected_output), "predicted_structure": _structure_summary(prediction), "metric_failure_reasons": reasons})
    metrics = semantic_metrics(predictions, expected)
    metrics.update({"generation_parse_failure_count": parse_failures, "empty_output_count": empty_outputs, "truncated_prediction_count": truncated})
    if diagnostics:
        metrics["diagnostic_samples"] = samples
    return metrics


def _score(metrics: dict[str, Any]) -> float:
    keys = ("semantic_schema_valid_rate", "intent_accuracy", "binding_accuracy", "predicate_coverage", "logical_structure_accuracy", "fallback_accuracy")
    return sum(float(metrics.get(key, 0.0)) for key in keys) / len(keys)


def _failure(root: Path, stage: str, exc: BaseException, run_id: str, expected: str | None, executed: str | None) -> None:
    import traceback
    _write_json(root / "smoke_failure.json", {"run_id": run_id, "stage": stage, "exception_type": type(exc).__name__, "sanitized_message": str(exc).replace("\n", " ")[:1000], "traceback_tail": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-12:]), "package_versions": {name: _version(name) for name in ("torch", "bitsandbytes", "transformers", "tokenizers", "accelerate", "peft")}, "expected_git_commit": expected, "executed_git_commit": executed})


def run_qwen_qlora_learning_experiment(*, output_root: Path, run_id: str, expected_git_commit: str | None, source_root: Path) -> dict[str, Any]:
    root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    executed: str | None = None
    stage = "source_identity"
    try:
        resolved = resolve_executed_source_commit(run_root=root, repo_root=source_root, expected_git_commit=expected_git_commit)
        executed = resolved.get("executed_source_commit")
        write_source_identity(root, run_id=run_id, expected_git_commit=expected_git_commit, executed_source_commit=executed, source_identity_method=str(resolved.get("source_identity_method") or "unknown"), source_identity_verified=bool(resolved.get("source_identity_verified")))
        if expected_git_commit and executed != expected_git_commit:
            raise RuntimeError("stale_kaggle_checkout")
        stage = "qwen_nf4_runtime"
        qwen_gate = run_qwen_nf4_load_cycle(output_root=output_root, run_id=run_id, expected_git_commit=expected_git_commit, source_root=source_root)
        if qwen_gate.get("verdict") != "QWEN_0_5B_NF4_P100_RUNTIME_PASSED":
            raise RuntimeError("QWEN_NF4_RUNTIME_GATE_FAILED")
        stage = "dataset"
        dataset = _load_experiment_dataset()
        dataset_result = {"canonical_dataset_root": dataset["root"], "train_total": dataset["train_total"], "validation_total": dataset["validation_total"], "test_total": dataset["test_total"], "experiment_train_examples": TRAIN_EXAMPLES, "experiment_validation_examples": VALIDATION_EXAMPLES, "test_split_accessed": False, "selected_train_hashes": dataset["selected_hashes"]["train"], "selected_validation_hashes": dataset["selected_hashes"]["validation"]}
        _write_json(root / "learning_experiment_dataset_result.json", dataset_result)
        _marker("LEARNING_EXPERIMENT_DATASET_RESULT_JSON", dataset_result)
        privacy = {"privacy_gate": True, "eligibility_gate": True, "test_split_accessed": False, "selected_train": TRAIN_EXAMPLES, "selected_validation": VALIDATION_EXAMPLES}
        _write_json(root / "learning_experiment_privacy_result.json", privacy)
        _marker("LEARNING_EXPERIMENT_PRIVACY_RESULT_JSON", privacy)
        contract = _audit_contract(dataset["selected"]["train"])
        _write_json(root / "semantic_contract_audit.json", contract)
        _marker("SEMANTIC_CONTRACT_AUDIT_JSON", contract)
        stage = "peft_install"
        dependency = _install_peft()
        _write_json(root / "learning_experiment_peft_result.json", dependency)
        _marker("LEARNING_EXPERIMENT_PEFT_RESULT_JSON", dependency)
        if not dependency["ok"]:
            raise RuntimeError("PEFT_INSTALL_FAILED")
        stage = "model_load"
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.float16, trust_remote_code=True)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        if any(target not in module_names for target in targets):
            raise RuntimeError("LORA_TARGET_MODULE_NOT_FOUND")
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=targets))
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        if not trainable or any(parameter.requires_grad and "lora_" not in name.lower() for name, parameter in model.named_parameters()):
            raise RuntimeError("LORA_FREEZE_CONTRACT_FAILED")
        _write_json(root / "learning_experiment_lora_result.json", {"r": 16, "alpha": 32, "dropout": 0.05, "targets": targets, "trainable_parameters": trainable, "total_parameters": total, "base_model_frozen": True})
        train_rows, validation_rows = dataset["selected"]["train"], dataset["selected"]["validation"]
        encoded_train, audits = [], []
        max_target_tokens = 0
        for row in train_rows:
            encoded, audit = _tokenize_supervised(tokenizer, row)
            encoded_train.append(encoded)
            audits.append(audit)
            max_target_tokens = max(max_target_tokens, audit["target_tokens"])
        tokenization = {"max_sequence_length": MAX_SEQUENCE_LENGTH, "max_target_tokens": max_target_tokens, "generation_max_new_tokens": max(GENERATION_MAX_NEW_TOKENS, max_target_tokens + 16), "audit_samples": audits[:4], "supervised_labeling_verified": True, "truncated_training_examples": sum(item["input_tokens"] >= MAX_SEQUENCE_LENGTH for item in audits)}
        _write_json(root / "learning_experiment_tokenization_result.json", tokenization)
        _marker("LEARNING_EXPERIMENT_TOKENIZATION_RESULT_JSON", tokenization)
        evaluations: dict[str, Any] = {"step_0": _evaluate(model, tokenizer, validation_rows, torch, diagnostics=True)}
        _write_json(root / "learning_experiment_validation_metrics.json", evaluations)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LEARNING_RATE)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if "lora_" in name.lower()}
        steps: list[dict[str, Any]] = []
        best_step, best_score = 0, _score(evaluations["step_0"])
        adapter_root = root / "adapters"
        for index, encoded in enumerate(encoded_train, start=1):
            if index == 1:
                optimizer.zero_grad(set_to_none=True)
            output = model(**encoded)
            loss = output.loss / GRADIENT_ACCUMULATION
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("TRAINING_LOSS_NONFINITE")
            loss.backward()
            if index % GRADIENT_ACCUMULATION == 0:
                gradients = [parameter.grad for name, parameter in model.named_parameters() if "lora_" in name.lower() and parameter.grad is not None]
                norm = float(torch.sqrt(sum(torch.sum(gradient.detach().float() ** 2) for gradient in gradients)).item()) if gradients else 0.0
                if not gradients or norm <= 0 or not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
                    raise RuntimeError("TRAINING_GRADIENT_NONFINITE")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step = index // GRADIENT_ACCUMULATION
                step_result = {"step": step, "loss": float((loss * GRADIENT_ACCUMULATION).detach().cpu()), "gradient_norm": norm, "learning_rate": LEARNING_RATE, "vram": _memory(torch)}
                steps.append(step_result)
                if step in VALIDATION_STEPS[1:]:
                    metrics = _evaluate(model, tokenizer, validation_rows, torch, diagnostics=step == 16)
                    evaluations[f"step_{step}"] = metrics
                    score = _score(metrics)
                    if score > best_score:
                        best_score, best_step = score, step
                        best_dir = adapter_root / f"best_step_{step}"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir, safe_serialization=True)
        changed = any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters() if name in before)
        if not changed:
            raise RuntimeError("LORA_PARAMETER_UNCHANGED")
        train_sanity = _evaluate(model, tokenizer, train_rows[:8], torch, diagnostics=True)
        final_dir = adapter_root / f"best_step_{best_step}"
        if not final_dir.exists():
            final_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_dir, safe_serialization=True)
        adapter_hashes = [{"name": str(path.relative_to(final_dir)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for path in sorted(final_dir.rglob("*")) if path.is_file()]
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reload_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.float16, trust_remote_code=True)
        reload_model = PeftModel.from_pretrained(reload_base, final_dir)
        reload_model.eval()
        reload_inputs = tokenizer(_prompt(validation_rows[0]), return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            reload_model.generate(**reload_inputs, max_new_tokens=tokenization["generation_max_new_tokens"], do_sample=False, pad_token_id=tokenizer.eos_token_id)
        final = {"run_id": run_id, "expected_git_commit": expected_git_commit, "executed_git_commit": executed, "dataset": dataset_result, "privacy": privacy, "contract": contract, "tokenization": tokenization, "config": {"model": MODEL_ID, "quantization": "4-bit NF4", "double_quant": True, "compute_dtype": "float16", "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "micro_batch_size": 1, "gradient_accumulation": 8, "effective_batch_size": 8, "learning_rate": LEARNING_RATE, "optimizer_steps": OPTIMIZER_STEPS}, "validation_metrics": evaluations, "train_sanity_metrics": train_sanity, "training_steps": steps, "loss_first": steps[0]["loss"], "loss_last": steps[-1]["loss"], "loss_mean": sum(item["loss"] for item in steps) / len(steps), "loss_trend": "decreasing" if steps[-1]["loss"] < steps[0]["loss"] else "not_decreasing", "lora_parameter_changed": changed, "base_model_frozen": True, "best_validation_step": best_step, "best_validation_score": best_score, "adapter_saved": True, "adapter_reload": True, "adapter_path": str(final_dir), "adapter_files": adapter_hashes, "vram_peak": _memory(torch), "train_data_used": True, "validation_data_used": True, "test_data_used": False, "test_split_accessed": False, "test_used": False, "checkpoint_saved": False, "verdict": "SEMANTIC_LEARNING_SIGNAL_CONFIRMED" if any(_score(evaluations[f"step_{step}"]) > _score(evaluations["step_0"]) and evaluations[f"step_{step}"]["semantic_schema_valid_rate"] >= evaluations["step_0"]["semantic_schema_valid_rate"] for step in VALIDATION_STEPS[1:]) else "MODEL_NOT_LEARNING"}
        _write_json(root / "learning_experiment_report.json", final)
        _marker("LEARNING_EXPERIMENT_FINAL_RESULT_JSON", final)
        return final
    except Exception as exc:
        _failure(root, stage, exc, run_id, expected_git_commit, executed)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/kaggle/working")
    parser.add_argument("--run-id")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--source-root")
    args = parser.parse_args(argv)
    root = Path(args.output_root)
    run_id = args.run_id or resolve_current_run_id(base_root=root / "smoke_runs") or generate_run_id()
    result = run_qwen_qlora_learning_experiment(output_root=root, run_id=run_id, expected_git_commit=args.expected_git_commit, source_root=Path(args.source_root) if args.source_root else root / "data_analysis_LLM")
    return 0 if result.get("verdict") in {"SEMANTIC_LEARNING_SIGNAL_CONFIRMED", "MODEL_NOT_LEARNING", "MODEL_FITS_TRAIN_NOT_VALIDATION"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
