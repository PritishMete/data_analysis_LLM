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

from .bootstrap import (
    KAGGLE_INPUT_ROOT,
    KAGGLE_WORKING_ROOT,
    _load_canonical_record,
    _load_json_if_exists,
    _split_row_count,
    discover_semantic_dataset,
    ensure_kaggle_paths,
    resolve_canonical_dataset_root,
)
from learning.semantic_extractor_training import (
    ALLOWED_SEMANTIC_OUTPUT_KEYS,
    build_semantic_extractor_targets,
    validate_semantic_target,
)
from learning.training_export import TrainingExportBundle, TrainingExportPolicy
from .qwen_nf4_load_cycle import MODEL_ID, _memory, _write_json, run_qwen_nf4_load_cycle
from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


WORKFLOW_MODE = "qwen_qlora_backward"
PEFT_VERSION = "0.13.2"
PEFT_SPEC = f"peft=={PEFT_VERSION}"
MAX_SMOKE_SEQUENCE_LENGTH = 128
PEFT_INSTALL_TIMEOUT_SECONDS = 600
TRAIN_EXAMPLES = 32
VALIDATION_EXAMPLES = 8
GRADIENT_ACCUMULATION = 8
OPTIMIZER_STEPS = 4
LEARNING_RATE = 1e-5


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


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_target_hash(target: dict[str, Any]) -> str:
    return _hash_text(json.dumps({"input": target.get("input"), "output": target.get("output")}, sort_keys=True, separators=(",", ":")))


def _target_from_row(item: dict[str, Any], split: str) -> dict[str, Any] | None:
    output = item.get("output") if isinstance(item, dict) else None
    if isinstance(output, dict) and set(output) <= ALLOWED_SEMANTIC_OUTPUT_KEYS and "intent" in output:
        target = dict(item)
        target["split"] = split
        return target
    return None


def _load_training_rows(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    # Only the requested training/validation file is opened. There is no test-file path here.
    path = dataset_root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{split}_split_missing")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    direct_targets = [_target_from_row(item, split) for item in rows]
    if all(target is not None for target in direct_targets):
        return [target for target in direct_targets if target is not None]

    records = [_load_canonical_record(item, split) for item in rows]
    bundle = TrainingExportBundle(
        records=records,
        rejected_count=0,
        rejected_reasons=__import__("collections").Counter(),
        duplicates_removed=0,
        inspected_count=len(records),
        policy=TrainingExportPolicy.from_env(),
        source_distribution=__import__("collections").Counter(record.plan_source or record.source_kind for record in records),
        intent_distribution=__import__("collections").Counter(record.intent for record in records),
        tool_graph_distribution=__import__("collections").Counter("|".join(record.tool_graph) for record in records),
        step_distribution=__import__("collections").Counter(len(record.tool_graph) for record in records),
        predicate_complexity_distribution=__import__("collections").Counter(int(record.predicate_graph.get("predicate_count") or 0) for record in records),
        average_quality=sum(record.quality_score for record in records) / len(records) if records else 0.0,
        dataset_version="",
    )
    targets = build_semantic_extractor_targets(bundle)
    for target in targets:
        target.split = split
    return [target.to_dict() for target in targets]


def _load_smoke_dataset() -> dict[str, Any]:
    resolved = resolve_canonical_dataset_root(KAGGLE_INPUT_ROOT)
    root_value = resolved.get("root")
    if not root_value:
        raise RuntimeError(str(resolved.get("reason") or "canonical_dataset_missing"))
    root = Path(root_value)
    manifest = _load_json_if_exists(root / "dataset_manifest.json") or _load_json_if_exists(root / "manifest.json") or {}
    report = _load_json_if_exists(root / "dataset_report.json") or _load_json_if_exists(root / "report.json") or {}
    train_total = _split_row_count(root / "train.jsonl")
    validation_total = _split_row_count(root / "validation.jsonl")
    metadata_test_total = manifest.get("test_count", report.get("test_count", 0))
    if train_total != 407 or validation_total != 47 or int(metadata_test_total or 0) != 46:
        raise RuntimeError("canonical_split_counts_unexpected")
    train_rows = _load_training_rows(root, "train")
    validation_rows = _load_training_rows(root, "validation")
    train_rows = sorted(train_rows, key=_safe_target_hash)
    validation_rows = sorted(validation_rows, key=_safe_target_hash)
    selected_train = train_rows[:TRAIN_EXAMPLES]
    selected_validation = validation_rows[:VALIDATION_EXAMPLES]
    if len(selected_train) != TRAIN_EXAMPLES:
        raise RuntimeError("training_split_too_small")
    if len(selected_validation) != VALIDATION_EXAMPLES:
        raise RuntimeError("validation_split_too_small")
    selected = {"train": selected_train, "validation": selected_validation}
    privacy_reasons: list[str] = []
    for split, rows in selected.items():
        for row in rows:
            valid, reason = validate_semantic_target(row)
            safe_text = json.dumps({"input": row.get("input"), "output": row.get("output")}, sort_keys=True)
            if not valid:
                privacy_reasons.append(f"{split}:{reason}")
            metadata_payload = dict(row.get("metadata") or {})
            if float(metadata_payload.get("quality") or 0.0) < 0.95 or bool(metadata_payload.get("ambiguity")):
                privacy_reasons.append(f"{split}:training_eligibility_failed")
            if any(token in safe_text.lower() for token in ("query_text", "normalized_query", "workbook", "sheet_name", "filename", "customer name", "sql", "tool_graph")):
                privacy_reasons.append(f"{split}:unsafe_payload")
    if privacy_reasons:
        raise RuntimeError("TRAINING_PRIVACY_GATE_FAILED")
    return {
        "root": str(root),
        "train_total": train_total,
        "validation_total": validation_total,
        "test_total_metadata": int(metadata_test_total),
        "selected": selected,
        "selected_hashes": {split: [_safe_target_hash(row) for row in rows] for split, rows in selected.items()},
        "test_split_accessed": False,
    }


def _prompt_for_target(target: dict[str, Any]) -> str:
    return "Extract the semantic analytics contract. Return JSON only. INPUT=" + json.dumps(target.get("input") or {}, sort_keys=True, separators=(",", ":")) + " OUTPUT="


def _target_output(target: dict[str, Any]) -> dict[str, Any]:
    return dict(target.get("output") or {})


def _parse_generated(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except Exception:
        return None
    return value if isinstance(value, dict) and not any(key.lower() in {"sql", "tool_graph", "tools"} for key in value) else None


def _evaluate_semantic_model(model: Any, tokenizer: Any, rows: list[dict[str, Any]], torch_module: Any) -> dict[str, float]:
    predictions: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    model.eval()
    for row in rows:
        prompt = _prompt_for_target(row)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SMOKE_SEQUENCE_LENGTH).to("cuda:0")
        with torch_module.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(generated[0][encoded["input_ids"].shape[-1] :], skip_special_tokens=True)
        predictions.append(_parse_generated(decoded) or {})
        expected.append(_target_output(row))
    metrics = {
        "semantic_schema_valid_rate": sum(set(pred) == set(exp) and bool(pred) for pred, exp in zip(predictions, expected)) / len(expected),
    }
    for key in ("intent", "semantic_bindings", "predicate_graph", "requires_fallback"):
        metrics[{"intent": "intent_accuracy", "semantic_bindings": "binding_accuracy", "predicate_graph": "predicate_coverage", "requires_fallback": "fallback_accuracy"}[key]] = sum(pred.get(key) == exp.get(key) for pred, exp in zip(predictions, expected)) / len(expected)
    metrics["logical_structure_accuracy"] = sum((pred.get("predicate_graph") or {}).get("logical_structure") == (exp.get("predicate_graph") or {}).get("logical_structure") for pred, exp in zip(predictions, expected)) / len(expected)
    return metrics


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

        dataset = _load_smoke_dataset()
        dataset_payload = {
            "canonical_dataset_root": dataset["root"],
            "train_split_total": dataset["train_total"],
            "validation_split_total": dataset["validation_total"],
            "test_split_total_from_metadata": dataset["test_total_metadata"],
            "smoke_train_examples": TRAIN_EXAMPLES,
            "smoke_validation_examples": VALIDATION_EXAMPLES,
            "selected_train_hashes": dataset["selected_hashes"]["train"],
            "selected_validation_hashes": dataset["selected_hashes"]["validation"],
            "test_split_accessed": False,
        }
        _write_json(report_root / "training_dataset_result.json", dataset_payload)
        _marker("TRAINING_DATASET_RESULT_JSON", dataset_payload)
        privacy_payload = {"privacy_gate": True, "eligibility_gate": True, "selected_train": TRAIN_EXAMPLES, "selected_validation": VALIDATION_EXAMPLES, "test_split_accessed": False}
        _write_json(report_root / "training_privacy_result.json", privacy_payload)
        _marker("TRAINING_PRIVACY_RESULT_JSON", privacy_payload)

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

        selected_train = dataset["selected"]["train"]
        selected_validation = dataset["selected"]["validation"]
        max_observed_sequence_length = 0
        truncated_examples = 0
        encoded_train: list[dict[str, Any]] = []
        for row in selected_train:
            prompt = _prompt_for_target(row)
            target_text = json.dumps(_target_output(row), sort_keys=True, separators=(",", ":"))
            encoded_row = tokenizer(prompt + target_text, return_tensors="pt", truncation=True, max_length=768, padding=False)
            observed = int(encoded_row["input_ids"].shape[-1])
            max_observed_sequence_length = max(max_observed_sequence_length, observed)
            truncated_examples += int(observed >= 768)
            encoded_train.append({key: value.to("cuda:0") for key, value in encoded_row.items()})
            encoded_train[-1]["labels"] = encoded_train[-1]["input_ids"].clone()
        _write_json(report_root / "training_tokenization_result.json", {"max_seq_len": 768, "max_observed_sequence_length": max_observed_sequence_length, "truncated_example_count": truncated_examples})
        model.eval()
        pretrain_metrics = _evaluate_semantic_model(model, tokenizer, selected_validation, torch)
        _write_json(report_root / "pretrain_validation_result.json", pretrain_metrics)
        _marker("PRETRAIN_VALIDATION_RESULT_JSON", pretrain_metrics)

        model.train()
        torch.cuda.reset_peak_memory_stats()
        memory_after_lora = _memory_or_none(torch)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LEARNING_RATE)
        training_steps: list[dict[str, Any]] = []
        before_params = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if "lora_" in name.lower()}
        for microbatch_index, encoded in enumerate(encoded_train, start=1):
            optimizer.zero_grad(set_to_none=True) if microbatch_index == 1 else None
            output = model(**encoded)
            loss = output.loss / GRADIENT_ACCUMULATION
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("TRAINING_LOSS_NONFINITE")
            loss.backward()
            if microbatch_index % GRADIENT_ACCUMULATION == 0:
                gradients = [parameter.grad for name, parameter in model.named_parameters() if "lora_" in name.lower() and parameter.grad is not None]
                gradient_norm = float(torch.sqrt(sum(torch.sum(gradient.detach().float() ** 2) for gradient in gradients)).item()) if gradients else 0.0
                if not gradients or not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients) or gradient_norm <= 0:
                    raise RuntimeError("TRAINING_GRADIENT_NONFINITE")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step = microbatch_index // GRADIENT_ACCUMULATION
                step_payload = {"step": step, "microbatches_consumed": microbatch_index, "loss": float((loss * GRADIENT_ACCUMULATION).detach().cpu()), "gradient_norm": gradient_norm, "learning_rate": LEARNING_RATE, "vram_allocated_mb": _memory_or_none(torch)["allocated_mb"], "vram_reserved_mb": _memory_or_none(torch)["reserved_mb"]}
                training_steps.append(step_payload)
                _marker("TRAINING_STEP_RESULT_JSON", step_payload)
        if len(training_steps) != OPTIMIZER_STEPS:
            raise RuntimeError("TRAINING_OPTIMIZER_FAILED")
        changed = any(not torch.equal(before_params[name], parameter.detach()) for name, parameter in model.named_parameters() if name in before_params)
        if not changed:
            raise RuntimeError("LORA_PARAMETER_UNCHANGED")
        _write_json(report_root / "training_steps.json", {"steps": training_steps, "optimizer_steps": len(training_steps), "learning_rate": LEARNING_RATE})

        model.eval()
        posttrain_metrics = _evaluate_semantic_model(model, tokenizer, selected_validation, torch)
        _write_json(report_root / "posttrain_validation_result.json", posttrain_metrics)
        _marker("POSTTRAIN_VALIDATION_RESULT_JSON", posttrain_metrics)

        adapter_dir = report_root / "adapters" / "qwen_qlora_training_smoke"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
        adapter_hashes = [{"name": str(path.relative_to(adapter_dir)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in adapter_files]
        adapter_payload = {"adapter_saved": True, "adapter_path": str(adapter_dir), "adapter_size_bytes": sum(item["bytes"] for item in adapter_hashes), "files": adapter_hashes}
        _write_json(report_root / "adapter_save_result.json", adapter_payload)
        _marker("ADAPTER_SAVE_RESULT_JSON", adapter_payload)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        reload_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quantization, device_map={"": 0}, torch_dtype=torch.float16, trust_remote_code=True)
        from peft import PeftModel

        reload_model = PeftModel.from_pretrained(reload_base, adapter_dir)
        reload_model.eval()
        reload_inputs = tokenizer(_prompt_for_target(selected_validation[0]), return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            reload_generated = reload_model.generate(**reload_inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        reload_text = tokenizer.decode(reload_generated[0][reload_inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        reload_prediction = _parse_generated(reload_text)
        reload_payload = {"adapter_reload": True, "model_device": sorted({str(parameter.device) for parameter in reload_model.parameters()}), "cpu_fallback": any(str(parameter.device).startswith("cpu") for parameter in reload_model.parameters()), "generation": True, "semantic_parse": reload_prediction is not None}
        _write_json(report_root / "adapter_reload_result.json", reload_payload)
        _marker("ADAPTER_RELOAD_RESULT_JSON", reload_payload)
        memory = {"after_lora": memory_after_lora, "peak_training": _memory_or_none(torch), "peak_validation": _memory_or_none(torch), "peak_reload": _memory_or_none(torch)}
        _write_json(report_root / "qlora_memory_result.json", memory)
        _marker("QLORA_MEMORY_RESULT_JSON", memory)
        final = {"run_id": run_id, "expected_git_commit": expected_git_commit, "executed_git_commit": executed, "torch": qwen_gate.get("dependencies", {}), "qwen_gate": qwen_gate, "peft": dependency, "preparation": preparation, "lora": lora_payload, "parameters": parameter_payload, "dataset": dataset_payload, "privacy": privacy_payload, "synthetic_example": False, "max_seq_len": 768, "max_observed_sequence_length": max_observed_sequence_length, "truncated_example_count": truncated_examples, "micro_batch_size": 1, "gradient_accumulation": GRADIENT_ACCUMULATION, "effective_batch_size": GRADIENT_ACCUMULATION, "learning_rate": LEARNING_RATE, "optimizer_steps": training_steps, "pretrain_metrics": pretrain_metrics, "posttrain_metrics": posttrain_metrics, "lora_parameter_changed": changed, "base_model_frozen": not base_trainable, "adapter": adapter_payload, "reload": reload_payload, "memory": memory, "train_data_used": True, "validation_data_used": True, "test_data_used": False, "test_split_accessed": False, "checkpoint_saved": False, "adapter_saved": True, "verdict": "QLORA_REAL_TRAINING_SMOKE_PASSED"}
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
