from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .bootstrap import (
    KAGGLE_WORKING_ROOT,
    build_artifact_manifest,
    build_semantic_kaggle_report,
    create_final_zip,
    detect_resume_checkpoint,
    discover_semantic_dataset,
    ensure_kaggle_paths,
    inspect_kaggle_environment,
    resolve_canonical_dataset_root,
    load_semantic_config,
    semantic_verdict,
    verify_attached_dataset,
    build_semantic_dataset_from_canonical,
)


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


def _run_real_smoke_training(
    *,
    base_model: str,
    smoke_root: Path,
    output_root: Path,
    resume_from: str | None = None,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    train_path = smoke_root / "train.jsonl"
    val_path = smoke_root / "validation.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("smoke_dataset_missing")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

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

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, texts: list[str]) -> None:
            self.texts = texts

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int) -> dict[str, Any]:
            encoded = tokenizer(
                self.texts[index],
                truncation=True,
                max_length=768,
                padding=False,
                return_tensors=None,
            )
            encoded["labels"] = list(encoded["input_ids"])
            return encoded

    train_dataset = _ListDataset(train_texts)
    val_dataset = _ListDataset(val_texts)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    smoke_steps = min(3, len(train_dataset))
    output_dir = output_root / "checkpoints" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=smoke_steps,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=1,
        eval_steps=1,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=True,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    duration = time.perf_counter() - start
    eval_metrics = trainer.evaluate()
    checkpoint_dir = output_dir / f"checkpoint-{smoke_steps}"
    checkpoint_created = checkpoint_dir.exists()

    adapter_dir = output_root / "adapters" / "smoke"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
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
        "train_rows": len(train_dataset),
        "validation_rows": len(val_dataset),
        "metrics": {
            "train_loss": float(train_result.training_loss) if train_result.training_loss is not None else None,
            "eval_loss": float(eval_metrics.get("eval_loss")) if eval_metrics.get("eval_loss") is not None else None,
        },
        "quantization": {
            "backend": "bitsandbytes",
            "bnb_4bit_quant_type": "nf4",
            "compute_dtype": "fp16",
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
        },
    }
    return {
        "smoke_training_report": smoke_training_report,
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_dir": str(adapter_dir),
        "train_steps": smoke_steps,
        "train_metrics": {"train_loss": smoke_training_report["metrics"]["train_loss"]},
        "validation_metrics": {"eval_loss": smoke_training_report["metrics"]["eval_loss"]},
    }


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


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


def run_notebook_flow(*, output_root: Path = KAGGLE_WORKING_ROOT, resume_from: str | None = None) -> dict[str, Any]:
    paths = ensure_kaggle_paths(output_root)
    resolved = resolve_canonical_dataset_root()
    dataset_dir = Path(resolved["root"]) if resolved.get("root") else None
    if dataset_dir is None:
        return {
            "verdict": "KAGGLE_GPU_NOT_AVAILABLE",
            "reason": resolved.get("reason") or "no_attached_dataset_found",
            "canonical_dataset_root": None,
            "paths": paths.to_dict(),
        }
    canonical_verification = verify_attached_dataset(dataset_dir)
    if not canonical_verification.get("verified"):
        return {
            "verdict": "TRAINING_FAILED",
            "reason": "canonical_dataset_verification_failed",
            "canonical_dataset_root": str(dataset_dir),
            "dataset_verification": canonical_verification,
            "paths": paths.to_dict(),
        }
    semantic_data = build_semantic_dataset_from_canonical(dataset_dir, output_root / "semantic_training")
    resume_checkpoint = detect_resume_checkpoint(paths.checkpoints, resume_from=resume_from)
    runtime = _safe_runtime_report(dataset_dir=dataset_dir, output_root=output_root)
    training_plan = build_training_plan(dataset_dir=Path(semantic_data["semantic_output_root"]), output_root=output_root)
    smoke_corpus = _build_smoke_corpus(Path(semantic_data["semantic_output_root"]), output_root)
    smoke_training = _run_real_smoke_training(
        base_model=str(training_plan["base_model"]),
        smoke_root=Path(smoke_corpus["root"]),
        output_root=output_root,
        resume_from=resume_from,
    )
    safe_report_path = paths.reports / "final_report.json"
    safe_metrics_path = paths.metrics / "semantic_metrics.json"
    smoke_report_path = paths.reports / "smoke_training_report.json"
    artifact_manifest_path = paths.manifests / "artifact_manifest.json"
    _write_json(
        safe_report_path,
        {
            "runtime": runtime,
            "training_plan": training_plan,
            "canonical_dataset_root": str(dataset_dir),
            "semantic_dataset_root": semantic_data["semantic_output_root"],
            "canonical_row_counts": semantic_data["bundle_report"].get("train_count", 0) + semantic_data["bundle_report"].get("validation_count", 0) + semantic_data["bundle_report"].get("test_count", 0),
            "semantic_row_count": semantic_data["semantic_row_count"],
            "smoke_training": smoke_training,
        },
    )
    _write_json(safe_metrics_path, semantic_data["readiness"])
    _write_json(smoke_report_path, smoke_training["smoke_training_report"])
    artifact_manifest = build_artifact_manifest([safe_report_path, safe_metrics_path, smoke_report_path])
    _write_json(artifact_manifest_path, artifact_manifest)
    final_zip = create_final_zip(paths.root, [safe_report_path, safe_metrics_path, smoke_report_path, artifact_manifest_path], zip_name="semantic_extractor_artifacts.zip")
    result = {
        "environment": runtime["environment"],
        "paths": paths.to_dict(),
        "canonical_dataset_root": str(dataset_dir),
        "semantic_dataset_root": semantic_data["semantic_output_root"],
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
    return result


def main() -> int:
    result = run_notebook_flow()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
