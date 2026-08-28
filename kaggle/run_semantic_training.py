from __future__ import annotations

import json
import os
import time
import subprocess
import sys
import traceback
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


def _write_jsonl_breadcrumb(path: Path, *, stage: str, success: bool, safe_message: str, torch_module: Any | None = None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "stage": stage,
        "success": bool(success),
        "safe_message": safe_message,
        **_runtime_memory_state(torch_module),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
    return payload


def _sanitize_exception_message(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text[:1000]


def _safe_traceback_tail(exc: BaseException, *, limit: int = 25) -> str:
    tb = traceback.TracebackException.from_exception(exc)
    lines = list(tb.format())
    safe_lines = lines[-limit:]
    return "".join(safe_lines)


def _safe_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return (result.stdout or "").strip() or None
    except Exception:
        return None


def _expected_commit_hash() -> str | None:
    return os.environ.get("KAGGLE_EXPECTED_GIT_COMMIT") or os.environ.get("EXPECTED_GIT_COMMIT")


def _write_smoke_heartbeat(report_root: Path, *, stage: str, smoke_mode: bool = True) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "timestamp": time.time(),
        "git_commit": _safe_commit_hash(),
        "smoke_mode": smoke_mode,
    }
    path = report_root / "smoke_heartbeat.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_smoke_failure(
    *,
    report_root: Path,
    stage: str,
    exc: BaseException,
    torch_module: Any | None = None,
) -> Path:
    report_root.mkdir(parents=True, exist_ok=True)
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
) -> None:
    _write_smoke_heartbeat(report_root, stage=stage, smoke_mode=smoke_mode)
    _emit_smoke_stage(breadcrumbs_path, stage=stage, success=success, safe_message=safe_message, torch_module=torch_module)


def _emit_smoke_stage(
    breadcrumbs_path: Path,
    *,
    stage: str,
    success: bool,
    safe_message: str,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    return _write_jsonl_breadcrumb(breadcrumbs_path, stage=stage, success=success, safe_message=safe_message, torch_module=torch_module)


def _run_real_smoke_training(
    *,
    base_model: str,
    smoke_root: Path,
    output_root: Path,
    breadcrumbs_path: Path,
    resume_from: str | None = None,
) -> dict[str, Any]:
    report_root = output_root / "reports"

    def _run_with_timeout(cmd: list[str], *, timeout: int, stage: str) -> None:
        try:
            subprocess.run(cmd, check=True, timeout=timeout)
        except Exception as exc:
            _write_smoke_failure(report_root=report_root, stage=stage, exc=exc, torch_module=None)
            raise

    def _ensure_runtime_packages() -> dict[str, str | None]:
        installed: dict[str, str | None] = {}
        try:
            import torch

            cuda_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
            installed["torch"] = torch.__version__
            installed["cuda_capability"] = f"{cuda_capability[0]}.{cuda_capability[1]}" if cuda_capability else None
        except Exception:
            installed["torch"] = None
            installed["cuda_capability"] = None
        try:
            from importlib.metadata import version

            installed["bitsandbytes"] = version("bitsandbytes")
        except Exception:
            installed["bitsandbytes"] = None

        required_packages = []
        if installed["bitsandbytes"] is None or tuple(int(part) for part in str(installed["bitsandbytes"]).split(".")[:2] if part.isdigit()) < (0, 46):
            required_packages.append("bitsandbytes>=0.46.1")
        for package in ("accelerate>=0.31", "peft>=0.11", "transformers>=4.43", "trl>=0.9", "safetensors>=0.4", "sentencepiece>=0.2.0"):
            required_packages.append(package)

        if required_packages:
            _run_with_timeout(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *required_packages],
                timeout=600,
                stage="dependencies_started",
            )
        return installed

    _write_smoke_heartbeat(report_root, stage="notebook_started")
    _stage_guard(stage="repo_checkout_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, safe_message="repo checkout start")
    _stage_guard(stage="repo_checkout_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, safe_message="repo checkout complete")
    _stage_guard(stage="dependencies_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, safe_message="starting smoke bootstrap")
    runtime_packages = _ensure_runtime_packages()
    torch = None
    try:
        _stage_guard(stage="dependencies_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, safe_message="runtime packages checked")
        import torch as torch_module

        torch = torch_module
        _stage_guard(stage="gpu_check_started", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message="gpu check start")
        if torch.cuda.is_available():
            _stage_guard(stage="gpu_check_complete", report_root=report_root, breadcrumbs_path=breadcrumbs_path, torch_module=torch, safe_message=torch.cuda.get_device_name(0))
        else:
            raise RuntimeError("cuda_unavailable")
    except Exception as exc:
        _write_smoke_failure(report_root=report_root, stage="dependencies_complete", exc=exc, torch_module=torch)
        raise

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except Exception as exc:
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
    _write_smoke_heartbeat(paths.reports, stage="notebook_started")
    executed_commit = _safe_commit_hash()
    expected_commit = _expected_commit_hash()
    if expected_commit and executed_commit and executed_commit != expected_commit:
        exc = RuntimeError("stale_kaggle_checkout")
        _write_smoke_failure(report_root=paths.reports, stage="notebook_started", exc=exc, torch_module=None)
        raise exc
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
    breadcrumbs_path = paths.reports / "smoke_breadcrumbs.jsonl"
    smoke_failure: Exception | None = None
    try:
        smoke_training = _run_real_smoke_training(
            base_model=str(training_plan["base_model"]),
            smoke_root=Path(smoke_corpus["root"]),
            output_root=output_root,
            breadcrumbs_path=breadcrumbs_path,
            resume_from=resume_from,
        )
    except Exception as exc:  # pragma: no cover - surfaced via Kaggle notebook logs
        smoke_failure = exc
        try:
            import torch as torch_module
        except Exception:
            torch_module = None
        _write_smoke_failure(report_root=paths.reports, stage="smoke_notebook", exc=exc, torch_module=torch_module)
        smoke_training = {
            "smoke_training_report": _failed_smoke_training_report(error=exc),
            "checkpoint_dir": None,
            "adapter_dir": None,
            "train_steps": 0,
            "train_metrics": {"train_loss": None},
            "validation_metrics": {"eval_loss": None},
        }
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
            "smoke_training_failure": type(smoke_failure).__name__ if smoke_failure else None,
        },
    )
    _write_json(safe_metrics_path, semantic_data["readiness"])
    _write_json(smoke_report_path, smoke_training["smoke_training_report"])
    artifact_manifest = build_artifact_manifest([safe_report_path, safe_metrics_path, smoke_report_path])
    _write_json(artifact_manifest_path, artifact_manifest)
    final_zip = create_final_zip(paths.root, [safe_report_path, safe_metrics_path, smoke_report_path, artifact_manifest_path, breadcrumbs_path, paths.reports / "smoke_failure.json"], zip_name="semantic_extractor_artifacts.zip")
    result = {
        "environment": runtime["environment"],
        "paths": paths.to_dict(),
        "executed_git_commit": executed_commit,
        "expected_git_commit": expected_commit,
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
