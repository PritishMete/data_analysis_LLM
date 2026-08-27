from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import random
import secrets
import string
from typing import Any

from .dataset import validate_dataset, verify_dataset_manifest, write_dataset_manifest
from .hardware import detect_hardware
from .metrics import TrainingMetrics
from .model_loader import DEFAULT_PROTOTYPE_MODEL
from .promotion import evaluate_promotion_gates
from .qlora import QLoRAConfig
from .profiles import select_model_profile, select_runtime_profile, choose_backend, PLANNER_BACKEND_AUTO, PLANNER_BACKEND_TRANSFORMERS, PLANNER_BACKEND_LLAMA_CPP


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_id(prefix: str = "exp") -> str:
    alphabet = string.ascii_lowercase + string.digits
    suffix = "".join(secrets.choice(alphabet) for _ in range(10))
    return f"{prefix}-{suffix}"


def create_experiment_id(prefix: str = "experiment") -> str:
    return _safe_id(prefix)


@dataclass(slots=True)
class SeedBundle:
    python_seed: int
    numpy_seed: int
    pytorch_seed: int
    trainer_seed: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def derive_seed_bundle(seed: int | None = None) -> SeedBundle:
    base = int(seed if seed is not None else random.SystemRandom().randrange(1, 2**31 - 1))
    return SeedBundle(
        python_seed=base,
        numpy_seed=base + 1,
        pytorch_seed=base + 2,
        trainer_seed=base + 3,
    )


@dataclass(slots=True)
class PreflightResult:
    ready: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _disk_free_gb(path: Path) -> float | None:
    try:
        usage = os.statvfs(path)
        return round((usage.f_bavail * usage.f_frsize) / 1024**3, 2)
    except Exception:
        return None


def _bf16_supported() -> bool | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(0)
        return (major, minor) >= (8, 0)
    except Exception:
        return None


def preflight_gpu_training(
    *,
    dataset_dir: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    minimum_vram_gb: float = 12.0,
    minimum_disk_gb: float = 40.0,
    planner_profile: str | None = None,
) -> PreflightResult:
    hardware = detect_hardware().to_dict()
    dataset = validate_dataset(dataset_dir)
    warnings: list[str] = []
    blockers: list[str] = []
    profile = select_model_profile(planner_profile)

    if not hardware.get("cuda_available"):
        blockers.append("cuda_unavailable")
    profile_min_vram = float(profile.training_min_vram_gb)
    required_vram = max(float(minimum_vram_gb), profile_min_vram)
    if hardware.get("vram_gb") is None or float(hardware.get("vram_gb") or 0.0) < required_vram:
        blockers.append("vram_below_threshold")

    free_disk_gb = _disk_free_gb(output_dir if output_dir.exists() else output_dir.parent)
    if free_disk_gb is not None and free_disk_gb < minimum_disk_gb:
        warnings.append(f"low_disk_space:{free_disk_gb:.2f}GB")
    bf16_supported = _bf16_supported()
    if bf16_supported is False:
        warnings.append("bf16_unavailable")
    if not dataset.ready_for_prototype:
        blockers.extend([f"dataset:{blocker}" for blocker in dataset.blockers])

    if manifest_path is not None:
        verification = verify_dataset_manifest(dataset_dir, manifest_path)
        if not verification["verified"]:
            blockers.append("dataset_hash_mismatch")
    else:
        write_dataset_manifest(dataset_dir)

    return PreflightResult(
        ready=not blockers,
        blockers=blockers,
        warnings=warnings,
        hardware={
            **hardware,
            "free_disk_gb": free_disk_gb,
            "bf16_supported": bf16_supported,
            "planner_profile": profile.profile_name,
            "training_min_vram_gb": profile.training_min_vram_gb,
            "training_recommended_vram_gb": profile.training_recommended_vram_gb,
            "inference_min_ram_gb": profile.inference_min_ram_gb,
            "inference_recommended_ram_gb": profile.inference_recommended_ram_gb,
            "inference_gpu_vram_gb": profile.inference_gpu_vram_gb,
        },
        dataset=dataset.to_dict(),
    )


def recommended_oom_actions() -> list[str]:
    return [
        "reduce max sequence length",
        "reduce batch size",
        "increase gradient accumulation",
        "enable gradient checkpointing",
        "reduce LoRA rank if necessary",
    ]


@dataclass(slots=True)
class ExperimentRunMetadata:
    experiment_id: str
    experiment_dir: Path
    dataset_dir: Path
    output_dir: Path
    base_model: str
    qlora: dict[str, Any]
    hardware: dict[str, Any]
    dataset_manifest_hash: str
    dataset_verification: dict[str, Any]
    seeds: SeedBundle
    created_at: str
    resume_from_checkpoint: str | None = None
    hf_home: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experiment_dir"] = str(self.experiment_dir)
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["seeds"] = self.seeds.to_dict()
        return payload


@dataclass(slots=True)
class ExperimentSummary:
    experiment_id: str
    base_model: str
    dataset_manifest_hash: str
    qlora_config: dict[str, Any]
    hardware: dict[str, Any]
    duration_seconds: float | None
    best_checkpoint: str | None
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    promotion_result: str
    warnings: list[str] = field(default_factory=list)
    status: str = "rejected"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def init_experiment(
    *,
    dataset_dir: Path,
    model_output_dir: Path,
    base_model: str,
    seed: int | None = None,
    resume_from_checkpoint: str | None = None,
    hf_home: str | None = None,
) -> ExperimentRunMetadata:
    experiment_id = create_experiment_id()
    experiment_dir = model_output_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest_info = verify_dataset_manifest(dataset_dir, dataset_dir / "dataset_manifest.sha256.json")
    if not manifest_info["verified"]:
        raise ValueError("dataset_hash_mismatch")
    hardware = detect_hardware().to_dict()
    seeds = derive_seed_bundle(seed)
    qlora = QLoRAConfig(model_id=base_model).to_dict()
    metadata = ExperimentRunMetadata(
        experiment_id=experiment_id,
        experiment_dir=experiment_dir,
        dataset_dir=dataset_dir,
        output_dir=model_output_dir,
        base_model=base_model,
        qlora=qlora,
        hardware=hardware,
        dataset_manifest_hash=_hash_text(_safe_json(manifest_info)),
        dataset_verification=manifest_info,
        seeds=seeds,
        created_at=_utcnow(),
        resume_from_checkpoint=resume_from_checkpoint,
        hf_home=hf_home,
    )
    (experiment_dir / "config.json").write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def record_safe_metrics(experiment_dir: Path, *, step: int, epoch: float, training_loss: float, validation_loss: float | None, learning_rate: float, vram_usage_gb: float | None, elapsed_seconds: float) -> Path:
    path = experiment_dir / "progress.jsonl"
    payload = {
        "step": step,
        "epoch": epoch,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        "learning_rate": learning_rate,
        "vram_usage_gb": vram_usage_gb,
        "elapsed_seconds": elapsed_seconds,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def write_experiment_summary(
    *,
    experiment_dir: Path,
    base_model: str,
    dataset_manifest_hash: str,
    qlora_config: dict[str, Any],
    hardware: dict[str, Any],
    duration_seconds: float | None,
    best_checkpoint: str | None,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    promotion_result: str,
    status: str,
    warnings: list[str] | None = None,
) -> ExperimentSummary:
    summary = ExperimentSummary(
        experiment_id=experiment_dir.name,
        base_model=base_model,
        dataset_manifest_hash=dataset_manifest_hash,
        qlora_config=qlora_config,
        hardware=hardware,
        duration_seconds=duration_seconds,
        best_checkpoint=best_checkpoint,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        promotion_result=promotion_result,
        warnings=list(warnings or []),
        status=status,
    )
    (experiment_dir / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def decide_final_promotion(*, readiness: dict[str, Any], metrics: dict[str, Any]) -> str:
    gate = evaluate_promotion_gates(readiness=readiness, metrics=metrics)
    if not readiness.get("ready_for_prototype"):
        return "TRAINING_FAILED"
    if gate.promotable:
        return "PROMOTE_TO_SHADOW"
    return "REJECT_MODEL"


def update_shadow_registry_status(status: str) -> str:
    if status == "PROMOTE_TO_SHADOW":
        return "shadow"
    if status == "REJECT_MODEL":
        return "rejected"
    return "failed"


def select_model_and_runtime_profile(
    *,
    planner_profile: str | None = None,
    runtime_profile: str | None = None,
    backend: str | None = None,
    cuda_available: bool = False,
    llama_cpp_available: bool = False,
) -> dict[str, Any]:
    model_profile = select_model_profile(planner_profile)
    runtime = select_runtime_profile(runtime_profile)
    selected_backend = choose_backend(
        backend=backend,
        runtime_profile=runtime,
        cuda_available=cuda_available,
        llama_cpp_available=llama_cpp_available,
    )
    return {
        "planner_profile": model_profile.to_dict(),
        "runtime_profile": runtime.to_dict(),
        "backend": selected_backend,
        "backend_policy": {
            "requested": (backend or PLANNER_BACKEND_AUTO),
            "auto_priority": list(runtime.backend_preference),
            "supported_backends": [PLANNER_BACKEND_TRANSFORMERS, PLANNER_BACKEND_LLAMA_CPP],
        },
    }
