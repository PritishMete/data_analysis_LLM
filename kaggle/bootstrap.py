from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import platform
import re
import shutil
import sys
from typing import Any, Iterable
from zipfile import ZIP_DEFLATED, ZipFile

from src.training.dataset import verify_dataset_manifest
from src.training.hardware import detect_hardware
from learning.semantic_extractor_training import build_semantic_readiness_report, build_semantic_extractor_targets
from learning.training_export import TrainingDatasetExporter, TrainingExportPolicy
from insight_learning.api.app import get_service


KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
DEFAULT_SEMANTIC_CONFIG = Path("configs") / "qwen25_0_5b_semantic_qlora.yaml"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-\d+$")
SAFE_ZIP_NAMES = {
    "adapter",
    "artifact_manifest",
    "final_report",
    "metrics",
    "model_registry",
    "training_config",
}


@dataclass(slots=True)
class KaggleEnvironmentReport:
    python_version: str
    platform: str
    machine: str
    gpu_name: str | None
    cuda_available: bool
    cuda_version: str | None
    torch_version: str | None
    vram_gb: float | None
    free_disk_gb: float | None
    dataset_dir: str | None = None
    attached_datasets: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KagglePaths:
    root: Path
    checkpoints: Path
    adapters: Path
    reports: Path
    metrics: Path
    manifests: Path
    final_zip: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_free_disk_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / 1024**3, 2)
    except Exception:
        return None


def ensure_kaggle_paths(root: Path = KAGGLE_WORKING_ROOT) -> KagglePaths:
    root.mkdir(parents=True, exist_ok=True)
    checkpoints = root / "checkpoints"
    adapters = root / "adapters"
    reports = root / "reports"
    metrics = root / "metrics"
    manifests = root / "manifests"
    for path in (checkpoints, adapters, reports, metrics, manifests):
        path.mkdir(parents=True, exist_ok=True)
    return KagglePaths(
        root=root,
        checkpoints=checkpoints,
        adapters=adapters,
        reports=reports,
        metrics=metrics,
        manifests=manifests,
        final_zip=root / "semantic_extractor_artifacts.zip",
    )


def inspect_kaggle_environment() -> KaggleEnvironmentReport:
    hardware = detect_hardware()
    return KaggleEnvironmentReport(
        python_version=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        gpu_name=hardware.gpu_name,
        cuda_available=hardware.cuda_available,
        cuda_version=hardware.cuda_version,
        torch_version=hardware.torch_version,
        vram_gb=hardware.vram_gb,
        free_disk_gb=_safe_free_disk_gb(KAGGLE_WORKING_ROOT if KAGGLE_WORKING_ROOT.exists() else Path.cwd()),
        attached_datasets=[str(path) for path in discover_attached_dataset(KAGGLE_INPUT_ROOT)],
    )


def discover_attached_dataset(input_root: Path = KAGGLE_INPUT_ROOT) -> list[Path]:
    if not input_root.exists():
        return []
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def discover_semantic_dataset(input_root: Path = KAGGLE_INPUT_ROOT) -> Path | None:
    candidates = discover_attached_dataset(input_root)
    for candidate in candidates:
        if (candidate / "dataset_manifest.sha256.json").exists():
            return candidate
        if (candidate / "train.jsonl").exists() and (candidate / "validation.jsonl").exists() and (candidate / "test.jsonl").exists():
            return candidate
    return candidates[0] if candidates else None


def verify_attached_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.sha256.json"
    if manifest_path.exists():
        verification = verify_dataset_manifest(dataset_dir, manifest_path)
    else:
        verification = {"verified": False, "mismatches": ["manifest_missing"], "manifest_path": str(manifest_path), "dataset_version": ""}
    return verification


def load_semantic_config(config_path: Path = DEFAULT_SEMANTIC_CONFIG) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def detect_resume_checkpoint(output_root: Path, resume_from: str | None = None) -> Path | None:
    if resume_from:
        candidate = Path(resume_from)
        if candidate.exists():
            return candidate
        named = output_root / resume_from
        if named.exists():
            return named
    if not output_root.exists():
        return None
    latest = output_root / "last_checkpoint"
    if latest.exists():
        text = latest.read_text(encoding="utf-8").strip()
        if text:
            candidate = output_root / text
            if candidate.exists():
                return candidate
    checkpoints = [path for path in output_root.iterdir() if path.is_dir() and CHECKPOINT_PATTERN.fullmatch(path.name)]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda path: int(path.name.split("-", 1)[1]), reverse=True)
    return checkpoints[0]


def build_artifact_manifest(files: Iterable[Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        if not path.exists() or not path.is_file():
            continue
        entries.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "manifest_version": 1,
        "artifacts": sorted(entries, key=lambda item: item["name"]),
    }
    return manifest


def create_final_zip(output_dir: Path, artifact_paths: Iterable[Path], zip_name: str = "semantic_extractor_artifacts.zip") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / zip_name
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in artifact_paths:
            if not path.exists() or not path.is_file():
                continue
            if path.name not in SAFE_ZIP_NAMES:
                continue
            archive.write(path, arcname=path.name)
    return zip_path


def semantic_verdict(*, gate_results: dict[str, float], readiness: bool, fallback_rate: float) -> str:
    if not readiness:
        return "TRAINING_FAILED"
    if fallback_rate > 0.0 and gate_results.get("semantic_schema_valid_rate", 0.0) < 0.99:
        return "REJECT_SEMANTIC_EXTRACTOR"
    thresholds = {
        "intent_accuracy": 0.95,
        "binding_accuracy": 0.90,
        "predicate_coverage": 0.90,
        "logical_structure_accuracy": 0.90,
        "semantic_schema_valid_rate": 0.99,
        "fallback_accuracy": 0.95,
    }
    for key, minimum in thresholds.items():
        if float(gate_results.get(key, 0.0)) < minimum:
            return "REJECT_SEMANTIC_EXTRACTOR"
    return "PROMOTE_SEMANTIC_EXTRACTOR_TO_SHADOW"


def build_semantic_kaggle_report(*, dataset_dir: Path, output_root: Path = KAGGLE_WORKING_ROOT) -> dict[str, Any]:
    paths = ensure_kaggle_paths(output_root)
    service = get_service()
    exporter = TrainingDatasetExporter(service.store, TrainingExportPolicy.from_env())
    bundle, preview = exporter.build_bundle(include_candidate_strategies=True)
    targets = build_semantic_extractor_targets(bundle)
    readiness = build_semantic_readiness_report(targets)
    verification = verify_attached_dataset(dataset_dir)
    manifest = build_artifact_manifest(
        [
            output_root / "training_config.json",
            output_root / "final_report.json",
        ]
    )
    return {
        "paths": paths.to_dict(),
        "dataset_dir": str(dataset_dir),
        "manifest_verification": verification,
        "semantic_readiness": readiness.to_dict(),
        "preview": preview[:5],
        "artifact_manifest": manifest,
    }
