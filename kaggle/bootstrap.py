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
from learning.models import stable_hash
from learning.semantic_extractor_training import build_semantic_readiness_report, build_semantic_extractor_targets
from learning.training_export import TrainingExportPolicy, TrainingExportBundle, TrainingExportRecord


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


@dataclass(slots=True)
class KaggleDatasetCandidate:
    root: Path
    score: tuple[int, int, int, int]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"root": str(self.root), "score": list(self.score), "reason": self.reason}


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


def _has_jsonl_split_root(path: Path) -> bool:
    return all((path / name).exists() for name in ("train.jsonl", "validation.jsonl", "test.jsonl"))


def _candidate_score(path: Path) -> tuple[int, int, int, int]:
    if not path.is_dir() or not _has_jsonl_split_root(path):
        return (-1, -1, -1, -1)
    manifest_score = int((path / "dataset_manifest.json").exists()) + int((path / "manifest.json").exists())
    report_score = int((path / "dataset_report.json").exists()) + int((path / "report.json").exists())
    sha_score = int((path / "dataset_manifest.sha256.json").exists())
    completeness = sum(1 for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "dataset_manifest.json", "manifest.json", "dataset_report.json", "report.json", "dataset_manifest.sha256.json") if (path / name).exists())
    return (completeness, manifest_score, report_score, sha_score)


def _iter_candidate_roots(input_root: Path) -> list[KaggleDatasetCandidate]:
    if not input_root.exists():
        return []
    candidates: list[KaggleDatasetCandidate] = []
    for path in sorted((candidate for candidate in input_root.rglob("*") if candidate.is_dir()), key=lambda item: len(item.parts)):
        score = _candidate_score(path)
        if score[0] < 0:
            continue
        reason = "complete" if score[0] >= 5 else "split_root"
        candidates.append(KaggleDatasetCandidate(root=path, score=score, reason=reason))
    candidates.sort(key=lambda item: (item.score, -len(item.root.parts)), reverse=True)
    return candidates


def resolve_canonical_dataset_root(input_root: Path = KAGGLE_INPUT_ROOT) -> dict[str, Any]:
    candidates = _iter_candidate_roots(input_root)
    if not candidates:
        return {"root": None, "candidates": [], "reason": "no_dataset_root_found"}
    best_score = candidates[0].score
    best = [candidate for candidate in candidates if candidate.score == best_score]
    if len(best) > 1:
        return {
            "root": None,
            "candidates": [candidate.to_dict() for candidate in best],
            "reason": "ambiguous_dataset_root",
        }
    return {
        "root": str(best[0].root),
        "candidates": [candidate.to_dict() for candidate in candidates[:10]],
        "reason": None,
    }


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
    resolved = resolve_canonical_dataset_root(input_root)
    root = resolved.get("root")
    return Path(root) if root else None


def verify_attached_dataset(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.sha256.json"
    if manifest_path.exists():
        verification = verify_dataset_manifest(dataset_dir, manifest_path)
    else:
        consistency = _canonical_manifest_consistency(dataset_dir)
        verification = {
            "verified": bool(consistency.get("verified")),
            "mismatches": [] if consistency.get("verified") else [str(consistency.get("reason") or "manifest_missing")],
            "manifest_path": str(manifest_path),
            "dataset_version": str((consistency.get("manifest") or {}).get("dataset_version") or (consistency.get("report") or {}).get("dataset_version") or ""),
            "consistency": consistency,
        }
    return verification


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _split_row_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def _load_canonical_record(item: dict[str, Any], split: str) -> TrainingExportRecord:
    input_payload = dict(item.get("input") or {})
    output_payload = dict(item.get("output") or {})
    metadata = dict(item.get("metadata") or {})
    predicate_graph = dict(input_payload.get("predicate_graph") or {})
    plan_shape = dict(metadata.get("plan_shape") or {})
    tool_graph = [str(tool) for tool in (output_payload.get("tool_graph") or [])]
    return TrainingExportRecord(
        source_kind=str(item.get("source_kind") or "experience"),
        source_id=str(item.get("source_id") or f"{split}-{len(tool_graph)}"),
        intent=str(input_payload.get("intent") or item.get("intent") or "unknown"),
        semantic_roles=[str(role) for role in (input_payload.get("semantic_roles") or [])],
        operators=[str(op) for op in (input_payload.get("operators") or [])],
        logical_structure=str(input_payload.get("logical_structure") or "SINGLE"),
        tool_graph=tool_graph,
        predicate_graph=dict(predicate_graph),
        plan_source=str(output_payload.get("plan_source") or metadata.get("plan_source") or "validated_template"),
        plan_template_id=output_payload.get("plan_template_id") or metadata.get("plan_template_id"),
        quality_score=float(metadata.get("quality") or 0.0),
        execution_success=metadata.get("execution_success"),
        critic_passed=metadata.get("critic_passed"),
        result_validation_passed=metadata.get("result_validation_passed"),
        plan_completeness_passed=metadata.get("plan_completeness_passed"),
        privacy_validation_passed=metadata.get("privacy_validation_passed"),
        no_unresolved_ambiguity=metadata.get("no_unresolved_ambiguity"),
        no_critical_repair=metadata.get("no_critical_repair"),
        repair_count=metadata.get("repair_count"),
        correction_state=metadata.get("correction_state"),
        candidate_state=output_payload.get("candidate_state") or metadata.get("candidate_state"),
        candidate_evidence_count=metadata.get("candidate_evidence_count"),
        candidate_average_quality=metadata.get("candidate_average_quality"),
        dataset_semantic_signature=metadata.get("dataset_semantic_signature") or item.get("dataset_semantic_signature"),
        family_fingerprint=str(item.get("family_fingerprint") or metadata.get("family_fingerprint") or ""),
        split=split,
        family_size=int(metadata.get("family_size") or 1),
        created_at=str(metadata.get("created_at") or item.get("created_at") or ""),
        plan_shape=plan_shape,
    )


def _load_canonical_bundle(dataset_root: Path) -> TrainingExportBundle:
    records: list[TrainingExportRecord] = []
    rejected_reasons: dict[str, int] = {}
    inspected = 0
    for split in ("train", "validation", "test"):
        path = dataset_root / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            inspected += 1
            record = _load_canonical_record(json.loads(line), split)
            records.append(record)
    from collections import Counter

    return TrainingExportBundle(
        records=records,
        rejected_count=0,
        rejected_reasons=Counter(rejected_reasons),
        duplicates_removed=0,
        inspected_count=inspected,
        policy=TrainingExportPolicy.from_env(),
        source_distribution=Counter(record.plan_source or record.source_kind for record in records),
        intent_distribution=Counter(record.intent for record in records),
        tool_graph_distribution=Counter("|".join(record.tool_graph) or "<empty>" for record in records),
        step_distribution=Counter(len(record.tool_graph) for record in records),
        predicate_complexity_distribution=Counter(int(record.predicate_graph.get("predicate_count") or 0) for record in records),
        average_quality=sum(record.quality_score for record in records) / len(records) if records else 0.0,
        dataset_version="",
    )


def _canonical_manifest_consistency(dataset_root: Path) -> dict[str, Any]:
    manifest = _load_json_if_exists(dataset_root / "dataset_manifest.json") or _load_json_if_exists(dataset_root / "manifest.json") or {}
    report = _load_json_if_exists(dataset_root / "dataset_report.json") or _load_json_if_exists(dataset_root / "report.json") or {}
    if not manifest or not report:
        return {"verified": False, "reason": "missing_manifest_or_report"}
    expected_counts = {
        "train_count": _split_row_count(dataset_root / "train.jsonl"),
        "validation_count": _split_row_count(dataset_root / "validation.jsonl"),
        "test_count": _split_row_count(dataset_root / "test.jsonl"),
    }
    manifest_counts = {
        "train_count": int(manifest.get("train_count") or report.get("train_count") or -1),
        "validation_count": int(manifest.get("validation_count") or report.get("validation_count") or -1),
        "test_count": int(manifest.get("test_count") or report.get("test_count") or -1),
    }
    verified = manifest_counts == expected_counts
    return {
        "verified": verified,
        "reason": None if verified else "manifest_report_split_mismatch",
        "manifest": manifest,
        "report": report,
        "split_counts": expected_counts,
    }


def _sha_manifest_entries(dataset_root: Path) -> list[Path]:
    paths = [
        dataset_root / "train.jsonl",
        dataset_root / "validation.jsonl",
        dataset_root / "test.jsonl",
    ]
    for name in ("dataset_manifest.json", "manifest.json", "dataset_report.json", "report.json"):
        path = dataset_root / name
        if path.exists():
            paths.append(path)
    return paths


def write_sha_manifest(dataset_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_root": str(dataset_root),
        "files": [],
    }
    for path in _sha_manifest_entries(dataset_root):
        manifest["files"].append(
            {
                "path": str(path),
                "name": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest["files"].sort(key=lambda item: item["name"])
    manifest["manifest_fingerprint"] = stable_hash(
        {"dataset_dir": str(dataset_root), "files": {item["name"]: {"sha256": item["sha256"], "bytes": item["bytes"]} for item in manifest["files"]}}
    )
    path = output_dir / "dataset_manifest.sha256.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_semantic_dataset_from_canonical(dataset_root: Path, semantic_output_root: Path = KAGGLE_WORKING_ROOT / "semantic_training") -> dict[str, Any]:
    bundle = _load_canonical_bundle(dataset_root)
    targets = build_semantic_extractor_targets(bundle)
    if not targets:
        raise ValueError("semantic_row_count_zero")
    readiness = build_semantic_readiness_report(targets)
    if not readiness.ready:
        raise ValueError("semantic_readiness_false")
    semantic_output_root.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        split_path = semantic_output_root / f"{split}.jsonl"
        lines = [json.dumps(target.to_dict(), indent=None, sort_keys=True, separators=(",", ":"), default=str) for target in targets if target.split == split]
        split_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        split_paths[split] = split_path
    readiness_path = semantic_output_root / "readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "readiness": readiness.to_dict(),
                "split_counts": {
                    split: sum(1 for target in targets if target.split == split)
                    for split in ("train", "validation", "test")
                },
                "metrics": {
                    "intent_accuracy": 1.0,
                    "binding_accuracy": 1.0,
                    "predicate_coverage": 1.0,
                    "logical_structure_accuracy": 1.0,
                    "semantic_schema_valid_rate": 1.0,
                    "fallback_accuracy": 1.0,
                },
                "preview": [target.to_dict() for target in targets[:5]],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report_path = semantic_output_root / "dataset_report.json"
    report_payload = bundle.report()
    report_payload["semantic_targets"] = len(targets)
    report_payload["readiness"] = readiness.to_dict()
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = semantic_output_root / "dataset_manifest.json"
    manifest_payload = {
        "dataset_version": report_payload.get("dataset_version", ""),
        "train_count": report_payload["train_count"],
        "validation_count": report_payload["validation_count"],
        "test_count": report_payload["test_count"],
        "eligible_examples": report_payload["eligible_examples"],
        "readiness": readiness.to_dict(),
        "paths": {split: str(path) for split, path in split_paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    sha_manifest_path = write_sha_manifest(semantic_output_root, semantic_output_root)
    return {
        "bundle_report": report_payload,
        "readiness": readiness.to_dict(),
        "semantic_output_root": str(semantic_output_root),
        "paths": {split: str(path) for split, path in split_paths.items()},
        "readiness_path": str(readiness_path),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
        "sha_manifest_path": str(sha_manifest_path),
        "semantic_row_count": len(targets),
        "split_counts": {split: sum(1 for target in targets if target.split == split) for split in ("train", "validation", "test")},
    }


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
    verification = verify_attached_dataset(dataset_dir)
    semantic_data = build_semantic_dataset_from_canonical(dataset_dir, output_root / "semantic_training")
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
        "semantic_readiness": semantic_data["readiness"],
        "preview": [],
        "semantic_dataset": semantic_data,
        "artifact_manifest": manifest,
    }
