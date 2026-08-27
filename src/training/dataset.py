from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json

from learning.canonical_training import TrainingReadinessAssessment
from learning.training_export import TrainingDatasetExporter


@dataclass(slots=True)
class DatasetValidationResult:
    ready_for_prototype: bool
    dataset_version: str
    report: dict[str, Any]
    readiness: dict[str, Any]
    blockers: list[str]
    files: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_for_prototype": self.ready_for_prototype,
            "dataset_version": self.dataset_version,
            "report": self.report,
            "readiness": self.readiness,
            "blockers": list(self.blockers),
            "files": {key: str(value) for key, value in self.files.items()},
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_sensitive_literals(text: str) -> bool:
    forbidden = [
        "John Smith",
        "john@example.com",
        "ACC-9988",
        "SecretCompanyXYZ",
        "9876543210",
        "query_text",
        "normalized_query",
        "workbook",
        "sheet_name",
        "filename",
        "customer name",
    ]
    lowered = text.lower()
    return any(item.lower() in lowered for item in forbidden)


def validate_dataset(dataset_dir: Path) -> DatasetValidationResult:
    manifest_path = dataset_dir / "dataset_manifest.json"
    report_path = dataset_dir / "dataset_report.json"
    files = {
        "train": dataset_dir / "train.jsonl",
        "validation": dataset_dir / "validation.jsonl",
        "test": dataset_dir / "test.jsonl",
        "manifest": manifest_path,
        "report": report_path,
    }
    blockers: list[str] = []
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        blockers.append(f"missing_files:{','.join(missing)}")
        return DatasetValidationResult(False, "", {}, {}, blockers, files)

    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    readiness_payload = dict(
        manifest.get("readiness")
        or report.get("readiness")
        or {}
    )
    readiness = readiness_payload
    if not readiness.get("ready_for_prototype"):
        blockers.append(str(readiness.get("reason") or "not_ready"))

    for split_name in ("train", "validation", "test"):
        if _contains_sensitive_literals(files[split_name].read_text(encoding="utf-8", errors="ignore")):
            blockers.append(f"privacy_violation:{split_name}")

    if manifest.get("eligible_examples") != report.get("eligible_examples"):
        blockers.append("manifest_report_mismatch")
    if manifest.get("readiness", {}).get("ready_for_prototype") is False:
        blockers.append("manifest_not_ready")
    if report.get("split_integrity_passed") is False:
        blockers.append("split_integrity_failed")
    if not report.get("privacy_rejections", 0) >= 0:
        blockers.append("privacy_gate_failed")

    return DatasetValidationResult(
        ready_for_prototype=not blockers,
        dataset_version=str(report.get("dataset_version") or manifest.get("dataset_version") or ""),
        report=report,
        readiness=readiness,
        blockers=blockers,
        files=files,
    )
