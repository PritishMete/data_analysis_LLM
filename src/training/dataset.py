from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from typing import Any


SPLITS = ("train", "validation", "test")


@dataclass(slots=True)
class DatasetValidationResult:
    ready_for_prototype: bool
    dataset_version: str
    report: dict[str, Any]
    readiness: dict[str, Any]
    blockers: list[str]
    files: dict[str, Path]
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_for_prototype": self.ready_for_prototype,
            "dataset_version": self.dataset_version,
            "report": self.report,
            "readiness": self.readiness,
            "blockers": list(self.blockers),
            "files": {key: str(value) for key, value in self.files.items()},
            "split_counts": dict(self.split_counts),
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def build_manifest_fingerprint(dataset_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"dataset_dir": str(dataset_dir), "files": {}}
    for name in (*SPLITS, "dataset_report.json"):
        file_path = dataset_dir / f"{name}.jsonl" if name in SPLITS else dataset_dir / name
        if file_path.exists():
            payload["files"][name] = {"sha256": sha256_file(file_path), "bytes": file_path.stat().st_size}
    return payload


def create_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest = {
        "dataset_version": "",
        "created_at": None,
        "fingerprint": build_manifest_fingerprint(dataset_dir),
        "files": {},
    }
    for name in (*SPLITS, "dataset_report"):
        path = dataset_dir / (f"{name}.jsonl" if name in SPLITS else "dataset_report.json")
        if path.exists():
            manifest["files"][name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return manifest


def write_dataset_manifest(dataset_dir: Path, manifest_path: Path | None = None) -> Path:
    manifest_path = manifest_path or (dataset_dir / "dataset_manifest.sha256.json")
    manifest = create_dataset_manifest(dataset_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def verify_dataset_manifest(dataset_dir: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path or (dataset_dir / "dataset_manifest.sha256.json")
    manifest = _load_json(manifest_path)
    mismatches: list[str] = []
    files = manifest.get("files") or {}
    for name, meta in files.items():
        path = Path(meta.get("path") or dataset_dir / f"{name}.jsonl")
        if not path.exists():
            mismatches.append(f"missing:{name}")
            continue
        expected = str(meta.get("sha256") or "")
        actual = sha256_file(path)
        if expected != actual:
            mismatches.append(f"sha256:{name}")
    return {"verified": not mismatches, "mismatches": mismatches, "manifest_path": str(manifest_path), "dataset_version": manifest.get("dataset_version", "")}


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
    sha_manifest_path = dataset_dir / "dataset_manifest.sha256.json"
    files = {
        "train": dataset_dir / "train.jsonl",
        "validation": dataset_dir / "validation.jsonl",
        "test": dataset_dir / "test.jsonl",
        "manifest": manifest_path,
        "report": report_path,
        "sha_manifest": sha_manifest_path,
    }
    blockers: list[str] = []
    missing = [name for name, path in files.items() if name != "sha_manifest" and not path.exists()]
    if missing:
        blockers.append(f"missing_files:{','.join(missing)}")
    if not manifest_path.exists() or not report_path.exists():
        return DatasetValidationResult(False, "", {}, {}, blockers, files, {split: 0 for split in SPLITS})

    report = _load_json(report_path)
    manifest = _load_json(manifest_path)
    readiness = dict(manifest.get("readiness") or report.get("readiness") or {})
    if not readiness.get("ready_for_prototype"):
        blockers.append(str(readiness.get("reason") or "not_ready"))

    split_counts = {}
    for split in SPLITS:
        path = files[split]
        split_counts[split] = sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        if _contains_sensitive_literals(path.read_text(encoding="utf-8", errors="ignore")):
            blockers.append(f"privacy_violation:{split}")

    if manifest.get("eligible_examples") != report.get("eligible_examples"):
        blockers.append("manifest_report_mismatch")
    if manifest.get("train_count") != split_counts["train"] or manifest.get("validation_count") != split_counts["validation"] or manifest.get("test_count") != split_counts["test"]:
        blockers.append("split_count_mismatch")
    if report.get("split_integrity_passed") is False:
        blockers.append("split_integrity_failed")

    if sha_manifest_path.exists():
        verification = verify_dataset_manifest(dataset_dir, sha_manifest_path)
        if not verification["verified"]:
            blockers.append("sha_manifest_mismatch")

    return DatasetValidationResult(
        ready_for_prototype=not blockers,
        dataset_version=str(report.get("dataset_version") or manifest.get("dataset_version") or ""),
        report=report,
        readiness=readiness,
        blockers=blockers,
        files=files,
        split_counts=split_counts,
    )
