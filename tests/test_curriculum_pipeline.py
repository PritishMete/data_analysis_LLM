from __future__ import annotations

import json
from pathlib import Path

from curriculum.analytics_curriculum import run_curriculum


SENSITIVE_VALUES = [
    "John Smith",
    "john@example.com",
    "ACC-9988",
    "SecretCompanyXYZ",
    "9876543210",
]


def _assert_no_sensitive_values(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for value in SENSITIVE_VALUES:
            assert value not in text, f"found sensitive value {value!r} in {path}"


def test_curriculum_runner_exports_safe_deduped_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("TEACHER_REPO_ROOT", r"E:\teacher_ref")

    runtime_root = tmp_path / "runtime"
    training_dir = tmp_path / "training"
    docs_path = tmp_path / "docs" / "analytics_curriculum_report.md"
    report_path = runtime_root / "report.json"

    result = run_curriculum(
        runtime_root=runtime_root,
        training_export_dir=training_dir,
        docs_path=docs_path,
        report_path=report_path,
        variants_per_family=2,
        max_examples_per_fingerprint=1,
    )

    report = result.training_report
    assert result.privacy_verified is True
    assert result.restart_verified is True
    assert report["eligible_examples"] == 100
    assert report["family_count"] == 100
    assert report["intent_count"] == 14
    assert report["duplicates_removed"] == 0
    assert report["rejected_examples"] == 10
    assert report["rejection_reasons"]["critic_failed"] == 10
    assert report["train_count"] + report["validation_count"] + report["test_count"] == report["eligible_examples"]

    export_files = [
        runtime_root / "report.json",
        docs_path,
        training_dir / "train.jsonl",
        training_dir / "validation.jsonl",
        training_dir / "test.jsonl",
        training_dir / "dataset_report.json",
        training_dir / "dataset_manifest.json",
    ]
    _assert_no_sensitive_values(export_files)

    splits: dict[str, set[str]] = {}
    for split_name in ("train", "validation", "test"):
        path = training_dir / f"{split_name}.jsonl"
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            splits.setdefault(str(record["family_fingerprint"]), set()).add(split_name)

    assert splits
    assert all(len(split_names) == 1 for split_names in splits.values())
