from __future__ import annotations

import json
from pathlib import Path

from kaggle.bootstrap import (
    build_artifact_manifest,
    create_final_zip,
    detect_resume_checkpoint,
    discover_attached_dataset,
    discover_semantic_dataset,
    ensure_kaggle_paths,
    semantic_verdict,
)
from kaggle.run_semantic_training import build_training_plan


def test_kaggle_dataset_discovery_and_output_paths(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    dataset_dir = input_root / "semantic-dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "train.jsonl").write_text("{}", encoding="utf-8")
    (dataset_dir / "validation.jsonl").write_text("{}", encoding="utf-8")
    (dataset_dir / "test.jsonl").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("kaggle.bootstrap.KAGGLE_INPUT_ROOT", input_root)
    monkeypatch.setattr("kaggle.run_semantic_training.KAGGLE_WORKING_ROOT", tmp_path / "working")

    attached = discover_attached_dataset(input_root)
    semantic = discover_semantic_dataset(input_root)
    paths = ensure_kaggle_paths(tmp_path / "working")
    plan = build_training_plan(dataset_dir=dataset_dir, output_root=tmp_path / "working")

    assert attached == [dataset_dir]
    assert semantic == dataset_dir
    assert paths.checkpoints.exists()
    assert paths.adapters.exists()
    assert plan["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_resume_checkpoint_detection_prefers_latest(tmp_path):
    output_dir = tmp_path / "working" / "checkpoints"
    (output_dir / "checkpoint-2").mkdir(parents=True)
    (output_dir / "checkpoint-9").mkdir()
    (output_dir.parent / "last_checkpoint").write_text("checkpoint-9", encoding="utf-8")

    assert detect_resume_checkpoint(output_dir) == output_dir / "checkpoint-9"
    assert detect_resume_checkpoint(output_dir, resume_from=str(output_dir / "checkpoint-2")) == output_dir / "checkpoint-2"


def test_artifact_manifest_and_zip_exclusions(tmp_path):
    safe = tmp_path / "adapter"
    unsafe = tmp_path / "dataset.jsonl"
    safe.write_text("safe", encoding="utf-8")
    unsafe.write_text("secret", encoding="utf-8")

    manifest = build_artifact_manifest([safe, unsafe])
    zip_path = create_final_zip(tmp_path, [safe, unsafe], zip_name="bundle.zip")

    assert manifest["manifest_version"] == 1
    assert {item["name"] for item in manifest["artifacts"]} == {"adapter", "dataset.jsonl"}
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as archive:
        assert archive.namelist() == ["adapter"]


def test_semantic_verdict_logic():
    promotable = semantic_verdict(
        gate_results={
            "intent_accuracy": 0.96,
            "binding_accuracy": 0.91,
            "predicate_coverage": 0.92,
            "logical_structure_accuracy": 0.92,
            "semantic_schema_valid_rate": 0.99,
            "fallback_accuracy": 0.95,
        },
        readiness=True,
        fallback_rate=0.0,
    )
    rejected = semantic_verdict(
        gate_results={"intent_accuracy": 0.2},
        readiness=True,
        fallback_rate=1.0,
    )
    failed = semantic_verdict(
        gate_results={},
        readiness=False,
        fallback_rate=0.0,
    )

    assert promotable == "PROMOTE_SEMANTIC_EXTRACTOR_TO_SHADOW"
    assert rejected == "REJECT_SEMANTIC_EXTRACTOR"
    assert failed == "TRAINING_FAILED"


def test_privacy_invariants_for_safe_artifacts(tmp_path):
    report = tmp_path / "final_report.json"
    metrics = tmp_path / "metrics.json"
    manifest = tmp_path / "artifact_manifest.json"
    report.write_text(json.dumps({"no raw data": True}), encoding="utf-8")
    metrics.write_text(json.dumps({"intent_accuracy": 1.0}), encoding="utf-8")
    manifest.write_text(json.dumps({"manifest_version": 1}), encoding="utf-8")

    zip_path = create_final_zip(tmp_path, [report, metrics, manifest])
    payload = zip_path.read_bytes()

    for needle in [b"John Smith", b"john@example.com", b"ACC-9988", b"SecretCompanyXYZ", b"9876543210"]:
        assert needle not in payload
