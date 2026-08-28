from __future__ import annotations

import json
from pathlib import Path

from kaggle.bootstrap import (
    build_artifact_manifest,
    build_semantic_dataset_from_canonical,
    create_final_zip,
    discover_semantic_dataset,
    load_semantic_config,
    resolve_canonical_dataset_root,
    semantic_verdict,
    verify_attached_dataset,
    write_sha_manifest,
)


def _canonical_record(
    *,
    source_id: str,
    intent: str,
    family_fingerprint: str,
    split: str,
    logical_structure: str = "AND",
    quality: float = 0.99,
    semantic_roles: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source_kind": "experience",
        "source_id": source_id,
        "split": split,
        "family_fingerprint": family_fingerprint,
        "input": {
            "intent": intent,
            "semantic_roles": semantic_roles or ["numeric_metric", "filter_value"],
            "operators": ["equals", "greater_than"],
            "logical_structure": logical_structure,
            "predicate_graph": {"predicate_count": 2, "shape": "safe"},
        },
        "output": {
            "tool_graph": ["sql.filter"],
            "plan_source": "validated_template",
            "plan_template_id": "plan.template.semantic",
            "source_kind": "experience",
            "candidate_state": None,
        },
        "metadata": {
            "quality": quality,
            "execution_success": True,
            "critic_passed": True,
            "result_validation_passed": True,
            "plan_completeness_passed": True,
            "privacy_validation_passed": True,
            "no_unresolved_ambiguity": True,
            "no_critical_repair": True,
            "repair_count": 0,
            "correction_state": "validated",
            "candidate_state": None,
            "candidate_evidence_count": None,
            "candidate_average_quality": None,
            "dataset_semantic_signature": "0123456789abcdef",
            "family_fingerprint": family_fingerprint,
            "split": split,
            "family_size": 1,
            "created_at": "2026-08-27T00:00:00+00:00",
            "plan_shape": {"limit": 5, "metric_count": 1, "tool_sequence": ["sql.filter"]},
        },
    }


def _write_canonical_dataset(root: Path, *, with_sha_manifest: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = [
        _canonical_record(source_id="evt_a", intent="filter", family_fingerprint="a" * 64, split="train", semantic_roles=["numeric_metric", "filter_value", "dimension_label"]),
        _canonical_record(source_id="evt_b", intent="analytics", family_fingerprint="b" * 64, split="train", logical_structure="OR", semantic_roles=["trend_metric", "time_window", "dimension_label"]),
        _canonical_record(source_id="evt_c", intent="operation", family_fingerprint="c" * 64, split="train", logical_structure="MIXED", semantic_roles=["action", "constraint", "status_flag"]),
        _canonical_record(source_id="evt_d", intent="cleaning", family_fingerprint="d" * 64, split="train", logical_structure="NOT", semantic_roles=["column_name", "null_check", "threshold"]),
        _canonical_record(source_id="evt_e", intent="sentiment", family_fingerprint="e" * 64, split="validation", semantic_roles=["text_span", "sentiment_label", "source_field"]),
        _canonical_record(source_id="evt_f", intent="filter", family_fingerprint="f" * 64, split="validation", semantic_roles=["numeric_metric", "range_bound", "dimension_label"]),
        _canonical_record(source_id="evt_g", intent="analytics", family_fingerprint="g" * 64, split="validation", logical_structure="OR", semantic_roles=["trend_metric", "aggregation_target", "time_window"]),
        _canonical_record(source_id="evt_h", intent="operation", family_fingerprint="h" * 64, split="test", logical_structure="MIXED", semantic_roles=["action", "workflow_step", "status_flag"]),
        _canonical_record(source_id="evt_i", intent="cleaning", family_fingerprint="i" * 64, split="test", logical_structure="NOT", semantic_roles=["column_name", "duplicate_check", "null_check"]),
        _canonical_record(source_id="evt_j", intent="sentiment", family_fingerprint="j" * 64, split="test", semantic_roles=["text_span", "sentiment_label", "channel_source"]),
    ]
    for split in ("train", "validation", "test"):
        lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records if record["split"] == split]
        (root / f"{split}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    split_counts = {split: sum(1 for record in records if record["split"] == split) for split in ("train", "validation", "test")}
    manifest = {
        "dataset_version": "canonical-test",
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "eligible_examples": sum(split_counts.values()),
        "readiness": {"ready_for_prototype": True},
    }
    report = {
        "dataset_version": "canonical-test",
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "eligible_examples": sum(split_counts.values()),
        "readiness": {"ready_for_prototype": True},
        "split_integrity_passed": True,
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / "dataset_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if with_sha_manifest:
        write_sha_manifest(root, root)
    return root


def test_recursive_discovery_resolves_nested_kaggle_root(tmp_path):
    kaggle_input = tmp_path / "kaggle" / "input"
    canonical_root = _write_canonical_dataset(kaggle_input / "datasets" / "jaistudio" / "data-analysis-llm")

    resolved = resolve_canonical_dataset_root(kaggle_input)
    discovered = discover_semantic_dataset(kaggle_input)
    verification = verify_attached_dataset(canonical_root)

    assert resolved["root"] == str(canonical_root)
    assert discovered == canonical_root
    assert verification["verified"] is True
    assert verification["mismatches"] == []


def test_multiple_candidate_roots_fail_clearly(tmp_path):
    kaggle_input = tmp_path / "input"
    root_a = _write_canonical_dataset(kaggle_input / "vendor-a" / "dataset-a")
    root_b = _write_canonical_dataset(kaggle_input / "vendor-b" / "dataset-b")

    resolved = resolve_canonical_dataset_root(kaggle_input)

    assert resolved["root"] is None
    assert resolved["reason"] == "ambiguous_dataset_root"
    candidate_roots = {item["root"] for item in resolved["candidates"]}
    assert candidate_roots == {str(root_a), str(root_b)}


def test_missing_sha_manifest_falls_back_to_consistency_and_generates_sha(tmp_path):
    canonical_root = _write_canonical_dataset(tmp_path / "canonical")
    verification = verify_attached_dataset(canonical_root)
    semantic_root = tmp_path / "semantic_training"

    assert verification["verified"] is True
    assert verification["mismatches"] == []
    assert "consistency" in verification

    sha_manifest_path = write_sha_manifest(canonical_root, semantic_root)
    payload = json.loads(sha_manifest_path.read_text(encoding="utf-8"))
    names = [item["name"] for item in payload["files"]]

    assert "train.jsonl" in names
    assert "validation.jsonl" in names
    assert "test.jsonl" in names
    assert "dataset_manifest.json" in names
    assert "manifest.json" in names
    assert "dataset_report.json" in names
    assert "report.json" in names
    assert payload["dataset_root"] == str(canonical_root)


def test_sha_manifest_verifier_accepts_list_format(tmp_path):
    canonical_root = _write_canonical_dataset(tmp_path / "canonical")
    sha_manifest_path = write_sha_manifest(canonical_root, canonical_root)

    verification = verify_attached_dataset(canonical_root)

    assert sha_manifest_path.exists()
    assert verification["verified"] is True
    assert verification["mismatches"] == []


def test_canonical_to_semantic_conversion_preserves_privacy_and_rows(tmp_path):
    canonical_root = _write_canonical_dataset(tmp_path / "canonical")
    semantic_report = build_semantic_dataset_from_canonical(canonical_root, tmp_path / "semantic_training")
    semantic_root = Path(semantic_report["semantic_output_root"])

    train = (semantic_root / "train.jsonl").read_text(encoding="utf-8")
    validation = (semantic_root / "validation.jsonl").read_text(encoding="utf-8")
    test = (semantic_root / "test.jsonl").read_text(encoding="utf-8")
    combined = train + validation + test

    assert semantic_report["semantic_row_count"] > 0
    assert semantic_report["readiness"]["ready"] is True
    assert sum(semantic_report["split_counts"].values()) == semantic_report["semantic_row_count"]
    assert "intent" in combined
    assert "semantic_bindings" in combined
    assert "predicate_graph" in combined
    assert "tool_graph" not in combined
    assert "sql" not in combined.lower()
    for needle in ["John Smith", "john@example.com", "ACC-9988", "SecretCompanyXYZ", "9876543210"]:
        assert needle not in combined


def test_artifact_manifest_and_zip_exclusions(tmp_path):
    safe = tmp_path / "training_config"
    unsafe = tmp_path / "dataset.jsonl"
    safe.write_text("safe", encoding="utf-8")
    unsafe.write_text("secret", encoding="utf-8")

    manifest = build_artifact_manifest([safe, unsafe])
    zip_path = create_final_zip(tmp_path, [safe, unsafe], zip_name="bundle.zip")

    assert manifest["manifest_version"] == 1
    assert {item["name"] for item in manifest["artifacts"]} == {"dataset.jsonl", "training_config"}
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as archive:
        assert archive.namelist() == ["training_config"]


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


def test_semantic_config_loader_resolves_repo_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_semantic_config()

    assert config["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["training"]["max_seq_len"] == 768
