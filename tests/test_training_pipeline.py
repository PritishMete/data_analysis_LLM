from __future__ import annotations

import json
from pathlib import Path

from training.cli import main
from training.dataset import create_dataset_manifest, validate_dataset, verify_dataset_manifest, write_dataset_manifest
from training.hardware import HardwareReport
from training.metrics import evaluate_training_metrics
from training.model_loader import DEFAULT_PROTOTYPE_MODEL, DEFAULT_MODEL_REGISTRY_ENTRY
from training.promotion import evaluate_promotion_gates


def test_validate_dataset_flags_canonical_dataset_ready():
    result = validate_dataset(Path("runtime") / "training")
    assert result.ready_for_prototype is True
    assert result.dataset_version
    assert result.report["eligible_examples"] == 500
    assert result.readiness["ready_for_prototype"] is True
    assert result.split_counts == {"train": 407, "validation": 47, "test": 46}


def test_dataset_manifest_create_and_verify(tmp_path):
    dataset_dir = Path("runtime") / "training"
    manifest = create_dataset_manifest(dataset_dir)
    assert "files" in manifest
    manifest_path = write_dataset_manifest(dataset_dir, tmp_path / "dataset_manifest.sha256.json")
    verification = verify_dataset_manifest(dataset_dir, manifest_path)
    assert verification["verified"] is True


def test_hardware_report_serializes():
    report = HardwareReport(
        python_version="3.13",
        platform="Windows",
        machine="AMD64",
        processor="AMD64",
        ram_gb=16.0,
        torch_version=None,
        cuda_available=False,
        cuda_version=None,
        gpu_name=None,
        vram_gb=None,
    )
    payload = report.to_dict()
    assert payload["cuda_available"] is False
    assert payload["ram_gb"] == 16.0


def test_qlora_and_model_registry_metadata():
    qlora = DEFAULT_PROTOTYPE_MODEL.to_dict()["qlora_config"]
    assert DEFAULT_PROTOTYPE_MODEL.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert qlora["load_in_4bit"] is True
    assert qlora["bnb_4bit_quant_type"] == "nf4"
    assert DEFAULT_MODEL_REGISTRY_ENTRY.to_dict()["qlora_enabled"] is True


def test_training_metrics_and_promotion_gates():
    metrics = evaluate_training_metrics(
        predicted=[{"plan_valid": True, "predicate_keys": ["a"], "logical_structure": "AND", "semantic_roles": ["x"], "tool_graph": ["tool.a"]}],
        expected=[{"plan_valid": True, "predicate_keys": ["a"], "logical_structure": "AND", "semantic_roles": ["x"], "tool_graph": ["tool.a"]}],
    )
    assert metrics.json_validity == 1.0
    assert metrics.tool_selection_f1 == 1.0
    promotion = evaluate_promotion_gates(
        readiness={"ready_for_prototype": True},
        metrics=metrics.to_dict(),
    )
    assert promotion.promotable is True


def test_cli_hardware_and_validate_dataset(capsys):
    assert main(["hardware"]) == 0
    hardware_out = capsys.readouterr().out
    assert "python_version" in hardware_out

    assert main(["validate-dataset"]) == 0
    validate_out = capsys.readouterr().out
    parsed = json.loads(validate_out)
    assert parsed["ready_for_prototype"] is True
    assert parsed["split_counts"] == {"train": 407, "validation": 47, "test": 46}


def test_cli_manifest_and_dry_run(capsys, tmp_path):
    assert main(["manifest-create", "--manifest-path", str(tmp_path / "sha.json")]) == 0
    create_out = json.loads(capsys.readouterr().out)
    assert Path(create_out["manifest_path"]).exists()

    assert main(["manifest-verify", "--manifest-path", str(tmp_path / "sha.json")]) == 0
    verify_out = json.loads(capsys.readouterr().out)
    assert verify_out["verified"] is True

    assert main(["dry-run"]) == 0
    dry_run_out = json.loads(capsys.readouterr().out)
    assert dry_run_out["status"] == "dry_run"
    assert dry_run_out["promotion_gate"]["promotable"] is True


def test_cli_refuses_real_training_without_cuda():
    assert main(["train"]) == 3
