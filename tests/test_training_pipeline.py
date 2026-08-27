from __future__ import annotations

import json
from pathlib import Path

from training.cli import main
from training.dataset import validate_dataset
from training.hardware import HardwareReport


def test_validate_dataset_flags_canonical_dataset_ready():
    result = validate_dataset(Path("runtime") / "training")
    assert result.ready_for_prototype is True
    assert result.dataset_version
    assert result.report["eligible_examples"] == 500
    assert result.readiness["ready_for_prototype"] is True


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


def test_cli_hardware_and_validate_dataset(capsys):
    assert main(["hardware"]) == 0
    hardware_out = capsys.readouterr().out
    assert "python_version" in hardware_out

    assert main(["validate-dataset"]) == 0
    validate_out = capsys.readouterr().out
    parsed = json.loads(validate_out)
    assert parsed["ready_for_prototype"] is True
