from __future__ import annotations

import pytest

from kaggle import qwen_qlora_128_16_controlled as controlled
from kaggle import qwen_qlora_learning_experiment as experiment


def test_controlled_128_16_contract_matches_baseline():
    result = controlled.verify_128_16_contract()
    assert result["experiment"] == "128_TRAIN_16_VALIDATION"
    assert result["train_examples"] == 128
    assert result["validation_examples"] == 16
    assert result["learning_rate"] == 1e-5
    assert result["optimizer_steps"] == 16
    assert result["validation_steps"] == [0, 4, 8, 12, 16]
    assert result["gradient_accumulation"] == 8
    assert result["max_sequence_length"] == 768
    assert result["test_split_accessed"] is False


def test_controlled_128_16_refuses_baseline_drift(monkeypatch):
    monkeypatch.setattr(experiment, "TRAIN_EXAMPLES", 127)
    with pytest.raises(RuntimeError, match="CONTROLLED_128_16_BASELINE_DRIFTED"):
        controlled.verify_128_16_contract()


def test_controlled_128_16_refuses_memorization_flag():
    with pytest.raises(RuntimeError, match="CONTROLLED_128_16_REFUSES_MEMORIZATION_FLAG"):
        controlled.main(["--memorization"])


def test_controlled_128_16_dispatches_baseline_without_mutation(monkeypatch):
    observed = {}

    def fake_main(argv):
        observed["argv"] = list(argv)
        return 0

    monkeypatch.setattr(experiment, "main", fake_main)
    before = {
        "train": experiment.TRAIN_EXAMPLES,
        "validation": experiment.VALIDATION_EXAMPLES,
        "lr": experiment.LEARNING_RATE,
        "steps": experiment.OPTIMIZER_STEPS,
        "checkpoints": experiment.VALIDATION_STEPS,
        "grad_accum": experiment.GRADIENT_ACCUMULATION,
        "max_seq": experiment.MAX_SEQUENCE_LENGTH,
    }

    assert controlled.main(["--run-id", "controlled-run"]) == 0
    assert observed["argv"] == ["--run-id", "controlled-run"]
    assert experiment.TRAIN_EXAMPLES == before["train"]
    assert experiment.VALIDATION_EXAMPLES == before["validation"]
    assert experiment.LEARNING_RATE == before["lr"]
    assert experiment.OPTIMIZER_STEPS == before["steps"]
    assert experiment.VALIDATION_STEPS == before["checkpoints"]
    assert experiment.GRADIENT_ACCUMULATION == before["grad_accum"]
    assert experiment.MAX_SEQUENCE_LENGTH == before["max_seq"]
