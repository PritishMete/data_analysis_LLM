from __future__ import annotations

from kaggle import qwen_qlora_learning_experiment as experiment
from kaggle import qwen_qlora_memorization_convergence as convergence


def test_memorization_convergence_only_overrides_learning_rate(monkeypatch):
    monkeypatch.setattr(experiment, "LEARNING_RATE", convergence.EXPECTED_BASE_LEARNING_RATE)
    original_steps = experiment.MEMORIZATION_STEPS
    original_train_examples = experiment.MEMORIZATION_TRAIN_EXAMPLES
    original_model = experiment.MODEL_ID
    original_seq_len = experiment.MAX_SEQUENCE_LENGTH

    convergence.configure_memorization_convergence()

    assert experiment.LEARNING_RATE == convergence.MEMORIZATION_CONVERGENCE_LR
    assert experiment.MEMORIZATION_STEPS == original_steps
    assert experiment.MEMORIZATION_TRAIN_EXAMPLES == original_train_examples
    assert experiment.MODEL_ID == original_model
    assert experiment.MAX_SEQUENCE_LENGTH == original_seq_len


def test_memorization_flag_is_added_once():
    assert convergence._with_memorization_flag([]) == ["--memorization"]
    assert convergence._with_memorization_flag(["--memorization"]) == ["--memorization"]
    assert convergence._with_memorization_flag(["--run-id", "abc"]) == ["--run-id", "abc", "--memorization"]


def test_memorization_convergence_refuses_baseline_lr_drift(monkeypatch):
    monkeypatch.setattr(experiment, "LEARNING_RATE", 2e-5)
    try:
        convergence.configure_memorization_convergence()
    except RuntimeError as exc:
        assert "BASE_LEARNING_RATE_DRIFTED" in str(exc)
    else:
        raise AssertionError("expected baseline drift protection")
