from __future__ import annotations

from kaggle import qwen_qlora_learning_experiment as experiment
from kaggle import qwen_qlora_memorization_convergence as convergence
from kaggle import execute_smoke_training
from scripts import kaggle_runner


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


def test_remote_dispatch_reuses_existing_learning_module(monkeypatch, tmp_path):
    run_id = "run-convergence"
    commit = "a" * 40
    observed = {}
    monkeypatch.setattr(experiment, "LEARNING_RATE", execute_smoke_training.EXPECTED_BASE_LEARNING_RATE)

    def fake_run(**kwargs):
        observed.update(kwargs)
        observed["learning_rate_during_run"] = experiment.LEARNING_RATE
        return {
            "run_id": run_id,
            "expected_git_commit": commit,
            "config": {
                "learning_rate": execute_smoke_training.MEMORIZATION_CONVERGENCE_LR,
                "memorization": True,
            },
        }

    monkeypatch.setattr(experiment, "run_qwen_qlora_learning_experiment", fake_run)
    result = execute_smoke_training._run_memorization_convergence(
        output_root=tmp_path,
        run_id=run_id,
        expected_git_commit=commit,
        source_root=tmp_path / "source",
    )

    assert observed["memorization"] is True
    assert observed["learning_rate_during_run"] == 1e-4
    assert experiment.LEARNING_RATE == 1e-5
    assert result["run_id"] == run_id


def test_remote_dispatch_refuses_baseline_lr_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "LEARNING_RATE", 2e-5)
    try:
        execute_smoke_training._run_memorization_convergence(
            output_root=tmp_path,
            run_id="run-convergence",
            expected_git_commit="a" * 40,
            source_root=tmp_path / "source",
        )
    except RuntimeError as exc:
        assert "BASE_LEARNING_RATE_DRIFTED" in str(exc)
    else:
        raise AssertionError("expected baseline drift protection")


def test_remote_dispatch_restores_lr_after_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "LEARNING_RATE", 1e-5)

    def fail(**kwargs):
        assert experiment.LEARNING_RATE == 1e-4
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(experiment, "run_qwen_qlora_learning_experiment", fail)
    try:
        execute_smoke_training._run_memorization_convergence(
            output_root=tmp_path,
            run_id="run-convergence",
            expected_git_commit="a" * 40,
            source_root=tmp_path / "source",
        )
    except RuntimeError as exc:
        assert "synthetic failure" in str(exc)
    else:
        raise AssertionError("expected synthetic failure")
    assert experiment.LEARNING_RATE == 1e-5


def test_runner_exposes_convergence_cycle_workflow(monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setattr(kaggle_runner, "preflight", lambda spec, stage_root: {"ready": True})
    monkeypatch.setattr(kaggle_runner, "run", lambda spec, stage_root, run_id=None: observed.setdefault("workflow_mode", spec.workflow_mode) or {})
    monkeypatch.setattr(kaggle_runner, "outputs", lambda spec, stage_root, run_id=None: {"status": "retrieved"})
    monkeypatch.setattr(kaggle_runner, "_resolve_stage_root", lambda root, run_id: root)

    result = kaggle_runner.qwen_semantic_memorization_convergence_cycle(
        stage_root=tmp_path,
        run_id="run-convergence",
    )

    assert result["run_id"] == "run-convergence"
    assert observed["workflow_mode"] == "qwen_semantic_memorization_convergence"
    parsed = kaggle_runner.build_parser().parse_args(["qwen-semantic-memorization-convergence-cycle"])
    assert parsed.command == "qwen-semantic-memorization-convergence-cycle"
