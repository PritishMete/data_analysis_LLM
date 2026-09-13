from __future__ import annotations

import json

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


def test_runtime_dispatch_runs_convergence_module_and_reads_current_run_report(monkeypatch, tmp_path):
    run_id = "run-convergence"
    commit = "a" * 40
    report_path = tmp_path / "smoke_runs" / run_id / "learning_experiment_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"run_id": run_id, "expected_git_commit": commit}), encoding="utf-8")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(execute_smoke_training.subprocess, "run", fake_run)
    result = execute_smoke_training._run_memorization_convergence_module(
        output_root=tmp_path,
        run_id=run_id,
        expected_git_commit=commit,
        source_root=tmp_path / "source",
    )

    command = observed["command"]
    assert command[:3] == [execute_smoke_training.sys.executable, "-m", "kaggle.qwen_qlora_memorization_convergence"]
    assert command[command.index("--run-id") + 1] == run_id
    assert command[command.index("--expected-git-commit") + 1] == commit
    assert result["run_id"] == run_id
    assert observed["kwargs"]["capture_output"] is True


def test_runtime_dispatch_rejects_report_from_another_run(monkeypatch, tmp_path):
    report_path = tmp_path / "smoke_runs" / "expected-run" / "learning_experiment_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"run_id": "historical-run", "expected_git_commit": "a" * 40}), encoding="utf-8")
    monkeypatch.setattr(
        execute_smoke_training.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )

    try:
        execute_smoke_training._run_memorization_convergence_module(
            output_root=tmp_path,
            run_id="expected-run",
            expected_git_commit="a" * 40,
            source_root=tmp_path / "source",
        )
    except RuntimeError as exc:
        assert "RUN_ID_MISMATCH" in str(exc)
    else:
        raise AssertionError("expected current-run report identity enforcement")


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
