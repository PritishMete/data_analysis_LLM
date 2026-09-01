from pathlib import Path

from kaggle import qwen_qlora_learning_experiment as experiment


SOURCE = Path(experiment.__file__).read_text(encoding="utf-8")


def test_experiment_uses_fixed_train_validation_subsets_and_seals_test():
    assert experiment.TRAIN_EXAMPLES == 128
    assert experiment.VALIDATION_EXAMPLES == 16
    assert '"test_split_accessed": False' in SOURCE
    assert 'root / "test.jsonl"' not in SOURCE
    assert '"test_data_used": False' in SOURCE


def test_experiment_preserves_frozen_runtime_and_training_profile():
    assert experiment.MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert experiment.PEFT_VERSION == "0.13.2"
    assert 'bnb_4bit_quant_type="nf4"' in SOURCE
    assert 'bnb_4bit_compute_dtype=torch.float16' in SOURCE
    assert "GRADIENT_ACCUMULATION = 8" in SOURCE
    assert "OPTIMIZER_STEPS = 16" in SOURCE
    assert "LEARNING_RATE = 1e-5" in SOURCE


def test_target_parser_evaluator_contract_is_explicitly_shared():
    assert "ALLOWED_SEMANTIC_OUTPUT_KEYS" in SOURCE
    assert "set(value) != ALLOWED_SEMANTIC_OUTPUT_KEYS" in SOURCE
    assert "training_target_schema_matches_parser" in SOURCE
    assert "semantic_metrics(predictions, expected)" in SOURCE


def test_label_masking_audit_supervises_only_target_span():
    assert "labels[:, :prompt_count] = -100" in SOURCE
    assert '"masked_label_count"' in SOURCE
    assert '"supervised_label_count"' in SOURCE
    assert '"supervised_labeling_verified": True' in SOURCE


def test_validation_schedule_and_train_sanity_are_present():
    assert '"step_0"' in SOURCE
    assert "VALIDATION_STEPS = (0, 4, 8, 12, 16)" in SOURCE
    assert "train_sanity = _evaluate" in SOURCE
    assert "best_validation_step" in SOURCE
    assert "best_validation_score" in SOURCE
    assert "adapter_root / f\"best_step_{step}\"" in SOURCE


def test_generation_is_deterministic_and_bounded_by_target_audit():
    assert "do_sample=False" in SOURCE
    assert "max_target_tokens" in SOURCE
    assert "generation_max_new_tokens" in SOURCE
    assert "truncated_prediction_count" in SOURCE


def test_diagnostics_are_privacy_safe_structural_summaries():
    assert "_safe_target_hash" in SOURCE
    assert "_structure_summary" in SOURCE
    assert "metric_failure_reasons" in SOURCE
    assert "query_text" in SOURCE
    assert "workbook" in SOURCE


def test_experiment_command_is_wired():
    from scripts import kaggle_runner

    assert "qwen-qlora-learning-experiment-cycle" in kaggle_runner.build_parser().format_help()
    assert kaggle_runner.qwen_qlora_learning_experiment_cycle.__name__ == "qwen_qlora_learning_experiment_cycle"
