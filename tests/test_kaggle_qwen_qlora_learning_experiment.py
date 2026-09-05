import json
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


def test_training_and_inference_share_one_canonical_prompt_builder():
    assert "def build_semantic_prompt" in SOURCE
    assert "semantic_prompt_token_ids(tokenizer, row)" in SOURCE
    assert "tokenizer(build_semantic_prompt(row)" in SOURCE
    assert "\n_prompt(" not in SOURCE


def test_schema_failure_diagnostics_are_structural_and_safe():
    value = {"intent": 7, "requires_fallback": "no", "extra": True}
    diagnostics = experiment.schema_failure_diagnostics(value)
    assert "semantic_bindings" in diagnostics["missing_keys"]
    assert diagnostics["unexpected_keys"] == ["extra"]
    assert diagnostics["wrong_types"] == ["intent", "requires_fallback"]
    assert diagnostics["invalid_shapes"] == []


def test_schema_failure_diagnostics_detect_invalid_shapes():
    diagnostics = experiment.schema_failure_diagnostics({
        "intent": "filter",
        "semantic_bindings": {},
        "predicate_graph": {"logical_structure": []},
        "aggregation": {"required": "yes"},
        "ranking": {"required": 1},
        "limit": None,
        "requires_fallback": False,
        "confidence": 1.0,
    })
    assert diagnostics["invalid_shapes"] == [
        "aggregation.required",
        "predicate_graph.logical_structure",
        "ranking.required",
    ]


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


def test_parser_distinguishes_generation_outcomes():
    assert experiment._parse_prediction_diagnostic("", generated_tokens=0, max_new_tokens=10)[1] == "NO_GENERATION"
    assert experiment._parse_prediction_diagnostic('{"intent":', generated_tokens=10, max_new_tokens=10)[1] == "INCOMPLETE_JSON"
    assert experiment._parse_prediction_diagnostic('{"intent":', generated_tokens=10, max_new_tokens=10, termination_reason="MAX_NEW_TOKENS_REACHED")[1] == "MAX_NEW_TOKENS_REACHED"
    assert experiment._parse_prediction_diagnostic("not json", generated_tokens=2, max_new_tokens=10)[1] == "MALFORMED_JSON"
    assert experiment._parse_prediction_diagnostic("[]", generated_tokens=2, max_new_tokens=10)[1] == "NON_OBJECT_JSON"


def test_balanced_json_extractor_handles_nested_objects_and_braces_in_strings():
    text = 'prefix {"predicate_graph":{"operator":"AND"},"note":"brace } and \\\"quote"} trailing garbage'
    assert experiment._extract_first_json_object(text) == text[text.index("{") : -len(" trailing garbage")]


def test_generation_termination_reason_is_based_on_actual_completion_tokens():
    assert experiment._generation_termination_reason([1, 2, 151645], generated_tokens=3, max_new_tokens=3, eos_token_id=151645) == "EOS"
    assert experiment._generation_termination_reason([1, 2, 3], generated_tokens=3, max_new_tokens=3, eos_token_id=151645) == "MAX_NEW_TOKENS_REACHED"
    assert experiment._generation_termination_reason([], generated_tokens=0, max_new_tokens=3, eos_token_id=151645) == "NO_GENERATION"


def test_complete_first_json_object_survives_trailing_text():
    value = {"intent": "filter", "semantic_bindings": {}, "predicate_graph": {}, "aggregation": {}, "ranking": {}, "limit": None, "requires_fallback": False, "confidence": 1.0}
    text = json.dumps(value) + " trailing prose"
    parsed, classification = experiment._parse_prediction_diagnostic(text, generated_tokens=20, max_new_tokens=192, termination_reason="OTHER_STOPPING_CRITERION")
    assert parsed == value
    assert classification == "VALID_SEMANTIC_OUTPUT"


def test_generation_budget_exceeds_audited_target_with_margin():
    assert experiment._generation_budget(137) > 137
    assert experiment._generation_budget(137) <= 768


def test_target_distribution_is_measured_without_raw_inputs():
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(max(1, len(text) // 10)))}

    rows = [{"output": {"intent": "filter"}}]
    distribution = experiment._target_token_distribution(Tokenizer(), rows)
    assert distribution["count"] == 1
    assert distribution["minimum"] == distribution["maximum"]


def test_memorization_contract_is_sealed_and_uses_required_milestones():
    assert experiment.MEMORIZATION_TRAIN_EXAMPLES == 8
    assert experiment.MEMORIZATION_STEPS == (0, 10, 25, 50)
    assert "memorization" in SOURCE
    assert 'validation_data_used": not memorization' in SOURCE
    assert 'test_used": False' in SOURCE


def test_model_not_learning_is_not_selected_when_generation_is_truncated():
    assert '"GENERATION_TRUNCATION_FOUND" if has_truncation' in SOURCE
    assert '"MODEL_NOT_LEARNING"' in SOURCE


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


def test_corpus_audit_command_is_wired_without_training():
    from scripts import kaggle_runner

    assert "semantic-corpus-audit-cycle" in kaggle_runner.build_parser().format_help()
    assert "qwen" not in Path("kaggle/semantic_corpus_audit_cycle.py").read_text(encoding="utf-8").lower()
