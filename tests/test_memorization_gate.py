from kaggle.memorization_gate import assess_memorization_report


def _report(step_0_score: float = 0.0, step_50_score: float = 1.0):
    def metrics(value: float):
        return {
            "semantic_schema_valid_rate": value,
            "intent_accuracy": value,
            "binding_accuracy": value,
            "predicate_coverage": value,
            "logical_structure_accuracy": value,
            "fallback_accuracy": value,
            "valid_json_rate": value,
            "exact_semantic_match_rate": value,
            "truncated_prediction_count": 0,
            "parser_classification_counts": {},
            "generation_parse_failure_count": 0,
        }

    return {
        "config": {"memorization": True},
        "dataset": {"experiment_train_examples": 8},
        "tokenization": {"target_truncation_count": 0},
        "validation_metrics": {
            "step_0": metrics(step_0_score),
            "step_50": metrics(step_50_score),
        },
        "training_steps": [{"step": step} for step in range(1, 51)],
        "lora_parameter_changed": True,
        "lora_gradients_finite_nonzero": True,
        "base_parameters_changed": False,
        "base_gradient_violation": False,
        "test_split_accessed": False,
        "test_data_used": False,
        "verdict": "SEMANTIC_LEARNING_SIGNAL_CONFIRMED",
    }


def test_strong_memorization_is_ready_for_128_16():
    result = assess_memorization_report(_report())
    assert result["decision"] == "READY_FOR_128_16"
    assert result["ready_for_128_16"] is True
    assert result["test_split_accessed"] is False


def test_test_access_blocks_progression():
    report = _report()
    report["test_split_accessed"] = True
    result = assess_memorization_report(report)
    assert result["decision"] == "MEMORIZATION_BLOCKED"
    assert "test_split_accessed" in result["failures"]


def test_target_truncation_blocks_progression():
    report = _report()
    report["tokenization"]["target_truncation_count"] = 1
    result = assess_memorization_report(report)
    assert result["decision"] == "MEMORIZATION_BLOCKED"
    assert "target_truncation" in result["failures"]


def test_improvement_without_perfect_schema_requires_review():
    result = assess_memorization_report(_report(step_0_score=0.0, step_50_score=0.75))
    assert result["decision"] == "MEMORIZATION_IMPROVING_REVIEW"
    assert result["ready_for_128_16"] is False


def test_no_learning_signal_is_not_proven():
    result = assess_memorization_report(_report(step_0_score=0.0, step_50_score=0.0))
    assert result["decision"] == "MEMORIZATION_NOT_PROVEN"
    assert result["ready_for_128_16"] is False
