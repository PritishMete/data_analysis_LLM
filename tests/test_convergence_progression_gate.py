from kaggle.convergence_progression_gate import assess_convergence_report


def _metrics(value: float):
    return {
        "semantic_schema_valid_rate": value,
        "intent_accuracy": value,
        "binding_accuracy": value,
        "predicate_coverage": value,
        "logical_structure_accuracy": value,
        "fallback_accuracy": value,
        "exact_semantic_match_rate": value,
        "truncated_prediction_count": 0,
        "parser_classification_counts": {},
    }


def _report(step_0: float = 0.0, step_50: float = 1.0):
    return {
        "config": {"memorization": True, "learning_rate": 1e-4},
        "dataset": {"experiment_train_examples": 8},
        "tokenization": {"target_truncation_count": 0},
        "validation_metrics": {
            "step_0": _metrics(step_0),
            "step_50": _metrics(step_50),
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


def test_strong_convergence_is_ready_for_128_16():
    result = assess_convergence_report(_report())
    assert result["decision"] == "READY_FOR_128_16"
    assert result["ready_for_128_16"] is True


def test_partial_improvement_requires_review():
    result = assess_convergence_report(_report(step_0=0.0, step_50=0.75))
    assert result["decision"] == "CONVERGENCE_IMPROVING_REVIEW"
    assert result["ready_for_128_16"] is False


def test_no_improvement_is_not_proven():
    result = assess_convergence_report(_report(step_0=0.25, step_50=0.25))
    assert result["decision"] == "CONVERGENCE_NOT_PROVEN"


def test_test_split_access_blocks_progression():
    report = _report()
    report["test_split_accessed"] = True
    result = assess_convergence_report(report)
    assert result["decision"] == "CONVERGENCE_BLOCKED"
    assert "test_split_accessed" in result["blockers"]


def test_generation_truncation_blocks_progression():
    report = _report()
    report["validation_metrics"]["step_50"]["truncated_prediction_count"] = 1
    result = assess_convergence_report(report)
    assert result["decision"] == "CONVERGENCE_BLOCKED"
    assert "generation_truncation" in result["blockers"]


def test_wrong_learning_rate_blocks_progression():
    report = _report()
    report["config"]["learning_rate"] = 1e-5
    result = assess_convergence_report(report)
    assert result["decision"] == "CONVERGENCE_BLOCKED"
    assert "unexpected_learning_rate" in result["blockers"]


def test_exact_match_is_preferred_not_required():
    report = _report()
    report["validation_metrics"]["step_50"].pop("exact_semantic_match_rate")
    result = assess_convergence_report(report)
    assert result["decision"] == "READY_FOR_128_16"
