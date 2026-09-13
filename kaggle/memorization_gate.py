"""Decision gate for the 8-example semantic memorization experiment.

This module is intentionally side-effect free.  It reads only the generated
learning-experiment report and decides whether the next 128/16 experiment is
justified.  It never opens train/validation/test corpus files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHECKPOINT_METRICS = (
    "semantic_schema_valid_rate",
    "intent_accuracy",
    "binding_accuracy",
    "predicate_coverage",
    "logical_structure_accuracy",
    "fallback_accuracy",
)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _score(metrics: dict[str, Any]) -> float:
    return sum(_as_float(metrics.get(key)) for key in CHECKPOINT_METRICS) / len(CHECKPOINT_METRICS)


def _count(metrics: dict[str, Any], key: str) -> int:
    try:
        return int(metrics.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def assess_memorization_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a conservative next-step decision from a memorization report."""
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    tokenization = report.get("tokenization") if isinstance(report.get("tokenization"), dict) else {}
    evaluations = report.get("validation_metrics") if isinstance(report.get("validation_metrics"), dict) else {}
    step_0 = evaluations.get("step_0") if isinstance(evaluations.get("step_0"), dict) else {}
    step_50 = evaluations.get("step_50") if isinstance(evaluations.get("step_50"), dict) else {}

    failures: list[str] = []
    if config.get("memorization") is not True:
        failures.append("not_memorization_run")
    if int(dataset.get("experiment_train_examples", 0) or 0) != 8:
        failures.append("train_example_count_not_8")
    if report.get("test_split_accessed") is not False or report.get("test_data_used") is not False:
        failures.append("test_split_accessed")
    if int(tokenization.get("target_truncation_count", 0) or 0) != 0:
        failures.append("target_truncation")
    if report.get("lora_parameter_changed") is not True:
        failures.append("lora_parameters_unchanged")
    if report.get("base_parameters_changed") is not False:
        failures.append("base_parameters_changed")
    if report.get("base_gradient_violation") is not False:
        failures.append("base_gradient_violation")
    if report.get("lora_gradients_finite_nonzero") is not True:
        failures.append("invalid_lora_gradients")
    if len(report.get("training_steps") or []) < 50:
        failures.append("optimizer_steps_below_50")
    if not step_50:
        failures.append("missing_step_50_metrics")

    truncation = _count(step_50, "truncated_prediction_count")
    max_token_stops = _count(step_50.get("parser_classification_counts", {}) if isinstance(step_50.get("parser_classification_counts"), dict) else {}, "MAX_NEW_TOKENS_REACHED")
    if truncation or max_token_stops:
        failures.append("generation_truncation")

    schema_valid = _as_float(step_50.get("semantic_schema_valid_rate"))
    intent_accuracy = _as_float(step_50.get("intent_accuracy"))
    fallback_accuracy = _as_float(step_50.get("fallback_accuracy"))
    valid_json_rate = _as_float(step_50.get("valid_json_rate", step_50.get("json_valid_rate", 1.0 if _count(step_50, "generation_parse_failure_count") == 0 and step_50 else 0.0)))
    exact_match = _as_float(step_50.get("exact_semantic_match_rate"))
    score_0 = _score(step_0) if step_0 else 0.0
    score_50 = _score(step_50) if step_50 else 0.0
    meaningful_improvement = score_50 > score_0 + 1e-9

    hard_failures = [
        item
        for item in failures
        if item
        in {
            "not_memorization_run",
            "train_example_count_not_8",
            "test_split_accessed",
            "target_truncation",
            "lora_parameters_unchanged",
            "base_parameters_changed",
            "base_gradient_violation",
            "invalid_lora_gradients",
            "optimizer_steps_below_50",
            "missing_step_50_metrics",
            "generation_truncation",
        }
    ]

    strong_memorization = (
        not hard_failures
        and schema_valid >= 1.0
        and intent_accuracy >= 1.0
        and fallback_accuracy >= 1.0
        and valid_json_rate >= 1.0
        and meaningful_improvement
    )

    if strong_memorization:
        decision = "READY_FOR_128_16"
    elif hard_failures:
        decision = "MEMORIZATION_BLOCKED"
    elif meaningful_improvement:
        decision = "MEMORIZATION_IMPROVING_REVIEW"
    else:
        decision = "MEMORIZATION_NOT_PROVEN"

    return {
        "decision": decision,
        "ready_for_128_16": decision == "READY_FOR_128_16",
        "failures": failures,
        "step_0_score": score_0,
        "step_50_score": score_50,
        "meaningful_improvement": meaningful_improvement,
        "step_50": {
            "semantic_schema_valid_rate": schema_valid,
            "intent_accuracy": intent_accuracy,
            "fallback_accuracy": fallback_accuracy,
            "valid_json_rate": valid_json_rate,
            "exact_semantic_match_rate": exact_match,
            "generation_truncation_count": truncation + max_token_stops,
        },
        "test_split_accessed": report.get("test_split_accessed"),
        "reported_verdict": report.get("verdict"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assess an 8-example memorization report.")
    parser.add_argument("report", type=Path, help="Path to learning_experiment_report.json")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = assess_memorization_report(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] in {"READY_FOR_128_16", "MEMORIZATION_IMPROVING_REVIEW", "MEMORIZATION_NOT_PROVEN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
