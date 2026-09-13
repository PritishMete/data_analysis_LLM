"""Progression gate for the 8-example LR convergence experiment.

This module is side-effect free. It only inspects a completed
learning_experiment_report.json and decides whether the controlled 128/16 stage
is justified. It never opens corpus files or touches the sealed test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CORE_METRICS = (
    "semantic_schema_valid_rate",
    "intent_accuracy",
    "binding_accuracy",
    "predicate_coverage",
    "logical_structure_accuracy",
    "fallback_accuracy",
)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _score(metrics: dict[str, Any]) -> float:
    return sum(_float(metrics.get(key)) for key in CORE_METRICS) / len(CORE_METRICS)


def _checkpoint(report: dict[str, Any], step: int) -> dict[str, Any]:
    metrics = report.get("validation_metrics")
    if not isinstance(metrics, dict):
        return {}
    value = metrics.get(f"step_{step}")
    return value if isinstance(value, dict) else {}


def _generation_truncation(metrics: dict[str, Any]) -> int:
    count = _int(metrics.get("truncated_prediction_count"))
    parser_counts = metrics.get("parser_classification_counts")
    if isinstance(parser_counts, dict):
        count += _int(parser_counts.get("MAX_NEW_TOKENS_REACHED"))
    return count


def assess_convergence_report(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    tokenization = report.get("tokenization") if isinstance(report.get("tokenization"), dict) else {}
    step_0 = _checkpoint(report, 0)
    step_50 = _checkpoint(report, 50)

    blockers: list[str] = []
    if config.get("memorization") is not True:
        blockers.append("not_memorization_run")
    if int(dataset.get("experiment_train_examples", 0) or 0) != 8:
        blockers.append("train_example_count_not_8")

    learning_rate = _float(config.get("learning_rate", report.get("learning_rate")))
    if abs(learning_rate - 1e-4) > 1e-12:
        blockers.append("unexpected_learning_rate")

    if report.get("test_split_accessed") is not False or report.get("test_data_used") is not False:
        blockers.append("test_split_accessed")
    if _int(tokenization.get("target_truncation_count")) != 0:
        blockers.append("target_truncation")
    if report.get("lora_parameter_changed") is not True:
        blockers.append("lora_parameters_unchanged")
    if report.get("base_parameters_changed") is not False:
        blockers.append("base_parameters_changed")
    if report.get("base_gradient_violation") is not False:
        blockers.append("base_gradient_violation")
    if report.get("lora_gradients_finite_nonzero") is not True:
        blockers.append("invalid_lora_gradients")
    if len(report.get("training_steps") or []) < 50:
        blockers.append("optimizer_steps_below_50")
    if not step_50:
        blockers.append("missing_step_50_metrics")

    generation_truncation = _generation_truncation(step_50) if step_50 else 0
    if generation_truncation:
        blockers.append("generation_truncation")

    score_0 = _score(step_0) if step_0 else 0.0
    score_50 = _score(step_50) if step_50 else 0.0
    meaningful_improvement = score_50 > score_0 + 1e-9

    schema = _float(step_50.get("semantic_schema_valid_rate"))
    intent = _float(step_50.get("intent_accuracy"))
    binding = _float(step_50.get("binding_accuracy"))
    predicate = _float(step_50.get("predicate_coverage"))
    logic = _float(step_50.get("logical_structure_accuracy"))
    fallback = _float(step_50.get("fallback_accuracy"))
    exact = _float(step_50.get("exact_semantic_match_rate"))

    # Exact-semantic-match is preferred but not required because older reports
    # did not always emit it. The six stable semantic metrics remain authoritative.
    strong_semantic_fit = (
        schema >= 1.0
        and intent >= 1.0
        and fallback >= 1.0
        and binding >= 0.875
        and predicate >= 0.875
        and logic >= 0.875
    )

    if blockers:
        decision = "CONVERGENCE_BLOCKED"
    elif strong_semantic_fit and meaningful_improvement:
        decision = "READY_FOR_128_16"
    elif meaningful_improvement:
        decision = "CONVERGENCE_IMPROVING_REVIEW"
    else:
        decision = "CONVERGENCE_NOT_PROVEN"

    return {
        "decision": decision,
        "ready_for_128_16": decision == "READY_FOR_128_16",
        "blockers": blockers,
        "learning_rate": learning_rate,
        "step_0_score": score_0,
        "step_50_score": score_50,
        "meaningful_improvement": meaningful_improvement,
        "step_50": {
            "semantic_schema_valid_rate": schema,
            "intent_accuracy": intent,
            "binding_accuracy": binding,
            "predicate_coverage": predicate,
            "logical_structure_accuracy": logic,
            "fallback_accuracy": fallback,
            "exact_semantic_match_rate": exact,
            "generation_truncation_count": generation_truncation,
        },
        "test_split_accessed": report.get("test_split_accessed"),
        "reported_verdict": report.get("verdict"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assess convergence experiment progression")
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = assess_convergence_report(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] != "CONVERGENCE_BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
