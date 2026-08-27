from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainingMetrics:
    json_validity: float = 0.0
    plan_validity: float = 0.0
    predicate_coverage: float = 0.0
    logical_structure_accuracy: float = 0.0
    semantic_role_coverage: float = 0.0
    tool_selection_f1: float = 0.0
    tool_sequence_accuracy: float = 0.0
    invalid_tool_rate: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "json_validity": self.json_validity,
            "plan_validity": self.plan_validity,
            "predicate_coverage": self.predicate_coverage,
            "logical_structure_accuracy": self.logical_structure_accuracy,
            "semantic_role_coverage": self.semantic_role_coverage,
            "tool_selection_f1": self.tool_selection_f1,
            "tool_sequence_accuracy": self.tool_sequence_accuracy,
            "invalid_tool_rate": self.invalid_tool_rate,
            "notes": list(self.notes),
        }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def evaluate_training_metrics(
    *,
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> TrainingMetrics:
    if not predicted or not expected:
        return TrainingMetrics(notes=["empty_predictions_or_targets"])

    total = min(len(predicted), len(expected))
    exact_json = 0
    exact_plan = 0
    predicate_hits = 0
    structure_hits = 0
    semantic_hits = 0
    tool_hits = 0
    tool_predicted = 0
    tool_expected = 0
    invalid_tools = 0

    for pred, exp in zip(predicted, expected):
        if pred == exp:
            exact_json += 1
        if pred.get("plan_valid") is True:
            exact_plan += 1
        pred_predicates = set(pred.get("predicate_keys") or [])
        exp_predicates = set(exp.get("predicate_keys") or [])
        if exp_predicates:
            predicate_hits += len(pred_predicates & exp_predicates) / len(exp_predicates)
        if pred.get("logical_structure") == exp.get("logical_structure"):
            structure_hits += 1
        pred_roles = set(pred.get("semantic_roles") or [])
        exp_roles = set(exp.get("semantic_roles") or [])
        if exp_roles:
            semantic_hits += len(pred_roles & exp_roles) / len(exp_roles)
        pred_tools = list(pred.get("tool_graph") or [])
        exp_tools = list(exp.get("tool_graph") or [])
        tool_predicted += len(pred_tools)
        tool_expected += len(exp_tools)
        tool_hits += len([tool for tool in pred_tools if tool in exp_tools])
        invalid_tools += sum(1 for tool in pred_tools if isinstance(tool, str) and tool.startswith("invalid."))

    precision = _safe_ratio(tool_hits, tool_predicted)
    recall = _safe_ratio(tool_hits, tool_expected)
    tool_selection_f1 = _safe_ratio(2 * precision * recall, precision + recall)

    return TrainingMetrics(
        json_validity=exact_json / total,
        plan_validity=exact_plan / total,
        predicate_coverage=predicate_hits / total,
        logical_structure_accuracy=structure_hits / total,
        semantic_role_coverage=semantic_hits / total,
        tool_selection_f1=tool_selection_f1,
        tool_sequence_accuracy=exact_json / total,
        invalid_tool_rate=_safe_ratio(invalid_tools, tool_predicted),
    )
