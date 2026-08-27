from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EvaluationResult:
    score: float
    notes: list[str]


def evaluate_shadow_predictions(*, predicted: list[dict[str, Any]], expected: list[dict[str, Any]]) -> EvaluationResult:
    if not predicted or not expected:
        return EvaluationResult(score=0.0, notes=["empty_predictions_or_targets"])
    exact = sum(1 for p, e in zip(predicted, expected) if p == e)
    return EvaluationResult(score=exact / min(len(predicted), len(expected)), notes=[])
