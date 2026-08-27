from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ShadowPrediction:
    plan_source: str
    skill_id: str | None
    plan: dict[str, Any]


def compare_plans(predicted: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_match": predicted == reference,
        "predicted_keys": sorted(predicted.keys()),
        "reference_keys": sorted(reference.keys()),
    }
