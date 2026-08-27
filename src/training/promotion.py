from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PromotionGateResult:
    promotable: bool
    blockers: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotable": self.promotable,
            "blockers": list(self.blockers),
            "thresholds": dict(self.thresholds),
        }


def evaluate_promotion_gates(*, readiness: dict[str, Any], metrics: dict[str, Any]) -> PromotionGateResult:
    thresholds = {
        "ready_for_prototype": True,
        "min_quality": 0.95,
        "min_train_count": 400,
        "min_validation_count": 40,
        "min_test_count": 40,
        "min_json_validity": 0.98,
        "min_plan_validity": 0.95,
        "min_tool_f1": 0.90,
        "max_invalid_tool_rate": 0.02,
    }
    blockers: list[str] = []
    if not readiness.get("ready_for_prototype"):
        blockers.append("dataset_not_ready")
    if float(metrics.get("json_validity") or 0.0) < thresholds["min_json_validity"]:
        blockers.append("json_validity_below_threshold")
    if float(metrics.get("plan_validity") or 0.0) < thresholds["min_plan_validity"]:
        blockers.append("plan_validity_below_threshold")
    if float(metrics.get("tool_selection_f1") or 0.0) < thresholds["min_tool_f1"]:
        blockers.append("tool_selection_below_threshold")
    if float(metrics.get("invalid_tool_rate") or 0.0) > thresholds["max_invalid_tool_rate"]:
        blockers.append("invalid_tool_rate_above_threshold")
    return PromotionGateResult(promotable=not blockers, blockers=blockers, thresholds=thresholds)
