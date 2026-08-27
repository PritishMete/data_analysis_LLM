from __future__ import annotations

from typing import Any

from learning.config import PROMOTION_THRESHOLDS
from learning.models import SkillState


_LIFECYCLE_RANKS = {
    "demoted": -1,
    "bootstrap": 0,
    "observed": 0,
    "candidate": 1,
    "validated": 2,
    "promoted": 2,
    "trusted": 3,
}


def promotion_label(state: SkillState) -> str:
    if state.failure_count >= 3 and state.failure_count > state.success_count:
        return "demoted"
    if state.success_count >= PROMOTION_THRESHOLDS["trusted_successes"] and state.average_quality_score >= PROMOTION_THRESHOLDS["trusted_quality"]:
        return "trusted"
    if state.success_count >= PROMOTION_THRESHOLDS["validated_successes"] and state.average_quality_score >= PROMOTION_THRESHOLDS["validated_quality"]:
        return "validated"
    if state.success_count >= PROMOTION_THRESHOLDS["candidate_successes"]:
        return "candidate"
    return "bootstrap"


def lifecycle_rank(label: str | None) -> int:
    return _LIFECYCLE_RANKS.get(str(label or "bootstrap"), 0)


def strategy_next_stage(label: str | None) -> str | None:
    current = str(label or "bootstrap")
    if current in {"demoted", "trusted"}:
        return None
    if current in {"bootstrap", "observed"}:
        return "candidate"
    if current == "candidate":
        return "validated"
    if current == "validated":
        return "trusted"
    if current == "promoted":
        return "validated"
    return "candidate"


def strategy_next_requirements(label: str | None) -> dict[str, Any]:
    next_stage = strategy_next_stage(label)
    if next_stage is None:
        return {
            "next_stage": None,
            "required_evidence": None,
            "required_quality": None,
            "failure_tolerance": 3,
        }
    if next_stage == "candidate":
        return {
            "next_stage": "candidate",
            "required_evidence": PROMOTION_THRESHOLDS["candidate_successes"],
            "required_quality": None,
            "failure_tolerance": 3,
        }
    if next_stage == "validated":
        return {
            "next_stage": "validated",
            "required_evidence": PROMOTION_THRESHOLDS["validated_successes"],
            "required_quality": PROMOTION_THRESHOLDS["validated_quality"],
            "failure_tolerance": 3,
        }
    return {
        "next_stage": "trusted",
        "required_evidence": PROMOTION_THRESHOLDS["trusted_successes"],
        "required_quality": PROMOTION_THRESHOLDS["trusted_quality"],
        "failure_tolerance": 3,
    }


def strategy_promotion_blockers(
    *,
    label: str | None,
    evidence_count: int,
    average_quality: float,
    failure_count: int = 0,
    success_count: int | None = None,
) -> list[str]:
    blockers: list[str] = []
    current = str(label or "bootstrap")
    if current == "demoted":
        blockers.append("demoted")
    if failure_count >= 3 and (success_count is None or failure_count > success_count):
        blockers.append("failure_tolerance_exceeded")
    requirements = strategy_next_requirements(current)
    required_evidence = requirements.get("required_evidence")
    required_quality = requirements.get("required_quality")
    if isinstance(required_evidence, int) and evidence_count < required_evidence:
        blockers.append(f"need_{required_evidence - evidence_count}_more_evidence")
    if isinstance(required_quality, (int, float)) and average_quality < float(required_quality):
        blockers.append(f"quality_below_{float(required_quality):.2f}")
    return blockers
