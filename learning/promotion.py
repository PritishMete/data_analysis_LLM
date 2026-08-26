from __future__ import annotations

from learning.config import PROMOTION_THRESHOLDS
from learning.models import SkillState


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
