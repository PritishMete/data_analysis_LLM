from __future__ import annotations

from learning.models import SkillState


def promotion_label(state: SkillState) -> str:
    if state.failure_count >= 3 and state.failure_count > state.success_count:
        return "demoted"
    if state.success_count >= 3 and state.average_quality_score >= 0.8:
        return "promoted"
    if state.success_count > 0:
        return "candidate"
    return "bootstrap"

