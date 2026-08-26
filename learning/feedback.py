from __future__ import annotations

from learning.models import SkillState


def feedback_adjustment(state: SkillState, feedback_score: int | None) -> float:
    if feedback_score is None:
        return 0.0
    if feedback_score > 0:
        return 0.05
    if feedback_score < 0:
        return -0.07
    return 0.0
