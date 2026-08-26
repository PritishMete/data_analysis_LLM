from __future__ import annotations

from learning.models import ExperienceRecord, LearningDecision, SkillState
from learning.models import stable_hash


class ReflectionEngine:
    def build_experience(
        self,
        *,
        decision: LearningDecision,
        query_text: str,
        normalized_query: str,
        schema_signature: str | None,
        success: bool,
        quality_score: float,
        result_summary: dict | None = None,
        failure_reason: str | None = None,
        feedback_score: int | None = None,
        skill_state_before: SkillState | None = None,
        skill_state_after: SkillState | None = None,
    ) -> ExperienceRecord:
        plan_summary = {}
        if decision.plan:
            plan_summary = {
                "route": decision.route,
                "skill_id": decision.skill_id,
                "plan_keys": sorted(decision.plan.keys()),
            }
        return ExperienceRecord(
            query_text=query_text,
            normalized_query=normalized_query,
            schema_signature=schema_signature,
            route=decision.route,
            skill_id=decision.skill_id,
            confidence=decision.confidence,
            success=success,
            score=quality_score,
            plan_hash=stable_hash(decision.plan) if decision.plan else None,
            plan_summary=plan_summary,
            result_summary=result_summary or {},
            failure_reason=failure_reason,
            feedback_score=feedback_score,
            skill_state_before=skill_state_before.to_dict() if skill_state_before is not None else None,
            skill_state_after=skill_state_after.to_dict() if skill_state_after is not None else None,
        )
