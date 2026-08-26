from __future__ import annotations

from learning.models import ExperienceRecord, LearningDecision, QueryFeatures, SkillState, stable_hash


class ReflectionEngine:
    def build_experience(
        self,
        *,
        decision: LearningDecision,
        features: QueryFeatures,
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
                "filter_count": len(decision.plan.get("filters") or []),
                "metric_count": len(decision.plan.get("metrics") or []),
                "has_group_by": bool(decision.plan.get("group_by")),
                "has_window": bool(decision.plan.get("window")),
                "has_categorize": "categorize" in decision.plan,
            }
        semantic_signature = features.semantic_signature
        return ExperienceRecord(
            intent=features.intent,
            query_features=features.to_dict(),
            semantic_roles=list(features.semantic_roles),
            operators=list(features.operators),
            logical_structure=features.logical_structure,
            tool_sequence=list(decision.tool_sequence or []),
            result_summary=result_summary or {},
            dataset_semantic_signature=features.dataset_semantic_signature,
            semantic_signature=semantic_signature,
            route=decision.route,
            skill_id=decision.skill_id,
            confidence=decision.confidence,
            success=success,
            score=quality_score,
            plan_hash=stable_hash(decision.plan) if decision.plan else None,
            plan_summary=plan_summary,
            failure_reason=failure_reason,
            feedback_score=feedback_score,
            skill_state_before=skill_state_before.to_dict() if skill_state_before is not None else None,
            skill_state_after=skill_state_after.to_dict() if skill_state_after is not None else None,
            candidate_strategy_id=None,
        )
