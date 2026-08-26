from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from agent.critic import PlanCritic
from agent.evaluator import ResultEvaluator
from agent.planner import LearningPlanner
from agent.reflection import ReflectionEngine
from agent.result_validator import ResultValidator
from learning.candidate_strategy import promote_candidate_strategy, update_candidate_memory
from learning.experience_store import LearningExperienceStore
from learning.feature_extractor import build_planner_context
from learning.models import CandidateStrategy, ExperienceRecord, FailureLesson, LearningDecision, PlannerContext
from learning.retriever import rank_records, score_candidate_strategy, score_experience, score_failure_lesson
from learning.skill_registry import SkillRegistry, get_skill_registry


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LearningExecutionTracker(AbstractContextManager["LearningExecutionTracker"]):
    def __init__(
        self,
        orchestrator: "AgenticLearningOrchestrator",
        *,
        decision: LearningDecision,
        context: PlannerContext,
    ):
        self.orchestrator = orchestrator
        self.decision = decision
        self.context = context
        self.result_summary: dict[str, Any] | None = None
        self.success = False
        self.feedback_score: int | None = None
        self.failure_reason: str | None = None
        self._finalized = False

    def set_result(
        self,
        *,
        result_summary: dict[str, Any] | None = None,
        success: bool = True,
        feedback_score: int | None = None,
        failure_reason: str | None = None,
    ) -> None:
        if result_summary is not None:
            self.result_summary = result_summary
        self.success = success
        self.feedback_score = feedback_score
        self.failure_reason = failure_reason

    def finalize(self) -> ExperienceRecord | None:
        if self._finalized:
            return None
        self._finalized = True
        return self.orchestrator.record_result(
            user_text="",
            decision=self.decision,
            df=None,
            available_columns=self.context.dataset_profile.get("available_columns") or [],
            result_summary=self.result_summary,
            success=self.success,
            feedback_score=self.feedback_score,
            failure_reason=self.failure_reason,
            planner_context=self.context,
        )

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        if exc_type is not None:
            self.set_result(success=False, failure_reason=str(exc_value))
        self.finalize()
        return False


class AgenticLearningOrchestrator:
    def __init__(
        self,
        registry: SkillRegistry | None = None,
        store: LearningExperienceStore | None = None,
    ):
        self.registry = registry or get_skill_registry()
        self.store = store or LearningExperienceStore()
        self.planner = LearningPlanner(registry=self.registry)
        self.critic = PlanCritic()
        self.evaluator = ResultEvaluator()
        self.reflection = ReflectionEngine()
        self.validator = ResultValidator()

    def build_context(
        self,
        user_text: str,
        df: pd.DataFrame | None = None,
        available_columns: list[str] | None = None,
    ) -> PlannerContext:
        context = build_planner_context(user_text, df, available_columns)
        recent = [ExperienceRecord.from_dict(item) for item in self.store.load_recent(limit=25)]
        lessons = [FailureLesson.from_dict(item) for item in self.store.load_failure_lessons(limit=25)]
        candidates = [CandidateStrategy.from_dict(item) for item in self.store.load_candidate_strategies(limit=25)]

        similar_experiences = rank_records(recent, score_experience, context.features, limit=5)
        failure_lessons = rank_records(lessons, score_failure_lesson, context.features, limit=5)
        candidate_strategies = rank_records(candidates, score_candidate_strategy, context.features, limit=5)

        context.similar_experiences = similar_experiences
        context.failure_lessons = failure_lessons
        context.candidate_strategies = candidate_strategies
        context.retrieval_trace = {
            "experience_count": len(similar_experiences),
            "failure_lesson_count": len(failure_lessons),
            "candidate_strategy_count": len(candidate_strategies),
            "dataset_semantic_signature": context.features.dataset_semantic_signature,
        }
        return context

    def plan(
        self,
        user_text: str,
        df: pd.DataFrame | None = None,
        available_columns: list[str] | None = None,
        planner_context: PlannerContext | None = None,
    ) -> LearningDecision:
        context = planner_context or self.build_context(user_text, df=df, available_columns=available_columns)
        decision = self.planner.plan(user_text, df, available_columns, planner_context=context)
        ok, notes = self.critic.review(decision, context=context)
        if not ok:
            return LearningDecision(
                route="unknown",
                confidence=0.0,
                message="A learned plan was considered unsafe or incomplete.",
                validation_notes=notes,
                features=decision.features,
                retrieval_trace=decision.retrieval_trace,
            )
        decision.validation_notes = notes
        return decision

    def track(
        self,
        *,
        user_text: str,
        df: pd.DataFrame | None = None,
        available_columns: list[str] | None = None,
        decision: LearningDecision | None = None,
    ) -> LearningExecutionTracker:
        context = self.build_context(user_text, df=df, available_columns=available_columns)
        decision = decision or self.plan(user_text, df=df, available_columns=available_columns, planner_context=context)
        return LearningExecutionTracker(self, decision=decision, context=context)

    def _sanitize_result_summary(self, result_summary: dict[str, Any] | None) -> dict[str, Any]:
        summary = dict(result_summary or {})
        allowed = {
            "result_kind",
            "row_count",
            "column_count",
            "shape",
            "summary_kind",
            "action",
            "confidence",
            "error",
        }
        return {key: summary.get(key) for key in allowed if key in summary}

    def _build_plan_summary(self, decision: LearningDecision) -> dict[str, Any]:
        plan = decision.plan or {}
        return {
            "route": decision.route,
            "skill_id": decision.skill_id,
            "plan_keys": sorted(plan.keys()),
            "filter_count": len(plan.get("filters") or []),
            "metric_count": len(plan.get("metrics") or []),
            "has_group_by": bool(plan.get("group_by")),
            "has_window": bool(plan.get("window")),
            "has_categorize": "categorize" in plan,
        }

    def _learn_from_experience(self, experience: ExperienceRecord, context: PlannerContext) -> None:
        recent = [ExperienceRecord.from_dict(item) for item in self.store.load_recent(limit=25)]
        candidate = update_candidate_memory(experience=experience, recent_experiences=recent)
        if candidate is not None:
            candidate.last_seen_at = experience.created_at
            self.store.append_candidate_strategy(candidate)
            self.store.update_candidate_strategy(candidate)
            promoted, spec = promote_candidate_strategy(candidate)
            if promoted and spec is not None:
                candidate.state = "promoted"
                candidate.promoted_at = experience.created_at
                candidate.validated_at = experience.created_at
                self.store.append_candidate_strategy(candidate)
                self.store.update_candidate_strategy(candidate)
                self.registry.register_dynamic_skill(spec)

        if not experience.success and experience.failure_reason:
            lesson = {
                "lesson_id": f"lesson.{experience.semantic_signature[:12]}",
                "intent": experience.intent,
                "failure_signature": experience.semantic_signature,
                "condition_structure": experience.logical_structure,
                "lesson": "Preserve every explicit predicate and validate the output before execution.",
                "severity": "high" if experience.score < 0.5 else "medium",
                "semantic_roles": experience.semantic_roles,
                "operators": experience.operators,
                "tool_sequence": experience.tool_sequence,
                "occurrence_count": 1,
                "average_quality": experience.score,
            }
            from learning.models import FailureLesson

            self.store.append_failure_lesson(FailureLesson.from_dict(lesson))

    def record_result(
        self,
        *,
        user_text: str,
        decision: LearningDecision,
        df: pd.DataFrame | None = None,
        available_columns: list[str] | None = None,
        result_summary: dict[str, Any] | None = None,
        success: bool = True,
        feedback_score: int | None = None,
        failure_reason: str | None = None,
        planner_context: PlannerContext | None = None,
    ) -> ExperienceRecord:
        context = planner_context or self.build_context(user_text, df=df, available_columns=available_columns)
        result_summary = self._sanitize_result_summary(result_summary)
        validation = self.validator.validate(decision, result_summary)
        final_success = bool(success and validation.success)
        if not final_success and validation.notes and not failure_reason:
            failure_reason = "; ".join(validation.notes)
        quality = self.evaluator.score(success=final_success, result_summary=result_summary, feedback_score=feedback_score)
        before, after = self.registry.update_from_experience(
            decision.skill_id,
            success=final_success,
            score=quality,
            now_iso=_utcnow_iso(),
        )
        if decision.skill_id:
            self.store.update_skill_state(decision.skill_id, before, after)
        experience = self.reflection.build_experience(
            decision=decision,
            features=context.features,
            success=final_success,
            quality_score=quality,
            result_summary=result_summary,
            failure_reason=failure_reason,
            feedback_score=feedback_score,
            skill_state_before=before,
            skill_state_after=after,
        )
        stored = self.store.append(experience)
        self._learn_from_experience(stored, context)
        return stored

    def result_summary_for_dataframe(self, df: pd.DataFrame | None) -> dict[str, Any]:
        if df is None:
            return {}
        return {
            "result_kind": "table",
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "shape": [int(len(df)), int(len(df.columns))],
        }


_ORCHESTRATOR: AgenticLearningOrchestrator | None = None


def get_agentic_orchestrator() -> AgenticLearningOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = AgenticLearningOrchestrator()
    return _ORCHESTRATOR
