from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any
import hashlib
import json

import pandas as pd

from agent.critic import PlanCritic
from agent.evaluator import ResultEvaluator
from agent.planner import LearningPlanner
from agent.reflection import ReflectionEngine
from learning.experience_store import LearningExperienceStore
from learning.models import ExperienceRecord, LearningDecision, SkillState
from learning.skill_registry import SkillRegistry, get_skill_registry


def _schema_signature(columns: list[str] | None, df: pd.DataFrame | None) -> str | None:
    if df is not None:
        payload = {
            "columns": [str(column) for column in df.columns],
            "dtypes": [str(dtype) for dtype in df.dtypes.tolist()],
        }
    elif columns is not None:
        payload = {"columns": [str(column) for column in columns]}
    else:
        return None
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LearningExecutionTracker(AbstractContextManager["LearningExecutionTracker"]):
    def __init__(
        self,
        orchestrator: "AgenticLearningOrchestrator",
        *,
        decision: LearningDecision,
        query_text: str,
        normalized_query: str,
        schema_signature: str | None,
    ):
        self.orchestrator = orchestrator
        self.decision = decision
        self.query_text = query_text
        self.normalized_query = normalized_query
        self.schema_signature = schema_signature
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
        quality = self.orchestrator.evaluator.score(
            success=self.success,
            result_summary=self.result_summary,
            feedback_score=self.feedback_score,
        )
        before, after = self.orchestrator.registry.update_from_experience(
            self.decision.skill_id,
            success=self.success,
            score=quality,
        )
        self.orchestrator.store.update_skill_state(self.decision.skill_id or "", before, after)
        experience = self.orchestrator.reflection.build_experience(
            decision=self.decision,
            query_text=self.query_text,
            normalized_query=self.normalized_query,
            schema_signature=self.schema_signature,
            success=self.success,
            quality_score=quality,
            result_summary=self.result_summary,
            failure_reason=self.failure_reason,
            feedback_score=self.feedback_score,
            skill_state_before=before,
            skill_state_after=after,
        )
        self.orchestrator.store.append(experience)
        return experience

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
        self.planner = LearningPlanner()
        self.critic = PlanCritic()
        self.evaluator = ResultEvaluator()
        self.reflection = ReflectionEngine()

    def plan(self, user_text: str, df: pd.DataFrame | None = None, available_columns: list[str] | None = None) -> LearningDecision:
        decision = self.planner.plan(user_text, df, available_columns)
        ok, notes = self.critic.review(decision)
        if not ok:
            return LearningDecision(
                route="unknown",
                confidence=0.0,
                message="A learned plan was considered unsafe or incomplete.",
                validation_notes=notes,
                features=decision.features,
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
        normalized = (user_text or "").strip().lower()
        schema_signature = _schema_signature(available_columns, df)
        decision = decision or self.plan(user_text, df=df, available_columns=available_columns)
        return LearningExecutionTracker(
            self,
            decision=decision,
            query_text=user_text,
            normalized_query=normalized,
            schema_signature=schema_signature,
        )

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
    ) -> ExperienceRecord:
        normalized = (user_text or "").strip().lower()
        schema_signature = _schema_signature(available_columns, df)
        quality = self.evaluator.score(success=success, result_summary=result_summary, feedback_score=feedback_score)
        before, after = self.registry.update_from_experience(
            decision.skill_id,
            success=success,
            score=quality,
        )
        self.store.update_skill_state(decision.skill_id or "", before, after)
        experience = self.reflection.build_experience(
            decision=decision,
            query_text=user_text,
            normalized_query=normalized,
            schema_signature=schema_signature,
            success=success,
            quality_score=quality,
            result_summary=result_summary,
            failure_reason=failure_reason,
            feedback_score=feedback_score,
            skill_state_before=before,
            skill_state_after=after,
        )
        return self.store.append(experience)

    def result_summary_for_dataframe(self, df: pd.DataFrame | None) -> dict[str, Any]:
        if df is None:
            return {}
        return {
            "result_kind": "table",
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": [str(column) for column in df.columns[:10]],
        }


_ORCHESTRATOR: AgenticLearningOrchestrator | None = None


def get_agentic_orchestrator() -> AgenticLearningOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = AgenticLearningOrchestrator()
    return _ORCHESTRATOR
