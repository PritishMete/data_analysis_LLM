from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from agent.orchestrator import AgenticLearningOrchestrator
from learning.models import CandidateStrategy, CorrectionRecord, DatasetSemanticProfile, LearningDecision, PlannerContext, QueryFeatures, stable_hash
from learning.training_export import TrainingDatasetExporter, TrainingExportBundle, TrainingExportPolicy
from learning.skill_registry import SkillRegistry
from learning.experience_store import LearningExperienceStore

from .schemas import (
    ExperienceRequest,
    ExperienceResponse,
    FeedbackRequest,
    FeedbackResponse,
    MetricsResponse,
    PlanRequest,
    PlanResponse,
    SkillResponse,
)


def _state_root() -> Path:
    root = os.environ.get("INSIGHT_LEARNING_RUNTIME_DIR") or os.environ.get("DATA_ANALYSIS_LLM_STATE_DIR")
    if root:
        return Path(root)
    return Path("runtime")


def _synthetic_user_text(intent: str, query_features: dict[str, Any], dataset_profile: dict[str, Any]) -> str:
    roles = " ".join(query_features.get("semantic_roles") or [])
    operators = " ".join(query_features.get("operators") or [])
    field_roles = " ".join(str(field.get("semantic_role") or "") for field in dataset_profile.get("fields", []))
    return " ".join(part for part in [intent, roles, operators, field_roles] if part).strip()


def _query_features_payload(request: PlanRequest | ExperienceRequest) -> QueryFeatures:
    payload = request.query_features.model_dump()
    roles = list(payload.get("semantic_roles") or [])
    operators = list(payload.get("operators") or [])
    return QueryFeatures(
        intent=request.intent,
        predicate_count=int(payload.get("predicate_count") or len(operators)),
        boolean_predicate_count=sum(1 for role in roles if role in {"boolean", "boolean_capability"}),
        numeric_comparison_count=sum(1 for op in operators if op in {"greater_than", "greater_than_equal", "less_than", "less_than_equal", "between"}),
        entity_reference_count=0,
        logical_structure=str(payload.get("logical_structure") or "SINGLE"),
        semantic_roles=roles,
        operators=operators,
        operation_hints=[request.intent],
        tool_hints=["sql.filter" if request.intent == "filter" else "sql.group_by" if request.intent in {"aggregate", "rank"} else "analytics.summary"],
        query_shape="statement",
        dataset_semantic_signature=None,
        semantic_signature=stable_hash({"intent": request.intent, "roles": roles, "operators": operators}),
        confidence=0.8,
    )


def _dataset_profile_payload(dataset_profile: dict[str, Any]) -> DatasetSemanticProfile:
    raw_fields = list(dataset_profile.get("fields") or [])
    fields = [
        {
            "id": str(field.get("id") or ""),
            "semantic_role": str(field.get("semantic_role") or "unknown"),
            "dtype": str(field.get("dtype") or "unknown"),
        }
        for field in raw_fields
        if isinstance(field, dict)
    ]
    available_columns = [field["id"] for field in fields if field["id"]]
    column_roles = {field["id"]: field["semantic_role"] for field in fields if field["id"]}
    return DatasetSemanticProfile(
        available_columns=available_columns,
        safe_profile={"fields": fields},
        column_roles=column_roles,
        dataset_semantic_signature=stable_hash({"fields": fields}) if fields else None,
    )


class InsightLearningService:
    def __init__(self) -> None:
        root = _state_root()
        self.store = LearningExperienceStore(root=root)
        self.registry = SkillRegistry(state_path=root / "skills_state.json")
        self.orchestrator = AgenticLearningOrchestrator(registry=self.registry, store=self.store)
        self.training_export_policy = TrainingExportPolicy.from_env()

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "service": "insight-learning"}

    def plan(self, request: PlanRequest) -> dict[str, Any]:
        features = _query_features_payload(request)
        dataset_profile = _dataset_profile_payload(request.dataset_profile.model_dump())
        context = PlannerContext(
            features=features,
            dataset_profile=dataset_profile.to_dict() | {"available_columns": dataset_profile.available_columns},
            dataset_semantic_profile=dataset_profile,
            retrieval_trace={},
        )
        decision = self.orchestrator.planner.plan(
            _synthetic_user_text(request.intent, request.query_features.model_dump(), request.dataset_profile.model_dump()),
            df=None,
            available_columns=dataset_profile.available_columns,
            planner_context=context,
        )
        critic_ok, critic_notes = self.orchestrator.critic.review(decision, context=context)
        return {
            "plan_source": decision.plan_source,
            "confidence": decision.confidence,
            "tool_graph": list(decision.tool_sequence or (decision.plan or {}).get("tool_sequence") or []),
            "plan_template_id": decision.plan_template_id,
            "critic_status": {"passed": critic_ok, "notes": list(dict.fromkeys([*(decision.validation_notes or []), *critic_notes]))},
            "plan": decision.plan,
            "skill_id": decision.skill_id,
        }

    def experience(self, request: ExperienceRequest) -> dict[str, Any]:
        features = _query_features_payload(request)
        dataset_profile_payload = request.dataset_profile.model_dump() if request.dataset_profile is not None else {"fields": []}
        dataset_profile = _dataset_profile_payload(dataset_profile_payload)
        context = PlannerContext(
            features=features,
            dataset_profile=dataset_profile.to_dict() | {"available_columns": dataset_profile.available_columns},
            dataset_semantic_profile=dataset_profile,
            retrieval_trace={},
        )
        tool_sequence = list(request.plan.get("tool_sequence") or [])
        route = "sql" if any(tool.startswith("sql") for tool in tool_sequence) or request.intent in {"filter", "aggregate", "rank", "compare", "trend"} else "operation"
        decision = LearningDecision(
            route=route,
            confidence=min(0.99, max(0.0, float(request.quality_score))),
            message="safe learning event",
            plan=dict(request.plan),
            features=features.to_dict(),
            retrieval_trace={},
            tool_sequence=tool_sequence,
            plan_source=str(request.plan_source or "validated_template"),
            plan_template_id=request.plan_template_id,
        )
        stored = self.orchestrator.record_result(
            user_text=_synthetic_user_text(request.intent, request.query_features.model_dump(), {"fields": []}),
            decision=decision,
            df=None,
            available_columns=[],
            result_summary={
                "result_kind": "table" if request.validation.get("success") else "error",
                "row_count": 1 if request.validation.get("success") else None,
                "column_count": max(1, len(dataset_profile.available_columns)) if request.validation.get("success") else None,
                "quality": request.quality_score,
            },
            success=bool(request.execution.get("success")) and bool(request.validation.get("success")),
            feedback_score=None,
            event_id=request.event_id,
            critic_passed=request.critic_passed,
            result_validation_passed=request.result_validation_passed,
            plan_completeness_passed=request.plan_completeness_passed,
            privacy_validation_passed=request.privacy_validation_passed,
            no_unresolved_ambiguity=request.no_unresolved_ambiguity,
            no_critical_repair=request.no_critical_repair,
            correction_state=request.correction_state,
            planner_context=context,
        )
        return {
            "stored": True,
            "learning_outcome": {
                "experience_id": stored.semantic_signature,
                "plan_source": stored.plan_source,
                "quality_score": stored.score,
            },
        }

    def feedback(self, request: FeedbackRequest) -> dict[str, Any]:
        outcome: dict[str, Any] = {"accepted": True}
        if request.correction_type and request.affected_intent and request.generalized_lesson:
            correction = self.orchestrator.record_correction(
                decision=LearningDecision(route="unknown", confidence=0.0, message="feedback"),
                correction_type=request.correction_type,
                affected_intent=request.affected_intent,
                generalized_lesson=request.generalized_lesson,
                dataset_semantic_signature=request.dataset_semantic_signature,
                requested_role=request.requested_role,
                resolution_preference=request.resolution_preference,
                preferred_semantic_candidate=request.preferred_semantic_candidate,
            )
            outcome["correction_id"] = correction.correction_id
        return outcome

    def skills(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.registry.all()]

    def export_training_dataset(
        self,
        *,
        include_candidate_strategies: bool = True,
        limit: int = 1000,
    ) -> TrainingExportBundle:
        exporter = TrainingDatasetExporter(self.store, self.training_export_policy)
        bundle, _ = exporter.build_bundle(limit=limit, include_candidate_strategies=include_candidate_strategies)
        return bundle

    def export_training_dataset_files(
        self,
        *,
        output_dir: Path | None = None,
        include_candidate_strategies: bool = True,
        limit: int = 1000,
    ) -> dict[str, str]:
        exporter = TrainingDatasetExporter(self.store, self.training_export_policy)
        paths = exporter.export_files(
            output_dir=output_dir,
            include_candidate_strategies=include_candidate_strategies,
            limit=limit,
        )
        return {key: str(path) for key, path in paths.items()}

    def metrics(self) -> dict[str, Any]:
        return {
            "experiences": len(self.store.load_recent(limit=10_000)),
            "plan_templates": len(self.store.load_plan_templates(limit=10_000)),
            "candidate_strategies": len(self.store.load_candidate_strategies(limit=10_000)),
            "failure_lessons": len(self.store.load_failure_lessons(limit=10_000)),
            "corrections": len(self.store.load_corrections(limit=10_000)),
            "skills": len(self.registry.all()),
        }


_SERVICE: InsightLearningService | None = None


def get_service() -> InsightLearningService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = InsightLearningService()
    return _SERVICE


def create_app() -> FastAPI:
    app = FastAPI(title="Insight Learning", version="0.1.0")

    from .routes_plan import router as plan_routes
    from .routes_experience import router as experience_routes
    from .routes_feedback import router as feedback_routes
    from .routes_export import router as export_routes
    from .routes_skills import router as skills_routes

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return get_service().health()

    @app.get("/v1/metrics")
    def metrics() -> dict[str, Any]:
        return {"metrics": get_service().metrics()}

    app.include_router(plan_routes)
    app.include_router(experience_routes)
    app.include_router(feedback_routes)
    app.include_router(export_routes)
    app.include_router(skills_routes)
    return app


app = create_app()
