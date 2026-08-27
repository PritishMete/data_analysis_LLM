from __future__ import annotations

import os
from pathlib import Path
from collections import Counter
from typing import Any

from fastapi import FastAPI

from agent.orchestrator import AgenticLearningOrchestrator
from learning.canonical_training import PlannerTrainingBackend
from learning.config import PROMOTION_THRESHOLDS
from learning.models import CandidateStrategy, CorrectionRecord, DatasetSemanticProfile, LearningDecision, PlannerContext, QueryFeatures, stable_hash
from learning.training_export import TrainingDatasetExporter, TrainingExportBundle, TrainingExportPolicy
from learning.skill_registry import SkillRegistry
from learning.experience_store import LearningExperienceStore
from learning.promotion import lifecycle_rank, strategy_next_requirements, strategy_promotion_blockers

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
        self.training_backend = PlannerTrainingBackend()
        self.learner_plan_requests = 0
        self.learner_plan_accepts = 0
        self.learner_plan_rejections = 0
        self.learner_experience_requests = 0
        self.learner_experience_accepts = 0
        self.last_safe_plan_source: str | None = None

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "service": "insight-learning"}

    def plan(self, request: PlanRequest) -> dict[str, Any]:
        self.learner_plan_requests += 1
        features = _query_features_payload(request)
        dataset_profile = _dataset_profile_payload(request.dataset_profile.model_dump())
        planner_text = request.text or _synthetic_user_text(request.intent, request.query_features.model_dump(), request.dataset_profile.model_dump())
        context = PlannerContext(
            features=features,
            dataset_profile=dataset_profile.to_dict() | {"available_columns": dataset_profile.available_columns},
            dataset_semantic_profile=dataset_profile,
            retrieval_trace={},
        )
        decision = self.orchestrator.planner.plan(
            planner_text,
            df=None,
            available_columns=dataset_profile.available_columns,
            planner_context=context,
        )
        critic_ok, critic_notes = self.orchestrator.critic.review(decision, context=context)
        accepted = bool(critic_ok and decision.plan and decision.plan_source != "deterministic_fallback" and decision.confidence >= 0.82)
        if accepted:
            self.learner_plan_accepts += 1
        else:
            self.learner_plan_rejections += 1
        self.last_safe_plan_source = decision.plan_source
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
        self.learner_experience_requests += 1
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
        if stored.success:
            self.learner_experience_accepts += 1
        self.last_safe_plan_source = stored.plan_source
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

    def build_training_dataset_manifest(
        self,
        *,
        include_candidate_strategies: bool = True,
        limit: int = 1000,
    ) -> dict[str, Any]:
        exporter = TrainingDatasetExporter(self.store, self.training_export_policy)
        bundle, _ = exporter.build_bundle(limit=limit, include_candidate_strategies=include_candidate_strategies)
        manifest = exporter.build_manifest(bundle)
        return manifest.to_dict()

    def evaluate_training_dataset_readiness(
        self,
        *,
        include_candidate_strategies: bool = True,
        limit: int = 1000,
    ) -> dict[str, Any]:
        exporter = TrainingDatasetExporter(self.store, self.training_export_policy)
        bundle, _ = exporter.build_bundle(limit=limit, include_candidate_strategies=include_candidate_strategies)
        readiness = exporter.evaluate_readiness(bundle)
        return readiness.to_dict()

    def create_training_dataset(
        self,
        *,
        include_candidate_strategies: bool = True,
        limit: int = 1000,
    ) -> dict[str, Any]:
        paths = self.export_training_dataset_files(
            include_candidate_strategies=include_candidate_strategies,
            limit=limit,
        )
        manifest = self.build_training_dataset_manifest(
            include_candidate_strategies=include_candidate_strategies,
            limit=limit,
        )
        return {"manifest": manifest, "paths": paths}

    def invalidate_training_candidate(
        self,
        *,
        source_id: str | None = None,
        family_fingerprint: str | None = None,
        reason: str = "manual",
    ) -> dict[str, Any]:
        exporter = TrainingDatasetExporter(self.store, self.training_export_policy)
        return exporter.invalidate_training_candidate(
            source_id=source_id,
            family_fingerprint=family_fingerprint,
            reason=reason,
        )

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

    def learning_status(self) -> dict[str, Any]:
        exporter = TrainingDatasetExporter(self.store, self.training_export_policy)
        bundle, _ = exporter.build_bundle(include_candidate_strategies=True)
        report = bundle.report()
        recent = self.store.load_recent(limit=1)
        recent_plan_source = None
        if recent:
            recent_plan_source = str(recent[0].get("plan_source") or "")
        strategies = self.store.load_candidate_strategies(limit=10_000)
        trusted_strategies = sum(1 for item in strategies if str(item.get("state") or "") == "trusted")
        readiness = exporter.evaluate_readiness(bundle)
        return {
            "experience_count": len(self.store.load_recent(limit=10_000)),
            "eligible_experience_count": report.get("eligible_examples", 0),
            "rejected_experience_count": report.get("rejected_examples", 0),
            "rejected_by_reason": report.get("rejection_reasons", {}),
            "plan_template_count": len(self.store.load_plan_templates(limit=10_000)),
            "candidate_strategy_count": len(strategies),
            "trusted_strategy_count": trusted_strategies,
            "failure_lesson_count": len(self.store.load_failure_lessons(limit=10_000)),
            "correction_count": len(self.store.load_corrections(limit=10_000)),
            "last_safe_plan_source": recent_plan_source or self.last_safe_plan_source,
            "learner_plan_requests": self.learner_plan_requests,
            "learner_plan_accepts": self.learner_plan_accepts,
            "learner_plan_rejections": self.learner_plan_rejections,
            "learner_experience_requests": self.learner_experience_requests,
            "learner_experience_accepts": self.learner_experience_accepts,
            "privacy_gate_passed": bool(report.get("privacy_rejections", 0) == 0 and report.get("eligible_examples", 0) > 0),
            "readiness": {
                "ready": readiness.ready,
                "ready_for_prototype": readiness.ready_for_prototype,
                "reason": readiness.reason,
            },
            "strategy_status": self.strategy_status(),
        }

    def strategy_status(self) -> dict[str, Any]:
        candidate_strategies = self.store.load_candidate_strategies(limit=10_000)
        registry_states = self.registry.snapshot().states
        statuses: list[dict[str, Any]] = []
        lifecycle_counts: Counter[str] = Counter()
        trusted_count = 0
        validated_count = 0
        candidate_count = 0
        total_evidence = 0
        total_quality = 0.0

        for item in candidate_strategies:
            strategy_id = str(item.get("strategy_id") or "")
            learned_skill_id = strategy_id.replace("strategy.", "learned.", 1) if strategy_id.startswith("strategy.") else strategy_id
            registry_state = registry_states.get(learned_skill_id)
            lifecycle = str(item.get("lifecycle_state") or item.get("state") or "observed")
            evidence_count = int(item.get("evidence_count") or 0)
            success_count = evidence_count
            failure_count = 0
            average_quality = float(item.get("average_quality") or 0.0)
            registry_lifecycle = None
            if registry_state is not None:
                registry_lifecycle = registry_state.state
                evidence_count = max(evidence_count, registry_state.success_count + registry_state.failure_count)
                success_count = registry_state.success_count
                failure_count = registry_state.failure_count
                average_quality = registry_state.average_quality_score or average_quality
                if lifecycle_rank(registry_state.state) > lifecycle_rank(lifecycle):
                    lifecycle = registry_state.state

            next_requirements = strategy_next_requirements(lifecycle)
            blockers = strategy_promotion_blockers(
                label=lifecycle,
                evidence_count=evidence_count,
                average_quality=average_quality,
                failure_count=failure_count,
                success_count=success_count,
            )
            status = {
                "strategy_id": strategy_id,
                "learned_skill_id": learned_skill_id,
                "intent": str(item.get("intent") or "unknown"),
                "lifecycle": lifecycle,
                "registry_lifecycle": registry_lifecycle,
                "candidate_state": str(item.get("state") or "candidate"),
                "evidence_count": evidence_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "average_quality": round(average_quality, 4),
                "required_evidence_for_next_stage": next_requirements["required_evidence"],
                "required_quality_for_next_stage": next_requirements["required_quality"],
                "failure_tolerance": next_requirements["failure_tolerance"],
                "promotion_blockers": blockers,
                "tool_graph_signature": stable_hash({"tool_sequence": list(item.get("tool_sequence") or [])})[:16],
                "semantic_role_signature": stable_hash({"semantic_roles": list(item.get("semantic_roles") or [])})[:16],
                "plan_template_id": item.get("plan_template_id"),
            }
            statuses.append(status)
            lifecycle_counts[lifecycle] += 1
            total_evidence += evidence_count
            total_quality += average_quality
            if lifecycle == "trusted":
                trusted_count += 1
            elif lifecycle == "validated":
                validated_count += 1
            elif lifecycle in {"candidate", "promoted"}:
                candidate_count += 1

        statuses.sort(key=lambda item: (lifecycle_rank(item["lifecycle"]), item["evidence_count"], item["average_quality"], item["strategy_id"]), reverse=True)
        average_quality = round((total_quality / len(statuses)) if statuses else 0.0, 4)
        average_evidence = round((total_evidence / len(statuses)) if statuses else 0.0, 4)
        return {
            "strategy_count": len(statuses),
            "candidate_strategy_count": candidate_count,
            "validated_strategy_count": validated_count,
            "trusted_strategy_count": trusted_count,
            "average_evidence_count": average_evidence,
            "average_quality": average_quality,
            "lifecycle_counts": dict(sorted(lifecycle_counts.items())),
            "promotion_thresholds": {
                "candidate_successes": PROMOTION_THRESHOLDS["candidate_successes"],
                "validated_successes": PROMOTION_THRESHOLDS["validated_successes"],
                "validated_quality": PROMOTION_THRESHOLDS["validated_quality"],
                "trusted_successes": PROMOTION_THRESHOLDS["trusted_successes"],
                "trusted_quality": PROMOTION_THRESHOLDS["trusted_quality"],
                "demotion_failures": 3,
            },
            "strategies": statuses,
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
    from .routes_status import router as status_routes
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
    app.include_router(status_routes)
    app.include_router(skills_routes)
    return app


app = create_app()
