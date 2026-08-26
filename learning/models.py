from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib
import json


SCHEMA_VERSION = 2


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalise_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


@dataclass(slots=True)
class SkillSpec:
    id: str
    name: str
    description: str
    intents: list[str]
    examples: list[str]
    required_semantic_roles: list[str]
    required_input_types: list[str]
    preconditions: list[str]
    supported_parameters: list[str]
    tool: str
    validation_rules: list[str]
    failure_conditions: list[str]
    expected_result: str
    post_execution_checks: list[str]
    confidence: float
    source_implementation: str
    version: int = 1
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SkillState:
    skill_id: str
    confidence: float
    success_count: int = 0
    failure_count: int = 0
    average_quality_score: float = 0.0
    state: str = "bootstrap"
    promoted_at: str | None = None
    last_seen_at: str | None = None
    candidate_promotions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillState":
        data = dict(payload)
        data.setdefault("confidence", 0.5)
        data.setdefault("success_count", 0)
        data.setdefault("failure_count", 0)
        data.setdefault("average_quality_score", 0.0)
        data.setdefault("state", "bootstrap")
        data.setdefault("promoted_at", None)
        data.setdefault("last_seen_at", None)
        data.setdefault("candidate_promotions", 0)
        return cls(**data)


@dataclass(slots=True)
class SkillMatch:
    spec: SkillSpec
    score: float
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = self.spec.to_dict()
        return payload


@dataclass(slots=True)
class QueryFeatures:
    intent: str
    predicate_count: int
    boolean_predicate_count: int
    numeric_comparison_count: int
    entity_reference_count: int
    logical_structure: str
    semantic_roles: list[str]
    operators: list[str]
    operation_hints: list[str]
    tool_hints: list[str]
    query_shape: str
    dataset_semantic_signature: str | None
    semantic_signature: str
    confidence: float
    step_count: int = 1
    has_multiple_steps: bool = False
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanStep:
    step_id: str
    tool: str
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentPlan:
    plan_id: str
    intent: str
    steps: list[PlanStep]
    expected_result: dict[str, Any]
    confidence: float
    requested_predicate_count: int = 0
    planned_predicate_count: int = 0
    logical_structure: str = "SINGLE"
    semantic_roles: list[str] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload


@dataclass(slots=True)
class LearningDecision:
    route: str
    confidence: float
    message: str
    skill_id: str | None = None
    skill_name: str | None = None
    plan: dict[str, Any] | None = None
    validation_notes: list[str] = field(default_factory=list)
    fallback_used: str | None = None
    features: dict[str, Any] = field(default_factory=dict)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    tool_sequence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperienceRecord:
    intent: str
    query_features: dict[str, Any]
    semantic_roles: list[str]
    operators: list[str]
    logical_structure: str
    tool_sequence: list[str]
    result_summary: dict[str, Any]
    dataset_semantic_signature: str | None
    semantic_signature: str
    route: str
    skill_id: str | None
    confidence: float
    success: bool
    score: float
    plan_hash: str | None = None
    plan_summary: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    feedback_score: int | None = None
    skill_state_before: dict[str, Any] | None = None
    skill_state_after: dict[str, Any] | None = None
    correction_type: str | None = None
    correction_summary: dict[str, Any] | None = None
    candidate_strategy_id: str | None = None
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceRecord":
        data = dict(payload)
        data.setdefault("query_features", {})
        data.setdefault("semantic_roles", [])
        data.setdefault("operators", [])
        data.setdefault("logical_structure", "SINGLE")
        data.setdefault("tool_sequence", [])
        data.setdefault("result_summary", {})
        data.setdefault("dataset_semantic_signature", None)
        data.setdefault(
            "semantic_signature",
            stable_hash({"intent": data.get("intent"), "route": data.get("route"), "skill_id": data.get("skill_id")}),
        )
        data.setdefault("route", "unknown")
        data.setdefault("skill_id", None)
        data.setdefault("confidence", 0.0)
        data.setdefault("success", False)
        data.setdefault("score", 0.0)
        data.setdefault("plan_hash", None)
        data.setdefault("plan_summary", {})
        data.setdefault("failure_reason", None)
        data.setdefault("feedback_score", None)
        data.setdefault("skill_state_before", None)
        data.setdefault("skill_state_after", None)
        data.setdefault("correction_type", None)
        data.setdefault("correction_summary", None)
        data.setdefault("candidate_strategy_id", None)
        data.setdefault("created_at", utcnow().isoformat())
        data.setdefault("version", SCHEMA_VERSION)
        return cls(**data)


@dataclass(slots=True)
class FailureLesson:
    lesson_id: str
    intent: str
    failure_signature: str
    condition_structure: str
    lesson: str
    severity: str
    semantic_roles: list[str]
    operators: list[str]
    tool_sequence: list[str]
    occurrence_count: int = 1
    average_quality: float = 0.0
    last_seen_at: str = field(default_factory=lambda: utcnow().isoformat())
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailureLesson":
        return cls(**payload)


@dataclass(slots=True)
class CorrectionRecord:
    correction_id: str
    correction_type: str
    affected_intent: str
    generalized_lesson: str
    dataset_semantic_signature: str | None
    requested_role: str | None = None
    resolution_preference: str | None = None
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorrectionRecord":
        return cls(**payload)


@dataclass(slots=True)
class CandidateStrategy:
    strategy_id: str
    intent: str
    semantic_signature: str
    tool_sequence: list[str]
    semantic_roles: list[str]
    evidence_count: int
    average_quality: float
    state: str = "candidate"
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    validated_at: str | None = None
    promoted_at: str | None = None
    last_seen_at: str | None = None
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateStrategy":
        return cls(**payload)


@dataclass(slots=True)
class PlannerContext:
    features: QueryFeatures
    dataset_profile: dict[str, Any]
    retrieved_skills: list[dict[str, Any]] = field(default_factory=list)
    similar_experiences: list[dict[str, Any]] = field(default_factory=list)
    failure_lessons: list[dict[str, Any]] = field(default_factory=list)
    candidate_strategies: list[dict[str, Any]] = field(default_factory=list)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
