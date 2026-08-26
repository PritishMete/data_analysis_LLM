from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SafeField(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    semantic_role: str
    dtype: str


class DatasetProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    fields: list[SafeField] = Field(default_factory=list)


class QueryFeaturesPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    predicate_count: int = 0
    logical_structure: str = "SINGLE"
    semantic_roles: list[str] = Field(default_factory=list)
    operators: list[str] = Field(default_factory=list)
    predicate_graph: list[dict[str, Any]] = Field(default_factory=list)
    tool_hints: list[str] = Field(default_factory=list)


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str | None = None
    intent: str
    query_features: QueryFeaturesPayload
    dataset_profile: DatasetProfile


class LearningEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 2
    event_id: str | None = None
    intent: str
    query_features: QueryFeaturesPayload
    dataset_profile: DatasetProfile | None = None
    tool_graph: list[str] = Field(default_factory=list)
    plan: dict[str, Any]
    execution: dict[str, Any]
    validation: dict[str, Any]
    quality_score: float = 0.0
    route: str | None = None
    plan_source: str | None = None
    skill_id: str | None = None
    plan_template_id: str | None = None
    dataset_semantic_signature: str | None = None
    execution_success: bool | None = None
    critic_passed: bool | None = None
    result_validation_passed: bool | None = None
    plan_completeness_passed: bool | None = None
    privacy_validation_passed: bool | None = None
    no_unresolved_ambiguity: bool | None = None
    no_critical_repair: bool | None = None
    repair_count: int | None = None
    correction_state: str | None = None
    safe_query_abstraction: dict[str, Any] = Field(default_factory=dict)


class ExperienceRequest(LearningEvent):
    model_config = ConfigDict(extra="allow")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    decision_id: str | None = None
    feedback_score: int | None = None
    correction_type: str | None = None
    affected_intent: str | None = None
    generalized_lesson: str | None = None
    dataset_semantic_signature: str | None = None
    requested_role: str | None = None
    resolution_preference: str | None = None
    preferred_semantic_candidate: str | None = None


class PlanResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    plan_source: str
    confidence: float
    tool_graph: list[str]
    plan_template_id: str | None = None
    critic_status: dict[str, Any] = Field(default_factory=dict)


class ExperienceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    stored: bool
    learning_outcome: dict[str, Any]


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    accepted: bool
    outcome: dict[str, Any] = Field(default_factory=dict)


class SkillResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    skills: list[dict[str, Any]]


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    metrics: dict[str, Any]


class TrainingDatasetCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_candidate_strategies: bool = True
    limit: int = Field(1000, ge=1, le=10_000)


class TrainingCandidateInvalidationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_id: str | None = None
    family_fingerprint: str | None = None
    reason: str = "manual"
