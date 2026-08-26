from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import hashlib
import json


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
        return cls(**payload)


@dataclass(slots=True)
class SkillMatch:
    spec: SkillSpec
    score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = self.spec.to_dict()
        return payload


@dataclass(slots=True)
class QueryFeatures:
    query: str
    normalized_query: str
    tokens: list[str]
    available_columns: list[str]
    semantic_roles: dict[str, str]
    numeric_columns: list[str] = field(default_factory=list)
    boolean_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    schema_signature: str | None = None
    intent_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperienceRecord:
    query_text: str
    normalized_query: str
    schema_signature: str | None
    route: str
    skill_id: str | None
    confidence: float
    success: bool
    score: float
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    plan_hash: str | None = None
    plan_summary: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    feedback_score: int | None = None
    skill_state_before: dict[str, Any] | None = None
    skill_state_after: dict[str, Any] | None = None
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

