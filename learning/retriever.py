from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Iterable

from learning.models import CandidateStrategy, ExperienceRecord, FailureLesson, QueryFeatures, SkillSpec


_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _overlap(left: Iterable[str], right: Iterable[str]) -> tuple[float, list[str]]:
    left_set = {str(item) for item in left if item}
    right_set = {str(item) for item in right if item}
    matched = sorted(left_set & right_set)
    if not matched:
        return 0.0, []
    return min(1.0, 0.12 * len(matched)), matched


def _age_boost(created_at: str | None) -> float:
    if not created_at:
        return 0.0
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    now = datetime.now(timezone.utc)
    days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    return max(0.0, 0.08 - min(0.08, days / 365.0 * 0.08))


def score_skill(spec: SkillSpec, features: QueryFeatures) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    if spec.id.startswith("filter.") and features.intent in {"filter", "analytics"}:
        score += 0.18
        reasons.append("filter intent")
    if spec.id.startswith("clean.") and features.intent in {"cleaning", "operation"}:
        score += 0.18
        reasons.append("cleaning intent")
    if spec.id.startswith("analytics.") and features.intent in {"analytics", "filter"}:
        score += 0.14
        reasons.append("analytics intent")
    if spec.id.startswith("transform.") and features.intent in {"cleaning", "operation"}:
        score += 0.14
        reasons.append("transformation intent")
    if spec.id.startswith("privacy.") and features.intent == "operation":
        score += 0.08
        reasons.append("privacy guard")

    role_score, matched_roles = _overlap(spec.required_semantic_roles, features.semantic_roles)
    if matched_roles:
        score += min(0.28, role_score)
        reasons.append(f"semantic roles: {matched_roles}")

    if spec.tool and any(tool_hint in spec.tool.lower() for tool_hint in features.tool_hints):
        score += 0.12
        reasons.append("tool hint")

    if any(hint in features.operation_hints for hint in tokenize(" ".join(spec.intents + spec.examples + [spec.name, spec.description]))):
        score += 0.12
        reasons.append("operation hint overlap")

    if features.logical_structure in {"AND", "MIXED"} and "multi" in spec.id:
        score += 0.08
        reasons.append("multi-condition structure")
    if features.logical_structure == "SEQUENTIAL" and ("step" in spec.description.lower() or "chain" in spec.description.lower()):
        score += 0.08
        reasons.append("sequential structure")

    if features.predicate_count > 1 and spec.id.startswith("filter."):
        score += 0.06
        reasons.append("multiple predicates")
    if features.boolean_predicate_count and "boolean_capability" in spec.required_semantic_roles:
        score += 0.08
        reasons.append("boolean capability")
    if features.numeric_comparison_count and any(role in spec.required_semantic_roles for role in {"rating_metric", "numeric_metric", "currency_metric"}):
        score += 0.06
        reasons.append("numeric comparison")

    score += max(0.0, min(0.1, spec.confidence - 0.8))
    if score <= 0.03:
        return 0.0, []
    return round(min(score, 0.99), 4), reasons


def score_experience(record: ExperienceRecord, features: QueryFeatures) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    if record.intent == features.intent:
        score += 0.25
        reasons.append("intent match")
    if record.semantic_signature == features.semantic_signature:
        score += 0.35
        reasons.append("semantic signature match")
    if record.dataset_semantic_signature and record.dataset_semantic_signature == features.dataset_semantic_signature:
        score += 0.12
        reasons.append("dataset signature match")

    role_score, matched_roles = _overlap(record.semantic_roles, features.semantic_roles)
    if matched_roles:
        score += min(0.18, role_score)
        reasons.append(f"roles: {matched_roles}")
    if record.logical_structure == features.logical_structure:
        score += 0.08
        reasons.append("structure match")

    tool_score, matched_tools = _overlap(record.tool_sequence, features.tool_hints)
    if matched_tools:
        score += min(0.12, tool_score)
        reasons.append(f"tool sequence: {matched_tools}")

    if record.success:
        score += 0.08
    score += max(0.0, min(0.08, record.score * 0.08))
    score += _age_boost(record.created_at)
    if record.failure_reason:
        score -= 0.05

    if score <= 0.04:
        return 0.0, []
    return round(min(score, 0.99), 4), reasons


def score_failure_lesson(lesson: FailureLesson, features: QueryFeatures) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    if lesson.intent == features.intent:
        score += 0.25
        reasons.append("intent match")
    if lesson.condition_structure == features.logical_structure:
        score += 0.15
        reasons.append("structure match")
    role_score, matched_roles = _overlap(lesson.semantic_roles, features.semantic_roles)
    if matched_roles:
        score += min(0.18, role_score)
        reasons.append(f"roles: {matched_roles}")
    tool_score, matched_tools = _overlap(lesson.tool_sequence, features.tool_hints)
    if matched_tools:
        score += min(0.12, tool_score)
        reasons.append(f"tools: {matched_tools}")
    if lesson.failure_signature == features.semantic_signature:
        score += 0.2
        reasons.append("failure signature match")
    score += min(0.08, lesson.average_quality * 0.05)
    if score <= 0.04:
        return 0.0, []
    return round(min(score, 0.99), 4), reasons


def score_candidate_strategy(strategy: CandidateStrategy, features: QueryFeatures) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    if strategy.intent == features.intent:
        score += 0.25
        reasons.append("intent match")
    if strategy.semantic_signature == features.semantic_signature:
        score += 0.35
        reasons.append("semantic signature match")
    role_score, matched_roles = _overlap(strategy.semantic_roles, features.semantic_roles)
    if matched_roles:
        score += min(0.18, role_score)
        reasons.append(f"roles: {matched_roles}")
    tool_score, matched_tools = _overlap(strategy.tool_sequence, features.tool_hints)
    if matched_tools:
        score += min(0.14, tool_score)
        reasons.append(f"tools: {matched_tools}")
    score += min(0.1, strategy.average_quality * 0.1)
    score += min(0.05, strategy.evidence_count * 0.01)
    if strategy.state == "promoted":
        score += 0.05
    if score <= 0.04:
        return 0.0, []
    return round(min(score, 0.99), 4), reasons


def rank_records(records: list[Any], scorer, features: QueryFeatures, limit: int = 5) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for record in records:
        score, reasons = scorer(record, features)
        if score <= 0:
            continue
        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        payload["score"] = score
        payload["reasons"] = reasons
        ranked.append(payload)
    ranked.sort(key=lambda item: (item.get("score", 0.0), item.get("created_at", "")), reverse=True)
    return ranked[:limit]
