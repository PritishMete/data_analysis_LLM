from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from learning.models import QueryFeatures, SkillSpec


_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def score_skill(spec: SkillSpec, features: QueryFeatures) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    query_tokens = Counter(features.tokens)
    intent_tokens = set(tokenize(" ".join(spec.intents + spec.examples + [spec.name, spec.description])))
    overlap = sum(query_tokens[token] for token in intent_tokens if token in query_tokens)
    if overlap:
        score += min(0.45, overlap * 0.08)
        reasons.append(f"intent overlap: {overlap}")

    normalized_query = features.normalized_query
    if any(intent.replace(" ", "") in normalized_query for intent in spec.intents):
        score += 0.15
        reasons.append("intent phrase match")

    available_roles = set(features.semantic_roles.values())
    required_roles = set(spec.required_semantic_roles)
    role_overlap = available_roles & required_roles
    if role_overlap:
        score += min(0.25, 0.1 * len(role_overlap))
        reasons.append(f"semantic roles: {sorted(role_overlap)}")

    if spec.id.startswith("filter.") and any(token in features.tokens for token in {"show", "find", "filter", "having", "where"}):
        score += 0.1
        reasons.append("filter intent")
    if spec.id.startswith("clean.") and any(token in features.tokens for token in {"normalize", "normalise", "convert", "fill", "missing", "clean"}):
        score += 0.1
        reasons.append("cleaning intent")
    if spec.id.startswith("analytics.") and any(token in features.tokens for token in {"total", "average", "sum", "count", "top", "sort", "rank"}):
        score += 0.1
        reasons.append("analytics intent")

    if "rating_metric" in required_roles and any(token in normalized_query for token in {"rating", "stars", "score"}):
        score += 0.1
        reasons.append("rating phrase")
    if "boolean_capability" in required_roles and any(token in normalized_query for token in {"online delivery", "table booking", "yes", "no", "true", "false"}):
        score += 0.08
        reasons.append("boolean phrase")
    if "geographic_area" in required_roles and any(token in normalized_query for token in {"country", "city", "region", "state"}):
        score += 0.06
        reasons.append("geo phrase")

    score += max(0.0, min(0.1, spec.confidence - 0.8))
    if score <= 0.05:
        return 0.0, []
    return round(min(score, 0.99), 4), reasons
