from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import pandas as pd

from learning.models import DatasetSemanticProfile, LogicalGroup, PlannerContext, PredicateNode, QueryFeatures, stable_hash
from secure_excel.semantic_roles import anonymized_schema_summary, build_schema_profile, detect_column_role


_OPERATOR_PATTERNS: list[tuple[str, str]] = [
    ("greater_than_equal", r"\b(?:at least|greater than or equal to|greater than equal to|not less than)\b"),
    ("less_than_equal", r"\b(?:at most|less than or equal to|less than equal to|not more than)\b"),
    ("greater_than", r"\b(?:above|over|greater than|more than|higher than)\b"),
    ("less_than", r"\b(?:below|under|less than|lower than)\b"),
    ("equals", r"\b(?:equal to|equals?|is)\b"),
    ("contains", r"\b(?:contains?|including|with)\b"),
]

_INTENT_HINTS: list[tuple[str, str]] = [
    ("operation", r"\b(?:add column|insert column|create column|tag each row|mark each row|label each row|pivot table|colour formatting|color formatting)\b"),
    ("cleaning", r"\b(?:normalize|normalise|clean|fill missing|null|dedupe|duplicate|standardize|standardise)\b"),
    ("analytics", r"\b(?:group by|aggregate|average|avg|sum|count|total|rank|top|bottom|compare)\b"),
    ("filter", r"\b(?:show|find|filter|list|display|rows having|rows with|where)\b"),
    ("sentiment", r"\b(?:sentiment|review tone|customer satisfaction)\b"),
]

_BOOLEAN_HINTS = {"online delivery", "table booking", "yes", "no", "true", "false", "available", "unavailable", "open", "closed"}
_NUMBER_RE = re.compile(r"(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?|between)\s+(?P<value>-?\d+(?:\.\d+)?)", re.I)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _count_matches(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))


def _classify_intent(text: str) -> tuple[str, list[str]]:
    normalized = _normalize(text)
    hints: list[str] = []
    for intent, pattern in _INTENT_HINTS:
        if re.search(pattern, normalized, flags=re.I):
            hints.append(intent)
    if "operation" in hints:
        return "operation", hints
    if "sentiment" in hints:
        return "sentiment", hints
    if "cleaning" in hints:
        return "cleaning", hints
    if "analytics" in hints and "filter" not in hints:
        return "analytics", hints
    if "filter" in hints:
        return "filter", hints
    return "unknown", hints


def _logical_structure(text: str) -> str:
    normalized = _normalize(text)
    has_and = " and " in normalized
    has_or = " or " in normalized
    has_then = " then " in normalized or ", then " in normalized or "; then " in normalized
    if has_then:
        return "SEQUENTIAL"
    if has_and and has_or:
        return "MIXED"
    if has_and:
        return "AND"
    if has_or:
        return "OR"
    return "SINGLE"


def _count_boolean_predicates(text: str) -> int:
    normalized = _normalize(text)
    return sum(1 for hint in _BOOLEAN_HINTS if hint in normalized)


def _count_numeric_comparisons(text: str) -> int:
    normalized = _normalize(text)
    patterns = [
        r"\b(?:above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+-?\d+(?:\.\d+)?",
        r"-?\d+(?:\.\d+)?\s*(?:above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\b",
        r"\b(?:between)\s+-?\d+(?:\.\d+)?\s+(?:and)\s+-?\d+(?:\.\d+)?",
    ]
    return sum(_count_matches(pattern, normalized) for pattern in patterns)


def _count_entities(text: str) -> int:
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text or "")
    capitalized = [token for token in re.findall(r"\b[A-Z][A-Za-z0-9&'.-]+\b", text or "") if len(token) > 1]
    return len(quoted) + min(len(capitalized), 3)


def _extract_predicate_graph(text: str) -> list[dict[str, Any]]:
    normalized = (text or "").lower()
    predicates: list[PredicateNode] = []
    seen_spans: list[tuple[int, int]] = []

    for hint in sorted(_BOOLEAN_HINTS, key=len, reverse=True):
        for match in re.finditer(re.escape(hint), normalized):
            span = match.span()
            if any(not (span[1] <= existing[0] or span[0] >= existing[1]) for existing in seen_spans):
                continue
            seen_spans.append(span)
            window = normalized[max(0, span[0] - 12):span[0]]
            predicates.append(
                PredicateNode(
                    kind="predicate",
                    role="boolean_capability",
                    operator="equals",
                    value_kind="boolean_true" if "not" not in window and "without" not in window else "boolean_false",
                    negated=("not" in window or "without" in window or "exclude" in window),
                )
            )

    for match in _NUMBER_RE.finditer(normalized):
        span = match.span()
        if any(not (span[1] <= existing[0] or span[0] >= existing[1]) for existing in seen_spans):
            continue
        seen_spans.append(span)
        op = match.group("op").lower()
        operator_map = {
            "above": "greater_than",
            "over": "greater_than",
            "greater than": "greater_than",
            "more than": "greater_than",
            "at least": "greater_than_equal",
            "below": "less_than",
            "under": "less_than",
            "less than": "less_than",
            "equal to": "equals",
            "equals": "equals",
            "equal": "equals",
            "between": "between",
        }
        predicates.append(
            PredicateNode(
                kind="predicate",
                role="numeric_measure",
                operator=operator_map.get(op, "greater_than"),
                value_kind="numeric_comparison",
                comparison_direction="from_query",
            )
        )

    if not predicates:
        return []
    logical_operator = "OR" if " or " in normalized and " and " not in normalized else "AND"
    group = LogicalGroup(operator=logical_operator, children=predicates)
    return [group.to_dict()]


def _derive_role_list(profile: dict[str, Any] | None) -> list[str]:
    if not profile:
        return []
    roles = [str(column.get("role") or "unknown") for column in profile.get("columns", [])]
    return sorted({role for role in roles if role and role != "unknown"})


def _derive_dataset_signature(profile: dict[str, Any] | None) -> str | None:
    if not profile:
        return None
    return stable_hash(
        {
            "columns": profile.get("columns", []),
            "role_index": {key: len(value) for key, value in (profile.get("role_index") or {}).items()},
        }
    )


def _safe_profile(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {}
    profile = build_schema_profile(df)
    return {
        "columns": anonymized_schema_summary(profile),
        "role_index": profile.get("role_index", {}),
        "dataset_semantic_signature": _derive_dataset_signature(profile),
        "row_count": int(len(df)),
    }


def build_planner_context(
    user_text: str,
    df: pd.DataFrame | None = None,
    available_columns: list[str] | None = None,
) -> PlannerContext:
    columns = [str(column) for column in (available_columns or (list(df.columns) if df is not None else []))]
    safe_profile = _safe_profile(df)
    raw_roles: dict[str, str] = {}
    if df is not None:
        for column in df.columns:
            try:
                raw_roles[str(column)] = detect_column_role(str(column), df[column])["role"]
            except Exception:
                raw_roles[str(column)] = "unknown"
    else:
        raw_roles = {column: "unknown" for column in columns}

    intent, intent_hints = _classify_intent(user_text)
    logical_structure = _logical_structure(user_text)
    operators = [name for name, pattern in _OPERATOR_PATTERNS if re.search(pattern, user_text or "", flags=re.I)]
    if re.search(r"\bnot\b", user_text or "", flags=re.I) and "not_equals" not in operators:
        operators.append("not_equals")

    predicate_count = len(operators)
    predicate_count += 1 if logical_structure in {"AND", "OR", "MIXED"} else 0
    predicate_count += _count_boolean_predicates(user_text)
    numeric_comparison_count = _count_numeric_comparisons(user_text)
    entity_reference_count = _count_entities(user_text)
    boolean_predicate_count = _count_boolean_predicates(user_text)
    step_count = 1 + (1 if logical_structure == "SEQUENTIAL" else 0)
    has_multiple_steps = step_count > 1

    semantic_roles = _derive_role_list(safe_profile)
    role_candidates = {role: list(values) for role, values in (safe_profile.get("role_index") or {}).items()}
    query_shape = "multi_step" if has_multiple_steps else ("question" if "?" in (user_text or "") else "statement")
    operation_hints = [hint for hint in ("filter", "group_by", "aggregate", "cleaning", "sentiment", "operation") if hint in intent_hints or hint == intent]
    tool_hints: list[str] = []
    if intent in {"filter", "analytics"}:
        tool_hints.extend(["sql", "sql.filter"])
    if intent in {"cleaning", "operation"}:
        tool_hints.extend(["operation", "categorization_agent._deterministic_special_mapping"])
    if intent == "sentiment":
        tool_hints.extend(["sentiment", "sentiment.analyzer"])
    if "boolean_capability" in semantic_roles:
        tool_hints.append("boolean")
    if "rating_metric" in semantic_roles:
        tool_hints.append("rating")
    confidence = 0.55
    confidence += 0.08 if intent != "unknown" else -0.08
    confidence += min(0.15, 0.03 * len(operators))
    confidence += 0.05 if semantic_roles else 0.0
    confidence += 0.04 if entity_reference_count else 0.0
    confidence = max(0.05, min(0.99, confidence))

    features = QueryFeatures(
        intent=intent,
        predicate_count=predicate_count,
        boolean_predicate_count=boolean_predicate_count,
        numeric_comparison_count=numeric_comparison_count,
        entity_reference_count=entity_reference_count,
        logical_structure=logical_structure,
        semantic_roles=semantic_roles,
        operators=sorted({operator for operator in operators}),
        operation_hints=sorted({hint for hint in operation_hints}),
        tool_hints=sorted({hint for hint in tool_hints}),
        query_shape=query_shape,
        dataset_semantic_signature=safe_profile.get("dataset_semantic_signature"),
        semantic_signature=stable_hash(
            {
                "intent": intent,
                "predicate_count": predicate_count,
                "boolean_predicate_count": boolean_predicate_count,
                "numeric_comparison_count": numeric_comparison_count,
                "entity_reference_count": entity_reference_count,
                "logical_structure": logical_structure,
                "semantic_roles": semantic_roles,
                "operators": sorted({operator for operator in operators}),
                "operation_hints": sorted({hint for hint in operation_hints}),
                "tool_hints": sorted({hint for hint in tool_hints}),
                "query_shape": query_shape,
                "dataset_semantic_signature": safe_profile.get("dataset_semantic_signature"),
            }
        ),
        confidence=round(confidence, 4),
        predicate_graph=_extract_predicate_graph(user_text),
        role_candidates=role_candidates,
        step_count=step_count,
        has_multiple_steps=has_multiple_steps,
    )

    dataset_semantic_profile = DatasetSemanticProfile(
        available_columns=columns,
        safe_profile=safe_profile,
        column_roles=raw_roles,
        dataset_semantic_signature=safe_profile.get("dataset_semantic_signature"),
    )
    dataset_profile = {
        "available_columns": columns,
        "column_roles": raw_roles,
        "safe_profile": safe_profile,
        "dataset_semantic_signature": safe_profile.get("dataset_semantic_signature"),
        "dataset_semantic_profile": dataset_semantic_profile.to_dict(),
    }

    return PlannerContext(
        features=features,
        dataset_profile=dataset_profile,
        dataset_semantic_profile=dataset_semantic_profile,
        retrieval_trace={
            "intent_hints": intent_hints,
            "operators": features.operators,
            "logical_structure": logical_structure,
        },
    )
