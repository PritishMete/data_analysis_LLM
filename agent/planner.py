from __future__ import annotations

from typing import Any
import json
import re

import pandas as pd

from learning.feature_extractor import build_planner_context
from learning.models import LearningDecision, PlannerContext
from learning.skill_registry import SkillRegistry, get_skill_registry

from secure_excel.semantic_roles import detect_column_role


_GENERIC_ENTITY_STOPWORDS = {
    "restaurant",
    "restaurants",
    "row",
    "rows",
    "record",
    "records",
    "item",
    "items",
    "data",
    "entry",
    "entries",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _column_roles(df: pd.DataFrame | None, columns: list[str]) -> dict[str, str]:
    if df is None:
        return {str(column): "unknown" for column in columns}
    roles: dict[str, str] = {}
    for column in columns:
        try:
            roles[str(column)] = detect_column_role(str(column), df[column])["role"]
        except Exception:
            roles[str(column)] = "unknown"
    return roles


def _choose_column(columns: list[str], text: str, *, role_hint: str | None = None, roles: dict[str, str] | None = None) -> str | None:
    normalized = _normalize_text(text)
    compact = _compact(text)
    for column in columns:
        n = _normalize_text(column)
        c = _compact(column)
        if n and (n in normalized or c and c in compact):
            return column
    if role_hint:
        role_aliases = {
            "entity_name": {"name", "restaurant", "customer", "product", "entity", "title"},
            "rating_metric": {"rating", "score", "stars"},
            "boolean_capability": {"delivery", "book", "booking", "available", "open"},
            "geographic_area": {"city", "country", "region", "state", "location"},
            "numeric_metric": {"amount", "value", "count", "total", "price", "revenue", "sales"},
        }
        aliases = role_aliases.get(role_hint, set())
        for column in columns:
            column_norm = _normalize_text(column)
            if any(alias in column_norm for alias in aliases):
                return column
    if roles:
        for column, role in roles.items():
            if role == role_hint:
                return column
    return columns[0] if columns else None


def _build_sql_filter_plan(text: str, columns: list[str], roles: dict[str, str]) -> dict[str, Any] | None:
    normalized = _normalize_text(text)
    if not any(token in normalized for token in {"show", "find", "filter", "list", "display", "view", "return"}):
        return None

    filters: list[dict[str, Any]] = []
    seen_filters: set[tuple[str, str, str]] = set()

    def add_filter(column: str, operator: str, value: Any) -> None:
        key = (column, operator, json.dumps(value, sort_keys=True, default=str))
        if key in seen_filters:
            return
        seen_filters.add(key)
        filters.append({"column": column, "operator": operator, "value": value})

    entity_column = None
    for column, role in roles.items():
        if role in {"restaurant_entity", "customer_entity", "product_entity", "entity_name", "category"}:
            entity_column = column
            break
    if entity_column:
        cleaned = normalized
        for verb in ("show", "find", "filter", "display", "list", "view", "return", "rows", "records"):
            cleaned = re.sub(rf"\b{verb}\b", " ", cleaned)
        cleaned = re.split(r"\b(having|with|where|that|which|whose|and|or)\b", cleaned, maxsplit=1)[0]
        cleaned = re.sub(r"\b(over|above|below|under|more|less|than|at least)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            candidate = cleaned.split(" and ")[0].strip()
            if candidate and candidate not in _GENERIC_ENTITY_STOPWORDS:
                add_filter(entity_column, "contains", candidate.title())

    for phrase, column_hint, value in (
        ("online delivery", "boolean_capability", True),
        ("table booking", "boolean_capability", True),
        ("delivery", "boolean_capability", True),
        ("booking", "boolean_capability", True),
    ):
        if phrase in normalized:
            column = next((col for col, role in roles.items() if role == "boolean_capability" or phrase.replace(" ", "") in _compact(col)), None)
            column = column or _choose_column(columns, phrase, role_hint=column_hint, roles=roles)
            if column:
                add_filter(column, "equals", value)

    rating_source = (text or "").lower()
    rating_match = re.search(
        r"\brating\b.*?(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)",
        rating_source,
    )
    if rating_match is None:
        rating_match = re.search(
            r"(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)\s*\b(?:rating|score|stars?)\b",
            rating_source,
        )
    if rating_match:
        op = rating_match.group("op")
        value = rating_match.group("value")
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
        }
        column = next((col for col, role in roles.items() if role == "rating_metric" or "rating" in _compact(col)), None)
        column = column or _choose_column(columns, "rating", role_hint="rating_metric", roles=roles)
        if column:
            add_filter(column, operator_map.get(op, "greater_than"), value)

    explicit_conditions = [token for token in (" and ", " with ", " having ", " where ") if token in normalized]
    if len(filters) >= 2 or explicit_conditions:
        return {"group_by": [], "metrics": [], "filters": filters, "limit": None, "order_by": []}

    return {"group_by": [], "metrics": [], "filters": filters, "limit": None, "order_by": []} if filters else None


def _build_operation_plan(text: str, columns: list[str]) -> dict[str, Any] | None:
    normalized = _normalize_text(text)
    if not any(token in normalized for token in {"categorize", "classify", "normalize", "normalise", "bucket", "band", "bin"}):
        return None
    selected = columns[:]
    if not selected:
        return None
    if "all columns" not in normalized and "every column" not in normalized:
        picked = []
        for column in columns:
            c = _compact(column)
            if c and c in _compact(text):
                picked.append(column)
        if picked:
            selected = picked
        else:
            selected = columns[:1]
    return {
        "action": "categorize",
        "categorize": {
            "sourceColumn": selected[0],
            "sourceColumns": selected,
            "allColumns": len(selected) == len(columns),
            "newColumnName": selected[0],
            "categories": [],
            "unmatchedLabel": "Other",
        },
    }


def _score_retrieval_context(context: PlannerContext, registry: SkillRegistry) -> dict[str, Any]:
    features = context.features
    skills = registry.match(features)[:5]
    experiences = sorted(
        context.similar_experiences,
        key=lambda item: (item.get("score", 0.0), item.get("created_at", "")),
        reverse=True,
    )[:5]
    lessons = sorted(
        context.failure_lessons,
        key=lambda item: (item.get("score", 0.0), item.get("created_at", "")),
        reverse=True,
    )[:5]
    candidates = sorted(
        context.candidate_strategies,
        key=lambda item: (item.get("score", 0.0), item.get("created_at", "")),
        reverse=True,
    )[:5]
    return {
        "skills": [match.to_dict() for match in skills],
        "experiences": experiences,
        "failure_lessons": lessons,
        "candidate_strategies": candidates,
        "feature_signature": features.semantic_signature,
    }


class LearningPlanner:
    def __init__(self, registry: SkillRegistry | None = None):
        self.registry = registry or get_skill_registry()

    def plan(
        self,
        user_text: str,
        df: pd.DataFrame | None,
        available_columns: list[str] | None = None,
        planner_context: PlannerContext | None = None,
    ) -> LearningDecision:
        columns = list(available_columns or (list(df.columns) if df is not None else []))
        context = planner_context or build_planner_context(user_text, df, columns)
        features = context.features
        roles = _column_roles(df, columns)
        retrieval_trace = _score_retrieval_context(context, self.registry)
        retrieval_trace.update(context.retrieval_trace)

        if retrieval_trace["skills"]:
            retrieval_trace["top_skill_id"] = retrieval_trace["skills"][0]["spec"]["id"]
            retrieval_trace["top_skill_score"] = retrieval_trace["skills"][0]["score"]

        route = "unknown"
        plan: dict[str, Any] | None = None
        skill_id: str | None = None
        skill_name: str | None = None
        message = "No learned skill matched with sufficient confidence."
        confidence = 0.0

        top_skill = retrieval_trace["skills"][0] if retrieval_trace["skills"] else None
        if top_skill and top_skill["score"] >= 0.35:
            skill_id = top_skill["spec"]["id"]
            skill_name = top_skill["spec"]["name"]

        if features.intent in {"filter", "analytics"} or (skill_id and skill_id.startswith("filter.")):
            plan = _build_sql_filter_plan(user_text, columns, roles)
            if plan is not None:
                route = "sql"
                confidence = 0.9 if len(plan.get("filters") or []) > 1 else 0.84
                if len(plan.get("filters") or []) > 1:
                    skill_id = "filter.multi_condition.v1"
                    skill_name = "Multi-condition filtering"
                else:
                    skill_id = "filter.entity_search.v1"
                    skill_name = "Entity search"
                message = "Matched a learned filter skill and built a local plan."

        if route == "unknown" and (features.intent in {"cleaning", "operation"} or (skill_id and skill_id.startswith("clean."))):
            plan = _build_operation_plan(user_text, columns)
            if plan is not None:
                route = "operation"
                confidence = 0.82
                skill_id = skill_id or "clean.boolean_normalization.v1"
                skill_name = skill_name or "Categorization / normalization"
                message = "Matched a learned operation skill and built a local command."

        if route == "unknown" and features.intent == "sentiment":
            route = "sentiment"
            confidence = 0.78
            message = "Matched a learned sentiment route."

        if route == "unknown" and features.intent == "analytics":
            route = "sql"
            confidence = 0.62
            message = "Matched an analytical query shape but no confident local plan."

        validation_notes: list[str] = []
        if route == "sql" and plan is not None:
            requested = features.predicate_count
            planned = len(plan.get("filters") or [])
            if requested and planned < requested:
                validation_notes.append("planned fewer predicates than requested")
            if features.logical_structure in {"AND", "MIXED"} and planned < 2:
                validation_notes.append("did not preserve multi-condition structure")
        if skill_id:
            for record in retrieval_trace["experiences"]:
                if record.get("skill_id") == skill_id:
                    validation_notes.append("retrieved prior experience for this skill")
                    break

        return LearningDecision(
            route=route,
            confidence=round(confidence, 4),
            message=message,
            skill_id=skill_id,
            skill_name=skill_name,
            plan=plan,
            validation_notes=validation_notes,
            features=features.to_dict(),
            retrieval_trace=retrieval_trace,
            tool_sequence=(
                ["sql.filter"] if route == "sql" else
                ["categorization_agent._deterministic_special_mapping"] if route == "operation" else
                ["sentiment.analyzer"] if route == "sentiment" else
                []
            ),
        )
