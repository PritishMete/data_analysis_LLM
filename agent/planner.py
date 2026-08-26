from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from learning.models import LearningDecision, QueryFeatures
from learning.retriever import tokenize
from secure_excel.semantic_roles import detect_column_role


_NUMBER_RE = re.compile(r"(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)", re.I)
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
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _column_roles(df: pd.DataFrame) -> dict[str, str]:
    roles: dict[str, str] = {}
    for column in df.columns:
        try:
            roles[str(column)] = detect_column_role(str(column), df[column])["role"]
        except Exception:
            roles[str(column)] = "unknown"
    return roles


def _choose_column(columns: list[str], text: str, *, role_hint: str | None = None) -> str | None:
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
        # Try to detect explicit entity values from the user text by stripping known verbs.
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
            column = column or _choose_column(columns, phrase, role_hint=column_hint)
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
        column = column or _choose_column(columns, "rating", role_hint="rating_metric")
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


class LearningPlanner:
    def plan(self, user_text: str, df: pd.DataFrame | None, available_columns: list[str] | None = None) -> LearningDecision:
        columns = list(available_columns or (list(df.columns) if df is not None else []))
        roles = _column_roles(df) if df is not None else {}
        normalized = _normalize_text(user_text)
        features = QueryFeatures(
            query=user_text,
            normalized_query=normalized,
            tokens=tokenize(user_text),
            available_columns=columns,
            semantic_roles=roles,
            numeric_columns=[col for col, role in roles.items() if role in {"numeric_metric", "currency_metric", "rating_metric", "count", "percentage"}],
            boolean_columns=[col for col, role in roles.items() if role == "boolean_capability"],
            text_columns=[col for col, role in roles.items() if role in {"description", "category", "entity_name"}],
            schema_signature=None,
            intent_hints=[token for token in ("filter", "categorize", "classify", "normalize", "aggregate", "group", "show") if token in normalized],
        )

        sql_plan = _build_sql_filter_plan(user_text, columns, roles)
        if sql_plan is not None:
            confidence = 0.88 if len(sql_plan.get("filters") or []) > 1 else 0.82
            return LearningDecision(
                route="sql",
                confidence=confidence,
                message="Matched a learned filter skill and built a local SQL plan.",
                skill_id="filter.multi_condition.v1" if len(sql_plan.get("filters") or []) > 1 else "filter.entity_search.v1",
                skill_name="Multi-condition filtering" if len(sql_plan.get("filters") or []) > 1 else "Entity search",
                plan=sql_plan,
                validation_notes=[],
                features=features.to_dict(),
            )

        operation_plan = _build_operation_plan(user_text, columns)
        if operation_plan is not None:
            return LearningDecision(
                route="operation",
                confidence=0.8,
                message="Matched a learned operation skill and built a local command.",
                skill_id="clean.boolean_normalization.v1" if "normalize" in normalized else "clean.gender_normalization.v1",
                skill_name="Categorization / normalization",
                plan=operation_plan,
                validation_notes=[],
                features=features.to_dict(),
            )

        return LearningDecision(
            route="unknown",
            confidence=0.0,
            message="No learned skill matched with sufficient confidence.",
            features=features.to_dict(),
        )
