"""Local natural-language parsing for Excel assistant commands."""

from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any

from .query_types import QueryCondition, StructuredQuery


ROLE_SYNONYMS = {
    "geographic_area": {"city", "area", "location", "in", "at"},
    "delivery_capability": {"delivery", "online delivery", "home delivery", "deliver"},
    "table_booking_capability": {"table booking", "booking", "reservation", "reservations", "table reservation"},
    "rating_metric": {"rating", "score", "stars"},
    "restaurant_entity": {"restaurant", "restaurants"},
    "count": {"count", "how many"},
}


def _match_role(text: str, role: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ROLE_SYNONYMS.get(role, set()))


def _extract_between(text: str) -> tuple[float, float] | None:
    match = re.search(r"\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", text, re.I)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None


def _extract_number(text: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _resolve_best_column(schema: dict[str, Any], role: str) -> str | None:
    candidates = schema.get("role_index", {}).get(role, [])
    return candidates[0] if candidates else None


def _parse_conditions(text: str, schema: dict[str, Any]) -> list[QueryCondition]:
    conditions: list[QueryCondition] = []

    area_col = _resolve_best_column(schema, "geographic_area")
    if area_col:
        area_match = re.search(
            r"\b(?:in|at|from)\s+([a-zA-Z][a-zA-Z0-9\s&'\-\.]{1,60}?)(?:\s+(?:having|with|and|where|show|find|list|sort|group|count|above|below|over|under)|[,.!?]|$)",
            text,
            re.I,
        )
        if area_match:
            value = area_match.group(1).strip()
            if value:
                conditions.append(QueryCondition(column_id=area_col, operator="equals", value=value))

    rating_col = _resolve_best_column(schema, "rating_metric")
    if rating_col and _match_role(text, "rating_metric"):
        between = _extract_between(text)
        if between:
            conditions.append(QueryCondition(column_id=rating_col, operator="between", value=between[0], value2=between[1]))
        else:
            number = _extract_number(text)
            if number is not None:
                if re.search(r"\b(at least|minimum|>=|greater than or equal)\b", text, re.I):
                    op = "greater_equal"
                elif re.search(r"\b(at most|maximum|<=|less than or equal)\b", text, re.I):
                    op = "less_equal"
                elif re.search(r"\b(above|over|greater than|more than|>)\b", text, re.I):
                    op = "greater_than"
                elif re.search(r"\b(below|under|less than|<)\b", text, re.I):
                    op = "less_than"
                else:
                    op = "greater_than"
                conditions.append(QueryCondition(column_id=rating_col, operator=op, value=number))

    delivery_col = _resolve_best_column(schema, "delivery_capability")
    if delivery_col and _match_role(text, "delivery_capability"):
        conditions.append(QueryCondition(column_id=delivery_col, operator="equals", value=True))

    booking_col = _resolve_best_column(schema, "table_booking_capability")
    if booking_col and _match_role(text, "table_booking_capability"):
        conditions.append(QueryCondition(column_id=booking_col, operator="equals", value=True))

    return conditions


def parse_query(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    lowered = (text or "").strip()
    query = StructuredQuery(operation="report")

    if re.search(r"\b(count|how many)\b", lowered, re.I):
        query.operation = "count"
    elif re.search(r"\b(sort|order|ascending|descending|highest|lowest|top)\b", lowered, re.I):
        query.operation = "sort"
    elif re.search(r"\b(group by|group|aggregate|sum|avg|average|total)\b", lowered, re.I):
        query.operation = "aggregate"
    else:
        query.operation = "filter" if _parse_conditions(lowered, schema) else "report"

    query.conditions = _parse_conditions(lowered, schema)

    if query.operation == "sort":
        rating_col = _resolve_best_column(schema, "rating_metric")
        if rating_col:
            direction = "desc" if re.search(r"\b(highest|descending|desc|top)\b", lowered, re.I) else "asc"
            query.sort = [{"column_id": rating_col, "direction": direction}]
    elif query.operation == "count":
        query.report = "count_rows"

    if re.search(r"\bshow\b", lowered, re.I) and query.operation == "report":
        query.operation = "filter" if query.conditions else "report"

    return query.as_dict()


def parse_anonymized_remote_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Placeholder for a future remote planner.

    The secure build keeps this local-only. If remote AI is ever enabled, the
    payload should already be anonymized and validated before reaching that
    point.
    """
    return payload

