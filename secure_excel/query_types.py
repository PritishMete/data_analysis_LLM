"""Structured query types used by the secure Excel pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_OPERATIONS = {"filter", "sort", "group", "aggregate", "search", "count", "report"}
SUPPORTED_OPERATORS = {
    "equals",
    "not_equals",
    "contains",
    "starts_with",
    "ends_with",
    "greater_than",
    "less_than",
    "greater_equal",
    "less_equal",
    "between",
    "is_null",
    "is_not_null",
}


@dataclass
class QueryCondition:
    column_id: str
    operator: str
    value: Any = None
    value2: Any = None


@dataclass
class StructuredQuery:
    operation: str
    conditions: list[QueryCondition] = field(default_factory=list)
    sort: list[dict[str, str]] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    aggregates: list[dict[str, Any]] = field(default_factory=list)
    limit: int | None = None
    search: str | None = None
    report: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "conditions": [condition.__dict__ for condition in self.conditions],
            "sort": self.sort,
            "group_by": self.group_by,
            "aggregates": self.aggregates,
            "limit": self.limit,
            "search": self.search,
            "report": self.report,
        }

