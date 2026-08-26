from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from learning.models import LearningDecision


@dataclass(slots=True)
class ValidationResult:
    success: bool
    notes: list[str]
    repair_hint: str | None = None


class ResultValidator:
    def validate(self, decision: LearningDecision, result_summary: dict[str, Any] | None) -> ValidationResult:
        notes: list[str] = []
        summary = result_summary or {}
        route = decision.route
        if route == "sql":
            filters = (decision.plan or {}).get("filters") or []
            row_count = summary.get("row_count")
            if filters and row_count is None:
                notes.append("sql result is missing a row count")
            if filters and isinstance(row_count, int) and row_count < 0:
                notes.append("sql result row count cannot be negative")
            if decision.features.get("logical_structure") in {"AND", "MIXED"} and len(filters) < 2:
                notes.append("sql result may have dropped an explicit condition")
        elif route == "operation":
            if summary.get("result_kind") == "error":
                notes.append("operation reported an error result")
        elif route == "sentiment":
            if not summary and not decision.validation_notes:
                notes.append("sentiment result is empty")

        success = not notes
        repair_hint = None
        if not success and route == "sql":
            repair_hint = "rebuild the filter plan and preserve every requested predicate"
        elif not success and route == "operation":
            repair_hint = "repair the operation plan and retry locally"
        return ValidationResult(success=success, notes=notes, repair_hint=repair_hint)
