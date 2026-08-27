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
    @staticmethod
    def _plan_predicate_count(plan: dict[str, Any] | None) -> int:
        plan = plan or {}
        filters = plan.get("filters")
        if isinstance(filters, list) and filters:
            return len(filters)
        predicate_graph = plan.get("predicate_graph")
        if isinstance(predicate_graph, dict):
            value = predicate_graph.get("predicate_count")
            if isinstance(value, int) and value >= 0:
                return value
        tool_graph = plan.get("tool_graph")
        if isinstance(tool_graph, list) and tool_graph:
            return len(tool_graph)
        return 0

    def validate(
        self,
        decision: LearningDecision,
        result_summary: dict[str, Any] | None,
        result_payload: dict[str, Any] | None = None,
    ) -> ValidationResult:
        notes: list[str] = []
        summary = result_summary or result_payload or {}
        payload = result_payload or {}
        route = decision.route
        if route == "sql":
            plan = decision.plan or {}
            filters = plan.get("filters") or []
            planned_predicates = self._plan_predicate_count(plan)
            row_count = summary.get("row_count")
            if filters and row_count is None:
                notes.append("sql result is missing a row count")
            if filters and isinstance(row_count, int) and row_count < 0:
                notes.append("sql result row count cannot be negative")
            if decision.features.get("logical_structure") in {"AND", "MIXED"} and planned_predicates < 2:
                notes.append("sql result may have dropped an explicit condition")
            expected_columns = plan.get("group_by") or []
            if expected_columns and isinstance(summary.get("columns"), list):
                missing = [column for column in expected_columns if column not in summary.get("columns", [])]
                if missing:
                    notes.append("sql result is missing a requested grouping column")
            if filters and isinstance(payload.get("rows"), list):
                rows = payload.get("rows") or []
                for row in rows:
                    if not isinstance(row, dict):
                        notes.append("sql result rows are not dictionaries")
                        break
                    for predicate in filters:
                        column = predicate.get("column")
                        if column not in row:
                            notes.append(f"sql result row is missing column {column}")
                            break
                        value = row.get(column)
                        operator = predicate.get("operator")
                        target = predicate.get("value")
                        if operator == "equals" and str(value) != str(target):
                            notes.append(f"row does not satisfy {column} equals {target}")
                        elif operator == "not_equals" and str(value) == str(target):
                            notes.append(f"row does not satisfy {column} not_equals {target}")
                        elif operator == "greater_than" and float(value) <= float(target):
                            notes.append(f"row does not satisfy {column} greater_than {target}")
                        elif operator == "less_than" and float(value) >= float(target):
                            notes.append(f"row does not satisfy {column} less_than {target}")
                        elif operator == "greater_than_equal" and float(value) < float(target):
                            notes.append(f"row does not satisfy {column} greater_than_equal {target}")
                        elif operator == "less_than_equal" and float(value) > float(target):
                            notes.append(f"row does not satisfy {column} less_than_equal {target}")
                    if notes:
                        break
        elif route == "operation":
            if summary.get("result_kind") == "error":
                notes.append("operation reported an error result")
            if (decision.plan or {}).get("action") == "multi_step" and not summary.get("steps"):
                notes.append("multi-step operation did not report step details")
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
