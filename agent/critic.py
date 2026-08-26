from __future__ import annotations

from typing import Any

from learning.models import LearningDecision, PlannerContext


class PlanCritic:
    """Lightweight plan safety and completeness checks."""

    def review(self, decision: LearningDecision, context: PlannerContext | None = None) -> tuple[bool, list[str]]:
        notes: list[str] = []
        features = context.features if context is not None else None
        if decision.route not in {"sql", "operation", "sentiment", "unknown"}:
            notes.append(f"unsupported route: {decision.route}")
        if decision.route == "sql":
            if not decision.plan:
                notes.append("sql decision is missing a plan")
            else:
                filters = decision.plan.get("filters") or []
                group_by = decision.plan.get("group_by") or []
                metrics = decision.plan.get("metrics") or []
                if not filters and not group_by and not metrics:
                    notes.append("sql plan is too vague")
                if features is not None and features.logical_structure in {"AND", "MIXED"} and len(filters) < 2:
                    notes.append("sql plan does not preserve the requested multi-condition structure")
        if decision.route == "operation" and not decision.plan:
            notes.append("operation decision is missing a command")
        if decision.confidence < 0:
            notes.append("confidence cannot be negative")
        if context is not None and not context.dataset_profile.get("available_columns"):
            notes.append("no available columns were supplied")
        return not notes, notes
