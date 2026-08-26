from __future__ import annotations

from learning.models import LearningDecision


class PlanCritic:
    """Lightweight plan safety and completeness checks."""

    def review(self, decision: LearningDecision) -> tuple[bool, list[str]]:
        notes: list[str] = []
        if decision.route not in {"sql", "operation", "sentiment", "unknown"}:
            notes.append(f"unsupported route: {decision.route}")
        if decision.route == "sql":
            if not decision.plan:
                notes.append("sql decision is missing a plan")
            elif not decision.plan.get("filters") and not decision.plan.get("group_by") and not decision.plan.get("metrics"):
                notes.append("sql plan is too vague")
        if decision.route == "operation" and not decision.plan:
            notes.append("operation decision is missing a command")
        if decision.confidence < 0:
            notes.append("confidence cannot be negative")
        return not notes, notes
