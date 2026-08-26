from __future__ import annotations

from typing import Any

from learning.models import LearningDecision, PlannerContext
from agent.tool_registry import get_tool_registry


class PlanCritic:
    """Lightweight plan safety and completeness checks."""

    @staticmethod
    def _predicate_leaf_count(features) -> int:
        total = 0
        graph = getattr(features, "predicate_graph", None) or []
        stack = list(graph)
        while stack:
            node = stack.pop()
            children = node.get("children") if isinstance(node, dict) else None
            if children:
                stack.extend(children)
                continue
            total += 1
        return total

    def review(self, decision: LearningDecision, context: PlannerContext | None = None) -> tuple[bool, list[str]]:
        notes: list[str] = []
        features = context.features if context is not None else None
        tool_registry = get_tool_registry()
        if decision.route not in {"sql", "operation", "sentiment", "unknown"}:
            notes.append(f"unsupported route: {decision.route}")
        if decision.route == "sql":
            if not decision.plan:
                notes.append("sql decision is missing a plan")
            else:
                filters = decision.plan.get("filters") or []
                group_by = decision.plan.get("group_by") or []
                metrics = decision.plan.get("metrics") or []
                tool_sequence = decision.tool_sequence or []
                if not filters and not group_by and not metrics:
                    notes.append("sql plan is too vague")
                requested_predicates = self._predicate_leaf_count(features) if features is not None else 0
                if requested_predicates and len(filters) < requested_predicates:
                    notes.append("sql plan dropped an explicit predicate")
                if features is not None and features.logical_structure in {"AND", "MIXED"} and len(filters) < 2:
                    notes.append("sql plan does not preserve the requested multi-condition structure")
                if features is not None and features.logical_structure == "OR" and len(filters) < 2:
                    notes.append("sql plan does not preserve disjunction structure")
                for tool_name in tool_sequence:
                    if not tool_registry.is_allowed(tool_name):
                        notes.append(f"disallowed tool: {tool_name}")
        if decision.route == "operation" and not decision.plan:
            notes.append("operation decision is missing a command")
        if decision.route == "operation" and decision.plan is not None:
            action = str(decision.plan.get("action") or "")
            if action and action not in {"categorize", "filter", "deduplicate", "color_scale", "fill_missing", "multi_step"}:
                notes.append(f"unsupported operation action: {action}")
        if decision.confidence < 0:
            notes.append("confidence cannot be negative")
        if context is not None and not context.dataset_profile.get("available_columns"):
            notes.append("no available columns were supplied")
        return not notes, notes
