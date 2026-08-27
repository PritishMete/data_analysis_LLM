from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from learning.plan_schema import canonical_plan_schema


@dataclass(slots=True)
class PlannerTrainingExample:
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"input": self.input, "output": self.output, "metadata": self.metadata}


def fine_tuning_candidate_to_example(candidate: dict[str, Any]) -> PlannerTrainingExample:
    available_tools = list(candidate.get("available_tools") or candidate.get("tool_graph") or candidate.get("tool_sequence") or [])
    plan_schema = canonical_plan_schema(
        intent=str(candidate.get("intent") or ""),
        semantic_roles=[str(item) for item in candidate.get("semantic_roles") or []],
        predicate_graph=dict(candidate.get("predicate_graph") or {}),
        logical_structure=str(candidate.get("logical_structure") or "SINGLE"),
        available_tools=available_tools,
        tool_graph=list(candidate.get("tool_graph") or candidate.get("tool_sequence") or []),
        output_contract=dict(candidate.get("output_contract") or {}),
    )
    return PlannerTrainingExample(
        input={
            "intent": candidate.get("intent"),
            "safe_semantic_schema": candidate.get("semantic_roles") or [],
            "predicate_graph": candidate.get("predicate_graph") or {},
            "logical_structure": candidate.get("logical_structure") or "SINGLE",
            "available_tools": available_tools,
            "tool_definitions": [tool.to_dict() for tool in plan_schema.tool_definitions],
            "safe_constraints": {
                "privacy": True,
                "raw_values_allowed": False,
                "raw_column_names_allowed": False,
            },
        },
        output={
            "structured_plan": candidate.get("output") or {},
            "tool_graph": list(candidate.get("tool_graph") or candidate.get("tool_sequence") or []),
            "expected_output_contract": candidate.get("output_contract") or {},
        },
        metadata={
            "source_kind": candidate.get("source_kind"),
            "source_id": candidate.get("source_id"),
            "quality": candidate.get("quality_score") or candidate.get("quality"),
            "plan_source": candidate.get("plan_source"),
            "family_fingerprint": candidate.get("family_fingerprint"),
            "schema_version": plan_schema.schema_version,
        },
    )
