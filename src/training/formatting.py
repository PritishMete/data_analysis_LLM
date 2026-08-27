from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PlannerTrainingExample:
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"input": self.input, "output": self.output, "metadata": self.metadata}


def fine_tuning_candidate_to_example(candidate: dict[str, Any]) -> PlannerTrainingExample:
    return PlannerTrainingExample(
        input={
            "intent": candidate.get("intent"),
            "safe_semantic_schema": candidate.get("semantic_roles") or [],
            "predicate_graph": candidate.get("predicate_graph") or {},
            "logical_structure": candidate.get("logical_structure") or "SINGLE",
            "available_tools": candidate.get("tool_graph") or candidate.get("tool_sequence") or [],
            "safe_constraints": {
                "privacy": True,
                "raw_values_allowed": False,
                "raw_column_names_allowed": False,
            },
        },
        output={
            "structured_plan": candidate.get("output") or {},
            "tool_graph": candidate.get("tool_graph") or candidate.get("tool_sequence") or [],
            "expected_output_contract": candidate.get("output_contract") or {},
        },
        metadata={
            "source_kind": candidate.get("source_kind"),
            "source_id": candidate.get("source_id"),
            "quality": candidate.get("quality_score") or candidate.get("quality"),
            "plan_source": candidate.get("plan_source"),
            "family_fingerprint": candidate.get("family_fingerprint"),
        },
    )
