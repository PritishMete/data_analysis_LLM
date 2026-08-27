from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent.tool_registry import get_tool_registry


CANONICAL_PLAN_SCHEMA_VERSION = 1
CANONICAL_PLAN_TOOL_KEYS = tuple(get_tool_registry().allowed_names())


@dataclass(slots=True)
class ToolDefinition:
    tool: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanSchema:
    schema_version: int = CANONICAL_PLAN_SCHEMA_VERSION
    intent: str = ""
    semantic_roles: list[str] = field(default_factory=list)
    predicate_graph: dict[str, Any] = field(default_factory=dict)
    logical_structure: str = "SINGLE"
    available_tools: list[str] = field(default_factory=list)
    tool_definitions: list[ToolDefinition] = field(default_factory=list)
    output_contract: dict[str, Any] = field(default_factory=dict)
    tool_graph: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_definitions"] = [tool.to_dict() for tool in self.tool_definitions]
        return payload

    def validate(self) -> list[str]:
        notes: list[str] = []
        registry = get_tool_registry()
        if not self.intent:
            notes.append("missing_intent")
        if self.logical_structure not in {"SINGLE", "AND", "OR", "MIXED", "SEQUENTIAL"}:
            notes.append("invalid_logical_structure")
        for tool in self.available_tools:
            if not registry.is_allowed(tool):
                notes.append(f"unknown_tool:{tool}")
        for tool in self.tool_graph:
            if not registry.is_allowed(tool):
                notes.append(f"unknown_tool:{tool}")
        if not self.output_contract:
            notes.append("missing_output_contract")
        return notes


def canonical_tool_definitions() -> list[ToolDefinition]:
    registry = get_tool_registry()
    return [
        ToolDefinition(tool=spec.name, description=spec.description)
        for spec in (registry.get(name) for name in registry.allowed_names())
        if spec is not None
    ]


def canonical_plan_schema(
    *,
    intent: str,
    semantic_roles: list[str],
    predicate_graph: dict[str, Any],
    logical_structure: str,
    available_tools: list[str],
    tool_graph: list[str],
    output_contract: dict[str, Any] | None = None,
) -> PlanSchema:
    return PlanSchema(
        intent=intent,
        semantic_roles=list(semantic_roles),
        predicate_graph=dict(predicate_graph),
        logical_structure=logical_structure,
        available_tools=list(available_tools),
        tool_definitions=canonical_tool_definitions(),
        output_contract=dict(output_contract or {}),
        tool_graph=list(tool_graph),
    )


def validate_plan_schema(payload: dict[str, Any]) -> list[str]:
    schema = PlanSchema(
        schema_version=int(payload.get("schema_version") or CANONICAL_PLAN_SCHEMA_VERSION),
        intent=str(payload.get("intent") or ""),
        semantic_roles=[str(item) for item in payload.get("semantic_roles") or []],
        predicate_graph=dict(payload.get("predicate_graph") or {}),
        logical_structure=str(payload.get("logical_structure") or "SINGLE"),
        available_tools=[str(item) for item in payload.get("available_tools") or []],
        output_contract=dict(payload.get("output_contract") or {}),
        tool_graph=[str(item) for item in payload.get("tool_graph") or []],
    )
    return schema.validate()
