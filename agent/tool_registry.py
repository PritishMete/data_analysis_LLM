from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    kind: str
    description: str


LOCAL_TOOL_ALLOWLIST: dict[str, ToolSpec] = {
    "sql.filter": ToolSpec(
        name="sql.filter",
        kind="sql",
        description="Builds a local row filter against the active dataframe.",
    ),
    "sql.group_by": ToolSpec(
        name="sql.group_by",
        kind="sql",
        description="Builds a grouped summary query against the active dataframe.",
    ),
    "analytics.summary": ToolSpec(
        name="analytics.summary",
        kind="analytics",
        description="Builds a privacy-safe shape and summary report.",
    ),
    "common.transformations.range_binning": ToolSpec(
        name="common.transformations.range_binning",
        kind="operation",
        description="Bins numeric values into labelled ranges.",
    ),
    "categorization_agent._deterministic_special_mapping": ToolSpec(
        name="categorization_agent._deterministic_special_mapping",
        kind="operation",
        description="Performs deterministic column categorization or normalization.",
    ),
    "data_cleaning_utils.fill_nulls": ToolSpec(
        name="data_cleaning_utils.fill_nulls",
        kind="operation",
        description="Fills missing values with a deterministic strategy.",
    ),
    "secure_excel.executor": ToolSpec(
        name="secure_excel.executor",
        kind="privacy",
        description="Keeps workbook execution local and privacy-safe.",
    ),
}


class ToolRegistry:
    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in LOCAL_TOOL_ALLOWLIST

    def get(self, tool_name: str) -> ToolSpec | None:
        return LOCAL_TOOL_ALLOWLIST.get(tool_name)

    def allowed_names(self) -> list[str]:
        return sorted(LOCAL_TOOL_ALLOWLIST)


_REGISTRY = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    return _REGISTRY
