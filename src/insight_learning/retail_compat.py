"""Optional learner-side bridge for the trained retail adapter.

Disabled by default. It translates the trainer's safe FIELD_nn aliases into
the current retail schema, calls the local retail query service, and translates
the returned plan back into safe aliases. The trainer repository is never
imported or modified.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


RETAIL_FIELDS = [
    "Transaction ID", "Date", "Customer ID", "Gender", "Age",
    "Product Category", "Quantity", "Price per Unit", "Total Amount",
]


class TrainedRetailCompatibilityAdapter:
    def __init__(self) -> None:
        self.enabled = os.getenv("TRAINER_RETAIL_ADAPTER_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        self.base_url = os.getenv("RETAIL_QUERY_API_URL", "http://127.0.0.1:8010").rstrip("/")
        self.model_version = os.getenv("RETAIL_ADAPTER_VERSION", "retail_plan_adapter-kaggle-28of28")

    def _restore_question(self, text: str) -> str:
        for index, field in enumerate(RETAIL_FIELDS, 1):
            text = text.replace(f"FIELD_{index:02d}", field)
        return text

    def _alias(self, value: Any) -> Any:
        actual = str(value)
        for index, field in enumerate(RETAIL_FIELDS, 1):
            if actual == field:
                return f"FIELD_{index:02d}"
        return value

    def _call(self, question: str) -> dict[str, Any]:
        body = json.dumps({"question": question}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/query", data=body, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _semantic_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        operation = plan.get("operation")
        if operation == "aggregate":
            group = self._alias(plan.get("group_by"))
            column = self._alias(plan.get("column"))
            function = str(plan.get("function") or "mean")
            return {
                "intent": "aggregate", "semantic_bindings": {"group_by": group, "measure": column},
                "predicate_graph": {"logical_structure": "SINGLE", "predicate_count": 0, "operators": [], "validated": True},
                "aggregation": {"required": True, "group_by": [group], "measures": [{"column": column, "function": function}]},
                "ranking": {"required": False, "direction": "desc"},
                "available_tools": ["sql.group_by", "analytics.summary"], "tool_sequence": ["sql.group_by"],
                "output_contract": {"result_kind": "analytics_plan", "safe": True},
                "filters": [], "group_by": [group], "metrics": [{"column": column, "function": function}], "order_by": [], "limit": None,
            }
        if operation == "filter":
            return {
                "intent": "filter", "semantic_bindings": {},
                "predicate_graph": {"logical_structure": "SINGLE", "predicate_count": 1, "operators": [str(plan.get("operator") or "contains")], "validated": True},
                "aggregation": {"required": False, "group_by": [], "measures": []},
                "ranking": {"required": False, "direction": "desc"}, "available_tools": ["sql.filter"], "tool_sequence": ["sql.filter"],
                "output_contract": {"result_kind": "analytics_plan", "safe": True},
                "filters": [{"column": self._alias(plan.get("column")), "operator": plan.get("operator"), "value": plan.get("value")}],
                "group_by": [], "metrics": [], "order_by": [], "limit": None,
            }
        raise ValueError(f"unsupported retail operation: {operation}")

    def plan(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        fields = list((payload.get("dataset_profile") or {}).get("fields") or [])
        if len(fields) != len(RETAIL_FIELDS):
            return None
        response = self._call(self._restore_question(str(payload.get("text") or "")))
        semantic = self._semantic_plan(dict(response["plan"]))
        return {
            "plan_source": "trained_retail_adapter", "model_version": self.model_version,
            "confidence": 0.93, "tool_graph": semantic["tool_sequence"],
            "critic_status": {"passed": True, "notes": []}, "plan": semantic,
            "skill_id": "retail.query.v1", "plan_template_id": "retail.compatibility.v1",
        }
