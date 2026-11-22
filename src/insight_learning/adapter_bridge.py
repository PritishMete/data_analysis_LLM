"""Runtime bridge for packaged QLoRA adapters.

Adapters are optional: a missing checkpoint never prevents InsightFlow from
using its existing planner. The bridge only returns validated JSON plans.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


class TrainedAdapter:
    def __init__(self, domain: str, path: Path, version: str):
        self.domain, self.path, self.version = domain, path, version
        self._model = self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not (self.path / "adapter_config.json").exists():
            raise FileNotFoundError(f"adapter checkpoint missing: {self.path}")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_name = os.getenv("LEARNER_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        local_only = os.getenv("LEARNER_BASE_MODEL_LOCAL_ONLY", "false").lower() in {"1", "true", "yes", "on"}
        self._tokenizer = AutoTokenizer.from_pretrained(base_name, local_files_only=local_only)
        base = AutoModelForCausalLM.from_pretrained(base_name, local_files_only=local_only)
        self._model = PeftModel.from_pretrained(base, self.path)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    @staticmethod
    def _json(raw: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("adapter_returned_no_json")
        value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("adapter_returned_non_object")
        return value

    def plan(self, question: str, fields: list[str]) -> dict[str, Any]:
        self._load()
        import torch
        prompt = f"{self.domain.title()} query to JSON plan. Columns={json.dumps(fields)} Question={question} Plan="
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            ids = self._model.generate(**inputs, max_new_tokens=160, do_sample=False, pad_token_id=self._tokenizer.eos_token_id)
        raw = self._tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        plan = self._json(raw)
        if plan.get("operation") not in {"filter", "aggregate", "rank", "trend"}:
            raise ValueError("adapter_returned_unsupported_operation")
        return plan


def deterministic_restaurant_plan(question: str) -> dict[str, Any] | None:
    """High-confidence guardrail for common restaurant requests.

    The small adapter remains useful for novel wording, but these explicit
    forms should never be sent through an uncertain generative prediction.
    """
    text = question.lower().strip()
    cities = {c.lower(): c for c in ("Bangalore", "Kolkata", "Mumbai", "Delhi", "Chennai")}
    cuisines = {c.lower(): c for c in ("Biryani", "Chinese", "North Indian", "South Indian", "Italian")}
    city = next((v for k, v in cities.items() if k in text), None)
    cuisine = next((v for k, v in cuisines.items() if k in text), None)
    if "top" in text or "most expensive" in text or "highest" in text:
        match = re.search(r"\btop\s+(\d+)", text) or re.search(r"\b(?:most expensive|expensive restaurants?)\s+(\d+)", text) or re.search(r"\b(\d+)\s+most expensive", text)
        limit = int(match.group(1)) if match else 1
        column = "cost" if "expensive" in text or "cost" in text else "rating"
        plan: dict[str, Any] = {"operation": "rank", "column": column, "direction": "desc", "limit": limit}
        if city:
            plan["where"] = {"column": "city", "operator": "eq", "value": city}
        return plan
    match = re.search(r"(?:rated|rating)\s+(?:above|over|greater than)\s+(\d+(?:\.\d+)?)", text)
    if match:
        return {"operation": "filter", "column": "rating", "operator": "gt", "value": float(match.group(1))}
    if city and ("restaurant" in text or "restaurants" in text):
        return {"operation": "filter", "column": "city", "operator": "eq", "value": city}
    if cuisine:
        return {"operation": "filter", "column": "cuisine", "operator": "eq", "value": cuisine}
    match = re.search(r"(?:average|mean)\s+(rating|cost)\s+by\s+(city|cuisine)", text)
    if match:
        return {"operation": "aggregate", "group_by": match.group(2), "column": match.group(1), "function": "mean"}
    return None


def semantic_envelope(domain: str, version: str, raw: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    operation = raw["operation"]
    def alias(value: Any) -> Any:
        return aliases.get(str(value), value)
    group = alias(raw.get("group_by")) if operation != "filter" else None
    column = alias(raw.get("column")) if raw.get("column") else None
    filters = []
    if operation == "filter" and column:
        filters.append({"column": column, "operator": raw.get("operator", "eq"), "value": raw.get("value")})
    where = raw.get("where")
    if where is not None:
        if not isinstance(where, dict) or not {"column", "operator", "value"} <= where.keys():
            raise ValueError("adapter_returned_invalid_where")
        if where["operator"] not in {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}:
            raise ValueError("adapter_returned_invalid_where_operator")
        filters.append({"column": alias(where["column"]), "operator": where["operator"], "value": where["value"]})
    metrics = [{"column": column, "function": raw.get("function", "mean")}] if operation in {"aggregate", "rank"} else []
    return {
        "intent": operation, "semantic_bindings": {},
        "predicate_graph": {"logical_structure": "SINGLE", "predicate_count": len(filters), "operators": [f["operator"] for f in filters], "validated": True},
        "aggregation": {"required": operation == "aggregate", "group_by": [group] if group else [], "measures": metrics},
        "ranking": {"required": operation == "rank", "direction": raw.get("direction", "desc")},
        "available_tools": ["sql.filter" if operation == "filter" else "sql.group_by"],
        "tool_sequence": ["sql.filter" if operation == "filter" else "sql.group_by"],
        "output_contract": {"result_kind": "analytics_plan", "safe": True},
        "filters": filters, "group_by": [group] if group else [], "metrics": metrics,
        "order_by": [{"column": column, "direction": raw.get("direction", "desc")}] if operation == "rank" else [],
        "limit": raw.get("limit"),
        **({"frequency": raw["frequency"]} if operation == "trend" else {}),
    }
