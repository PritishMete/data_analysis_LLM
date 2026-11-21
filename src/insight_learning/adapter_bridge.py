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
        self._tokenizer = AutoTokenizer.from_pretrained(base_name, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(base_name, local_files_only=True)
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


def semantic_envelope(domain: str, version: str, raw: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    operation = raw["operation"]
    def alias(value: Any) -> Any:
        return aliases.get(str(value), value)
    group = alias(raw.get("group_by")) if operation != "filter" else None
    column = alias(raw.get("column")) if raw.get("column") else None
    filters = []
    if operation == "filter" and column:
        filters.append({"column": column, "operator": raw.get("operator", "eq"), "value": raw.get("value")})
    metrics = [] if operation == "filter" else [{"column": column, "function": raw.get("function", "mean")}]
    return {
        "intent": operation, "semantic_bindings": {},
        "predicate_graph": {"logical_structure": "SINGLE", "predicate_count": len(filters), "operators": [f["operator"] for f in filters], "validated": True},
        "aggregation": {"required": operation != "filter", "group_by": [group] if group else [], "measures": metrics},
        "ranking": {"required": operation == "rank", "direction": raw.get("direction", "desc")},
        "available_tools": ["sql.filter" if operation == "filter" else "sql.group_by"],
        "tool_sequence": ["sql.filter" if operation == "filter" else "sql.group_by"],
        "output_contract": {"result_kind": "analytics_plan", "safe": True},
        "filters": filters, "group_by": [group] if group else [], "metrics": metrics,
        "order_by": [{"column": column, "direction": raw.get("direction", "desc")}] if operation == "rank" else [],
        "limit": raw.get("limit"),
        **({"frequency": raw["frequency"]} if operation == "trend" else {}),
    }
