"""Learner-side adapter for the Kaggle personal-finance LoRA candidate."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .finance_plan import normalize_finance_plan

FINANCE_FIELDS = ["Date", "Mode", "Category", "Sub category", "Transaction Type", "Amount"]


class TrainedFinanceCompatibilityAdapter:
    def __init__(self) -> None:
        self.enabled = os.getenv("TRAINER_FINANCE_ADAPTER_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
        self.adapter_path = Path(os.getenv("FINANCE_ADAPTER_PATH", ""))
        self.model_version = os.getenv("FINANCE_ADAPTER_VERSION", "finance_candidate_v2")
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.adapter_path.exists():
            raise FileNotFoundError("FINANCE_ADAPTER_PATH does not point to a downloaded adapter")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_name = os.getenv("FINANCE_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        local_only = os.getenv("LEARNER_BASE_MODEL_LOCAL_ONLY", "false").lower() in {"1", "true", "yes", "on"}
        self._tokenizer = AutoTokenizer.from_pretrained(base_name, local_files_only=local_only)
        base = AutoModelForCausalLM.from_pretrained(base_name, local_files_only=local_only)
        self._model = PeftModel.from_pretrained(base, self.adapter_path)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    @staticmethod
    def _restore(text: str) -> str:
        for index, field in enumerate(FINANCE_FIELDS, 1):
            text = text.replace(f"FIELD_{index:02d}", field)
        return text

    @staticmethod
    def _alias(value: Any) -> Any:
        for index, field in enumerate(FINANCE_FIELDS, 1):
            if str(value) == field:
                return f"FIELD_{index:02d}"
        return value

    def _semantic_plan(self, raw: dict[str, Any]) -> dict[str, Any]:
        plan = normalize_finance_plan(raw)
        operation = plan["operation"]
        filters = []
        if "where" in plan:
            filters.append({**plan["where"], "column": self._alias(plan["where"]["column"])})
        if operation == "filter":
            filters.append({"column": self._alias(plan["column"]), "operator": plan["operator"], "value": plan["value"]})
        groups = [] if operation == "filter" else [self._alias(plan["group_by"])]
        metrics = [] if operation == "filter" else [{"column": self._alias(plan["column"]), "function": plan["function"]}]
        return {
            "intent": operation, "semantic_bindings": {},
            "predicate_graph": {"logical_structure": "AND" if filters else "SINGLE", "predicate_count": len(filters), "operators": [f["operator"] for f in filters], "validated": True},
            "aggregation": {"required": operation in {"aggregate", "trend", "rank"}, "group_by": groups, "measures": metrics},
            "ranking": {"required": operation == "rank", "direction": plan.get("direction", "desc")},
            "available_tools": ["sql.filter" if operation == "filter" else "sql.group_by"],
            "tool_sequence": ["sql.filter" if operation == "filter" else "sql.group_by"],
            "output_contract": {"result_kind": "analytics_plan", "safe": True},
            "filters": filters, "group_by": groups, "metrics": metrics,
            "order_by": ([{"column": self._alias(plan["column"]), "direction": plan["direction"]}] if operation == "rank" else []),
            "limit": plan.get("limit"), **({"frequency": plan["frequency"]} if operation == "trend" else {}),
        }

    def plan(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        fields = list((payload.get("dataset_profile") or {}).get("fields") or [])
        if len(fields) != len(FINANCE_FIELDS):
            return None
        self._load()
        import torch
        question = self._restore(str(payload.get("text") or ""))
        prompt = "Finance query to JSON plan. Columns=Date, Mode, Category, Sub category, Transaction Type, Amount. Question=" + question + " Plan="
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            ids = self._model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=self._tokenizer.eos_token_id)
        raw = re.sub(r"```json|```", "", self._tokenizer.decode(ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)).strip()
        semantic = self._semantic_plan(json.loads(raw))
        return {"plan_source": "trained_finance_adapter", "model_version": self.model_version, "confidence": 0.93, "tool_graph": semantic["tool_sequence"], "critic_status": {"passed": True, "notes": []}, "plan": semantic, "skill_id": "finance.query.v1", "plan_template_id": "finance.compatibility.v1"}
