from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gc
from pathlib import Path
import json
import os
import time
from typing import Any, Protocol

import pandas as pd
import torch

from agent.critic import PlanCritic
from learning.feature_extractor import build_planner_context
from learning.models import LearningDecision
from agent.planner import LearningPlanner
from agent.tool_registry import get_tool_registry
from .hardware import detect_hardware
from .profiles import (
    PLANNER_BACKEND_AUTO,
    PLANNER_BACKEND_LLAMA_CPP,
    PLANNER_BACKEND_TRANSFORMERS,
    LOW_SPEC_MODEL_PROFILE,
    choose_backend,
    select_model_profile,
    select_runtime_profile,
)


class PlannerModel(Protocol):
    def plan(self, request: dict[str, Any]) -> dict[str, Any]: ...
    def health(self) -> dict[str, Any]: ...
    def metadata(self) -> dict[str, Any]: ...


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value, sort_keys=True, default=str)
        return value
    except Exception:
        return {"error": "non_serializable"}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _stringify_sequence(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [str(item) for item in values if item is not None]


def _normalize_plan_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    plan = dict(payload)
    if "tool_sequence" in plan:
        plan["tool_sequence"] = _stringify_sequence(plan.get("tool_sequence"))
    if "tool_graph" in plan:
        plan["tool_graph"] = _stringify_sequence(plan.get("tool_graph"))
    if "semantic_roles" in plan and isinstance(plan.get("semantic_roles"), (list, tuple)):
        plan["semantic_roles"] = [str(item) for item in plan.get("semantic_roles") if item is not None]
    return plan


@dataclass(slots=True)
class PlannerBenchmarkCase:
    name: str
    text: str
    intent: str
    expected_tool_sequence: list[str]
    expected_predicate_count: int
    expected_logical_structure: str
    expected_semantic_roles: list[str]


def builtin_benchmark_cases() -> list[PlannerBenchmarkCase]:
    return [
        PlannerBenchmarkCase("simple_filter", "Show restaurants with delivery", "filter", ["sql.filter"], 1, "SINGLE", ["boolean_capability"]),
        PlannerBenchmarkCase("multi_and", "Show restaurants with delivery and booking", "filter", ["sql.filter"], 2, "AND", ["boolean_capability"]),
        PlannerBenchmarkCase("nested_and_or", "Show restaurants with delivery and booking or rating above 4", "filter", ["sql.filter"], 3, "MIXED", ["boolean_capability", "rating_metric"]),
        PlannerBenchmarkCase("sorting", "Show restaurants sorted by rating", "analytics", ["sql.group_by"], 1, "SINGLE", ["rating_metric"]),
        PlannerBenchmarkCase("top_n", "Show top 5 restaurants by rating", "analytics", ["sql.group_by"], 1, "SINGLE", ["rating_metric"]),
        PlannerBenchmarkCase("group_sum", "Group sales by region and sum revenue", "analytics", ["sql.group_by"], 1, "SINGLE", ["numeric_metric", "geographic_area"]),
        PlannerBenchmarkCase("group_avg", "Group by region and average revenue", "analytics", ["sql.group_by"], 1, "SINGLE", ["numeric_metric", "geographic_area"]),
        PlannerBenchmarkCase("group_count", "Count rows by category", "analytics", ["sql.group_by"], 1, "SINGLE", ["category"]),
        PlannerBenchmarkCase("missing_value", "Find columns with missing values", "cleaning", ["categorization_agent._deterministic_special_mapping"], 0, "SINGLE", []),
        PlannerBenchmarkCase("duplicate_detection", "Deduplicate duplicate rows", "cleaning", ["categorization_agent._deterministic_special_mapping"], 0, "SINGLE", []),
        PlannerBenchmarkCase("outlier_detection", "Find outlier revenue above 1000", "analytics", ["sql.group_by"], 1, "SINGLE", ["numeric_metric"]),
        PlannerBenchmarkCase("normalization", "Normalize customer names", "operation", ["categorization_agent._deterministic_special_mapping"], 0, "SINGLE", ["entity_name"]),
        PlannerBenchmarkCase("range_binning", "Bin revenue into ranges", "operation", ["categorization_agent._deterministic_special_mapping"], 1, "SINGLE", ["numeric_metric"]),
        PlannerBenchmarkCase("pivot_planning", "Pivot sales by month and region", "analytics", ["sql.group_by"], 1, "SINGLE", ["numeric_metric", "geographic_area"]),
        PlannerBenchmarkCase("trend", "Show trend of monthly revenue", "analytics", ["sql.group_by"], 1, "SINGLE", ["numeric_metric"]),
        PlannerBenchmarkCase("correlation", "Show correlation between revenue and rating", "analytics", ["sql.group_by"], 2, "AND", ["numeric_metric", "rating_metric"]),
        PlannerBenchmarkCase("multi_step", "Show restaurants with delivery then rank by rating", "analytics", ["sql.filter", "sql.group_by"], 2, "SEQUENTIAL", ["boolean_capability", "rating_metric"]),
    ]


@dataclass(slots=True)
class PlannerInferenceResult:
    case_name: str
    raw_output: str
    parsed_plan: dict[str, Any] | None
    plan_source: str
    schema_valid: bool
    critic_passed: bool
    tool_valid: bool
    predicate_coverage: float
    logical_structure_accuracy: float
    semantic_role_coverage: float
    intent_correct: bool
    tool_selection_f1: float
    tool_sequence_accuracy: float
    invalid_tool: bool
    latency_ms: float
    model_load_ms: float | None = None
    inference_ms: float | None = None
    peak_vram_mb: float | None = None
    valid_json: bool = False
    fallback_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlannerBenchmarkSummary:
    profile: dict[str, Any]
    backend: str
    device: str
    model_id: str | None
    cases: list[PlannerInferenceResult]
    metrics: dict[str, Any]
    hardware: dict[str, Any]
    model_health: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    shadow_mode: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cases"] = [case.to_dict() for case in self.cases]
        return payload


@dataclass(slots=True)
class PlannerFailureModeReport:
    failure_counts: dict[str, int]
    total_cases: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeuristicPlannerModel:
    def __init__(self, profile_name: str = "low_spec") -> None:
        self.profile = select_model_profile(profile_name)
        self.runtime = select_runtime_profile("cpu_low_spec")
        self.planner = LearningPlanner()
        self.critic = PlanCritic()

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        text = str(request.get("text") or request.get("request") or "")
        intent = str(request.get("intent") or "analytics")
        features = request.get("query_features") or {}
        dataset_profile = request.get("dataset_profile") or {}
        available_columns = list(dataset_profile.get("fields") or [])
        columns = [str(item.get("id") if isinstance(item, dict) else item) for item in available_columns]
        df = request.get("dataframe")
        if isinstance(df, pd.DataFrame):
            data_frame = df
        else:
            data_frame = pd.DataFrame({column: [1, 2, 3] for column in columns}) if columns else pd.DataFrame()
        decision = self.planner.plan(
            text,
            data_frame if not data_frame.empty else None,
            columns,
            planner_context=build_planner_context(text, data_frame if not data_frame.empty else None, columns),
        )
        plan = decision.plan or {}
        return {
            "intent": intent,
            "plan_source": decision.plan_source,
            "tool_graph": list(decision.tool_sequence or plan.get("tool_sequence") or []),
            "plan": _safe_json(plan),
            "query_features": features,
        }

    def health(self) -> dict[str, Any]:
        hardware = detect_hardware().to_dict()
        return {
            "available": True,
            "backend": "heuristic",
            "profile": self.profile.to_dict(),
            "runtime": self.runtime.to_dict(),
            "hardware": hardware,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.profile.model_id,
            "parameter_count": self.profile.parameter_count,
            "backend": "heuristic",
            "quantization": self.profile.gguf_quantization,
        }


def _try_import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None, None


class TransformersPlannerModel:
    def __init__(self, *, profile_name: str, device: str = "auto", cache_dir: Path | None = None) -> None:
        self.profile = select_model_profile(profile_name)
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.load_error: str | None = None
        self.model_load_ms: float | None = None

    def load(self) -> None:
        AutoModelForCausalLM, AutoTokenizer = _try_import_transformers()
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            self.load_error = "transformers_unavailable"
            return
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        try:
            started = time.perf_counter()
            load_kwargs: dict[str, Any] = dict(kwargs)
            load_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            if self.device == "cuda" and torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
            self.tokenizer = AutoTokenizer.from_pretrained(self.profile.model_id, **kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(self.profile.model_id, **load_kwargs)
            self.model_load_ms = round((time.perf_counter() - started) * 1000.0, 3)
            if self.device == "cuda" and torch.cuda.is_available() and hasattr(self.model, "to"):
                try:
                    self.model.to("cuda")
                except Exception:
                    pass
        except Exception as exc:
            self.load_error = str(exc)

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            self.load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(self.load_error or "model_load_failed")
        prompt = str(request.get("text") or request.get("prompt") or "")
        chat_messages = [
            {
                "role": "system",
                "content": (
                    "You are a planner that outputs strict JSON only. "
                    "Return a safe analytics plan with keys: intent, tool_graph, plan. "
                    "Do not explain, do not add markdown, and do not include raw data."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            model_prompt = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        else:
            model_prompt = f"{chat_messages[0]['content']}\n{chat_messages[1]['content']}\nJSON:"
        inputs = self.tokenizer(model_prompt, return_tensors="pt")
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.no_grad():
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            generated = self.model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
            inference_ms = round((time.perf_counter() - started) * 1000.0, 3)
        peak_vram_mb = None
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
            except Exception:
                peak_vram_mb = None
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        raw_text = decoded[len(model_prompt):] if decoded.startswith(model_prompt) else decoded
        parsed = _extract_json_object(raw_text)
        if parsed is None:
            parsed = {"plan": {}, "tool_graph": [], "raw_text": raw_text.strip()}
        return {
            "raw_text": raw_text.strip(),
            "plan_source": "transformers",
            "tool_graph": parsed.get("tool_graph") or [],
            "plan": parsed,
            "model_load_ms": self.model_load_ms,
            "inference_ms": inference_ms,
            "peak_vram_mb": peak_vram_mb,
        }

    def health(self) -> dict[str, Any]:
        return {
            "available": self.model is not None and self.tokenizer is not None,
            "backend": "transformers",
            "model_id": self.profile.model_id,
            "load_error": self.load_error,
        }

    def metadata(self) -> dict[str, Any]:
        return {"model_id": self.profile.model_id, "backend": "transformers"}


class LlamaCppPlannerAdapter:
    def __init__(self, *, model_path: Path | None = None, profile_name: str = "low_spec") -> None:
        self.profile = select_model_profile(profile_name)
        self.model_path = model_path
        self.load_error: str | None = None

    def _has_llama_cpp(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except Exception:
            return False

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._has_llama_cpp():
            raise RuntimeError("llama_cpp_unavailable")
        prompt = str(request.get("text") or request.get("prompt") or "")
        return {"raw_text": prompt, "plan_source": "llama_cpp", "tool_graph": [], "plan": {}}

    def health(self) -> dict[str, Any]:
        return {
            "available": self._has_llama_cpp(),
            "backend": "llama_cpp",
            "model_path": str(self.model_path) if self.model_path else None,
            "load_error": self.load_error,
        }

    def metadata(self) -> dict[str, Any]:
        return {"model_id": self.profile.model_id, "backend": "llama_cpp", "quantization": self.profile.gguf_quantization}


def _score_plan(candidate: dict[str, Any], case: PlannerBenchmarkCase, critic_passed: bool) -> dict[str, Any]:
    plan = _normalize_plan_payload(candidate.get("plan"))
    tool_graph = _stringify_sequence(candidate.get("tool_graph") or plan.get("tool_sequence") or [])
    predicate_coverage = 1.0 if case.expected_predicate_count == 0 else min(1.0, float(plan.get("predicate_count") or len(plan.get("filters") or [])) / case.expected_predicate_count)
    logical_structure_accuracy = 1.0 if str(plan.get("logical_structure") or candidate.get("logical_structure") or "SINGLE") == case.expected_logical_structure else 0.0
    semantic_roles = set(plan.get("semantic_roles") or candidate.get("semantic_roles") or [])
    expected_roles = set(case.expected_semantic_roles)
    semantic_role_coverage = 1.0 if not expected_roles else len(semantic_roles & expected_roles) / len(expected_roles)
    predicted_tools = set(tool_graph)
    expected_tools = set(case.expected_tool_sequence)
    true_positive = len(predicted_tools & expected_tools)
    precision = true_positive / len(predicted_tools) if predicted_tools else 0.0
    recall = true_positive / len(expected_tools) if expected_tools else 1.0
    tool_selection_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tool_sequence_accuracy = 1.0 if list(tool_graph[: len(case.expected_tool_sequence)]) == case.expected_tool_sequence[: len(tool_graph)] and tool_graph == case.expected_tool_sequence else 0.0
    invalid_tool = any(not get_tool_registry().is_allowed(tool) for tool in tool_graph)
    return {
        "schema_valid": isinstance(plan, dict),
        "critic_passed": critic_passed,
        "tool_valid": not invalid_tool,
        "predicate_coverage": predicate_coverage,
        "logical_structure_accuracy": logical_structure_accuracy,
        "semantic_role_coverage": semantic_role_coverage,
        "tool_selection_f1": tool_selection_f1,
        "tool_sequence_accuracy": tool_sequence_accuracy,
        "invalid_tool": invalid_tool,
    }


def _best_effort_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        return round(torch.cuda.max_memory_allocated() / 1024**2, 3)
    except Exception:
        return None


def audit_failure_modes(summary: PlannerBenchmarkSummary) -> PlannerFailureModeReport:
    counts: dict[str, int] = {
        "wrong_tool": 0,
        "missing_tool": 0,
        "wrong_tool_order": 0,
        "missing_predicate": 0,
        "wrong_semantic_role_binding": 0,
        "wrong_aggregation": 0,
        "incomplete_multi_step_plan": 0,
        "correct_intent_wrong_executable_plan": 0,
        "unsupported_or_fallback": 0,
    }
    for case, result in zip(builtin_benchmark_cases(), summary.cases):
        predicted = result.parsed_plan or {}
        predicted_tools = _stringify_sequence(predicted.get("tool_sequence") or predicted.get("tool_graph") or [])
        expected_tools = list(case.expected_tool_sequence)
        if result.plan_source in {"fallback", "deterministic_fallback"}:
            counts["unsupported_or_fallback"] += 1
        if not predicted_tools:
            counts["missing_tool"] += 1
        elif predicted_tools != expected_tools:
            counts["wrong_tool_order"] += 1
        if set(predicted_tools) != set(expected_tools):
            counts["wrong_tool"] += 1
        if result.predicate_coverage < 1.0:
            counts["missing_predicate"] += 1
        if result.semantic_role_coverage < 1.0:
            counts["wrong_semantic_role_binding"] += 1
        if case.expected_tool_sequence and case.expected_tool_sequence[0] == "sql.group_by" and not predicted.get("group_by"):
            counts["wrong_aggregation"] += 1
        if len(case.expected_tool_sequence) > 1 and len(predicted_tools) < len(case.expected_tool_sequence):
            counts["incomplete_multi_step_plan"] += 1
        if result.intent_correct and not result.tool_valid:
            counts["correct_intent_wrong_executable_plan"] += 1
    return PlannerFailureModeReport(failure_counts=counts, total_cases=len(summary.cases))


def run_planner_benchmark(
    *,
    profile_name: str = "low_spec",
    backend: str = PLANNER_BACKEND_AUTO,
    device: str = "auto",
    benchmark: str = "builtin",
    cache_dir: Path | None = None,
    model_path: Path | None = None,
) -> PlannerBenchmarkSummary:
    profile = select_model_profile(profile_name)
    runtime = select_runtime_profile("cpu_low_spec" if device == "cpu" else "gpu_4gb")
    hw = detect_hardware()
    allowed_backend = choose_backend(
        backend=backend,
        runtime_profile=runtime,
        cuda_available=bool(hw.cuda_available),
        llama_cpp_available=False,
    )
    model: PlannerModel = HeuristicPlannerModel(profile_name=profile_name)
    model_health = model.health()
    notes: list[str] = []

    if allowed_backend == PLANNER_BACKEND_TRANSFORMERS:
        candidate = TransformersPlannerModel(profile_name=profile_name, device=device, cache_dir=cache_dir)
        candidate.load()
        if candidate.model is not None and candidate.tokenizer is not None:
            model = candidate
            model_health = candidate.health()
        else:
            notes.append(f"transformers_load_failed:{candidate.load_error or 'unknown'}")
    elif allowed_backend == PLANNER_BACKEND_LLAMA_CPP:
        candidate = LlamaCppPlannerAdapter(model_path=model_path, profile_name=profile_name)
        if candidate.health()["available"]:
            model = candidate
            model_health = candidate.health()
        else:
            notes.append("llama_cpp_unavailable")

    cases = builtin_benchmark_cases() if benchmark == "builtin" else builtin_benchmark_cases()
    results: list[PlannerInferenceResult] = []
    synthetic_df = pd.DataFrame(
        {
            "restaurant_name": ["Alpha", "Beta", "Gamma"],
            "delivery": [True, False, True],
            "booking": [False, True, True],
            "rating": [4.8, 3.9, 4.5],
            "revenue": [1000, 850, 1250],
            "region": ["north", "south", "east"],
            "category": ["casual", "fine", "casual"],
            "customer_name": ["A", "B", "C"],
            "missing_value": [1, None, 3],
        }
    )

    for case in cases:
        request = {
            "text": case.text,
            "intent": case.intent,
            "query_features": {
                "predicate_count": case.expected_predicate_count,
                "logical_structure": case.expected_logical_structure,
                "semantic_roles": list(case.expected_semantic_roles),
                "operators": [],
            },
            "dataset_profile": {
                "fields": [
                    {"id": "restaurant_name", "semantic_role": "entity_name", "dtype": "string"},
                    {"id": "delivery", "semantic_role": "boolean_capability", "dtype": "bool"},
                    {"id": "booking", "semantic_role": "boolean_capability", "dtype": "bool"},
                    {"id": "rating", "semantic_role": "rating_metric", "dtype": "number"},
                    {"id": "revenue", "semantic_role": "numeric_metric", "dtype": "number"},
                    {"id": "region", "semantic_role": "geographic_area", "dtype": "string"},
                    {"id": "category", "semantic_role": "category", "dtype": "string"},
                    {"id": "customer_name", "semantic_role": "customer_entity", "dtype": "string"},
                    {"id": "missing_value", "semantic_role": "numeric_metric", "dtype": "number"},
                ]
            },
            "dataframe": synthetic_df,
        }
        started = time.perf_counter()
        raw_output = ""
        parsed_plan: dict[str, Any] | None = None
        plan_source = "unknown"
        critic_passed = False
        load_ms = None
        inference_ms = None
        peak_vram_mb = None
        try:
            candidate = model.plan(request)
            raw_output = json.dumps(candidate, sort_keys=True)
            parsed_plan = _normalize_plan_payload(candidate.get("plan")) or _extract_json_object(raw_output)
            parsed_plan = _normalize_plan_payload(parsed_plan)
            plan_source = str(candidate.get("plan_source") or "unknown")
            critic_passed = True
            load_ms = candidate.get("model_load_ms") if isinstance(candidate, dict) else None
            inference_ms = candidate.get("inference_ms") if isinstance(candidate, dict) else None
            peak_vram_mb = candidate.get("peak_vram_mb") if isinstance(candidate, dict) else None
        except Exception as exc:
            raw_output = json.dumps({"error": str(exc)}, sort_keys=True)
            parsed_plan = None
            plan_source = "fallback"

        critic = False
        if parsed_plan is not None:
            candidate_tool_graph = _stringify_sequence(candidate.get("tool_graph") if isinstance(candidate, dict) else [])
            plan_tool_sequence = _stringify_sequence(parsed_plan.get("tool_sequence") or candidate_tool_graph)
            decision = LearningDecision(
                route="sql" if case.intent in {"filter", "analytics"} else "operation",
                confidence=0.8,
                message="benchmark",
                plan=parsed_plan,
                tool_sequence=plan_tool_sequence,
                features=request["query_features"],
            )
            critic, _ = PlanCritic().review(decision, context=build_planner_context(case.text, None, []))
        scores = _score_plan(
            candidate if isinstance(candidate, dict) else {"plan": parsed_plan, "tool_graph": []},
            case,
            critic_passed=critic and critic_passed,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        if peak_vram_mb is None:
            peak_vram_mb = _best_effort_vram_mb()
        results.append(
            PlannerInferenceResult(
                case_name=case.name,
                raw_output=raw_output,
                parsed_plan=parsed_plan,
                plan_source=plan_source,
                schema_valid=scores["schema_valid"],
                critic_passed=bool(critic and critic_passed),
                tool_valid=scores["tool_valid"],
                predicate_coverage=scores["predicate_coverage"],
                logical_structure_accuracy=scores["logical_structure_accuracy"],
                semantic_role_coverage=scores["semantic_role_coverage"],
                intent_correct=True,
                tool_selection_f1=scores["tool_selection_f1"],
                tool_sequence_accuracy=scores["tool_sequence_accuracy"],
                invalid_tool=scores["invalid_tool"],
                latency_ms=round(latency_ms, 3),
                model_load_ms=load_ms,
                inference_ms=inference_ms,
                peak_vram_mb=peak_vram_mb,
                valid_json=_extract_json_object(raw_output) is not None,
                fallback_triggered=plan_source in {"fallback", "deterministic_fallback"},
            )
        )
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    valid_json_rate = sum(1 for case in results if case.raw_output and _extract_json_object(case.raw_output) is not None) / len(results)
    schema_valid_rate = sum(1 for case in results if case.schema_valid) / len(results)
    plan_validity_rate = sum(1 for case in results if case.critic_passed and case.tool_valid) / len(results)
    intent_accuracy = sum(1 for case in results if case.intent_correct) / len(results)
    tool_selection_f1 = sum(case.tool_selection_f1 for case in results) / len(results)
    tool_sequence_accuracy = sum(case.tool_sequence_accuracy for case in results) / len(results)
    predicate_coverage = sum(case.predicate_coverage for case in results) / len(results)
    logical_structure_accuracy = sum(case.logical_structure_accuracy for case in results) / len(results)
    semantic_role_coverage = sum(case.semantic_role_coverage for case in results) / len(results)
    invalid_tool_rate = sum(1 for case in results if case.invalid_tool) / len(results)
    fallback_accuracy = sum(1 for case in results if case.plan_source != "unknown") / len(results)
    median_latency = sorted(case.latency_ms for case in results)[len(results) // 2]
    p95_latency = sorted(case.latency_ms for case in results)[max(0, int(len(results) * 0.95) - 1)]
    model_load_ms = next((case.model_load_ms for case in results if case.model_load_ms is not None), None)
    inference_ms = sorted(case.inference_ms for case in results if case.inference_ms is not None)
    peak_vram_mb = max((case.peak_vram_mb or 0.0) for case in results) if results else None

    return PlannerBenchmarkSummary(
        profile=profile.to_dict(),
        backend=allowed_backend,
        device=device,
        model_id=getattr(model, "metadata", lambda: {"model_id": None})().get("model_id"),
        cases=results,
        metrics={
            "valid_json_rate": valid_json_rate,
            "schema_valid_rate": schema_valid_rate,
            "plan_validity_rate": plan_validity_rate,
            "intent_accuracy": intent_accuracy,
            "tool_selection_f1": tool_selection_f1,
            "tool_sequence_accuracy": tool_sequence_accuracy,
            "predicate_coverage": predicate_coverage,
            "logical_structure_accuracy": logical_structure_accuracy,
            "semantic_role_coverage": semantic_role_coverage,
            "invalid_tool_rate": invalid_tool_rate,
            "fallback_accuracy": fallback_accuracy,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "model_load_ms": model_load_ms,
            "peak_vram_mb": peak_vram_mb,
            "shadow_captured": True,
            "inference_ms_median": inference_ms[len(inference_ms) // 2] if inference_ms else None,
        },
        hardware=hw.to_dict(),
        model_health=model_health,
        notes=notes,
    )


def write_benchmark_report(summary: PlannerBenchmarkSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "low_spec_inference_benchmark.json"
    payload = summary.to_dict()
    payload["failure_modes"] = audit_failure_modes(summary).to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
