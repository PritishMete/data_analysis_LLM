from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import re
from typing import Any

from learning.canonical_training import (
    PlannerTrainingBackend,
    TrainingCandidateInvalidation,
    TrainingDatasetManifest,
    TrainingReadinessAssessment,
    build_training_dataset_manifest,
)
from learning.experience_store import LearningExperienceStore
from learning.models import stable_hash


SAFE_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")
SAFE_HEX_RE = re.compile(r"^[a-f0-9]{16,}$")
SAFE_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}$")

ALLOWED_LOGICAL_STRUCTURES = {"SINGLE", "AND", "OR", "MIXED"}
ALLOWED_SPLITS = ("train", "validation", "test")
ALLOWED_SOURCE_KINDS = {"experience", "strategy"}
ALLOWED_CANDIDATE_STATES = {"validated", "trusted", "promoted"}
ALLOWED_CANDIDATE_LIFECYCLES = {"validated", "trusted"}
ALLOWED_PLAN_SOURCES = {
    "bootstrap_skill",
    "deterministic_fallback",
    "experience_transfer",
    "validated_template",
    "teacher_execution",
    "gemini",
    "gemini_validated",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(0, int(raw))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _safe_token(value: str) -> bool:
    return bool(SAFE_TOKEN_RE.fullmatch(value or ""))


def _safe_string(value: str, *, allow_uppercase_enum: set[str] | None = None) -> bool:
    if allow_uppercase_enum and value in allow_uppercase_enum:
        return True
    if value in {"", "AND", "OR", "MIXED"}:
        return True
    return _safe_token(value)


def _shape_signature(value: Any) -> Any:
    if isinstance(value, dict):
        items = []
        for key in sorted(value.keys(), key=lambda item: str(item)):
            items.append((str(key), _shape_signature(value[key])))
        return {"type": "dict", "keys": items}
    if isinstance(value, list):
        return {"type": "list", "items": [_shape_signature(item) for item in value[:20]]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_shape_signature(item) for item in value[:20]]}
    if isinstance(value, set):
        return {"type": "set", "items": sorted(_shape_signature(item) for item in list(value)[:20])}
    if isinstance(value, bool):
        return {"type": "bool"}
    if isinstance(value, int):
        return {"type": "int"}
    if isinstance(value, float):
        return {"type": "float"}
    if value is None:
        return {"type": "null"}
    return {"type": type(value).__name__}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _normalise_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _normalise_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _normalise_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(slots=True)
class TrainingExportPolicy:
    minimum_quality: float = 0.95
    minimum_readiness_quality: float = 0.96
    minimum_eligible_examples: int = 500
    minimum_family_count: int = 50
    minimum_intent_count: int = 10
    max_single_intent_share: float = 0.40
    max_single_tool_graph_share: float = 0.40
    require_execution_success: bool = True
    require_critic_pass: bool = True
    require_result_validation: bool = True
    require_plan_completeness: bool = True
    require_privacy_pass: bool = True
    require_no_unresolved_ambiguity: bool = True
    require_no_critical_repair: bool = True
    allow_repaired_examples: bool = False
    max_examples_per_fingerprint: int = 1
    max_examples_per_intent: int = 100
    max_examples_per_tool_graph: int = 100
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    preview_limit: int = 25
    export_limit: int = 10_000
    allow_candidate_lifecycles: tuple[str, ...] = ("validated", "trusted")

    @classmethod
    def from_env(cls) -> "TrainingExportPolicy":
        policy = cls(
            minimum_quality=_env_float("INSIGHT_LEARNING_EXPORT_MIN_QUALITY", 0.95),
            minimum_readiness_quality=_env_float("INSIGHT_LEARNING_EXPORT_MIN_READINESS_QUALITY", 0.96),
            minimum_eligible_examples=max(1, _env_int("INSIGHT_LEARNING_EXPORT_MIN_ELIGIBLE_EXAMPLES", 500)),
            minimum_family_count=max(1, _env_int("INSIGHT_LEARNING_EXPORT_MIN_FAMILY_COUNT", 50)),
            minimum_intent_count=max(1, _env_int("INSIGHT_LEARNING_EXPORT_MIN_INTENT_COUNT", 10)),
            max_single_intent_share=_env_float("INSIGHT_LEARNING_EXPORT_MAX_SINGLE_INTENT_SHARE", 0.40),
            max_single_tool_graph_share=_env_float("INSIGHT_LEARNING_EXPORT_MAX_SINGLE_TOOL_GRAPH_SHARE", 0.40),
            require_execution_success=_env_bool("INSIGHT_LEARNING_EXPORT_REQUIRE_EXECUTION_SUCCESS", True),
            require_critic_pass=_env_bool("INSIGHT_LEARNING_EXPORT_REQUIRE_CRITIC_PASS", True),
            require_result_validation=_env_bool("INSIGHT_LEARNING_EXPORT_REQUIRE_RESULT_VALIDATION", True),
            require_plan_completeness=_env_bool("INSIGHT_LEARNING_EXPORT_REQUIRE_PLAN_COMPLETENESS", True),
            require_privacy_pass=_env_bool("INSIGHT_LEARNING_EXPORT_REQUIRE_PRIVACY_PASS", True),
            require_no_unresolved_ambiguity=_env_bool("INSIGHT_LEARNING_EXPORT_REQUIRE_NO_AMBIGUITY", True),
            require_no_critical_repair=_env_bool("INSIGHT_LEARNING_EXPORT_REQUIRE_NO_CRITICAL_REPAIR", True),
            allow_repaired_examples=_env_bool("INSIGHT_LEARNING_EXPORT_ALLOW_REPAIRED", False),
            max_examples_per_fingerprint=max(1, _env_int("INSIGHT_LEARNING_EXPORT_MAX_PER_FINGERPRINT", 1)),
            max_examples_per_intent=max(1, _env_int("INSIGHT_LEARNING_EXPORT_MAX_PER_INTENT", 100)),
            max_examples_per_tool_graph=max(1, _env_int("INSIGHT_LEARNING_EXPORT_MAX_PER_TOOL_GRAPH", 100)),
            train_ratio=_env_float("INSIGHT_LEARNING_EXPORT_TRAIN_RATIO", 0.8),
            validation_ratio=_env_float("INSIGHT_LEARNING_EXPORT_VALIDATION_RATIO", 0.1),
            test_ratio=_env_float("INSIGHT_LEARNING_EXPORT_TEST_RATIO", 0.1),
            preview_limit=max(1, _env_int("INSIGHT_LEARNING_EXPORT_PREVIEW_LIMIT", 25)),
            export_limit=max(1, _env_int("INSIGHT_LEARNING_EXPORT_LIMIT", 10_000)),
        )
        total = policy.train_ratio + policy.validation_ratio + policy.test_ratio
        if total <= 0:
            return cls()
        policy.train_ratio /= total
        policy.validation_ratio /= total
        policy.test_ratio /= total
        return policy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingExportRecord:
    source_kind: str
    source_id: str
    intent: str
    semantic_roles: list[str]
    operators: list[str]
    logical_structure: str
    tool_graph: list[str]
    predicate_graph: dict[str, Any]
    plan_source: str | None
    plan_template_id: str | None
    quality_score: float
    execution_success: bool | None
    critic_passed: bool | None
    result_validation_passed: bool | None
    plan_completeness_passed: bool | None
    privacy_validation_passed: bool | None
    no_unresolved_ambiguity: bool | None
    no_critical_repair: bool | None
    repair_count: int | None
    correction_state: str | None
    candidate_state: str | None
    candidate_evidence_count: int | None
    candidate_average_quality: float | None
    dataset_semantic_signature: str | None
    family_fingerprint: str
    split: str = ""
    family_size: int = 1
    created_at: str | None = None
    plan_shape: dict[str, Any] = field(default_factory=dict)

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "split": self.split,
            "family_fingerprint": self.family_fingerprint,
            "input": {
                "intent": self.intent,
                "semantic_roles": list(self.semantic_roles),
                "operators": list(self.operators),
                "logical_structure": self.logical_structure,
                "predicate_graph": dict(self.predicate_graph),
            },
            "output": {
                "tool_graph": list(self.tool_graph),
                "plan_source": self.plan_source,
                "plan_template_id": self.plan_template_id,
                "source_kind": self.source_kind,
                "candidate_state": self.candidate_state,
            },
            "metadata": {
                "quality": self.quality_score,
                "execution_success": self.execution_success,
                "critic_passed": self.critic_passed,
                "result_validation_passed": self.result_validation_passed,
                "plan_completeness_passed": self.plan_completeness_passed,
                "privacy_validation_passed": self.privacy_validation_passed,
                "no_unresolved_ambiguity": self.no_unresolved_ambiguity,
                "no_critical_repair": self.no_critical_repair,
                "repair_count": self.repair_count,
                "correction_state": self.correction_state,
                "candidate_state": self.candidate_state,
                "candidate_evidence_count": self.candidate_evidence_count,
                "candidate_average_quality": self.candidate_average_quality,
                "dataset_semantic_signature": self.dataset_semantic_signature,
                "family_fingerprint": self.family_fingerprint,
                "split": self.split,
                "family_size": self.family_size,
                "created_at": self.created_at,
                "plan_shape": self.plan_shape,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_export_dict()


@dataclass(slots=True)
class TrainingExportBundle:
    records: list[TrainingExportRecord]
    rejected_count: int
    rejected_reasons: Counter[str]
    duplicates_removed: int
    inspected_count: int
    policy: TrainingExportPolicy
    source_distribution: Counter[str]
    intent_distribution: Counter[str]
    tool_graph_distribution: Counter[str]
    step_distribution: Counter[int]
    predicate_complexity_distribution: Counter[int]
    average_quality: float
    dataset_version: str = ""

    def report(self) -> dict[str, Any]:
        split_counts = Counter(record.split for record in self.records)
        eligible = len(self.records)
        intent_distribution = dict(sorted(self.intent_distribution.items()))
        tool_distribution = dict(sorted(self.tool_graph_distribution.items()))
        step_distribution = dict(sorted(self.step_distribution.items()))
        family_count = len({record.family_fingerprint for record in self.records})
        intent_count = len(intent_distribution)
        tool_graph_count = len(tool_distribution)
        largest_intent_share = (max(intent_distribution.values()) / eligible) if eligible and intent_distribution else 0.0
        largest_tool_graph_share = (max(tool_distribution.values()) / eligible) if eligible and tool_distribution else 0.0
        average_steps = (
            sum(step * count for step, count in self.step_distribution.items()) / eligible
            if eligible
            else 0.0
        )
        single_step_pct = round((self.step_distribution.get(1, 0) / eligible) * 100, 2) if eligible else 0.0
        multi_step_pct = round(100.0 - single_step_pct, 2) if eligible else 0.0
        privacy_rejections = sum(
            count
            for reason, count in self.rejected_reasons.items()
            if reason.startswith("unsafe_") or "privacy" in reason
        )
        invalidated_excluded = int(self.rejected_reasons.get("invalidated", 0))
        balance_warnings: list[str] = []
        if eligible and largest_intent_share > self.policy.max_single_intent_share:
            balance_warnings.append(f"largest_intent_share={largest_intent_share:.3f}")
        if eligible and largest_tool_graph_share > self.policy.max_single_tool_graph_share:
            balance_warnings.append(f"largest_tool_graph_share={largest_tool_graph_share:.3f}")
        return {
            "total_experiences_inspected": self.inspected_count,
            "eligible_examples": eligible,
            "rejected_examples": self.rejected_count,
            "rejection_reasons": dict(sorted(self.rejected_reasons.items())),
            "duplicates_removed": self.duplicates_removed,
            "invalidated_excluded": invalidated_excluded,
            "privacy_rejections": privacy_rejections,
            "family_count": family_count,
            "intent_count": intent_count,
            "tool_graph_count": tool_graph_count,
            "largest_intent_share": round(largest_intent_share, 4),
            "largest_tool_graph_share": round(largest_tool_graph_share, 4),
            "average_tool_steps": round(average_steps, 4),
            "single_step_pct": single_step_pct,
            "multi_step_pct": multi_step_pct,
            "balance_warnings": balance_warnings,
            "intent_distribution": intent_distribution,
            "tool_graph_distribution": tool_distribution,
            "number_of_steps_distribution": step_distribution,
            "predicate_complexity_distribution": dict(sorted(self.predicate_complexity_distribution.items())),
            "plan_source_distribution": dict(sorted(self.source_distribution.items())),
            "average_quality": round(self.average_quality, 4) if self.records else 0.0,
            "dataset_version": self.dataset_version,
            "train_count": int(split_counts.get("train", 0)),
            "validation_count": int(split_counts.get("validation", 0)),
            "test_count": int(split_counts.get("test", 0)),
            "policy": self.policy.to_dict(),
        }

    def to_preview(self, limit: int) -> list[dict[str, Any]]:
        return [record.to_export_dict() for record in self.records[:limit]]


class TrainingDatasetExporter:
    def __init__(self, store: LearningExperienceStore | None = None, policy: TrainingExportPolicy | None = None):
        self.store = store or LearningExperienceStore()
        self.policy = policy or TrainingExportPolicy.from_env()
        self.backend = PlannerTrainingBackend()

    def _query_features_shape(self, features: dict[str, Any]) -> dict[str, Any]:
        return {
            "predicate_count": int(features.get("predicate_count") or 0),
            "logical_structure": str(features.get("logical_structure") or "SINGLE"),
            "semantic_roles": list(features.get("semantic_roles") or []),
            "operators": list(features.get("operators") or []),
            "predicate_graph": _shape_signature(features.get("predicate_graph") or []),
        }

    def _predicate_graph(self, features: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        predicate_graph = features.get("predicate_graph")
        if isinstance(predicate_graph, list) and predicate_graph:
            return {
                "operator": str(features.get("logical_structure") or "SINGLE"),
                "shape": _shape_signature(predicate_graph),
                "predicate_count": len(predicate_graph),
            }
        filters = list(plan.get("filters") or [])
        return {
            "operator": str(features.get("logical_structure") or "SINGLE"),
            "shape": _shape_signature(filters),
            "predicate_count": len(filters) or int(features.get("predicate_count") or 0),
        }

    def _plan_shape(self, plan: dict[str, Any]) -> dict[str, Any]:
        return _shape_signature({
            "keys": sorted(plan.keys()),
            "tool_sequence": list(plan.get("tool_sequence") or []),
            "group_by_count": len(plan.get("group_by") or []),
            "metric_count": len(plan.get("metrics") or []),
            "filter_count": len(plan.get("filters") or []),
            "window_present": bool(plan.get("window")),
            "keep_top_n_per_partition": bool(plan.get("keep_top_n_per_partition") is not None),
            "derived_columns_count": len(plan.get("derived_columns") or []),
            "action": plan.get("action"),
        })

    def _safe_record_from_experience(self, item: dict[str, Any]) -> tuple[TrainingExportRecord | None, str | None]:
        intent = str(item.get("intent") or "unknown")
        query_features = dict(item.get("query_features") or {})
        semantic_roles = [str(role) for role in (item.get("semantic_roles") or query_features.get("semantic_roles") or [])]
        operators = [str(op) for op in (item.get("operators") or query_features.get("operators") or [])]
        logical_structure = str(item.get("logical_structure") or query_features.get("logical_structure") or "SINGLE")
        if logical_structure not in ALLOWED_LOGICAL_STRUCTURES:
            return None, "invalid_logical_structure"
        tool_graph = [str(tool) for tool in (item.get("tool_sequence") or [])]
        plan = dict(item.get("plan_provenance", {}).get("template") or {})
        if not plan:
            plan = dict(item.get("plan_summary") or {})
        if not plan and tool_graph:
            plan = {"tool_sequence": list(tool_graph)}

        execution_success = _normalise_bool(item.get("success"))
        critic_passed = _normalise_bool(item.get("critic_passed"))
        result_validation_passed = _normalise_bool(item.get("result_validation_passed"))
        plan_completeness_passed = _normalise_bool(item.get("plan_completeness_passed"))
        privacy_validation_passed = _normalise_bool(item.get("privacy_validation_passed"))
        no_unresolved_ambiguity = _normalise_bool(item.get("no_unresolved_ambiguity"))
        no_critical_repair = _normalise_bool(item.get("no_critical_repair"))
        repair_count = _normalise_int(item.get("repair_count"))
        correction_state = item.get("correction_state")
        if correction_state is not None:
            correction_state = str(correction_state)

        quality = _normalise_float(item.get("score"))
        if quality is None:
            return None, "missing_quality"

        if item.get("event_id") and not _safe_token(str(item["event_id"])):
            return None, "unsafe_event_id"
        if not _safe_token(intent):
            return None, "unsafe_intent"
        if any(not _safe_token(role) for role in semantic_roles):
            return None, "unsafe_semantic_role"
        if any(not _safe_token(op) for op in operators):
            return None, "unsafe_operator"
        if any(not _safe_token(tool) for tool in tool_graph):
            return None, "unsafe_tool_graph"
        if logical_structure not in ALLOWED_LOGICAL_STRUCTURES:
            return None, "unsafe_logical_structure"
        plan_source = item.get("plan_source")
        if plan_source is not None and not _safe_token(str(plan_source)):
            return None, "unsafe_plan_source"
        plan_template_id = item.get("plan_template_id")
        if plan_template_id is not None and not _safe_token(str(plan_template_id)):
            return None, "unsafe_plan_template_id"
        dataset_signature = item.get("dataset_semantic_signature")
        if dataset_signature is not None and not SAFE_HEX_RE.fullmatch(str(dataset_signature)):
            return None, "unsafe_dataset_signature"

        if self.policy.require_execution_success and execution_success is not True:
            return None, "execution_failed"
        if self.policy.require_critic_pass and critic_passed is not True:
            return None, "critic_failed"
        if self.policy.require_result_validation and result_validation_passed is not True:
            return None, "result_validation_failed"
        if self.policy.require_plan_completeness and plan_completeness_passed is not True:
            return None, "plan_completeness_failed"
        if self.policy.require_privacy_pass and privacy_validation_passed is not True:
            return None, "privacy_failed"
        if self.policy.require_no_unresolved_ambiguity and no_unresolved_ambiguity is not True:
            return None, "ambiguity_not_resolved"
        if self.policy.require_no_critical_repair and no_critical_repair is not True:
            return None, "critical_repair_present"
        if not self.policy.allow_repaired_examples and (repair_count or 0) > 0:
            return None, "repaired_example_rejected"
        if quality < self.policy.minimum_quality:
            return None, "quality_below_threshold"

        predicate_graph = self._predicate_graph(query_features, plan)
        fingerprint_payload = {
            "source_kind": "experience",
            "intent": intent,
            "semantic_roles": semantic_roles,
            "operators": operators,
            "logical_structure": logical_structure,
            "tool_graph": tool_graph,
            "predicate_graph": _shape_signature(predicate_graph),
            "plan_shape": self._plan_shape(plan),
        }
        family_fingerprint = stable_hash(fingerprint_payload)
        return (
            TrainingExportRecord(
                source_kind="experience",
                source_id=str(item.get("event_id") or item.get("semantic_signature") or family_fingerprint[:16]),
                intent=intent,
                semantic_roles=semantic_roles,
                operators=operators,
                logical_structure=logical_structure,
                tool_graph=tool_graph,
                predicate_graph=predicate_graph,
                plan_source=str(plan_source or "teacher_execution"),
                plan_template_id=str(plan_template_id) if plan_template_id is not None else None,
                quality_score=quality,
                execution_success=execution_success,
                critic_passed=critic_passed,
                result_validation_passed=result_validation_passed,
                plan_completeness_passed=plan_completeness_passed,
                privacy_validation_passed=privacy_validation_passed,
                no_unresolved_ambiguity=no_unresolved_ambiguity,
                no_critical_repair=no_critical_repair,
                repair_count=repair_count,
                correction_state=correction_state,
                candidate_state=None,
                candidate_evidence_count=None,
                candidate_average_quality=None,
                dataset_semantic_signature=str(dataset_signature) if dataset_signature is not None else None,
                family_fingerprint=family_fingerprint,
                created_at=str(item.get("created_at") or ""),
                plan_shape=self._plan_shape(plan),
            ),
            None,
        )

    def _safe_record_from_strategy(self, item: dict[str, Any]) -> tuple[TrainingExportRecord | None, str | None]:
        state = str(item.get("state") or "candidate")
        lifecycle = str(item.get("lifecycle_state") or "observed")
        allowed_lifecycles = set(self.policy.allow_candidate_lifecycles or ALLOWED_CANDIDATE_LIFECYCLES)
        if state not in ALLOWED_CANDIDATE_STATES or lifecycle not in allowed_lifecycles:
            return None, "strategy_lifecycle_not_trusted"
        evidence_count = _normalise_int(item.get("evidence_count")) or 0
        average_quality = _normalise_float(item.get("average_quality")) or 0.0
        if average_quality < self.policy.minimum_quality:
            return None, "strategy_quality_below_threshold"
        if evidence_count <= 0:
            return None, "strategy_has_no_evidence"

        intent = str(item.get("intent") or "unknown")
        semantic_roles = [str(role) for role in (item.get("semantic_roles") or [])]
        tool_graph = [str(tool) for tool in (item.get("tool_sequence") or [])]
        if not _safe_token(intent):
            return None, "unsafe_intent"
        if any(not _safe_token(role) for role in semantic_roles):
            return None, "unsafe_semantic_role"
        if any(not _safe_token(tool) for tool in tool_graph):
            return None, "unsafe_tool_graph"

        template = dict(item.get("plan_template") or {})
        predicate_graph = {
            "operator": "SINGLE",
            "shape": _shape_signature(template),
            "predicate_count": len(template.get("predicate_structure") or []),
        }
        plan_template_id = item.get("plan_template_id")
        if plan_template_id is not None and not _safe_token(str(plan_template_id)):
            return None, "unsafe_plan_template_id"
        family_fingerprint = stable_hash(
            {
                "source_kind": "strategy",
                "intent": intent,
                "semantic_roles": semantic_roles,
                "tool_graph": tool_graph,
                "logical_structure": str(item.get("logical_structure") or "SINGLE"),
                "plan_template_id": plan_template_id,
                "plan_template_shape": _shape_signature(template),
            }
        )
        return (
            TrainingExportRecord(
                source_kind="strategy",
                source_id=str(item.get("strategy_id") or family_fingerprint[:16]),
                intent=intent,
                semantic_roles=semantic_roles,
                operators=[],
                logical_structure=str(item.get("logical_structure") or "SINGLE"),
                tool_graph=tool_graph,
                predicate_graph=predicate_graph,
                plan_source=state,
                plan_template_id=str(plan_template_id) if plan_template_id is not None else None,
                quality_score=average_quality,
                execution_success=None,
                critic_passed=None,
                result_validation_passed=None,
                plan_completeness_passed=None,
                privacy_validation_passed=None,
                no_unresolved_ambiguity=None,
                no_critical_repair=None,
                repair_count=None,
                correction_state=None,
                candidate_state=state,
                candidate_evidence_count=evidence_count,
                candidate_average_quality=average_quality,
                dataset_semantic_signature=str(item.get("semantic_signature") or "") or None,
                family_fingerprint=family_fingerprint,
                created_at=str(item.get("created_at") or ""),
                plan_shape=self._plan_shape(template),
            ),
            None,
        )

    def _stable_split(self, family_fingerprint: str) -> str:
        raw = int(hashlib.sha256(family_fingerprint.encode("utf-8")).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)
        train_cutoff = self.policy.train_ratio
        validation_cutoff = train_cutoff + self.policy.validation_ratio
        if raw < train_cutoff:
            return "train"
        if raw < validation_cutoff:
            return "validation"
        return "test"

    def _recursively_validate_record(self, record: dict[str, Any]) -> tuple[bool, str | None]:
        def scan(value: Any, path: str = "record") -> str | None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key not in {"input", "output", "metadata"} and path == "record":
                        return f"unexpected_top_level_key:{key}"
                    if path == "record" and key in {"input", "output", "metadata"}:
                        continue
                    if isinstance(key, str) and not _safe_token(key):
                        return f"unsafe_key:{path}.{key}"
                    reason = scan(item, f"{path}.{key}")
                    if reason:
                        return reason
                return None
            if isinstance(value, list):
                for index, item in enumerate(value):
                    reason = scan(item, f"{path}[{index}]")
                    if reason:
                        return reason
                return None
            if isinstance(value, str):
                if value in ALLOWED_LOGICAL_STRUCTURES or value in ALLOWED_SPLITS or value in ALLOWED_SOURCE_KINDS or value in ALLOWED_CANDIDATE_STATES or value in ALLOWED_CANDIDATE_LIFECYCLES or value in ALLOWED_PLAN_SOURCES:
                    return None
                if _safe_token(value) or SAFE_HEX_RE.fullmatch(value) or SAFE_TIMESTAMP_RE.fullmatch(value):
                    return None
                return f"unsafe_string:{path}"
            if isinstance(value, (bool, int, float)) or value is None:
                return None
            return f"unsafe_type:{path}:{type(value).__name__}"

        return (scan(record) is None, scan(record))

    def build_bundle(
        self,
        *,
        limit: int | None = None,
        include_candidate_strategies: bool = True,
    ) -> tuple[TrainingExportBundle, list[dict[str, Any]]]:
        inspected = 0
        rejected_reasons: Counter[str] = Counter()
        dedupe_removed = 0
        candidates: list[TrainingExportRecord] = []
        invalidations = self.store.load_training_invalidations(limit=self.policy.export_limit)
        invalidated_source_ids = {
            str(item.get("source_id") or "")
            for item in invalidations
            if str(item.get("source_id") or "")
        }
        invalidated_fingerprints = {
            str(item.get("family_fingerprint") or "")
            for item in invalidations
            if str(item.get("family_fingerprint") or "")
        }

        source_items: list[tuple[str, dict[str, Any]]] = []
        for item in self.store.load_recent(limit=self.policy.export_limit):
            source_items.append(("experience", item))
        if include_candidate_strategies:
            for item in self.store.load_candidate_strategies(limit=self.policy.export_limit):
                source_items.append(("strategy", item))

        for source_kind, item in source_items:
            inspected += 1
            if source_kind == "experience":
                record, rejection = self._safe_record_from_experience(item)
            else:
                record, rejection = self._safe_record_from_strategy(item)
            if record is None:
                rejected_reasons[rejection or "rejected"] += 1
                continue
            if record.source_id in invalidated_source_ids or record.family_fingerprint in invalidated_fingerprints:
                rejected_reasons["invalidated"] += 1
                continue
            candidates.append(record)

        if not candidates:
            bundle = TrainingExportBundle(
                records=[],
                rejected_count=sum(rejected_reasons.values()),
                rejected_reasons=rejected_reasons,
                duplicates_removed=dedupe_removed,
                inspected_count=inspected,
                policy=self.policy,
                source_distribution=Counter(),
                intent_distribution=Counter(),
                tool_graph_distribution=Counter(),
                step_distribution=Counter(),
                predicate_complexity_distribution=Counter(),
                average_quality=0.0,
                dataset_version="",
            )
            return bundle, []

        grouped: dict[str, list[TrainingExportRecord]] = defaultdict(list)
        for record in candidates:
            grouped[record.family_fingerprint].append(record)

        deduped: list[TrainingExportRecord] = []
        for fingerprint, group in grouped.items():
            group.sort(key=lambda item: (item.quality_score, item.created_at or "", item.source_id), reverse=True)
            kept = group[: self.policy.max_examples_per_fingerprint]
            dedupe_removed += max(0, len(group) - len(kept))
            deduped.extend(kept)
            split = self._stable_split(fingerprint)
            for record in kept:
                record.split = split
                record.family_size = len(group)

        deduped.sort(key=lambda item: (item.quality_score, item.created_at or "", item.source_id), reverse=True)

        selected: list[TrainingExportRecord] = []
        intent_counts: Counter[str] = Counter()
        tool_counts: Counter[str] = Counter()
        for record in deduped:
            if intent_counts[record.intent] >= self.policy.max_examples_per_intent:
                rejected_reasons["intent_cap"] += 1
                continue
            tool_key = "|".join(record.tool_graph) or "<empty>"
            if tool_counts[tool_key] >= self.policy.max_examples_per_tool_graph:
                rejected_reasons["tool_graph_cap"] += 1
                continue
            valid, reason = self.validate_export_record(record.to_export_dict())
            if not valid:
                rejected_reasons[reason or "privacy_validation_failed"] += 1
                continue
            intent_counts[record.intent] += 1
            tool_counts[tool_key] += 1
            selected.append(record)

        selected.sort(key=lambda item: (item.split, -item.quality_score, item.source_id))
        if limit is not None:
            selected = selected[:limit]

        source_distribution = Counter(record.plan_source or record.source_kind for record in selected)
        intent_distribution = Counter(record.intent for record in selected)
        tool_graph_distribution = Counter("|".join(record.tool_graph) or "<empty>" for record in selected)
        step_distribution = Counter(len(record.tool_graph) for record in selected)
        predicate_complexity_distribution = Counter(int(record.predicate_graph.get("predicate_count") or 0) for record in selected)
        avg_quality = sum(record.quality_score for record in selected) / len(selected) if selected else 0.0
        report = TrainingExportBundle(
            records=selected,
            rejected_count=inspected - len(selected),
            rejected_reasons=rejected_reasons,
            duplicates_removed=dedupe_removed,
            inspected_count=inspected,
            policy=self.policy,
            source_distribution=source_distribution,
            intent_distribution=Counter(record.intent for record in selected),
            tool_graph_distribution=Counter("|".join(record.tool_graph) or "<empty>" for record in selected),
            step_distribution=Counter(len(record.tool_graph) for record in selected),
            predicate_complexity_distribution=Counter(int(record.predicate_graph.get("predicate_count") or 0) for record in selected),
            average_quality=avg_quality,
            dataset_version="",
        )

        preview = [record.to_export_dict() for record in selected[: self.policy.preview_limit]]
        return report, preview

    def _records_to_jsonl(self, records: list[TrainingExportRecord]) -> bytes:
        lines = [json.dumps(record.to_export_dict(), sort_keys=True, separators=(",", ":"), default=str) for record in records]
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    def _records_to_csv(self, records: list[TrainingExportRecord]) -> bytes:
        rows = []
        for record in records:
            exported = record.to_export_dict()
            row = {
                "source_kind": record.source_kind,
                "source_id": record.source_id,
                "split": record.split,
                "family_fingerprint": record.family_fingerprint,
                "intent": record.intent,
                "quality_score": record.quality_score,
                "tool_graph": _stable_json(exported["output"]["tool_graph"]),
                "input": _stable_json(exported["input"]),
                "output": _stable_json(exported["output"]),
                "metadata": _stable_json(exported["metadata"]),
            }
            rows.append(row)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["source_kind", "source_id", "split", "family_fingerprint", "intent", "quality_score", "tool_graph", "input", "output", "metadata"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buffer.getvalue().encode("utf-8")

    def export_bytes(self, records: list[TrainingExportRecord], *, fmt: str) -> bytes:
        if fmt == "jsonl":
            return self._records_to_jsonl(records)
        if fmt == "csv":
            return self._records_to_csv(records)
        raise ValueError(f"Unsupported export format: {fmt}")

    def build_manifest(
        self,
        bundle: TrainingExportBundle,
        *,
        split_paths: dict[str, str] | None = None,
    ) -> TrainingDatasetManifest:
        report = bundle.report()
        report["dataset_version"] = report.get("dataset_version") or stable_hash(
            {
                "policy": self.policy.to_dict(),
                "eligible_examples": report.get("eligible_examples", 0),
                "family_count": report.get("family_count", 0),
                "train_count": report.get("train_count", 0),
                "validation_count": report.get("validation_count", 0),
                "test_count": report.get("test_count", 0),
                "intent_distribution": report.get("intent_distribution", {}),
                "tool_graph_distribution": report.get("tool_graph_distribution", {}),
                "average_quality": report.get("average_quality", 0.0),
                "duplicates_removed": report.get("duplicates_removed", 0),
            }
        )[:16]
        manifest = build_training_dataset_manifest(report, policy=self.policy, split_paths=split_paths)
        bundle.dataset_version = manifest.dataset_version
        return manifest

    def evaluate_readiness(self, bundle: TrainingExportBundle) -> TrainingReadinessAssessment:
        manifest = self.build_manifest(bundle)
        return manifest.readiness or self.backend.evaluate(bundle.report(), policy=self.policy)

    def invalidate_training_candidate(
        self,
        *,
        source_id: str | None = None,
        family_fingerprint: str | None = None,
        reason: str = "manual",
    ) -> dict[str, Any]:
        invalidation = TrainingCandidateInvalidation(
            source_id=source_id,
            family_fingerprint=family_fingerprint,
            reason=reason,
        )
        return self.store.append_training_invalidation(invalidation)

    def validate_export_record(self, record: dict[str, Any]) -> tuple[bool, str | None]:
        allowed_keys = {"input", "output", "metadata", "source_kind", "source_id", "split", "family_fingerprint"}
        if not isinstance(record, dict):
            return False, "record_not_object"
        if any(key not in allowed_keys for key in record.keys()):
            return False, "unexpected_record_key"
        source_kind = str(record.get("source_kind") or "")
        split = str(record.get("split") or "")
        source_id = str(record.get("source_id") or "")
        family_fingerprint = str(record.get("family_fingerprint") or "")
        if source_kind not in ALLOWED_SOURCE_KINDS:
            return False, "unsafe_source_kind"
        if split not in ALLOWED_SPLITS:
            return False, "unsafe_split"
        if not (_safe_token(source_id) or SAFE_HEX_RE.fullmatch(source_id)):
            return False, "unsafe_source_id"
        if not SAFE_HEX_RE.fullmatch(family_fingerprint):
            return False, "unsafe_family_fingerprint"
        ok, reason = self._recursively_validate_record({k: v for k, v in record.items() if k in {"input", "output", "metadata"}})
        return ok, reason

    def export_files(
        self,
        *,
        output_dir: Path | None = None,
        include_candidate_strategies: bool = True,
        limit: int | None = None,
    ) -> dict[str, Path]:
        bundle, _ = self.build_bundle(limit=limit, include_candidate_strategies=include_candidate_strategies)
        target_dir = Path(output_dir) if output_dir is not None else self.store.root / "training"
        target_dir.mkdir(parents=True, exist_ok=True)
        split_paths: dict[str, Path] = {}
        for split in ALLOWED_SPLITS:
            split_records = [record for record in bundle.records if record.split == split]
            path = target_dir / f"{split}.jsonl"
            path.write_bytes(self._records_to_jsonl(split_records))
            split_paths[split] = path
        manifest = self.build_manifest(bundle, split_paths={key: str(value) for key, value in split_paths.items()})
        report_path = target_dir / "dataset_report.json"
        report_path.write_text(json.dumps(bundle.report(), indent=2, sort_keys=True), encoding="utf-8")
        manifest_path = target_dir / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        split_paths["report"] = report_path
        split_paths["manifest"] = manifest_path
        return split_paths
