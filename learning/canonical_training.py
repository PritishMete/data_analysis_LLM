from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .models import SCHEMA_VERSION, stable_hash


CANONICAL_CORPUS_VERSION = 1
DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_MINIMUM_QUALITY = 0.95
DEFAULT_MINIMUM_READINESS_QUALITY = 0.96
DEFAULT_MINIMUM_ELIGIBLE_EXAMPLES = 500
DEFAULT_MINIMUM_FAMILY_COUNT = 50
DEFAULT_MINIMUM_INTENT_COUNT = 10
DEFAULT_MAX_SINGLE_INTENT_SHARE = 0.40
DEFAULT_MAX_SINGLE_TOOL_GRAPH_SHARE = 0.40


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _report_payload(bundle_or_report: Any) -> dict[str, Any]:
    if bundle_or_report is None:
        return {}
    if isinstance(bundle_or_report, dict):
        return dict(bundle_or_report)
    report = getattr(bundle_or_report, "report", None)
    if callable(report):
        value = report()
        if isinstance(value, dict):
            return dict(value)
    return {}


def _policy_payload(policy: Any) -> dict[str, Any]:
    if policy is None:
        return {}
    if isinstance(policy, dict):
        return dict(policy)
    if hasattr(policy, "to_dict") and callable(policy.to_dict):
        value = policy.to_dict()
        if isinstance(value, dict):
            return dict(value)
    return {
        key: getattr(policy, key)
        for key in (
            "minimum_quality",
            "require_execution_success",
            "require_critic_pass",
            "require_result_validation",
            "require_plan_completeness",
            "require_privacy_pass",
            "require_no_unresolved_ambiguity",
            "require_no_critical_repair",
            "allow_repaired_examples",
            "max_examples_per_fingerprint",
            "max_examples_per_intent",
            "max_examples_per_tool_graph",
            "train_ratio",
            "validation_ratio",
            "test_ratio",
        )
        if hasattr(policy, key)
    }


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    return [value]


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dataset_version(report: dict[str, Any], policy: dict[str, Any]) -> str:
    payload = {
        "policy": policy,
        "eligible_examples": report.get("eligible_examples", 0),
        "train_count": report.get("train_count", 0),
        "validation_count": report.get("validation_count", 0),
        "test_count": report.get("test_count", 0),
        "intent_distribution": report.get("intent_distribution", {}),
        "tool_graph_distribution": report.get("tool_graph_distribution", {}),
        "average_quality": report.get("average_quality", 0.0),
        "duplicates_removed": report.get("duplicates_removed", 0),
    }
    return stable_hash(payload)[:16]


def _distribution_share(distribution: Mapping[str, Any]) -> float:
    total = sum(int(value or 0) for value in distribution.values())
    if total <= 0:
        return 0.0
    return max(int(value or 0) for value in distribution.values()) / float(total)


@dataclass(slots=True)
class StructuralFamily:
    fingerprint: str
    intent: str
    semantic_roles: list[str]
    operators: list[str]
    logical_structure: str
    tool_graph: list[str]
    predicate_graph: dict[str, Any]
    plan_source: str | None = None
    plan_template_id: str | None = None
    source_kinds: list[str] = field(default_factory=list)
    example_count: int = 0
    best_quality: float = 0.0
    split: str = "train"
    version: int = CANONICAL_CORPUS_VERSION

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "StructuralFamily":
        input_payload = _safe_dict(record.get("input"))
        output_payload = _safe_dict(record.get("output"))
        metadata = _safe_dict(record.get("metadata"))
        return cls(
            fingerprint=str(record.get("family_fingerprint") or metadata.get("family_fingerprint") or ""),
            intent=str(input_payload.get("intent") or ""),
            semantic_roles=[str(item) for item in _safe_list(input_payload.get("semantic_roles"))],
            operators=[str(item) for item in _safe_list(input_payload.get("operators"))],
            logical_structure=str(input_payload.get("logical_structure") or "SINGLE"),
            tool_graph=[str(item) for item in _safe_list(output_payload.get("tool_graph"))],
            predicate_graph=_safe_dict(input_payload.get("predicate_graph")),
            plan_source=(
                str(output_payload.get("plan_source"))
                if output_payload.get("plan_source") is not None
                else None
            ),
            plan_template_id=(
                str(output_payload.get("plan_template_id"))
                if output_payload.get("plan_template_id") is not None
                else None
            ),
            source_kinds=[str(item) for item in _safe_list(metadata.get("source_kinds"))],
            example_count=int(metadata.get("family_size") or 0),
            best_quality=float(metadata.get("quality") or 0.0),
            split=str(record.get("split") or metadata.get("split") or "train"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FineTuningCandidate:
    source_kind: str
    source_id: str
    family: StructuralFamily
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]
    eligible: bool = True
    invalidated: bool = False
    rejection_reason: str | None = None
    corpus_version: int = CANONICAL_CORPUS_VERSION

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        eligible: bool = True,
        invalidated: bool = False,
        rejection_reason: str | None = None,
    ) -> "FineTuningCandidate":
        family = StructuralFamily.from_record(record)
        input_payload = _safe_dict(record.get("input"))
        output_payload = _safe_dict(record.get("output"))
        metadata = _safe_dict(record.get("metadata"))
        return cls(
            source_kind=str(record.get("source_kind") or ""),
            source_id=str(record.get("source_id") or ""),
            family=family,
            input=input_payload,
            output=output_payload,
            metadata=metadata,
            eligible=eligible,
            invalidated=invalidated,
            rejection_reason=rejection_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["family"] = self.family.to_dict()
        return payload


@dataclass(slots=True)
class TrainingCandidateInvalidation:
    source_id: str | None = None
    family_fingerprint: str | None = None
    reason: str = "manual"
    created_at: str = field(default_factory=_utcnow_iso)
    corpus_version: int = CANONICAL_CORPUS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingCandidateInvalidation":
        data = dict(payload)
        data.setdefault("source_id", None)
        data.setdefault("family_fingerprint", None)
        data.setdefault("reason", "manual")
        data.setdefault("created_at", _utcnow_iso())
        data.setdefault("corpus_version", CANONICAL_CORPUS_VERSION)
        return cls(**data)


@dataclass(slots=True)
class TrainingBenchmarkResult:
    dataset_version: str
    eligible_examples: int
    family_count: int
    intent_count: int
    tool_graph_count: int
    average_quality: float
    readiness_score: float
    largest_intent_share: float = 0.0
    largest_tool_graph_share: float = 0.0
    notes: list[str] = field(default_factory=list)
    version: int = CANONICAL_CORPUS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingReadinessAssessment:
    ready: bool
    ready_for_prototype: bool
    reason: str | None
    checked_at: str
    dataset_version: str
    eligible_examples: int
    family_count: int
    intent_count: int
    tool_graph_count: int
    rejected_examples: int
    minimum_quality: float
    minimum_readiness_quality: float
    minimum_eligible_examples: int
    minimum_family_count: int
    minimum_intent_count: int
    max_single_intent_share: float
    max_single_tool_graph_share: float
    all_required_gates_passed: bool
    privacy_gate_passed: bool
    dedupe_gate_passed: bool
    split_integrity_passed: bool
    average_quality: float
    largest_intent_share: float
    largest_tool_graph_share: float
    benchmark: TrainingBenchmarkResult | None = None
    notes: list[str] = field(default_factory=list)
    balance_warnings: list[str] = field(default_factory=list)
    version: int = CANONICAL_CORPUS_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.benchmark is not None:
            payload["benchmark"] = self.benchmark.to_dict()
        payload["ready_for_prototype"] = self.ready_for_prototype
        return payload


@dataclass(slots=True)
class PlannerModelVersion:
    model_name: str = "planner"
    model_version: str = "planner.corpus.v1"
    schema_version: int = SCHEMA_VERSION
    corpus_version: int = CANONICAL_CORPUS_VERSION
    dataset_version: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingDatasetManifest:
    manifest_version: int
    corpus_version: int
    dataset_version: str
    generated_at: str
    policy: dict[str, Any]
    inspected_count: int
    eligible_examples: int
    rejected_examples: int
    duplicates_removed: int
    rejection_reasons: dict[str, int]
    source_distribution: dict[str, int]
    intent_distribution: dict[str, int]
    tool_graph_distribution: dict[str, int]
    step_distribution: dict[str, int]
    predicate_complexity_distribution: dict[str, int]
    average_quality: float
    train_count: int
    validation_count: int
    test_count: int
    family_count: int
    split_paths: dict[str, str] = field(default_factory=dict)
    readiness: TrainingReadinessAssessment | None = None
    benchmark: TrainingBenchmarkResult | None = None
    model_version: PlannerModelVersion | None = None
    version: int = CANONICAL_CORPUS_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.readiness is not None:
            payload["readiness"] = self.readiness.to_dict()
        if self.benchmark is not None:
            payload["benchmark"] = self.benchmark.to_dict()
        if self.model_version is not None:
            payload["model_version"] = self.model_version.to_dict()
        return payload


def evaluate_training_readiness(
    report: Mapping[str, Any],
    *,
    policy: Any = None,
    benchmark: TrainingBenchmarkResult | None = None,
) -> TrainingReadinessAssessment:
    report_payload = dict(report)
    policy_payload = _policy_payload(policy)
    minimum_quality = float(policy_payload.get("minimum_quality", DEFAULT_MINIMUM_QUALITY) or DEFAULT_MINIMUM_QUALITY)
    minimum_readiness_quality = float(policy_payload.get("minimum_readiness_quality", DEFAULT_MINIMUM_READINESS_QUALITY) or DEFAULT_MINIMUM_READINESS_QUALITY)
    minimum_eligible_examples = int(policy_payload.get("minimum_eligible_examples", DEFAULT_MINIMUM_ELIGIBLE_EXAMPLES) or DEFAULT_MINIMUM_ELIGIBLE_EXAMPLES)
    minimum_family_count = int(policy_payload.get("minimum_family_count", DEFAULT_MINIMUM_FAMILY_COUNT) or DEFAULT_MINIMUM_FAMILY_COUNT)
    minimum_intent_count = int(policy_payload.get("minimum_intent_count", DEFAULT_MINIMUM_INTENT_COUNT) or DEFAULT_MINIMUM_INTENT_COUNT)
    max_single_intent_share = float(policy_payload.get("max_single_intent_share", DEFAULT_MAX_SINGLE_INTENT_SHARE) or DEFAULT_MAX_SINGLE_INTENT_SHARE)
    max_single_tool_graph_share = float(policy_payload.get("max_single_tool_graph_share", DEFAULT_MAX_SINGLE_TOOL_GRAPH_SHARE) or DEFAULT_MAX_SINGLE_TOOL_GRAPH_SHARE)
    eligible_examples = int(report_payload.get("eligible_examples") or 0)
    family_count = int(report_payload.get("family_count") or 0)
    intent_count = int(report_payload.get("intent_count") or len(report_payload.get("intent_distribution") or {}))
    tool_graph_count = int(report_payload.get("tool_graph_count") or len(report_payload.get("tool_graph_distribution") or {}))
    rejected_examples = int(report_payload.get("rejected_examples") or 0)
    average_quality = float(report_payload.get("average_quality") or 0.0)
    train_count = int(report_payload.get("train_count") or 0)
    validation_count = int(report_payload.get("validation_count") or 0)
    test_count = int(report_payload.get("test_count") or 0)
    duplicates_removed = int(report_payload.get("duplicates_removed") or 0)
    largest_intent_share = float(report_payload.get("largest_intent_share") or _distribution_share(report_payload.get("intent_distribution") or {}))
    largest_tool_graph_share = float(report_payload.get("largest_tool_graph_share") or _distribution_share(report_payload.get("tool_graph_distribution") or {}))
    privacy_gate_passed = eligible_examples > 0 and rejected_examples >= 0
    dedupe_gate_passed = duplicates_removed >= 0
    split_integrity_passed = (train_count + validation_count + test_count) == eligible_examples
    all_required_gates_passed = (
        eligible_examples > 0
        and average_quality >= minimum_quality
        and privacy_gate_passed
        and dedupe_gate_passed
        and split_integrity_passed
    )
    volume_gate_passed = eligible_examples >= minimum_eligible_examples
    family_gate_passed = family_count >= minimum_family_count
    intent_gate_passed = intent_count >= minimum_intent_count
    quality_gate_passed = average_quality >= minimum_readiness_quality
    concentration_gate_passed = largest_intent_share <= max_single_intent_share and largest_tool_graph_share <= max_single_tool_graph_share
    notes: list[str] = []
    balance_warnings: list[str] = []
    if eligible_examples < minimum_eligible_examples:
        notes.append("eligible_examples_below_threshold")
    if family_count < minimum_family_count:
        notes.append("family_count_below_threshold")
    if intent_count < minimum_intent_count:
        notes.append("intent_count_below_threshold")
    if average_quality < minimum_readiness_quality:
        notes.append("quality_below_readiness_threshold")
    if largest_intent_share > max_single_intent_share:
        warning = f"largest_intent_share={largest_intent_share:.3f}"
        notes.append(warning)
        balance_warnings.append(warning)
    if largest_tool_graph_share > max_single_tool_graph_share:
        warning = f"largest_tool_graph_share={largest_tool_graph_share:.3f}"
        notes.append(warning)
        balance_warnings.append(warning)
    if eligible_examples <= 0:
        notes.append("no_eligible_examples")
    if not split_integrity_passed:
        notes.append("split_integrity_failed")
    ready = all_required_gates_passed and volume_gate_passed and family_gate_passed and intent_gate_passed and quality_gate_passed and concentration_gate_passed
    reason = None if ready else ",".join(notes) if notes else "not_ready"
    readiness_score = min(
        1.0,
        max(
            0.0,
            (
                0.30 * min(1.0, eligible_examples / float(minimum_eligible_examples))
                + 0.20 * min(1.0, family_count / float(minimum_family_count))
                + 0.15 * min(1.0, intent_count / float(minimum_intent_count))
                + 0.20 * min(1.0, average_quality / minimum_readiness_quality)
                + 0.075 * max(0.0, 1.0 - (largest_intent_share / max_single_intent_share if max_single_intent_share else largest_intent_share))
                + 0.075 * max(0.0, 1.0 - (largest_tool_graph_share / max_single_tool_graph_share if max_single_tool_graph_share else largest_tool_graph_share))
            ),
        ),
    )
    benchmark = benchmark or TrainingBenchmarkResult(
        dataset_version=str(report_payload.get("dataset_version") or ""),
        eligible_examples=eligible_examples,
        family_count=family_count,
        intent_count=intent_count,
        tool_graph_count=tool_graph_count,
        average_quality=average_quality,
        readiness_score=readiness_score,
        largest_intent_share=largest_intent_share,
        largest_tool_graph_share=largest_tool_graph_share,
        notes=list(notes),
    )
    return TrainingReadinessAssessment(
        ready=ready,
        ready_for_prototype=ready,
        reason=reason,
        checked_at=_utcnow_iso(),
        dataset_version=str(report_payload.get("dataset_version") or ""),
        eligible_examples=eligible_examples,
        family_count=family_count,
        intent_count=intent_count,
        tool_graph_count=tool_graph_count,
        rejected_examples=rejected_examples,
        minimum_quality=minimum_quality,
        minimum_readiness_quality=minimum_readiness_quality,
        minimum_eligible_examples=minimum_eligible_examples,
        minimum_family_count=minimum_family_count,
        minimum_intent_count=minimum_intent_count,
        max_single_intent_share=max_single_intent_share,
        max_single_tool_graph_share=max_single_tool_graph_share,
        all_required_gates_passed=all_required_gates_passed,
        privacy_gate_passed=privacy_gate_passed,
        dedupe_gate_passed=dedupe_gate_passed,
        split_integrity_passed=split_integrity_passed,
        average_quality=average_quality,
        largest_intent_share=largest_intent_share,
        largest_tool_graph_share=largest_tool_graph_share,
        benchmark=benchmark,
        notes=notes,
        balance_warnings=balance_warnings,
    )


def build_training_dataset_manifest(
    report: Mapping[str, Any],
    *,
    policy: Any = None,
    split_paths: Mapping[str, str] | None = None,
    benchmark: TrainingBenchmarkResult | None = None,
) -> TrainingDatasetManifest:
    report_payload = dict(report)
    policy_payload = _policy_payload(policy)
    dataset_version = _dataset_version(report_payload, policy_payload)
    report_payload.setdefault("dataset_version", dataset_version)
    readiness = evaluate_training_readiness(report_payload, policy=policy_payload, benchmark=benchmark)
    return TrainingDatasetManifest(
        manifest_version=1,
        corpus_version=CANONICAL_CORPUS_VERSION,
        dataset_version=dataset_version,
        generated_at=_utcnow_iso(),
        policy=policy_payload,
        inspected_count=int(report_payload.get("total_experiences_inspected") or 0),
        eligible_examples=int(report_payload.get("eligible_examples") or 0),
        rejected_examples=int(report_payload.get("rejected_examples") or 0),
        duplicates_removed=int(report_payload.get("duplicates_removed") or 0),
        rejection_reasons=dict(report_payload.get("rejection_reasons") or {}),
        source_distribution=dict(report_payload.get("plan_source_distribution") or {}),
        intent_distribution=dict(report_payload.get("intent_distribution") or {}),
        tool_graph_distribution=dict(report_payload.get("tool_graph_distribution") or {}),
        step_distribution={str(key): int(value) for key, value in dict(report_payload.get("number_of_steps_distribution") or {}).items()},
        predicate_complexity_distribution={str(key): int(value) for key, value in dict(report_payload.get("predicate_complexity_distribution") or {}).items()},
        average_quality=float(report_payload.get("average_quality") or 0.0),
        train_count=int(report_payload.get("train_count") or 0),
        validation_count=int(report_payload.get("validation_count") or 0),
        test_count=int(report_payload.get("test_count") or 0),
        family_count=int(report_payload.get("family_count") or report_payload.get("eligible_examples") or 0),
        split_paths={str(key): str(value) for key, value in dict(split_paths or {}).items()},
        readiness=readiness,
        benchmark=readiness.benchmark,
        model_version=PlannerModelVersion(dataset_version=dataset_version),
    )


class PlannerTrainingBackend:
    def __init__(self, *, model_name: str = "planner", model_version: str = "planner.corpus.v1") -> None:
        self.model_version = PlannerModelVersion(model_name=model_name, model_version=model_version)

    def build_manifest(
        self,
        bundle_or_report: Any,
        *,
        policy: Any = None,
        split_paths: Mapping[str, str] | None = None,
    ) -> TrainingDatasetManifest:
        report = _report_payload(bundle_or_report)
        manifest = build_training_dataset_manifest(report, policy=policy, split_paths=split_paths)
        manifest.model_version = self.model_version
        manifest.model_version.dataset_version = manifest.dataset_version
        return manifest

    def evaluate(
        self,
        bundle_or_report: Any,
        *,
        policy: Any = None,
    ) -> TrainingReadinessAssessment:
        report = _report_payload(bundle_or_report)
        return evaluate_training_readiness(report, policy=policy)

    def benchmark(
        self,
        bundle_or_report: Any,
        *,
        policy: Any = None,
    ) -> TrainingBenchmarkResult:
        report = _report_payload(bundle_or_report)
        readiness = evaluate_training_readiness(report, policy=policy)
        return readiness.benchmark or TrainingBenchmarkResult(
            dataset_version=readiness.dataset_version,
            eligible_examples=readiness.eligible_examples,
            family_count=readiness.family_count,
            intent_count=readiness.intent_count,
            tool_graph_count=readiness.tool_graph_count,
            average_quality=readiness.average_quality,
            readiness_score=1.0 if readiness.ready_for_prototype else readiness.benchmark.readiness_score if readiness.benchmark else 0.0,
            largest_intent_share=readiness.largest_intent_share,
            largest_tool_graph_share=readiness.largest_tool_graph_share,
            notes=list(readiness.notes),
        )
