from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .canonical_training import StructuralFamily
from .models import stable_hash
from .training_export import ALLOWED_LOGICAL_STRUCTURES, SAFE_HEX_RE, SAFE_TOKEN_RE, TrainingExportBundle, _shape_signature


ALLOWED_SEMANTIC_INTENTS = {"filter", "analytics", "cleaning", "operation", "sentiment"}
ALLOWED_SEMANTIC_OUTPUT_KEYS = {
    "intent",
    "semantic_bindings",
    "predicate_graph",
    "aggregation",
    "ranking",
    "limit",
    "requires_fallback",
    "confidence",
}
ALLOWED_PREDICATE_OPERATORS = {"AND", "OR", "NOT", "SINGLE", "MIXED"}
ALLOWED_SEMANTIC_LOGICAL_STRUCTURES = set(ALLOWED_LOGICAL_STRUCTURES) | {"NOT"}


@dataclass(slots=True)
class SemanticExtractorTarget:
    source_kind: str
    source_id: str
    split: str
    family_fingerprint: str
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SemanticDatasetReadinessReport:
    row_count: int
    intent_diversity: int
    predicate_diversity: int
    role_coverage: float
    ambiguity_rate: float
    average_quality: float
    ready: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_token(value: str) -> bool:
    return bool(SAFE_TOKEN_RE.fullmatch(value or ""))


def _semantic_input_from_record(record: Any) -> dict[str, Any]:
    predicate_graph = dict(record.predicate_graph or {})
    return {
        "intent": record.intent,
        "safe_field_aliases": list(dict.fromkeys(record.semantic_roles or [])),
        "semantic_roles": list(record.semantic_roles or []),
        "dtypes": list(record.plan_shape.get("dtypes") or []),
        "logical_hints": {
            "logical_structure": record.logical_structure,
            "predicate_count": int(predicate_graph.get("predicate_count") or len(record.operators) or 0),
            "operators": list(record.operators or []),
        },
    }


def _semantic_output_from_record(record: Any) -> dict[str, Any]:
    predicate_graph = dict(record.predicate_graph or {})
    logical_structure = record.logical_structure if record.logical_structure in ALLOWED_SEMANTIC_LOGICAL_STRUCTURES else "SINGLE"
    semantic_bindings = {
        "dataset_signature": record.dataset_semantic_signature,
        "intent_hint": record.intent,
        "query_shape": record.plan_shape.get("query_shape"),
    }
    output = {
        "intent": record.intent,
        "semantic_bindings": semantic_bindings,
        "predicate_graph": {
            "logical_structure": logical_structure,
            "predicate_count": int(predicate_graph.get("predicate_count") or len(record.operators) or 0),
            "operators": [logical_structure] if logical_structure else [],
            "roles": list(record.semantic_roles or []),
            "validated": True,
        },
        "aggregation": {
            "measure_roles": list(record.plan_shape.get("measure_roles") or []),
            "required": bool(record.plan_shape.get("metric_count")),
        },
        "ranking": {
            "required": bool(record.plan_shape.get("tool_sequence")),
            "direction": "desc",
        },
        "limit": record.plan_shape.get("limit"),
        "requires_fallback": False,
        "confidence": min(1.0, max(0.0, float(record.quality_score or 0.0))),
    }
    return output


def validate_semantic_target(target: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(target, dict):
        return False, "target_not_object"
    if set(target.keys()) - {"source_kind", "source_id", "split", "family_fingerprint", "input", "output", "metadata"}:
        return False, "unexpected_top_level_key"
    source_kind = str(target.get("source_kind") or "")
    split = str(target.get("split") or "")
    family_fingerprint = str(target.get("family_fingerprint") or "")
    if source_kind not in {"experience", "strategy"}:
        return False, "unsafe_source_kind"
    if split not in {"train", "validation", "test"}:
        return False, "unsafe_split"
    if not SAFE_HEX_RE.fullmatch(family_fingerprint):
        return False, "unsafe_family_fingerprint"
    output = dict(target.get("output") or {})
    if not output or set(output.keys()) - ALLOWED_SEMANTIC_OUTPUT_KEYS:
        return False, "invalid_output_schema"
    intent = str(output.get("intent") or "")
    if intent not in ALLOWED_SEMANTIC_INTENTS:
        return False, "unsupported_intent"
    predicate_graph = output.get("predicate_graph")
    if not isinstance(predicate_graph, dict):
        return False, "invalid_predicate_ast"
    logical_structure = str(predicate_graph.get("logical_structure") or "SINGLE")
    if logical_structure not in ALLOWED_SEMANTIC_LOGICAL_STRUCTURES:
        return False, "invalid_logical_structure"
    operators = [str(item) for item in (predicate_graph.get("operators") or [])]
    if any(operator.upper() not in ALLOWED_PREDICATE_OPERATORS for operator in operators):
        return False, "invalid_predicate_operator"
    aggregation = output.get("aggregation")
    ranking = output.get("ranking")
    if not isinstance(aggregation, dict) or not isinstance(ranking, dict):
        return False, "invalid_aggregation_ranking"
    if "confidence" not in output or not isinstance(output.get("confidence"), (int, float)):
        return False, "missing_confidence"
    return True, None


def _split_by_family(family_fingerprint: str, *, train_ratio: float = 0.8, validation_ratio: float = 0.1) -> str:
    raw = int(stable_hash({"family_fingerprint": family_fingerprint})[:12], 16) / float(0xFFFFFFFFFFFF)
    if raw < train_ratio:
        return "train"
    if raw < train_ratio + validation_ratio:
        return "validation"
    return "test"


def _family_signature(record: Any) -> str:
    payload = {
        "intent": record.intent,
        "semantic_roles": list(record.semantic_roles or []),
        "logical_structure": record.logical_structure,
        "predicate_graph": _shape_signature(record.predicate_graph or {}),
        "plan_shape": _shape_signature(record.plan_shape or {}),
    }
    return stable_hash(payload)


def build_semantic_extractor_targets(bundle: TrainingExportBundle) -> list[SemanticExtractorTarget]:
    targets: list[SemanticExtractorTarget] = []
    for record in bundle.records:
        family_fingerprint = _family_signature(record)
        split = _split_by_family(family_fingerprint)
        target = SemanticExtractorTarget(
            source_kind=record.source_kind,
            source_id=record.source_id,
            split=split,
            family_fingerprint=family_fingerprint,
            input=_semantic_input_from_record(record),
            output=_semantic_output_from_record(record),
            metadata={
                "quality": record.quality_score,
                "plan_source": record.plan_source,
                "source_kind": record.source_kind,
                "family_size": record.family_size,
                "ambiguity": bool(record.no_unresolved_ambiguity is False),
                "dataset_semantic_signature": record.dataset_semantic_signature,
            },
        )
        ok, reason = validate_semantic_target(target.to_dict())
        if not ok:
            continue
        targets.append(target)
    return targets


def semantic_split_counts(targets: Iterable[SemanticExtractorTarget]) -> dict[str, int]:
    counts = Counter(target.split for target in targets)
    return {split: int(counts.get(split, 0)) for split in ("train", "validation", "test")}


def build_semantic_readiness_report(targets: list[SemanticExtractorTarget]) -> SemanticDatasetReadinessReport:
    row_count = len(targets)
    intents = Counter(target.output["intent"] for target in targets)
    predicates = Counter(target.output["predicate_graph"]["logical_structure"] for target in targets)
    roles = Counter(role for target in targets for role in target.input.get("semantic_roles", []))
    ambiguity_rate = (
        sum(1 for target in targets if bool(target.metadata.get("ambiguity"))) / row_count
        if row_count
        else 0.0
    )
    average_quality = (
        sum(float(target.metadata.get("quality") or 0.0) for target in targets) / row_count
        if row_count
        else 0.0
    )
    unique_roles = len({role for target in targets for role in target.input.get("semantic_roles", [])})
    role_coverage = min(1.0, unique_roles / 8.0) if row_count else 0.0
    notes: list[str] = []
    if row_count <= 0:
        notes.append("no_rows")
    if len(intents) < 3:
        notes.append("low_intent_diversity")
    if len(predicates) < 2:
        notes.append("low_predicate_diversity")
    if ambiguity_rate > 0.1:
        notes.append("ambiguity_rate_high")
    ready = (
        row_count > 0
        and role_coverage >= 0.5
        and ambiguity_rate <= 0.1
        and average_quality >= 0.95
        and len(intents) >= 1
        and len(predicates) >= 1
    )
    return SemanticDatasetReadinessReport(
        row_count=row_count,
        intent_diversity=len(intents),
        predicate_diversity=len(predicates),
        role_coverage=round(float(role_coverage), 4),
        ambiguity_rate=round(float(ambiguity_rate), 4),
        average_quality=round(float(average_quality), 4),
        ready=ready,
        notes=notes,
    )


def semantic_metrics(predicted: list[dict[str, Any]], expected: list[dict[str, Any]]) -> dict[str, float]:
    total = max(1, len(expected))
    intent_hits = binding_hits = predicate_hits = logical_hits = aggregation_hits = ranking_hits = fallback_hits = schema_hits = 0
    for pred, exp in zip(predicted, expected):
        if pred.get("intent") == exp.get("intent"):
            intent_hits += 1
        if pred.get("semantic_bindings") == exp.get("semantic_bindings"):
            binding_hits += 1
        if pred.get("predicate_graph") == exp.get("predicate_graph"):
            predicate_hits += 1
        if pred.get("predicate_graph", {}).get("logical_structure") == exp.get("predicate_graph", {}).get("logical_structure"):
            logical_hits += 1
        if pred.get("aggregation") == exp.get("aggregation"):
            aggregation_hits += 1
        if pred.get("ranking") == exp.get("ranking"):
            ranking_hits += 1
        if bool(pred.get("requires_fallback")) == bool(exp.get("requires_fallback")):
            fallback_hits += 1
        if set(pred.keys()) <= ALLOWED_SEMANTIC_OUTPUT_KEYS:
            schema_hits += 1
    return {
        "intent_accuracy": intent_hits / total,
        "binding_accuracy": binding_hits / total,
        "predicate_coverage": predicate_hits / total,
        "logical_structure_accuracy": logical_hits / total,
        "aggregation_accuracy": aggregation_hits / total,
        "ranking_accuracy": ranking_hits / total,
        "fallback_accuracy": fallback_hits / total,
        "semantic_schema_valid_rate": schema_hits / total,
    }
