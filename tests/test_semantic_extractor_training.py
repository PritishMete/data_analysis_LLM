from __future__ import annotations

from collections import Counter

from learning.semantic_extractor_training import (
    build_semantic_extractor_targets,
    build_semantic_readiness_report,
    semantic_metrics,
    semantic_split_counts,
    validate_semantic_target,
)
from learning.training_export import TrainingExportBundle, TrainingExportPolicy, TrainingExportRecord


def _record(
    *,
    source_id: str,
    intent: str,
    logical_structure: str,
    semantic_roles: list[str],
    operators: list[str],
    quality: float,
    family_fingerprint: str,
    plan_source: str = "validated_template",
) -> TrainingExportRecord:
    return TrainingExportRecord(
        source_kind="experience",
        source_id=source_id,
        intent=intent,
        semantic_roles=list(semantic_roles),
        operators=list(operators),
        logical_structure=logical_structure,
        tool_graph=["sql.filter"],
        predicate_graph={"operator": logical_structure, "predicate_count": len(operators)},
        plan_source=plan_source,
        plan_template_id="plan.template.semantic",
        quality_score=quality,
        execution_success=True,
        critic_passed=True,
        result_validation_passed=True,
        plan_completeness_passed=True,
        privacy_validation_passed=True,
        no_unresolved_ambiguity=True,
        no_critical_repair=True,
        repair_count=0,
        correction_state="validated",
        candidate_state=None,
        candidate_evidence_count=None,
        candidate_average_quality=None,
        dataset_semantic_signature="0123456789abcdef",
        family_fingerprint=family_fingerprint,
        split="train",
        family_size=1,
        plan_shape={
            "limit": 5,
            "metric_count": 1,
            "tool_sequence": ["sql.filter"],
            "dtypes": ["string", "bool"],
            "query_shape": "statement",
            "measure_roles": ["numeric_metric"],
        },
    )


def _bundle() -> TrainingExportBundle:
    records = [
        _record(
            source_id="evt_a",
            intent="filter",
            logical_structure="AND",
            semantic_roles=["boolean_capability", "rating_metric"],
            operators=["equals_true", "greater_than"],
            quality=0.98,
            family_fingerprint="a" * 64,
        ),
        _record(
            source_id="evt_b",
            intent="analytics",
            logical_structure="SINGLE",
            semantic_roles=["numeric_metric", "geographic_area"],
            operators=["avg"],
            quality=0.99,
            family_fingerprint="b" * 64,
        ),
        _record(
            source_id="evt_c",
            intent="operation",
            logical_structure="MIXED",
            semantic_roles=["entity_name", "numeric_metric"],
            operators=["normalize", "bin"],
            quality=0.97,
            family_fingerprint="c" * 64,
        ),
    ]
    return TrainingExportBundle(
        records=records,
        rejected_count=0,
        rejected_reasons=Counter(),
        duplicates_removed=0,
        inspected_count=len(records),
        policy=TrainingExportPolicy(max_examples_per_fingerprint=1),
        source_distribution=Counter({"experience": len(records)}),
        intent_distribution=Counter(record.intent for record in records),
        tool_graph_distribution=Counter("|".join(record.tool_graph) for record in records),
        step_distribution=Counter(len(record.tool_graph) for record in records),
        predicate_complexity_distribution=Counter(int(record.predicate_graph["predicate_count"]) for record in records),
        average_quality=sum(record.quality_score for record in records) / len(records),
        dataset_version="semantic-test",
    )


def test_semantic_extractor_targets_are_semantic_only():
    bundle = _bundle()
    targets = build_semantic_extractor_targets(bundle)

    assert len(targets) == 3
    sample = targets[0].to_dict()
    valid, reason = validate_semantic_target(sample)
    assert valid is True
    assert reason is None
    assert set(sample["output"].keys()) == {
        "intent",
        "semantic_bindings",
        "predicate_graph",
        "aggregation",
        "ranking",
        "limit",
        "requires_fallback",
        "confidence",
    }
    assert "tool_graph" not in sample["output"]
    assert "sql" not in str(sample).lower()


def test_semantic_extractor_split_families_and_readiness():
    bundle = _bundle()
    targets = build_semantic_extractor_targets(bundle)
    report = build_semantic_readiness_report(targets)
    split_counts = semantic_split_counts(targets)
    family_splits = {}
    for target in targets:
        family_splits.setdefault(target.family_fingerprint, set()).add(target.split)

    assert sum(split_counts.values()) == len(targets)
    assert report.row_count == 3
    assert report.intent_diversity == 3
    assert report.predicate_diversity >= 2
    assert report.average_quality >= 0.95
    assert report.ambiguity_rate == 0.0
    assert report.ready is True
    assert all(len(value) == 1 for value in family_splits.values())


def test_semantic_metrics_cover_expected_fields():
    predicted = [
        {
            "intent": "filter",
            "semantic_bindings": {"intent_hint": "filter"},
            "predicate_graph": {"logical_structure": "AND"},
            "aggregation": {"required": True},
            "ranking": {"required": False},
            "limit": 5,
            "requires_fallback": False,
            "confidence": 1.0,
        }
    ]
    expected = [dict(predicted[0])]
    metrics = semantic_metrics(predicted, expected)

    assert metrics["intent_accuracy"] == 1.0
    assert metrics["binding_accuracy"] == 1.0
    assert metrics["semantic_schema_valid_rate"] == 1.0


def test_semantic_metrics_scores_invalid_predictions_without_crashing():
    valid = {
        "intent": "filter",
        "semantic_bindings": {"intent_hint": "filter"},
        "predicate_graph": {"logical_structure": "AND"},
        "aggregation": {"required": True},
        "ranking": {"required": False},
        "limit": 5,
        "requires_fallback": False,
        "confidence": 1.0,
    }
    metrics = semantic_metrics([None, "", {}, [], "not json", valid], [valid] * 6)

    assert metrics["intent_accuracy"] == 1.0 / 6.0
    assert metrics["semantic_schema_valid_rate"] == 1.0 / 6.0
    assert metrics["prediction_none_count"] == 1
    assert metrics["empty_output_count"] == 1
    assert metrics["parse_failure_count"] == 1
    assert metrics["schema_failure_count"] == 2
    assert metrics["valid_prediction_count"] == 1


def test_sixteen_invalid_predictions_produce_zero_metrics():
    expected = [{
        "intent": "filter",
        "semantic_bindings": {"intent_hint": "filter"},
        "predicate_graph": {"logical_structure": "AND"},
        "aggregation": {"required": True},
        "ranking": {"required": False},
        "limit": 5,
        "requires_fallback": False,
        "confidence": 1.0,
    }] * 16
    metrics = semantic_metrics([None] * 16, expected)

    assert metrics["intent_accuracy"] == 0.0
    assert metrics["binding_accuracy"] == 0.0
    assert metrics["predicate_coverage"] == 0.0
    assert metrics["logical_structure_accuracy"] == 0.0
    assert metrics["fallback_accuracy"] == 0.0
    assert metrics["semantic_schema_valid_rate"] == 0.0
    assert metrics["prediction_none_count"] == 16


def test_canonical_planner_intents_are_supported_by_semantic_targets():
    assert {"aggregate", "compare", "trend", "benchmark", "risk_review"}.issubset(
        __import__("learning.semantic_extractor_training", fromlist=["ALLOWED_SEMANTIC_INTENTS"]).ALLOWED_SEMANTIC_INTENTS
    )
