from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

app_module = importlib.import_module("insight_learning.api.app")
from insight_learning.api.app import create_app
from learning.experience_store import LearningExperienceStore
from learning.models import ExperienceRecord, stable_hash
from learning.training_export import TrainingDatasetExporter, TrainingExportPolicy


SENSITIVE_VALUES = [
    "John Smith",
    "john@example.com",
    "ACC-9988",
    "SecretCompanyXYZ",
    "9876543210",
]


def _experience_record(
    *,
    event_id: str,
    intent: str = "filter",
    quality: float = 0.97,
    success: bool = True,
    critic_passed: bool = True,
    result_validation_passed: bool = True,
    plan_completeness_passed: bool = True,
    privacy_validation_passed: bool = True,
    no_unresolved_ambiguity: bool = True,
    no_critical_repair: bool = True,
    repair_count: int = 0,
    plan_source: str = "gemini_validated",
    tool_sequence: list[str] | None = None,
    semantic_roles: list[str] | None = None,
    operators: list[str] | None = None,
    logical_structure: str = "AND",
    created_at: str = "2026-08-26T00:00:00+00:00",
) -> ExperienceRecord:
    tool_sequence = tool_sequence or ["sql.filter", "sql.validate"]
    semantic_roles = semantic_roles or ["boolean", "numeric_measure"]
    operators = operators or ["equals_true", "less_than"]
    query_features = {
        "predicate_count": len(operators),
        "logical_structure": logical_structure,
        "semantic_roles": list(semantic_roles),
        "operators": list(operators),
    }
    plan_summary = {
        "tool_sequence": list(tool_sequence),
        "filters": [],
        "group_by": [],
        "metrics": [],
    }
    return ExperienceRecord(
        intent=intent,
        query_features=query_features,
        semantic_roles=list(semantic_roles),
        operators=list(operators),
        logical_structure=logical_structure,
        tool_sequence=list(tool_sequence),
        result_summary={"row_count": 3, "quality": quality},
        dataset_semantic_signature=stable_hash({"intent": intent, "tool_sequence": tool_sequence})[:16],
        semantic_signature=stable_hash({
            "intent": intent,
            "tool_sequence": tool_sequence,
            "semantic_roles": semantic_roles,
            "operators": operators,
        }),
        route="sql",
        skill_id="filter.multi_condition.v1",
        confidence=quality,
        success=success,
        score=quality,
        event_id=event_id,
        plan_hash=stable_hash(plan_summary),
        plan_summary=plan_summary,
        failure_reason=None,
        feedback_score=None,
        repair_count=repair_count,
        critic_passed=critic_passed,
        result_validation_passed=result_validation_passed,
        plan_completeness_passed=plan_completeness_passed,
        privacy_validation_passed=privacy_validation_passed,
        no_unresolved_ambiguity=no_unresolved_ambiguity,
        no_critical_repair=no_critical_repair,
        correction_state="validated" if no_critical_repair else "corrected",
        skill_state_before=None,
        skill_state_after=None,
        plan_source=plan_source,
        plan_template_id="plan.template.safe",
        plan_provenance={"template": {"tool_sequence": list(tool_sequence)}},
        correction_type=None,
        correction_summary=None,
        candidate_strategy_id=None,
        created_at=created_at,
        version=2,
    )


def _store_and_export(tmp_path: Path) -> tuple[LearningExperienceStore, TrainingDatasetExporter]:
    store = LearningExperienceStore(root=tmp_path / "state")
    exporter = TrainingDatasetExporter(store)
    return store, exporter


def _assert_no_sensitive_text(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for value in SENSITIVE_VALUES:
            assert value not in text, f"found sensitive value {value!r} in {path}"


def test_strict_eligibility_rejects_bad_results_and_low_quality(tmp_path):
    store, exporter = _store_and_export(tmp_path)
    store.append(_experience_record(event_id="evt_good"))
    store.append(_experience_record(event_id="evt_bad_validation", result_validation_passed=False))
    store.append(_experience_record(event_id="evt_bad_critic", critic_passed=False))
    store.append(_experience_record(event_id="evt_bad_quality", quality=0.80))

    bundle, preview = exporter.build_bundle()

    report = bundle.report()
    assert report["eligible_examples"] == 1
    assert report["rejected_examples"] >= 3
    assert report["rejection_reasons"]["result_validation_failed"] == 1
    assert report["rejection_reasons"]["critic_failed"] == 1
    assert report["rejection_reasons"]["quality_below_threshold"] == 1
    assert len(preview) == 1


def test_event_id_idempotency_prevents_double_learning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("INSIGHT_LEARNING_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("DATA_ANALYSIS_LLM_STATE_DIR", str(tmp_path / "state"))
    app_module._SERVICE = None
    client = TestClient(create_app())
    caplog.set_level("INFO")

    payload = {
        "schema_version": 1,
        "event_id": "evt_idempotent",
        "intent": "summarize",
        "query_features": {
            "predicate_count": 0,
            "logical_structure": "SINGLE",
            "semantic_roles": ["text_summary"],
            "operators": [],
        },
        "dataset_profile": {
            "fields": [
                {"id": "field_001", "semantic_role": "text_summary", "dtype": "string"},
            ]
        },
        "plan": {"tool_sequence": ["operation.categorize"], "action": "categorize"},
        "execution": {"success": True, "route": "operation"},
        "validation": {"success": True, "warnings": []},
        "quality_score": 0.97,
        "route": "operation",
        "plan_source": "validated_template",
        "skill_id": "operation.summarize.v1",
        "plan_template_id": "plan.template.safe",
        "dataset_semantic_signature": "0123456789abcdef",
        "critic_passed": True,
        "result_validation_passed": True,
        "plan_completeness_passed": True,
        "privacy_validation_passed": True,
        "no_unresolved_ambiguity": True,
        "no_critical_repair": True,
        "correction_state": "validated",
        "safe_query_abstraction": {
            "query_text": "John Smith",
            "normalized_query": "john@example.com",
            "account_number": "ACC-9988",
        },
        "query_features": {
            "predicate_count": 2,
            "logical_structure": "AND",
            "semantic_roles": ["boolean", "numeric_measure"],
            "operators": ["equals_true", "less_than"],
            "query_text": "SecretCompanyXYZ",
        },
    }

    first = client.post("/v1/experience", json=payload)
    second = client.post("/v1/experience", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200

    service = app_module.get_service()
    service.store.append(_experience_record(event_id="evt_safe"))
    export = client.get("/v1/export/training-dataset?format=report&persist=true")
    assert export.status_code == 200
    report = export.json()["report"]
    assert report["eligible_examples"] == 1

    experiences_path = service.store.experiences_path
    candidate_path = service.store.candidate_strategies_path
    assert experiences_path.exists()
    training_dir = tmp_path / "runtime" / "training"
    _assert_no_sensitive_text(
        [
            experiences_path,
            candidate_path,
            training_dir / "train.jsonl",
            training_dir / "validation.jsonl",
            training_dir / "test.jsonl",
            training_dir / "dataset_report.json",
        ]
    )
    first_json = json.dumps(first.json(), sort_keys=True)
    export_json = json.dumps(export.json(), sort_keys=True)
    assert all(value not in first_json for value in SENSITIVE_VALUES)
    assert all(value not in export_json for value in SENSITIVE_VALUES)
    assert all(value not in caplog.text for value in SENSITIVE_VALUES)


def test_event_id_idempotency_prevents_double_learning(tmp_path):
    store = LearningExperienceStore(root=tmp_path / "runtime")
    record = _experience_record(event_id="evt_idempotent")
    store.append(record)
    store.append(record)

    experiences_path = store.experiences_path
    lines = [line for line in experiences_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert sum(1 for line in lines if json.loads(line).get("event_id") == "evt_idempotent") == 1

    exporter = TrainingDatasetExporter(store)
    bundle, _ = exporter.build_bundle()
    assert bundle.report()["eligible_examples"] == 1


def test_structural_dedup_prefers_highest_quality(tmp_path):
    store, exporter = _store_and_export(tmp_path)
    store.append(_experience_record(event_id="evt_a", quality=0.96))
    store.append(_experience_record(event_id="evt_b", quality=0.99))
    store.append(_experience_record(event_id="evt_c", quality=0.97))

    bundle, _ = exporter.build_bundle()
    report = bundle.report()

    assert report["eligible_examples"] == 1
    assert report["duplicates_removed"] == 2
    assert bundle.records[0].quality_score == 0.99


def test_split_assignment_stays_with_family(tmp_path):
    store, _ = _store_and_export(tmp_path)
    policy = TrainingExportPolicy(max_examples_per_fingerprint=3)
    exporter = TrainingDatasetExporter(store, policy)

    store.append(_experience_record(event_id="evt_split_1", tool_sequence=["sql.filter", "sql.validate"]))
    store.append(_experience_record(event_id="evt_split_2", tool_sequence=["sql.filter", "sql.validate"]))
    store.append(_experience_record(event_id="evt_split_3", tool_sequence=["sql.filter", "sql.validate"]))

    bundle, _ = exporter.build_bundle()
    splits = {record.split for record in bundle.records}

    assert splits == {bundle.records[0].split}
    assert len(bundle.records) == 3


def test_dataset_balance_applies_intent_caps(tmp_path):
    store, _ = _store_and_export(tmp_path)
    policy = TrainingExportPolicy(max_examples_per_intent=2, max_examples_per_fingerprint=1)
    exporter = TrainingDatasetExporter(store, policy)

    for index in range(6):
        store.append(
            _experience_record(
                event_id=f"evt_filter_{index}",
                intent="filter",
                tool_sequence=["sql.filter", f"sql.step_{index}"],
                created_at=f"2026-08-26T00:00:{index:02d}+00:00",
            )
        )
    for index in range(3):
        store.append(
            _experience_record(
                event_id=f"evt_aggregate_{index}",
                intent="aggregate",
                tool_sequence=["sql.group_by", f"sql.step_{index}"],
                created_at=f"2026-08-26T00:01:{index:02d}+00:00",
            )
        )

    bundle, _ = exporter.build_bundle()
    report = bundle.report()

    assert report["eligible_examples"] == 4
    assert report["rejection_reasons"]["intent_cap"] == 5
    assert report["intent_distribution"]["filter"] == 2
    assert report["intent_distribution"]["aggregate"] == 2
