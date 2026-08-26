from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

app_module = importlib.import_module("insight_learning.api.app")
from insight_learning.api.app import create_app
from learning.models import ExperienceRecord, stable_hash


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("INSIGHT_LEARNING_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("DATA_ANALYSIS_LLM_STATE_DIR", str(tmp_path / "state"))
    app_module._SERVICE = None
    app = create_app()
    return TestClient(app)


def _eligible_record(*, event_id: str, intent: str = "summarize") -> ExperienceRecord:
    tool_sequence = ["operation.categorize"]
    plan_summary = {"tool_sequence": list(tool_sequence), "action": "categorize"}
    return ExperienceRecord(
        intent=intent,
        query_features={
            "intent": intent,
            "predicate_count": 0,
            "logical_structure": "SINGLE",
            "semantic_roles": ["text_summary"],
            "operators": [],
            "operation_hints": [intent],
            "tool_hints": ["analytics.summary"],
            "query_shape": "statement",
            "semantic_signature": stable_hash({"intent": intent, "tool_sequence": tool_sequence}),
            "confidence": 0.8,
        },
        semantic_roles=["text_summary"],
        operators=[],
        logical_structure="SINGLE",
        tool_sequence=tool_sequence,
        result_summary={"result_kind": "table", "row_count": 1, "column_count": 1},
        dataset_semantic_signature="0123456789abcdef",
        semantic_signature=stable_hash({"intent": intent, "tool_sequence": tool_sequence}),
        route="operation",
        skill_id="operation.summarize.v1",
        confidence=0.97,
        success=True,
        score=0.97,
        event_id=event_id,
        plan_hash=stable_hash(plan_summary),
        plan_summary=plan_summary,
        feedback_score=5,
        repair_count=0,
        critic_passed=True,
        result_validation_passed=True,
        plan_completeness_passed=True,
        privacy_validation_passed=True,
        no_unresolved_ambiguity=True,
        no_critical_repair=True,
        correction_state="validated",
        plan_source="validated_template",
        plan_template_id="plan.template.safe",
        plan_provenance={"template": {"tool_sequence": list(tool_sequence)}},
        created_at="2026-08-26T00:00:00+00:00",
        version=2,
    )


def test_health_and_skills(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    skills = client.get("/v1/skills")
    assert skills.status_code == 200
    assert isinstance(skills.json()["skills"], list)


def test_plan_endpoint_returns_safe_plan(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    response = client.post(
        "/v1/plan",
        json={
            "intent": "filter",
            "query_features": {
                "predicate_count": 3,
                "logical_structure": "AND",
                "semantic_roles": ["boolean", "boolean", "numeric_measure"],
                "operators": ["equals_true", "equals_true", "less_than"],
            },
            "dataset_profile": {
                "fields": [
                    {"id": "field_001", "semantic_role": "boolean", "dtype": "boolean"},
                    {"id": "field_002", "semantic_role": "boolean", "dtype": "boolean"},
                    {"id": "field_003", "semantic_role": "numeric_measure", "dtype": "float"},
                ]
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "plan_source" in payload
    assert isinstance(payload["tool_graph"], list)
    assert "critic_status" in payload


def test_experience_and_feedback_endpoints(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    experience = client.post(
        "/v1/experience",
        json={
            "schema_version": 1,
            "event_id": "evt_001",
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
            "quality_score": 0.96,
            "route": "operation",
            "plan_source": "validated_template",
            "skill_id": "operation.summarize.v1",
            "dataset_semantic_signature": "0123456789abcdef",
            "critic_passed": True,
            "result_validation_passed": True,
            "plan_completeness_passed": True,
            "privacy_validation_passed": True,
            "no_unresolved_ambiguity": True,
            "no_critical_repair": True,
            "correction_state": "validated",
            "safe_query_abstraction": {"text": "filter request", "available_columns": ["field_001"]},
        },
    )
    assert experience.status_code == 200
    assert experience.json()["stored"] is True

    feedback = client.post(
        "/v1/feedback",
        json={
            "decision_id": "decision.001",
            "feedback_score": 5,
            "correction_type": "wrong_semantic_field",
            "affected_intent": "filter",
            "generalized_lesson": "preserve the semantic role and avoid inventing columns",
            "dataset_semantic_signature": "signature-001",
            "requested_role": "numeric_measure",
            "resolution_preference": "Defect Rate",
            "preferred_semantic_candidate": "Defect Rate",
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["accepted"] is True


def test_training_dataset_export_includes_records(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    service = app_module.get_service()
    service.store.append(_eligible_record(event_id="evt_002"))

    export = client.get("/v1/export/training-dataset?format=json&persist=true")
    assert export.status_code == 200
    payload = export.json()
    assert payload["exported"] is True
    assert payload["format"] == "json"
    assert payload["eligible_examples"] >= 1
    assert isinstance(payload["records"], list)
    assert payload["records"][0]["input"]["intent"] == "summarize"
    assert "report" in payload

    report_export = client.get("/v1/export/training-dataset?format=report")
    assert report_export.status_code == 200
    report_payload = report_export.json()
    assert report_payload["format"] == "report"
    assert "eligible_examples" in report_payload["report"]


def test_metrics_endpoint(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    assert "metrics" in metrics.json()
