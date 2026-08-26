from __future__ import annotations

from fastapi.testclient import TestClient

from insight_learning.api.app import create_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("INSIGHT_LEARNING_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("DATA_ANALYSIS_LLM_STATE_DIR", str(tmp_path / "state"))
    app = create_app()
    return TestClient(app)


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
            "intent": "filter",
            "query_features": {
                "predicate_count": 3,
                "logical_structure": "AND",
                "semantic_roles": ["boolean", "boolean", "numeric_measure"],
                "operators": ["equals_true", "equals_true", "less_than"],
            },
            "plan": {"tool_sequence": ["sql.filter"], "filters": [], "group_by": [], "metrics": []},
            "execution": {"success": True},
            "validation": {"success": True},
            "quality_score": 0.96,
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


def test_metrics_endpoint(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    assert "metrics" in metrics.json()

