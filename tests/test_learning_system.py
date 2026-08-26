from __future__ import annotations

import json

import pandas as pd

from agent.orchestrator import AgenticLearningOrchestrator
from learning.experience_store import LearningExperienceStore
from learning.models import LearningDecision
from learning.skill_registry import SkillRegistry


def _orchestrator(tmp_path):
    registry = SkillRegistry(state_path=tmp_path / "skills_state.json")
    store = LearningExperienceStore(root=tmp_path)
    return AgenticLearningOrchestrator(registry=registry, store=store)


def test_bootstrap_skills_include_the_learning_core(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    skill_ids = {spec.id for spec in orchestrator.registry.all()}

    assert "filter.multi_condition.v1" in skill_ids
    assert "filter.entity_search.v1" in skill_ids
    assert "clean.boolean_normalization.v1" in skill_ids
    assert "privacy.safe_local_execution.v1" in skill_ids


def test_multi_condition_filter_planner_keeps_every_requested_condition(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    df = pd.DataFrame(
        {
            "Restaurant Name": ["Pizza Hut", "Domino's Pizza"],
            "Online Delivery": [True, False],
            "Table Booking": [True, True],
            "Aggregate rating": [4.2, 3.1],
        }
    )

    decision = orchestrator.plan(
        "show restaurants having online delivery and table booking above 3.5 rating",
        df=df,
        available_columns=list(df.columns),
    )

    assert decision.route == "sql"
    assert decision.skill_id == "filter.multi_condition.v1"
    assert decision.plan is not None
    assert len(decision.plan["filters"]) == 3
    columns = {predicate["column"] for predicate in decision.plan["filters"]}
    assert {"Online Delivery", "Table Booking", "Aggregate rating"} <= columns


def test_planner_reuses_similar_experience_memory(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    df = pd.DataFrame({"Restaurant Name": ["Pizza Hut", "KFC"], "Revenue": [100, 200]})

    decision = orchestrator.plan("show Pizza Hut", df=df, available_columns=list(df.columns))
    assert decision.route == "sql"

    orchestrator.record_result(
        user_text="show Pizza Hut",
        decision=decision,
        df=df,
        available_columns=list(df.columns),
        result_summary={"result_kind": "table", "row_count": 1, "column_count": 2, "columns": ["Restaurant Name", "Revenue"]},
        success=True,
    )

    next_decision = orchestrator.plan("show Pizza Hut", df=df, available_columns=list(df.columns))
    assert next_decision.retrieval_trace["experience_count"] >= 1
    assert next_decision.skill_id == "filter.entity_search.v1"


def test_privacy_safe_migration_and_candidate_promotion(tmp_path):
    legacy_payload = {
        "query_text": "show John Smith",
        "normalized_query": "show john smith",
        "schema_signature": "legacy-schema",
        "route": "sql",
        "skill_id": "filter.entity_search.v1",
        "confidence": 0.95,
        "success": True,
        "score": 0.92,
        "plan_hash": "abc123",
        "plan_summary": {"route": "sql", "plan_keys": ["filters"]},
        "result_summary": {"result_kind": "table", "row_count": 1, "column_count": 2, "columns": ["Name", "Email"]},
        "created_at": "2026-08-26T00:00:00+00:00",
        "version": 1,
    }
    (tmp_path / "experiences.jsonl").write_text(json.dumps(legacy_payload) + "\n", encoding="utf-8")

    orchestrator = _orchestrator(tmp_path)
    migrated_log = (tmp_path / "experiences.jsonl").read_text(encoding="utf-8")
    assert "query_text" not in migrated_log
    assert "normalized_query" not in migrated_log
    assert "John Smith" not in migrated_log

    recent = orchestrator.store.load_recent(limit=1)
    assert recent
    assert "query_text" not in recent[0]
    assert "normalized_query" not in recent[0]

    df = pd.DataFrame({"Restaurant Name": ["Pizza Hut"], "Revenue": [100]})
    decision = orchestrator.plan("show Pizza Hut", df=df, available_columns=list(df.columns))
    assert isinstance(decision, LearningDecision)

    for _ in range(3):
        orchestrator.record_result(
            user_text="show Pizza Hut",
            decision=decision,
            df=df,
            available_columns=list(df.columns),
            result_summary={"result_kind": "table", "row_count": 1, "column_count": 2, "columns": ["Restaurant Name", "Revenue"]},
            success=True,
        )

    experience_log = (tmp_path / "experiences.jsonl").read_text(encoding="utf-8")
    assert '"rows"' not in experience_log
    assert '"sheet_name"' not in experience_log

    strategies = orchestrator.store.load_candidate_strategies(limit=10)
    assert strategies
    assert any(strategy["state"] == "promoted" for strategy in strategies)

    registry = SkillRegistry(state_path=tmp_path / "skills_state.json")
    assert any(spec.id.startswith("learned.") for spec in registry.all())


def test_failure_learning_creates_safe_lesson(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    df = pd.DataFrame({"Restaurant Name": ["Pizza Hut"], "Revenue": [100]})
    decision = orchestrator.plan("show Pizza Hut", df=df, available_columns=list(df.columns))

    orchestrator.record_result(
        user_text="show Pizza Hut",
        decision=decision,
        df=df,
        available_columns=list(df.columns),
        result_summary={"result_kind": "error"},
        success=False,
        failure_reason="forced failure",
    )

    lessons = orchestrator.store.load_failure_lessons(limit=10)
    assert lessons
    assert lessons[0]["intent"] in {"filter", "unknown"}

    retried = orchestrator.plan("show Pizza Hut", df=df, available_columns=list(df.columns))
    assert retried.retrieval_trace["failure_lesson_count"] >= 1
