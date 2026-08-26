from __future__ import annotations

import json

import pandas as pd

from agent.orchestrator import AgenticLearningOrchestrator
from learning.experience_store import LearningExperienceStore
from learning.models import FailureLesson, LearningDecision
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


def test_planner_reuses_similar_experience_memory_and_changes_plan_source(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    df = pd.DataFrame({"Restaurant Name": ["Pizza Hut", "KFC"], "Revenue": [100, 200]})

    decision = orchestrator.plan("show Pizza Hut", df=df, available_columns=list(df.columns))
    assert decision.route == "sql"
    assert decision.plan_source == "bootstrap_skill"

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
    assert next_decision.plan_source == "experience_transfer"


def test_domain_transfer_reuses_learned_template_across_datasets(tmp_path):
    orchestrator = _orchestrator(tmp_path)

    customers = pd.DataFrame(
        {
            "Customer": ["A", "B", "C"],
            "Active": [True, True, False],
            "Verified": [True, False, True],
            "Score": [91, 84, 72],
        }
    )
    customer_decision = orchestrator.plan(
        "show customers with active and verified and score above 80",
        df=customers,
        available_columns=list(customers.columns),
    )
    assert customer_decision.route == "sql"
    orchestrator.record_result(
        user_text="show customers with active and verified and score above 80",
        decision=customer_decision,
        df=customers,
        available_columns=list(customers.columns),
        result_summary={"result_kind": "table", "row_count": 1, "column_count": 4},
        success=True,
    )

    suppliers = pd.DataFrame(
        {
            "Supplier": ["X", "Y", "Z"],
            "Approved": [True, False, True],
            "Express": [True, True, False],
            "Defect Rate": [0.12, 0.31, 0.08],
        }
    )
    supplier_decision = orchestrator.plan(
        "approved suppliers with express and defect rate below 0.2",
        df=suppliers,
        available_columns=list(suppliers.columns),
    )

    assert supplier_decision.route == "sql"
    assert supplier_decision.plan_source == "experience_transfer"
    assert supplier_decision.plan_template_id is not None
    assert supplier_decision.plan is not None
    assert len(supplier_decision.plan["filters"]) == 3
    assert {item["column"] for item in supplier_decision.plan["filters"]} == {"Approved", "Express", "Defect Rate"}


def test_multi_step_template_transfers_across_domains(tmp_path):
    orchestrator = _orchestrator(tmp_path)

    category_df = pd.DataFrame(
        {
            "Category": ["A", "B", "C"],
            "Revenue": [10, 30, 20],
        }
    )
    first_decision = orchestrator.plan(
        "top categories by average revenue",
        df=category_df,
        available_columns=list(category_df.columns),
    )
    assert first_decision.route == "sql"
    orchestrator.record_result(
        user_text="top categories by average revenue",
        decision=first_decision,
        df=category_df,
        available_columns=list(category_df.columns),
        result_summary={"result_kind": "table", "row_count": 3, "column_count": 2},
        success=True,
    )

    region_df = pd.DataFrame(
        {
            "Region": ["North", "South", "West"],
            "Margin": [0.2, 0.6, 0.4],
        }
    )
    second_decision = orchestrator.plan(
        "top regions by average margin",
        df=region_df,
        available_columns=list(region_df.columns),
    )

    assert second_decision.route == "sql"
    assert second_decision.plan_source == "experience_transfer"
    assert second_decision.plan is not None
    assert second_decision.plan.get("group_by") == ["Region"]
    assert second_decision.plan.get("metrics")
    assert second_decision.plan.get("order_by")


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


def test_failure_learning_creates_safe_lesson_and_guides_next_plan(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    df = pd.DataFrame({"Restaurant Name": ["Pizza Hut"], "Revenue": [100]})

    lesson = FailureLesson(
        lesson_id="lesson.filter.001",
        intent="filter",
        failure_signature="failure.sig.001",
        condition_structure="AND",
        lesson="Preserve every explicit predicate and validate the output before execution.",
        severity="high",
        semantic_roles=["restaurant_entity", "boolean_capability", "rating_metric"],
        operators=["greater_than"],
        tool_sequence=["sql.filter"],
        occurrence_count=1,
        average_quality=0.2,
        strict_predicate_parity=True,
        required_roles=["restaurant_entity", "boolean_capability", "rating_metric"],
        required_operators=["greater_than"],
        required_tool_sequence=["sql.filter"],
    )
    orchestrator.store.append_failure_lesson(lesson)

    decision = orchestrator.plan(
        "show restaurants having online delivery and table booking above 3.5 rating",
        df=df.assign(**{"Online Delivery": [True], "Table Booking": [True], "Aggregate rating": [4.8]}),
        available_columns=["Restaurant Name", "Online Delivery", "Table Booking", "Aggregate rating"],
    )

    assert decision.retrieval_trace["failure_lesson_count"] >= 1
    assert any("failure lesson" in note or "predicate" in note for note in decision.validation_notes)
