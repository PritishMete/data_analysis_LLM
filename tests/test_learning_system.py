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


def test_entity_search_planner_is_generic_and_not_brand_hardcoded(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    df = pd.DataFrame({"Restaurant Name": ["Pizza Hut", "KFC"], "Revenue": [100, 200]})

    decision = orchestrator.plan("show Pizza Hut", df=df, available_columns=list(df.columns))

    assert decision.route == "sql"
    assert decision.skill_id == "filter.entity_search.v1"
    assert decision.plan is not None
    assert decision.plan["filters"][0]["column"] == "Restaurant Name"
    assert decision.plan["filters"][0]["value"] == "Pizza Hut"


def test_experience_logging_is_privacy_safe_and_persists_skill_promotion(tmp_path):
    orchestrator = _orchestrator(tmp_path)
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
    assert "100" not in experience_log
    assert '"sheet_name"' not in experience_log

    registry = SkillRegistry(state_path=tmp_path / "skills_state.json")
    state = registry.state_for("filter.entity_search.v1")
    assert state.success_count >= 3
    assert state.state == "promoted"
    assert state.confidence >= 0.9

    store = LearningExperienceStore(root=tmp_path)
    recent = store.load_recent(limit=1)
    assert recent
    assert recent[0]["skill_id"] == "filter.entity_search.v1"

    summary = store.load_summary()
    assert summary["skills"]["filter.entity_search.v1"]["state"] == "promoted"
