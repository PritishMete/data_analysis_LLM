# test_transformation_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Unit + integration tests for the centralized Transformation Engine:
#   common/transformations/transformation_registry.py
#   common/transformations/transformation_engine.py
#   common/transformations/transformation_history.py
#   common/transformations/transformation_result.py
#   common/transformations/adapters/*.py
#
# Usage:
#   python test_transformation_engine.py
# ─────────────────────────────────────────────────────────────────────────────

import time

import pandas as pd

from common.transformations import (
    TransformationEngine,
    TransformationHistory,
    all_transformations,
    detect_transformation,
    get_transformation,
    transformation_names,
)


# ── Registry ─────────────────────────────────────────────────────────────
def test_registry_has_all_built_ins():
    expected = {
        "range_binning", "rename_columns", "drop_columns", "fill_missing",
        "remove_duplicates", "merge_columns", "split_column",
        "type_conversion", "date_features",
    }
    assert expected.issubset(set(transformation_names()))
    print("test_registry_has_all_built_ins: PASS")


def test_registry_get_returns_correct_instance():
    t = get_transformation("range_binning")
    assert t is not None
    assert t.name == "range_binning"
    assert t.display_name == "Range Binning"
    print("test_registry_get_returns_correct_instance: PASS")


def test_registry_get_unknown_returns_none():
    assert get_transformation("does_not_exist") is None
    print("test_registry_get_unknown_returns_none: PASS")


def test_registry_all_transformations_is_a_copy():
    snapshot = all_transformations()
    snapshot["fake"] = object()
    assert "fake" not in transformation_names()
    print("test_registry_all_transformations_is_a_copy: PASS")


def test_registry_detect_transformation_routes_correctly():
    df = pd.DataFrame({"Rating": [0.1, 1.2, 2.3], "Notes": [None, "x", None]})
    located = detect_transformation("Create rating ranges 0-1,1-2,2-3", df)
    assert located is not None
    transformation, detection = located
    assert transformation.name == "range_binning"
    assert detection["params"]["source_column"] == "Rating"
    print("test_registry_detect_transformation_routes_correctly: PASS")


def test_registry_detect_transformation_returns_none_for_unrelated_text():
    df = pd.DataFrame({"Revenue": [1, 2, 3]})
    assert detect_transformation("What is the total revenue?", df) is None
    print("test_registry_detect_transformation_returns_none_for_unrelated_text: PASS")


# ── Engine: locate/validate/apply ───────────────────────────────────────
def test_engine_run_by_explicit_name_and_params():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8], "Sales": [10, 20, 15]})
    engine = TransformationEngine()
    result = engine.run(
        df, transformation_name="range_binning",
        params={"source_column": "Rating", "ranges": ["0-1", "1-2", "2-3"]},
    )
    assert result.success is True
    assert "Rating_Range" in result.dataframe.columns
    print("test_engine_run_by_explicit_name_and_params: PASS")


def test_engine_run_by_natural_language_query():
    df = pd.DataFrame({"Age": [5, 18, 19, 60, 61], "Income": [0, 100, 200, 300, 400]})
    engine = TransformationEngine()
    result = engine.run(df, query="Group age into 0-18,19-30,31-45,46-60,60+")
    assert result.success is True
    assert list(result.dataframe["Age_Range"]) == ["0-18", "0-18", "19-30", "46-60", "60+"]
    print("test_engine_run_by_natural_language_query: PASS")


def test_engine_run_unlocatable_transformation_fails_cleanly():
    df = pd.DataFrame({"A": [1, 2, 3]})
    engine = TransformationEngine()
    result = engine.run(df, query="please make me a sandwich")
    assert result.success is False
    assert result.error
    print("test_engine_run_unlocatable_transformation_fails_cleanly: PASS")


def test_engine_run_validation_failure_is_not_a_crash():
    df = pd.DataFrame({"Name": ["a", "b", "c"]})
    engine = TransformationEngine()
    result = engine.run(
        df, transformation_name="range_binning",
        params={"source_column": "Name", "ranges": ["0-1"]},
    )
    assert result.success is False
    assert "not numeric" in result.error
    print("test_engine_run_validation_failure_is_not_a_crash: PASS")


def test_engine_unknown_transformation_name_fails_cleanly():
    df = pd.DataFrame({"A": [1, 2, 3]})
    engine = TransformationEngine()
    result = engine.run(df, transformation_name="not_a_real_transformation")
    assert result.success is False
    print("test_engine_unknown_transformation_name_fails_cleanly: PASS")


# ── Preview mode ─────────────────────────────────────────────────────────
def test_engine_preview_does_not_mutate_or_commit():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8]})
    original_columns = list(df.columns)
    engine = TransformationEngine()
    result = engine.preview(df, query="Create rating range 0-1,1-2,2-3")
    assert result.success is True
    assert result.preview["affected_columns"] == ["Rating", "Rating_Range"]
    assert list(df.columns) == original_columns  # original untouched
    print("test_engine_preview_does_not_mutate_or_commit: PASS")


def test_engine_preview_only_includes_changed_columns():
    df = pd.DataFrame({"Rating": [0.4, 1.6], "Untouched": ["a", "b"]})
    engine = TransformationEngine()
    result = engine.preview(df, query="Create rating range 0-1,1-2")
    assert "Untouched" not in result.preview["affected_columns"]
    print("test_engine_preview_only_includes_changed_columns: PASS")


# ── Undo / Redo / History ────────────────────────────────────────────────
def test_history_records_each_transformation():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8]})
    engine = TransformationEngine()
    history = TransformationHistory()
    engine.run(df, query="Create rating range 0-1,1-2,2-3", history=history)
    assert len(history) == 1
    entries = history.list()
    assert entries[0]["transformation_name"] == "range_binning"
    assert entries[0]["target_columns"] == ["Rating_Range"]
    print("test_history_records_each_transformation: PASS")


def test_history_undo_restores_previous_dataframe():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8]})
    engine = TransformationEngine()
    history = TransformationHistory()
    result = engine.run(df, query="Create rating range 0-1,1-2,2-3", history=history)
    assert "Rating_Range" in result.dataframe.columns

    undo_result = engine.undo(history)
    assert undo_result.success is True
    assert "Rating_Range" not in undo_result.dataframe.columns
    assert not history.can_undo()
    print("test_history_undo_restores_previous_dataframe: PASS")


def test_history_redo_reapplies_undone_transformation():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8]})
    engine = TransformationEngine()
    history = TransformationHistory()
    engine.run(df, query="Create rating range 0-1,1-2,2-3", history=history)
    engine.undo(history)
    redo_result = engine.redo(history)
    assert redo_result.success is True
    assert "Rating_Range" in redo_result.dataframe.columns
    assert history.can_undo()
    assert not history.can_redo()
    print("test_history_redo_reapplies_undone_transformation: PASS")


def test_history_new_action_clears_redo_stack():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8], "Age": [10, 20, 30]})
    engine = TransformationEngine()
    history = TransformationHistory()
    r1 = engine.run(df, query="Create rating range 0-1,1-2,2-3", history=history)
    engine.undo(history)
    assert history.can_redo()
    engine.run(r1.dataframe.drop(columns=["Rating_Range"]), query="Bucket age 0-18,19-100", history=history)
    assert not history.can_redo()
    print("test_history_new_action_clears_redo_stack: PASS")


def test_history_undo_with_empty_history_fails_cleanly():
    engine = TransformationEngine()
    history = TransformationHistory()
    result = engine.undo(history)
    assert result.success is False
    print("test_history_undo_with_empty_history_fails_cleanly: PASS")


def test_history_replay_reapplies_sequence_on_fresh_dataframe():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8]})
    engine = TransformationEngine()
    history = TransformationHistory()
    engine.run(df, query="Create rating range 0-1,1-2,2-3", history=history)

    fresh_df = pd.DataFrame({"Rating": [0.9, 2.9]})
    replayed = history.replay(engine, fresh_df)
    assert list(replayed["Rating_Range"]) == ["0-1", "2-3"]
    print("test_history_replay_reapplies_sequence_on_fresh_dataframe: PASS")


def test_history_multi_step_chain_tracks_all_entries():
    df = pd.DataFrame({"Rating": [0.4, 1.6], "Sales": [10, 20], "Notes": [None, "x"]})
    engine = TransformationEngine()
    history = TransformationHistory()

    r1 = engine.run(df, query="Create rating range 0-1,1-2", history=history)
    r2 = engine.run(r1.dataframe, query="rename Sales to Revenue", history=history)
    r3 = engine.run(r2.dataframe, query="drop column Notes", history=history)

    assert r3.success is True
    assert len(history) == 3
    assert list(r3.dataframe.columns) == ["Rating", "Revenue", "Rating_Range"]
    print("test_history_multi_step_chain_tracks_all_entries: PASS")


# ── Pipeline integration (schema/stats/kpis/charts/executive summary) ───
def test_pipeline_schema_diff_reports_added_column():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8]})
    engine = TransformationEngine()
    result = engine.run(df, query="Create rating range 0-1,1-2,2-3")
    assert result.updated_schema["added_columns"] == ["Rating_Range"]
    assert result.updated_schema["removed_columns"] == []
    print("test_pipeline_schema_diff_reports_added_column: PASS")


def test_pipeline_refreshes_statistics_kpis_charts_and_executive_summary():
    df = pd.DataFrame({
        "Rating": [0.4, 1.6, 2.8, 3.9, 4.8, 0.8, 1.9, 2.4, 4.2, 3.1, 2.2, 1.1],
        "Sales": [100, 200, 150, 300, 250, 120, 180, 90, 310, 220, 175, 140],
    })
    engine = TransformationEngine()
    result = engine.run(
        df, query="Create column for rating range 0-1,1-2,2-3,3-4,4-5", value_column="Sales",
    )
    assert result.updated_statistics.get("mean") is not None
    assert isinstance(result.updated_kpis, list)
    assert result.updated_charts.get("chart") == "column"
    exec_summary = result.updated_ai_report.get("executive_summary", {})
    assert "derived_columns" in exec_summary
    assert exec_summary["derived_columns"][0]["new_column"] == "Rating_Range"
    print("test_pipeline_refreshes_statistics_kpis_charts_and_executive_summary: PASS")


def test_pipeline_transformations_applied_metadata_present():
    """AI-report-facing 'Transformations Applied' data: engine surfaces
    everything needed (name, method, source/new columns, rows_modified via
    history) to render that section — see main.py's /transform/apply."""
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8]})
    engine = TransformationEngine()
    history = TransformationHistory()
    result = engine.run(df, query="Create rating range 0-1,1-2,2-3", history=history)
    entry = history.list()[0]
    assert entry["transformation_name"] == "range_binning"
    assert entry["metadata"]["new_column"] == "Rating_Range"
    assert entry["rows_modified"] >= 0
    assert result.execution_time >= 0
    print("test_pipeline_transformations_applied_metadata_present: PASS")


# ── Performance: no duplicate dataframe copies / no dataset rescans ─────
def test_performance_engine_does_not_rescan_or_reread_dataset():
    """The engine receives an in-memory DataFrame and returns an in-memory
    DataFrame — it never touches disk/network. This proxies that guarantee
    by checking total wall time stays fast for a moderately sized frame."""
    df = pd.DataFrame({
        "Rating": [i % 5 + 0.1 for i in range(5000)],
        "Sales": list(range(5000)),
    })
    engine = TransformationEngine()
    start = time.perf_counter()
    result = engine.run(df, query="Create rating range 0-1,1-2,2-3,3-4,4-5,5-6")
    elapsed = time.perf_counter() - start
    assert result.success is True
    assert elapsed < 5.0  # generous ceiling; catches accidental O(n^2)/rescans
    print(f"test_performance_engine_does_not_rescan_or_reread_dataset: PASS ({elapsed:.3f}s)")


def test_performance_original_dataframe_is_never_mutated():
    """apply() must always return a NEW dataframe — critical so the engine
    never has to re-fetch/rescan the source to get an undo snapshot."""
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8]})
    original_columns = list(df.columns)
    engine = TransformationEngine()
    engine.run(df, query="Create rating range 0-1,1-2,2-3")
    assert list(df.columns) == original_columns
    print("test_performance_original_dataframe_is_never_mutated: PASS")


if __name__ == "__main__":
    test_registry_has_all_built_ins()
    test_registry_get_returns_correct_instance()
    test_registry_get_unknown_returns_none()
    test_registry_all_transformations_is_a_copy()
    test_registry_detect_transformation_routes_correctly()
    test_registry_detect_transformation_returns_none_for_unrelated_text()

    test_engine_run_by_explicit_name_and_params()
    test_engine_run_by_natural_language_query()
    test_engine_run_unlocatable_transformation_fails_cleanly()
    test_engine_run_validation_failure_is_not_a_crash()
    test_engine_unknown_transformation_name_fails_cleanly()

    test_engine_preview_does_not_mutate_or_commit()
    test_engine_preview_only_includes_changed_columns()

    test_history_records_each_transformation()
    test_history_undo_restores_previous_dataframe()
    test_history_redo_reapplies_undone_transformation()
    test_history_new_action_clears_redo_stack()
    test_history_undo_with_empty_history_fails_cleanly()
    test_history_replay_reapplies_sequence_on_fresh_dataframe()
    test_history_multi_step_chain_tracks_all_entries()

    test_pipeline_schema_diff_reports_added_column()
    test_pipeline_refreshes_statistics_kpis_charts_and_executive_summary()
    test_pipeline_transformations_applied_metadata_present()

    test_performance_engine_does_not_rescan_or_reread_dataset()
    test_performance_original_dataframe_is_never_mutated()

    print("\nAll transformation engine tests passed.")
