# test_range_binning_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Integration tests for the range-binning PIPELINE wiring:
#   apply_range_binning() -> common/report/orchestrator.generate_structured_report_data()
#   -> common/insights/chart_recommender.py (DerivedCategoryRule)
#   -> common/insights/executive_summary.py ("Derived Columns" section)
#
# These exercise the SAME entry point main.py's /transform/range_binning and
# query_router.py's handle_smart_query() fast-path both call, so a pass here
# means both routes are getting a correctly refreshed statistics/KPI/trend/
# outlier/recommendation/chart/executive-summary bundle for the transformed
# dataframe — without a second, duplicate analytics pipeline.
#
# Usage:
#   python test_range_binning_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

from common.transformations.range_binning import apply_range_binning
from common.report.orchestrator import ALL_ANALYSIS_IDS, generate_structured_report_data


def _bin_and_report(df: pd.DataFrame, source_column: str, ranges: list[str], value_column: str):
    result = apply_range_binning(df, source_column, ranges)
    new_df = result["dataframe"]
    metadata = result["metadata"]
    derived_columns = [{
        "new_column": metadata["new_column"],
        "source_column": metadata["source_column"],
        "method": "Range Binning",
        "category_count": len(metadata["ranges"]),
    }]
    report = generate_structured_report_data(
        new_df,
        list(ALL_ANALYSIS_IDS),
        value_column=value_column,
        derived_column=metadata["new_column"],
        derived_source_column=metadata["source_column"],
        derived_columns=derived_columns,
    )
    return result, report


def test_new_column_present_in_transformed_dataframe():
    """Step 3/4: the transformed dataframe (not the original) is what schema
    and every downstream analysis must operate on."""
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8, 3.9, 4.8], "Sales": [10, 20, 15, 30, 25]})
    result, _ = _bin_and_report(df, "Rating", ["0-1", "1-2", "2-3", "3-4", "4-5"], "Sales")
    new_df = result["dataframe"]
    assert "Rating_Range" in new_df.columns
    assert len(new_df.columns) == len(df.columns) + 1
    # Original untouched (apply_range_binning never mutates its input).
    assert "Rating_Range" not in df.columns
    print("test_new_column_present_in_transformed_dataframe: PASS")


def test_statistics_recomputed_on_transformed_dataframe():
    """Step 5: statistics reflect the value column on the NEW dataframe."""
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8, 3.9, 4.8], "Sales": [10, 20, 15, 30, 25]})
    _, report = _bin_and_report(df, "Rating", ["0-1", "1-2", "2-3", "3-4", "4-5"], "Sales")
    assert report["statistics"]["mean"] == 20.0
    print("test_statistics_recomputed_on_transformed_dataframe: PASS")


def test_chart_recommendation_targets_new_column():
    """Steps 10: the binned column should drive the chart recommendation —
    'Rating_Range' with 5 buckets -> a 'column' chart, grounded in the
    derived column, not a generic guess."""
    df = pd.DataFrame({
        "Rating": [0.4, 1.6, 2.8, 3.9, 4.8, 0.8, 1.9, 2.4, 4.2, 3.1, 2.2, 1.1],
        "Sales": [100, 200, 150, 300, 250, 120, 180, 90, 310, 220, 175, 140],
    })
    _, report = _bin_and_report(df, "Rating", ["0-1", "1-2", "2-3", "3-4", "4-5"], "Sales")
    chart = report["chart_recommendation"]
    assert chart["chart"] == "column"
    assert chart["x_axis"] == "Rating Range"
    assert chart["y_axis"] == "Sales"  # not the binned source column itself
    assert chart["confidence"] >= 0.9
    print("test_chart_recommendation_targets_new_column: PASS")


def test_chart_recommendation_falls_back_to_horizontal_bar_for_many_buckets():
    """More than MODERATE_CARDINALITY_MAX buckets -> horizontal_bar, same
    convention ManyCategoriesRule already uses elsewhere."""
    df = pd.DataFrame({
        "Score": list(range(0, 100, 2)),
        "Value": list(range(50)),
    })
    ranges = [f"{i}-{i+9}" for i in range(0, 100, 10)]  # 10 buckets > MODERATE_CARDINALITY_MAX (8)
    _, report = _bin_and_report(df, "Score", ranges, "Value")
    assert report["chart_recommendation"]["chart"] == "horizontal_bar"
    print("test_chart_recommendation_falls_back_to_horizontal_bar_for_many_buckets: PASS")


def test_executive_summary_has_derived_columns_section():
    """Step 11: executive_summary output includes a 'Derived Columns'
    section describing what was created, by what method, with how many
    categories — and folds a matching line into key_findings."""
    df = pd.DataFrame({
        "Age": [5, 18, 19, 30, 31, 60, 61, 100],
        "Income": [0, 500, 20000, 45000, 60000, 80000, 30000, 90000],
    })
    _, report = _bin_and_report(
        df, "Age", ["0-18", "19-30", "31-45", "46-60", "60+"], "Income"
    )
    summary = report["executive_summary"]
    assert "derived_columns" in summary
    assert summary["derived_columns"] == [{
        "new_column": "Age_Range",
        "source_column": "Age",
        "method": "Range Binning",
        "category_count": 5,
    }]
    assert any("Age_Range" in finding for finding in summary["key_findings"])
    print("test_executive_summary_has_derived_columns_section: PASS")


def test_schema_selection_ids_unaffected_by_derived_columns():
    """Passing derived_column/derived_columns must not change which keys are
    exposed for a given selection — selection gating (existing behavior)
    still governs the response shape."""
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8], "Sales": [10, 20, 15]})
    result = apply_range_binning(df, "Rating", ["0-1", "1-2", "2-3"])
    new_df = result["dataframe"]
    report = generate_structured_report_data(
        new_df, ["statistics"], value_column="Sales",
        derived_column="Rating_Range", derived_columns=[{"new_column": "Rating_Range"}],
    )
    assert set(report.keys()) == {"statistics"}
    print("test_schema_selection_ids_unaffected_by_derived_columns: PASS")


if __name__ == "__main__":
    test_new_column_present_in_transformed_dataframe()
    test_statistics_recomputed_on_transformed_dataframe()
    test_chart_recommendation_targets_new_column()
    test_chart_recommendation_falls_back_to_horizontal_bar_for_many_buckets()
    test_executive_summary_has_derived_columns_section()
    test_schema_selection_ids_unaffected_by_derived_columns()
    print("\nAll range_binning pipeline integration tests passed.")
