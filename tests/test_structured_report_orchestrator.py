# tests/test_structured_report_orchestrator.py
# ─────────────────────────────────────────────────────────────────────────────
# Covers common/report/orchestrator.generate_structured_report_data() — the
# selection-driven, dependency-aware wiring around the EXISTING deterministic
# detectors (trend_detector, outlier_detector, kpi_detector,
# recommendation_engine, chart_recommender, executive_summary,
# statistics/service). No Gemini/LLM/ADK import happens anywhere in this
# module or in the orchestrator it tests, so these tests run with zero
# network calls and zero API keys required — which is itself proof that
# the structured analytics never depend on Gemini to be computed (test 12).
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import pytest

from common.report.orchestrator import generate_structured_report_data


@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "revenue": [1000, 1100, 1200, 1300, 1400, 1500],
        "cost": [400, 420, 450, 470, 500, 520],
        "customer_id": ["c1", "c2", "c3", "c1", "c2", "c1"],
    })


@pytest.fixture
def outlier_df():
    return pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
        "revenue": [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070],
        "amount": [10, 11, 9, 12, 10, 11, 9, 5000],
    })


# 1. statistics selection returns statistics
def test_statistics_selection_returns_statistics(sales_df):
    result = generate_structured_report_data(sales_df, ["statistics"], value_column="revenue")
    assert "statistics" in result
    assert result["statistics"]["max"] == 1500.0
    assert result["statistics"]["min"] == 1000.0


# 2. trend selection returns trend_insight
def test_trend_selection_returns_trend_insight(sales_df):
    result = generate_structured_report_data(
        sales_df, ["trend_detection"], value_column="revenue", period_column="month"
    )
    assert "trend_insight" in result
    assert result["trend_insight"]["trend"] == "Increasing"


# 3. outlier selection returns outliers
def test_outlier_selection_returns_outliers(outlier_df):
    result = generate_structured_report_data(outlier_df, ["outlier_detection"], value_column="revenue")
    assert "outliers" in result
    assert any(f["column"] == "amount" for f in result["outliers"])


# 4. KPI selection returns detected_kpis
def test_kpi_selection_returns_detected_kpis(sales_df):
    result = generate_structured_report_data(sales_df, ["kpi_analysis"], value_column="revenue")
    assert "detected_kpis" in result
    names = {k["name"] for k in result["detected_kpis"]}
    assert "Revenue" in names


# 5. recommendation selection returns recommendations
def test_recommendation_selection_returns_recommendations():
    declining_df = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr"],
        "revenue": [1000, 900, 800, 700],
    })
    result = generate_structured_report_data(
        declining_df, ["recommendations"], value_column="revenue", period_column="month"
    )
    assert "recommendations" in result
    assert any(r["recommendation"] == "Improve marketing strategy" for r in result["recommendations"])


# 6. chart selection returns chart_recommendation
def test_chart_selection_returns_chart_recommendation(sales_df):
    result = generate_structured_report_data(sales_df, ["chart_recommendation"], value_column="revenue")
    assert "chart_recommendation" in result
    assert result["chart_recommendation"]["chart"]


# 7. data quality selection returns data_quality
def test_data_quality_selection_returns_data_quality(sales_df):
    result = generate_structured_report_data(sales_df, ["data_quality"])
    assert "data_quality" in result
    assert "quality_score" in result["data_quality"]


# 8. executive summary selection returns executive_summary
def test_executive_summary_selection_returns_executive_summary(sales_df):
    result = generate_structured_report_data(
        sales_df, ["executive_summary"], value_column="revenue", period_column="month"
    )
    assert "executive_summary" in result
    assert "overall_health" in result["executive_summary"]
    assert "key_findings" in result["executive_summary"]


# 9. multiple selections return all requested sections
def test_multiple_selections_return_all_requested_sections(sales_df):
    result = generate_structured_report_data(
        sales_df, ["statistics", "outlier_detection"], value_column="revenue"
    )
    assert set(result.keys()) == {"statistics", "outliers"}


# 10. unselected sections are not incorrectly presented as selected
def test_unselected_sections_are_not_presented(sales_df):
    result = generate_structured_report_data(sales_df, ["trend_detection"], value_column="revenue", period_column="month")
    assert set(result.keys()) == {"trend_insight"}
    # dependencies used internally must not leak into the response
    assert "statistics" not in result
    assert "detected_kpis" not in result
    assert "recommendations" not in result
    assert "chart_recommendation" not in result
    assert "data_quality" not in result
    assert "executive_summary" not in result


def test_executive_summary_alone_only_exposes_executive_summary(sales_df):
    """executive_summary internally computes statistics/kpis/trend/
    recommendations/outliers/data_quality as dependencies, but none of
    those should be exposed unless independently selected."""
    result = generate_structured_report_data(
        sales_df, ["executive_summary"], value_column="revenue", period_column="month"
    )
    assert set(result.keys()) == {"executive_summary"}


def test_empty_selection_returns_empty_dict(sales_df):
    assert generate_structured_report_data(sales_df, [], value_column="revenue") == {}
    assert generate_structured_report_data(sales_df, None, value_column="revenue") == {}


# 11. chart_recommendation uses the canonical "chart" field
def test_chart_recommendation_uses_canonical_chart_field(sales_df):
    result = generate_structured_report_data(sales_df, ["chart_recommendation"], value_column="revenue")
    chart = result["chart_recommendation"]
    assert set(chart.keys()) == {
        "chart", "title", "subtitle", "x_axis", "y_axis", "series", "confidence", "reason",
    }
    # not "chart_type" or any other legacy/alternate name
    assert "chart_type" not in chart


# 12. no Gemini call is required to calculate the structured analytics
def test_no_gemini_or_llm_import_used_by_orchestrator():
    """The orchestrator module must not import google.adk/genai or any LLM
    client — the structured analytics are pure Python/pandas, computable
    with zero network calls or API keys."""
    import common.report.orchestrator as orchestrator_module

    source = orchestrator_module.__file__
    with open(source, "r", encoding="utf-8") as f:
        content = f.read()
    assert "google.adk" not in content
    assert "google.genai" not in content
    assert "genai" not in content.lower().replace("generate", "")  # no stray genai import


def test_structured_analytics_are_fully_computed_without_any_network_access(sales_df):
    """End-to-end sanity check: requesting every analysis type still only
    exercises pandas/statistics code (no monkeypatched network calls are
    needed for this to pass), proving Gemini plays no role in computing
    the values themselves."""
    result = generate_structured_report_data(
        sales_df,
        [
            "statistics", "trend_detection", "outlier_detection", "kpi_analysis",
            "recommendations", "chart_recommendation", "data_quality", "executive_summary",
        ],
        value_column="revenue",
        period_column="month",
    )
    assert set(result.keys()) == {
        "statistics", "trend_insight", "outliers", "detected_kpis", "recommendations",
        "chart_recommendation", "data_quality", "executive_summary",
    }


# ── value_column / period_column auto-detection ─────────────────────────────

def test_auto_detects_value_column_by_name_hint():
    df = pd.DataFrame({"id": [1, 2, 3], "revenue": [10, 20, 30]})
    result = generate_structured_report_data(df, ["statistics"])
    assert result["statistics"]["max"] == 30.0


def test_auto_detects_period_column_by_name_hint():
    df = pd.DataFrame({"order_date": ["Jan", "Feb", "Mar"], "revenue": [10, 20, 30]})
    result = generate_structured_report_data(df, ["trend_detection"])
    assert result["trend_insight"]["trend"] == "Increasing"
