# tests/test_ai_engine_insights.py
# Only exercises the NEW, additive functions (generate_statistics,
# generate_dataframe_insights) — parse_command()/train_model()/etc. are
# unchanged and already have their own coverage (or lack thereof) elsewhere;
# this file doesn't touch them.
import pytest
import pandas as pd

from ai_engine import (
    generate_dataframe_insights,
    generate_missing_value_report,
    generate_outlier_report,
    generate_statistics,
)


def test_generate_statistics_returns_plain_descriptive_stats():
    df = pd.DataFrame({"revenue": [1000, 1100, 1184]})
    stats = generate_statistics(df, value_column="revenue")

    assert stats["count"] == 3.0
    assert stats["min"] == 1000.0
    assert stats["max"] == 1184.0
    assert stats["mean"] == pytest.approx(1094.6667, abs=1e-3)


def test_generate_statistics_raises_for_unknown_column():
    with pytest.raises(ValueError):
        generate_statistics(pd.DataFrame({"x": [1, 2]}), value_column="nope")


def test_generate_dataframe_insights_runs_statistics_then_trend_detection():
    df = pd.DataFrame({
        "month": ["January", "February", "March"],
        "revenue": [1000, 1100, 1184],
    })
    result = generate_dataframe_insights(df, value_column="revenue", period_column="month", label="Revenue")

    assert set(result.keys()) == {
        "statistics", "trend_insight", "outliers", "detected_kpis", "recommendations",
        "chart_recommendation", "data_quality", "executive_summary",
    }
    assert result["statistics"]["max"] == 1184.0
    assert result["trend_insight"]["trend"] == "Increasing"
    assert result["trend_insight"]["growth_percent"] == 18.4
    assert result["trend_insight"]["summary"] == "Revenue increased consistently throughout the period."
    assert result["recommendations"] == []  # healthy, increasing revenue -> nothing to flag
    assert result["chart_recommendation"]["chart"] == "line"
    assert set(result["chart_recommendation"]) == {
        "chart", "title", "subtitle", "x_axis", "y_axis", "series", "confidence", "reason"
    }


def test_generate_dataframe_insights_includes_detected_kpis():
    df = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr"],
        "revenue": [1000, 1100, 1200, 1300],
        "cost": [400, 420, 450, 470],
    })
    result = generate_dataframe_insights(df, value_column="revenue", period_column="month", label="Revenue")

    names = {f["name"] for f in result["detected_kpis"]}
    assert {"Revenue", "Profit", "Growth Rate"} <= names


def test_generate_dataframe_insights_chart_recommendation_uses_the_question():
    df = pd.DataFrame({"region": ["North", "South", "East", "West"], "revenue": [400, 300, 200, 100]})
    result = generate_dataframe_insights(
        df, value_column="revenue", period_column="region", label="Revenue",
        question="Compare revenue by region",
    )
    assert result["chart_recommendation"]["chart"] == "bar"
    assert set(result["chart_recommendation"]) == {
        "chart", "title", "subtitle", "x_axis", "y_axis", "series", "confidence", "reason"
    }


def test_generate_dataframe_insights_surfaces_recommendations_for_declining_revenue():
    df = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr"],
        "revenue": [1000, 900, 800, 700],
    })
    result = generate_dataframe_insights(df, value_column="revenue", period_column="month", label="Revenue")

    assert any(r["recommendation"] == "Improve marketing strategy" for r in result["recommendations"])


def test_generate_dataframe_insights_passes_through_caller_supplied_kpis():
    df = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "revenue": [1000, 1100, 1200]})
    result = generate_dataframe_insights(
        df, value_column="revenue", period_column="month", label="Revenue",
        kpis={"top_customer_revenue_share_percent": 70.0},
    )

    assert any(r["recommendation"] == "Diversify customer base" for r in result["recommendations"])


def test_generate_dataframe_insights_surfaces_recommendations_from_detected_outliers():
    df = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
        "revenue": [1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070],
        "amount": [10, 11, 9, 12, 10, 11, 9, 5000],
    })
    result = generate_dataframe_insights(df, value_column="revenue", period_column="month", label="Revenue")

    assert any(f["column"] == "amount" for f in result["outliers"])
    assert any(r["recommendation"] == "Investigate and clean outlier records" for r in result["recommendations"])


def test_generate_missing_value_report_and_outlier_report():
    df = pd.DataFrame({
        "email": ["a@x.com", None, None, None],
        "amount": [10, 11, 9, 500],
    })

    missing = generate_missing_value_report(df)
    assert missing["email"] == 75.0

    outliers = generate_outlier_report(df, columns=["amount"])
    amount_iqr = next(f for f in outliers if f["column"] == "amount" and f["method"] == "iqr")
    assert amount_iqr["outlier_count"] == 1
