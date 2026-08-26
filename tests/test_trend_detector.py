# tests/test_trend_detector.py
import pytest
import pandas as pd

from common.insights.trend_detector import detect_trend
from common.insights.service import InsightsService
from common.insights.schemas import TrendInsight


def test_matches_the_documented_example_exactly():
    df = pd.DataFrame({
        "month": ["January", "February", "March"],
        "revenue": [1000, 1100, 1184],
    })
    result = detect_trend(df, value_column="revenue", period_column="month", label="Revenue")

    assert result["trend"] == "Increasing"
    assert result["growth_percent"] == 18.4
    assert result["highest_period"] == "March"
    assert result["lowest_period"] == "January"
    assert result["summary"] == "Revenue increased consistently throughout the period."


def test_detects_consistent_decrease():
    df = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr"],
        "revenue": [1000, 900, 800, 700],
    })
    result = detect_trend(df, value_column="revenue", period_column="month", label="Revenue")

    assert result["trend"] == "Decreasing"
    assert result["growth_percent"] == -30.0
    assert result["decline_percent"] == 30.0
    assert result["highest_period"] == "Jan"
    assert result["lowest_period"] == "Apr"
    assert result["consecutive_decrease"] == 3
    assert result["consecutive_increase"] == 0
    assert "decreased consistently" in result["summary"]


def test_detects_stable_trend_for_small_fluctuation():
    df = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr"],
        "revenue": [1000, 1005, 995, 1002],
    })
    result = detect_trend(df, value_column="revenue", period_column="month", label="Revenue")

    assert result["trend"] == "Stable"
    assert result["decline_percent"] == 0.0
    assert "remained stable" in result["summary"]


def test_volatile_but_net_increasing_is_not_called_consistent():
    df = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "revenue": [1000, 1200, 1100, 1300, 1250],
    })
    result = detect_trend(df, value_column="revenue", period_column="month", label="Revenue")

    assert result["trend"] == "Increasing"
    assert result["consecutive_increase"] < 4  # not every step increased
    assert "with some fluctuation" in result["summary"]


def test_uses_dataframe_index_when_no_period_column_given():
    df = pd.DataFrame({"revenue": [10, 20, 30]})
    result = detect_trend(df, value_column="revenue")

    assert result["trend"] == "Increasing"
    assert result["highest_period"] == "2"
    assert result["lowest_period"] == "0"


def test_formats_datetime_period_column_as_month_name():
    dates = pd.date_range("2024-01-01", periods=3, freq="MS")
    df = pd.DataFrame({"date": dates, "revenue": [500, 600, 720]})
    result = detect_trend(df, value_column="revenue", period_column="date", label="Revenue")

    assert result["highest_period"] == "March"
    assert result["lowest_period"] == "January"


def test_single_row_is_stable_with_no_growth_percent_of_zero():
    df = pd.DataFrame({"revenue": [42]})
    result = detect_trend(df, value_column="revenue")

    assert result["trend"] == "Stable"
    assert result["growth_percent"] == 0.0
    assert result["highest_period"] == result["lowest_period"]


def test_growth_percent_is_none_when_first_value_is_zero():
    df = pd.DataFrame({"revenue": [0, 10, 20]})
    result = detect_trend(df, value_column="revenue")

    assert result["growth_percent"] is None
    assert result["trend"] == "Increasing"  # still determinable via slope


def test_drops_missing_values_before_analysis():
    df = pd.DataFrame({"revenue": [10.0, None, 30.0]})
    result = detect_trend(df, value_column="revenue")

    assert result["trend"] == "Increasing"
    assert result["growth_percent"] == 200.0


def test_raises_for_unknown_value_column():
    with pytest.raises(ValueError):
        detect_trend(pd.DataFrame({"x": [1, 2]}), value_column="nope")


def test_raises_for_unknown_period_column():
    with pytest.raises(ValueError):
        detect_trend(pd.DataFrame({"x": [1, 2]}), value_column="x", period_column="nope")


def test_raises_for_empty_dataframe():
    with pytest.raises(ValueError):
        detect_trend(pd.DataFrame({"x": []}), value_column="x")


def test_insights_service_returns_validated_trend_insight():
    df = pd.DataFrame({
        "month": ["January", "February", "March"],
        "revenue": [1000, 1100, 1184],
    })
    insight = InsightsService().detect_trend(df, value_column="revenue", period_column="month", label="Revenue")

    assert isinstance(insight, TrendInsight)
    assert insight.trend == "Increasing"
    assert insight.growth_percent == 18.4
    assert insight.summary == "Revenue increased consistently throughout the period."
