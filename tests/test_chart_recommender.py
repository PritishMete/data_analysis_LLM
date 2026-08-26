import pandas as pd

from common.insights.chart_recommender import ChartRecommenderEngine, TimeSeriesRule, recommend_chart
from common.insights.trend_detector import detect_trend

EXPECTED_KEYS = {"chart", "title", "subtitle", "x_axis", "y_axis", "series", "confidence", "reason"}


def assert_metadata(result):
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["chart"] in {"line", "bar", "horizontal_bar", "pie", "scatter", "histogram"}
    assert isinstance(result["title"], str) and result["title"]
    assert isinstance(result["subtitle"], str) and result["subtitle"]
    assert isinstance(result["x_axis"], str) and result["x_axis"]
    assert isinstance(result["y_axis"], str) and result["y_axis"]
    assert isinstance(result["series"], list)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["reason"], str) and result["reason"]


def test_time_series_returns_flutter_metadata():
    df = pd.DataFrame({"month": ["Jan", "Feb", "Mar", "Apr"], "revenue": [1000, 1100, 1200, 1300]})
    trend = detect_trend(df, value_column="revenue", period_column="month", label="Revenue")
    result = recommend_chart(question="How has revenue changed over time?", df=df, trend=trend)
    assert_metadata(result)
    assert result["chart"] == "line"
    assert result["title"] == "Revenue by Month"
    assert result["x_axis"] == "Month"
    assert result["y_axis"] == "Revenue"
    assert result["series"][0]["field"] == "revenue"
    assert result["reason"]


def test_time_series_fires_from_trend_alone_without_keywords():
    df = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "revenue": [1000, 1100, 1200]})
    trend = detect_trend(df, value_column="revenue", period_column="month", label="Revenue")
    result = recommend_chart(question="what happened", df=df, trend=trend)
    assert_metadata(result)
    assert result["chart"] == "line"


def test_correlation_recommends_scatter_metadata():
    df = pd.DataFrame({"price": [10, 20, 30, 40, 50], "sales": [100, 90, 70, 50, 30]})
    result = recommend_chart(question="What is the correlation between price and sales?", df=df)
    assert_metadata(result)
    assert result["chart"] == "scatter"
    assert result["x_axis"] == "Price"
    assert result["y_axis"] == "Sales"


def test_correlation_requires_two_numeric_columns():
    df = pd.DataFrame({"price": [10, 20, 30, 40, 50]})
    result = recommend_chart(question="What is the correlation between price and sales?", df=df)
    assert result["chart"] != "scatter"


def test_distribution_recommends_histogram_metadata():
    df = pd.DataFrame({"order_amount": [10, 12, 15, 9, 11, 14, 300, 13, 10, 16, 12, 50, 11, 13, 9]})
    result = recommend_chart(question="What is the distribution of order amounts?", df=df)
    assert_metadata(result)
    assert result["chart"] == "histogram"


def test_part_to_whole_recommends_pie_metadata():
    df = pd.DataFrame({"region": ["North", "South", "East", "West"], "revenue": [400, 300, 200, 100]})
    result = recommend_chart(question="What percentage of revenue comes from each region?", df=df)
    assert_metadata(result)
    assert result["chart"] == "pie"


def test_many_categories_recommends_horizontal_bar_metadata():
    df = pd.DataFrame({"state": [f"S{i}" for i in range(30)], "sales": list(range(30))})
    result = recommend_chart(question="Compare sales across all states", df=df)
    assert_metadata(result)
    assert result["chart"] == "horizontal_bar"


def test_category_comparison_recommends_bar_metadata():
    df = pd.DataFrame({"region": ["North", "South", "East", "West"], "revenue": [400, 300, 200, 100]})
    result = recommend_chart(question="Compare revenue by region", df=df)
    assert_metadata(result)
    assert result["chart"] == "bar"


def test_falls_back_to_bar_metadata_when_no_signal_present():
    df = pd.DataFrame({"flag": [0, 1] * 5})
    result = recommend_chart(question="Tell me something interesting", df=df)
    assert_metadata(result)
    assert result["chart"] == "bar"
    assert result["confidence"] == 0.4


def test_works_with_no_arguments_at_all():
    result = recommend_chart()
    assert_metadata(result)
    assert result["chart"] == "bar"


def test_confidence_is_always_within_bounds():
    df = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "revenue": [1000, 1100, 1200]})
    trend = detect_trend(df, value_column="revenue", period_column="month", label="Revenue")
    result = recommend_chart(question="revenue over time trend", df=df, trend=trend)
    assert 0.0 <= result["confidence"] <= 1.0


def test_never_returns_a_rendered_chart_or_image_payload():
    result = recommend_chart(question="revenue over time", df=pd.DataFrame({"x": [1, 2, 3]}))
    for forbidden in ("image", "figure", "svg", "png", "base64", "plot"):
        assert forbidden not in result


def test_engine_is_reusable_with_a_custom_rule_subset():
    engine = ChartRecommenderEngine(rules=[TimeSeriesRule()])
    df = pd.DataFrame({"region": ["North", "South", "East", "West"], "revenue": [400, 300, 200, 100]})
    result = engine.recommend(question="Compare revenue by region", df=df)
    assert_metadata(result)
    assert result["chart"] == "bar"
    assert result["confidence"] == 0.4


def test_engine_instance_is_stateless_across_calls():
    engine = ChartRecommenderEngine()
    df = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "revenue": [1000, 1100, 1200]})
    trend = detect_trend(df, value_column="revenue", period_column="month", label="Revenue")
    first = engine.recommend(question="revenue over time", df=df, trend=trend)
    second = engine.recommend()
    assert first["chart"] == "line"
    assert second["chart"] == "bar"


def test_flutter_contract_has_no_legacy_recommended_chart_field():
    result = recommend_chart(question="revenue over time", df=pd.DataFrame({"month": ["Jan", "Feb"], "revenue": [100, 120]}))
    assert "recommended_chart" not in result
