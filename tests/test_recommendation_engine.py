# tests/test_recommendation_engine.py
from common.insights.recommendation_engine import (
    CustomerConcentrationRule,
    HighMissingValuesRule,
    OutlierQualityRule,
    ProfitMarginDeclineRule,
    Recommendation,
    RecommendationEngine,
    RevenueDeclineRule,
)


def _priorities(recs):
    return [r["priority"] for r in recs]


def _recommendations(recs):
    return {r["recommendation"] for r in recs}


def test_revenue_decline_triggers_marketing_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(trend={"revenue": {"trend": "Decreasing", "decline_percent": 20.0}})

    assert len(recs) == 1
    assert recs[0]["category"] == "Revenue"
    assert recs[0]["priority"] == "High"
    assert 0.0 < recs[0]["confidence"] <= 1.0
    assert recs[0]["impact"] == "High"
    assert recs[0]["recommendation"] == "Improve marketing strategy"
    assert "20.0%" in recs[0]["reason"]
    assert list(recs[0]) == ["category", "priority", "confidence", "impact", "recommendation", "reason"]


def test_increasing_revenue_does_not_trigger():
    engine = RecommendationEngine()
    recs = engine.generate(trend={"revenue": {"trend": "Increasing", "decline_percent": 0.0}})
    assert "Improve marketing strategy" not in _recommendations(recs)


def test_profit_margin_decline_via_trend():
    engine = RecommendationEngine()
    recs = engine.generate(trend={"profit_margin": {"trend": "Decreasing", "decline_percent": 12.0}})

    assert any(r["recommendation"] == "Review pricing strategy" and r["category"] == "Profitability" for r in recs)


def test_profit_margin_decline_via_kpi_fallback():
    engine = RecommendationEngine()
    recs = engine.generate(kpis={"profit_margin_change_percent": -6.5})

    assert len(recs) == 1
    assert recs[0]["recommendation"] == "Review pricing strategy"
    assert recs[0]["category"] == "Profitability"
    assert recs[0]["impact"] == recs[0]["priority"]
    assert "6.5%" in recs[0]["reason"]


def test_positive_profit_margin_change_does_not_trigger():
    engine = RecommendationEngine()
    recs = engine.generate(kpis={"profit_margin_change_percent": 3.0})
    assert recs == []


def test_named_revenue_trend_does_not_leak_into_profit_margin_rule():
    # Regression test: a single, explicitly-named "revenue" trend must not
    # be silently claimed by ProfitMarginDeclineRule just because it's the
    # only trend present.
    engine = RecommendationEngine()
    recs = engine.generate(trend={"revenue": {"trend": "Decreasing", "decline_percent": 30.0}})

    assert "Review pricing strategy" not in _recommendations(recs)
    assert "Improve marketing strategy" in _recommendations(recs)


def test_unnamed_raw_trend_dict_is_still_usable():
    engine = RecommendationEngine()
    raw_trend = {"trend": "Decreasing", "decline_percent": 20.0, "growth_percent": -20.0}
    recs = engine.generate(trend=raw_trend)
    assert "Improve marketing strategy" in _recommendations(recs)


def test_high_missing_values_triggers_data_quality_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(statistics={"missing_percentage": {"email": 55.0, "phone": 2.0}})

    assert len(recs) == 1
    assert recs[0]["recommendation"] == "Improve data quality"
    assert recs[0]["category"] == "Data Quality"
    assert recs[0]["impact"] == recs[0]["priority"]
    assert "email" in recs[0]["reason"]


def test_low_missing_values_do_not_trigger():
    engine = RecommendationEngine()
    recs = engine.generate(statistics={"missing_percentage": {"phone": 2.0}})
    assert recs == []


def test_customer_concentration_triggers_diversify_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(kpis={"top_customer_revenue_share_percent": 65.0})

    assert len(recs) == 1
    assert recs[0]["recommendation"] == "Diversify customer base"
    assert recs[0]["priority"] == "High"


def test_outlier_rule_triggers_investigate_recommendation():
    engine = RecommendationEngine()
    recs = engine.generate(outliers={"amount": {"outlier_count": 5, "outlier_percentage": 30.0}})

    assert len(recs) == 1
    assert recs[0]["recommendation"] == "Investigate and clean outlier records"
    assert recs[0]["priority"] == "High"


def test_no_inputs_produce_no_recommendations():
    engine = RecommendationEngine()
    assert engine.generate() == []


def test_recommendations_sorted_by_priority_then_confidence():
    engine = RecommendationEngine()
    recs = engine.generate(
        trend={"revenue": {"trend": "Decreasing", "decline_percent": 30.0}},  # high
        outliers={"amount": {"outlier_count": 1, "outlier_percentage": 12.0}},  # medium
    )
    assert _priorities(recs) == sorted(_priorities(recs), key=lambda p: {"High": 0, "Medium": 1, "Low": 2}[p])


def test_engine_is_reusable_with_a_custom_rule_subset():
    engine = RecommendationEngine(rules=[RevenueDeclineRule()])
    recs = engine.generate(
        trend={"revenue": {"trend": "Decreasing", "decline_percent": 10.0}},
        kpis={"top_customer_revenue_share_percent": 90.0},  # would trigger CustomerConcentrationRule too
    )
    assert _recommendations(recs) == {"Improve marketing strategy"}


def test_engine_instance_has_no_call_to_call_state_leakage():
    # Stateless: two independent generate() calls on the same instance must
    # not affect each other.
    engine = RecommendationEngine()
    first = engine.generate(trend={"revenue": {"trend": "Decreasing", "decline_percent": 30.0}})
    second = engine.generate()
    assert first != []
    assert second == []


def test_recommendation_dataclass_has_the_required_fields():
    fields = Recommendation.__dataclass_fields__.keys()
    assert set(fields) == {"category", "priority", "confidence", "impact", "recommendation", "reason"}


def test_individual_rules_return_none_when_their_input_is_absent():
    from common.insights.recommendation_engine import RecommendationContext

    empty_context = RecommendationContext()
    for rule in (
        RevenueDeclineRule(),
        ProfitMarginDeclineRule(),
        HighMissingValuesRule(),
        CustomerConcentrationRule(),
        OutlierQualityRule(),
    ):
        assert rule.evaluate(empty_context) is None
