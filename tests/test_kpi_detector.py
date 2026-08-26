# tests/test_kpi_detector.py
import pandas as pd

from common.insights.kpi_detector import KpiDetectorEngine, RevenueKpiRule, detect_kpis

EXPECTED_KEYS = {"name", "value", "trend", "unit", "confidence", "rank", "importance", "category", "description"}


def _sales_df():
    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(1, 13)],
        "month": ["Jan", "Jan", "Jan", "Feb", "Feb", "Feb", "Mar", "Mar", "Mar", "Apr", "Apr", "Apr"],
        "customer_name": ["Acme", "Acme", "Beta", "Acme", "Gamma", "Beta", "Acme", "Beta", "Gamma", "Acme", "Beta", "Gamma"],
        "product_name": ["Widget", "Gadget", "Widget", "Widget", "Gadget", "Widget", "Gizmo", "Widget", "Gadget", "Widget", "Gizmo", "Widget"],
        "revenue": [100, 150, 90, 120, 200, 95, 300, 110, 220, 130, 310, 105],
        "cost": [60, 80, 50, 70, 120, 55, 180, 65, 130, 75, 190, 60],
    })


def test_detects_all_nine_example_kpis():
    findings = detect_kpis(_sales_df())
    names = {f["name"] for f in findings}
    expected = {
        "Revenue", "Profit", "Profit Margin", "Average Order Value", "Growth Rate",
        "Top Customer", "Top Product", "Order Count", "Customer Count",
    }
    assert expected <= names


def test_every_finding_has_the_required_shape():
    findings = detect_kpis(_sales_df())
    assert len(findings) > 0
    for finding in findings:
        assert set(finding.keys()) == EXPECTED_KEYS
        assert 0.0 <= finding["confidence"] <= 1.0


def test_kpis_are_ranked_and_classified():
    findings = detect_kpis(_sales_df())
    assert [f["rank"] for f in findings] == list(range(1, len(findings) + 1))
    assert findings[0]["name"] in {"Revenue", "Profit"}
    assert findings[0]["importance"] == "Critical"
    revenue = next(f for f in findings if f["name"] == "Revenue")
    assert revenue["category"] == "Revenue"
    assert revenue["trend"] in ("Increasing", "Decreasing", "Stable")
    assert 0.0 <= revenue["confidence"] <= 1.0
    assert revenue["description"]


def test_backward_compatible_legacy_kpi_fields_remain():
    findings = detect_kpis(_sales_df())
    revenue = next(f for f in findings if f["name"] == "Revenue")
    assert revenue["name"] == "Revenue"
    assert revenue["value"] == 1930.0
    assert revenue["unit"] == "currency"


def test_revenue_value_and_unit():
    findings = detect_kpis(_sales_df())
    revenue = next(f for f in findings if f["name"] == "Revenue")
    assert revenue["value"] == 1930.0
    assert revenue["unit"] == "currency"
    assert revenue["trend"] in ("Increasing", "Decreasing", "Stable")


def test_top_customer_and_top_product_are_identity_strings():
    findings = detect_kpis(_sales_df())
    top_customer = next(f for f in findings if f["name"] == "Top Customer")
    top_product = next(f for f in findings if f["name"] == "Top Product")
    assert isinstance(top_customer["value"], str)
    assert isinstance(top_product["value"], str)
    assert top_customer["unit"] is None
    assert top_customer["trend"] is None


def test_order_count_uses_distinct_order_ids():
    findings = detect_kpis(_sales_df())
    order_count = next(f for f in findings if f["name"] == "Order Count")
    assert order_count["value"] == 12


def test_no_columns_matching_any_kpi_convention_returns_nothing():
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    assert detect_kpis(df) == []


def test_order_count_does_not_fire_on_unrelated_data_without_a_financial_signal():
    # Regression test: previously fell back to len(df) unconditionally,
    # producing a bogus "Order Count" on any non-empty DataFrame.
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    findings = detect_kpis(df)
    assert all(f["name"] != "Order Count" for f in findings)


def test_order_count_still_fires_with_a_value_column_but_no_explicit_id():
    df = pd.DataFrame({"amount": [10, 20, 15]})
    findings = detect_kpis(df)
    order_count = next(f for f in findings if f["name"] == "Order Count")
    assert order_count["value"] == 3
    assert order_count["confidence"] < 0.7  # weaker signal than an explicit order id


def test_empty_dataframe_returns_nothing():
    assert detect_kpis(pd.DataFrame()) == []


def test_works_with_synonym_column_names_no_hardcoded_dataset():
    # Different naming convention entirely -- proves this isn't tied to
    # one specific dataset's exact column names.
    df = pd.DataFrame({
        "sales": [1000, 1200, 900],
        "expense": [400, 500, 350],
        "client_name": ["A", "B", "A"],
    })
    findings = detect_kpis(df)
    names = {f["name"] for f in findings}
    assert {"Revenue", "Profit"} <= names


def test_profit_is_derived_when_no_direct_profit_column_exists():
    df = pd.DataFrame({"revenue": [100, 200], "cost": [40, 90]})
    findings = detect_kpis(df)
    profit = next(f for f in findings if f["name"] == "Profit")
    assert profit["value"] == 170.0  # (100-40) + (200-90)
    assert profit["confidence"] < 0.92  # derived, less confident than a direct column


def test_statistics_input_is_reused_for_unique_counts():
    df = pd.DataFrame({"order_id": ["o1", "o2", "o3", "o1"], "revenue": [10, 20, 30, 10]})
    # Deliberately wrong count, to prove it's actually being REUSED and not recomputed.
    stats = {"distribution": {"unique_values": {"order_id": 999}}}
    findings = KpiDetectorEngine().detect(df, statistics=stats)
    order_count = next(f for f in findings if f["name"] == "Order Count")
    assert order_count["value"] == 999


def test_engine_is_reusable_with_a_custom_rule_subset():
    engine = KpiDetectorEngine(rules=[RevenueKpiRule()])
    findings = engine.detect(_sales_df())
    assert {f["name"] for f in findings} == {"Revenue"}


def test_no_llm_or_ai_imports_anywhere_in_module():
    import common.insights.kpi_detector as module
    source = open(module.__file__).read().lower()
    for forbidden in ("gemini", "openai", "google.genai", "google.adk", "llm"):
        assert forbidden not in source
