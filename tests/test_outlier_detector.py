# tests/test_outlier_detector.py
import pytest
import pandas as pd

from common.insights.outlier_detector import (
    _severity,
    detect_outliers,
    detect_outliers_iqr,
    detect_outliers_zscore,
)
from common.insights.service import InsightsService

EXPECTED_KEYS = {
    "column", "method", "outlier_count", "percentage", "severity",
    "recommended_action", "examples", "summary", "confidence"
}


def _df_with_outlier():
    return pd.DataFrame({
        "amount": [10, 11, 9, 12, 10, 11, 9, 500, 10, 11],
        "steady": [5] * 10,
        "category": ["a", "b"] * 5,
    })


def test_iqr_flags_the_obvious_outlier():
    findings = detect_outliers_iqr(_df_with_outlier())
    amount = next(f for f in findings if f["column"] == "amount")
    assert amount["method"] == "iqr"
    assert amount["outlier_count"] == 1
    assert 500 in amount["examples"]
    assert set(amount.keys()) == EXPECTED_KEYS


def test_zscore_runs_and_returns_expected_shape():
    findings = detect_outliers_zscore(_df_with_outlier(), columns=["amount"], threshold=1.0)
    assert len(findings) == 1
    assert findings[0]["method"] == "zscore"
    assert findings[0]["outlier_count"] >= 1
    assert set(findings[0].keys()) == EXPECTED_KEYS


def test_non_numeric_columns_are_excluded_automatically():
    findings = detect_outliers(_df_with_outlier())
    assert all(f["column"] != "category" for f in findings)


def test_zero_spread_column_is_skipped():
    findings = detect_outliers(_df_with_outlier())
    assert all(f["column"] != "steady" for f in findings)


def test_detect_outliers_runs_both_methods_by_default():
    findings = detect_outliers(_df_with_outlier())
    methods = {f["method"] for f in findings if f["column"] == "amount"}
    assert methods == {"iqr", "zscore"}


def test_can_restrict_to_a_single_method():
    findings = detect_outliers(_df_with_outlier(), methods=("iqr",))
    assert all(f["method"] == "iqr" for f in findings)


def test_severity_reflects_percentage_thresholds():
    # 30% outliers should be "high" severity, well above the 15% cutoff.
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 1000, 2000, 3000]})
    findings = detect_outliers_iqr(df, columns=["x"])
    assert findings[0]["severity"] == "Critical"


def test_severity_action_and_confidence_are_business_ready():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 1000, 2000, 3000]})
    finding = detect_outliers_iqr(df, columns=["x"])[0]
    assert finding["severity"] == "Critical"
    assert finding["recommended_action"] == "Validate source"
    assert 0.0 <= finding["confidence"] <= 1.0


def test_critical_severity_threshold():
    assert _severity(20.0) == "Critical"
    assert _severity(30.0) == "Critical"


def test_summary_reports_zero_when_nothing_found():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    findings = detect_outliers_iqr(df, columns=["x"])
    assert findings[0]["outlier_count"] == 0
    assert "No outliers" in findings[0]["summary"]


def test_examples_are_capped_and_json_safe():
    df = pd.DataFrame({"x": list(range(1, 51)) + [100000] * 8})
    findings = detect_outliers_iqr(df, columns=["x"])
    assert findings[0]["outlier_count"] > 5
    assert len(findings[0]["examples"]) == 5
    assert all(isinstance(v, (int, float)) for v in findings[0]["examples"])


def test_raises_for_unknown_column():
    with pytest.raises(ValueError):
        detect_outliers_iqr(_df_with_outlier(), columns=["nope"])


def test_raises_for_non_numeric_explicit_column():
    with pytest.raises(ValueError):
        detect_outliers_iqr(_df_with_outlier(), columns=["category"])


def test_insights_service_delegates_to_detect_outliers():
    findings = InsightsService().detect_outliers(_df_with_outlier())
    assert any(f["column"] == "amount" for f in findings)
    assert all(set(f.keys()) == EXPECTED_KEYS for f in findings)
