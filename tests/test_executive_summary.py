import pandas as pd

from common.insights.executive_summary import generate_executive_summary
from common.statistics.service import calculate_data_quality_score
from ai_engine import generate_dataframe_insights


def test_executive_summary_returns_structured_concise_sections():
    result = generate_executive_summary(
        statistics={"mean": 100.0, "min": 80.0, "max": 120.0},
        kpis=[{"rank": 1, "name": "Revenue", "value": 1200, "trend": "Increasing"}],
        trend={"trend": "Increasing", "growth_rate": 18.2, "confidence": 0.94},
        recommendations=[],
        outliers=[],
        data_quality={"quality_score": 97, "quality_grade": "Excellent", "quality_summary": "Data quality score is 97.00/100 (Excellent)."},
    )
    assert set(result) == {"overall_health", "key_findings", "business_risks", "business_opportunities", "data_quality_summary"}
    assert result["overall_health"] == "Healthy"
    assert isinstance(result["key_findings"], list)
    assert len(result["key_findings"]) <= 6
    assert "97.00/100" in result["data_quality_summary"]


def test_executive_summary_flags_high_priority_declining_revenue():
    result = generate_executive_summary(
        trend={"trend": "Decreasing", "decline_percent": 20, "confidence": 0.95},
        recommendations=[{
            "category": "Revenue", "priority": "High", "confidence": 0.9,
            "impact": "High", "recommendation": "Improve marketing strategy",
            "reason": "Revenue is trending downward.",
        }],
        outliers=[],
        data_quality={"quality_score": 100},
    )
    assert result["overall_health"] == "At Risk"
    assert any("Improve marketing strategy" in item for item in result["business_risks"])


def test_ai_engine_includes_quality_and_executive_summary_without_gemini():
    df = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr"],
        "revenue": [1000, 900, 800, 700],
    })
    result = generate_dataframe_insights(df, value_column="revenue", period_column="month", label="Revenue")
    assert "data_quality" in result
    assert "executive_summary" in result
    assert set(result["executive_summary"]) == {
        "overall_health", "key_findings", "business_risks", "business_opportunities", "data_quality_summary",
    }
    assert result["data_quality"] == calculate_data_quality_score(df)
