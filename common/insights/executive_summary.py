"""Rule-based executive summary builder.

Combines already-computed analytics into a concise, structured JSON-friendly
summary. This module deliberately does not call Gemini, an LLM, or any ML
model and does not calculate new business metrics.
"""

from __future__ import annotations

from typing import Any


_HEALTH_ORDER = {"Critical": 4, "At Risk": 3, "Needs Attention": 2, "Healthy": 1}


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _priority_rank(priority: Any) -> int:
    return {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}.get(str(priority), 0)


def _quality_health(quality_score: Any) -> str:
    try:
        score = float(quality_score)
    except (TypeError, ValueError):
        return "Healthy"
    if score < 40:
        return "Critical"
    if score < 60:
        return "At Risk"
    if score < 75:
        return "Needs Attention"
    return "Healthy"


def _recommendation_health(recommendations: list[dict]) -> str:
    priorities = {_priority_rank(r.get("priority")) for r in recommendations}
    if 4 in priorities:
        return "Critical"
    if 3 in priorities:
        return "At Risk"
    if 2 in priorities:
        return "Needs Attention"
    return "Healthy"


def _outlier_health(outliers: list[dict]) -> str:
    severities = {str(o.get("severity")) for o in outliers}
    if "Critical" in severities:
        return "Critical"
    if "High" in severities:
        return "At Risk"
    if "Medium" in severities:
        return "Needs Attention"
    return "Healthy"


def _trend_health(trend: dict) -> str:
    if trend.get("trend") == "Decreasing":
        confidence = _clamp01(trend.get("confidence"))
        decline = abs(float(trend.get("decline_percent") or 0.0))
        if confidence >= 0.8 and decline >= 15:
            return "At Risk"
        if confidence >= 0.65 or decline >= 5:
            return "Needs Attention"
    return "Healthy"


def _highest_health(*health_values: str) -> str:
    return max(health_values, key=lambda value: _HEALTH_ORDER.get(value, 0), default="Healthy")


def _kpi_findings(kpis: list[dict]) -> list[str]:
    findings: list[str] = []
    for kpi in sorted(kpis, key=lambda item: item.get("rank", 999))[:4]:
        name = kpi.get("name") or "KPI"
        value = kpi.get("value")
        trend = kpi.get("trend")
        if trend in {"Increasing", "Decreasing"}:
            findings.append(f"{name} is {trend.lower()} at {value}.")
        else:
            findings.append(f"{name} is {value}.")
    return findings


def _statistics_findings(statistics: dict) -> list[str]:
    findings: list[str] = []
    if not statistics:
        return findings
    if statistics.get("mean") is not None:
        findings.append(f"Average value is {statistics['mean']}.")
    if statistics.get("max") is not None:
        findings.append(f"Maximum value is {statistics['max']}.")
    if statistics.get("min") is not None:
        findings.append(f"Minimum value is {statistics['min']}.")
    return findings


def _trend_findings(trend: dict) -> list[str]:
    if not trend or not trend.get("trend"):
        return []
    label = trend["trend"]
    growth = trend.get("growth_rate")
    if growth is None:
        growth = trend.get("growth_percent")
    if growth is not None:
        direction = "growth" if float(growth) >= 0 else "decline"
        return [f"The primary trend is {label.lower()} with {abs(float(growth)):.1f}% {direction}."]
    return [f"The primary trend is {label.lower()}."]


def _outlier_findings(outliers: list[dict]) -> list[str]:
    flagged = [o for o in outliers if (o.get("outlier_count") or 0) > 0]
    flagged.sort(key=lambda o: (_priority_rank(o.get("severity")), o.get("outlier_count", 0)), reverse=True)
    return [
        f"{o.get('column')} has {o.get('outlier_count')} outlier(s) by {str(o.get('method', '')).upper()} detection."
        for o in flagged[:3]
    ]


def _risks(recommendations: list[dict], outliers: list[dict], trend: dict) -> list[str]:
    risks: list[str] = []
    for rec in sorted(recommendations, key=lambda r: _priority_rank(r.get("priority")), reverse=True):
        if _priority_rank(rec.get("priority")) >= 2:
            risks.append(f"{rec.get('recommendation')}: {rec.get('reason')}")
        if len(risks) >= 3:
            break

    if len(risks) < 3:
        for finding in _outlier_findings(outliers):
            risks.append(finding)
            if len(risks) >= 3:
                break

    if len(risks) < 3 and trend.get("trend") == "Decreasing":
        risks.append("The primary metric is declining and may require attention.")
    return risks[:3]


def _opportunities(kpis: list[dict], trend: dict, recommendations: list[dict]) -> list[str]:
    opportunities: list[str] = []
    if trend.get("trend") == "Increasing":
        growth = trend.get("growth_rate")
        if growth is None:
            growth = trend.get("growth_percent")
        if growth is not None:
            opportunities.append(f"Build on the positive trend while growth is {float(growth):.1f}%.")
        else:
            opportunities.append("Build on the positive primary trend.")

    for kpi in sorted(kpis, key=lambda item: item.get("rank", 999)):
        if kpi.get("trend") == "Increasing":
            opportunities.append(f"Scale the positive movement in {kpi.get('name')}.")
        if len(opportunities) >= 3:
            break

    if len(opportunities) < 3:
        for rec in recommendations:
            if rec.get("category") in {"Revenue", "Profitability", "Customers", "Inventory", "Operations"}:
                opportunities.append(f"Act on the {rec.get('category').lower()} insight: {rec.get('recommendation')}.")
            if len(opportunities) >= 3:
                break
    return opportunities[:3]


def _derived_columns_findings(derived_columns: list[dict]) -> list[str]:
    findings = []
    for entry in derived_columns:
        new_col = entry.get("new_column")
        method = entry.get("method", "Transformation")
        count = entry.get("category_count")
        source = entry.get("source_column")
        if not new_col:
            continue
        detail = f"Created {new_col}"
        if source:
            detail += f" from {source}"
        detail += f" using {method}"
        if count is not None:
            detail += f" ({count} categories)"
        detail += "."
        findings.append(detail)
    return findings


def generate_executive_summary(
    *,
    statistics: dict | None = None,
    kpis: list[dict] | None = None,
    trend: dict | None = None,
    recommendations: list[dict] | None = None,
    outliers: list[dict] | None = None,
    data_quality: dict | None = None,
    derived_columns: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a concise executive summary from existing analysis outputs.

    derived_columns: optional list of columns created by a transformation
    this session (e.g. range binning), each shaped like
    {"new_column": "Rating_Range", "source_column": "Rating",
     "method": "Range Binning", "category_count": 5}. When present, these
    show up in their own "Derived Columns" section AND are folded into
    key_findings so they surface in the headline summary too.
    """
    statistics = statistics or {}
    kpis = kpis or []
    trend = trend or {}
    recommendations = recommendations or []
    outliers = outliers or []
    data_quality = data_quality or {}
    derived_columns = derived_columns or []

    derived_findings = _derived_columns_findings(derived_columns)

    key_findings = (
        derived_findings + _statistics_findings(statistics) + _kpi_findings(kpis)
        + _trend_findings(trend) + _outlier_findings(outliers)
    )[:6]
    if not key_findings:
        key_findings = ["No material findings were detected from the available analytics."]

    risks = _risks(recommendations, outliers, trend)
    if not risks:
        risks = ["No material business risks were detected by the current rules."]

    opportunities = _opportunities(kpis, trend, recommendations)
    if not opportunities:
        opportunities = ["No material growth opportunity was identified by the current rules."]

    quality_summary = data_quality.get("quality_summary") or "Data quality score was not available."
    health = _highest_health(
        _quality_health(data_quality.get("quality_score")),
        _recommendation_health(recommendations),
        _outlier_health(outliers),
        _trend_health(trend),
    )

    summary: dict[str, Any] = {
        "overall_health": health,
        "key_findings": key_findings,
        "business_risks": risks,
        "business_opportunities": opportunities,
        "data_quality_summary": quality_summary,
    }

    # Only added when there's actually something derived this session —
    # keeps the shape identical to before for every existing caller.
    if derived_columns:
        summary["derived_columns"] = [
            {
                "new_column": entry.get("new_column"),
                "source_column": entry.get("source_column"),
                "method": entry.get("method", "Transformation"),
                "category_count": entry.get("category_count"),
            }
            for entry in derived_columns
        ]

    return summary


# Short aliases for callers that prefer service-style naming.
generate = generate_executive_summary
build_executive_summary = generate_executive_summary
