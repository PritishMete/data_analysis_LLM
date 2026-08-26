# common/insights/schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# Thin Pydantic wrapper around trend_detector.detect_trend()'s return dict —
# gives InsightsService (service.py) and any future route a validated,
# typed, JSON-serializable shape without trend_detector.py itself needing to
# know Pydantic exists (it stays pure Python + pandas, importable anywhere,
# including outside a FastAPI request).
# ─────────────────────────────────────────────────────────────────────────────

from typing import Literal

from pydantic import BaseModel


class TrendInsight(BaseModel):
    trend: Literal["Increasing", "Decreasing", "Stable"]
    confidence: float = 0.0
    growth_rate: float | None = None
    method: Literal["linear_regression", "exponential_growth", "stable", "seasonal_basic"] = "stable"
    significance_level: Literal["High", "Medium", "Low"] = "Low"
    slope: float = 0.0
    growth_percent: float | None
    decline_percent: float
    highest_period: str | None
    lowest_period: str | None
    consecutive_increase: int
    consecutive_decrease: int
    summary: str


class RecommendationInsight(BaseModel):
    """Validated, JSON-serializable recommendation contract for frontend consumers."""

    category: Literal[
        "Revenue",
        "Profitability",
        "Customers",
        "Inventory",
        "Data Quality",
        "Operations",
    ]
    priority: Literal["High", "Medium", "Low"]
    confidence: float
    impact: Literal["High", "Medium", "Low"]
    recommendation: str
    reason: str


class KpiInsight(BaseModel):
    """Validated KPI contract. Legacy value/unit fields remain available."""

    rank: int
    importance: Literal["Critical", "High", "Medium", "Low"]
    category: Literal[
        "Revenue",
        "Profitability",
        "Customers",
        "Inventory",
        "Data Quality",
        "Operations",
    ]
    confidence: float
    trend: Literal["Increasing", "Decreasing", "Stable"] | None
    description: str
    # Backward-compatible fields from the existing KPI response.
    name: str
    value: float | int | str
    unit: Literal["currency", "%", "count"] | None
