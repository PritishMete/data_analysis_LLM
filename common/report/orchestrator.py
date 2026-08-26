# common/report/orchestrator.py
# ─────────────────────────────────────────────────────────────────────────────
# Selection-aware structured analytics orchestration.
#
# This module is the single place that decides, given the analysis type IDs
# a user selected on the Flutter side, which of the EXISTING deterministic
# Python detectors (common/insights/*.py, common/statistics/service.py) need
# to run, in what order (to satisfy their internal dependencies), and which
# of their results actually get exposed in the final response.
#
# Nothing in here recomputes or reimplements what trend_detector.py,
# outlier_detector.py, kpi_detector.py, recommendation_engine.py,
# chart_recommender.py, or executive_summary.py already do — this module
# only wires their existing entry points together and gates exposure by
# selection. No Gemini/LLM call happens anywhere in this file.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import pandas as pd

from common.insights.chart_recommender import ChartRecommenderEngine
from common.insights.kpi_detector import KpiDetectorEngine
from common.insights.executive_summary import generate_executive_summary
from common.insights.recommendation_engine import RecommendationEngine
from common.insights.service import InsightsService
from common.statistics.service import calculate_data_quality_score

# ── Canonical selection IDs (as sent by the Flutter client) ────────────────
STATISTICS = "statistics"
TREND_DETECTION = "trend_detection"
OUTLIER_DETECTION = "outlier_detection"
KPI_ANALYSIS = "kpi_analysis"
RECOMMENDATIONS = "recommendations"
CHART_RECOMMENDATION = "chart_recommendation"
DATA_QUALITY = "data_quality"
EXECUTIVE_SUMMARY = "executive_summary"

ALL_ANALYSIS_IDS = (
    STATISTICS, TREND_DETECTION, OUTLIER_DETECTION, KPI_ANALYSIS,
    RECOMMENDATIONS, CHART_RECOMMENDATION, DATA_QUALITY, EXECUTIVE_SUMMARY,
)

# Maps a selection id to the key it's exposed under in the response.
_RESULT_KEY = {
    STATISTICS: "statistics",
    TREND_DETECTION: "trend_insight",
    OUTLIER_DETECTION: "outliers",
    KPI_ANALYSIS: "detected_kpis",
    RECOMMENDATIONS: "recommendations",
    CHART_RECOMMENDATION: "chart_recommendation",
    DATA_QUALITY: "data_quality",
    EXECUTIVE_SUMMARY: "executive_summary",
}

_PERIOD_NAME_HINTS = ("date", "month", "year", "period", "quarter", "week", "time", "day")

# Generic, dataset-agnostic hints for picking a default numeric value
# column when the caller doesn't supply one — same "match by convention,
# never by a specific dataset's exact names" philosophy as
# common/insights/kpi_detector.py's hint groups.
_VALUE_COLUMN_HINTS = (
    "revenue", "sales", "total_sales", "gross_sales", "net_sales",
    "amount", "total", "price", "value", "income", "profit", "cost",
)

_insights_service = InsightsService()
_recommendation_engine = RecommendationEngine()
_chart_recommender_engine = ChartRecommenderEngine()
_kpi_detector_engine = KpiDetectorEngine()


def _detect_value_column(df: pd.DataFrame) -> str | None:
    """Best-effort default numeric column when the caller doesn't specify
    one. Prefers a column whose name matches a common business-metric
    convention; falls back to the first numeric column; None if there are
    no numeric columns at all."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        return None
    lowered = [(col, str(col).lower()) for col in numeric_columns]
    for hint in _VALUE_COLUMN_HINTS:
        for col, low in lowered:
            if hint in low:
                return col
    return numeric_columns[0]


def _detect_period_column(df: pd.DataFrame) -> str | None:
    """Same spirit as kpi_detector.py / chart_recommender.py's own period
    detection: a real datetime column first, otherwise a column whose name
    looks period-like."""
    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if datetime_columns:
        return datetime_columns[0]
    for column in df.columns:
        if any(hint in str(column).lower() for hint in _PERIOD_NAME_HINTS):
            return column
    return None


def _generate_statistics(df: pd.DataFrame, value_column: str | None) -> dict:
    """Plain descriptive statistics (count/mean/std/min/quartiles/max) for
    the value column — nothing ML-derived."""
    if value_column is None or value_column not in df.columns:
        return {}
    described = df[value_column].astype(float).describe()
    return {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in described.to_dict().items()}


def _generate_missing_value_report(df: pd.DataFrame) -> dict:
    """% of missing values per column — the signal
    recommendation_engine.HighMissingValuesRule looks for."""
    return {str(col): round(float(pct), 2) for col, pct in (df.isna().mean() * 100).items()}


def _outliers_by_column_for_recommendations(outlier_findings: list[dict]) -> dict:
    """Reduces the detector's per-(column, method) findings down to the
    one-entry-per-column shape recommendation_engine.OutlierQualityRule
    expects, keeping whichever method flagged the higher percentage."""
    worst_by_column: dict = {}
    for finding in outlier_findings:
        column = finding["column"]
        current = worst_by_column.get(column)
        if current is None or finding["percentage"] > current["outlier_percentage"]:
            worst_by_column[column] = {
                "outlier_count": finding["outlier_count"],
                "outlier_percentage": finding["percentage"],
            }
    return worst_by_column


def generate_structured_report_data(
    df: pd.DataFrame,
    selected_analysis_ids: list[str],
    value_column: str | None = None,
    period_column: str | None = None,
    question: str | None = None,
    label: str | None = None,
    derived_column: str | None = None,
    derived_columns: list[dict] | None = None,
    derived_source_column: str | None = None,
) -> dict:
    """The reusable orchestration function.

    Given a (cleaned) DataFrame and the list of analysis-type ids the user
    selected, runs ONLY the Python analytics needed to satisfy that
    selection — computing internal dependencies (e.g. executive_summary
    needs statistics/kpis/trend/recommendations/outliers/data_quality)
    without exposing them unless they were independently selected too.

    derived_column: name of a column just created by a transformation this
        request (e.g. range binning) — passed to the chart recommender so
        it can ground its recommendation in that specific new column
        instead of a generic guess.
    derived_columns: list of {"new_column", "source_column", "method",
        "category_count"} dicts for ALL columns derived this session —
        surfaced in the executive summary's "Derived Columns" section.

    Returns a dict containing only the keys corresponding to
    `selected_analysis_ids` (using the canonical Flutter-facing field
    names: statistics, trend_insight, outliers, detected_kpis,
    recommendations, chart_recommendation, data_quality,
    executive_summary). An empty/None selection returns an empty dict.
    """
    selected = set(selected_analysis_ids or [])
    if not selected:
        return {}

    if value_column is None:
        value_column = _detect_value_column(df)
    if period_column is None:
        period_column = _detect_period_column(df)

    # ── Resolve what needs to be COMPUTED (may exceed what's exposed) ──────
    need_data_quality = DATA_QUALITY in selected or EXECUTIVE_SUMMARY in selected
    need_trend = (
        TREND_DETECTION in selected or RECOMMENDATIONS in selected
        or CHART_RECOMMENDATION in selected or EXECUTIVE_SUMMARY in selected
    )
    need_statistics = (
        STATISTICS in selected or CHART_RECOMMENDATION in selected or EXECUTIVE_SUMMARY in selected
    )
    need_outliers = OUTLIER_DETECTION in selected or RECOMMENDATIONS in selected or EXECUTIVE_SUMMARY in selected
    need_kpis = KPI_ANALYSIS in selected or EXECUTIVE_SUMMARY in selected
    need_recommendations = RECOMMENDATIONS in selected or EXECUTIVE_SUMMARY in selected

    statistics = _generate_statistics(df, value_column) if need_statistics else {}

    trend_insight_model = None
    if need_trend and value_column is not None:
        try:
            trend_insight_model = _insights_service.detect_trend(
                df, value_column, period_column, label=label
            )
        except ValueError:
            trend_insight_model = None
    trend_insight = trend_insight_model.model_dump() if trend_insight_model is not None else {}

    outlier_findings = _insights_service.detect_outliers(df) if need_outliers else []

    detected_kpis = _kpi_detector_engine.detect(df, statistics=statistics) if need_kpis else []

    recommendations: list[dict] = []
    if need_recommendations:
        missing_value_report = _generate_missing_value_report(df)
        recommendations = _recommendation_engine.generate(
            statistics={"missing_percentage": missing_value_report},
            trend=({value_column: trend_insight} if value_column and trend_insight else None),
            kpis=None,
            outliers=_outliers_by_column_for_recommendations(outlier_findings),
        )

    chart_recommendation = {}
    if CHART_RECOMMENDATION in selected:
        chart_recommendation = _chart_recommender_engine.recommend(
            question=question, df=df, statistics=statistics, trend=trend_insight,
            derived_column=derived_column, derived_source_column=derived_source_column,
        )

    data_quality = calculate_data_quality_score(df) if need_data_quality else {}

    executive_summary = {}
    if EXECUTIVE_SUMMARY in selected:
        executive_summary = generate_executive_summary(
            statistics=statistics,
            kpis=detected_kpis,
            trend=trend_insight,
            recommendations=recommendations,
            outliers=outlier_findings,
            data_quality=data_quality,
            derived_columns=derived_columns,
        )

    computed = {
        STATISTICS: statistics,
        TREND_DETECTION: trend_insight,
        OUTLIER_DETECTION: outlier_findings,
        KPI_ANALYSIS: detected_kpis,
        RECOMMENDATIONS: recommendations,
        CHART_RECOMMENDATION: chart_recommendation,
        DATA_QUALITY: data_quality,
        EXECUTIVE_SUMMARY: executive_summary,
    }

    # Only expose sections that were actually selected — dependencies
    # computed above to satisfy e.g. executive_summary are never leaked
    # into the response unless independently requested.
    return {_RESULT_KEY[analysis_id]: computed[analysis_id] for analysis_id in selected if analysis_id in computed}
