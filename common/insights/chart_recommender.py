"""Deterministic chart metadata recommender.

This module recommends a chart type and returns metadata only. It never renders,
generates, or returns chart images. Flutter owns the actual visualization.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import re

import pandas as pd

LOW_CARDINALITY_MAX = 4
MODERATE_CARDINALITY_MAX = 8
HIGH_CARDINALITY_MIN = 9

_PERIOD_NAME_HINTS = ("date", "month", "year", "period", "quarter", "week", "time", "day")
_TIME_SERIES_KEYWORDS = (
    "over time", "trend", "monthly", "yearly", "quarterly", "weekly", "daily",
    "growth", "by month", "by year", "by day", "timeline", "history",
    "year over year", "month over month", "change over",
)
_CORRELATION_KEYWORDS = (
    "correlation", "correlate", "relationship between", " vs ", " versus ",
    "related to", "impact of", "influence of", "association between", "affect",
)
_DISTRIBUTION_KEYWORDS = (
    "distribution", "histogram", "spread of", "distributed", "frequency of", "range of",
)
_PART_TO_WHOLE_KEYWORDS = (
    "share of", "percentage of", "proportion", "breakdown of", " % of total", "% of total",
    "made up of", "split between", "of the total",
)
_MANY_CATEGORIES_KEYWORDS = ("top ", "ranking", "rank ", "each ", "every ", "all categories")
_CATEGORY_COMPARISON_KEYWORDS = (
    "compare", "by category", "across", "highest", "lowest", "most", "least",
    "by region", "by product", "by department", "by segment",
)


@dataclass(frozen=True)
class ChartRecommendation:
    chart: str
    title: str
    subtitle: str
    x_axis: str
    y_axis: str
    series: list[dict]
    confidence: float
    reason: str


@dataclass
class ChartContext:
    question: str
    df: pd.DataFrame
    statistics: dict
    trend: dict
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    datetime_columns: list = field(default_factory=list)
    row_count: int = 0
    derived_column: str | None = None
    derived_source_column: str | None = None


class ChartRule(ABC):
    name: str = "unnamed_chart_rule"

    @abstractmethod
    def evaluate(self, context: ChartContext) -> ChartRecommendation | None:
        raise NotImplementedError


def _question_has_any(question: str, *keywords: str) -> bool:
    return any(keyword in question for keyword in keywords)


def _has_period_like_column(df: pd.DataFrame) -> bool:
    return any(any(hint in str(col).lower() for hint in _PERIOD_NAME_HINTS) for col in df.columns)


def _confidence(strength: float, *, floor: float = 0.55, ceiling: float = 0.95) -> float:
    return round(min(ceiling, floor + strength), 2)


def _low_cardinality_categoricals(context: ChartContext, max_unique: int, min_unique: int = 2) -> list[str]:
    return [
        col for col in context.categorical_columns
        if min_unique <= context.df[col].nunique() <= max_unique
    ]


def _high_cardinality_categoricals(context: ChartContext, min_unique: int) -> list[str]:
    return [col for col in context.categorical_columns if context.df[col].nunique() >= min_unique]


def _display_name(column: str | None, fallback: str = "Value") -> str:
    if column is None:
        return fallback
    text = re.sub(r"[_-]+", " ", str(column)).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:1].upper() + text[1:] if text else fallback


def _find_question_column(question: str, columns: list[str]) -> str | None:
    """Pick a column explicitly mentioned in the question, when possible."""
    for column in columns:
        normalized = str(column).lower().replace("_", " ").replace("-", " ")
        if normalized and normalized in question:
            return column
    return None


def _select_y(context: ChartContext, excluded: set[str] | None = None) -> str | None:
    excluded = excluded or set()
    candidates = [c for c in context.numeric_columns if c not in excluded]
    return _find_question_column(context.question, candidates) or (candidates[0] if candidates else None)


def _select_x_category(context: ChartContext) -> str | None:
    period = [c for c in context.datetime_columns if c in context.df.columns]
    period += [c for c in context.df.columns if c not in period and any(h in str(c).lower() for h in _PERIOD_NAME_HINTS)]
    if period:
        return _find_question_column(context.question, period) or period[0]
    return _find_question_column(context.question, context.categorical_columns) or (
        context.categorical_columns[0] if context.categorical_columns else None
    )


def _series(column: str | None, label: str | None = None, aggregation: str = "sum") -> list[dict]:
    if column is None:
        return []
    return [{"name": label or _display_name(column), "field": str(column), "aggregation": aggregation}]


def _metadata(
    *, chart: str, title: str, subtitle: str, x_axis: str, y_axis: str,
    series: list[dict], confidence: float, reason: str,
) -> ChartRecommendation:
    return ChartRecommendation(
        chart=chart,
        title=title,
        subtitle=subtitle,
        x_axis=x_axis,
        y_axis=y_axis,
        series=series,
        confidence=confidence,
        reason=reason,
    )


class TimeSeriesRule(ChartRule):
    name = "time_series"

    def evaluate(self, context: ChartContext) -> ChartRecommendation | None:
        strength = 0.0
        reasons = []
        if context.trend:
            strength += 0.2
            reasons.append("a trend has already been computed for this data")
        if context.datetime_columns or _has_period_like_column(context.df):
            strength += 0.35
            reasons.append("the data includes a date/period-like column")
        if _question_has_any(context.question, *_TIME_SERIES_KEYWORDS):
            strength += 0.35
            reasons.append("the question asks about change over time")
        if strength == 0.0:
            return None
        x = _select_x_category(context)
        y = _select_y(context)
        if x is None or y is None:
            return None
        x_label = _display_name(x, "Time")
        y_label = _display_name(y)
        return _metadata(
            chart="line",
            title=f"{y_label} by {x_label}",
            subtitle="Trend over time",
            x_axis=x_label,
            y_axis=y_label,
            series=_series(y, y_label),
            confidence=_confidence(strength),
            reason="Time series pattern detected: " + "; ".join(reasons) + ".",
        )


class CorrelationRule(ChartRule):
    name = "correlation"

    def evaluate(self, context: ChartContext) -> ChartRecommendation | None:
        strength = 0.0
        reasons = []
        if _question_has_any(context.question, *_CORRELATION_KEYWORDS):
            strength += 0.4
            reasons.append("the question asks about a relationship between two variables")
        if self._has_strong_reported_correlation(context.statistics):
            strength += 0.4
            reasons.append("a strong correlation was already reported in the statistics")
        if strength == 0.0 or len(context.numeric_columns) < 2:
            return None
        x = _find_question_column(context.question, context.numeric_columns)
        y = _select_y(context, {x} if x else set())
        x = x or context.numeric_columns[0]
        if y is None:
            return None
        strength += 0.15
        x_label, y_label = _display_name(x), _display_name(y)
        return _metadata(
            chart="scatter",
            title=f"{y_label} vs {x_label}",
            subtitle="Relationship between two numeric variables",
            x_axis=x_label,
            y_axis=y_label,
            series=_series(y, y_label, aggregation="none"),
            confidence=_confidence(strength),
            reason="Correlation between two numeric variables detected: " + "; ".join(reasons) + ".",
        )

    @staticmethod
    def _has_strong_reported_correlation(statistics: dict) -> bool:
        correlations = statistics.get("correlations") or statistics.get("correlation")
        if not isinstance(correlations, dict):
            return False
        for row in correlations.values():
            if not isinstance(row, dict):
                continue
            for value in row.values():
                try:
                    if 0.5 <= abs(float(value)) < 0.999:
                        return True
                except (TypeError, ValueError):
                    continue
        return False


class DistributionRule(ChartRule):
    name = "distribution"

    def evaluate(self, context: ChartContext) -> ChartRecommendation | None:
        strength = 0.0
        reasons = []
        if _question_has_any(context.question, *_DISTRIBUTION_KEYWORDS):
            strength += 0.45
            reasons.append("the question asks about a distribution or spread")
        continuous_column = self._most_continuous_numeric_column(context)
        if continuous_column is not None:
            strength += 0.3
            reasons.append(f"'{continuous_column}' has many distinct numeric values")
        if strength == 0.0 or continuous_column is None:
            return None
        label = _display_name(continuous_column)
        return _metadata(
            chart="histogram",
            title=f"Distribution of {label}",
            subtitle="Frequency distribution",
            x_axis=label,
            y_axis="Frequency",
            series=_series(continuous_column, label, aggregation="distribution"),
            confidence=_confidence(strength),
            reason="Distribution of a numeric variable detected: " + "; ".join(reasons) + ".",
        )

    @staticmethod
    def _most_continuous_numeric_column(context: ChartContext) -> str | None:
        if context.row_count == 0:
            return None
        best_column, best_ratio = None, 0.0
        for column in context.numeric_columns:
            ratio = context.df[column].nunique() / context.row_count
            if ratio > best_ratio:
                best_column, best_ratio = column, ratio
        return best_column if best_ratio >= 0.5 else None


class PartToWholeRule(ChartRule):
    name = "part_to_whole"

    def evaluate(self, context: ChartContext) -> ChartRecommendation | None:
        strength = 0.0
        reasons = []
        if _question_has_any(context.question, *_PART_TO_WHOLE_KEYWORDS):
            strength += 0.45
            reasons.append("the question asks about a share/percentage of a total")
        few = _low_cardinality_categoricals(context, LOW_CARDINALITY_MAX)
        if few and len(context.categorical_columns) <= 2:
            strength += 0.25
            reasons.append(f"'{few[0]}' has only a few categories")
        y = _select_y(context)
        x = _select_x_category(context)
        if strength == 0.0 or x is None or y is None:
            return None
        x_label, y_label = _display_name(x), _display_name(y)
        return _metadata(
            chart="pie",
            title=f"{y_label} by {x_label}",
            subtitle="Share of total",
            x_axis=x_label,
            y_axis=y_label,
            series=_series(y, y_label),
            confidence=_confidence(strength),
            reason="Part-to-whole composition detected: " + "; ".join(reasons) + ".",
        )


class ManyCategoriesRule(ChartRule):
    name = "many_categories"

    def evaluate(self, context: ChartContext) -> ChartRecommendation | None:
        many = _high_cardinality_categoricals(context, HIGH_CARDINALITY_MIN)
        if not many or not context.numeric_columns:
            return None
        x = many[0]
        y = _select_y(context)
        strength = 0.45
        reasons = [f"'{x}' has {context.df[x].nunique()} distinct categories"]
        if _question_has_any(context.question, *_MANY_CATEGORIES_KEYWORDS):
            strength += 0.25
            reasons.append("the question asks to compare many/all categories")
        x_label, y_label = _display_name(x), _display_name(y)
        return _metadata(
            chart="horizontal_bar",
            title=f"{y_label} by {x_label}",
            subtitle="Comparison across many categories",
            x_axis=x_label,
            y_axis=y_label,
            series=_series(y, y_label),
            confidence=_confidence(strength),
            reason="Many categories detected: " + "; ".join(reasons) + ".",
        )


class CategoryComparisonRule(ChartRule):
    name = "category_comparison"

    def evaluate(self, context: ChartContext) -> ChartRecommendation | None:
        moderate = _low_cardinality_categoricals(context, MODERATE_CARDINALITY_MAX)
        has_shape = bool(moderate) and bool(context.numeric_columns)
        matched = _question_has_any(context.question, *_CATEGORY_COMPARISON_KEYWORDS)
        if (not has_shape and not matched) or not context.numeric_columns:
            return None
        x = _find_question_column(context.question, moderate) if moderate else None
        x = x or (moderate[0] if moderate else None)
        y = _select_y(context)
        if x is None or y is None:
            return None
        strength = (0.4 if has_shape else 0.0) + (0.3 if matched else 0.0)
        reasons = []
        if has_shape:
            reasons.append(f"'{x}' has a small, comparable set of categories")
        if matched:
            reasons.append("the question asks to compare categories")
        x_label, y_label = _display_name(x), _display_name(y)
        return _metadata(
            chart="bar",
            title=f"{y_label} by {x_label}",
            subtitle="Category comparison",
            x_axis=x_label,
            y_axis=y_label,
            series=_series(y, y_label),
            confidence=_confidence(strength),
            reason="Category comparison detected: " + "; ".join(reasons) + ".",
        )


class DerivedCategoryRule(ChartRule):
    """Recommends a chart for a column just produced by a transformation like
    range_binning (see common/transformations/range_binning.py) — e.g.
    Rating_Range, Age_Range, Salary_Range. Reuses the exact same cardinality
    thresholds as every other categorical rule in this module; it doesn't
    introduce a new concept, just prioritizes the column the pipeline knows
    was JUST derived, so the recommendation is grounded in that column
    instead of whatever the generic rules would otherwise pick.
    """
    name = "derived_category"

    def evaluate(self, context: ChartContext) -> ChartRecommendation | None:
        column = context.derived_column
        if not column or column not in context.categorical_columns:
            return None

        unique_count = context.df[column].nunique()
        if unique_count < 2:
            return None

        # Ordinal/bucketed columns read naturally as a vertical "column"
        # chart when they have a manageable number of buckets, and fall
        # back to the same horizontal_bar treatment ManyCategoriesRule
        # already uses once there are too many buckets to read comfortably.
        chart_type = "column" if unique_count <= MODERATE_CARDINALITY_MAX else "horizontal_bar"

        excluded_for_y = {column}
        if context.derived_source_column:
            excluded_for_y.add(context.derived_source_column)
        y = _select_y(context, excluded=excluded_for_y)
        label = _display_name(column)
        if y is not None:
            y_label = _display_name(y)
            series = _series(y, y_label)
            title = f"{y_label} by {label}"
            y_axis = y_label
        else:
            # No other numeric column to aggregate — fall back to counting
            # rows per bucket, which is always meaningful for a freshly
            # binned column.
            series = [{"name": "Count", "field": str(column), "aggregation": "count"}]
            title = f"Record Count by {label}"
            y_axis = "Count"

        return _metadata(
            chart=chart_type,
            title=title,
            subtitle="Distribution across derived ranges",
            x_axis=label,
            y_axis=y_axis,
            series=series,
            # Deliberately near-ceiling and NOT computed from generic
            # signal strength like the other rules: this rule only fires
            # for the exact column the pipeline just derived, so it should
            # always win over a generic category-comparison guess.
            confidence=0.97,
            reason=(
                f"'{column}' was just created by range binning ({unique_count} categories); "
                f"a {chart_type} chart shows how records are distributed across those buckets."
            ),
        )


DEFAULT_RULES: tuple[ChartRule, ...] = (
    DerivedCategoryRule(),
    TimeSeriesRule(),
    CorrelationRule(),
    DistributionRule(),
    PartToWholeRule(),
    ManyCategoriesRule(),
    CategoryComparisonRule(),
)

FALLBACK_RECOMMENDATION = _metadata(
    chart="bar",
    title="Data Comparison",
    subtitle="General-purpose comparison",
    x_axis="Category",
    y_axis="Value",
    series=[],
    confidence=0.4,
    reason="No strong time-series, correlation, distribution, or category-composition signal was found; a bar chart is a safe general-purpose default.",
)


def _build_context(
    question: str | None,
    df: pd.DataFrame | None,
    statistics: dict | None,
    trend: dict | None,
    derived_column: str | None = None,
    derived_source_column: str | None = None,
) -> ChartContext:
    df = df if df is not None else pd.DataFrame()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_columns = [c for c in df.columns if c not in numeric_columns and c not in datetime_columns]
    return ChartContext(
        question=(question or "").strip().lower(),
        df=df,
        statistics=statistics or {},
        trend=trend or {},
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        row_count=len(df),
        derived_column=derived_column,
        derived_source_column=derived_source_column,
    )


class ChartRecommenderEngine:
    """Return chart metadata for Flutter. No rendering or image generation."""

    def __init__(self, rules: list[ChartRule] | tuple[ChartRule, ...] | None = None):
        self.rules: list[ChartRule] = list(rules) if rules is not None else list(DEFAULT_RULES)

    def recommend(
        self,
        question: str | None = None,
        df: pd.DataFrame | None = None,
        statistics: dict | None = None,
        trend: dict | None = None,
        derived_column: str | None = None,
        derived_source_column: str | None = None,
    ) -> dict:
        context = _build_context(
            question, df, statistics, trend,
            derived_column=derived_column, derived_source_column=derived_source_column,
        )
        candidates = [rule.evaluate(context) for rule in self.rules]
        candidates = [candidate for candidate in candidates if candidate is not None]
        best = max(candidates, key=lambda candidate: candidate.confidence) if candidates else FALLBACK_RECOMMENDATION

        # Flutter consumes this object as chart metadata. Keep the response
        # deliberately data-only: no rendering, image, or plot payloads.
        return asdict(best)


def recommend_chart(
    question: str | None = None,
    df: pd.DataFrame | None = None,
    statistics: dict | None = None,
    trend: dict | None = None,
    derived_column: str | None = None,
    derived_source_column: str | None = None,
) -> dict:
    return ChartRecommenderEngine().recommend(
        question, df, statistics, trend,
        derived_column=derived_column, derived_source_column=derived_source_column,
    )
