# common/insights/kpi_detector.py
# ─────────────────────────────────────────────────────────────────────────────
# Rule-based KPI detector. Given a DataFrame (and,
# optionally, already-computed statistics), it looks for common business
# KPIs by matching column NAMES against generic, well-known business-data
# naming conventions ("revenue", "cost", "customer_id", "order_id", ...) —
# never by hardcoding a specific dataset's exact column names or business
# domain. The same rule set works unmodified on a sales spreadsheet, a
# SaaS subscription export, or a retail orders table, as long as the
# columns are named the way people conventionally name them; a dataset
# whose columns match none of these hints simply yields fewer (or zero)
# KPI findings — this module never invents a metric it can't actually
# compute from what's there.
#
# Same package, same philosophy as trend_detector.py, outlier_detector.py,
# recommendation_engine.py, and chart_recommender.py: deterministic rules
# over already-known data, every run on the same input produces the exact
# same output, and — where a KPI has a time dimension — this module reuses
# trend_detector.detect_trend() rather than reimplementing trend logic.
#
# ── Return shape (per KPI found) ────────────────────────────────────────
#   name         "Revenue", "Profit", "Profit Margin", "Average Order Value",
#                "Growth Rate", "Top Customer", "Top Product", "Order Count",
#                "Customer Count"
#   value        float for magnitude KPIs (Revenue, Profit, Profit Margin,
#                Average Order Value, Growth Rate, Order Count, Customer
#                Count); str for identity KPIs (Top Customer, Top Product —
#                the winning entity's label, e.g. "Acme Corp")
#   trend        "Increasing" | "Decreasing" | "Stable" | None — from
#                trend_detector.detect_trend() when a period-like column is
#                present; None for identity KPIs and whenever no period
#                column can be found
#   unit         "currency" | "%" | "count" | None (identity KPIs).
#                "currency" is deliberately generic — this module has no
#                way to know USD vs EUR vs anything else, and never guesses
#   confidence   0.0-1.0 — how well the matched column name(s) fit the
#                expected convention, and how direct the computation was
#                (a KPI read straight from a column is more confident than
#                one derived from two, e.g. profit = revenue - cost)
# ─────────────────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import pandas as pd

from .trend_detector import detect_trend

_PERIOD_NAME_HINTS = ("date", "month", "year", "period", "quarter", "week", "time", "day")

# Each hint group is (keywords, confidence) — checked in order, so an exact
# convention like "revenue" scores higher than a looser synonym like
# "sales". Purely generic English business-data naming conventions; no
# dataset-specific names appear anywhere in this file.
_REVENUE_HINTS = [
    (("revenue",), 0.92),
    (("total_sales", "gross_sales", "net_sales"), 0.88),
    (("sales", "income", "turnover"), 0.75),
]
_PROFIT_HINTS = [
    (("profit",), 0.92),
    (("net_income", "earnings"), 0.85),
]
_COST_HINTS = [
    (("cost", "expense", "cogs"), 0.8),
]
_ORDER_VALUE_HINTS = [
    (("order_total", "order_value", "order_amount"), 0.9),
    (("amount", "total"), 0.7),
    (("price",), 0.65),
]
_ORDER_ID_HINTS = [
    (("order_id", "order_number", "invoice_id", "transaction_id"), 0.9),
]
_CUSTOMER_ID_HINTS = [
    (("customer_id", "client_id", "customer_name", "client_name"), 0.88),
    (("customer", "client"), 0.7),
]
_PRODUCT_HINTS = [
    (("product_id", "product_name", "item_name", "sku"), 0.88),
    (("product", "item"), 0.7),
]


@dataclass(frozen=True)
class KpiFinding:
    # Existing fields are intentionally retained for backward compatibility.
    name: str
    value: float | int | str
    trend: str | None
    unit: str | None
    confidence: float
    # Presentation/ranking metadata added without changing any KPI detection rule.
    rank: int = 0
    importance: str = "Medium"
    category: str = "Operations"
    description: str = ""


@dataclass
class KpiContext:
    df: pd.DataFrame
    statistics: dict
    numeric_columns: list = field(default_factory=list)
    period_column: str | None = None


class KpiRule(ABC):
    """Extension point — same shape as RecommendationRule/ChartRule
    elsewhere in this package. Add a new KPI by writing one more class,
    never by editing KpiDetectorEngine itself."""

    name: str = "unnamed_kpi_rule"

    @abstractmethod
    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        raise NotImplementedError


def _clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 2)


def _find_column(columns, hint_groups: list[tuple[tuple[str, ...], float]]) -> tuple[str, float] | None:
    """Returns (column_name, confidence) for the first column matching the
    highest-scoring hint group, or None. `columns` can be any iterable of
    column names/dtypes — id-like columns aren't necessarily numeric or
    necessarily string, so callers pass whichever subset is appropriate."""
    lowered = [(col, str(col).lower()) for col in columns]
    for keywords, confidence in hint_groups:
        for col, low in lowered:
            if any(kw in low for kw in keywords):
                return col, confidence
    return None


def _find_numeric_column(context: KpiContext, hint_groups: list[tuple[tuple[str, ...], float]]) -> tuple[str, float] | None:
    return _find_column(context.numeric_columns, hint_groups)


def _trend_label(context: KpiContext, value_column: str) -> str | None:
    """Reuses trend_detector.detect_trend() rather than computing a trend
    independently — the ONE place this package decides what "Increasing/
    Decreasing/Stable" means. Returns None whenever no period column is
    available or the detector can't produce a result (e.g. too little
    data), never raises."""
    if context.period_column is None:
        return None
    try:
        result = detect_trend(context.df, value_column=value_column, period_column=context.period_column)
    except ValueError:
        return None
    return result["trend"]


def _series_trend_label(context: KpiContext, series: pd.Series, label: str) -> str | None:
    """Same as _trend_label, but for a derived series (e.g. revenue - cost)
    that isn't an actual column on context.df."""
    if context.period_column is None:
        return None
    working = context.df[[context.period_column]].copy()
    working[label] = series
    try:
        result = detect_trend(working, value_column=label, period_column=context.period_column)
    except ValueError:
        return None
    return result["trend"]


def _unique_count(context: KpiContext, column: str) -> int:
    """Prefers an already-computed unique-value count from the `statistics`
    input (supports both main.py's grouped analyze_dataframe shape,
    {"distribution": {"unique_values": {...}}}, and a flat
    {"unique_values": {...}}) over recomputing df[column].nunique() —
    "reuse existing metrics" in the same spirit as the rest of this
    package. Falls back to computing it directly when statistics doesn't
    have it."""
    distribution = context.statistics.get("distribution", {})
    unique_values = distribution.get("unique_values") or context.statistics.get("unique_values")
    if isinstance(unique_values, dict) and column in unique_values:
        return int(unique_values[column])
    return int(context.df[column].nunique())


# ── Rules ────────────────────────────────────────────────────────────────


class RevenueKpiRule(KpiRule):
    name = "revenue"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        match = _find_numeric_column(context, _REVENUE_HINTS)
        if match is None:
            return None
        column, confidence = match
        series = context.df[column].dropna()
        if series.empty:
            return None
        return KpiFinding(
            name="Revenue",
            value=round(float(series.sum()), 2),
            trend=_trend_label(context, column),
            unit="currency",
            confidence=confidence,
        )


class ProfitKpiRule(KpiRule):
    name = "profit"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        direct = _find_numeric_column(context, _PROFIT_HINTS)
        if direct is not None:
            column, confidence = direct
            series = context.df[column].dropna()
            if series.empty:
                return None
            return KpiFinding(
                name="Profit",
                value=round(float(series.sum()), 2),
                trend=_trend_label(context, column),
                unit="currency",
                confidence=confidence,
            )

        # No direct profit-labeled column — derive it as revenue - cost
        # when both are present. A generic formula, not a dataset-specific
        # assumption: any dataset with revenue-like and cost-like columns
        # supports this, regardless of domain.
        revenue_match = _find_numeric_column(context, _REVENUE_HINTS)
        cost_match = _find_numeric_column(context, _COST_HINTS)
        if revenue_match is None or cost_match is None:
            return None
        revenue_column, revenue_confidence = revenue_match
        cost_column, cost_confidence = cost_match
        aligned = context.df[[revenue_column, cost_column]].dropna()
        if aligned.empty:
            return None
        derived = aligned[revenue_column] - aligned[cost_column]
        confidence = _clamp01(min(revenue_confidence, cost_confidence) * 0.9)  # derived -> a bit less confident
        return KpiFinding(
            name="Profit",
            value=round(float(derived.sum()), 2),
            trend=_series_trend_label(context, derived, "_derived_profit"),
            unit="currency",
            confidence=confidence,
        )


class MarginKpiRule(KpiRule):
    name = "margin"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        profit = ProfitKpiRule().evaluate(context)
        revenue_match = _find_numeric_column(context, _REVENUE_HINTS)
        if profit is None or revenue_match is None or not isinstance(profit.value, (int, float)):
            return None
        revenue_column, revenue_confidence = revenue_match
        revenue_total = context.df[revenue_column].dropna().sum()
        if not revenue_total:
            return None
        margin_percent = round(profit.value / revenue_total * 100, 2)
        # Reuses Profit's own trend as an approximation of the margin
        # trend — an exact per-period profit/revenue RATIO trend would
        # need its own series; this is a reasonable, clearly-documented
        # stand-in rather than new, more complex trend machinery.
        return KpiFinding(
            name="Profit Margin",
            value=margin_percent,
            trend=profit.trend,
            unit="%",
            confidence=_clamp01(min(profit.confidence, revenue_confidence) * 0.95),
        )


class AverageOrderValueKpiRule(KpiRule):
    name = "average_order_value"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        value_match = _find_numeric_column(context, _ORDER_VALUE_HINTS) or _find_numeric_column(context, _REVENUE_HINTS)
        if value_match is None:
            return None
        value_column, value_confidence = value_match
        series = context.df[value_column].dropna()
        if series.empty:
            return None

        order_match = _find_column(context.df.columns, _ORDER_ID_HINTS)
        if order_match is not None:
            order_column, order_confidence = order_match
            distinct_orders = _unique_count(context, order_column)
            if not distinct_orders:
                return None
            average = float(series.sum() / distinct_orders)
            confidence = _clamp01(min(value_confidence, order_confidence))
        else:
            # No explicit order identifier -> treat each row as one order.
            # A reasonable default for already-transactional data, but
            # less certain than an explicit order id, hence the discount.
            average = float(series.mean())
            confidence = _clamp01(value_confidence * 0.8)

        return KpiFinding(
            name="Average Order Value",
            value=round(average, 2),
            trend=_trend_label(context, value_column),
            unit="currency",
            confidence=confidence,
        )


class GrowthRateKpiRule(KpiRule):
    name = "growth_rate"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        if context.period_column is None:
            return None
        value_match = _find_numeric_column(context, _REVENUE_HINTS)
        if value_match is None:
            return None
        value_column, confidence = value_match
        try:
            result = detect_trend(context.df, value_column=value_column, period_column=context.period_column)
        except ValueError:
            return None
        if result["growth_percent"] is None:
            return None
        return KpiFinding(
            name="Growth Rate",
            value=result["growth_percent"],
            trend=result["trend"],
            unit="%",
            confidence=confidence,
        )


class TopCustomerKpiRule(KpiRule):
    name = "top_customer"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        customer_match = _find_column(context.df.columns, _CUSTOMER_ID_HINTS)
        if customer_match is None:
            return None
        return _top_entity(context, customer_match, kpi_name="Top Customer")


class TopProductKpiRule(KpiRule):
    name = "top_product"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        product_match = _find_column(context.df.columns, _PRODUCT_HINTS)
        if product_match is None:
            return None
        return _top_entity(context, product_match, kpi_name="Top Product")


def _top_entity(context: KpiContext, entity_match: tuple[str, float], *, kpi_name: str) -> KpiFinding | None:
    """Shared logic behind TopCustomerKpiRule and TopProductKpiRule: rank
    by total value (revenue/order-value-like column) when one is
    available, falling back to plain occurrence count when it isn't."""
    entity_column, entity_confidence = entity_match
    value_match = _find_numeric_column(context, _REVENUE_HINTS) or _find_numeric_column(context, _ORDER_VALUE_HINTS)

    entities = context.df[entity_column].dropna()
    if entities.empty:
        return None

    if value_match is not None:
        value_column, value_confidence = value_match
        totals = context.df.dropna(subset=[entity_column]).groupby(entity_column)[value_column].sum()
        if totals.empty:
            return None
        top_name = totals.idxmax()
        confidence = _clamp01(min(entity_confidence, value_confidence))
    else:
        counts = entities.value_counts()
        if counts.empty:
            return None
        top_name = counts.idxmax()
        confidence = _clamp01(entity_confidence * 0.75)  # ranked by frequency only -> weaker signal

    return KpiFinding(name=kpi_name, value=str(top_name), trend=None, unit=None, confidence=confidence)


class OrderCountKpiRule(KpiRule):
    name = "order_count"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        order_match = _find_column(context.df.columns, _ORDER_ID_HINTS)
        if order_match is not None:
            order_column, confidence = order_match
            value = _unique_count(context, order_column)
            trend = None
            if context.period_column is not None:
                grouped = context.df.groupby(context.period_column)[order_column].nunique().reset_index()
                try:
                    trend = detect_trend(grouped, value_column=order_column, period_column=context.period_column)["trend"]
                except ValueError:
                    trend = None
            return KpiFinding(name="Order Count", value=int(value), trend=trend, unit="count", confidence=confidence)

        # No explicit order identifier column. Falling back to "one row =
        # one order" is only reasonable when there's SOME corroborating
        # sign this is actually transactional data (a revenue/order-value
        # column) — without that, an arbitrary DataFrame with no financial
        # columns at all has no business getting an "Order Count" KPI just
        # because it happens to have rows.
        value_match = _find_numeric_column(context, _REVENUE_HINTS) or _find_numeric_column(context, _ORDER_VALUE_HINTS)
        if value_match is None or len(context.df) == 0:
            return None
        return KpiFinding(name="Order Count", value=len(context.df), trend=None, unit="count", confidence=0.55)


class CustomerCountKpiRule(KpiRule):
    name = "customer_count"

    def evaluate(self, context: KpiContext) -> KpiFinding | None:
        customer_match = _find_column(context.df.columns, _CUSTOMER_ID_HINTS)
        if customer_match is None:
            return None
        customer_column, confidence = customer_match
        value = _unique_count(context, customer_column)
        return KpiFinding(name="Customer Count", value=int(value), trend=None, unit="count", confidence=confidence)




# Importance/category metadata is intentionally separate from detection rules.
# This means automatic KPI discovery and its thresholds remain unchanged.
_KPI_METADATA = {
    "Revenue": {"importance": "Critical", "category": "Revenue"},
    "Profit": {"importance": "Critical", "category": "Profitability"},
    "Profit Margin": {"importance": "High", "category": "Profitability"},
    "Growth Rate": {"importance": "High", "category": "Revenue"},
    "Order Count": {"importance": "High", "category": "Operations"},
    "Customer Count": {"importance": "High", "category": "Customers"},
    "Average Order Value": {"importance": "Medium", "category": "Revenue"},
    "Top Customer": {"importance": "Medium", "category": "Customers"},
    "Top Product": {"importance": "Medium", "category": "Inventory"},
}


def _kpi_metadata(name: str) -> dict:
    return _KPI_METADATA.get(name, {"importance": "Low", "category": "Operations"})


def _kpi_description(finding: KpiFinding) -> str:
    value = finding.value
    if isinstance(value, float):
        value_text = f"{value:,.2f}"
    else:
        value_text = f"{value:,}" if isinstance(value, int) else str(value)
    if finding.unit == "currency":
        value_text = f"{value_text} (currency)"
    elif finding.unit == "%":
        value_text = f"{value_text}%"
    elif finding.unit == "count":
        value_text = f"{value_text} records"

    trend_text = f" Trend: {finding.trend}." if finding.trend else ""
    return f"{finding.name} is {value_text}.{trend_text}"


DEFAULT_RULES: tuple[KpiRule, ...] = (
    RevenueKpiRule(),
    ProfitKpiRule(),
    MarginKpiRule(),
    AverageOrderValueKpiRule(),
    GrowthRateKpiRule(),
    TopCustomerKpiRule(),
    TopProductKpiRule(),
    OrderCountKpiRule(),
    CustomerCountKpiRule(),
)


def _build_context(df: pd.DataFrame | None, statistics: dict | None) -> KpiContext:
    df = df if df is not None else pd.DataFrame()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    period_column = datetime_columns[0] if datetime_columns else _find_period_like_column(df)
    return KpiContext(df=df, statistics=statistics or {}, numeric_columns=numeric_columns, period_column=period_column)


def _find_period_like_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if any(hint in str(column).lower() for hint in _PERIOD_NAME_HINTS):
            return column
    return None


class KpiDetectorEngine:
    """Stateless and reusable — same pattern as RecommendationEngine and
    ChartRecommenderEngine elsewhere in this package. Safe to build once
    and share, or construct fresh per call; pass a custom `rules` list to
    change which KPIs are considered without touching this class."""

    def __init__(self, rules: list[KpiRule] | tuple[KpiRule, ...] | None = None):
        self.rules: list[KpiRule] = list(rules) if rules is not None else list(DEFAULT_RULES)

    def detect(self, df: pd.DataFrame, statistics: dict | None = None) -> list[dict]:
        context = _build_context(df, statistics)
        findings = []
        for rule in self.rules:
            result = rule.evaluate(context)
            if result is not None:
                findings.append(result)

        # Detection above is deliberately unchanged. Ranking/classification is
        # applied only after the existing rules have found their KPIs.
        enriched = []
        for finding in findings:
            metadata = _kpi_metadata(finding.name)
            enriched.append({
                **asdict(finding),
                "importance": metadata["importance"],
                "category": metadata["category"],
                "description": _kpi_description(finding),
            })

        importance_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        enriched.sort(
            key=lambda item: (
                importance_order.get(item["importance"], 99),
                -float(item["confidence"]),
                item["name"],
            )
        )
        for index, item in enumerate(enriched, start=1):
            item["rank"] = index
        return enriched


def detect_kpis(df: pd.DataFrame, statistics: dict | None = None) -> list[dict]:
    """Module-level convenience wrapper around KpiDetectorEngine for a
    one-off call — equivalent to KpiDetectorEngine().detect(df, statistics)."""
    return KpiDetectorEngine().detect(df, statistics)
