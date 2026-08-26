# common/insights/recommendation_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Rule-based business recommendation engine. Consumes already-computed
# analytics — statistics, a trend read (the exact shape
# common.insights.trend_detector.detect_trend() returns), KPIs, and an
# outlier report — and produces prioritized, human-readable recommendations.
#
# This module computes NOTHING itself: no DataFrame ever touches this file,
# and it never calls Gemini, any other LLM, or any ML model. Every
# recommendation comes from a fixed threshold rule below, so the same
# inputs always produce the same output — reproducible, auditable, and free
# to run inline with no external API call and no training.
#
# Reusable by construction: RecommendationEngine takes a list of
# RecommendationRule objects (defaulting to DEFAULT_RULES below). Adding a
# new kind of recommendation is "write one more class that implements
# evaluate()", never a change to the engine itself — the same pattern
# schema_intelligence/rules/ already uses for column-role detection.
#
# ── Input contract ──────────────────────────────────────────────────────
# Every input is optional; a rule that needs data it wasn't given simply
# declines (returns None) rather than raising, so callers can pass whatever
# subset of analytics they actually have.
#
#   statistics: dict, optional. Any keys are accepted; the rules below
#       specifically look for:
#         "missing_percentage": {column_name: percent_missing (0-100), ...}
#       — e.g. from ai_engine.py's own missing-value report. Additional
#       keys (like a per-column describe()-style dict) are ignored by the
#       current rules but are accepted without error, so a caller can pass
#       its full statistics payload as-is.
#
#   trend: dict, optional. Either:
#         (a) a single detect_trend() dict, i.e. it has a top-level "trend"
#             key — used as-is, treated as "the metric being asked about"
#             whatever it's named, or
#         (b) {"revenue": {...detect_trend() dict...}, "profit_margin": {...}, ...}
#             for multiple metrics at once.
#       See _normalize_trend_input()/_select_trend() below for exactly how
#       a rule picks the metric it cares about out of either shape.
#
#   kpis: dict, optional. Flat, e.g.:
#         {"profit_margin_change_percent": -3.1,
#          "top_customer_revenue_share_percent": 62.0}
#       Any KPI-producing part of the backend can hand this engine
#       whatever subset of these keys it has; unrecognized/absent keys are
#       ignored by whichever rule doesn't use them.
#
#   outliers: dict, optional. Per-column, e.g.:
#         {"revenue": {"outlier_count": 5, "outlier_percentage": 8.2}, ...}
# ─────────────────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Literal

Priority = Literal["High", "Medium", "Low"]
Category = Literal["Revenue", "Profitability", "Customers", "Inventory", "Data Quality", "Operations"]
Impact = Literal["High", "Medium", "Low"]


@dataclass(frozen=True)
class Recommendation:
    category: Category
    priority: Priority
    confidence: float  # 0.0-1.0 — derived from the triggering signal's magnitude, never arbitrary
    impact: Impact
    recommendation: str
    reason: str


@dataclass
class RecommendationContext:
    """Bundles the four documented input types for a single generate()
    call. See module docstring for the exact shape of each field."""

    statistics: dict = field(default_factory=dict)
    trend: dict = field(default_factory=dict)  # always normalized to {metric_name: detect_trend()-dict, ...}
    kpis: dict = field(default_factory=dict)
    outliers: dict = field(default_factory=dict)


class RecommendationRule(ABC):
    """The extension point. A rule reads whatever it needs off the
    RecommendationContext and either returns one Recommendation or
    declines (None) — never raises for "this data wasn't provided", so one
    rule's missing input never breaks another rule's evaluation."""

    name: str = "unnamed_recommendation_rule"

    @abstractmethod
    def evaluate(self, context: RecommendationContext) -> Recommendation | None:
        raise NotImplementedError


def _normalize_trend_input(trend: dict | None) -> dict:
    """Accepts either input shape (a) or (b) from the module docstring and
    always returns shape (b) — {metric_name: detect_trend()-dict}."""
    if not trend:
        return {}
    if "trend" in trend:  # looks like a raw detect_trend() dict, not {metric: {...}}
        return {"_primary": trend}
    return trend


def _select_trend(trend: dict, preferred_names: tuple[str, ...]) -> dict | None:
    """Picks the trend a particular rule cares about (e.g. revenue) out of
    the normalized multi-metric trend dict. Prefers an exact
    (case-insensitive) name match; falls back to the caller's single
    UNNAMED trend (the "_primary" sentinel _normalize_trend_input sets
    when it was handed a raw detect_trend() dict with no metric name at
    all) — but never to some other, explicitly-named single metric. If a
    caller names their one trend "revenue", that name is a real signal:
    ProfitMarginDeclineRule must NOT claim it just because it's the only
    entry present. Returns None when nothing usable is found."""
    if not trend:
        return None
    lowered = {str(k).lower(): v for k, v in trend.items()}
    for name in preferred_names:
        if name in lowered:
            return lowered[name]
    if "_primary" in lowered:
        return lowered["_primary"]
    return None


def _scale_confidence(value: float, *, low: float, high: float, floor: float = 0.55, ceiling: float = 0.95) -> float:
    """Maps `value` linearly from [low, high] to [floor, ceiling], clamped
    at both ends. Deterministic, no ML: a bigger deviation from normal
    always yields a higher confidence, never lower — and confidence never
    claims more certainty than `ceiling` or less than `floor` (a rule that
    fired at all has SOME basis, so it's never reported near-zero)."""
    if high <= low:
        return floor
    fraction = (value - low) / (high - low)
    fraction = max(0.0, min(1.0, fraction))
    return round(floor + fraction * (ceiling - floor), 2)


def _priority_from_magnitude(value: float, *, medium_at: float, high_at: float) -> Priority:
    if value >= high_at:
        return "High"
    if value >= medium_at:
        return "Medium"
    return "Low"


def _impact_from_priority(priority: Priority) -> Impact:
    """Maps the existing rule severity to the user-facing impact field.

    The existing rule thresholds remain unchanged; this only exposes their
    already-derived priority as a second, explicit impact dimension.
    """
    return priority


# ── Rules ────────────────────────────────────────────────────────────────


class RevenueDeclineRule(RecommendationRule):
    """Revenue trending down -> improve marketing strategy (the example
    given in the spec)."""

    name = "revenue_decline"
    METRIC_NAMES = ("revenue", "sales", "total_revenue", "income")

    def evaluate(self, context: RecommendationContext) -> Recommendation | None:
        trend = _select_trend(context.trend, self.METRIC_NAMES)
        if trend is None or trend.get("trend") != "Decreasing":
            return None
        decline = trend.get("decline_percent") or 0.0
        priority = _priority_from_magnitude(decline, medium_at=5.0, high_at=15.0)
        return Recommendation(
            category="Revenue",
            priority=priority,
            recommendation="Improve marketing strategy",
            reason=f"Revenue is trending downward, declining {decline}% over the analyzed period.",
            impact=_impact_from_priority(priority),
            confidence=_scale_confidence(decline, low=2.0, high=25.0),
        )


class ProfitMarginDeclineRule(RecommendationRule):
    """Profit margin trending down, or down vs. a prior period per KPIs ->
    review pricing (the example given in the spec). Prefers an explicit
    trend read when one is given; falls back to a KPI-reported change."""

    name = "profit_margin_decline"
    METRIC_NAMES = ("profit_margin", "margin", "gross_margin", "net_margin")
    KPI_KEY = "profit_margin_change_percent"

    def evaluate(self, context: RecommendationContext) -> Recommendation | None:
        trend = _select_trend(context.trend, self.METRIC_NAMES)
        if trend is not None and trend.get("trend") == "Decreasing":
            decline = trend.get("decline_percent") or 0.0
            reason = f"Profit margin is trending downward, declining {decline}% over the analyzed period."
        else:
            change = context.kpis.get(self.KPI_KEY)
            if change is None or change >= 0:
                return None
            decline = abs(change)
            reason = f"Profit margin fell by {decline}% compared to the prior period."
        priority = _priority_from_magnitude(decline, medium_at=3.0, high_at=10.0)
        return Recommendation(
            category="Profitability",
            priority=priority,
            recommendation="Review pricing strategy",
            reason=reason,
            impact=_impact_from_priority(priority),
            confidence=_scale_confidence(decline, low=1.0, high=20.0),
        )


class HighMissingValuesRule(RecommendationRule):
    """A column with a high missing-value rate -> improve data quality (the
    example given in the spec)."""

    name = "high_missing_values"
    THRESHOLD_PERCENT = 20.0

    def evaluate(self, context: RecommendationContext) -> Recommendation | None:
        missing = context.statistics.get("missing_percentage")
        if not missing:
            return None
        worst_column, worst_value = max(missing.items(), key=lambda kv: kv[1])
        if worst_value < self.THRESHOLD_PERCENT:
            return None
        affected = sorted(c for c, v in missing.items() if v >= self.THRESHOLD_PERCENT)
        priority = _priority_from_magnitude(worst_value, medium_at=30.0, high_at=50.0)
        return Recommendation(
            category="Data Quality",
            priority=priority,
            recommendation="Improve data quality",
            reason=(
                f"{len(affected)} column(s) have high missing-value rates "
                f"(worst: '{worst_column}' at {worst_value}% missing)."
            ),
            impact=_impact_from_priority(priority),
            confidence=_scale_confidence(worst_value, low=20.0, high=80.0),
        )


class CustomerConcentrationRule(RecommendationRule):
    """A small number of customers account for most revenue -> diversify
    customer base (the example given in the spec)."""

    name = "customer_concentration"
    KPI_KEYS = ("top_customer_revenue_share_percent", "top_customer_share_percent", "customer_concentration_percent")
    THRESHOLD_PERCENT = 40.0

    def evaluate(self, context: RecommendationContext) -> Recommendation | None:
        share = next((context.kpis[key] for key in self.KPI_KEYS if key in context.kpis), None)
        if share is None or share < self.THRESHOLD_PERCENT:
            return None
        priority = _priority_from_magnitude(share, medium_at=50.0, high_at=60.0)
        return Recommendation(
            category="Customers",
            priority=priority,
            recommendation="Diversify customer base",
            reason=(
                f"A single customer (or small group) accounts for {share}% of revenue, "
                "a significant concentration risk."
            ),
            impact=_impact_from_priority(priority),
            confidence=_scale_confidence(share, low=40.0, high=80.0),
        )


class OutlierQualityRule(RecommendationRule):
    """A column with an unusually high rate of statistical outliers ->
    investigate/clean the data. Not one of the spec's named examples, but a
    direct, natural use of the "Outliers" input this engine explicitly
    accepts."""

    name = "outlier_quality"
    THRESHOLD_PERCENT = 10.0

    def evaluate(self, context: RecommendationContext) -> Recommendation | None:
        if not context.outliers:
            return None
        worst_column, worst_pct = None, 0.0
        for column, report in context.outliers.items():
            pct = report.get("outlier_percentage", 0.0) if isinstance(report, dict) else 0.0
            if pct > worst_pct:
                worst_column, worst_pct = column, pct
        if worst_column is None or worst_pct < self.THRESHOLD_PERCENT:
            return None
        priority = _priority_from_magnitude(worst_pct, medium_at=15.0, high_at=25.0)
        return Recommendation(
            category="Data Quality",
            priority=priority,
            recommendation="Investigate and clean outlier records",
            reason=(
                f"Column '{worst_column}' has an unusually high rate of outliers "
                f"({worst_pct}% of values), which may distort analysis."
            ),
            impact=_impact_from_priority(priority),
            confidence=_scale_confidence(worst_pct, low=10.0, high=40.0),
        )


DEFAULT_RULES: tuple[RecommendationRule, ...] = (
    RevenueDeclineRule(),
    ProfitMarginDeclineRule(),
    HighMissingValuesRule(),
    CustomerConcentrationRule(),
    OutlierQualityRule(),
)

_PRIORITY_ORDER: dict[Priority, int] = {"High": 0, "Medium": 1, "Low": 2}


class RecommendationEngine:
    """Stateless and reusable: safe to build once and share across
    requests/datasets/organizations (rules carry no per-call state), or to
    construct fresh each time — either way behaves identically. Pass a
    custom `rules` list to run a different rule set entirely (e.g. a
    subset, or additional organization-specific rules alongside
    DEFAULT_RULES) without touching this class."""

    def __init__(self, rules: list[RecommendationRule] | tuple[RecommendationRule, ...] | None = None):
        self.rules: list[RecommendationRule] = list(rules) if rules is not None else list(DEFAULT_RULES)

    def generate(
        self,
        *,
        statistics: dict | None = None,
        trend: dict | None = None,
        kpis: dict | None = None,
        outliers: dict | None = None,
    ) -> list[dict]:
        """Evaluates every rule against the given inputs and returns the
        triggered recommendations as plain dicts (priority/recommendation/
        reason/confidence — JSON-serializable, no dataclass leaking out),
        sorted by priority (high first) and then by confidence (highest
        first) within a priority tier, so the most actionable item is
        always first."""
        context = RecommendationContext(
            statistics=statistics or {},
            trend=_normalize_trend_input(trend),
            kpis=kpis or {},
            outliers=outliers or {},
        )
        triggered: list[Recommendation] = []
        for rule in self.rules:
            result = rule.evaluate(context)
            if result is not None:
                triggered.append(result)
        triggered.sort(key=lambda r: (_PRIORITY_ORDER[r.priority], -r.confidence))
        return [asdict(r) for r in triggered]
