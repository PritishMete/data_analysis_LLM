# common/insights/service.py
# ─────────────────────────────────────────────────────────────────────────────
# Thin integration layer over trend_detector.py and outlier_detector.py —
# the one place that wraps their plain functions for callers (ai_engine.py
# today; a future route, tomorrow), rather than every caller reaching into
# each detector module directly. Both detector modules stay
# dependency-free, pure Python + pandas functions; this class adds no logic
# of its own beyond delegating.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

from .outlier_detector import detect_outliers
from .schemas import TrendInsight
from .trend_detector import detect_trend


class InsightsService:
    """Stateless on purpose — no database, no persistence, no config beyond
    what's passed to each call. Both trend detection and outlier detection
    are cheap enough to run fresh every time they're needed, so there's
    nothing here to cache or store."""

    def detect_trend(
        self,
        df: pd.DataFrame,
        value_column: str,
        period_column: str | None = None,
        *,
        label: str | None = None,
    ) -> TrendInsight:
        result = detect_trend(df, value_column, period_column, label=label)
        return TrendInsight(**result)

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        methods: tuple[str, ...] = ("iqr", "zscore"),
        **kwargs,
    ) -> list[dict]:
        """Returns the plain list-of-dicts outlier_detector.detect_outliers
        produces (column/method/outlier_count/percentage/severity/examples/
        summary per entry) — not wrapped in a Pydantic model, since no
        schema for it currently exists in schemas.py and every current
        caller (ai_engine.py) consumes it as plain dicts anyway."""
        return detect_outliers(df, columns=columns, methods=methods, **kwargs)
