# common/insights/trend_detector.py
"""Deterministic trend detection over an ordered pandas Series.

The original trend fields and OLS direction logic are intentionally preserved.
The detector now also exposes confidence, growth rate, detection method,
significance level, and slope while supporting linear, exponential, stable and
basic seasonal patterns.
"""

from datetime import datetime
import math

import pandas as pd
from scipy.stats import t as student_t

STABLE_SLOPE_THRESHOLD = 0.01  # 1% of average value per period


def _label_for_period(value) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%B")
    return str(value)


def _longest_run(flags: list[bool]) -> int:
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _linear_regression(values: pd.Series) -> tuple[float, float, float, float]:
    """Return slope, r2, p-value and standard error for y ~ x."""
    n = len(values)
    if n < 2:
        return 0.0, 0.0, 1.0, 0.0

    x = pd.Series(range(n), dtype=float)
    y = values.astype(float).reset_index(drop=True)
    x_mean, y_mean = x.mean(), y.mean()
    x_centered, y_centered = x - x_mean, y - y_mean
    sxx = float((x_centered ** 2).sum())
    syy = float((y_centered ** 2).sum())
    sxy = float((x_centered * y_centered).sum())
    slope = sxy / sxx if sxx else 0.0

    if syy == 0 or sxx == 0:
        return slope, 1.0 if syy == 0 else 0.0, 1.0, 0.0

    r2 = max(0.0, min(1.0, (sxy * sxy) / (sxx * syy)))
    residual_ss = max(0.0, syy - (sxy * sxy / sxx))
    if n <= 2 or residual_ss == 0:
        p_value = 0.0 if residual_ss == 0 and abs(slope) > 0 else 1.0
        stderr = 0.0
    else:
        stderr = math.sqrt(residual_ss / (n - 2) / sxx)
        t_stat = abs(slope / stderr) if stderr else float("inf")
        p_value = float(2 * student_t.sf(t_stat, n - 2)) if math.isfinite(t_stat) else 0.0
    return float(slope), float(r2), p_value, float(stderr)


def _seasonal_period(values: pd.Series) -> int | None:
    """Basic seasonality check on detrended residuals.

    Looks for a repeated residual pattern at a short lag. Requiring at least
    six observations and a strong residual autocorrelation keeps ordinary
    monotonic trends from being mislabeled as seasonal.
    """
    n = len(values)
    if n < 6:
        return None

    x = pd.Series(range(n), dtype=float)
    slope, _, _, _ = _linear_regression(values)
    intercept = float(values.mean() - slope * x.mean())
    residuals = values.astype(float).reset_index(drop=True) - (intercept + slope * x)

    for lag in range(2, min(n // 2, 12) + 1):
        a = residuals.iloc[:-lag]
        b = residuals.iloc[lag:]
        if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
            continue
        corr = float(a.corr(b))
        if corr >= 0.75:
            return lag
    return None


def _significance_level(p_value: float) -> str:
    if p_value < 0.01:
        return "High"
    if p_value < 0.05:
        return "Medium"
    return "Low"


def _confidence(r2: float, p_value: float, *, stable: bool = False) -> float:
    if stable:
        value = 0.60 + min(0.35, max(0.0, 1.0 - r2) * 0.35)
    else:
        significance = max(0.0, min(1.0, 1.0 - p_value))
        value = 0.50 * r2 + 0.50 * significance
    return round(max(0.0, min(1.0, value)), 2)


def detect_trend(
    df: pd.DataFrame,
    value_column: str,
    period_column: str | None = None,
    *,
    label: str | None = None,
) -> dict:
    """Detect a linear, exponential, stable, or basic seasonal trend.

    Backward-compatible fields remain unchanged. New fields are:
    confidence, growth_rate, method, significance_level, and slope.
    """
    if value_column not in df.columns:
        raise ValueError(f"value_column {value_column!r} is not a column in the given DataFrame")
    if period_column is not None and period_column not in df.columns:
        raise ValueError(f"period_column {period_column!r} is not a column in the given DataFrame")
    if len(df) == 0:
        raise ValueError("cannot detect a trend in an empty DataFrame")

    display_label = label or value_column.replace("_", " ").title()
    working = df[[value_column]].copy()
    working["__period_label__"] = (
        df[period_column].map(_label_for_period) if period_column is not None else df.index.map(_label_for_period)
    )
    working = working.dropna(subset=[value_column])
    values = working[value_column].astype(float).reset_index(drop=True)
    period_labels = working["__period_label__"].reset_index(drop=True)
    n = len(values)

    base = {
        "trend": "Stable",
        "growth_percent": None,
        "decline_percent": 0.0,
        "highest_period": None,
        "lowest_period": None,
        "consecutive_increase": 0,
        "consecutive_decrease": 0,
        "summary": f"No {display_label.lower()} data available to determine a trend.",
        "confidence": 0.0,
        "growth_rate": None,
        "method": "stable",
        "significance_level": "Low",
        "slope": 0.0,
    }
    if n == 0:
        return base

    highest_period = period_labels[values.idxmax()]
    lowest_period = period_labels[values.idxmin()]
    base.update({"highest_period": highest_period, "lowest_period": lowest_period})

    if n == 1:
        base.update({
            "growth_percent": 0.0,
            "growth_rate": 0.0,
            "confidence": 0.60,
            "summary": f"Only one period of {display_label.lower()} data is available, so no trend can be determined.",
        })
        return base

    first_value, last_value = float(values.iloc[0]), float(values.iloc[-1])
    growth_percent = None if first_value == 0 else round((last_value - first_value) / abs(first_value) * 100, 1)

    slope, linear_r2, p_value, _ = _linear_regression(values)
    mean_value = float(values.mean())
    relative_slope = slope / abs(mean_value) if mean_value else 0.0

    # Preserve the existing OLS-based direction decision.
    if abs(relative_slope) < STABLE_SLOPE_THRESHOLD:
        trend = "Stable"
    elif relative_slope > 0:
        trend = "Increasing"
    else:
        trend = "Decreasing"

    method = "stable" if trend == "Stable" else "linear_regression"
    method_r2 = linear_r2

    # Basic seasonal detection is intentionally conservative and does not
    # replace the original direction label.
    seasonal_period = _seasonal_period(values)
    if seasonal_period is not None:
        method = "seasonal_basic"

    # Detect exponential growth only when all values are positive and the
    # log-linear fit explains the series materially better than linear fit.
    if trend == "Increasing" and (values > 0).all():
        log_values = values.map(math.log)
        _, log_r2, _, _ = _linear_regression(log_values)
        if log_r2 >= 0.80 and log_r2 > linear_r2 + 0.05:
            method = "exponential_growth"
            method_r2 = log_r2

    significance = _significance_level(p_value)
    confidence = _confidence(method_r2, p_value, stable=(trend == "Stable"))

    decline_percent = round(abs(growth_percent), 1) if (trend == "Decreasing" and growth_percent is not None) else 0.0
    diffs = values.diff().dropna()
    consecutive_increase = _longest_run((diffs > 0).tolist())
    consecutive_decrease = _longest_run((diffs < 0).tolist())

    total_steps = n - 1
    if trend == "Increasing":
        summary = (
            f"{display_label} increased consistently throughout the period."
            if consecutive_increase == total_steps
            else f"{display_label} increased overall throughout the period, with some fluctuation."
        )
    elif trend == "Decreasing":
        summary = (
            f"{display_label} decreased consistently throughout the period."
            if consecutive_decrease == total_steps
            else f"{display_label} decreased overall throughout the period, with some fluctuation."
        )
    else:
        summary = f"{display_label} remained stable throughout the period."

    return {
        "trend": trend,
        "growth_percent": growth_percent,
        "decline_percent": decline_percent,
        "highest_period": highest_period,
        "lowest_period": lowest_period,
        "consecutive_increase": consecutive_increase,
        "consecutive_decrease": consecutive_decrease,
        "summary": summary,
        "confidence": confidence,
        "growth_rate": growth_percent,
        "method": method,
        "significance_level": significance,
        "slope": round(slope, 6),
    }
