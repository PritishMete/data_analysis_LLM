# common/insights/outlier_detector.py
# ─────────────────────────────────────────────────────────────────────────────
# Deterministic outlier detection over a pandas DataFrame — Python + pandas
# only, same spirit as trend_detector.py in this package: no model, no LLM
# call, no external service. Given the same DataFrame and method twice,
# this always returns the exact same result.
#
# Two independent, well-known statistical methods, each usable on its own:
#   - IQR (interquartile range): flags values outside
#     [Q1 - multiplier*IQR, Q3 + multiplier*IQR]. Robust to skewed
#     distributions and doesn't assume normality.
#   - Z-score: flags values more than `threshold` standard deviations from
#     the mean. Simple and fast, but assumes an approximately normal
#     distribution — which is exactly why both methods are offered rather
#     than picking one: they can (and often do) disagree, and a caller
#     comparing both gets a more complete picture than either alone.
#
# Reusable by design: every function here is a plain, stateless function
# (or a thin wrapper in InsightsService — see service.py) that takes a
# DataFrame and returns plain dicts. No class to instantiate, no
# configuration to set up beforehand, safe to call from anywhere —
# ai_engine.py's insights pipeline today, any future route or batch job
# tomorrow.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

DEFAULT_IQR_MULTIPLIER = 1.5
DEFAULT_ZSCORE_THRESHOLD = 3.0
MAX_EXAMPLES = 5
MIN_POINTS_FOR_IQR = 4
MIN_POINTS_FOR_ZSCORE = 2

# Percentage-of-values-flagged thresholds for the "severity" bucket.
# Business-judgment thresholds, not fit from data — same style as
# trend_detector.py's STABLE_SLOPE_THRESHOLD and
# recommendation_engine.py's magnitude thresholds.
SEVERITY_MEDIUM_AT = 5.0
SEVERITY_HIGH_AT = 10.0
SEVERITY_CRITICAL_AT = 20.0


def _severity(percentage: float) -> str:
    """Map the percentage of flagged values to a business-facing severity."""
    if percentage >= SEVERITY_CRITICAL_AT:
        return "Critical"
    if percentage >= SEVERITY_HIGH_AT:
        return "High"
    if percentage >= SEVERITY_MEDIUM_AT:
        return "Medium"
    return "Low"


def _recommended_action(severity: str, outlier_count: int) -> str:
    """Return a deterministic action suitable for a business-facing UI."""
    if outlier_count == 0:
        return "Ignore"
    if severity in {"High", "Critical"}:
        return "Validate source"
    return "Review transaction"


def _confidence(method: str, percentage: float, outlier_count: int) -> float:
    """Return a deterministic confidence score for the detected finding.

    IQR is generally more robust to skew and extreme values, so it receives
    a slightly higher base confidence than Z-score. The score is intentionally
    conservative when no observations are flagged because an absence of
    outliers is weaker evidence than a clear statistical finding.
    """
    base = 0.95 if method == "iqr" else 0.90
    if outlier_count == 0:
        base -= 0.05
    elif percentage >= SEVERITY_CRITICAL_AT:
        base += 0.02
    return round(min(base, 0.99), 2)


def _json_safe(value):
    """numpy scalar (np.float64, np.int64, ...) -> plain Python scalar, so
    the result is safe to json.dumps directly without a custom encoder."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_finding(column: str, method: str, outlier_values: list, total: int) -> dict:
    outlier_count = len(outlier_values)
    percentage = round(outlier_count / total * 100, 2) if total else 0.0
    severity = _severity(percentage)
    recommended_action = _recommended_action(severity, outlier_count)
    confidence = _confidence(method, percentage, outlier_count)
    examples = [_json_safe(v) for v in outlier_values[:MAX_EXAMPLES]]

    if outlier_count == 0:
        summary = f"No outliers detected in '{column}' using the {method.upper()} method."
    else:
        summary = (
            f"{outlier_count} outlier(s) detected in '{column}' using the {method.upper()} method "
            f"({percentage}% of values) — severity: {severity}."
        )

    return {
        "column": str(column),
        "method": method,
        "outlier_count": outlier_count,
        "percentage": percentage,
        "severity": severity,
        "recommended_action": recommended_action,
        "examples": examples,
        "summary": summary,
        "confidence": confidence,
    }


def _resolve_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if columns is None:
        return df.select_dtypes(include="number").columns.tolist()
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"column {column!r} is not a column in the given DataFrame")
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"column {column!r} is not numeric; outlier detection requires numeric data")
    return list(columns)


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> list[dict]:
    """IQR-method outlier findings, one dict per analyzed column (see
    module docstring for the exact shape). `columns` defaults to every
    numeric column in `df`; pass an explicit list to restrict analysis.
    Columns with fewer than 4 non-null values, or with zero spread
    (Q1 == Q3), are skipped — IQR is undefined/meaningless there.
    """
    findings = []
    for column in _resolve_columns(df, columns):
        series = df[column].dropna()
        if len(series) < MIN_POINTS_FOR_IQR:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower_bound, upper_bound = q1 - multiplier * iqr, q3 + multiplier * iqr
        outlier_values = series[(series < lower_bound) | (series > upper_bound)].tolist()
        findings.append(_build_finding(column, "iqr", outlier_values, len(series)))
    return findings


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = DEFAULT_ZSCORE_THRESHOLD,
) -> list[dict]:
    """Z-score-method outlier findings, one dict per analyzed column (see
    module docstring for the exact shape). `columns` defaults to every
    numeric column in `df`; pass an explicit list to restrict analysis.
    Columns with fewer than 2 non-null values, or with zero standard
    deviation (every value identical), are skipped — z-score is
    undefined/meaningless there.
    """
    findings = []
    for column in _resolve_columns(df, columns):
        series = df[column].dropna()
        if len(series) < MIN_POINTS_FOR_ZSCORE:
            continue
        std = series.std()
        if not std or pd.isna(std):
            continue
        mean = series.mean()
        z_scores = (series - mean).abs() / std
        outlier_values = series[z_scores > threshold].tolist()
        findings.append(_build_finding(column, "zscore", outlier_values, len(series)))
    return findings


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    methods: tuple[str, ...] = ("iqr", "zscore"),
    *,
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER,
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
) -> list[dict]:
    """Runs the requested methods (default: both) and returns their
    findings concatenated — one entry per (column, method) pair, so a
    caller can see where IQR and Z-score agree or disagree on the same
    column rather than only getting one method's opinion silently.
    """
    findings: list[dict] = []
    if "iqr" in methods:
        findings.extend(detect_outliers_iqr(df, columns, multiplier=iqr_multiplier))
    if "zscore" in methods:
        findings.extend(detect_outliers_zscore(df, columns, threshold=zscore_threshold))
    return findings
