"""Statistics utilities for dataset quality assessment.

This module intentionally adds only a Data Quality Score. It does not add
new descriptive statistics or business recommendations.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


_DATE_NAME_HINTS = (
    "date",
    "time",
    "timestamp",
    "datetime",
    "dob",
    "created",
    "updated",
    "modified",
)


def _is_missing(value: Any) -> bool:
    """Treat nulls and whitespace-only strings as missing."""
    if pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.isna()
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            mask[column] = df[column].map(_is_missing)
    return mask


def _looks_like_date_column(series: pd.Series, column_name: Any) -> bool:
    """Identify likely date columns without treating ordinary numeric columns as dates."""
    name = str(column_name).strip().lower()
    if any(hint in name for hint in _DATE_NAME_HINTS):
        return True

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False

    non_missing = series.dropna().astype(str).str.strip()
    non_missing = non_missing[non_missing != ""]
    if non_missing.empty:
        return False

    parsed = pd.to_datetime(non_missing, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= 0.80


def _invalid_date_cells(df: pd.DataFrame, missing_mask: pd.DataFrame) -> tuple[int, int]:
    """Return invalid and applicable date-cell counts.

    Missing date values are not counted as invalid dates because they are
    already accounted for by the Missing Values component.
    """
    invalid = 0
    applicable = 0

    for column in df.columns:
        series = df[column]
        if not _looks_like_date_column(series, column):
            continue

        valid_population = ~missing_mask[column]
        values = series.loc[valid_population]
        applicable += int(len(values))
        if values.empty:
            continue

        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        invalid += int(parsed.isna().sum())

    return invalid, applicable


def _quality_grade(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    if score >= 40:
        return "Poor"
    return "Critical"


def calculate_data_quality_score(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate a 0-100 data quality score.

    The score is the equal-weighted average of four quality dimensions:
    missing values, duplicate rows, invalid dates, and completely empty
    columns. A clean dataset scores 100.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    rows, columns = df.shape
    if rows == 0 and columns == 0:
        return {
            "quality_score": 100.0,
            "quality_grade": "Excellent",
            "quality_summary": "Dataset is empty, so no quality issues were detected.",
        }

    missing_mask = _missing_mask(df)
    total_cells = rows * columns
    missing_cells = int(missing_mask.sum().sum()) if total_cells else 0
    missing_rate = missing_cells / total_cells if total_cells else 0.0

    duplicate_rate = float(df.duplicated(keep=False).sum() / rows) if rows else 0.0

    invalid_date_cells, applicable_date_cells = _invalid_date_cells(df, missing_mask)
    invalid_date_rate = (
        invalid_date_cells / applicable_date_cells if applicable_date_cells else 0.0
    )

    empty_columns = 0
    if columns:
        for column in df.columns:
            if bool(missing_mask[column].all()):
                empty_columns += 1
    empty_column_rate = empty_columns / columns if columns else 0.0

    component_scores = [
        100.0 * (1.0 - missing_rate),
        100.0 * (1.0 - duplicate_rate),
        100.0 * (1.0 - invalid_date_rate),
        100.0 * (1.0 - empty_column_rate),
    ]
    score = round(max(0.0, min(100.0, sum(component_scores) / len(component_scores))), 2)
    grade = _quality_grade(score)

    issues = []
    if missing_cells:
        issues.append(f"{missing_cells} missing value(s)")
    duplicate_rows = int(df.duplicated(keep=False).sum()) if rows else 0
    if duplicate_rows:
        issues.append(f"{duplicate_rows} duplicate row(s)")
    if invalid_date_cells:
        issues.append(f"{invalid_date_cells} invalid date value(s)")
    if empty_columns:
        issues.append(f"{empty_columns} empty column(s)")

    if issues:
        summary = f"Data quality score is {score:.2f}/100 ({grade}). Issues detected: " + ", ".join(issues) + "."
    else:
        summary = f"Data quality score is {score:.2f}/100 ({grade}). No missing values, duplicates, invalid dates, or empty columns detected."

    return {
        "quality_score": score,
        "quality_grade": grade,
        "quality_summary": summary,
    }


# Convenient aliases for callers that use service-style naming.
data_quality_score = calculate_data_quality_score
get_data_quality_score = calculate_data_quality_score
