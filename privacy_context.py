"""Centralized privacy helpers for AI/planner calls.

The backend runs in local-only mode by default. When strict privacy is
enabled, Gemini-facing payloads are anonymized to field IDs and structural
metadata only. Real column names and any row-level values stay on the local
side of the application.
"""

from __future__ import annotations

from typing import Any
import re

import pandas as pd

try:
    from privacy_policy import LOCAL_ONLY
except Exception:  # pragma: no cover - defensive fallback
    LOCAL_ONLY = True

try:
    from secure_excel.semantic_roles import detect_column_role
except Exception:  # pragma: no cover - optional dependency path
    detect_column_role = None

PRIVACY_MODE = "local_only" if LOCAL_ONLY else "remote_allowed"


def strict_enabled() -> bool:
    return bool(LOCAL_ONLY)


def _safe_id(index: int) -> str:
    return f"FIELD_{index + 1:02d}"


def safe_columns(columns):
    """Return anonymized column labels plus forward/backward mappings."""
    original = [str(c) for c in columns]
    safe = [_safe_id(i) for i in range(len(original))]
    forward = {real: anon for real, anon in zip(original, safe)}
    reverse = {anon: real for real, anon in zip(original, safe)}
    return safe, forward, reverse


def _replace_tokens(text: str, replacements: dict[str, str]) -> str:
    out = str(text or "")
    if not replacements:
        return out
    for source in sorted(replacements, key=len, reverse=True):
        target = replacements[source]
        if not source:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(source) + r"(?!\w)", re.IGNORECASE)
        out = pattern.sub(target, out)
    return out


def sanitize_user_text(text: str, columns: list[str] | None = None, df: pd.DataFrame | None = None) -> str:
    """Redact obvious PII and replace real column names with field IDs."""
    safe_text = str(text or "")
    if not strict_enabled():
        return safe_text

    # Replace obvious contact data before any planner sees the request.
    safe_text = re.sub(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "<EMAIL>", safe_text, flags=re.I)
    safe_text = re.sub(r"https?://\S+|www\.\S+", "<URL>", safe_text, flags=re.I)
    safe_text = re.sub(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", "<PHONE>", safe_text)

    source_columns = columns if columns is not None else (list(df.columns) if df is not None else [])
    _, forward, _ = safe_columns(source_columns)
    return _replace_tokens(safe_text, forward)


def remap_plan(plan, reverse_map: dict[str, str] | None = None):
    """Recursively remap anonymized field IDs back to real column names."""
    if not reverse_map:
        return plan

    def _remap(value: Any):
        if isinstance(value, dict):
            return {(_remap(k) if isinstance(k, str) else k): _remap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_remap(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_remap(v) for v in value)
        if isinstance(value, str):
            return reverse_map.get(value, value)
        return value

    return _remap(plan)


def value_aliases(df: pd.DataFrame):
    """Placeholder for value-level alias remapping.

    The current privacy layer keeps values local and avoids sending them to
    Gemini, so the mapping remains empty. The hook is preserved for future
    value-safe planners that may need reversible symbolic labels.
    """
    return {}, {}


def dataframe_profile(df: pd.DataFrame, include_samples: bool = True) -> dict:
    """Build a Gemini-safe dataset profile.

    In strict mode the profile is anonymized to field IDs and structural
    metadata only. Outside strict mode the legacy verbose profile is kept for
    backwards compatibility.
    """
    if strict_enabled():
        safe, forward, _ = safe_columns(df.columns)
        columns = []
        for idx, col in enumerate(df.columns):
            series = df[col]
            role_info = None
            if detect_column_role is not None:
                try:
                    role_info = detect_column_role(str(col), series)
                except Exception:
                    role_info = None
            columns.append(
                {
                    "column_id": safe[idx],
                    "role": role_info["role"] if role_info else "unknown",
                    "confidence": role_info["confidence"] if role_info else 0.5,
                    "dtype": str(series.dtype),
                    "missing_ratio": round(float(series.isna().mean()), 4),
                    "unique_ratio": round(float(series.nunique(dropna=True) / len(series.dropna())) if len(series.dropna()) else 0.0, 4),
                }
            )
        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": columns,
            "summary": {
                "duplicate_rows": int(df.duplicated().sum()),
                "total_missing": int(df.isna().sum().sum()),
            },
            "anonymized": True,
        }

    profile = {
        "columns": list(df.columns),
        "row_count": int(len(df)),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }
    if include_samples:
        profile["samples"] = {
            str(c): df[c].dropna().astype(str).head(5).tolist()
            for c in df.columns
        }
    return profile
