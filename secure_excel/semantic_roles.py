"""Local semantic role detection for Excel columns.

All reasoning happens against the in-memory DataFrame only. No column names or
cell values are sent outside the process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import re

import numpy as np
import pandas as pd


ROLE_PRIORITY = [
    "identifier",
    "restaurant_entity",
    "customer_entity",
    "product_entity",
    "supplier_entity",
    "employee_entity",
    "geographic_area",
    "date",
    "rating_metric",
    "currency_metric",
    "numeric_metric",
    "count",
    "percentage",
    "email",
    "phone",
    "url",
    "address",
    "boolean_capability",
    "table_booking_capability",
    "delivery_capability",
    "status",
    "category",
    "description",
    "unknown",
]


NAME_HINTS = {
    "identifier": {"id", "identifier", "uid", "uuid", "code", "key", "ref", "reference"},
    "restaurant_entity": {"restaurant", "restaurantname", "restaurant_name", "cafe", "diner", "outlet", "branch", "store"},
    "customer_entity": {"customer", "client", "buyer", "account"},
    "product_entity": {"product", "item", "sku", "goods", "service"},
    "supplier_entity": {"supplier", "vendor", "partner"},
    "employee_entity": {"employee", "staff", "worker", "agent", "associate"},
    "geographic_area": {"city", "area", "location", "locality", "region", "state", "country", "zone", "place"},
    "date": {"date", "time", "created", "updated", "month", "year", "day"},
    "rating_metric": {"rating", "score", "stars", "reviewscore", "review_rating"},
    "currency_metric": {"amount", "price", "revenue", "sales", "total", "cost", "income", "salary", "profit"},
    "numeric_metric": {"value", "metric", "qty", "quantity", "count", "number", "num"},
    "count": {"count", "totalcount", "visits", "orders", "items"},
    "percentage": {"pct", "percent", "percentage", "ratio", "share"},
    "email": {"email", "mail"},
    "phone": {"phone", "mobile", "contact", "tel"},
    "url": {"url", "website", "link", "site"},
    "address": {"address", "addr", "street", "road", "postal", "zip"},
    "boolean_capability": {"available", "status", "flag", "enabled", "active", "has"},
    "table_booking_capability": {"booking", "bookable", "reservation", "table"},
    "delivery_capability": {"delivery", "deliver", "dispatch", "takeaway", "online"},
    "status": {"status", "state", "phase"},
    "category": {"category", "type", "segment", "class", "group", "kind"},
    "description": {"description", "details", "notes", "comment", "review", "summary"},
}

BOOLEAN_LIKE = {"yes", "no", "true", "false", "y", "n", "1", "0", "available", "not available", "open", "closed"}
BOOLEAN_TRUE = {"yes", "true", "y", "1", "available", "open", "on", "enabled"}
BOOLEAN_FALSE = {"no", "false", "n", "0", "not available", "closed", "off", "disabled"}


def _clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _name_tokens(name: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
    return {token for token in cleaned.split() if token}


def _series_text(series: pd.Series, limit: int = 50) -> list[str]:
    values = series.dropna().astype(str).head(limit).tolist()
    return [value.strip() for value in values if value.strip()]


def _datetime_ratio(series: pd.Series) -> float:
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0 if len(series) else 0.0
    if pd.api.types.is_numeric_dtype(series):
        return 0.0
    parsed = pd.to_datetime(series, errors="coerce")
    return float(parsed.notna().mean()) if len(series) else 0.0


def _numeric_ratio(series: pd.Series) -> float:
    converted = pd.to_numeric(series, errors="coerce")
    return float(converted.notna().mean()) if len(series) else 0.0


def _bool_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    values = series.dropna().astype(str).str.strip().str.lower()
    return float(values.isin(BOOLEAN_LIKE).mean()) if len(values) else 0.0


def _unique_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    return float(non_null.nunique(dropna=True) / len(non_null)) if len(non_null) else 0.0


def _match_name_role(name: str) -> tuple[str | None, float]:
    tokens = _name_tokens(name)
    compact = _clean_name(name)
    best_role = None
    best_score = 0.0
    for role, hints in NAME_HINTS.items():
        score = 0.0
        for hint in hints:
            if hint in tokens or hint == compact or hint in compact:
                score += 1.0
        if role == "delivery_capability" and "delivery" in tokens and "table" not in tokens:
            score += 0.5
        if role == "table_booking_capability" and {"table", "booking"}.issubset(tokens):
            score += 0.8
        if role == "restaurant_entity" and "name" in tokens and any(term in tokens for term in {"restaurant", "cafe", "diner", "branch"}):
            score += 0.9
        if score > best_score:
            best_role, best_score = role, score
    return best_role, min(best_score, 1.0)


def detect_column_role(column_name: str, series: pd.Series) -> dict[str, Any]:
    name_role, name_score = _match_name_role(column_name)
    non_null = series.dropna()
    text_values = _series_text(series)
    unique_ratio = _unique_ratio(series)
    numeric_ratio = _numeric_ratio(series)
    datetime_ratio = _datetime_ratio(series)
    bool_ratio = _bool_ratio(series)
    dtype_name = str(series.dtype)

    role = name_role or "unknown"
    confidence = name_score
    evidence: list[str] = []

    if datetime_ratio >= 0.8:
        role = "date"
        confidence = max(confidence, min(1.0, datetime_ratio))
        evidence.append("values parse as datetime")
    elif bool_ratio >= 0.8:
        if role in {"delivery_capability", "table_booking_capability"}:
            confidence = max(confidence, 0.9)
            evidence.append("binary capability-like values")
        else:
            role = "boolean_capability"
            confidence = max(confidence, 0.85)
            evidence.append("boolean-like values")
    elif numeric_ratio >= 0.9:
        if name_role == "rating_metric" or ("rating" in _clean_name(column_name) and non_null.max() <= 5 and non_null.min() >= 0):
            role = "rating_metric"
            confidence = max(confidence, 0.95)
            evidence.append("rating-shaped numeric range")
        elif name_role == "currency_metric" or any(token in _clean_name(column_name) for token in {"price", "cost", "sales", "revenue", "income", "profit", "amount"}):
            role = "currency_metric"
            confidence = max(confidence, 0.92)
            evidence.append("currency-style numeric column")
        elif name_role == "percentage" or any(token in _clean_name(column_name) for token in {"pct", "percent", "percentage", "ratio", "share"}):
            role = "percentage"
            confidence = max(confidence, 0.92)
            evidence.append("percentage-style numeric column")
        elif name_role == "count":
            role = "count"
            confidence = max(confidence, 0.9)
            evidence.append("count-like numeric column")
        else:
            role = name_role if name_role in {"numeric_metric", "count"} else "numeric_metric"
            confidence = max(confidence, 0.8 if role == "numeric_metric" else 0.9)
            evidence.append("numeric dtype")
    else:
        lower_values = {value.lower() for value in text_values}
        if lower_values and lower_values.issubset(BOOLEAN_LIKE):
            if role in {"delivery_capability", "table_booking_capability"}:
                confidence = max(confidence, 0.9)
                evidence.append("capability label values")
            else:
                role = "boolean_capability"
                confidence = max(confidence, 0.85)
                evidence.append("boolean-like text values")
        elif name_role in {"email", "phone", "url", "address", "status", "category", "description", "geographic_area", "restaurant_entity", "customer_entity", "product_entity", "supplier_entity", "employee_entity", "identifier"}:
            role = name_role
            confidence = max(confidence, 0.82)
            evidence.append("name hint matched")
        elif unique_ratio > 0.9 and any(token in _clean_name(column_name) for token in {"id", "code", "key", "uuid"}):
            role = "identifier"
            confidence = max(confidence, 0.9)
            evidence.append("high-cardinality identifier-like column")
        elif any(token in _clean_name(column_name) for token in {"name", "title"}) and unique_ratio > 0.5:
            role = "entity_name"
            confidence = max(confidence, 0.75)
            evidence.append("name/title text column")
        elif text_values and max((len(value.split()) for value in text_values), default=0) >= 3:
            role = "description"
            confidence = max(confidence, 0.7)
            evidence.append("long-form text")
        else:
            role = role if role != "unknown" else "category"
            confidence = max(confidence, 0.55 if role == "category" else 0.4)
            evidence.append("fallback categorical/text role")

    return {
        "role": role,
        "confidence": round(float(min(1.0, confidence)), 3),
        "dtype": dtype_name,
        "evidence": evidence,
        "unique_ratio": round(unique_ratio, 3),
    }


def _anonymized_id(index: int) -> str:
    return f"c{index + 1}"


def build_schema_profile(df: pd.DataFrame) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    role_index: dict[str, list[str]] = {}
    original_to_id: dict[str, str] = {}
    id_to_original: dict[str, str] = {}

    for idx, column in enumerate(df.columns):
        column_id = _anonymized_id(idx)
        series = df[column]
        role_info = detect_column_role(str(column), series)
        column_profile = {
            "column_id": column_id,
            "role": role_info["role"],
            "confidence": role_info["confidence"],
            "dtype": role_info["dtype"],
            "row_count": int(len(df)),
            "unique_ratio": role_info["unique_ratio"],
            "evidence": role_info["evidence"],
        }
        columns.append(column_profile)
        role_index.setdefault(role_info["role"], []).append(column_id)
        original_to_id[str(column)] = column_id
        id_to_original[column_id] = str(column)

    return {
        "columns": columns,
        "original_to_id": original_to_id,
        "id_to_original": id_to_original,
        "role_index": role_index,
    }


def anonymized_schema_summary(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "column_id": column["column_id"],
            "role": column["role"],
            "confidence": column["confidence"],
            "dtype": column["dtype"],
        }
        for column in profile["columns"]
    ]
