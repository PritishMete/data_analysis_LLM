# common/transformations/adapters/fill_missing.py
# ─────────────────────────────────────────────────────────────────────────────
# Adapter around the ALREADY-EXISTING data_cleaning_utils.fill_nulls(). No
# filling logic is reimplemented here.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register
from data_cleaning_utils import fill_nulls

_METHOD_WORDS = ("mean", "median", "mode")
_FILL_RE = re.compile(r"fill\s+(?:missing|null[s]?|na)\s*(?:values?)?\s*(?:in\s+)?(.*)", re.IGNORECASE)


class FillMissingTransformation(BaseTransformation):
    name = "fill_missing"
    display_name = "Fill Missing Values"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        text = text or ""
        low = text.lower()
        if "fill" not in low or ("missing" not in low and "null" not in low and " na " not in f" {low} "):
            return {"detected": False, "params": {}, "confidence": 0.0}

        method = next((m for m in _METHOD_WORDS if m in low), "mean")
        match = _FILL_RE.search(text)
        column = None
        if match:
            fragment = match.group(1)
            for m in _METHOD_WORDS:
                fragment = re.sub(rf"\b(?:with|using)\s+{m}\b", "", fragment, flags=re.IGNORECASE)
            fragment = fragment.strip(" .")
            column = next((c for c in df.columns if str(c).lower() == fragment.lower()), None)

        return {
            "detected": True,
            "confidence": 0.8 if column else 0.5,
            "params": {"columns": [column] if column else None, "method": method},
        }

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        method = (params.get("method") or "mean").lower()
        if method not in {"mean", "median", "mode", "custom"}:
            raise TransformationError(f"Unknown fill method '{method}'. Use mean, median, mode, or custom.")
        columns = params.get("columns")
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise TransformationError(f"Column(s) not found: {', '.join(missing)}.")
        if method == "custom" and params.get("custom_value") is None:
            raise TransformationError("`custom_value` is required when method='custom'.")

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        columns = params.get("columns") or df.columns[df.isnull().any()].tolist()
        before = df[columns].head(sample_rows) if columns else df.head(0)
        after_full = fill_nulls(df, columns=columns, method=params.get("method", "mean"),
                                 custom_value=params.get("custom_value"))
        after = after_full[columns].head(sample_rows) if columns else after_full.head(0)
        return {
            "affected_columns": columns,
            "affected_rows": int(df[columns].isnull().any(axis=1).sum()) if columns else 0,
            "before": before.to_dict(orient="records"),
            "after": after.to_dict(orient="records"),
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        columns = params.get("columns") or df.columns[df.isnull().any()].tolist()
        cells_before = int(df[columns].isnull().sum().sum()) if columns else 0
        new_df = fill_nulls(df, columns=columns, method=params.get("method", "mean"),
                             custom_value=params.get("custom_value"))
        metadata = {
            "type": "column_transformation",
            "transformation": "fill_missing",
            "columns": columns,
            "method": params.get("method", "mean"),
            "cells_filled": cells_before,
        }
        return {"dataframe": new_df, "metadata": metadata}


register(FillMissingTransformation())
