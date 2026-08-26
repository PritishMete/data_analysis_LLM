# common/transformations/adapters/date_features.py
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register

_VALID_FEATURES = ("year", "month", "day", "weekday", "quarter", "week")
_DATE_FEATURE_RE = re.compile(r"extract\s+(.+?)\s+from\s+(\w+)", re.IGNORECASE)


class DateFeatureTransformation(BaseTransformation):
    name = "date_features"
    display_name = "Date Feature Extraction"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        low = (text or "").lower()
        if "extract" not in low or not any(f in low for f in _VALID_FEATURES):
            return {"detected": False, "params": {}, "confidence": 0.0}
        match = _DATE_FEATURE_RE.search(text)
        if not match:
            return {"detected": True, "params": {}, "confidence": 0.3}
        features_raw, raw_column = match.group(1), match.group(2).strip()
        features = [f for f in _VALID_FEATURES if f in features_raw.lower()]
        column = next((c for c in df.columns if str(c).lower() == raw_column.lower()), None)
        return {
            "detected": True,
            "confidence": 0.85 if column and features else 0.4,
            "params": {"column": column, "features": features or ["year", "month", "day"]},
        }

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        column = params.get("column")
        if not column:
            raise TransformationError("`column` is required for date_features.")
        if column not in df.columns:
            raise TransformationError(f"Column '{column}' does not exist in the dataset.")
        features = params.get("features") or []
        bad = [f for f in features if f not in _VALID_FEATURES]
        if bad:
            raise TransformationError(f"Unsupported date feature(s): {', '.join(bad)}. Use: {', '.join(_VALID_FEATURES)}.")
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.isna().all():
            raise TransformationError(f"Column '{column}' does not contain any parseable dates.")

    def _extract(self, series: pd.Series, features: list[str]) -> dict[str, pd.Series]:
        parsed = pd.to_datetime(series, errors="coerce")
        out = {}
        for feature in features:
            if feature == "year":
                out[f"{series.name}_year"] = parsed.dt.year
            elif feature == "month":
                out[f"{series.name}_month"] = parsed.dt.month
            elif feature == "day":
                out[f"{series.name}_day"] = parsed.dt.day
            elif feature == "weekday":
                out[f"{series.name}_weekday"] = parsed.dt.day_name()
            elif feature == "quarter":
                out[f"{series.name}_quarter"] = parsed.dt.quarter
            elif feature == "week":
                out[f"{series.name}_week"] = parsed.dt.isocalendar().week.astype("Int64")
        return out

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        column, features = params["column"], params["features"]
        sample = df[[column]].head(sample_rows)
        new_cols = self._extract(sample[column], features)
        after = sample.copy()
        for name, values in new_cols.items():
            after[name] = values
        return {
            "affected_columns": [column] + list(new_cols.keys()),
            "affected_rows": len(df),
            "before": sample.to_dict(orient="records"),
            "after": after.to_dict(orient="records"),
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        column, features = params["column"], params["features"]
        new_cols = self._extract(df[column], features)
        new_df = df.copy()
        for name, values in new_cols.items():
            new_df[name] = values
        metadata = {
            "type": "column_transformation",
            "transformation": "date_features",
            "source_column": column,
            "features": features,
            "new_columns": list(new_cols.keys()),
        }
        return {"dataframe": new_df, "metadata": metadata}

    def undo(self, before_df: pd.DataFrame, after_df: pd.DataFrame, apply_result: dict[str, Any]) -> pd.DataFrame:
        new_columns = apply_result.get("metadata", {}).get("new_columns") or []
        if new_columns and set(new_columns).issubset(after_df.columns) and len(before_df) == len(after_df):
            return after_df.drop(columns=new_columns)
        return super().undo(before_df, after_df, apply_result)


register(DateFeatureTransformation())
