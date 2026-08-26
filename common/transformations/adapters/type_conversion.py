# common/transformations/adapters/type_conversion.py
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register

_VALID_TYPES = {"int", "float", "string", "str", "datetime", "date", "bool", "category"}
_CONVERT_RE = re.compile(
    r"convert\s+(.+?)\s+to\s+(int|integer|float|number|string|str|text|datetime|date|bool|boolean|category|categorical)",
    re.IGNORECASE,
)


def _normalize_type(target_type: str) -> str:
    t = target_type.lower()
    if t in {"integer"}:
        return "int"
    if t in {"number"}:
        return "float"
    if t in {"text", "str"}:
        return "string"
    if t in {"boolean"}:
        return "bool"
    if t in {"categorical"}:
        return "category"
    if t in {"date"}:
        return "datetime"
    return t


class TypeConversionTransformation(BaseTransformation):
    name = "type_conversion"
    display_name = "Type Conversion"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        match = _CONVERT_RE.search(text or "")
        if not match:
            return {"detected": False, "params": {}, "confidence": 0.0}
        raw_column, raw_type = match.group(1).strip(), match.group(2).strip()
        column = next((c for c in df.columns if str(c).lower() == raw_column.lower()), None)
        return {
            "detected": True,
            "confidence": 0.85 if column else 0.4,
            "params": {"column": column, "target_type": _normalize_type(raw_type)},
        }

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        column, target_type = params.get("column"), params.get("target_type")
        if not column:
            raise TransformationError("`column` is required for type conversion.")
        if column not in df.columns:
            raise TransformationError(f"Column '{column}' does not exist in the dataset.")
        if target_type not in _VALID_TYPES:
            raise TransformationError(
                f"Unsupported target type '{target_type}'. Use one of: {', '.join(sorted(_VALID_TYPES))}."
            )

    def _convert(self, series: pd.Series, target_type: str) -> pd.Series:
        if target_type == "int":
            return pd.to_numeric(series, errors="coerce").round().astype("Int64")
        if target_type == "float":
            return pd.to_numeric(series, errors="coerce")
        if target_type in {"string", "str"}:
            return series.astype(str)
        if target_type in {"datetime", "date"}:
            return pd.to_datetime(series, errors="coerce")
        if target_type == "bool":
            return series.astype(bool)
        if target_type == "category":
            return series.astype("category")
        raise TransformationError(f"Unsupported target type '{target_type}'.")

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        column, target_type = params["column"], params["target_type"]
        sample = df[[column]].head(sample_rows)
        converted = sample.copy()
        converted[column] = self._convert(sample[column], target_type)
        return {
            "affected_columns": [column],
            "affected_rows": len(df),
            "before": sample.to_dict(orient="records"),
            "after": converted.to_dict(orient="records"),
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        column, target_type = params["column"], params["target_type"]
        new_df = df.copy()
        original_dtype = str(new_df[column].dtype)
        converted = self._convert(new_df[column], target_type)
        failed = int(converted.isna().sum() - new_df[column].isna().sum()) if target_type in {"int", "float", "datetime", "date"} else 0
        new_df[column] = converted
        metadata = {
            "type": "column_transformation",
            "transformation": "type_conversion",
            "source_column": column,
            "before_dtype": original_dtype,
            "after_dtype": str(new_df[column].dtype),
            "values_failed_to_convert": max(failed, 0),
        }
        return {"dataframe": new_df, "metadata": metadata}


register(TypeConversionTransformation())
