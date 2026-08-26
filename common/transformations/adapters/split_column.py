# common/transformations/adapters/split_column.py
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register

_SPLIT_RE = re.compile(r"split\s+(?:column\s+)?(\w+)\s*(?:by|on)?\s*['\"]?([^'\"]*?)['\"]?\s*$", re.IGNORECASE)


class SplitColumnTransformation(BaseTransformation):
    name = "split_column"
    display_name = "Split Column"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        if "split" not in (text or "").lower():
            return {"detected": False, "params": {}, "confidence": 0.0}
        match = _SPLIT_RE.search(text.strip())
        if not match:
            return {"detected": True, "params": {}, "confidence": 0.3}
        raw_column = match.group(1).strip()
        delimiter = match.group(2) if match.group(2) else ","
        column = next((c for c in df.columns if str(c).lower() == raw_column.lower()), None)
        return {
            "detected": True,
            "confidence": 0.8 if column else 0.4,
            "params": {"column": column, "delimiter": delimiter},
        }

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        column = params.get("column")
        if not column:
            raise TransformationError("`column` is required for split_column.")
        if column not in df.columns:
            raise TransformationError(f"Column '{column}' does not exist in the dataset.")
        if not params.get("delimiter"):
            raise TransformationError("`delimiter` is required for split_column.")

    def _split(self, df: pd.DataFrame, column: str, delimiter: str) -> tuple[pd.DataFrame, list[str]]:
        split_df = df[column].astype(str).str.split(delimiter, expand=True)
        new_names = [f"{column}_part{i + 1}" for i in range(split_df.shape[1])]
        split_df.columns = new_names
        return split_df, new_names

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        column, delimiter = params["column"], params["delimiter"]
        sample = df[[column]].head(sample_rows)
        split_sample, new_names = self._split(sample, column, delimiter)
        after = pd.concat([sample, split_sample], axis=1)
        return {
            "affected_columns": [column] + new_names,
            "affected_rows": len(df),
            "before": sample.to_dict(orient="records"),
            "after": after.to_dict(orient="records"),
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        column, delimiter = params["column"], params["delimiter"]
        split_result, new_names = self._split(df, column, delimiter)
        new_df = pd.concat([df.copy(), split_result], axis=1)
        metadata = {
            "type": "column_transformation",
            "transformation": "split_column",
            "source_column": column,
            "delimiter": delimiter,
            "new_columns": new_names,
        }
        return {"dataframe": new_df, "metadata": metadata}

    def undo(self, before_df: pd.DataFrame, after_df: pd.DataFrame, apply_result: dict[str, Any]) -> pd.DataFrame:
        new_columns = apply_result.get("metadata", {}).get("new_columns") or []
        if new_columns and set(new_columns).issubset(after_df.columns) and len(before_df) == len(after_df):
            return after_df.drop(columns=new_columns)
        return super().undo(before_df, after_df, apply_result)


register(SplitColumnTransformation())
