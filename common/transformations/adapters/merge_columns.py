# common/transformations/adapters/merge_columns.py
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register

_MERGE_RE = re.compile(r"merge\s+(.+?)\s+(?:and|with)\s+(.+?)(?:\s+into\s+(\w+))?$", re.IGNORECASE)


class MergeColumnsTransformation(BaseTransformation):
    name = "merge_columns"
    display_name = "Merge Columns"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        text = text or ""
        if "merge" not in text.lower() and "combine" not in text.lower():
            return {"detected": False, "params": {}, "confidence": 0.0}
        match = _MERGE_RE.search(text)
        if not match:
            return {"detected": True, "params": {}, "confidence": 0.3}
        a_raw, b_raw, new_col = match.group(1).strip(), match.group(2).strip(), match.group(3)
        a = next((c for c in df.columns if str(c).lower() == a_raw.lower()), None)
        b = next((c for c in df.columns if str(c).lower() == b_raw.lower()), None)
        columns = [c for c in (a, b) if c]
        return {
            "detected": True,
            "confidence": 0.85 if len(columns) == 2 else 0.4,
            "params": {"columns": columns, "new_column": new_col, "separator": " "},
        }

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        columns = params.get("columns")
        if not columns or len(columns) < 2:
            raise TransformationError("`columns` must list at least two columns to merge.")
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise TransformationError(f"Column(s) not found: {', '.join(missing)}.")
        new_column = params.get("new_column")
        if new_column and new_column in df.columns:
            raise TransformationError(f"Column '{new_column}' already exists.")

    def _merged_series(self, df: pd.DataFrame, columns: list[str], separator: str) -> pd.Series:
        return df[columns].astype(str).agg(separator.join, axis=1)

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        columns = params["columns"]
        new_column = params.get("new_column") or "_".join(str(c) for c in columns)
        separator = params.get("separator", " ")
        sample = df[columns].head(sample_rows)
        after = sample.copy()
        after[new_column] = self._merged_series(sample, columns, separator)
        return {
            "affected_columns": columns + [new_column],
            "affected_rows": len(df),
            "before": sample.to_dict(orient="records"),
            "after": after.to_dict(orient="records"),
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        columns = params["columns"]
        separator = params.get("separator", " ")
        new_column = params.get("new_column") or "_".join(str(c) for c in columns)
        new_df = df.copy()
        new_df[new_column] = self._merged_series(new_df, columns, separator)
        metadata = {
            "type": "column_transformation",
            "transformation": "merge_columns",
            "source_columns": columns,
            "new_column": new_column,
            "separator": separator,
        }
        return {"dataframe": new_df, "metadata": metadata}

    def undo(self, before_df: pd.DataFrame, after_df: pd.DataFrame, apply_result: dict[str, Any]) -> pd.DataFrame:
        new_column = apply_result.get("metadata", {}).get("new_column")
        if new_column and new_column in after_df.columns and len(before_df) == len(after_df):
            return after_df.drop(columns=[new_column])
        return super().undo(before_df, after_df, apply_result)


register(MergeColumnsTransformation())
