# common/transformations/adapters/drop_columns.py
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register

_DROP_RE = re.compile(r"drop\s+(?:column[s]?\s+)?(.+)$", re.IGNORECASE)
_REMOVE_COL_RE = re.compile(r"remove\s+(?:column[s]?\s+)?(.+)$", re.IGNORECASE)


def _split_column_list(fragment: str, columns: list[str]) -> list[str]:
    """Matches real column names (longest first, case-insensitive) inside a
    free-text fragment like "rating and old_id" or "a, b, c"."""
    found = []
    remaining = fragment
    for col in sorted(columns, key=lambda c: -len(str(c))):
        pattern = re.compile(rf"\b{re.escape(str(col))}\b", re.IGNORECASE)
        if pattern.search(remaining):
            found.append(col)
            remaining = pattern.sub("", remaining)
    return found


class DropColumnsTransformation(BaseTransformation):
    name = "drop_columns"
    display_name = "Drop Columns"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        text = text or ""
        match = _DROP_RE.search(text) or _REMOVE_COL_RE.search(text)
        if not match:
            return {"detected": False, "params": {}, "confidence": 0.0}
        columns = _split_column_list(match.group(1), list(df.columns))
        return {
            "detected": True,
            "confidence": 0.85 if columns else 0.3,
            "params": {"columns": columns},
        }

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        columns = params.get("columns")
        if not columns:
            raise TransformationError("`columns` (a non-empty list of column names) is required.")
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise TransformationError(f"Column(s) not found: {', '.join(missing)}.")
        if set(columns) == set(df.columns):
            raise TransformationError("Cannot drop every column in the dataset.")

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        columns = params["columns"]
        sample = df[columns].head(sample_rows)
        return {
            "affected_columns": columns,
            "affected_rows": len(df),
            "before": sample.to_dict(orient="records"),
            "after": [],  # columns are gone entirely — nothing to show "after"
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        columns = params["columns"]
        new_df = df.drop(columns=columns)
        metadata = {
            "type": "column_transformation",
            "transformation": "drop_columns",
            "dropped_columns": columns,
        }
        return {"dataframe": new_df, "metadata": metadata}


register(DropColumnsTransformation())
