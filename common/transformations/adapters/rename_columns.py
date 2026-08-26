# common/transformations/adapters/rename_columns.py
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register

# "rename X to Y" / "rename column X as Y" / "rename 'old name' to 'new name'"
_RENAME_RE = re.compile(
    r"rename\s+(?:column\s+)?['\"]?([\w \-]+?)['\"]?\s+(?:to|as)\s+['\"]?([\w \-]+?)['\"]?\s*$",
    re.IGNORECASE,
)


class RenameColumnsTransformation(BaseTransformation):
    name = "rename_columns"
    display_name = "Rename Columns"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        if "rename" not in (text or "").lower():
            return {"detected": False, "params": {}, "confidence": 0.0}
        match = _RENAME_RE.search(text.strip())
        if not match:
            return {"detected": True, "params": {}, "confidence": 0.3}
        old, new = match.group(1).strip(), match.group(2).strip()
        # Match case-insensitively against real columns so "rename rating to score" works.
        resolved_old = next((c for c in df.columns if str(c).lower() == old.lower()), old)
        return {
            "detected": True,
            "confidence": 0.9 if resolved_old in df.columns else 0.5,
            "params": {"mapping": {resolved_old: new}},
        }

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        mapping = params.get("mapping")
        if not mapping or not isinstance(mapping, dict):
            raise TransformationError("`mapping` (a dict of {old_name: new_name}) is required.")
        missing = [old for old in mapping if old not in df.columns]
        if missing:
            raise TransformationError(f"Column(s) not found: {', '.join(missing)}.")
        collisions = [new for new in mapping.values() if new in df.columns and new not in mapping]
        if collisions:
            raise TransformationError(
                f"Cannot rename to {', '.join(collisions)} — a column with that name already exists."
            )

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        mapping = params["mapping"]
        old_cols = list(mapping.keys())
        sample = df[old_cols].head(sample_rows)
        after = sample.rename(columns=mapping)
        return {
            "affected_columns": old_cols + list(mapping.values()),
            "affected_rows": len(df),
            "before": sample.to_dict(orient="records"),
            "after": after.to_dict(orient="records"),
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        mapping = params["mapping"]
        new_df = df.rename(columns=mapping)
        metadata = {
            "type": "column_transformation",
            "transformation": "rename_columns",
            "renamed": mapping,
            "source_column": next(iter(mapping.keys()), None),
            "new_column": next(iter(mapping.values()), None),
        }
        return {"dataframe": new_df, "metadata": metadata}

    def undo(self, before_df: pd.DataFrame, after_df: pd.DataFrame, apply_result: dict[str, Any]) -> pd.DataFrame:
        mapping = apply_result.get("metadata", {}).get("renamed") or {}
        inverse = {new: old for old, new in mapping.items()}
        if inverse and set(inverse).issubset(after_df.columns):
            return after_df.rename(columns=inverse)
        return super().undo(before_df, after_df, apply_result)


register(RenameColumnsTransformation())
