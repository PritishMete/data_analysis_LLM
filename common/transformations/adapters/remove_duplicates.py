# common/transformations/adapters/remove_duplicates.py
from __future__ import annotations

from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register

_TRIGGER_WORDS = ("duplicate", "dedupe", "de-dupe", "distinct rows")


class RemoveDuplicatesTransformation(BaseTransformation):
    name = "remove_duplicates"
    display_name = "Duplicate Removal"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        low = (text or "").lower()
        if not any(word in low for word in _TRIGGER_WORDS):
            return {"detected": False, "params": {}, "confidence": 0.0}
        return {"detected": True, "confidence": 0.85, "params": {"keep": "first"}}

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        keep = params.get("keep", "first")
        if keep not in {"first", "last", False}:
            raise TransformationError("`keep` must be 'first', 'last', or False.")
        subset = params.get("subset")
        if subset:
            missing = [c for c in subset if c not in df.columns]
            if missing:
                raise TransformationError(f"Column(s) not found: {', '.join(missing)}.")

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        subset = params.get("subset")
        keep = params.get("keep", "first")
        duplicate_mask = df.duplicated(subset=subset, keep=keep)
        duplicated_rows = df[duplicate_mask].head(sample_rows)
        return {
            "affected_columns": subset or list(df.columns),
            "affected_rows": int(duplicate_mask.sum()),
            "before": duplicated_rows.to_dict(orient="records"),
            "after": [],  # these exact rows are removed entirely
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        subset = params.get("subset")
        keep = params.get("keep", "first")
        rows_before = len(df)
        new_df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        metadata = {
            "type": "row_transformation",
            "transformation": "remove_duplicates",
            "subset": subset,
            "rows_removed": rows_before - len(new_df),
        }
        return {"dataframe": new_df, "metadata": metadata}


register(RemoveDuplicatesTransformation())
