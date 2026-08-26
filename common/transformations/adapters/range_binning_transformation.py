# common/transformations/adapters/range_binning_transformation.py
# ─────────────────────────────────────────────────────────────────────────────
# Registers the ALREADY-EXISTING range_binning.py logic with the centralized
# TransformationEngine. Zero binning logic lives here — this is purely an
# adapter around common/transformations/range_binning.py's detect_range_binning()
# / apply_range_binning(), which are untouched.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_registry import register
from common.transformations.range_binning import (
    RangeBinningError,
    apply_range_binning,
    detect_range_binning,
)


class RangeBinningTransformation(BaseTransformation):
    name = "range_binning"
    display_name = "Range Binning"

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        detection = detect_range_binning(text, list(df.columns), df)
        # BUG FIX: this used to be
        #   bool(detection.get("detected") and detection.get("source_column"))
        # which silently reported "not detected at all" whenever the column
        # guess failed (e.g. a plural/prefixed/differently-worded column
        # name like "Ratings", "Overall_Rating", "Score" instead of an exact
        # substring match on "Rating"). That made the engine's _locate()
        # skip range binning entirely and fall through to the generic LLM
        # router, which has no concept of range binning and hallucinates a
        # "not supported" response — even though the intent WAS correctly
        # recognized (detect_range_binning already returns detected=True at
        # 0.7 confidence in exactly this case).
        #
        # Fix: report the real intent-detection result here. `validate()`
        # below already raises a specific, actionable TransformationError
        # when source_column can't be resolved — and the engine surfaces
        # that error message directly to the user (see query_router.py's
        # `elif transform_result.error != "Could not locate..."` branch)
        # instead of falling through to the LLM. So an unresolved column now
        # produces "I understood you want range binning, but couldn't tell
        # which column — try one of: X, Y, Z" instead of "not supported".
        return {
            "detected": bool(detection.get("detected")),
            "confidence": detection.get("confidence", 0.0),
            "params": {
                "source_column": detection.get("source_column"),
                "ranges": detection.get("ranges"),
                "new_column": detection.get("new_column"),
            },
        }

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        source_column = params.get("source_column")
        if not source_column:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            suggestion = f" Numeric columns available: {', '.join(numeric_cols)}." if numeric_cols else ""
            raise TransformationError(
                "I understood you want to bucket a column into ranges, but couldn't tell which "
                f"column — try including its exact name in your request.{suggestion}"
            )
        if source_column not in df.columns:
            raise TransformationError(f"Column '{source_column}' does not exist in the dataset.")
        if not pd.api.types.is_numeric_dtype(df[source_column]):
            raise TransformationError(
                f"Column '{source_column}' is not numeric — range binning requires a numeric column."
            )
        # Full parse/order/overlap validation is delegated to apply_range_binning
        # itself (single source of truth for that logic) via a dry-run below.
        try:
            apply_range_binning(df, source_column, params.get("ranges"), params.get("new_column"))
        except RangeBinningError as e:
            raise TransformationError(str(e)) from e

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        try:
            result = apply_range_binning(df, params["source_column"], params.get("ranges"), params.get("new_column"))
        except RangeBinningError as e:
            raise TransformationError(str(e)) from e
        preview = result["preview"]
        new_column = result["metadata"]["new_column"]
        return {
            "affected_columns": [params["source_column"], new_column],
            "affected_rows": len(df),
            "before": preview["before"],
            "after": preview["after"],
        }

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        try:
            result = apply_range_binning(df, params["source_column"], params.get("ranges"), params.get("new_column"))
        except RangeBinningError as e:
            raise TransformationError(str(e)) from e
        metadata = dict(result["metadata"])
        metadata["category_count"] = len(metadata["ranges"])
        metadata["explanation"] = result["explanation"]
        return {"dataframe": result["dataframe"], "metadata": metadata}

    def undo(self, before_df: pd.DataFrame, after_df: pd.DataFrame, apply_result: dict[str, Any]) -> pd.DataFrame:
        # Cheaper than a full snapshot restore: range binning only ADDS one
        # column, so undo is just dropping it back off — but only when the
        # rest of the dataframe is otherwise identical (guards against this
        # shortcut being wrong if something upstream changed row count etc).
        new_column = apply_result.get("metadata", {}).get("new_column")
        if new_column and new_column in after_df.columns and len(before_df) == len(after_df):
            return after_df.drop(columns=[new_column])
        return super().undo(before_df, after_df, apply_result)


register(RangeBinningTransformation())
