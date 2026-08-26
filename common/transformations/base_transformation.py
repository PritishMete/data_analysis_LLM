# common/transformations/base_transformation.py
# ─────────────────────────────────────────────────────────────────────────────
# The contract every transformation (existing or new) must implement to be
# usable by the centralized TransformationEngine (transformation_engine.py).
#
# This does NOT reimplement any transformation logic. It's the thin interface
# concrete transformations (see common/transformations/adapters/*.py) wrap
# around already-existing, already-tested logic — e.g. RangeBinningTransformation
# wraps common/transformations/range_binning.py's detect_range_binning() /
# apply_range_binning() rather than reimplementing binning.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class TransformationError(ValueError):
    """Raised by validate()/apply() for any user-facing failure (bad column,
    bad params, etc). The engine catches this and turns it into a failed
    TransformationResult rather than a 500."""


class BaseTransformation(ABC):
    """One entry in the transformation registry.

    Required methods:
      detect(text, df)      -> intent + parameter extraction from free text
      validate(df, params)  -> raises TransformationError on any problem
      preview(df, params)   -> before/after preview WITHOUT mutating df
      apply(df, params)     -> the actual transformation, returns a NEW df
      metadata(apply_result)-> transformation-specific metadata dict
      undo(before_df, after_df, apply_result) -> df to restore

    `name` is the registry key (e.g. "range_binning"); `display_name` is
    what shows up in the AI report / Flutter Transformation Center (e.g.
    "Range Binning").
    """

    name: str = "unnamed_transformation"
    display_name: str = "Unnamed Transformation"

    @abstractmethod
    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        """Rule-based (no LLM) intent + parameter extraction.

        Must return at least: {"detected": bool, "params": dict, "confidence": float}
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        """Raises TransformationError with a human-readable message on any
        validation failure (missing column, wrong dtype, bad params, ...).
        Returns None on success.
        """
        raise NotImplementedError

    @abstractmethod
    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        """Returns a preview WITHOUT mutating df or committing anything:
        {"affected_columns": [...], "affected_rows": int,
         "before": [...], "after": [...]}
        `before`/`after` only include the columns actually affected.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        """Performs the transformation. MUST NOT mutate `df` in place.
        Returns {"dataframe": new_df, "metadata": {...}}.
        """
        raise NotImplementedError

    def metadata(self, apply_result: dict[str, Any]) -> dict[str, Any]:
        """Default: pass through whatever apply() already put in "metadata".
        Override only if a transformation needs to post-process it."""
        return apply_result.get("metadata", {})

    def undo(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        apply_result: dict[str, Any],
    ) -> pd.DataFrame:
        """Default undo: restore the pre-transformation snapshot. This is
        always correct (every transformation supports undo this way via
        TransformationHistory's snapshots), so most transformations don't
        need to override this. Transformations where a cheaper, storage-free
        inverse exists (e.g. dropping the column range_binning just added)
        may override for efficiency — see adapters/range_binning_transformation.py.
        """
        return before_df.copy()
