# common/transformations/transformation_history.py
# ─────────────────────────────────────────────────────────────────────────────
# In-memory transformation ledger for one working session (one loaded
# dataset). This is what powers Undo / Redo / Replay and the Flutter
# "Transformation Center" timeline.
#
# Design choice: undo/redo restore from an in-memory DATAFRAME SNAPSHOT taken
# before each transformation, rather than relying on every transformation
# implementing a perfect mathematical inverse. This is the only approach
# that's correct for ALL transformations uniformly (you can't losslessly
# invert "fill missing values" or "drop columns" without the original data),
# and it's what BaseTransformation.undo() defaults to. `replay()` is the
# complementary operation: it re-runs each recorded step's apply() logic
# from a given point forward against a (possibly different) starting
# dataframe, rather than restoring a snapshot — useful for "apply this same
# sequence of cleanups to a re-uploaded/updated dataset".
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class TransformationHistoryEntry:
    id: str
    timestamp: str
    query: str | None
    transformation_name: str
    source_columns: list[str]
    target_columns: list[str]
    rows_modified: int
    execution_time: float
    metadata: dict[str, Any]
    params: dict[str, Any]
    # Snapshots kept for undo/redo. Not exposed in to_dict() (too big for a
    # Flutter payload) — callers that need the actual data use `export`.
    _before_df: pd.DataFrame = field(repr=False, compare=False)
    _after_df: pd.DataFrame = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "query": self.query,
            "transformation_name": self.transformation_name,
            "source_columns": self.source_columns,
            "target_columns": self.target_columns,
            "rows_modified": self.rows_modified,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


class TransformationHistory:
    """One instance per working session/dataset. NOT a global singleton —
    the caller (e.g. a per-session object in main.py, or per-request state
    passed in by the Flutter client) owns the instance so different users'
    histories never mix.
    """

    def __init__(self) -> None:
        self._entries: list[TransformationHistoryEntry] = []
        self._redo_stack: list[TransformationHistoryEntry] = []

    # ── Recording ────────────────────────────────────────────────────────
    def record(
        self,
        *,
        transformation_name: str,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        params: dict[str, Any],
        metadata: dict[str, Any],
        query: str | None = None,
        execution_time: float = 0.0,
    ) -> TransformationHistoryEntry:
        source_columns = [c for c in before_df.columns if c in after_df.columns and (
            not before_df[c].equals(after_df[c]) if len(before_df) == len(after_df) else True
        )]
        target_columns = [c for c in after_df.columns if c not in before_df.columns]

        if not source_columns and metadata.get("source_column"):
            source_columns = [metadata["source_column"]]
        if not target_columns and metadata.get("new_column"):
            target_columns = [metadata["new_column"]]

        rows_modified = self._count_modified_rows(before_df, after_df)

        entry = TransformationHistoryEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            transformation_name=transformation_name,
            source_columns=source_columns,
            target_columns=target_columns,
            rows_modified=rows_modified,
            execution_time=execution_time,
            metadata=metadata,
            params=params,
            _before_df=before_df,
            _after_df=after_df,
        )
        self._entries.append(entry)
        self._redo_stack.clear()  # a new action invalidates any pending redo
        return entry

    @staticmethod
    def _count_modified_rows(before_df: pd.DataFrame, after_df: pd.DataFrame) -> int:
        if len(before_df) != len(after_df):
            return abs(len(after_df) - len(before_df))
        shared_cols = [c for c in before_df.columns if c in after_df.columns]
        if not shared_cols:
            return len(after_df)
        try:
            changed_mask = (before_df[shared_cols].values != after_df[shared_cols].values)
            return int(changed_mask.any(axis=1).sum())
        except Exception:
            return len(after_df)

    # ── Undo / Redo ──────────────────────────────────────────────────────
    def can_undo(self) -> bool:
        return len(self._entries) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> tuple[pd.DataFrame, TransformationHistoryEntry] | None:
        """Pops the last transformation and returns (restored_df, entry)."""
        if not self._entries:
            return None
        entry = self._entries.pop()
        self._redo_stack.append(entry)
        return entry._before_df.copy(), entry

    def redo(self) -> tuple[pd.DataFrame, TransformationHistoryEntry] | None:
        """Re-applies the most recently undone transformation."""
        if not self._redo_stack:
            return None
        entry = self._redo_stack.pop()
        self._entries.append(entry)
        return entry._after_df.copy(), entry

    # ── Replay ───────────────────────────────────────────────────────────
    def replay(self, engine: "TransformationEngine", start_df: pd.DataFrame, up_to_index: int | None = None) -> pd.DataFrame:
        """Re-runs every recorded transformation's `apply()` (fresh
        computation, not snapshot restore) against `start_df` in order,
        up to and including `up_to_index` (default: the whole history).
        Used to replay a cleanup sequence onto a different/updated dataset.
        """
        entries = self._entries if up_to_index is None else self._entries[: up_to_index + 1]
        df = start_df
        for entry in entries:
            transformation = engine.registry_get(entry.transformation_name)
            if transformation is None:
                raise ValueError(f"Cannot replay: transformation '{entry.transformation_name}' is not registered.")
            apply_result = transformation.apply(df, entry.params)
            df = apply_result["dataframe"]
        return df

    # ── Inspection ───────────────────────────────────────────────────────
    def list(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)
