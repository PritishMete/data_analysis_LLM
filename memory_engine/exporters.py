# memory_engine/exporters.py
# ─────────────────────────────────────────────────────────────────────────────
# "ML Ready" step for this backend: turn the reusable experiences already
# sitting in query_history into a flat, structured table a FUTURE training
# job can consume directly — nothing in this file fits, trains, evaluates,
# or even imports a model/ML library. It reads already-stored rows (via
# QueryHistoryRepository) and reshapes them; that's the entire job.
#
# Exactly seven columns, per spec — no more, no less:
#   intent              QueryHistory.intent
#   question            QueryHistory.user_query
#   sql                 QueryHistory.generated_sql
#   pipeline            QueryHistory.python_pipeline (JSON-encoded — see below)
#   execution_time_ms   QueryHistory.execution_time_ms
#   feedback            QueryHistory.feedback_score
#   dataset_type        datasets.Dataset.source_type, resolved via
#                       QueryHistory.dataset_id (denormalized in at export
#                       time — query_history itself has no dataset_type
#                       column, and shouldn't grow one just for this: it
#                       would just be a second copy of the Dataset
#                       Registry's own source_type, going stale the moment
#                       a dataset is re-typed).
#
# `pipeline` is JSON-encoded to a plain string column (json.dumps), even in
# the Parquet output where a nested/struct column would otherwise be
# possible. Deliberate: query_router.py's plan dicts and cleaning_ops.py's
# steps lists don't share one fixed shape, and a column pyarrow has to
# infer a schema for needs one; a plain string column sidesteps that
# entirely and — just as importantly — keeps the CSV and Parquet outputs
# schema-identical, which is one less thing a future training pipeline
# needs to special-case per format.
#
# Why this lives in memory_engine/ rather than query_history/: query_history
# is deliberately just the raw append-only log (see its own module
# docstring) — reshaping that log into ML-ready tables is exactly the kind
# of "future ML integration" surface memory_engine exists to own (see
# contracts.py). datasets/ is only ever READ here (for source_type), never
# written to.
# ─────────────────────────────────────────────────────────────────────────────

import io
import json

import pandas as pd

from datasets.repository import DatasetRepository
from query_history.models import QueryHistory
from query_history.repository import QueryHistoryRepository

TRAINING_EXPORT_COLUMNS: list[str] = [
    "intent",
    "question",
    "sql",
    "pipeline",
    "execution_time_ms",
    "feedback",
    "dataset_type",
]

SUPPORTED_EXPORT_FORMATS: tuple[str, ...] = ("csv", "parquet")


def _serialize_pipeline(value: dict | list | None) -> str | None:
    """dict/list -> compact JSON string; None stays None (a real NULL in
    both CSV and Parquet, not the string "None" or "null")."""
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), default=str)


class TrainingDatasetExporter:
    """Prepares (never trains on) a flat table of past query executions for
    a future ML training pipeline. See module docstring for the exact
    column set and why each choice was made.
    """

    def __init__(self, history_repository: QueryHistoryRepository, dataset_repository: DatasetRepository):
        self.history_repository = history_repository
        self.dataset_repository = dataset_repository

    # ── Collection ──────────────────────────────────────────────────────

    def collect(
        self,
        *,
        organization_id: str | None = None,
        dataset_id: str | None = None,
        schema_hash: str | None = None,
        only_successful: bool = False,
        limit: int = 5000,
    ) -> list[QueryHistory]:
        """Which rows to include is delegated entirely to
        QueryHistoryRepository.list_candidates — the same general-purpose,
        multi-dimension query memory_engine's own find_*() methods already
        use, so this exporter adds no new query logic, just a different
        output SHAPE for the same underlying data.

        `only_successful` defaults to False (unlike
        QueryHistoryRepository.list_for_training's True default): a future
        model learning what NOT to produce needs failed executions too —
        see query_history/models.py's module docstring on why `success` is
        kept as a labeled column rather than filtered out at the source.
        Callers who only want positive examples can pass
        only_successful=True.
        """
        return self.history_repository.list_candidates(
            organization_id=organization_id,
            dataset_id=dataset_id,
            schema_hash=schema_hash,
            success=True if only_successful else None,
            limit=limit,
        )

    # ── Shaping ─────────────────────────────────────────────────────────

    def to_records(self, entries: list[QueryHistory]) -> list[dict]:
        """One dict per entry, exactly the seven columns in
        TRAINING_EXPORT_COLUMNS — nothing else is exported, even though
        QueryHistory has more fields available (e.g. success, planner_version,
        rows_returned). Batches dataset_type lookups through a small local
        cache keyed by dataset_id, so a 5000-row export against a handful of
        datasets does a handful of lookups, not 5000.
        """
        dataset_type_by_id: dict[str, str | None] = {}

        def _dataset_type(dataset_id: str | None) -> str | None:
            if dataset_id is None:
                return None
            if dataset_id not in dataset_type_by_id:
                dataset = self.dataset_repository.get_by_id(dataset_id)
                dataset_type_by_id[dataset_id] = dataset.source_type if dataset is not None else None
            return dataset_type_by_id[dataset_id]

        return [
            {
                "intent": entry.intent,
                "question": entry.user_query,
                "sql": entry.generated_sql,
                "pipeline": _serialize_pipeline(entry.python_pipeline),
                "execution_time_ms": entry.execution_time_ms,
                "feedback": entry.feedback_score,
                "dataset_type": _dataset_type(entry.dataset_id),
            }
            for entry in entries
        ]

    def to_dataframe(self, entries: list[QueryHistory]) -> pd.DataFrame:
        """Same seven columns as to_records(), as a DataFrame with nullable
        dtypes on the two numeric columns — so a missing execution_time_ms
        or feedback round-trips as an actual NULL in Parquet (and an empty
        cell in CSV), not a silently-wrong placeholder like 0 or NaN-as-str.
        """
        records = self.to_records(entries)
        df = pd.DataFrame(records, columns=TRAINING_EXPORT_COLUMNS)
        df["execution_time_ms"] = df["execution_time_ms"].astype("Float64")
        df["feedback"] = df["feedback"].astype("Int64")
        return df

    # ── Serialization ───────────────────────────────────────────────────
    # Both return raw bytes (never write to disk themselves) so a caller —
    # e.g. memory_engine/routes.py's /training-export endpoint — can stream
    # the result straight into an HTTP response or hand it to whatever
    # storage it likes, without this module needing an opinion on where
    # exports live.

    def export_csv(self, entries: list[QueryHistory]) -> bytes:
        df = self.to_dataframe(entries)
        return df.to_csv(index=False).encode("utf-8")

    def export_parquet(self, entries: list[QueryHistory]) -> bytes:
        df = self.to_dataframe(entries)
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        return buffer.getvalue()

    def export(self, entries: list[QueryHistory], *, fmt: str) -> bytes:
        """Single entry point taking a format string — what
        memory_engine/routes.py's endpoint calls, so adding a third format
        later is a one-line addition here rather than a new route."""
        fmt = fmt.lower()
        if fmt == "csv":
            return self.export_csv(entries)
        if fmt == "parquet":
            return self.export_parquet(entries)
        raise ValueError(f"unsupported export format: {fmt!r} (expected one of {SUPPORTED_EXPORT_FORMATS})")
