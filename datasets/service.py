# datasets/service.py
# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry business logic. Deterministic only — pandas for column
# stats, hashlib for identity, SQLAlchemy (via the repository) for
# persistence. No AI/LLM calls happen anywhere in this file, per the
# project requirement that registry/schema-detection work stays
# non-probabilistic and cheap to run on every upload.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass

import pandas as pd

from .hashing import compute_file_hash, compute_schema_hash
from .models import Dataset, DatasetColumn
from .repository import DatasetRepository


@dataclass
class DatasetRegistration:
    dataset: Dataset
    columns: list[DatasetColumn]
    was_duplicate: bool


class DatasetRegistryService:
    """Depends on a DatasetRepository via constructor injection (see
    routes.py's `get_dataset_repository` -> `get_dataset_registry_service`
    dependency chain) — this class never opens its own DB session, so it can
    be unit-tested with a repository backed by an in-memory SQLite engine or,
    for pure-logic tests, a hand-rolled fake repository.
    """

    def __init__(self, repository: DatasetRepository):
        self.repository = repository

    def register_dataset(
        self,
        *,
        df: pd.DataFrame,
        raw_bytes: bytes,
        organization_id: str,
        dataset_name: str,
        uploaded_by: str | None,
        source_type: str,
    ) -> DatasetRegistration:
        """Idempotent: re-uploading byte-identical content for the same
        organization returns the EXISTING dataset (with last_accessed
        refreshed) instead of creating a duplicate row. This is what lets the
        registry eventually recognize "this is the same file as before"
        rather than accumulating a new row per re-upload/re-analysis.
        """
        file_hash = compute_file_hash(raw_bytes)

        existing = self.repository.find_by_file_hash(file_hash, organization_id)
        if existing is not None:
            self.repository.touch_last_accessed(existing.dataset_id)
            existing_columns = self.repository.get_columns(existing.dataset_id)
            return DatasetRegistration(dataset=existing, columns=existing_columns, was_duplicate=True)

        schema_hash = compute_schema_hash(
            (str(col), str(df[col].dtype)) for col in df.columns
        )

        dataset = Dataset(
            organization_id=organization_id,
            dataset_name=dataset_name,
            uploaded_by=uploaded_by,
            schema_hash=schema_hash,
            file_hash=file_hash,
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
            source_type=source_type,
        )
        dataset = self.repository.create_dataset(dataset)

        column_rows = [self._build_column_row(dataset.dataset_id, df, col) for col in df.columns]
        column_rows = self.repository.add_columns(column_rows)

        return DatasetRegistration(dataset=dataset, columns=column_rows, was_duplicate=False)

    @staticmethod
    def _build_column_row(dataset_id: str, df: pd.DataFrame, column_name: str) -> DatasetColumn:
        series = df[column_name]
        row_count = len(series)
        missing_count = int(series.isnull().sum())
        missing_pct = (missing_count / row_count * 100.0) if row_count > 0 else 0.0

        return DatasetColumn(
            dataset_id=dataset_id,
            column_name=str(column_name),
            detected_type=str(series.dtype),
            nullable=missing_count > 0,
            unique_count=int(series.nunique(dropna=True)),
            missing_percentage=round(missing_pct, 4),
            inferred_role=None,  # filled in later by SchemaIntelligenceService
        )
