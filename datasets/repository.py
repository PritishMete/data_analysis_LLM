# datasets/repository.py
# ─────────────────────────────────────────────────────────────────────────────
# Repository layer: the ONLY place in this feature that talks SQLAlchemy
# Session/queries directly. DatasetRegistryService (service.py) depends on
# this via constructor injection and never touches the Session itself —
# standard repository-pattern separation so the service layer stays testable
# without spinning up a real DB (mock the repository in service-level tests;
# the repository itself is what test_dataset_registry.py exercises against
# real SQLite).
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Dataset, DatasetColumn


class DatasetRepository:
    def __init__(self, db: Session):
        self.db = db

    # ── Writes ──────────────────────────────────────────────────────────────

    def create_dataset(self, dataset: Dataset) -> Dataset:
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset

    def add_columns(self, columns: list[DatasetColumn]) -> list[DatasetColumn]:
        self.db.add_all(columns)
        self.db.commit()
        for c in columns:
            self.db.refresh(c)
        return columns

    def touch_last_accessed(self, dataset_id: str) -> None:
        dataset = self.get_by_id(dataset_id)
        if dataset is not None:
            dataset.last_accessed = datetime.now(timezone.utc)
            self.db.commit()

    def update_column_role(self, dataset_id: str, column_name: str, inferred_role: str | None) -> None:
        stmt = select(DatasetColumn).where(
            DatasetColumn.dataset_id == dataset_id,
            DatasetColumn.column_name == column_name,
        )
        column = self.db.execute(stmt).scalars().first()
        if column is not None:
            column.inferred_role = inferred_role
            self.db.commit()

    def update_column_role_metadata(
        self,
        dataset_id: str,
        column_name: str,
        *,
        confidence: float | None,
        evidence: dict | None = None,
        detected_at: datetime | None = None,
    ) -> None:
        """Additive companion to update_column_role() above — persists the
        confidence score (and supporting evidence) behind the winning
        inferred_role directly on this DatasetColumn row, so that
        information lives inside the Dataset Registry itself rather than
        only in schema_intelligence's separate audit tables. A SEPARATE
        method rather than new parameters on update_column_role(), so every
        existing caller of that method keeps working completely unchanged;
        this one is additive, not a replacement. `detected_at` defaults to
        now — pass an explicit value only if the detection actually
        happened earlier (e.g. re-persisting a stored result)."""
        stmt = select(DatasetColumn).where(
            DatasetColumn.dataset_id == dataset_id,
            DatasetColumn.column_name == column_name,
        )
        column = self.db.execute(stmt).scalars().first()
        if column is not None:
            column.inferred_role_confidence = confidence
            column.inferred_role_evidence = evidence
            column.role_detected_at = detected_at or datetime.now(timezone.utc)
            self.db.commit()

    # ── Reads ───────────────────────────────────────────────────────────────

    def get_by_id(self, dataset_id: str) -> Dataset | None:
        return self.db.get(Dataset, dataset_id)

    def find_by_file_hash(self, file_hash: str, organization_id: str) -> Dataset | None:
        stmt = select(Dataset).where(
            Dataset.file_hash == file_hash,
            Dataset.organization_id == organization_id,
        )
        return self.db.execute(stmt).scalars().first()

    def list_by_organization(self, organization_id: str, limit: int = 50) -> list[Dataset]:
        stmt = (
            select(Dataset)
            .where(Dataset.organization_id == organization_id)
            .order_by(Dataset.created_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def get_columns(self, dataset_id: str) -> list[DatasetColumn]:
        stmt = select(DatasetColumn).where(DatasetColumn.dataset_id == dataset_id)
        return list(self.db.execute(stmt).scalars().all())

    def list_all_for_organization_excluding(
        self, organization_id: str, exclude_dataset_id: str, limit: int = 25
    ) -> list[Dataset]:
        """Used by schema_intelligence's relationship detector to find OTHER
        datasets in the same org to compare candidate foreign keys against."""
        stmt = (
            select(Dataset)
            .where(
                Dataset.organization_id == organization_id,
                Dataset.dataset_id != exclude_dataset_id,
            )
            .order_by(Dataset.created_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())
