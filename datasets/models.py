# datasets/models.py
# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy ORM models for the Dataset Registry. Two tables, exactly as
# specified: `datasets` (one row per uploaded dataset) and `dataset_columns`
# (one row per column of a given dataset — a schema fingerprint that the
# Schema Intelligence service enriches with `inferred_role`).
# ─────────────────────────────────────────────────────────────────────────────

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    organization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    uploaded_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Identity/dedup fields — see datasets/hashing.py for how these are computed.
    schema_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)  # csv|tsv|xlsx|xls|json
    last_accessed: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    columns: Mapped[list["DatasetColumn"]] = relationship(
        "DatasetColumn", back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience only
        return f"<Dataset {self.dataset_id} '{self.dataset_name}' ({self.row_count}x{self.column_count})>"


class DatasetColumn(Base):
    __tablename__ = "dataset_columns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True
    )
    column_name: Mapped[str] = mapped_column(String(255), nullable=False)
    detected_type: Mapped[str] = mapped_column(String(32), nullable=False)  # pandas dtype name
    nullable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    unique_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    missing_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Populated by schema_intelligence.service.SchemaIntelligenceService, NOT
    # by the Dataset Registry itself — kept nullable here since a freshly
    # registered dataset hasn't been analyzed yet the instant it's inserted.
    inferred_role: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Confidence/evidence/timestamp behind THAT SAME winning inferred_role —
    # added so the Dataset Registry carries a complete, self-contained
    # summary of "what role, how confident, why" per column, rather than a
    # bare label with the reasoning only available by cross-referencing
    # schema_intelligence's own tables (column_role_detections), which still
    # separately hold the FULL multi-candidate audit trail (every rule's
    # opinion, not just the winner) for deeper inspection. These three are
    # deliberately nullable and independent of `inferred_role` itself: a
    # role can be set without them (e.g. by a future caller that doesn't
    # have a confidence score to report), and re-analysis simply overwrites
    # all four together — see DatasetRepository.update_column_role_metadata.
    inferred_role_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    inferred_role_evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    role_detected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="columns")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DatasetColumn {self.column_name} role={self.inferred_role}>"
