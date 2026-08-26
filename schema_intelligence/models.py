# schema_intelligence/models.py
# ─────────────────────────────────────────────────────────────────────────────
# Three tables, none of which touch datasets/models.py:
#   - DatasetRelationship    (existing — confirmed cross-dataset FK links)
#   - ColumnRoleDetection    (NEW — full, confidence-scored audit trail of
#                             every rule's opinion about every column, not
#                             just the single winning role)
#   - DuplicateColumnPair    (NEW — pairs of columns detected as duplicates)
#
# `dataset_columns.inferred_role` (in datasets/models.py) still holds the
# single WINNING role per column, written via the existing
# DatasetRepository.update_column_role() — reused, not modified. These new
# tables hold the richer, multi-candidate, evidence-carrying picture behind
# that one summary value, since a single string column has nowhere to put a
# confidence score or competing candidates without altering the Dataset
# Registry's own schema, which this feature is explicitly not allowed to do.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from core.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetRelationship(Base):
    __tablename__ = "dataset_relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source_dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_column: Mapped[str] = mapped_column(String(255), nullable=False)

    target_dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)

    # Fraction (0.0-1.0) of source_column's non-null values found in
    # target_column's value set — how the relationship was scored, deterministically.
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DatasetRelationship {self.source_dataset_id}.{self.source_column} -> "
            f"{self.target_dataset_id}.{self.target_column} ({self.confidence:.2f})>"
        )


class ColumnRoleDetection(Base):
    """One row per (column, candidate role) — a column with ambiguous
    evidence can have SEVERAL rows here (e.g. "percentage": 0.7 AND
    "numeric": 0.3), unlike dataset_columns.inferred_role which only ever
    holds the single winner. `rule_name` + `evidence` make every score
    traceable back to exactly why it was assigned, with no black box.
    """

    __tablename__ = "column_role_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True
    )
    column_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rule_name: Mapped[str] = mapped_column(String(128), nullable=False)
    evidence: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ColumnRoleDetection {self.column_name}={self.role} ({self.confidence:.2f})>"


class DuplicateColumnPair(Base):
    """One row per detected duplicate-column pair within a single dataset."""

    __tablename__ = "duplicate_column_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.dataset_id", ondelete="CASCADE"), nullable=False, index=True
    )
    column_a: Mapped[str] = mapped_column(String(255), nullable=False)
    column_b: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DuplicateColumnPair {self.column_a}=={self.column_b} ({self.confidence:.2f})>"
