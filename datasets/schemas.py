# datasets/schemas.py
# ─────────────────────────────────────────────────────────────────────────────
# Pydantic I/O models for the Dataset Registry API. Kept separate from the
# SQLAlchemy models in models.py on purpose (clean architecture: persistence
# shape and wire shape are allowed to diverge even when they look similar
# today).
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime

from pydantic import BaseModel


class DatasetColumnOut(BaseModel):
    column_name: str
    detected_type: str
    nullable: bool
    unique_count: int
    missing_percentage: float
    inferred_role: str | None
    # Additive: confidence/evidence/timestamp behind that same inferred_role,
    # persisted directly on DatasetColumn (see datasets/models.py) rather
    # than only living in schema_intelligence's separate audit tables — so
    # this same Dataset Registry endpoint already carries them, with zero
    # schema_intelligence import needed here.
    inferred_role_confidence: float | None = None
    inferred_role_evidence: dict | None = None
    role_detected_at: datetime | None = None

    model_config = {"from_attributes": True}


class DatasetOut(BaseModel):
    dataset_id: str
    organization_id: str
    dataset_name: str
    uploaded_by: str | None
    created_at: datetime
    schema_hash: str
    file_hash: str
    row_count: int
    column_count: int
    source_type: str
    last_accessed: datetime

    model_config = {"from_attributes": True}


class DatasetRegisterResponse(BaseModel):
    dataset: DatasetOut
    columns: list[DatasetColumnOut]
    was_duplicate: bool  # true if this exact file_hash was already registered
    # for this organization — the existing row was returned/touched instead
    # of inserting a new one.
