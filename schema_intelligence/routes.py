# schema_intelligence/routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.db import get_db
from datasets.repository import DatasetRepository
from datasets.routes import get_dataset_repository

from .repository import ColumnRoleDetectionRepository, DuplicateColumnRepository, RelationshipRepository
from .service import SchemaIntelligenceService

schema_intelligence_router = APIRouter(prefix="/v2/schema-intelligence", tags=["schema-intelligence"])


def get_relationship_repository(db: Session = Depends(get_db)) -> RelationshipRepository:
    return RelationshipRepository(db)


def get_column_detection_repository(db: Session = Depends(get_db)) -> ColumnRoleDetectionRepository:
    return ColumnRoleDetectionRepository(db)


def get_duplicate_repository(db: Session = Depends(get_db)) -> DuplicateColumnRepository:
    return DuplicateColumnRepository(db)


def get_schema_intelligence_service(
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    relationship_repo: RelationshipRepository = Depends(get_relationship_repository),
    column_detection_repo: ColumnRoleDetectionRepository = Depends(get_column_detection_repository),
    duplicate_repo: DuplicateColumnRepository = Depends(get_duplicate_repository),
) -> SchemaIntelligenceService:
    return SchemaIntelligenceService(dataset_repo, relationship_repo, column_detection_repo, duplicate_repo)


class ColumnRoleOut(BaseModel):
    column_name: str
    inferred_role: str | None  # the single winning role, from dataset_columns
    detected_type: str
    unique_count: int
    missing_percentage: float
    # Additive: same winning role's confidence/evidence, now persisted
    # directly on dataset_columns too (see datasets/models.py) — no longer
    # only reachable via /candidates below.
    inferred_role_confidence: float | None = None
    inferred_role_evidence: dict | None = None


class ColumnCandidateOut(BaseModel):
    """One scored candidate for a column — a column can have several of
    these (e.g. "percentage": 0.7 and "numeric": 0.3) even though only the
    top one becomes the winning `inferred_role`."""
    column_name: str
    role: str
    confidence: float
    rule_name: str
    evidence: dict


class DuplicateColumnPairOut(BaseModel):
    column_a: str
    column_b: str
    confidence: float


class RelationshipOut(BaseModel):
    source_dataset_id: str
    source_column: str
    target_dataset_id: str
    target_column: str
    confidence: float


@schema_intelligence_router.get("/{dataset_id}/roles", response_model=list[ColumnRoleOut])
async def get_column_roles(dataset_id: str, dataset_repo: DatasetRepository = Depends(get_dataset_repository)):
    """The single winning role per column — same shape as before this
    feature's rule-engine rewrite. For the FULL confidence-scored candidate
    list behind each winner, see /candidates below."""
    dataset = dataset_repo.get_by_id(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    columns = dataset_repo.get_columns(dataset_id)
    return [
        ColumnRoleOut(
            column_name=c.column_name,
            inferred_role=c.inferred_role,
            detected_type=c.detected_type,
            unique_count=c.unique_count,
            missing_percentage=c.missing_percentage,
            inferred_role_confidence=c.inferred_role_confidence,
            inferred_role_evidence=c.inferred_role_evidence,
        )
        for c in columns
    ]


@schema_intelligence_router.get("/{dataset_id}/candidates", response_model=list[ColumnCandidateOut])
async def get_column_candidates(
    dataset_id: str,
    column_detection_repo: ColumnRoleDetectionRepository = Depends(get_column_detection_repository),
):
    """Every rule's scored opinion about every column — the full audit trail
    behind the winning roles returned by /roles. Each entry names the exact
    rule that produced it and the evidence backing the score."""
    detections = column_detection_repo.list_for_dataset(dataset_id)
    return [
        ColumnCandidateOut(
            column_name=d.column_name,
            role=d.role,
            confidence=d.confidence,
            rule_name=d.rule_name,
            evidence=d.evidence,
        )
        for d in detections
    ]


@schema_intelligence_router.get("/{dataset_id}/duplicate-columns", response_model=list[DuplicateColumnPairOut])
async def get_duplicate_columns(
    dataset_id: str,
    duplicate_repo: DuplicateColumnRepository = Depends(get_duplicate_repository),
):
    pairs = duplicate_repo.list_for_dataset(dataset_id)
    return [
        DuplicateColumnPairOut(column_a=p.column_a, column_b=p.column_b, confidence=p.confidence)
        for p in pairs
    ]


@schema_intelligence_router.get("/{dataset_id}/relationships", response_model=list[RelationshipOut])
async def get_relationships(
    dataset_id: str,
    relationship_repo: RelationshipRepository = Depends(get_relationship_repository),
):
    rows = relationship_repo.list_for_dataset(dataset_id)
    return [
        RelationshipOut(
            source_dataset_id=r.source_dataset_id,
            source_column=r.source_column,
            target_dataset_id=r.target_dataset_id,
            target_column=r.target_column,
            confidence=r.confidence,
        )
        for r in rows
    ]
