# ingestion/routes.py
# ─────────────────────────────────────────────────────────────────────────────
# The single new "smart upload" endpoint. This is ADDITIVE — it does not
# replace /clean_data, /agentic_clean_data, or /smart_query in main.py; it
# sits alongside them. Existing endpoints keep working exactly as they do
# today. See INTEGRATION.md for the optional, minimal hook to also call this
# from inside an existing upload endpoint if/when you want automatic
# registration on every upload rather than only via this dedicated route.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from common.file_parsing import read_file_to_dataframe
from core.db import get_db
from datasets.repository import DatasetRepository
from datasets.routes import get_dataset_repository
from datasets.schemas import DatasetColumnOut, DatasetOut
from datasets.service import DatasetRegistryService
from schema_intelligence.repository import ColumnRoleDetectionRepository, DuplicateColumnRepository, RelationshipRepository
from schema_intelligence.routes import get_column_detection_repository, get_duplicate_repository, get_relationship_repository
from schema_intelligence.service import SchemaIntelligenceService

from .service import DatasetIngestionOrchestrator

ingestion_router = APIRouter(prefix="/v2/ingest", tags=["ingestion"])


def get_registry_service(repo: DatasetRepository = Depends(get_dataset_repository)) -> DatasetRegistryService:
    return DatasetRegistryService(repo)


def get_intelligence_service(
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    relationship_repo: RelationshipRepository = Depends(get_relationship_repository),
    column_detection_repo: ColumnRoleDetectionRepository = Depends(get_column_detection_repository),
    duplicate_repo: DuplicateColumnRepository = Depends(get_duplicate_repository),
) -> SchemaIntelligenceService:
    return SchemaIntelligenceService(dataset_repo, relationship_repo, column_detection_repo, duplicate_repo)


def get_orchestrator(
    registry_service: DatasetRegistryService = Depends(get_registry_service),
    intelligence_service: SchemaIntelligenceService = Depends(get_intelligence_service),
) -> DatasetIngestionOrchestrator:
    return DatasetIngestionOrchestrator(registry_service, intelligence_service)


class RelationshipCandidateOut(BaseModel):
    source_column: str
    target_dataset_id: str
    target_column: str
    confidence: float


class IngestResponse(BaseModel):
    dataset: DatasetOut
    columns: list[DatasetColumnOut]
    was_duplicate: bool
    relationships: list[RelationshipCandidateOut]


@ingestion_router.post("/dataset", response_model=IngestResponse)
async def ingest_dataset(
    file: UploadFile = File(...),
    organization_id: str = Form(...),
    dataset_name: str | None = Form(None),
    uploaded_by: str | None = Form(None),
    db: Session = Depends(get_db),
    orchestrator: DatasetIngestionOrchestrator = Depends(get_orchestrator),
):
    raw_bytes = await file.read()
    filename = file.filename or "upload.csv"
    df = read_file_to_dataframe(filename, raw_bytes)

    source_type = filename.lower().rsplit(".", 1)[-1] if "." in filename else "csv"

    result = orchestrator.ingest(
        df=df,
        raw_bytes=raw_bytes,
        organization_id=organization_id,
        dataset_name=dataset_name or filename,
        uploaded_by=uploaded_by,
        source_type=source_type,
    )

    columns_out = [
        DatasetColumnOut(
            column_name=c.column_name,
            detected_type=c.detected_type,
            nullable=c.nullable,
            unique_count=c.unique_count,
            missing_percentage=c.missing_percentage,
            inferred_role=result.column_roles.get(c.column_name, c.inferred_role),
        )
        for c in result.registration.columns
    ]

    return IngestResponse(
        dataset=result.registration.dataset,
        columns=columns_out,
        was_duplicate=result.registration.was_duplicate,
        relationships=[
            RelationshipCandidateOut(
                source_column=r.source_column,
                target_dataset_id=r.target_dataset_id,
                target_column=r.target_column,
                confidence=r.confidence,
            )
            for r in result.relationships
        ],
    )
