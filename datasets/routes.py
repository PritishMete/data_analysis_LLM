# datasets/routes.py
# ─────────────────────────────────────────────────────────────────────────────
# Read/management endpoints for the Dataset Registry. The actual "upload a
# file and register it" flow (which also triggers Schema Intelligence) lives
# in ingestion/routes.py — this router is deliberately just the registry's
# own CRUD-ish surface, so this package doesn't need to import
# schema_intelligence at all (keeps the dependency direction one-way).
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.db import get_db

from .repository import DatasetRepository
from .schemas import DatasetColumnOut, DatasetOut
from .service import DatasetRegistryService

dataset_registry_router = APIRouter(prefix="/v2/datasets", tags=["dataset-registry"])


# ── Dependency injection chain: Session -> Repository -> Service ────────────

def get_dataset_repository(db: Session = Depends(get_db)) -> DatasetRepository:
    return DatasetRepository(db)


def get_dataset_registry_service(
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetRegistryService:
    return DatasetRegistryService(repo)


# ── Routes ────────────────────────────────────────────────────────────────────

@dataset_registry_router.get("", response_model=list[DatasetOut])
async def list_datasets(
    organization_id: str,
    limit: int = 50,
    repo: DatasetRepository = Depends(get_dataset_repository),
):
    """Lists datasets registered for an organization, most recent first."""
    return repo.list_by_organization(organization_id, limit=limit)


@dataset_registry_router.get("/{dataset_id}", response_model=DatasetOut)
async def get_dataset(dataset_id: str, repo: DatasetRepository = Depends(get_dataset_repository)):
    dataset = repo.get_by_id(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return dataset


@dataset_registry_router.get("/{dataset_id}/columns", response_model=list[DatasetColumnOut])
async def get_dataset_columns(dataset_id: str, repo: DatasetRepository = Depends(get_dataset_repository)):
    dataset = repo.get_by_id(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return repo.get_columns(dataset_id)
