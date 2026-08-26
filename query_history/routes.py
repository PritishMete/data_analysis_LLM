# query_history/routes.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.db import get_db
from datasets.repository import DatasetRepository
from datasets.routes import get_dataset_repository

from .repository import QueryHistoryRepository
from .schemas import FeedbackUpdate, QueryHistoryCreate, QueryHistoryOut
from .service import QueryHistoryService

query_history_router = APIRouter(prefix="/v2/query-history", tags=["query-history"])


def get_query_history_repository(db: Session = Depends(get_db)) -> QueryHistoryRepository:
    return QueryHistoryRepository(db)


def get_query_history_service(
    repo: QueryHistoryRepository = Depends(get_query_history_repository),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
) -> QueryHistoryService:
    # dataset_repo is used ONLY to auto-resolve schema_hash from dataset_id
    # (see QueryHistoryService._resolve_schema_hash) — a read-only reuse of
    # the Dataset Registry's existing repository, nothing about it modified.
    return QueryHistoryService(repo, dataset_repo)


@query_history_router.post("", response_model=QueryHistoryOut)
async def log_query(
    payload: QueryHistoryCreate,
    service: QueryHistoryService = Depends(get_query_history_service),
):
    """Manual/explicit logging endpoint. For AUTOMATIC logging wrapped around
    an actual execution (with timing + exception-safe success detection
    built in), use QueryHistoryService.track(...) in-process instead — see
    the class docstring on QueryExecutionTracker in service.py, and
    INTEGRATION.md for where to drop it into query_router.py/command_agent.py.
    """
    return service.log_execution(
        user_query=payload.user_query,
        intent=payload.intent,
        generated_sql=payload.generated_sql,
        python_pipeline=payload.python_pipeline,
        visualization=payload.visualization,
        execution_time_ms=payload.execution_time_ms,
        rows_returned=payload.rows_returned,
        dataset_id=payload.dataset_id,
        organization_id=payload.organization_id,
        success=payload.success,
        error_message=payload.error_message,
        planner_version=payload.planner_version,
    )


@query_history_router.get("", response_model=list[QueryHistoryOut])
async def list_query_history(
    organization_id: str | None = None,
    dataset_id: str | None = None,
    success: bool | None = None,
    planner_version: str | None = None,
    limit: int = 50,
    service: QueryHistoryService = Depends(get_query_history_service),
):
    return service.get_history(
        organization_id=organization_id,
        dataset_id=dataset_id,
        success=success,
        planner_version=planner_version,
        limit=limit,
    )


@query_history_router.get("/training-export", response_model=list[QueryHistoryOut])
async def export_training_examples(
    schema_hash: str | None = None,
    only_successful: bool = True,
    limit: int = 5000,
    service: QueryHistoryService = Depends(get_query_history_service),
):
    """Bulk export shaped for a future ML training job — see
    QueryHistoryRepository.list_for_training. Optionally scoped to one
    schema_hash (i.e. "give me every example seen against datasets shaped
    like this one"), which is exactly the grouping key a model generalizing
    across datasets would want to train on.
    """
    return service.get_training_examples(schema_hash=schema_hash, only_successful=only_successful, limit=limit)


@query_history_router.patch("/{entry_id}/feedback", response_model=QueryHistoryOut)
async def submit_feedback(
    entry_id: int,
    payload: FeedbackUpdate,
    service: QueryHistoryService = Depends(get_query_history_service),
):
    entry = service.record_feedback(entry_id, payload.feedback_score)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Query history entry #{entry_id} not found.")
    return entry
