# memory_engine/routes.py
# ─────────────────────────────────────────────────────────────────────────────
# HTTP surface for inspection/testing. The REAL integration point for a
# future LLM-based planner is calling AnalyticsMemoryEngine's four methods
# directly, in-process — these endpoints exist so the engine's behavior can
# be verified/demoed without needing that planner wired up yet.
#
# Ranking itself is delegated to a CandidateRanker (see contracts.py) that
# AnalyticsMemoryEngine is constructed with — get_memory_engine() below is
# the one place that choice is wired up, so plugging in a future ML ranker
# means changing that one function, not any route body.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from datasets.repository import DatasetRepository
from datasets.routes import get_dataset_repository
from query_history.repository import QueryHistoryRepository
from query_history.routes import get_query_history_repository

from .exporters import SUPPORTED_EXPORT_FORMATS, TrainingDatasetExporter
from .service import AnalyticsMemoryEngine

memory_engine_router = APIRouter(prefix="/v2/memory-engine", tags=["memory-engine"])


def get_memory_engine(
    history_repo: QueryHistoryRepository = Depends(get_query_history_repository),
) -> AnalyticsMemoryEngine:
    # No `ranker=` passed here -> AnalyticsMemoryEngine falls back to
    # DefaultCandidateRanker (deterministic, no ML). Swapping the whole
    # backend's ranking behavior — including for a future ML-based ranker —
    # is a one-line change to this single function.
    return AnalyticsMemoryEngine(history_repo)


def get_training_dataset_exporter(
    history_repo: QueryHistoryRepository = Depends(get_query_history_repository),
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
) -> TrainingDatasetExporter:
    return TrainingDatasetExporter(history_repo, dataset_repo)


class MemoryMatchOut(BaseModel):
    query_history_id: int
    matched_query: str
    similarity_score: float
    intent: str | None
    schema_hash: str | None
    generated_sql: str | None
    python_pipeline: object | None
    visualization: object | None
    execution_time_ms: float | None
    rows_returned: int | None
    feedback_score: int | None
    planner_version: str | None


class SimilarQueryResponse(BaseModel):
    hit: bool
    match: MemoryMatchOut | None = None


class BestSqlResponse(BaseModel):
    hit: bool
    generated_sql: str | None = None


class BestPipelineResponse(BaseModel):
    hit: bool
    python_pipeline: object | None = None


class FeedbackRequest(BaseModel):
    feedback_score: int


class RankerInfoResponse(BaseModel):
    ranker_name: str


@memory_engine_router.get("/ranker-info", response_model=RankerInfoResponse)
async def ranker_info(engine: AnalyticsMemoryEngine = Depends(get_memory_engine)):
    """Reports which CandidateRanker is currently wired up (see
    contracts.py / get_memory_engine above). Mainly useful once a future ML
    ranker is deployed alongside the default one — this is how a caller (or
    a human debugging a weird match) can confirm which is actually live,
    without needing to read source or redeploy anything."""
    return RankerInfoResponse(ranker_name=engine.ranker.name)


@memory_engine_router.get("/similar-query", response_model=SimilarQueryResponse)
async def similar_query(
    user_query: str,
    dataset_id: str | None = None,
    organization_id: str | None = None,
    schema_hash: str | None = None,
    min_confidence: float | None = None,
    engine: AnalyticsMemoryEngine = Depends(get_memory_engine),
):
    match = engine.find_similar_query(
        user_query=user_query, dataset_id=dataset_id, organization_id=organization_id,
        schema_hash=schema_hash, min_confidence=min_confidence,
    )
    if match is None:
        return SimilarQueryResponse(hit=False)
    return SimilarQueryResponse(hit=True, match=MemoryMatchOut(**match.__dict__))


@memory_engine_router.get("/best-sql", response_model=BestSqlResponse)
async def best_sql(
    user_query: str,
    dataset_id: str | None = None,
    organization_id: str | None = None,
    schema_hash: str | None = None,
    min_confidence: float | None = None,
    engine: AnalyticsMemoryEngine = Depends(get_memory_engine),
):
    sql = engine.find_best_sql(
        user_query=user_query, dataset_id=dataset_id, organization_id=organization_id,
        schema_hash=schema_hash, min_confidence=min_confidence,
    )
    return BestSqlResponse(hit=sql is not None, generated_sql=sql)


@memory_engine_router.get("/best-pipeline", response_model=BestPipelineResponse)
async def best_pipeline(
    user_query: str,
    dataset_id: str | None = None,
    organization_id: str | None = None,
    schema_hash: str | None = None,
    min_confidence: float | None = None,
    engine: AnalyticsMemoryEngine = Depends(get_memory_engine),
):
    pipeline = engine.find_best_pipeline(
        user_query=user_query, dataset_id=dataset_id, organization_id=organization_id,
        schema_hash=schema_hash, min_confidence=min_confidence,
    )
    return BestPipelineResponse(hit=pipeline is not None, python_pipeline=pipeline)


@memory_engine_router.patch("/feedback/{query_history_id}", response_model=MemoryMatchOut)
async def feedback(
    query_history_id: int,
    payload: FeedbackRequest,
    engine: AnalyticsMemoryEngine = Depends(get_memory_engine),
):
    entry = engine.record_feedback(query_history_id, payload.feedback_score)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Query history entry #{query_history_id} not found.")
    return MemoryMatchOut(
        query_history_id=entry.id,
        matched_query=entry.user_query,
        similarity_score=1.0,  # not a match lookup — feedback was applied directly by id
        intent=entry.intent,
        schema_hash=entry.schema_hash,
        generated_sql=entry.generated_sql,
        python_pipeline=entry.python_pipeline,
        visualization=entry.visualization,
        execution_time_ms=entry.execution_time_ms,
        rows_returned=entry.rows_returned,
        feedback_score=entry.feedback_score,
        planner_version=entry.planner_version,
    )


# ── ML readiness: structured training-data export ──────────────────────────
# Prepares data only — see memory_engine/exporters.py's module docstring.
# No model is trained, fit, or evaluated anywhere on this path.

_EXPORT_MEDIA_TYPES = {"csv": "text/csv", "parquet": "application/octet-stream"}


@memory_engine_router.get("/training-export")
async def export_training_dataset(
    format: str = "csv",
    organization_id: str | None = None,
    dataset_id: str | None = None,
    schema_hash: str | None = None,
    only_successful: bool = False,
    limit: int = 5000,
    exporter: TrainingDatasetExporter = Depends(get_training_dataset_exporter),
):
    """Downloadable, structured export of query_history for a future ML
    training pipeline — intent, question, sql, pipeline, execution_time_ms,
    feedback, dataset_type. `format` is "csv" (default) or "parquet".
    Filters mirror /v2/memory-engine's other endpoints: scope by
    organization_id/dataset_id/schema_hash, and set only_successful=true to
    export positive examples only (failures are included by default — see
    TrainingDatasetExporter.collect for why).
    """
    fmt = format.lower()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        raise HTTPException(
            status_code=400, detail=f"format must be one of {SUPPORTED_EXPORT_FORMATS}, got {format!r}"
        )

    entries = exporter.collect(
        organization_id=organization_id,
        dataset_id=dataset_id,
        schema_hash=schema_hash,
        only_successful=only_successful,
        limit=limit,
    )
    body = exporter.export(entries, fmt=fmt)
    return Response(
        content=body,
        media_type=_EXPORT_MEDIA_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="training_export.{fmt}"'},
    )
