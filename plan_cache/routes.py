# plan_cache/routes.py
# ─────────────────────────────────────────────────────────────────────────────
# Exposes plan-cache lookup/invalidation over HTTP for completeness/
# inspection, but the real integration point is calling
# PlanCacheService.evaluate() / find_cached_plan() IN-PROCESS from
# query_router.py / command_agent.py's dispatch path, before the Gemini
# call — see INTEGRATION.md. This endpoint lets the Flutter app (or a
# curious engineer) check cache-hit/miss/expiration/invalidation behavior
# directly without needing that deeper integration done first, and without
# touching any planner code.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.db import get_db
from datasets.repository import DatasetRepository
from datasets.routes import get_dataset_repository

from .repository import PlanCacheRepository
from .service import DEFAULT_MIN_CONFIDENCE, PlanCacheService

plan_cache_router = APIRouter(prefix="/v2/plan-cache", tags=["plan-cache"])


def get_plan_cache_repository(db: Session = Depends(get_db)) -> PlanCacheRepository:
    return PlanCacheRepository(db)


def get_plan_cache_service(
    dataset_repo: DatasetRepository = Depends(get_dataset_repository),
    plan_cache_repo: PlanCacheRepository = Depends(get_plan_cache_repository),
) -> PlanCacheService:
    return PlanCacheService(dataset_repo, plan_cache_repo)


class PlanCacheHitOut(BaseModel):
    generated_sql: str | None
    python_pipeline: object | None
    intent: str | None
    planner_version: str | None
    confidence: float
    source_dataset_id: str
    matched_on: str
    original_query_history_id: int


class PlanCacheLookupResponse(BaseModel):
    # `hit`/`plan` kept exactly as before for any existing consumer that
    # only checks those two fields. `outcome`/`detail` are additive.
    hit: bool
    plan: PlanCacheHitOut | None
    outcome: str
    detail: str | None = None


class InvalidateQueryRequest(BaseModel):
    query_history_id: int
    reason: str | None = None


class InvalidateScopeRequest(BaseModel):
    dataset_id: str
    intent: str | None = None
    planner_version: str | None = None
    reason: str | None = None


class InvalidationOut(BaseModel):
    id: int
    query_history_id: int | None
    schema_hash: str | None
    intent: str | None
    planner_version: str | None
    reason: str | None


@plan_cache_router.get("/lookup", response_model=PlanCacheLookupResponse)
async def lookup_plan(
    dataset_id: str,
    user_query: str,
    intent: str | None = None,
    planner_version: str | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    service: PlanCacheService = Depends(get_plan_cache_service),
):
    result = service.evaluate(
        dataset_id=dataset_id,
        user_query=user_query,
        intent=intent,
        planner_version=planner_version,
        min_confidence=min_confidence,
    )
    plan_out = None
    if result.hit is not None:
        plan_out = PlanCacheHitOut(
            generated_sql=result.hit.generated_sql,
            python_pipeline=result.hit.python_pipeline,
            intent=result.hit.intent,
            planner_version=result.hit.planner_version,
            confidence=result.hit.confidence,
            source_dataset_id=result.hit.source_dataset_id,
            matched_on=result.hit.matched_on,
            original_query_history_id=result.hit.original_query_history_id,
        )
    return PlanCacheLookupResponse(
        hit=result.is_hit, plan=plan_out, outcome=result.outcome.value, detail=result.detail
    )


@plan_cache_router.post("/invalidate/query", response_model=InvalidationOut)
async def invalidate_query(
    payload: InvalidateQueryRequest,
    service: PlanCacheService = Depends(get_plan_cache_service),
):
    """Marks one specific cached plan (by its source query_history row id)
    as no longer reusable. Does not delete or alter that query_history
    row — it stays a permanent, accurate historical log."""
    entry = service.invalidate_plan(query_history_id=payload.query_history_id, reason=payload.reason)
    return InvalidationOut(
        id=entry.id,
        query_history_id=entry.query_history_id,
        schema_hash=entry.schema_hash,
        intent=entry.intent,
        planner_version=entry.planner_version,
        reason=entry.reason,
    )


@plan_cache_router.post("/invalidate/scope", response_model=InvalidationOut)
async def invalidate_scope(
    payload: InvalidateScopeRequest,
    service: PlanCacheService = Depends(get_plan_cache_service),
):
    """Marks every plan currently cached under `dataset_id`'s schema
    (optionally narrowed to one intent and/or planner_version) as no
    longer reusable. A fresh, successful execution logged AFTER this call
    is unaffected."""
    try:
        entry = service.invalidate_scope(
            dataset_id=payload.dataset_id,
            intent=payload.intent,
            planner_version=payload.planner_version,
            reason=payload.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return InvalidationOut(
        id=entry.id,
        query_history_id=entry.query_history_id,
        schema_hash=entry.schema_hash,
        intent=entry.intent,
        planner_version=entry.planner_version,
        reason=entry.reason,
    )
