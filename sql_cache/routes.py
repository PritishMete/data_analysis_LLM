# sql_cache/routes.py
# ─────────────────────────────────────────────────────────────────────────────
# The middleware is what actually intercepts real traffic — this endpoint
# exists purely so the cache's behavior can be inspected/tested directly
# (e.g. from Flutter, or curl) without needing to fire a real /agentic_command
# request and hope for a hit. Same reasoning as plan_cache_router's /lookup.
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from query_history.repository import QueryHistoryRepository
from query_history.routes import get_query_history_repository

from .service import SqlCacheService

sql_cache_router = APIRouter(prefix="/v2/sql-cache", tags=["sql-cache"])


def get_sql_cache_service(
    history_repo: QueryHistoryRepository = Depends(get_query_history_repository),
) -> SqlCacheService:
    return SqlCacheService(history_repo)


class SqlCacheLookupResponse(BaseModel):
    hit: bool
    generated_sql: str | None = None
    python_pipeline: object | None = None
    intent: str | None = None
    matched_query: str | None = None
    similarity_score: float | None = None
    source_query_history_id: int | None = None
    planner_version: str | None = None


@sql_cache_router.get("/lookup", response_model=SqlCacheLookupResponse)
async def lookup(
    user_query: str,
    dataset_id: str | None = None,
    organization_id: str | None = None,
    service: SqlCacheService = Depends(get_sql_cache_service),
):
    result = service.find_similar_cached_query(
        user_query=user_query, dataset_id=dataset_id, organization_id=organization_id
    )
    if result is None:
        return SqlCacheLookupResponse(hit=False)
    return SqlCacheLookupResponse(
        hit=True,
        generated_sql=result.generated_sql,
        python_pipeline=result.python_pipeline,
        intent=result.intent,
        matched_query=result.matched_query,
        similarity_score=result.similarity_score,
        source_query_history_id=result.source_query_history_id,
        planner_version=result.planner_version,
    )
