# sql_cache/service.py
# ─────────────────────────────────────────────────────────────────────────────
# The core decision: "has a query this similar already succeeded?" Reuses
# QueryHistoryRepository AS-IS (no changes to query_history/ at all) as the
# candidate source — this package adds a NEW way to search that existing
# data, not a new place to store it. No new tables, no new persistence layer.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass

from query_history.repository import QueryHistoryRepository

from .contracts import SimilarityStrategy
from .strategies import TextSimilarityStrategy


@dataclass
class SqlCacheHit:
    generated_sql: str | None
    python_pipeline: dict | list | None
    intent: str | None
    matched_query: str
    similarity_score: float
    source_query_history_id: int
    planner_version: str | None = None


class SqlCacheService:
    """Both the SIMILARITY ALGORITHM and the CANDIDATE SOURCE are injected —
    that's the entire "modular and replaceable" story. Construct with a
    different SimilarityStrategy (e.g. a future embedding-based one) or point
    it at a different repository entirely, and nothing else in this class,
    the middleware, or main.py's wiring needs to change.
    """

    def __init__(
        self,
        history_repository: QueryHistoryRepository,
        similarity_strategy: SimilarityStrategy | None = None,
        min_confidence: float = 0.95,
        max_candidates: int = 200,
    ):
        self.history_repository = history_repository
        self.similarity_strategy = similarity_strategy or TextSimilarityStrategy()
        self.min_confidence = min_confidence
        self.max_candidates = max_candidates

    def find_similar_cached_query(
        self,
        *,
        user_query: str,
        dataset_id: str | None = None,
        organization_id: str | None = None,
    ) -> SqlCacheHit | None:
        """Scans past SUCCESSFUL executions (scoped to `dataset_id` and/or
        `organization_id` when given — unscoped, i.e. neither provided, scans
        the most recent successful executions across everything, which is
        looser but still useful when no dataset context is available, e.g.
        an ad-hoc /smart_query call with no registered dataset). Returns the
        best-scoring match if it clears `min_confidence`, else None.
        """
        candidates = self.history_repository.list_recent(
            organization_id=organization_id,
            dataset_id=dataset_id,
            success=True,
            limit=self.max_candidates,
        )

        best_score = 0.0
        best_entry = None
        for entry in candidates:
            score = self.similarity_strategy.score(user_query, entry.user_query)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self.min_confidence:
            return None

        return SqlCacheHit(
            generated_sql=best_entry.generated_sql,
            python_pipeline=best_entry.python_pipeline,
            intent=best_entry.intent,
            matched_query=best_entry.user_query,
            similarity_score=best_score,
            source_query_history_id=best_entry.id,
            planner_version=best_entry.planner_version,
        )
