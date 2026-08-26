# query_history/repository.py
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import QueryHistory


class QueryHistoryRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, entry: QueryHistory) -> QueryHistory:
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def get_by_id(self, entry_id: int) -> QueryHistory | None:
        return self.db.get(QueryHistory, entry_id)

    def set_feedback(self, entry_id: int, feedback_score: int) -> QueryHistory | None:
        entry = self.get_by_id(entry_id)
        if entry is not None:
            entry.feedback_score = feedback_score
            self.db.commit()
            self.db.refresh(entry)
        return entry

    def list_recent(
        self,
        *,
        organization_id: str | None = None,
        dataset_id: str | None = None,
        success: bool | None = None,
        planner_version: str | None = None,
        limit: int = 50,
    ) -> list[QueryHistory]:
        stmt = select(QueryHistory)
        if organization_id is not None:
            stmt = stmt.where(QueryHistory.organization_id == organization_id)
        if dataset_id is not None:
            stmt = stmt.where(QueryHistory.dataset_id == dataset_id)
        if success is not None:
            stmt = stmt.where(QueryHistory.success == success)
        if planner_version is not None:
            stmt = stmt.where(QueryHistory.planner_version == planner_version)
        stmt = stmt.order_by(QueryHistory.created_at.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def list_for_training(
        self,
        *,
        schema_hash: str | None = None,
        only_successful: bool = True,
        limit: int = 5000,
    ) -> list[QueryHistory]:
        """Bulk export shape for a future ML training pipeline — optionally
        scoped to one schema shape (via the denormalized `schema_hash`, no
        join needed), and defaulting to successful-only examples since a
        model learning "given this question + this schema, produce this
        SQL/pipeline" needs labeled-correct examples first. Set
        only_successful=False to also pull failures for a future model that
        learns what NOT to produce.
        """
        stmt = select(QueryHistory)
        if schema_hash is not None:
            stmt = stmt.where(QueryHistory.schema_hash == schema_hash)
        if only_successful:
            stmt = stmt.where(QueryHistory.success.is_(True))
        stmt = stmt.order_by(QueryHistory.created_at.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def list_candidates(
        self,
        *,
        organization_id: str | None = None,
        dataset_id: str | None = None,
        schema_hash: str | None = None,
        success: bool | None = None,
        limit: int = 200,
    ) -> list[QueryHistory]:
        """The general-purpose candidate query — supports combining ALL
        THREE scoping dimensions (org, dataset, schema_hash) at once, unlike
        list_recent (org/dataset only) or list_for_training (schema_hash
        only). Added for memory_engine/service.py, which needs to gather
        candidates scoped however the caller specifies — e.g. "same
        organization AND same schema shape" for cross-dataset reuse within
        a tenant, without pulling every row and filtering in Python.
        """
        stmt = select(QueryHistory)
        if organization_id is not None:
            stmt = stmt.where(QueryHistory.organization_id == organization_id)
        if dataset_id is not None:
            stmt = stmt.where(QueryHistory.dataset_id == dataset_id)
        if schema_hash is not None:
            stmt = stmt.where(QueryHistory.schema_hash == schema_hash)
        if success is not None:
            stmt = stmt.where(QueryHistory.success == success)
        stmt = stmt.order_by(QueryHistory.created_at.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def find_similar_successful(
        self, *, user_query: str, dataset_id: str | None, limit: int = 5
    ) -> list[QueryHistory]:
        """Deterministic (non-fuzzy, non-AI) exact-text match against past
        SUCCESSFUL runs for the same dataset — the simplest possible seed for
        the "backend gets smarter than the LLM" goal: if we've literally seen
        this exact question against this exact dataset before and it
        succeeded, that's a free, instant answer with no LLM call needed.
        Fuzzy/semantic matching would be a natural next step, but that's a
        separate, larger feature — this is the deterministic floor it would
        build on.
        """
        stmt = (
            select(QueryHistory)
            .where(
                QueryHistory.user_query == user_query,
                QueryHistory.success.is_(True),
            )
        )
        if dataset_id is not None:
            stmt = stmt.where(QueryHistory.dataset_id == dataset_id)
        stmt = stmt.order_by(QueryHistory.created_at.desc()).limit(limit)
        return list(self.db.execute(stmt).scalars().all())
