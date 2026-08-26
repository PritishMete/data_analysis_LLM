# plan_cache/repository.py
# ─────────────────────────────────────────────────────────────────────────────
# The queries that make "smarter than the LLM" concrete: join query_history
# against datasets on schema_hash (not dataset_id), so a plan that succeeded
# for ONE customer's upload can be reused for a DIFFERENT dataset that
# merely happens to have the same column shape — e.g. two different orgs
# both uploading a "Region, Product, Quantity, Revenue" export. This is the
# cross-dataset generalization the plain dataset_id match in
# query_history.repository.find_similar_successful() doesn't give you on
# its own.
#
# Two tiers of candidate, both scoped to `success.is_(True)` and, now,
# excluded of anything invalidated (see PlanCacheInvalidation):
#   - find_reusable_plan(): EXACT (case-sensitive) text match on
#     user_query. Highest-confidence tier — see service.py's
#     _compute_confidence.
#   - find_by_intent(): broader match on (schema_hash, intent
#     [, planner_version]) regardless of exact wording. This is the new
#     "cache using intent / schema hash / planner version" tier — two
#     differently-phrased questions with the same intent, against
#     same-shaped data, produced by the same planner, are treated as the
#     same reusable experience.
# Both return a bounded, ordered-by-recency LIST (not just the single best
# row) — the caller (service.py) needs a few candidates, not just one, to
# correctly separate "genuinely nothing matches" from "something matched
# but every candidate is invalidated or has expired".
# ─────────────────────────────────────────────────────────────────────────────

from sqlalchemy import select
from sqlalchemy.orm import Session

from datasets.models import Dataset
from query_history.models import QueryHistory

from .models import PlanCacheInvalidation


class PlanCacheRepository:
    def __init__(self, db: Session):
        self.db = db

    # ── Candidate lookup ────────────────────────────────────────────────

    def find_reusable_plan(
        self,
        *,
        schema_hash: str,
        user_query: str,
        limit: int = 5,
    ) -> list[QueryHistory]:
        """Most recent SUCCESSFUL query_history rows whose associated
        dataset shares `schema_hash`, with an exact (case-sensitive) match on
        `user_query` text. Deterministic — no fuzzy/semantic matching here;
        see PlanCacheService docstring for why that's a deliberate, separate
        next step rather than bundled into this one. `limit` defaults to 5
        (not 1) so the service layer can skip over invalidated/expired
        candidates without a second round-trip in the common case.
        """
        stmt = (
            select(QueryHistory)
            .join(Dataset, Dataset.dataset_id == QueryHistory.dataset_id)
            .where(
                Dataset.schema_hash == schema_hash,
                QueryHistory.user_query == user_query,
                QueryHistory.success.is_(True),
            )
            .order_by(QueryHistory.created_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    def find_by_intent(
        self,
        *,
        schema_hash: str,
        intent: str,
        planner_version: str | None = None,
        limit: int = 5,
    ) -> list[QueryHistory]:
        """Broader tier: most recent SUCCESSFUL rows sharing `schema_hash`
        and `intent`, regardless of exact `user_query` wording. When
        `planner_version` is given, also requires an exact match on it —
        deliberately no "any planner_version" fallback here (that's what
        omitting the argument is for): a plan produced by one planner
        version reusable-by-default for every future planner version would
        quietly undermine the entire point of tracking planner_version in
        the first place.
        """
        conditions = [
            Dataset.schema_hash == schema_hash,
            QueryHistory.intent == intent,
            QueryHistory.success.is_(True),
        ]
        if planner_version is not None:
            conditions.append(QueryHistory.planner_version == planner_version)

        stmt = (
            select(QueryHistory)
            .join(Dataset, Dataset.dataset_id == QueryHistory.dataset_id)
            .where(*conditions)
            .order_by(QueryHistory.created_at.desc())
            .limit(limit)
        )
        return list(self.db.execute(stmt).scalars().all())

    # ── Invalidation ────────────────────────────────────────────────────

    def invalidated_ids_among(self, query_history_ids: list[int]) -> set[int]:
        """Which of these specific query_history ids have a per-row
        invalidation on file. Empty input -> empty result without a query."""
        if not query_history_ids:
            return set()
        stmt = select(PlanCacheInvalidation.query_history_id).where(
            PlanCacheInvalidation.query_history_id.in_(query_history_ids)
        )
        return {row[0] for row in self.db.execute(stmt).all()}

    def scope_invalidations(self, *, schema_hash: str) -> list[PlanCacheInvalidation]:
        """All scope-level invalidation rules (query_history_id IS NULL) on
        file for this schema_hash — service.py checks each candidate
        against these individually (intent / planner_version wildcards and
        the invalidated_at-vs-created_at cutoff can't be expressed as a
        single flat WHERE per-candidate, and this table is expected to stay
        small)."""
        stmt = select(PlanCacheInvalidation).where(
            PlanCacheInvalidation.schema_hash == schema_hash,
            PlanCacheInvalidation.query_history_id.is_(None),
        )
        return list(self.db.execute(stmt).scalars().all())

    def invalidate_query(
        self, *, query_history_id: int, reason: str | None = None
    ) -> PlanCacheInvalidation:
        """Invalidate one specific cached plan by its source query_history
        row id. Denormalizes schema_hash/intent/planner_version from that
        row (when it still exists) purely for auditability — matching logic
        never needs them for this kind of rule, only the id."""
        source = self.db.get(QueryHistory, query_history_id)
        entry = PlanCacheInvalidation(
            query_history_id=query_history_id,
            schema_hash=source.schema_hash if source else None,
            intent=source.intent if source else None,
            planner_version=source.planner_version if source else None,
            reason=reason,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def invalidate_scope(
        self,
        *,
        schema_hash: str,
        intent: str | None = None,
        planner_version: str | None = None,
        reason: str | None = None,
    ) -> PlanCacheInvalidation:
        """Invalidate every plan currently cached under this key (or this
        key's wildcarded superset, if intent/planner_version are omitted).
        Only affects rows that already existed as of now — see
        PlanCacheService.is_invalidated for the cutoff logic."""
        entry = PlanCacheInvalidation(
            query_history_id=None,
            schema_hash=schema_hash,
            intent=intent,
            planner_version=planner_version,
            reason=reason,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry
