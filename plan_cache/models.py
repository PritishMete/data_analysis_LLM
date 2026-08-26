# plan_cache/models.py
# ─────────────────────────────────────────────────────────────────────────────
# Plan Cache still has no table for the plans themselves — it stays a
# read-through cache computed live over query_history (see repository.py's
# module docstring), the same "no new tables, reuse query_history as the
# only source of truth" choice memory_engine and sql_cache both make. A
# derived VALUE (which SQL/pipeline to reuse) never needs its own storage;
# it's recomputed from query_history + this table on every lookup.
#
# But "cache expiration" and, especially, "cache invalidation" are STATEFUL
# by definition — you cannot invalidate a value that's recomputed fresh on
# every read unless something, somewhere, remembers that the invalidation
# happened. That's the one genuinely new fact this package needs to
# persist, and it deliberately does NOT belong on query_history: a
# query_history row is an immutable historical record of "this execution
# happened and produced this result" — it must stay true forever, even for
# a plan that a human later decides shouldn't be reused. So invalidation
# lives in its own tiny, purpose-built table instead of mutating that log.
#
# Two ways to invalidate (both write one row here — see
# PlanCacheRepository.invalidate_query / invalidate_scope):
#   - By query_history_id: "this ONE specific cached plan is no longer
#     good" (e.g. it was reused a few times but a user reported it's wrong).
#   - By scope (schema_hash [+ intent] [+ planner_version]): "nothing
#     currently cached under this key should be reused anymore" (e.g. a
#     planner version had a bug, or a schema's semantics changed in a way
#     schema_hash alone doesn't capture). Only affects rows that already
#     existed AT the time of invalidation (see is_invalidated() in
#     service.py) — a fresh, successful execution logged AFTER the
#     invalidation is fair game again, same as any normal cache repopulating
#     after a flush.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PlanCacheInvalidation(Base):
    __tablename__ = "plan_cache_invalidations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Set for a single-row invalidation; NULL for a scope-level one.
    # Deliberately a plain int, not a ForeignKey — plan_cache only ever
    # READS query_history, never owns or cascades against it, matching the
    # loose, read-only relationship this whole package has to that table.
    query_history_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    # Scope fields. For a single-row invalidation these are denormalized
    # from the target row purely for auditability (so a human inspecting
    # this table doesn't need to cross-reference query_history to see what
    # was invalidated) — matching logic never needs them in that case,
    # since the query_history_id already pins the exact row. For a
    # scope-level invalidation, schema_hash is required and intent /
    # planner_version are optional wildcards: NULL means "matches any".
    schema_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    planner_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    invalidated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        scope = f"query_history_id={self.query_history_id}" if self.query_history_id else f"scope={self.schema_hash}"
        return f"<PlanCacheInvalidation #{self.id} {scope}>"
