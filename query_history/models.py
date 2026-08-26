# query_history/models.py
# ─────────────────────────────────────────────────────────────────────────────
# Every successful (or attempted) query execution gets one row here. This is
# the raw material the project's stated long-term goal depends on: "the
# backend should eventually become smarter than the LLM by learning from
# previous executions" — that learning has to be trained/derived from SOME
# table of real query -> plan -> outcome examples, and this is it. Nothing in
# this package does any learning itself yet; it just makes sure the data
# exists and is queryable, so a future deterministic ranking/caching layer
# (e.g. "we've seen this exact question against this exact schema_hash
# before — reuse the plan") — already built in plan_cache/ — and any FUTURE
# ML model both have real, structured examples to work from.
#
# Schema design notes for future ML training:
#   - `schema_hash` is a DENORMALIZED copy of the dataset's schema fingerprint
#     (see datasets/hashing.py) at logging time. It's redundant with a join
#     through `dataset_id` -> datasets.schema_hash, but training pipelines
#     will constantly want to group/filter examples by "same shape of
#     dataset" — baking it in directly avoids every future consumer needing
#     to know about, or join against, the Dataset Registry at all.
#   - `success` + `error_message` are BOTH kept, deliberately. A model (or a
#     human) learning what NOT to do needs failure examples with a reason
#     attached just as much as it needs successful ones — a bare boolean
#     throws that signal away.
#   - `rows_returned` matters for training a future "does this query even
#     make sense against this schema" signal (e.g. a plan that reliably
#     returns 0 rows across many runs is informative, not just noise).
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class QueryHistory(Base):
    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Nullable FK: a query might run before a dataset is registered (e.g. an
    # ad-hoc /smart_query call against an upload that predates this feature),
    # so this deliberately doesn't hard-require a dataset row to exist.
    dataset_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("datasets.dataset_id", ondelete="SET NULL"), nullable=True, index=True
    )
    organization_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # Denormalized from datasets.schema_hash at logging time (see module
    # docstring above) — nullable because it can only be populated when
    # dataset_id is both provided AND resolves to a registered dataset.
    schema_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    generated_sql: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Arbitrary structured pipeline description — e.g. cleaning_ops.py's
    # steps[] list, or query_router.py's plan dict. Stored as JSON rather
    # than a fixed schema since different execution paths (SQL plan vs
    # cleaning steps vs add_column config) shape this differently.
    python_pipeline: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)

    # Structured description of how the result was VISUALIZED (e.g.
    # {"chart_type": "bar", "x": "region", "y": "revenue"}) — not the
    # rendered image/SVG itself, just enough to reproduce the same chart
    # choice for a similar future query without asking an LLM to re-decide
    # "what kind of chart fits this data" from scratch every time.
    visualization: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    execution_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rows_returned: Mapped[int | None] = mapped_column(Integer, nullable=True)

    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -1 (bad) / 0 (neutral, unrated) / 1 (good) by default; a UI could use a
    # wider range (e.g. 1-5) — this column doesn't enforce one, intentionally.
    feedback_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Free-form tag identifying WHICH planner/version produced this row
    # (e.g. "gemini-1.5-flash", "rules-engine-v2", "sql-cache-reuse"). Not an
    # enum/FK on purpose — new planners will come and go, and training/
    # analytics consumers just need to group and filter by this string, not
    # validate it against a fixed set. Nullable + indexed for the same
    # reason `intent` is: most existing callers won't send it (backward
    # compatible), but once they do, "compare success rate / avg feedback
    # across planner_version" becomes a simple GROUP BY.
    planner_version: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<QueryHistory #{self.id} intent={self.intent} success={self.success}>"
