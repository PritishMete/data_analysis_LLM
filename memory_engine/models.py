# memory_engine/models.py
# ─────────────────────────────────────────────────────────────────────────────
# Not a database model (that's query_history/models.py — this package adds
# no new tables; see the "Store reusable experiences" note in
# service.py's module docstring). This is the OUTPUT shape the Memory
# Engine hands back to callers: every dimension of one reusable experience,
# ranked and ready to use, regardless of which CandidateRanker produced the
# ranking. Lives in its own module (rather than inside service.py) so
# routes.py can import it without importing the engine implementation.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class MemoryMatch:
    """One reusable experience, ranked and ready to hand back to whatever
    planner asked for it. Carries EVERY stored dimension, not just the SQL —
    a caller can use as much or as little of this as it wants."""

    query_history_id: int
    matched_query: str
    similarity_score: float
    intent: str | None
    schema_hash: str | None
    generated_sql: str | None
    python_pipeline: dict | list | None
    visualization: dict | None
    execution_time_ms: float | None
    rows_returned: int | None
    feedback_score: int | None
    planner_version: str | None
