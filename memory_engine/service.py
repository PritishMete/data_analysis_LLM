# memory_engine/service.py
# ─────────────────────────────────────────────────────────────────────────────
# THE ANALYTICS MEMORY ENGINE.
#
# Responsibilities (and nothing more):
#   1. STORE reusable experiences — reuses QueryHistoryRepository as-is. No
#      new tables, no changes to query_history/. Every successful execution
#      already logged there (question, schema_hash, SQL/pipeline, how the
#      result was shown, runtime, feedback) IS the "experience."
#   2. PROVIDE four read/write operations over those experiences:
#      find_similar_query() / find_best_sql() / find_best_pipeline() /
#      record_feedback() — see below. This class is the only surface
#      that's supposed to exist for that.
#
# HARD REQUIREMENTS:
#   - ZERO dependency on Gemini, or on any LLM/AI library at all. Check the
#     imports in this whole package: no google.adk/google.genai/openai/
#     anthropic import anywhere. Enforced structurally, not just by
#     docstring claim — see tests/test_memory_engine.py, which AST-scans
#     every module in this package for forbidden imports, not just this
#     file, so that guarantee survives future refactors here too.
#   - NO TRAINING HAPPENS HERE. This class and everything it calls only
#     READS already-stored data and RANKS it deterministically (today) or
#     via inference (in a future ranker) — it never fits, fine-tunes, or
#     persists a model. If a future ML integration needs a training step,
#     that step belongs in its own package, not this one.
#
# DESIGNED FOR FUTURE ML INTEGRATION:
#   Ranking is delegated to an injected `CandidateRanker` (see
#   contracts.py) rather than implemented inline in this class. Today's
#   default (DefaultCandidateRanker, rankers.py) is deterministic text
#   similarity + a feedback tiebreak — equally usable by a Gemini-based
#   planner, a Claude-based planner, or a hand-written rules engine, since
#   none of them see this class's internals, only its four public methods.
#   Swapping in an embedding/ML-based ranker later means writing one new
#   CandidateRanker and passing it to the constructor — this class,
#   routes.py, and every external caller stay untouched.
# ─────────────────────────────────────────────────────────────────────────────

from query_history.models import QueryHistory
from query_history.repository import QueryHistoryRepository

from .contracts import CandidateRanker, RankedCandidate, SimilarityStrategy
from .models import MemoryMatch
from .rankers import DefaultCandidateRanker


class AnalyticsMemoryEngine:
    """Modular/replaceable in the same way sql_cache is: the ranking
    strategy is injected (default DefaultCandidateRanker — deterministic,
    no ML), and the storage layer is the existing QueryHistoryRepository.
    Swap either without touching this class or any caller.
    """

    def __init__(
        self,
        history_repository: QueryHistoryRepository,
        similarity_strategy: SimilarityStrategy | None = None,
        default_min_confidence: float = 0.95,
        max_candidates: int = 200,
        ranker: CandidateRanker | None = None,
    ):
        self.history_repository = history_repository
        self.default_min_confidence = default_min_confidence
        self.max_candidates = max_candidates

        # `ranker` is the forward-looking, ML-ready construction path (see
        # contracts.py) — pass a custom CandidateRanker to change HOW
        # candidates get scored and ordered, wholesale.
        # `similarity_strategy` is kept as a direct constructor argument for
        # backward compatibility with existing callers/tests that only want
        # to swap the per-pair scoring function (e.g. ExactMatchStrategy)
        # while keeping the default feedback-tiebreak ranking behavior; it's
        # threaded into DefaultCandidateRanker when no explicit `ranker` is
        # given. Passing both is fine — `ranker` wins, since it's the more
        # specific choice.
        self.ranker = ranker or DefaultCandidateRanker(similarity_strategy)

    # ── Internal: gather candidates, delegate scoring/ordering to the ranker ─

    def _ranked_candidates(
        self,
        *,
        user_query: str,
        dataset_id: str | None,
        organization_id: str | None,
        schema_hash: str | None,
        min_confidence: float | None,
    ) -> list[RankedCandidate]:
        threshold = self.default_min_confidence if min_confidence is None else min_confidence

        candidates = self.history_repository.list_candidates(
            organization_id=organization_id,
            dataset_id=dataset_id,
            schema_hash=schema_hash,
            success=True,
            limit=self.max_candidates,
        )

        # Scoring/ordering is entirely the ranker's job — this class only
        # knows how to gather the candidate pool and hand it off. See
        # contracts.py for why that split is the whole point.
        return self.ranker.rank(user_query, candidates, min_confidence=threshold)

    @staticmethod
    def _to_match(ranked: RankedCandidate) -> MemoryMatch:
        entry = ranked.entry
        return MemoryMatch(
            query_history_id=entry.id,
            matched_query=entry.user_query,
            similarity_score=ranked.score,
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

    # ── Public API — exactly the four methods requested ─────────────────────

    def find_similar_query(
        self,
        *,
        user_query: str,
        dataset_id: str | None = None,
        organization_id: str | None = None,
        schema_hash: str | None = None,
        min_confidence: float | None = None,
    ) -> MemoryMatch | None:
        """Returns the single best-matching past experience (by the
        ranker's ordering), or None if nothing clears `min_confidence`
        (default: this engine's configured threshold).
        """
        ranked = self._ranked_candidates(
            user_query=user_query, dataset_id=dataset_id, organization_id=organization_id,
            schema_hash=schema_hash, min_confidence=min_confidence,
        )
        if not ranked:
            return None
        return self._to_match(ranked[0])

    def find_best_sql(
        self,
        *,
        user_query: str,
        dataset_id: str | None = None,
        organization_id: str | None = None,
        schema_hash: str | None = None,
        min_confidence: float | None = None,
    ) -> str | None:
        """Like find_similar_query, but specifically hunts for the best
        match that actually HAS a generated_sql value — a top-ranked overall
        match might be, say, a pivot/add_column experience with no SQL at
        all, in which case this skips it and returns the next-best one that
        does have SQL, rather than returning None just because the #1 match
        wasn't SQL-shaped.
        """
        ranked = self._ranked_candidates(
            user_query=user_query, dataset_id=dataset_id, organization_id=organization_id,
            schema_hash=schema_hash, min_confidence=min_confidence,
        )
        for candidate in ranked:
            if candidate.entry.generated_sql:
                return candidate.entry.generated_sql
        return None

    def find_best_pipeline(
        self,
        *,
        user_query: str,
        dataset_id: str | None = None,
        organization_id: str | None = None,
        schema_hash: str | None = None,
        min_confidence: float | None = None,
    ) -> dict | list | None:
        """Same idea as find_best_sql, for python_pipeline instead."""
        ranked = self._ranked_candidates(
            user_query=user_query, dataset_id=dataset_id, organization_id=organization_id,
            schema_hash=schema_hash, min_confidence=min_confidence,
        )
        for candidate in ranked:
            if candidate.entry.python_pipeline:
                return candidate.entry.python_pipeline
        return None

    def record_feedback(self, query_history_id: int, feedback_score: int) -> QueryHistory | None:
        """Records a human's judgment of one stored experience — this is
        the signal find_best_sql/find_best_pipeline use to break ties
        between otherwise-equally-similar matches. Returns the updated
        entry, or None if `query_history_id` doesn't exist."""
        return self.history_repository.set_feedback(query_history_id, feedback_score)
