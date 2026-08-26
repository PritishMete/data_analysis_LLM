# memory_engine/contracts.py
# ─────────────────────────────────────────────────────────────────────────────
# THE swap points this package is designed around for future ML integration.
# Two of them:
#
#   - CandidateRanker: scores and orders already-gathered experiences (see
#     below). AnalyticsMemoryEngine (service.py) never scores or orders
#     candidates itself — it only asks a CandidateRanker to do so. Today's
#     default is DefaultCandidateRanker (rankers.py): deterministic text
#     similarity plus a feedback-score tiebreak, with zero ML/LLM involved.
#     Tomorrow it could be an embedding-based ranker, a learned re-ranking
#     model, or anything else — swapping it means writing ONE new class that
#     satisfies this contract and passing it into AnalyticsMemoryEngine's
#     constructor. Nothing in service.py, routes.py, sql_cache/, or any
#     external caller needs to change, because the method signatures below
#     are the entire surface any caller depends on.
#
#   - SimilarityStrategy: the narrower, single-pair-of-strings primitive a
#     CandidateRanker typically builds on ("how similar are these two
#     questions"). It lives here — not in sql_cache/ — because it's the
#     general-purpose primitive: memory_engine is THE reusable-experience
#     store for the whole backend, and sql_cache is one particular, narrower
#     consumer of that same idea (an ASGI middleware that short-circuits an
#     HTTP request using this same table). sql_cache/contracts.py and
#     sql_cache/strategies.py now just re-export from here, so any code (or
#     test) already importing `sql_cache.contracts.SimilarityStrategy` keeps
#     working unchanged.
#
# Deliberately narrow contracts: a ranker's (or strategy's) job is READING
# and ORDERING/SCORING already-stored experiences, nothing else.
#   - Must not write to the database (no training, no fine-tuning, no
#     persistence of any kind) — see the module docstring in service.py for
#     why "do not train models here" is a hard requirement of this package.
#   - A CandidateRanker must not fetch its own candidates — the engine
#     gathers candidates (via QueryHistoryRepository, already scoped by
#     org/dataset/schema_hash) and hands them to the ranker, so a ranker
#     implementation never needs to know about SQLAlchemy, the repository,
#     or how scoping works.
# ─────────────────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import dataclass

from query_history.models import QueryHistory


class SimilarityStrategy(ABC):
    """Scores how similar two natural-language queries are. Implementations
    must be deterministic and side-effect-free — same two strings in, same
    score out, every time, since a match/cache decision that isn't
    reproducible would be worse than no matching at all. This is also what
    keeps the contract ML-integration-ready without doing any training or
    inference INSIDE this package: an embedding-based strategy is free to
    call out to a (pre-trained, externally managed) model at score()-time —
    that's inference, not training — as long as scoring a given pair stays
    deterministic from the caller's point of view.
    """

    name: str = "unnamed_similarity_strategy"

    @abstractmethod
    def score(self, query_a: str, query_b: str) -> float:
        """Returns a similarity score in [0.0, 1.0]. 1.0 = effectively
        identical; 0.0 = unrelated."""
        raise NotImplementedError


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate experience, scored and ready to hand back to the
    engine. `score` is opaque to the engine — it only needs to be
    comparable (higher = better match) and to have already cleared
    whatever confidence bar the ranker was given; the engine does not
    interpret it beyond that, so a future ML ranker's score doesn't need to
    mean the same thing as today's [0.0, 1.0] similarity float."""

    score: float
    entry: QueryHistory


class CandidateRanker(ABC):
    """Scores and orders past experiences by relevance to a new question.

    Implementations must be deterministic-from-inputs and side-effect-free:
    same query + same candidates + same threshold in, same ranking out,
    every time. A ranker that reads a model checkpoint and runs inference
    still satisfies this (inference itself has no side effects on the
    stored data) — this rule is about not mutating history, not about
    forbidding ML.
    """

    name: str = "unnamed_candidate_ranker"

    @abstractmethod
    def rank(
        self,
        user_query: str,
        candidates: list[QueryHistory],
        *,
        min_confidence: float,
    ) -> list[RankedCandidate]:
        """Return only the candidates that clear `min_confidence`, ordered
        best match first. An empty list means nothing in `candidates` was a
        good enough match — that's a valid, expected result, not an error.
        """
        raise NotImplementedError
