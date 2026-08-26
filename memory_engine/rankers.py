# memory_engine/rankers.py
# ─────────────────────────────────────────────────────────────────────────────
# The DEFAULT CandidateRanker — deterministic, no ML, no LLM. Built on this
# package's own SimilarityStrategy contract and TextSimilarityStrategy
# implementation (contracts.py / strategies.py) rather than a second,
# duplicate copy of "how similar are two questions" logic; sql_cache reuses
# these same two classes (re-exported from sql_cache/contracts.py and
# sql_cache/strategies.py) as the other consumer of this exact,
# already-LLM-free primitive.
# ─────────────────────────────────────────────────────────────────────────────

from query_history.models import QueryHistory

from .contracts import CandidateRanker, RankedCandidate, SimilarityStrategy
from .strategies import TextSimilarityStrategy


class DefaultCandidateRanker(CandidateRanker):
    """Filters candidates by a per-pair SimilarityStrategy score, then
    breaks ties among equally-similar matches by feedback_score — among
    matches that are ALL good enough to reuse, prefer the one a human has
    actually endorsed. feedback_score defaults to 0 (neutral/unrated) when
    None, so an unrated-but-more-similar match still beats a less-similar
    rated one; feedback only breaks ties within the similarity band, it
    doesn't override similarity itself.

    This is a straight extraction of what AnalyticsMemoryEngine used to do
    inline — behavior is unchanged, only where it lives has moved, so it
    can be swapped out as a whole (see contracts.py) rather than requiring
    a future ML ranker to reimplement the tiebreak/sort logic too.
    """

    name = "default_text_similarity_ranker"

    def __init__(self, similarity_strategy: SimilarityStrategy | None = None):
        self.similarity_strategy = similarity_strategy or TextSimilarityStrategy()

    def rank(
        self,
        user_query: str,
        candidates: list[QueryHistory],
        *,
        min_confidence: float,
    ) -> list[RankedCandidate]:
        scored: list[RankedCandidate] = []
        for entry in candidates:
            score = self.similarity_strategy.score(user_query, entry.user_query)
            if score >= min_confidence:
                scored.append(RankedCandidate(score=score, entry=entry))

        scored.sort(key=lambda rc: (rc.score, rc.entry.feedback_score or 0), reverse=True)
        return scored
