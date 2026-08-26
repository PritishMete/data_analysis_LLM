# memory_engine/strategies.py
# ─────────────────────────────────────────────────────────────────────────────
# Deterministic, Python-only similarity — no embeddings, no LLM calls, no
# training. Two signals blended together, because each catches a different
# kind of "similar but not identical" phrasing that the other misses:
#   - character-level (difflib.SequenceMatcher): near-identical phrasing,
#     typos, minor rewording — "total revenue by region" vs "total revenue
#     by region " (trailing space) or "totl revenue by region" (typo).
#   - token-level (Jaccard on word sets): reordered or restructured
#     phrasing that's character-different but means the same thing —
#     "total revenue by region" vs "revenue total, grouped by region".
# Neither alone is reliable enough to safely gate a 95%-confidence reuse
# decision; blended, they're a reasonable deterministic proxy for "this is
# basically the same question."
#
# This is the DEFAULT SimilarityStrategy for both memory_engine's
# DefaultCandidateRanker and sql_cache's SqlCacheService — one deterministic
# primitive, two consumers. It lives here (not in sql_cache/) because
# memory_engine is the general-purpose reusable-experience store; see the
# module docstring in contracts.py for the full reasoning.
# ─────────────────────────────────────────────────────────────────────────────

import difflib
import re

from .contracts import SimilarityStrategy


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)  # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    return set(_normalize(text).split())


class TextSimilarityStrategy(SimilarityStrategy):
    """The default strategy. `char_weight`/`token_weight` are constructor
    parameters (not buried constants), so the blend is tunable per instance
    without touching this class."""

    name = "text_similarity"

    def __init__(self, char_weight: float = 0.5, token_weight: float = 0.5):
        if not (0.0 <= char_weight <= 1.0 and 0.0 <= token_weight <= 1.0):
            raise ValueError("weights must each be within [0.0, 1.0]")
        if abs((char_weight + token_weight) - 1.0) > 1e-9:
            raise ValueError("char_weight + token_weight must sum to 1.0")
        self.char_weight = char_weight
        self.token_weight = token_weight

    def score(self, query_a: str, query_b: str) -> float:
        norm_a, norm_b = _normalize(query_a), _normalize(query_b)
        if not norm_a or not norm_b:
            return 0.0
        if norm_a == norm_b:
            return 1.0

        char_score = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

        tokens_a, tokens_b = _tokenize(query_a), _tokenize(query_b)
        if tokens_a and tokens_b:
            token_score = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        else:
            token_score = 0.0

        return round(self.char_weight * char_score + self.token_weight * token_score, 4)


class ExactMatchStrategy(SimilarityStrategy):
    """Trivial baseline: 1.0 for identical normalized text, 0.0 otherwise.
    Useful as a strict-mode swap-in, or as the simplest possible reference
    implementation of the SimilarityStrategy contract."""

    name = "exact_match"

    def score(self, query_a: str, query_b: str) -> float:
        return 1.0 if _normalize(query_a) == _normalize(query_b) else 0.0
