# sql_cache/strategies.py
# ─────────────────────────────────────────────────────────────────────────────
# TextSimilarityStrategy / ExactMatchStrategy's canonical home is now
# memory_engine/strategies.py — see sql_cache/contracts.py's module
# docstring for why. Re-exported here so nothing importing
# `sql_cache.strategies.TextSimilarityStrategy` (or ExactMatchStrategy)
# needs to change — same class objects, just defined elsewhere now.
# ─────────────────────────────────────────────────────────────────────────────

from memory_engine.strategies import ExactMatchStrategy, TextSimilarityStrategy  # noqa: F401
