# sql_cache/contracts.py
# ─────────────────────────────────────────────────────────────────────────────
# SimilarityStrategy's canonical home is now memory_engine/contracts.py —
# memory_engine is the general-purpose reusable-experience store for the
# whole backend, and sql_cache is just one narrower consumer of the same
# deterministic "how similar are these two questions" primitive (an ASGI
# middleware that short-circuits an HTTP request using the same
# query_history table memory_engine reads from). See the module docstring
# there for the full reasoning.
#
# This module is kept as a re-export so nothing importing
# `sql_cache.contracts.SimilarityStrategy` (existing code, existing tests)
# needs to change — it's the exact same class object, just defined
# elsewhere now.
# ─────────────────────────────────────────────────────────────────────────────

from memory_engine.contracts import SimilarityStrategy  # noqa: F401
