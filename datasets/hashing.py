# datasets/hashing.py
# ─────────────────────────────────────────────────────────────────────────────
# Pure, deterministic (no AI, no randomness) hashing helpers used to identify
# datasets and detect re-uploads of the same file or same schema shape.
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import json
from collections.abc import Iterable


def compute_file_hash(raw_bytes: bytes) -> str:
    """SHA-256 of the raw uploaded bytes — identifies byte-for-byte identical
    re-uploads (e.g. someone re-uploading the same export twice)."""
    return hashlib.sha256(raw_bytes).hexdigest()


def compute_schema_hash(columns: Iterable[tuple[str, str]]) -> str:
    """SHA-256 of a normalized (name, dtype) column list — identifies datasets
    that share the same SHAPE even if the underlying values differ (e.g. two
    monthly exports of the same report). Column order doesn't matter and
    naming is case/whitespace-normalized, so trivial variations don't produce
    a different hash.
    """
    normalized = sorted((str(name).strip().lower(), str(dtype)) for name, dtype in columns)
    payload = json.dumps(normalized, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
