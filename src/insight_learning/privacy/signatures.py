from __future__ import annotations

from typing import Any

from learning.models import stable_hash


def fingerprint_learning_event(payload: dict[str, Any]) -> str:
    return stable_hash(payload)

