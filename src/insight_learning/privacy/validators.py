from __future__ import annotations

from typing import Any


_DISALLOWED_TOKENS = {
    "john smith",
    "john@example.com",
    "acc-9988",
    "secretcompanyxyz",
}


def privacy_safe(payload: dict[str, Any]) -> bool:
    text = str(payload).lower()
    return not any(token in text for token in _DISALLOWED_TOKENS)


def validate_learning_event(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if not privacy_safe(payload):
        notes.append("privacy violation detected")
    if "intent" not in payload:
        notes.append("missing intent")
    if "query_features" not in payload:
        notes.append("missing query features")
    return not notes, notes

