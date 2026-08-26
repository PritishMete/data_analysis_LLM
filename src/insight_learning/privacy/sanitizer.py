from __future__ import annotations

from copy import deepcopy
from typing import Any

from learning.models import stable_hash


_BLOCKED_KEYS = {
    "query_text",
    "normalized_query",
    "rows",
    "row_values",
    "sheet_name",
    "filename",
    "file_name",
    "column_names",
    "original_columns",
    "original_values",
}


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items() if key not in _BLOCKED_KEYS}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def sanitize_learning_event(payload: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(payload)
    for key in list(clean):
        if key in _BLOCKED_KEYS:
            clean.pop(key, None)
    clean = _sanitize_value(clean)
    clean["event_signature"] = stable_hash(clean)
    return clean

