# common/response_envelope.py
# ─────────────────────────────────────────────────────────────────────────────
# ONE function that builds the /smart_query response envelope, used by every
# return path in query_router.py's handle_smart_query() AND by main.py's
# top-level error handlers around it (data-load failures, unhandled
# exceptions). Previously each of those ~8 return sites hand-built its own
# dict, so successes and failures — and even different failure branches —
# didn't share the same set of keys (e.g. some branches had "statistics"/
# "schema"/"ai_report", most didn't; the SQL-route branches never had
# "operation" at all). A Flutter client that unconditionally reads
# `response["statistics"]` (say) would work for some responses and throw for
# others, which is indistinguishable, from the app's perspective, from the
# raw-fetch failures this whole effort is about eliminating.
#
# Every key listed in TASK 7 of the brief (success, route, operation,
# metadata, preview, statistics, schema, ai_report, warnings, errors) is
# ALWAYS present with a type-correct default ({} or []), on every single
# /smart_query response, success or failure, "operation" route or "sql"
# route. Route-specific extras (plan/sql/result for the "sql" route) are
# still passed through via **extra so no existing Flutter parsing code that
# already reads those breaks.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any


def smart_query_envelope(
    *,
    success: bool,
    route: str = "operation",
    message: str = "",
    confidence: float = 0.0,
    operation: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    preview: dict[str, Any] | None = None,
    statistics: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
    ai_report: dict[str, Any] | None = None,
    warnings: list[Any] | None = None,
    errors: list[Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Builds the single consistent /smart_query response shape.

    Always includes: success, route, message, confidence, operation,
    metadata, preview, statistics, schema, ai_report, warnings, errors —
    with safe defaults ({} / []) for anything the caller doesn't supply, so
    no route ever omits a key another route includes. Any extra keyword
    arguments (e.g. plan / sql / result on the "sql" route) are merged in on
    top, preserving each route's existing route-specific fields.
    """
    envelope: dict[str, Any] = {
        "success": success,
        "route": route,
        "message": message or "",
        "confidence": confidence,
        "operation": operation if operation is not None else {},
        "metadata": metadata if metadata is not None else {},
        "preview": preview if preview is not None else {},
        "statistics": statistics if statistics is not None else {},
        "schema": schema if schema is not None else {},
        "ai_report": ai_report if ai_report is not None else {},
        "warnings": warnings if warnings is not None else [],
        "errors": errors if errors is not None else [],
    }
    envelope.update(extra)
    return envelope
