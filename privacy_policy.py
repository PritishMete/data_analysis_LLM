"""Server-side privacy boundary for InsightFlow.

The hosted backend can still provide normal API functionality when explicitly
configured for remote processing, but the safe default is local-only. In that
mode, routes that accept workbook bytes or raw row values are rejected before
any file is read or AI code is invoked.
"""
from __future__ import annotations

import os
from fastapi import HTTPException, Request

PRIVACY_MODE = os.getenv("INSIGHTFLOW_PRIVACY_MODE", "local_only").strip().lower()
LOCAL_ONLY = PRIVACY_MODE not in {"remote_allowed", "remote", "off", "disabled"}

# Any endpoint in this set can receive workbook bytes or raw row values.
BLOCKED_LOCAL_ONLY_PATHS = {
    "/analyze",
    "/analyze-report",
    "/clean_data",
    "/transform/range_binning",
    "/transform/preview",
    "/transform/apply",
    "/agentic_categorize",
    "/location/enrich",
    "/sentiment_analysis",
    "/smart_query",
    "/v2/excel/scan",
    "/v2/excel/context",
    "/excel/session",
    "/v2/ingest/dataset",
    "/api/clean/dynamic_backtrack",
}


def reject_if_local_only(path: str, content_type: str = "") -> None:
    if LOCAL_ONLY and (path in BLOCKED_LOCAL_ONLY_PATHS or content_type.lower().startswith("multipart/form-data")):
        raise HTTPException(
            status_code=403,
            detail=(
                "Dataset processing is disabled on the hosted API in local-only privacy mode. "
                "Process the workbook in the InsightFlow client, or explicitly set "
                "INSIGHTFLOW_PRIVACY_MODE=remote_allowed if remote processing is intentionally required."
            ),
        )


def privacy_status() -> dict[str, object]:
    return {
        "mode": "local_only" if LOCAL_ONLY else "remote_allowed",
        "dataset_uploads_enabled": not LOCAL_ONLY,
    }
