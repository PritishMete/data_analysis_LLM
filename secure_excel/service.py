"""End-to-end secure Excel assistant service."""

from __future__ import annotations

from typing import Any

import pandas as pd

from common.excel_context import scan_workbook
from .config import CONFIG
from .executor import execute_structured_query
from .privacy_guard import PrivacyAuditRecord, assert_safe_remote_payload, log_privacy_audit
from .query_parser import parse_query
from .query_validator import validate_structured_query
from .semantic_roles import anonymized_schema_summary, build_schema_profile
from .session_store import SESSION_STORE


def load_excel_session(
    raw_bytes: bytes,
    filename: str,
    *,
    sheet_name: str | None = None,
    active_cell: str | None = None,
    dataset_range: str | None = None,
) -> dict[str, Any]:
    dataframe, workbook_context = scan_workbook(
        raw_bytes,
        filename=filename,
        sheet_name=sheet_name,
        active_cell=active_cell,
        requested_range=dataset_range,
    )
    schema = build_schema_profile(dataframe)
    session = SESSION_STORE.create(dataframe, schema, workbook_context)
    log_privacy_audit(
        PrivacyAuditRecord(
            request_type="excel_load",
            column_count=len(dataframe.columns),
            row_count=len(dataframe),
            anonymized_column_ids=[column["column_id"] for column in schema["columns"]],
            operation_type="load",
        )
    )
    return {
        "session_id": session.session_id,
        "context": workbook_context,
        "schema": anonymized_schema_summary(schema),
        "column_count": len(dataframe.columns),
        "row_count": len(dataframe),
    }


def interpret_query(session_id: str, text: str) -> dict[str, Any]:
    session = SESSION_STORE.get(session_id)
    parsed = parse_query(text, session.schema)
    validated = validate_structured_query(parsed, session.schema)
    assert_safe_remote_payload({
        "anonymized_schema": anonymized_schema_summary(session.schema),
        "query": validated,
    })
    return validated


def execute_query(session_id: str, text: str) -> dict[str, Any]:
    session = SESSION_STORE.get(session_id)
    parsed = parse_query(text, session.schema)
    validated = validate_structured_query(parsed, session.schema)
    result = execute_structured_query(session.dataframe, session.schema, validated)
    log_privacy_audit(
        PrivacyAuditRecord(
            request_type="excel_query",
            column_count=len(session.dataframe.columns),
            row_count=len(session.dataframe),
            anonymized_column_ids=[column["column_id"] for column in session.schema["columns"]],
            operation_type=validated["operation"],
        )
    )
    return {
        "session_id": session_id,
        "query": validated,
        "schema": anonymized_schema_summary(session.schema),
        **result,
    }


def list_supported_transforms() -> dict[str, Any]:
    return {
        "supported_operations": [
            "filter",
            "sort",
            "group",
            "aggregate",
            "search",
            "count",
            "report",
        ],
        "supported_operators": [
            "equals",
            "not_equals",
            "contains",
            "starts_with",
            "ends_with",
            "greater_than",
            "less_than",
            "greater_equal",
            "less_equal",
            "between",
            "is_null",
            "is_not_null",
        ],
        "remote_ai_enabled": CONFIG.remote_ai_enabled,
    }

