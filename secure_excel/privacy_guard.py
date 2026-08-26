"""Privacy guard for any outbound remote payloads.

The secure Excel path is local-only by default, but if a remote model is ever
enabled, this module rejects payloads that contain obvious PII or raw workbook
content. Logging stays metadata-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
URL_RE = re.compile(r"\bhttps?://\S+\b", re.I)


@dataclass(frozen=True)
class PrivacyAuditRecord:
    request_type: str
    column_count: int
    row_count: int
    anonymized_column_ids: list[str]
    operation_type: str


def _flatten(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def contains_sensitive_content(payload: Any) -> bool:
    try:
        serialized = json.dumps(payload, default=str)
    except Exception:
        serialized = str(payload)
    if EMAIL_RE.search(serialized) or PHONE_RE.search(serialized) or URL_RE.search(serialized):
        return True
    lowered = serialized.lower()
    forbidden_tokens = (
        "customer name",
        "restaurant name",
        "sheet name",
        "filename",
        "file_name",
        "address",
        "email",
        "phone",
        "url",
    )
    return any(token in lowered for token in forbidden_tokens)


def assert_safe_remote_payload(payload: Any) -> None:
    if contains_sensitive_content(payload):
        raise ValueError("Remote payload failed privacy guard checks.")


def log_privacy_audit(record: PrivacyAuditRecord) -> None:
    logger.info(
        "privacy_audit request_type=%s column_count=%s row_count=%s anonymized_column_ids=%s operation_type=%s",
        record.request_type,
        record.column_count,
        record.row_count,
        ",".join(record.anonymized_column_ids),
        record.operation_type,
    )

