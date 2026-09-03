"""Canonical dependency bootstrap report contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
STATUSES = frozenset({"STARTED", "FAILED", "SUCCESS"})
REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "stage",
    "status",
    "install_success",
    "stack_verified",
)


class DependencyReportError(ValueError):
    """Raised when a dependency report violates the canonical contract."""


def validate_dependency_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise DependencyReportError("REPORT_SCHEMA_INVALID: report must be an object")
    missing = [field for field in REQUIRED_FIELDS if field not in report]
    if missing:
        raise DependencyReportError("REPORT_SCHEMA_INVALID: missing " + ",".join(missing))
    if report["schema_version"] != SCHEMA_VERSION:
        raise DependencyReportError("REPORT_SCHEMA_INVALID: schema_version")
    if not isinstance(report["run_id"], str) or not report["run_id"]:
        raise DependencyReportError("REPORT_SCHEMA_INVALID: run_id")
    if report["status"] not in STATUSES:
        raise DependencyReportError("REPORT_SCHEMA_INVALID: status")
    for field in ("install_success", "stack_verified"):
        if not isinstance(report[field], bool):
            raise DependencyReportError(f"REPORT_SCHEMA_INVALID: {field} must be boolean")
    status = report["status"]
    installed = report["install_success"]
    verified = report["stack_verified"]
    if status == "STARTED" and (installed or verified):
        raise DependencyReportError("REPORT_SCHEMA_INVALID: STARTED must be false/false")
    if status == "FAILED" and verified:
        raise DependencyReportError("REPORT_SCHEMA_INVALID: FAILED cannot be stack_verified")
    if status == "SUCCESS" and (not installed or not verified):
        raise DependencyReportError("REPORT_SCHEMA_INVALID: SUCCESS requires true/true")
    if verified and not installed:
        raise DependencyReportError("REPORT_SCHEMA_INVALID: stack_verified requires install_success")
    return dict(report)


def dependency_report_allows_model_load(report: Any) -> dict[str, Any]:
    try:
        normalized = validate_dependency_report(report)
    except DependencyReportError as exc:
        message = str(exc)
        return {"allowed": False, "reason": message.split(":", 1)[0], "detail": message}
    if normalized["status"] == "FAILED":
        if not normalized["install_success"]:
            return {"allowed": False, "reason": "INSTALL_FAILED"}
        return {"allowed": False, "reason": "STACK_NOT_VERIFIED"}
    if normalized["status"] != "SUCCESS":
        return {"allowed": False, "reason": "REPORT_NOT_FINALIZED"}
    if not normalized["install_success"]:
        return {"allowed": False, "reason": "INSTALL_FAILED"}
    if not normalized["stack_verified"]:
        return {"allowed": False, "reason": "STACK_NOT_VERIFIED"}
    return {"allowed": True, "reason": "SUCCESS"}


def write_dependency_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_dependency_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(temporary, path)
    return normalized
