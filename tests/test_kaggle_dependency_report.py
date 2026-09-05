import json
from pathlib import Path

import pytest

from kaggle import bootstrap_environment
from kaggle.dependency_report import (
    DependencyReportError,
    dependency_report_allows_model_load,
    validate_dependency_report,
    write_dependency_report,
)


def report(status="STARTED", install_success=False, stack_verified=False):
    return {
        "schema_version": "1.0",
        "run_id": "run-1",
        "stage": "dependencies",
        "status": status,
        "install_success": install_success,
        "stack_verified": stack_verified,
    }


@pytest.mark.parametrize(
    ("payload", "allowed", "reason"),
    [
        (report(), False, "REPORT_NOT_FINALIZED"),
        (report("FAILED"), False, "INSTALL_FAILED"),
        (report("FAILED", True), False, "STACK_NOT_VERIFIED"),
        (report("SUCCESS", True, True), True, "SUCCESS"),
    ],
)
def test_dependency_gate_contract_states(payload, allowed, reason):
    result = dependency_report_allows_model_load(payload)
    assert result["allowed"] is allowed
    assert result["reason"] == reason


def test_impossible_success_state_is_rejected():
    with pytest.raises(DependencyReportError, match="SUCCESS requires true/true"):
        validate_dependency_report(report("SUCCESS", True, False))


def test_missing_field_is_schema_invalid():
    payload = report()
    del payload["stack_verified"]
    result = dependency_report_allows_model_load(payload)
    assert result == {"allowed": False, "reason": "REPORT_SCHEMA_INVALID", "detail": "REPORT_SCHEMA_INVALID: missing stack_verified"}


def test_report_is_validated_and_atomically_written(tmp_path):
    path = tmp_path / "dependency_install_result.json"
    payload = report("SUCCESS", True, True)
    write_dependency_report(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not path.with_name(path.name + ".tmp").exists()


def test_boolean_fields_must_be_booleans():
    payload = report()
    payload["install_success"] = "true"
    with pytest.raises(DependencyReportError, match="must be boolean"):
        validate_dependency_report(payload)


def test_unexpected_bootstrap_failure_publishes_terminal_failed_report(tmp_path, monkeypatch):
    monkeypatch.delenv("KAGGLE_RUN_DIR", raising=False)
    bootstrap_environment._finalize_unexpected_failure(
        ["--output-root", str(tmp_path), "--run-id", "run-failure"],
        RuntimeError("safe bootstrap failure"),
    )

    saved = json.loads((tmp_path / "smoke_runs" / "run-failure" / "dependency_install_result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "FAILED"
    assert saved["install_success"] is False
    assert saved["stack_verified"] is False
    assert saved["unexpected_exception"] is True


def test_bootstrap_public_entrypoint_is_exception_safe():
    source = Path(bootstrap_environment.__file__).read_text(encoding="utf-8")
    assert "def _run_bootstrap" in source
    assert "return _run_bootstrap(argv)" in source
    assert "_finalize_unexpected_failure(argv, exc)" in source


def test_bootstrap_json_probe_is_imported_before_runtime_use():
    source = Path(bootstrap_environment.__file__).read_text(encoding="utf-8")
    assert hasattr(bootstrap_environment, "_safe_run_json_probe")
    assert source.index("_safe_run_json_probe,") < source.index("def _probe_nf4_runtime")
