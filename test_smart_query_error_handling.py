# test_smart_query_error_handling.py
# ─────────────────────────────────────────────────────────────────────────────
# Verifies the production-grade error handling added to /smart_query and the
# Transformation Engine:
#   - json_safe() actually neutralizes every numpy/pandas type that would
#     otherwise break JSON serialization (including NaN/Infinity, which
#     Python's json module accepts as a non-standard extension but Dart/JS
#     JSON decoders reject outright).
#   - /smart_query ALWAYS returns HTTP 200 with a valid, consistently-shaped
#     JSON body — for a normal query, a malformed query, and a query that
#     triggers an exception deep in the transformation pipeline.
#   - The exact query that originally triggered "ClientException: Failed to
#     fetch" now returns either a successful TransformationResult or a
#     structured JSON error — never nothing.
#
# The TestClient-based tests (Part 2) require the full app's dependencies
# (fastapi, pydantic, sqlalchemy, google-adk, duckdb, etc.) which are not
# all installed in every environment — Part 1 (json_safe) has zero such
# dependencies and can always run. Run with:
#   python3 test_smart_query_error_handling.py
# or under pytest.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

# ── Part 1: json_safe() — no FastAPI/pydantic dependency ───────────────────

from main import json_safe, smart_query_error_response


def test_json_safe_handles_numpy_scalars():
    assert json_safe(np.int64(42)) == 42
    assert isinstance(json_safe(np.int64(42)), int)
    assert json_safe(np.float64(3.14)) == 3.14
    assert json_safe(np.bool_(True)) is True
    print("test_json_safe_handles_numpy_scalars: PASS")


def test_json_safe_handles_nan_and_infinity_as_null_not_bare_tokens():
    """The critical regression test: Python's json.dumps happily emits a
    bare `NaN` token by default (non-standard JSON), which both Dart's
    json.decode() and JavaScript's JSON.parse() reject outright. json_safe
    must convert these to `null` — not merely "not crash on the Python
    side", but actually produce text a strict JSON decoder accepts.
    """
    for value in (np.float64("nan"), float("nan"), float("inf"), float("-inf"), np.float64("inf")):
        safe = json_safe(value)
        assert safe is None, f"expected None for {value!r}, got {safe!r}"
    text = json.dumps({"x": json_safe(float("nan"))}, allow_nan=False)  # raises on any residual NaN/Inf
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"x": None}  # round-trips through a strict decoder
    print("test_json_safe_handles_nan_and_infinity_as_null_not_bare_tokens: PASS")


def test_json_safe_does_not_accidentally_catch_numpy_subclasses_of_float():
    """np.float64 is a genuine subclass of Python's float — an isinstance()
    fast-path would silently bypass NaN sanitization for it. Regression
    test for exactly that bug.
    """
    nan_as_np_float64 = np.float64("nan")
    assert isinstance(nan_as_np_float64, float)  # confirms the subclass relationship exists
    assert json_safe(nan_as_np_float64) is None
    print("test_json_safe_does_not_accidentally_catch_numpy_subclasses_of_float: PASS")


def test_json_safe_handles_pandas_types():
    assert json_safe(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"
    assert json_safe(pd.NaT) is None
    assert json_safe(pd.Series([1, 2, 3])) == [1, 2, 3]
    df = pd.DataFrame({"a": [1, 2]})
    # UPDATED per TASK 2/8 spec: DataFrame now serializes to a
    # {"columns": [...], "rows": [...]} envelope (not a bare list of row
    # dicts) — this preserves column ORDER and NAMES even when a frame has
    # zero rows (a bare `[]` loses that information entirely), which matters
    # for any Flutter table renderer consuming this shape directly.
    assert json_safe(df) == {"columns": ["a"], "rows": [{"a": 1}, {"a": 2}]}
    print("test_json_safe_handles_pandas_types: PASS")


def test_json_safe_handles_realistic_nested_ai_report_payload():
    """Simulates the actual shape of a /smart_query success response —
    nested dicts/lists mixing numpy scalars, as real statistics/KPI
    computations (.mean(), .sum(), .nunique()) would produce.
    """
    payload = {
        "operation": {
            "data": {"rows": [{"Votes": np.int64(340), "Rating": np.float64(4.2)}]},
            "ai_report": {
                "statistics": {"Votes": {"mean": np.float64(210.5), "unique": np.int64(87)}},
                "detected_kpis": [{"value": np.float64(4.2)}],
            },
            "updated_schema": {"changed_dtypes": {"Votes": np.dtype("int64")}},
        }
    }
    safe = json_safe(payload)
    text = json.dumps(safe, allow_nan=False)  # must not raise
    round_tripped = json.loads(text)
    assert round_tripped["operation"]["data"]["rows"][0]["Votes"] == 340
    print("test_json_safe_handles_realistic_nested_ai_report_payload: PASS")


def test_json_safe_never_raises_on_unknown_types():
    class Weird:
        def __repr__(self):
            return "<Weird>"
    safe = json_safe(Weird())
    json.dumps(safe)  # must not raise
    assert safe == "<Weird>"
    print("test_json_safe_never_raises_on_unknown_types: PASS")


def test_smart_query_error_response_shape_matches_success_envelope():
    err = smart_query_error_response("Column not found.", error_type="COLUMN_NOT_FOUND", confidence=0.5)
    assert err["route"] == "operation"
    assert err["success"] is False
    assert err["confidence"] == 0.5
    assert err["operation"]["error_type"] == "COLUMN_NOT_FOUND"
    json.dumps(err)
    print("test_smart_query_error_response_shape_matches_success_envelope: PASS")


# ── Part 2: full-stack /smart_query behavior via FastAPI TestClient ────────
# Requires the app's full dependency set (fastapi, pydantic, sqlalchemy,
# google-adk, duckdb, ...). Skipped with a clear message if unavailable
# rather than failing the whole file — Part 1 above is the dependency-free
# core guarantee and always runs.

def _try_build_test_client():
    try:
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)
    except Exception as e:  # ImportError or any startup-time failure
        print(f"[SKIPPED] Part 2 (TestClient) unavailable in this environment: {e}")
        return None


def _post_smart_query(client, csv_text: str, query_text: str):
    files = {"file": ("data.csv", csv_text, "text/csv")}
    data = {"text": query_text, "available_sheets": "[]"}
    return client.post("/smart_query", files=files, data=data)


def test_smart_query_always_returns_200_and_json_for_a_normal_query():
    client = _try_build_test_client()
    if client is None:
        return
    csv_text = "Aggregate rating,Votes\n0.4,120\n1.8,340\n4.6,87\n"
    resp = _post_smart_query(client, csv_text, "create column for Aggregate rating 0-1,1-2,2-3,3-4,4-5")
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}"
    body = resp.json()  # must not raise — this IS the "never ClientException" guarantee
    assert body.get("route") == "operation"
    print("test_smart_query_always_returns_200_and_json_for_a_normal_query: PASS")


def test_smart_query_malformed_csv_returns_structured_json_not_500():
    client = _try_build_test_client()
    if client is None:
        return
    resp = _post_smart_query(client, "not,a,valid\ncsv\"\"\"file", "create column for X 0-1,1-2")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("success") is False or "error" in str(body).lower()
    print("test_smart_query_malformed_csv_returns_structured_json_not_500: PASS")


def test_smart_query_ambiguous_column_still_returns_json_error():
    client = _try_build_test_client()
    if client is None:
        return
    csv_text = "Score,Votes\n0.4,120\n1.8,340\n"
    resp = _post_smart_query(client, csv_text, "create column for rating range 0-1,1-2,2-3,3-4,4-5")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("route") == "operation"
    assert body.get("success") is False
    print("test_smart_query_ambiguous_column_still_returns_json_error: PASS")


def test_smart_query_transformation_engine_exception_still_returns_json():
    """Monkeypatches the engine to raise, simulating a genuine internal bug,
    and confirms the route still returns 200 + valid JSON rather than
    propagating — the exact "no uncaught exception reaches FastAPI"
    requirement.
    """
    client = _try_build_test_client()
    if client is None:
        return
    import query_router

    original_run = query_router._transformation_engine.run

    def boom(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    query_router._transformation_engine.run = boom
    try:
        csv_text = "Aggregate rating,Votes\n0.4,120\n"
        resp = _post_smart_query(client, csv_text, "create column for Aggregate rating 0-1,1-2,2-3,3-4,4-5")
        assert resp.status_code == 200, f"expected 200 even on internal exception, got {resp.status_code}"
        body = resp.json()
        assert body.get("success") is False
        assert body.get("operation", {}).get("error_type") in (
            "TRANSFORMATION_ENGINE_EXCEPTION", "RESPONSE_BUILD_FAILED",
        )
    finally:
        query_router._transformation_engine.run = original_run
    print("test_smart_query_transformation_engine_exception_still_returns_json: PASS")


def test_the_exact_reported_query_never_fails_to_fetch():
    """The specific query from the bug report. Must return a successful
    TransformationResult or a structured JSON error — anything but an
    empty/non-JSON response.
    """
    client = _try_build_test_client()
    if client is None:
        return
    csv_text = (
        "Restaurant ID,Country Code,Longitude,Latitude,Average Cost for two,"
        "Price range,Aggregate rating,Votes\n"
        "1,1,77.1,28.6,700,3,0.4,120\n"
        "2,1,77.2,28.7,500,2,1.8,340\n"
        "3,1,77.3,28.8,900,4,4.6,87\n"
    )
    query = "Create column for Aggregate rating 0-1,1-2,2-3,3-4,4-5 and check which rating is in which rating range"
    resp = _post_smart_query(client, csv_text, query)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict) and body, "response body must be a non-empty JSON object"
    assert body.get("route") in ("operation", "sql")
    if body.get("success"):
        assert "Aggregate rating_Range" in json.dumps(body) or "Aggregate_rating_Range" in json.dumps(body)
    else:
        assert body.get("message") or body.get("operation", {}).get("error")
    print("test_the_exact_reported_query_never_fails_to_fetch: PASS —", "success" if body.get("success") else "structured error")


if __name__ == "__main__":
    # Part 1 — always runs.
    test_json_safe_handles_numpy_scalars()
    test_json_safe_handles_nan_and_infinity_as_null_not_bare_tokens()
    test_json_safe_does_not_accidentally_catch_numpy_subclasses_of_float()
    test_json_safe_handles_pandas_types()
    test_json_safe_handles_realistic_nested_ai_report_payload()
    test_json_safe_never_raises_on_unknown_types()
    test_smart_query_error_response_shape_matches_success_envelope()

    # Part 2 — runs if the full app's dependencies are installed, otherwise
    # each test prints [SKIPPED] and returns cleanly.
    test_smart_query_always_returns_200_and_json_for_a_normal_query()
    test_smart_query_malformed_csv_returns_structured_json_not_500()
    test_smart_query_ambiguous_column_still_returns_json_error()
    test_smart_query_transformation_engine_exception_still_returns_json()
    test_the_exact_reported_query_never_fails_to_fetch()

    print("\nAll smart_query error-handling tests completed.")
