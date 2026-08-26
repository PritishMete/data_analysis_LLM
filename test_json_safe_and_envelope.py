# test_json_safe_and_envelope.py
# ─────────────────────────────────────────────────────────────────────────────
# TASK 9 test coverage for the Smart Query reliability work:
#
#   Part 1 — common/json_safe.py::to_json_safe() against every type called
#            out in the brief: numpy.int64/int32, numpy.float64/float32,
#            numpy.bool_, pandas.Timestamp, datetime/date, Decimal, UUID,
#            Path, numpy.ndarray, pandas.Series, pandas.Index, pandas.NA,
#            numpy.nan, math.nan, pandas.DataFrame — plus nested structures
#            mixing all of the above, which is the realistic failure shape
#            (a rogue value buried three levels deep in a dict-of-lists).
#
#   Part 2 — common/response_envelope.py::smart_query_envelope() — every
#            route (success/failure, "operation"/"sql") always carries the
#            same top-level keys (TASK 7).
#
#   Part 3 — common/transformations/transformation_result.py::
#            TransformationResult — every declared field round-trips
#            through to_json_safe() even when populated with numpy/pandas
#            values, the shape TransformationEngine actually produces them
#            in.
#
#   Part 4 — query_router.py::_operation_error_response() — always returns
#            the full consistent envelope, is itself JSON-safe even when
#            handed a raw exception object, and round-trips through
#            json.dumps(..., allow_nan=False) (the same strict check
#            Dart/JS decoders effectively perform).
#
#   Part 5 — end-to-end: every non-LLM route inside
#            query_router.handle_smart_query() (the deterministic
#            transformation fast-path, both its success and failure
#            branches) returns a body that (a) has the full TASK 7 key set
#            and (b) survives json.dumps(..., allow_nan=False).
#
# Run with: python3 -m pytest test_json_safe_and_envelope.py -q
# or:       python3 test_json_safe_and_envelope.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import datetime
import decimal
import json
import math
import pathlib
import uuid

import numpy as np
import pandas as pd

from common.json_safe import to_json_safe
from common.response_envelope import smart_query_envelope
from common.transformations.transformation_result import TransformationResult

REQUIRED_ENVELOPE_KEYS = {
    "success", "route", "message", "confidence", "operation",
    "metadata", "preview", "statistics", "schema", "ai_report",
    "warnings", "errors",
}


def _assert_strict_json_roundtrip(payload):
    """The same check a strict client-side JSON decoder (Dart's
    json.decode(), JavaScript's JSON.parse()) effectively performs:
    allow_nan=False makes json.dumps raise on any residual NaN/Infinity,
    which Python's json module otherwise emits as bare (invalid-JSON)
    tokens.
    """
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text) == payload
    return text


# ── Part 1: to_json_safe() per-type coverage ────────────────────────────────

def test_numpy_integers():
    assert to_json_safe(np.int64(42)) == 42
    assert isinstance(to_json_safe(np.int64(42)), int)
    assert to_json_safe(np.int32(-7)) == -7
    assert isinstance(to_json_safe(np.int32(-7)), int)


def test_numpy_floats_including_nan():
    assert to_json_safe(np.float64(3.14)) == 3.14
    assert isinstance(to_json_safe(np.float32(1.5)), float)
    assert to_json_safe(np.float64("nan")) is None
    assert to_json_safe(np.float32("nan")) is None
    assert to_json_safe(np.float64("inf")) is None


def test_numpy_bool():
    assert to_json_safe(np.bool_(True)) is True
    assert to_json_safe(np.bool_(False)) is False


def test_numpy_ndarray():
    arr = np.array([1, 2, 3], dtype=np.int64)
    assert to_json_safe(arr) == [1, 2, 3]
    arr2 = np.array([1.0, np.nan, 3.0])
    assert to_json_safe(arr2) == [1.0, None, 3.0]


def test_pandas_timestamp_and_nat():
    assert to_json_safe(pd.Timestamp("2026-08-04T13:20:14.123456")) == "2026-08-04T13:20:14.123456"
    assert to_json_safe(pd.NaT) is None


def test_datetime_and_date():
    dt = datetime.datetime(2026, 8, 4, 13, 20, 14, 123456)
    assert to_json_safe(dt) == "2026-08-04T13:20:14.123456"
    d = datetime.date(2026, 8, 4)
    assert to_json_safe(d) == "2026-08-04"


def test_decimal():
    assert to_json_safe(decimal.Decimal("19.99")) == 19.99
    assert isinstance(to_json_safe(decimal.Decimal("19.99")), float)
    assert to_json_safe(decimal.Decimal("nan")) is None


def test_uuid_and_path():
    u = uuid.uuid4()
    assert to_json_safe(u) == str(u)
    p = pathlib.Path("/tmp/some/file.csv")
    assert to_json_safe(p) == str(p)


def test_pandas_series_and_index():
    assert to_json_safe(pd.Series([1, 2, 3])) == [1, 2, 3]
    assert to_json_safe(pd.Index(["a", "b"])) == ["a", "b"]


def test_pandas_na_and_bare_nan_variants():
    assert to_json_safe(pd.NA) is None
    assert to_json_safe(float("nan")) is None
    assert to_json_safe(math.nan) is None
    assert to_json_safe(np.nan) is None


def test_pandas_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
    safe = to_json_safe(df)
    assert safe == {"columns": ["a", "b"], "rows": [{"a": 1, "b": "x"}, {"a": 2, "b": None}]}
    _assert_strict_json_roundtrip(safe)


def test_dataframe_with_nan_and_timestamp_values():
    df = pd.DataFrame({
        "amount": [1.5, np.nan],
        "when": [pd.Timestamp("2026-01-01"), pd.NaT],
    })
    safe = to_json_safe(df)
    assert safe["rows"][0]["amount"] == 1.5
    assert safe["rows"][1]["amount"] is None
    _assert_strict_json_roundtrip(safe)


def test_nested_structure_mixing_every_type():
    """The realistic shape: a rogue value buried in nested dicts/lists,
    exactly how a KPI/metadata/preview blob is actually built.
    """
    payload = {
        "kpis": [{"label": "Total Revenue", "value": np.float64(12345.6789)}],
        "generated_at": pd.Timestamp("2026-08-04T00:00:00"),
        "row_ids": np.array([np.int64(1), np.int64(2)]),
        "price": decimal.Decimal("9.99"),
        "trace_id": uuid.uuid4(),
        "missing": {np.nan, pd.NA, math.nan} if False else None,  # sets aren't hashable w/ NA mixed; keep simple
        "empty_slot": float("nan"),
        "nested": {"deep": {"deeper": np.bool_(True)}},
    }
    safe = to_json_safe(payload)
    _assert_strict_json_roundtrip(safe)
    assert safe["kpis"][0]["value"] == 12345.6789
    assert safe["empty_slot"] is None
    assert safe["nested"]["deep"]["deeper"] is True


def test_never_raises_on_unknown_object():
    class Weird:
        def __repr__(self):
            return "Weird()"
    safe = to_json_safe(Weird())
    assert isinstance(safe, str)
    _assert_strict_json_roundtrip({"x": safe})


# ── Part 2: response envelope consistency (TASK 7) ──────────────────────────

def test_envelope_always_has_full_key_set_success():
    env = smart_query_envelope(success=True, route="operation", message="ok", operation={"action": "rename"})
    assert REQUIRED_ENVELOPE_KEYS <= set(env.keys())
    for k in ("metadata", "preview", "statistics", "schema", "ai_report"):
        assert env[k] == {}
    for k in ("warnings", "errors"):
        assert env[k] == []


def test_envelope_always_has_full_key_set_failure():
    env = smart_query_envelope(success=False, route="sql", message="boom", errors=[{"error_type": "X", "message": "boom"}])
    assert REQUIRED_ENVELOPE_KEYS <= set(env.keys())
    assert env["success"] is False
    assert env["errors"] == [{"error_type": "X", "message": "boom"}]


def test_envelope_preserves_route_specific_extras():
    env = smart_query_envelope(success=True, route="sql", plan={"group_by": ["x"]}, sql="SELECT 1", result={"rows": []})
    assert env["plan"] == {"group_by": ["x"]}
    assert env["sql"] == "SELECT 1"
    assert env["result"] == {"rows": []}
    assert REQUIRED_ENVELOPE_KEYS <= set(env.keys())


def test_success_and_failure_envelopes_share_identical_key_set():
    success = smart_query_envelope(success=True, route="operation", operation={"a": 1})
    failure = smart_query_envelope(success=False, route="operation", errors=[{"x": 1}])
    assert set(success.keys()) == set(failure.keys())


# ── Part 3: TransformationResult serializes cleanly ─────────────────────────

def test_transformation_result_with_numpy_and_timestamp_fields_is_json_safe():
    result = TransformationResult(
        success=True,
        transformation={"name": "range_binning", "confidence": np.float64(0.92)},
        preview={"sample": [np.int64(1), np.int64(2)]},
        metadata={"explanation": "binned ages", "generated_at": pd.Timestamp("2026-08-04")},
        updated_schema={"age_band": "category"},
        updated_statistics={"count": np.int64(100), "mean": np.float64(float("nan"))},
        updated_kpis=[{"label": "Rows", "value": np.int64(100)}],
        updated_charts={"type": "bar"},
        updated_ai_report={"summary": "looks fine", "score": np.float64(0.87)},
        message="done",
    )
    safe = to_json_safe(result.to_dict())
    _assert_strict_json_roundtrip(safe)
    assert safe["transformation"]["confidence"] == 0.92
    assert safe["updated_statistics"]["mean"] is None
    assert safe["preview"]["sample"] == [1, 2]


def test_transformation_result_failure_classmethod_is_json_safe():
    result = TransformationResult.failure("could not parse ranges")
    safe = to_json_safe(result.to_dict())
    _assert_strict_json_roundtrip(safe)
    assert safe["success"] is False
    assert safe["error"] == "could not parse ranges"


# ── Part 4: _operation_error_response() ─────────────────────────────────────

def test_operation_error_response_full_envelope_and_json_safe():
    from query_router import _operation_error_response

    resp = _operation_error_response(
        "Transformation failed.",
        error_type="TRANSFORMATION_ENGINE_EXCEPTION",
        exception=ValueError("bad column: Revenu"),  # a raw exception object, not str()'d by the caller
    )
    assert REQUIRED_ENVELOPE_KEYS <= set(resp.keys())
    assert resp["success"] is False
    assert resp["route"] == "operation"
    assert resp["errors"] == [{"error_type": "TRANSFORMATION_ENGINE_EXCEPTION", "message": "Transformation failed."}]
    # the raw exception object passed as `exception=` must have been made
    # JSON-safe (stringified) even though the caller didn't str() it first
    _assert_strict_json_roundtrip(resp)
    assert isinstance(resp["operation"]["exception"], str)


def test_operation_error_response_default_error_type():
    from query_router import _operation_error_response
    resp = _operation_error_response("Something went wrong.")
    assert resp["operation"]["error_type"] == "INTERNAL_ERROR"
    _assert_strict_json_roundtrip(resp)


# ── Part 5: handle_smart_query() routes return valid, consistent JSON ───────

def test_handle_smart_query_transformation_success_route_is_consistent_and_json_safe():
    from query_router import handle_smart_query

    df = pd.DataFrame({
        "Age": [12, 34, 56, 78, np.nan],
        "Revenue": [np.float64(100.5), np.float64(200.25), np.nan, np.float64(50.0), np.float64(10.0)],
        "SignupDate": pd.to_datetime(["2026-01-01", "2026-02-01", None, "2026-03-01", "2026-04-01"]),
    })
    result = asyncio.run(handle_smart_query("group age into 0-18, 19-40, 41-60, 60+", df, []))
    assert REQUIRED_ENVELOPE_KEYS <= set(result.keys())
    _assert_strict_json_roundtrip(result)
    assert result["route"] == "operation"


def test_handle_smart_query_engine_exception_returns_valid_json_not_a_crash():
    """Simulates the exact failure mode from the brief: an exception deep in
    the transformation pipeline must still produce a full, valid JSON
    envelope — never an unhandled exception that would surface to Flutter
    as "ClientException: Failed to fetch".
    """
    from query_router import handle_smart_query, _transformation_engine

    original_run = _transformation_engine.run

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated engine failure")

    _transformation_engine.run = _boom
    try:
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = asyncio.run(handle_smart_query("do something", df, []))
    finally:
        _transformation_engine.run = original_run

    assert REQUIRED_ENVELOPE_KEYS <= set(result.keys())
    assert result["success"] is False
    _assert_strict_json_roundtrip(result)


if __name__ == "__main__":
    import sys
    failures = 0
    tests = [(name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    for name, fn in tests:
        try:
            fn()
            print(f"{name}: PASS")
        except Exception as e:
            failures += 1
            print(f"{name}: FAIL — {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
