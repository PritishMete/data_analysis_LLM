# common/json_safe.py
# ─────────────────────────────────────────────────────────────────────────────
# Single, shared recursive JSON-safety serializer for the whole backend.
#
# WHY THIS EXISTS (root cause of the Flutter "ClientException: Failed to
# fetch" reports): a route handler can build a "successful" Python response
# dict, return it, and STILL fail — because Starlette's JSON encoding of
# that dict happens AFTER the handler returns, outside any try/except inside
# the handler. A single numpy.int64 (from a `.sum()`), a pandas.Timestamp
# (from a date column), a Decimal, a NaN, or any other non-JSON-native value
# buried anywhere in a nested dict/list is enough to blow up serialization
# for the *entire* response — Flutter never receives a body at all, and it
# looks like a network failure ("Failed to fetch") even though the server
# logic executed correctly.
#
# Previously this logic was duplicated (main.py had its own `json_safe`,
# query_router.py had none at all — its `_operation_error_response()` and
# several `handle_smart_query()` branches returned raw dicts that were only
# made safe if/when main.py happened to wrap them). This module is now the
# ONE place that knows how to flatten every value type this backend ever
# produces into something `json.dumps` can always handle. Both main.py and
# query_router.py import `to_json_safe` from here instead of maintaining
# their own copies, so a fix here fixes every route at once.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import datetime
import decimal
import math
import pathlib
import uuid
from typing import Any

import numpy as np
import pandas as pd


def to_json_safe(obj: Any) -> Any:
    """Recursively converts a value tree into something the standard `json`
    module (and therefore FastAPI/Starlette's JSONResponse) can ALWAYS
    serialize into valid JSON.

    Handles, at minimum:
      numpy.int64 / int32 / any numpy.integer
      numpy.float64 / float32 / any numpy.floating (NaN -> null)
      numpy.bool_
      numpy.ndarray
      pandas.Timestamp, pandas.NaT, datetime.datetime, datetime.date
      pandas.Series, pandas.Index
      pandas.DataFrame -> {"columns": [...], "rows": [...]}
      pandas.NA
      decimal.Decimal -> float
      uuid.UUID -> str
      pathlib.Path -> str
      math.nan / float('nan') / float('inf') -> null
      dict / list / tuple / set (recursively)

    Never raises. Anything not specifically recognized, and that plain
    `json.dumps` would otherwise reject, falls back to `str(obj)` — this
    function's entire purpose is to GUARANTEE serialization succeeds, even
    if that occasionally means a slightly lossy string representation for
    some exotic value not explicitly handled below.
    """
    if obj is None:
        return None

    # NOTE: exact-type check for the common Python scalars, not isinstance().
    # np.float64 is a genuine subclass of Python's float and np.bool_ can
    # behave like bool in comparisons — an isinstance() fast-path here would
    # let a NaN-valued np.float64 slip through as a bare (invalid-JSON) NaN
    # token. numpy/pandas types are handled explicitly further down instead.
    if type(obj) in (bool, int, str):
        return obj
    if type(obj) is float:
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # ── numpy scalars ────────────────────────────────────────────────────
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, np.dtype):
        return str(obj)

    # ── pandas ───────────────────────────────────────────────────────────
    if obj is pd.NaT:
        return None
    if obj is pd.NA:
        return None
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, pd.Series):
        return [to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Index):
        return [to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        safe_df = obj.replace({np.nan: None})
        return {
            "columns": [to_json_safe(c) for c in safe_df.columns.tolist()],
            "rows": [to_json_safe(r) for r in safe_df.to_dict(orient="records")],
        }

    # ── stdlib types ─────────────────────────────────────────────────────
    if isinstance(obj, decimal.Decimal):
        try:
            if obj.is_nan():
                return None
        except (ValueError, decimal.InvalidOperation):
            pass
        return float(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, pathlib.Path):
        return str(obj)

    # ── containers ───────────────────────────────────────────────────────
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(x) for x in obj]

    # ── remaining NaN-like scalars (e.g. Decimal('nan') already handled
    #    above; this catches anything else pandas considers NA) ──────────
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    # ── last resort: anything json.dumps already accepts passes through
    #    unchanged; anything else is stringified rather than raising ─────
    try:
        import json as _json
        _json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
