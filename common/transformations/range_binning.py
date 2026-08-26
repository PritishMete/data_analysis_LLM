# common/transformations/range_binning.py
# ─────────────────────────────────────────────────────────────────────────────
# Column Binning / Range Categorization
#
# Turns any numeric column into a new categorical column by slicing it into
# user-defined (or auto-generated) numeric ranges.
#
#   "Create column for rating range 0-1,1-2,2-3,3-4,4-5"
#   "Group age into 0-18,19-30,31-45,46-60,60+"
#   "Create salary bands"
#
# Two entry points, matching the rest of common/ — deterministic, no LLM call:
#
#   detect_range_binning(text, columns, df=None)  -> intent + parameter extraction
#   apply_range_binning(df, source_column, ranges, new_column=None)
#                                                   -> the actual transformation
#
# Both are pure functions with no side effects on the caller's dataframe —
# apply_range_binning always returns a NEW dataframe (df.copy()), so it slots
# into the existing "transform -> re-run analyze_dataframe / statistics /
# report / charts on the result" pattern already used by clean_dataframe()
# in data_cleaner.py.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


class RangeBinningError(ValueError):
    """Raised for any validation failure in range binning (bad column,
    unparsable range, unordered/overlapping ranges, etc). Callers should
    catch this and surface `str(e)` directly to the user — messages are
    written to be human-readable on their own."""


# ── Intent detection ────────────────────────────────────────────────────────

# Phrases that signal the user wants a derived categorical/bucketed column.
# Matched with word boundaries against the lowercased request text.
_TRIGGER_PHRASES = (
    "range binning", "column binning", "range categorization", "range categorisation",
    "create range", "create ranges", "rating range", "score range", "age group",
    "age band", "age bracket", "salary band", "salary bracket", "salary range",
    "revenue bucket", "revenue band", "revenue range", "income bracket",
    "group into ranges", "group into range", "group into buckets", "group into bands",
    "bucketize", "bucketise", "bucket", "buckets", "band", "bands",
    "bin", "bins", "binning",
)

# Compiled once: word-boundary match for every trigger phrase.
_TRIGGER_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _TRIGGER_PHRASES) + r")\b",
    re.IGNORECASE,
)

# A single numeric token, integer or decimal, optionally negative (e.g. "-10", "-3.5").
_NUM = r"-?\d+(?:\.\d+)?"

# Individual range-token shapes, tried in this order.
_RE_CLOSED = re.compile(rf"^\s*({_NUM})\s*(?:-|to|–|—)\s*({_NUM})\s*$", re.IGNORECASE)
_RE_OPEN_ABOVE_PLUS = re.compile(rf"^\s*({_NUM})\s*\+\s*$")
_RE_ABOVE = re.compile(rf"^\s*(?:above|over|greater than|more than)\s+({_NUM})\s*$", re.IGNORECASE)
_RE_BELOW = re.compile(rf"^\s*(?:below|under|less than|fewer than)\s+({_NUM})\s*$", re.IGNORECASE)

# Finds candidate range tokens embedded in a free-form sentence, e.g.
# "group age into 0-18,19-30,31-45,46-60,60+" -> ["0-18","19-30",...,"60+"]
_RANGE_LIST_RE = re.compile(
    rf"(?:{_NUM}\s*(?:-|to|–|—)\s*{_NUM}|{_NUM}\s*\+|"
    rf"(?:above|over|greater than|more than|below|under|less than|fewer than)\s+{_NUM})",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _extract_ranges_from_text(text: str) -> list[str]:
    """Pulls out every range-shaped token in the text, in the order they
    appear (e.g. "0-1,1-2,2-3,3-4,4-5" or "0-18, 19-30, 31-45, 46-60, 60+").
    """
    return [m.group(0).strip() for m in _RANGE_LIST_RE.finditer(text)]


def _guess_target_column(text: str, columns: list[str]) -> str | None:
    """Matches the user's wording against the real column names. Prefers an
    exact (normalized) match; falls back to the longest column name that
    appears as a normalized substring of the request, so "rating range" ->
    "Rating" rather than a shorter, coincidentally-matching column.

    Tolerant of a simple singular/plural mismatch in either direction (e.g.
    request says "rating", column is named "Ratings", or vice versa) — this
    is the single most common way an otherwise-obvious match was silently
    missed. It intentionally does NOT try to guess synonyms/prefixes (e.g.
    "Overall_Rating", "Score") — that's a genuine ambiguous case, and
    RangeBinningTransformation.validate() surfaces a clear "which column?"
    error for it rather than this function guessing wrong.
    """
    norm_text = _normalize(text)
    candidates = []
    for col in columns:
        norm_col = _normalize(str(col))
        if not norm_col:
            continue
        forms = {norm_col}
        if norm_col.endswith("s") and len(norm_col) > 1:
            forms.add(norm_col[:-1])  # "ratings" -> "rating"
        else:
            forms.add(norm_col + "s")  # "rating" -> "ratings"
        if any(form in norm_text for form in forms):
            candidates.append(col)
    if not candidates:
        return None
    # Longest/most-specific column name wins ties (e.g. "AgeGroup" vs "Age").
    return max(candidates, key=lambda c: len(_normalize(str(c))))


def detect_range_binning(text: str, columns: list[str], df: pd.DataFrame | None = None) -> dict:
    """Rule-based intent + parameter extraction for range-binning requests.

    Args:
        text: the user's natural-language request.
        columns: available column names in the currently loaded dataset.
        df: optional — the loaded dataframe. When given and the user didn't
            name explicit ranges (e.g. "create salary bands"), it's used to
            auto-generate 5 sensible equal-width ranges spanning the
            column's actual min/max.

    Returns:
        {
          "detected": bool,
          "source_column": str | None,
          "ranges": list[str] | None,   # raw range tokens, still unparsed
          "new_column": str | None,     # None -> apply_range_binning defaults it
          "confidence": float,
          "message": str,
        }
    """
    text = text or ""

    # Generic categorization/classification is handled by the agentic
    # categorization flow, not numeric range binning. Only explicit range/bin/
    # bucket/band language belongs here.
    generic_categorization = re.search(
        r"\b(?:categorize|categorise|categorization|categorisation|classify|classification)\b",
        text,
        re.IGNORECASE,
    )
    explicit_numeric_grouping = re.search(
        r"\b(?:range(?:s)?|range\s+binning|column\s+binning|bucket(?:s|ize|ise|ing)?|"
        r"band(?:s|ing)?|bin(?:s|ning)?|group\s+into\s+(?:ranges?|buckets?|bands?))\b",
        text,
        re.IGNORECASE,
    )
    if generic_categorization and not explicit_numeric_grouping:
        return {
            "detected": False,
            "source_column": None,
            "ranges": None,
            "new_column": None,
            "confidence": 0.0,
            "message": "Generic categorization is handled by the Categorization Agent, not range binning.",
        }

    trigger_matched = bool(_TRIGGER_RE.search(text))
    ranges = _extract_ranges_from_text(text)

    # An explicit list of 2+ range-shaped tokens (e.g. "0-18,19-30,31-45") is
    # an unambiguous signal on its own — covers phrasing like "group age
    # into 0-18,19-30,..." that doesn't contain one of the fixed trigger
    # phrases verbatim (the trigger phrase for that case is "group ... into",
    # not "group into").
    strong_range_signal = len(ranges) >= 2

    if not trigger_matched and not strong_range_signal:
        return {
            "detected": False,
            "source_column": None,
            "ranges": None,
            "new_column": None,
            "confidence": 0.0,
            "message": "No range-binning/bucketing phrasing detected.",
        }

    source_column = _guess_target_column(text, columns or [])

    confidence = 0.5
    if source_column:
        confidence += 0.3
    if ranges:
        confidence += 0.2

    if not ranges and source_column is not None and df is not None and source_column in df.columns:
        try:
            ranges = _auto_generate_ranges(df[source_column])
        except RangeBinningError:
            ranges = []

    message = "Detected a request to bucket a numeric column into ranges."
    if source_column is None:
        message += " Could not confidently match a target column."
    if not ranges:
        message += " No explicit ranges found; defaults will be used."

    return {
        "detected": True,
        "source_column": source_column,
        "ranges": ranges or None,
        "new_column": f"{source_column}_Range" if source_column else None,
        "confidence": round(min(confidence, 1.0), 2),
        "message": message,
    }


# ── Range parsing / validation ──────────────────────────────────────────────

@dataclass
class _Interval:
    raw: str            # original token, used as the display label
    low: float
    high: float          # np.inf for open-ended "above X" / "X+"
    low_open: bool = False   # True => low is exclusive (used for "below X")
    high_open: bool = False  # kept for symmetry / future use


def _parse_range_token(token: str) -> _Interval:
    token = token.strip()

    m = _RE_CLOSED.match(token)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        if high <= low:
            raise RangeBinningError(
                f"Invalid range '{token}': the upper bound must be greater than the lower bound."
            )
        return _Interval(raw=token, low=low, high=high)

    m = _RE_OPEN_ABOVE_PLUS.match(token)
    if m:
        low = float(m.group(1))
        return _Interval(raw=token, low=low, high=np.inf)

    m = _RE_ABOVE.match(token)
    if m:
        low = float(m.group(1))
        return _Interval(raw=token, low=low, high=np.inf, low_open=True)

    m = _RE_BELOW.match(token)
    if m:
        high = float(m.group(1))
        return _Interval(raw=token, low=-np.inf, high=high, high_open=True)

    raise RangeBinningError(
        f"Could not parse range '{token}'. Expected formats like '0-1', '10-20', "
        f"'20+', 'below 50', or 'above 100'."
    )


def _auto_generate_ranges(series: pd.Series, num_bins: int = 5) -> list[str]:
    """Builds `num_bins` equal-width range tokens spanning the column's
    actual min/max — used when the user asks for e.g. "salary bands"
    without giving explicit numbers.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        raise RangeBinningError("Cannot auto-generate ranges: the column has no numeric values.")

    lo, hi = float(numeric.min()), float(numeric.max())
    if lo == hi:
        raise RangeBinningError("Cannot auto-generate ranges: the column has a single constant value.")

    step = (hi - lo) / num_bins
    is_int_like = float(numeric.round().eq(numeric).all()) == 1.0 and step >= 1

    def fmt(v: float) -> str:
        return str(int(round(v))) if is_int_like else f"{round(v, 2):g}"

    ranges = []
    for i in range(num_bins):
        b_lo = lo + i * step
        b_hi = hi if i == num_bins - 1 else lo + (i + 1) * step
        ranges.append(f"{fmt(b_lo)}-{fmt(b_hi)}")
    return ranges


def _parse_and_validate_ranges(ranges: list[str]) -> list[_Interval]:
    if not ranges:
        raise RangeBinningError("At least one range must be provided.")

    intervals = [_parse_range_token(r) for r in ranges]
    ordered = sorted(intervals, key=lambda iv: iv.low)

    # Ordering check: reject if the caller supplied them out of order, since
    # a scrambled range list is almost always a mistake worth surfacing
    # rather than silently re-sorting and hiding it.
    if [iv.raw for iv in intervals] != [iv.raw for iv in ordered]:
        raise RangeBinningError(
            "Ranges are not in ascending order: "
            f"expected an order like {[iv.raw for iv in ordered]}."
        )

    # Overlap check on the sorted list.
    for a, b in zip(ordered, ordered[1:]):
        if b.low < a.high:
            raise RangeBinningError(
                f"Ranges '{a.raw}' and '{b.raw}' overlap between {b.low} and {a.high}."
            )

    return ordered


def _formula_intervals(intervals: list[_Interval]) -> list[dict]:
    """Turns the already-parsed/validated/ordered `_Interval` list into the
    plain-dict shape the Excel formula write-back layer consumes (see the
    "formula_intervals" metadata key in apply_range_binning). Mirrors
    _label_for_value's own semantics exactly (same low/high/open flags, same
    ascending-order-first-match evaluation) so a nested IF() built from this
    data in Excel always agrees with what this module would compute in
    Python for the same value.
    """
    return [
        {
            "low": None if np.isneginf(iv.low) else iv.low,
            "high": None if np.isposinf(iv.high) else iv.high,
            "low_open": iv.low_open,
            "high_open": iv.high_open,
            "label": iv.raw,
        }
        for iv in intervals
    ]


# ── Transformation ───────────────────────────────────────────────────────────

def _label_for_value(value: float, intervals: list[_Interval]) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    # Every bin is closed on both ends by default ([low, high]), which is
    # what makes disjoint integer ranges like "0-18,19-30,...,60+" work
    # exactly as written. "below X" / "above X" tokens are the exception —
    # their shared boundary is exclusive (below_open/above_open), so e.g.
    # "below 50" + "50-100" don't both claim the value 50.
    # For ranges that DO share an inclusive boundary (e.g. "0-1" / "1-2"),
    # intervals are checked in ascending order and the FIRST (lower) match
    # wins — so 1.0 belongs to "0-1", not "1-2".
    for iv in intervals:
        lower_ok = value > iv.low if iv.low_open else value >= iv.low
        upper_ok = value < iv.high if iv.high_open else value <= iv.high
        if lower_ok and upper_ok:
            return iv.raw
    return "Out of Range"


def _default_new_column_name(df: pd.DataFrame, source_column: str, requested: str | None) -> str:
    if requested:
        name = requested
    else:
        name = f"{source_column}_Range"
    if name not in df.columns:
        return name
    # Avoid clobbering an existing column if this transformation is applied twice.
    n = 2
    while f"{name}_{n}" in df.columns:
        n += 1
    return f"{name}_{n}"


def apply_range_binning(
    df: pd.DataFrame,
    source_column: str,
    ranges: list[str] | None = None,
    new_column: str | None = None,
) -> dict:
    """Creates a new categorical column from `source_column` by bucketing
    its numeric values into `ranges`.

    Validates:
      - source_column exists
      - source_column is numeric
      - ranges are parseable, ordered, and non-overlapping

    Returns:
        {
          "dataframe": pd.DataFrame,   # NEW dataframe (original untouched) with the added column
          "metadata": {
              "type": "column_transformation",
              "transformation": "range_binning",
              "source_column": "...",
              "new_column": "...",
              "ranges": ["0-1", "1-2", ...],
          },
          "preview": {"before": [...], "after": [...]},   # up to 5 sample rows
          "explanation": "Created a new column named ... .",
        }

    Raises:
        RangeBinningError on any validation failure.
    """
    if source_column not in df.columns:
        raise RangeBinningError(f"Column '{source_column}' does not exist in the dataset.")

    if not pd.api.types.is_numeric_dtype(df[source_column]):
        raise RangeBinningError(
            f"Column '{source_column}' is not numeric — range binning requires a numeric column."
        )

    ranges = ranges or _auto_generate_ranges(df[source_column])
    intervals = _parse_and_validate_ranges(ranges)

    result_column = _default_new_column_name(df, source_column, new_column)

    new_df = df.copy()
    new_df[result_column] = new_df[source_column].apply(lambda v: _label_for_value(v, intervals))

    sample_idx = df.index[:10]
    preview_before = df.loc[sample_idx, [source_column]].to_dict(orient="records")
    preview_after = new_df.loc[sample_idx, [source_column, result_column]].to_dict(orient="records")

    metadata = {
        "type": "column_transformation",
        "transformation": "range_binning",
        "source_column": source_column,
        "new_column": result_column,
        "ranges": [iv.raw for iv in intervals],
        # Structured, already-validated/ordered interval data for the Excel
        # write-back layer (lib/core/interop/excel_interop_web.dart ->
        # web/excel_data_processor.js::buildRangeBinningFormulaForRow) to turn
        # into a LIVE nested-IF Excel formula, so the derived column
        # recalculates automatically when the user edits the source cells
        # instead of staying a one-time computed value. The JS layer does NOT
        # re-derive bounds/ordering/overlap on its own — that logic stays
        # single-sourced here. `low`/`high` of None means "no bound on this
        # side" (from an "above X" / "below X" / "X+" token); `to_json_safe`
        # (common/json_safe.py) already turns math.inf into null on the wire,
        # so no extra serialization step is needed for that case.
        "formula_capable": True,
        "formula_intervals": _formula_intervals(intervals),
    }

    explanation = (
        f"Created a new column named {result_column}. "
        f"Each value in {source_column} has been categorized into the specified numeric intervals."
    )

    return {
        "dataframe": new_df,
        "metadata": metadata,
        "preview": {"before": preview_before, "after": preview_after},
        "explanation": explanation,
    }
