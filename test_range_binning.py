# test_range_binning.py
# ─────────────────────────────────────────────────────────────────────────────
# Test and usage examples for common/transformations/range_binning.py.
# Run this locally to validate the range-binning logic before deploying.
#
# Usage:
#   python test_range_binning.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

from common.transformations.range_binning import (
    RangeBinningError,
    apply_range_binning,
    detect_range_binning,
)


def test_detect_explicit_ranges():
    """'Create column for rating range 0-1,1-2,2-3,3-4,4-5' -> Rating, 5 ranges."""
    columns = ["Rating", "Name"]
    result = detect_range_binning(
        "Create column for rating range 0-1,1-2,2-3,3-4,4-5", columns
    )
    assert result["detected"] is True
    assert result["source_column"] == "Rating"
    assert result["ranges"] == ["0-1", "1-2", "2-3", "3-4", "4-5"]
    assert result["new_column"] == "Rating_Range"
    print("test_detect_explicit_ranges: PASS")


def test_detect_open_ended_range():
    """'Group age into 0-18,19-30,31-45,46-60,60+' -> Age, with an open-ended '60+'."""
    columns = ["Age", "Name"]
    result = detect_range_binning(
        "Group age into 0-18,19-30,31-45,46-60,60+", columns
    )
    assert result["detected"] is True
    assert result["source_column"] == "Age"
    assert result["ranges"][-1] == "60+"
    print("test_detect_open_ended_range: PASS")


def test_detect_implicit_intent_no_ranges():
    """'Create salary bands' -> detected, column matched, ranges left for auto-generation."""
    result = detect_range_binning("Create salary bands", ["Salary", "Region"])
    assert result["detected"] is True
    assert result["source_column"] == "Salary"
    print("test_detect_implicit_intent_no_ranges: PASS")


def test_detect_non_binning_query():
    """An unrelated analytical question must NOT be detected."""
    result = detect_range_binning("What is the total revenue by region?", ["Revenue", "Region"])
    assert result["detected"] is False
    print("test_detect_non_binning_query: PASS")


def test_apply_basic_binning():
    df = pd.DataFrame({"Rating": [0.4, 1.6, 2.8, 3.9, 4.8]})
    result = apply_range_binning(df, "Rating", ["0-1", "1-2", "2-3", "3-4", "4-5"])
    out = result["dataframe"]
    assert list(out["Rating_Range"]) == ["0-1", "1-2", "2-3", "3-4", "4-5"]
    # Checked as a subset rather than exact dict equality: "formula_capable"
    # / "formula_intervals" were added additively for the Excel live-formula
    # write-back path and are covered by their own dedicated tests below
    # (test_formula_intervals_closed_ranges etc.) — this test only needs to
    # confirm the original, pre-existing metadata contract still holds.
    for key, expected in {
        "type": "column_transformation",
        "transformation": "range_binning",
        "source_column": "Rating",
        "new_column": "Rating_Range",
        "ranges": ["0-1", "1-2", "2-3", "3-4", "4-5"],
    }.items():
        assert result["metadata"][key] == expected
    print("test_apply_basic_binning: PASS")


def test_apply_decimal_ranges():
    """Decimal ranges (e.g. GPA 0.0-1.0,1.0-2.0,...) parse and bin correctly."""
    df = pd.DataFrame({"GPA": [0.3, 0.9, 1.5, 2.75, 3.95]})
    result = apply_range_binning(df, "GPA", ["0.0-1.0", "1.0-2.0", "2.0-3.0", "3.0-4.0"])
    out = result["dataframe"]
    assert list(out["GPA_Range"]) == ["0.0-1.0", "0.0-1.0", "1.0-2.0", "2.0-3.0", "3.0-4.0"]
    print("test_apply_decimal_ranges: PASS")


def test_apply_negative_values():
    """Negative ranges (e.g. temperature buckets) parse and bin correctly."""
    df = pd.DataFrame({"Temp": [-60, -45, -20, -5, 0, 5, 20]})
    result = apply_range_binning(df, "Temp", ["-50--10", "-10-0", "0-10"])
    out = result["dataframe"]
    assert list(out["Temp_Range"]) == [
        "Out of Range", "-50--10", "-50--10", "-10-0", "-10-0", "0-10", "Out of Range"
    ]
    assert result["metadata"]["ranges"] == ["-50--10", "-10-0", "0-10"]
    print("test_apply_negative_values: PASS")


def test_apply_plus_open_ended():
    """'20+' style open-ended-above ranges are inclusive of the boundary."""
    df = pd.DataFrame({"Age": [15, 19, 20, 25, 60]})
    result = apply_range_binning(df, "Age", ["0-19", "20+"])
    out = result["dataframe"]
    assert list(out["Age_Range"]) == ["0-19", "0-19", "20+", "20+", "20+"]
    print("test_apply_plus_open_ended: PASS")


def test_apply_below_and_above_exclusive_boundary():
    """'below X' / 'above X' are exclusive at X so they never double-claim
    the same boundary value as an adjacent closed range."""
    df = pd.DataFrame({"Score": [10, 49, 50, 99, 100, 150]})
    result = apply_range_binning(df, "Score", ["below 50", "50-100", "above 100"])
    out = result["dataframe"]
    assert list(out["Score_Range"]) == [
        "below 50", "below 50", "50-100", "50-100", "50-100", "above 100"
    ]
    print("test_apply_below_and_above_exclusive_boundary: PASS")


def test_apply_open_ended_and_below_above():
    df = pd.DataFrame({"Score": [10, 49, 50, 99, 100, 150]})
    result = apply_range_binning(df, "Score", ["below 50", "50-100", "above 100"])
    out = result["dataframe"]
    assert list(out["Score_Range"]) == [
        "below 50", "below 50", "50-100", "50-100", "50-100", "above 100"
    ]
    print("test_apply_open_ended_and_below_above: PASS")


def test_apply_missing_column():
    df = pd.DataFrame({"Rating": [1, 2, 3]})
    try:
        apply_range_binning(df, "DoesNotExist", ["0-1"])
        assert False, "expected RangeBinningError"
    except RangeBinningError:
        pass
    print("test_apply_missing_column: PASS")


def test_apply_non_numeric_column():
    df = pd.DataFrame({"Name": ["a", "b", "c"]})
    try:
        apply_range_binning(df, "Name", ["0-1"])
        assert False, "expected RangeBinningError"
    except RangeBinningError:
        pass
    print("test_apply_non_numeric_column: PASS")


def test_apply_overlapping_ranges():
    df = pd.DataFrame({"Rating": [0.5, 1.5]})
    try:
        apply_range_binning(df, "Rating", ["0-2", "1-3"])
        assert False, "expected RangeBinningError"
    except RangeBinningError:
        pass
    print("test_apply_overlapping_ranges: PASS")


def test_apply_unordered_ranges():
    df = pd.DataFrame({"Rating": [0.5, 1.5]})
    try:
        apply_range_binning(df, "Rating", ["1-2", "0-1"])
        assert False, "expected RangeBinningError"
    except RangeBinningError:
        pass
    print("test_apply_unordered_ranges: PASS")


def test_apply_auto_generated_ranges():
    df = pd.DataFrame({"Salary": [30000, 45000, 52000, 61000, 75000, 90000, 120000]})
    result = apply_range_binning(df, "Salary")  # no ranges given -> auto-generated
    assert len(result["metadata"]["ranges"]) == 5
    print("test_apply_auto_generated_ranges: PASS")


def test_priority_explicit_ranges_never_auto_generated():
    # Auto-generation for this column's actual min/max (30000-120000) would
    # produce completely different bin edges than these hand-picked ones —
    # if detect_range_binning ever "fixed up" or ignored explicit ranges in
    # favor of auto-generated ones, this test would catch it immediately.
    df = pd.DataFrame({"Salary": [30000, 45000, 52000, 61000, 75000, 90000, 120000]})
    detection = detect_range_binning(
        "Create salary bands 0-40000,40000-80000,80000-200000", list(df.columns), df
    )
    assert detection["ranges"] == ["0-40000", "40000-80000", "80000-200000"]
    result = apply_range_binning(
        df, detection["source_column"], detection["ranges"], detection["new_column"]
    )
    assert result["metadata"]["ranges"] == ["0-40000", "40000-80000", "80000-200000"]
    print("test_priority_explicit_ranges_never_auto_generated: PASS")


def test_priority_auto_generate_only_when_no_ranges_supplied():
    df = pd.DataFrame({"Salary": [30000, 45000, 52000, 61000, 75000, 90000, 120000]})
    # No explicit ranges in the text at all -> auto-generation is the only
    # path that can produce ranges.
    detection = detect_range_binning("Create salary bands", list(df.columns), df)
    assert detection["ranges"] is not None
    assert len(detection["ranges"]) == 5
    print("test_priority_auto_generate_only_when_no_ranges_supplied: PASS")


def test_formula_intervals_closed_ranges():
    df = pd.DataFrame({"Rating": [0.4, 1.7, 2.5, 3.9, 4.8]})
    result = apply_range_binning(df, "Rating", ["0-1", "1-2", "2-3", "3-4", "4-5"])
    assert result["metadata"]["formula_capable"] is True
    intervals = result["metadata"]["formula_intervals"]
    assert intervals[0] == {
        "low": 0.0, "high": 1.0, "low_open": False, "high_open": False, "label": "0-1"
    }
    assert intervals[-1] == {
        "low": 4.0, "high": 5.0, "low_open": False, "high_open": False, "label": "4-5"
    }
    print("test_formula_intervals_closed_ranges: PASS")


def test_formula_intervals_open_ended_ranges_use_none_for_unbounded_side():
    df = pd.DataFrame({"Age": [10, 25, 50, 70]})
    result = apply_range_binning(df, "Age", ["below 18", "18-40", "40-60", "above 60"])
    intervals = result["metadata"]["formula_intervals"]
    assert intervals[0]["low"] is None and intervals[0]["high_open"] is True
    assert intervals[-1]["high"] is None and intervals[-1]["low_open"] is True
    print("test_formula_intervals_open_ended_ranges_use_none_for_unbounded_side: PASS")


def test_end_to_end_from_text():
    df = pd.DataFrame({"Rating": [0.8, 1.9, 2.4, 4.2]})
    detection = detect_range_binning(
        "Create column for rating range 0-1,1-2,2-3,3-4,4-5", list(df.columns)
    )
    result = apply_range_binning(
        df, detection["source_column"], detection["ranges"], detection["new_column"]
    )
    assert list(result["dataframe"]["Rating_Range"]) == ["0-1", "1-2", "2-3", "4-5"]
    assert result["explanation"] == (
        "Created a new column named Rating_Range. "
        "Each value in Rating has been categorized into the specified numeric intervals."
    )
    print("test_end_to_end_from_text: PASS")


if __name__ == "__main__":
    test_detect_explicit_ranges()
    test_detect_open_ended_range()
    test_detect_implicit_intent_no_ranges()
    test_detect_non_binning_query()
    test_apply_basic_binning()
    test_apply_decimal_ranges()
    test_apply_negative_values()
    test_apply_plus_open_ended()
    test_apply_below_and_above_exclusive_boundary()
    test_apply_open_ended_and_below_above()
    test_apply_missing_column()
    test_apply_non_numeric_column()
    test_apply_overlapping_ranges()
    test_apply_unordered_ranges()
    test_apply_auto_generated_ranges()
    test_priority_explicit_ranges_never_auto_generated()
    test_priority_auto_generate_only_when_no_ranges_supplied()
    test_formula_intervals_closed_ranges()
    test_formula_intervals_open_ended_ranges_use_none_for_unbounded_side()
    test_end_to_end_from_text()
    print("\nAll range_binning tests passed.")
