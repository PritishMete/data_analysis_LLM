# test_cleaning_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Tests cleaning_ops.run_steps() directly — the deterministic executor half
# of the pipeline — so you can verify multi-step behavior WITHOUT needing a
# live Gemini API key. Each test's `steps` list is exactly what
# cleaning_agent.parse_cleaning_query() would produce for the NL query shown
# in the comment above it.
#
# Run with: python3 test_cleaning_agent.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

from cleaning_ops import run_steps


def _show(title, df, report):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(df)
    print(f"\nSummary: {report['summary']}")
    for s in report["steps"]:
        status = s["status"]
        marker = "✅" if status == "done" else ("⚠️" if status == "skipped" else "❌")
        print(f"  {marker} step {s['step']} [{s['op']}] -> {s}")


# ── TEST 1: single op ─────────────────────────────────────────────────────
# NL query: "fill missing age with mean"
def test_single_step():
    df = pd.DataFrame({"Name": ["A", "B", "C"], "Age": [25, np.nan, 35]})
    steps = [{"op": "fill_nulls", "columns": ["Age"], "method": "mean"}]
    out, report = run_steps(df, steps)
    _show("TEST 1: single step — fill missing age with mean", out, report)
    assert out["Age"].isnull().sum() == 0


# ── TEST 2: three chained ops in one query, in order ──────────────────────
# NL query: "fill missing salary with median, then convert salary from usd
#            to inr at 83.5, and remove duplicate rows based on id"
def test_multi_step_chain():
    df = pd.DataFrame({
        "id":     [1, 1, 2, 3],
        "salary": [50000.0, 50000.0, np.nan, 80000.0],
    })
    steps = [
        {"op": "fill_nulls", "columns": ["salary"], "method": "median"},
        {"op": "convert_currency", "column": "salary", "from_currency": "USD",
         "to_currency": "INR", "rate": 83.5},
        {"op": "remove_duplicates", "subset": ["id"]},
    ]
    out, report = run_steps(df, steps)
    _show("TEST 2: multi-step chain (fill -> convert -> dedupe)", out, report)
    assert "salary_INR" in out.columns
    assert out["salary"].isnull().sum() == 0
    assert len(out) == 3  # duplicate id=1 row removed


# ── TEST 3: header cleanup BEFORE a later step references the new casing ──
# NL query: "clean up the headers (lowercase, trim, underscores) then fill
#            any missing age with the mean"
def test_headers_then_fill_survives_rename():
    df = pd.DataFrame({" Full Name ": ["A", "B", "C"], "Age": [25, np.nan, 35]})
    steps = [
        {"op": "clean_headers", "case": "lower", "trim": True, "replace_spaces_with": "_"},
        # Agent output "age" (post-clean casing) per the system instruction —
        # but even if it had said "Age", _resolve_column's case-insensitive
        # match against the CURRENT dataframe still finds it.
        {"op": "fill_nulls", "columns": ["age"], "method": "mean"},
    ]
    out, report = run_steps(df, steps)
    _show("TEST 3: clean_headers then fill_nulls (column renamed mid-pipeline)", out, report)
    assert list(out.columns) == ["full_name", "age"]
    assert out["age"].isnull().sum() == 0


# ── TEST 4: filter phrased as a removal, inverted correctly ───────────────
# NL query: "drop rows where status is cancelled"
def test_filter_inverted():
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "status": ["cancelled", "completed", "completed", "cancelled"],
    })
    steps = [{"op": "filter_rows", "column": "status", "operator": "not_equals", "value": "cancelled"}]
    out, report = run_steps(df, steps)
    _show("TEST 4: filter_rows — drop rows where status is cancelled", out, report)
    assert len(out) == 2
    assert (out["status"] == "cancelled").sum() == 0


# ── TEST 5: one bad step in a chain is skipped, the rest still run ────────
# NL query: "fill missing age with mean and fix the xyz_typo column"
def test_bad_step_is_skipped_not_fatal():
    df = pd.DataFrame({"Age": [25, np.nan, 35]})
    steps = [
        {"op": "fill_nulls", "columns": ["Age"], "method": "mean"},
        {"op": "convert_currency", "column": "xyz_typo", "from_currency": "USD", "to_currency": "INR", "rate": 83.5},
    ]
    out, report = run_steps(df, steps)
    _show("TEST 5: one unresolvable step is skipped, valid steps still complete", out, report)
    assert out["Age"].isnull().sum() == 0
    assert report["steps"][1]["status"] == "skipped"


# ── TEST 6: whole-request round trip through parse_cleaning_query ─────────
# Only runs if google-adk + a live model/API key are configured; otherwise
# skips gracefully so this file still runs standalone with just pandas.
def test_live_agent_roundtrip():
    import asyncio

    async def _run():
        try:
            from cleaning_agent import parse_cleaning_query
        except Exception as e:
            print(f"\n[SKIPPED] live agent test — google-adk not available: {e}")
            return

        df = pd.DataFrame({
            " Customer Name ": ["Alice", "alice", "Bob"],
            "Amount USD": [100.0, 100.0, np.nan],
        })
        query = ("lowercase the headers and replace spaces with underscores, "
                 "then fill missing amount usd with the mean, then convert "
                 "amount usd to inr at 83.5")
        try:
            parsed = await parse_cleaning_query(query, list(df.columns))
        except Exception as e:
            print(f"\n[SKIPPED] live agent call failed (likely no API key configured): {e}")
            return

        print("\n" + "=" * 78)
        print("TEST 6: live agent round trip")
        print("=" * 78)
        print(f"Query: {query}")
        print(f"Parsed steps: {json_dumps_safe(parsed)}")
        out, report = run_steps(df, parsed.get("steps", []))
        _show("  -> executed result", out, report)

    asyncio.run(_run())


def json_dumps_safe(obj):
    import json
    try:
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)


if __name__ == "__main__":
    test_single_step()
    test_multi_step_chain()
    test_headers_then_fill_survives_rename()
    test_filter_inverted()
    test_bad_step_is_skipped_not_fatal()

    import asyncio
    asyncio.run(test_live_agent_roundtrip())

    print("\nALL TESTS COMPLETE.")
