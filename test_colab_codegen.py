"""
test_colab_codegen.py
─────────────────────────────────────────────────────────────────────────────
Tests colab_codegen.py — the deterministic plan/action -> Python-source
renderer used by /generate_colab_code — WITHOUT needing a live Gemini API
key. Every test builds the exact JSON shape command_agent.py / query_router
would produce, generates code from it, then actually EXECUTES that code
against a real DataFrame to confirm it's not just syntactically valid but
behaviorally correct.

Run with: python3 test_colab_codegen.py
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

from colab_codegen import gen_sql_code, gen_operation_code
from query_router import build_sql_from_plan


def _run(code, df):
    ns = {"df": df.copy(), "pd": pd, "np": np}
    exec(compile(code, "<generated>", "exec"), ns)
    return ns["df"]


# ── SQL plan -> DuckDB code ────────────────────────────────────────────────

def test_sql_plan_group_by():
    df = pd.DataFrame({
        "Region": ["North", "South", "North", "East"],
        "Revenue": [100, 200, 150, 300],
    })
    plan = {
        "group_by": ["Region"],
        "metrics": [{"column": "Revenue", "function": "sum", "alias": "total_revenue"}],
        "order_by": [{"column": "total_revenue", "direction": "desc"}],
    }
    sql = build_sql_from_plan(plan, list(df.columns))
    code = gen_sql_code(sql, "df")
    ns = {"df": df, "pd": pd}
    exec(compile(code, "<g>", "exec"), ns)
    result = ns["result"]
    assert list(result["Region"]) == ["East", "North", "South"]
    print("[PASS] SQL plan group-by codegen")


# ── operation: filter ──────────────────────────────────────────────────────

def test_filter_not_equals():
    df = pd.DataFrame({"rating_count": [5, 0, 12, 0, 8]})
    action = {"action": "filter",
              "filter": {"columnName": "rating_count", "type": "not_equals", "value": "0", "value2": ""}}
    code = gen_operation_code(action, "df", list(df.columns))
    out = _run(code, df)
    assert (out["rating_count"] == 0).sum() == 0
    assert len(out) == 3
    print("[PASS] filter (not_equals) codegen")


def test_filter_top_n():
    df = pd.DataFrame({"score": [10, 50, 30, 90, 20]})
    action = {"action": "filter", "filter": {"columnName": "score", "type": "top_n", "value": "2", "value2": ""}}
    code = gen_operation_code(action, "df", list(df.columns))
    out = _run(code, df)
    assert sorted(out["score"].tolist()) == [50, 90]
    print("[PASS] filter (top_n) codegen")


# ── operation: deduplicate ─────────────────────────────────────────────────

def test_deduplicate():
    df = pd.DataFrame({"id": [1, 1, 2, 3], "v": ["a", "a", "b", "c"]})
    action = {"action": "deduplicate", "deduplicate": {"columns": ["id"]}}
    code = gen_operation_code(action, "df", list(df.columns))
    out = _run(code, df)
    assert len(out) == 3
    print("[PASS] deduplicate codegen")


# ── operation: add_column (condition — new vs returning) ──────────────────

def test_add_column_condition():
    df = pd.DataFrame({"CustomerName": ["Anita", "Anita", "Vikram", "Deepa", "Deepa"]})
    action = {
        "action": "add_column",
        "add_column": {
            "newColumnName": "Customer_Status",
            "condition": {"windowFunction": "count", "column": "CustomerName",
                          "partitionBy": ["CustomerName"], "operator": "greater_than", "value": "1"},
            "thenLabel": "Returning", "elseLabel": "New",
        },
    }
    code = gen_operation_code(action, "df", list(df.columns))
    out = _run(code, df)
    assert out.loc[out["CustomerName"] == "Vikram", "Customer_Status"].iloc[0] == "New"
    assert out.loc[out["CustomerName"] == "Anita", "Customer_Status"].iloc[0] == "Returning"
    print("[PASS] add_column (condition) codegen")


# ── operation: add_column (formula — compute) ──────────────────────────────

def test_add_column_formula_compute():
    df = pd.DataFrame({"UnitPrice": [100.0, 50.0], "DiscountPct": [20, 0]})
    action = {
        "action": "add_column",
        "add_column": {
            "newColumnName": "discounted_price", "condition": None,
            "formula": {"leftExpression": None, "rightExpression": "UnitPrice * (1 - DiscountPct/100)",
                        "operator": "equals", "tolerance": None, "mode": "compute"},
            "thenLabel": "Match", "elseLabel": "Mismatch",
        },
    }
    code = gen_operation_code(action, "df", list(df.columns))
    out = _run(code, df)
    assert out["discounted_price"].tolist() == [80.0, 50.0]
    print("[PASS] add_column (formula/compute) codegen")


# ── operation: fill_missing ─────────────────────────────────────────────────

def test_fill_missing_median():
    df = pd.DataFrame({"age": [20.0, np.nan, 40.0]})
    action = {"action": "fill_missing", "fill_missing": {"column": "age", "strategy": "median", "sourceFormulaColumn": None}}
    code = gen_operation_code(action, "df", list(df.columns))
    out = _run(code, df)
    assert out["age"].isnull().sum() == 0
    print("[PASS] fill_missing (median) codegen")


# ── operation: multi_step ──────────────────────────────────────────────────

def test_multi_step():
    df = pd.DataFrame({
        "Rating Count": [5, 0, 12, 0],
        "id": [1, 1, 2, 3],
    })
    action = {
        "action": "multi_step",
        "multi_step": {"outputSheetName": "Cleaned_Data", "steps": [
            {"op": "standardize_columns"},
            {"op": "filter_rows", "column": "rating_count", "operator": "not_equals", "value": "0"},
            {"op": "remove_duplicates", "subset": ["id"]},
        ]},
    }
    code = gen_operation_code(action, "df", list(df.columns))
    out = _run(code, df)
    assert "rating_count" in out.columns
    assert (out["rating_count"] == 0).sum() == 0
    print("[PASS] multi_step codegen")


if __name__ == "__main__":
    test_sql_plan_group_by()
    test_filter_not_equals()
    test_filter_top_n()
    test_deduplicate()
    test_add_column_condition()
    test_add_column_formula_compute()
    test_fill_missing_median()
    test_multi_step()
    print("\nALL TESTS COMPLETE.")
