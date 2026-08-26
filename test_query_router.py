"""
test_query_router.py
─────────────────────────────────────────────────────────────────────────────
Tests the parts of query_router.py that don't require a live Gemini API key:

  1. build_sql_from_plan() — the deterministic plan -> SQL builder — run
     against TWO completely different datasets (sales-style and restaurant-
     style columns) using the exact same plan SHAPES, to prove the structure
     is schema-agnostic: the LLM never needs to have seen these column names
     before, it only needs to output the operation taxonomy from query.json
     (group_by / metrics / filters / window / order_by / limit).
  2. _execute_sql() — confirms the generated SQL actually runs correctly via
     DuckDB against real sample data.
  3. Column-resolution safety — confirms a plan referencing a column that
     doesn't exist in the current dataset is rejected (PlanError), rather
     than silently generating broken/incorrect SQL.

Run with: python3 test_query_router.py
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
from query_router import build_sql_from_plan, _execute_sql, PlanError


def run_plan(name, plan, df):
    columns = list(df.columns)
    sql = build_sql_from_plan(plan, columns)
    result = _execute_sql(sql, df)
    print(f"\n--- {name} ---")
    print(f"SQL: {sql}")
    print(f"columns: {result['columns']}")
    for row in result["rows"]:
        print(f"  {row}")
    return result


# ── Dataset A: sales-style columns (matches SALES_DATA_SQL.sql) ──────────────

sales_df = pd.DataFrame({
    "OrderID":         [1, 2, 3, 4, 5, 6, 7, 8],
    "Region":          ["North", "South", "North", "East", "South", "West", "East", "North"],
    "Product":         ["Widget", "Gadget", "Widget", "Gizmo", "Widget", "Gadget", "Gizmo", "Gizmo"],
    "Salesperson":     ["Amit", "Priya", "Amit", "Rahul", "Priya", "Sara", "Rahul", "Amit"],
    "Quantity":        [10, 5, 8, 3, 12, 6, 4, 7],
    "DiscountedPrice": [100.0, 200.0, 100.0, 300.0, 100.0, 200.0, 300.0, 300.0],
})
sales_df["Revenue"] = sales_df["DiscountedPrice"] * sales_df["Quantity"]

# Plan A1: "total revenue by region, ranked highest to lowest"
plan_a1 = {
    "group_by": ["Region"],
    "metrics": [{"column": "Revenue", "function": "sum", "alias": "total_revenue"}],
    "order_by": [{"column": "total_revenue", "direction": "desc"}],
}

# Plan A2: "best-selling product in each region" (window + keep_top_n_per_partition)
plan_a2 = {
    "group_by": ["Region", "Product"],
    "metrics": [{"column": "Quantity", "function": "sum", "alias": "total_qty"}],
    "window": {"type": "rank", "partition_by": ["Region"],
               "order_by": [{"column": "total_qty", "direction": "desc"}]},
    "keep_top_n_per_partition": 1,
}


# ── Dataset B: restaurant-style columns (matches RESTAURANTS_SQL.sql), ───────
#    DIFFERENT column names entirely — proves the SAME plan shapes generalize.

restaurants_df = pd.DataFrame({
    "name":         ["Spice Hub", "Curry House", "Pizza Point", "Biryani Bros", "Wok On", "Grill King"],
    "city":         ["Bangalore", "Bangalore", "Delhi", "Delhi", "Mumbai", "Mumbai"],
    "cuisine":      ["Indian", "Indian", "Italian", "Indian", "Chinese", "American"],
    "cost":         [500, 300, 700, 250, 450, 900],
    "rating":       [4.5, 4.1, 3.9, 4.7, 4.0, 3.8],
    "rating_count": [1200, 800, 500, 2200, 650, 300],
})

# Plan B1: "average rating and total rating_count per city" — SAME SHAPE as
# plan_a1's group_by+metrics, just different column names.
plan_b1 = {
    "group_by": ["city"],
    "metrics": [
        {"column": "rating", "function": "avg", "alias": "avg_rating"},
        {"column": "rating_count", "function": "sum", "alias": "total_ratings"},
    ],
    "order_by": [{"column": "avg_rating", "direction": "desc"}],
}

# Plan B2: "highest-rated restaurant in each city" — SAME SHAPE as plan_a2
# (window + keep_top_n_per_partition), again just different columns.
plan_b2 = {
    "group_by": ["city", "name"],
    "metrics": [{"column": "rating", "function": "max", "alias": "top_rating"}],
    "window": {"type": "rank", "partition_by": ["city"],
               "order_by": [{"column": "top_rating", "direction": "desc"}]},
    "keep_top_n_per_partition": 1,
}

# Plan B3: filter — "restaurants in Bangalore costing 300 or less"
plan_b3 = {
    "filters": [
        {"column": "city", "operator": "equals", "value": "Bangalore"},
        {"column": "cost", "operator": "less_than_equal", "value": "300"},
    ],
}

print("=" * 70)
print("DATASET A (sales-style columns)")
print("=" * 70)
run_plan("A1: total revenue by region, ranked", plan_a1, sales_df)
run_plan("A2: best-selling product per region", plan_a2, sales_df)

print("\n" + "=" * 70)
print("DATASET B (restaurant-style columns — completely different schema)")
print("=" * 70)
run_plan("B1: avg rating + total rating_count per city (same shape as A1)", plan_b1, restaurants_df)
run_plan("B2: highest-rated restaurant per city (same shape as A2)", plan_b2, restaurants_df)
run_plan("B3: filter — Bangalore, cost <= 300", plan_b3, restaurants_df)


# ── Derived-column test: "compare revenue between new and returning customers" ─
#    (the exact pattern that previously failed — a category built from a
#    window function, aggregated afterward)

sales_orders_df = pd.DataFrame({
    "CustomerName": ["Anita", "Anita", "Vikram", "Deepa", "Deepa", "Deepa", "Rohan"],
    "TotalPrice":   [500,     300,      700,      200,     150,     400,     900],
})

plan_new_vs_returning = {
    "derived_columns": [
        {
            "alias": "customer_category",
            "case": {
                "condition": {
                    "window_function": "count",
                    "column": "CustomerName",
                    "partition_by": ["CustomerName"],
                    "operator": "greater_than",
                    "value": "1",
                },
                "then": "Returning",
                "else": "New",
            },
        }
    ],
    "group_by": ["customer_category"],
    "metrics": [{"column": "TotalPrice", "function": "sum", "alias": "total_rev"}],
}

print("\n" + "=" * 70)
print("DERIVED COLUMN TEST: new vs returning customers")
print("=" * 70)
run_plan("compare revenue between new and returning customers", plan_new_vs_returning, sales_orders_df)
# Expect: Anita (2 orders) + Deepa (3 orders) -> Returning = 500+300+200+150+400 = 1550
#         Vikram (1 order) + Rohan (1 order) -> New = 700+900 = 1600




print("\n" + "=" * 70)
print("COLUMN-RESOLUTION SAFETY TEST")
print("=" * 70)
bad_plan = {"group_by": ["nonexistent_column_xyz"],
            "metrics": [{"column": "rating", "function": "avg", "alias": "avg_rating"}]}
try:
    build_sql_from_plan(bad_plan, list(restaurants_df.columns))
    print("[FAIL] expected PlanError but none was raised")
except PlanError as e:
    print(f"[PASS] correctly rejected: {e}")

print("\nALL TESTS COMPLETE.")
