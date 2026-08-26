"""
query_router.py
─────────────────────────────────────────────────────────────────────────────
Implements the query.json principle directly:

    "LLM should decide the analytical operation, but backend should generate
     and execute queries."

The router agent NEVER writes raw SQL. It only decides:
  (a) route: "sql" (an analytical question) vs "operation" (a spreadsheet
      action — pivot/filter/dedupe/color_scale, handled by command_agent.py)
  (b) if "sql": a STRUCTURED PLAN using the exact operation taxonomy from
      query.json — aggregation / filter / ranking / window_function — built
      only from column names, functions, and conditions (no SQL syntax).

A deterministic Python builder (`build_sql_from_plan`) then turns that plan
into a DuckDB SQL statement, resolving every column name against whatever
dataset is actually loaded at request time. Because the builder — not the
LLM — is what touches SQL syntax, this works on ANY dataset without
depending on column names the model has seen before, and it's trivially
testable without an LLM at all (see test_query_router.py).

Wire into main.py with:

    from query_router import handle_smart_query
    result = await handle_smart_query(text, df, available_sheets)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import re
import traceback
import uuid
import logging
import os

import duckdb
import pandas as pd

try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
except Exception:  # pragma: no cover - import fallback for local tests
    class _UnavailableGemini:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("google.adk is unavailable in this environment.")

    class _UnavailableSessionService:
        async def create_session(self, *args, **kwargs):
            return None

    class _TypesNamespace:
        class Content:
            def __init__(self, *args, **kwargs):
                self.parts = kwargs.get("parts", [])

        class Part:
            def __init__(self, *args, **kwargs):
                self.text = kwargs.get("text", "")

    LlmAgent = _UnavailableGemini
    Runner = _UnavailableGemini
    InMemorySessionService = _UnavailableSessionService
    types = _TypesNamespace()

from command_agent import parse_agentic_command
from common.transformations import TransformationEngine
from common.json_safe import to_json_safe
from common.response_envelope import smart_query_envelope
from sentiment_agent import analyze_sentiment
from privacy_context import strict_enabled, safe_columns, sanitize_user_text, remap_plan, value_aliases

logger = logging.getLogger(__name__)


def _operation_error_response(message: str, *, error_type: str = "INTERNAL_ERROR", confidence: float = 0.0, **extra) -> dict:
    """Same envelope shape (success/route/operation/metadata/preview/
    statistics/schema/ai_report/warnings/errors — see TASK 7 /
    common/response_envelope.py) every successful response from this module
    already uses — kept as a local helper (rather than importing main.py's
    smart_query_error_response) to avoid a circular import, since main.py
    already imports handle_smart_query FROM this module.

    Passed through `to_json_safe()` before being returned: this is the
    error path, most likely to be carrying a raw exception object or other
    non-JSON-native value in `extra` (e.g. `exception=e` instead of
    `str(e)`), so it gets the same serialization guarantee as every success
    path instead of relying on every call site remembering to stringify
    first.
    """
    operation = {"action": "transformation_error", "error_type": error_type, "error": message}
    operation.update(extra)
    envelope = smart_query_envelope(
        success=False,
        route="operation",
        message=message,
        confidence=confidence,
        operation=operation,
        errors=[{"error_type": error_type, "message": message}],
    )
    return to_json_safe(envelope)

# Single shared engine instance — stateless, see common/transformations/
# transformation_engine.py. The fast-path below routes ANY transformation
# request (range binning, rename, drop, fill missing, dedupe, merge/split
# columns, type conversion, date features, ...) through it, deterministically
# and without an LLM call, before falling back to the Gemini router.
_transformation_engine = TransformationEngine()

MODEL = os.getenv("QUERY_ROUTER_MODEL", "gemini-2.5-flash")

# Name the uploaded dataframe is registered under inside DuckDB.
TABLE_NAME = "data"

AGG_FUNCTIONS = {"sum", "avg", "count", "min", "max"}
FILTER_OPERATORS = {
    "equals", "not_equals", "contains",
    "greater_than", "less_than", "greater_than_equal", "less_than_equal",
    "between", "above_average", "below_average",
}
WINDOW_TYPES = {"rank", "dense_rank", "running_total", "moving_average"}


# ── Router prompt — outputs a STRUCTURED PLAN, never raw SQL ─────────────────

ROUTER_SYSTEM_INSTRUCTION = """You are a routing + query-planning agent for a natural language
data analysis tool. You are given the list of available column names for the currently loaded
dataset, plus the user's request. You NEVER write SQL — you only decide the route and, if
applicable, a structured plan built from the operation types below.

Decide ONE of three routes:

1. "sentiment" — the user asks to analyze customer-review sentiment, customer satisfaction, review tone, or restaurant performance based on a review-text column. Use this route even when the user says "show me" or "based on ReviewText". The sentiment analyzer will batch the review texts and produce row-level sentiment plus restaurant-level performance.
2. "sql" — the request is an analytical QUESTION about the data: aggregations, filtering to
   answer a question, ranking / top-N / bottom-N, or window functions (rank, dense_rank, running
   total, moving average). Any read-only question whose answer is a table or a single value.
3. "operation" — the request asks to MODIFY or reshape the spreadsheet itself: building a pivot
   table into a new sheet, permanently filtering/keeping/removing rows in a sheet, deduplicating,
   conditional colour formatting, or ADDING A NEW PERSISTENT COLUMN to the sheet (e.g. "add a
   column that marks customers as new or returning"). These go through a separate handler — set
   "plan" to null.

   DISAMBIGUATION for classification-style requests: if the user says "add", "create", "insert" a
   column, or "mark"/"label"/"tag" each row — that PERSISTS a change to the sheet, so route
   "operation". If instead the user is asking a QUESTION or wants a comparison/report (e.g.
   "compare revenue between new and returning customers") without asking to modify the sheet,
   route "sql" and use derived_columns (below) to compute the classification as part of the query.

If route is "sentiment", build plan as {"review_column":"<review text column>","restaurant_column":"<restaurant name column or null>"}. Match ReviewText/review/comment/content to the closest real review-text column and Restaurant Name to the closest real restaurant column.

If route is "sql", build a "plan" object using ONLY these fields (omit any that don't apply,
never invent a column that isn't in the available list — match wording to the closest real
column, case-insensitive):

{
  "group_by": ["<column>", ...],              // columns to group by, for aggregation/ranking
  "metrics": [{"column": "<column>", "function": "sum|avg|count|min|max", "alias": "<short_name>"}],
  "filters": [{"column": "<column>", "operator": "equals|not_equals|contains|greater_than|
                less_than|greater_than_equal|less_than_equal|between|above_average|below_average",
               "value": "<string>", "value2": "<string, only for between>"}],
  "window": {"type": "rank|dense_rank|running_total|moving_average",
             "partition_by": ["<column>", ...],
             "order_by": [{"column": "<column or metric alias>", "direction": "asc|desc"}],
             "window_size": <int, only for moving_average, default 3>},
  "keep_top_n_per_partition": <int, ONLY when the user wants just the top/bottom result WITHIN
                                each group, e.g. "best product in each region" -> 1>,
  "order_by": [{"column": "<column or metric alias>", "direction": "asc|desc"}],
  "limit": <int, for a plain top-N/bottom-N over the whole result, not per group>,
  "derived_columns": [
    {
      "alias": "<short_name, e.g. customer_category>",
      "case": {
        "condition": {
          "window_function": "count|sum|avg|min|max",
          "column": "<column to aggregate, omit for count(*)>",
          "partition_by": ["<column>", ...],
          "operator": "equals|not_equals|greater_than|less_than|greater_than_equal|less_than_equal",
          "value": "<string>"
        },
        "then": "<label if condition is true>",
        "else": "<label if condition is false>"
      }
    }
  ]
}

Use "derived_columns" whenever the user wants to CLASSIFY rows based on a per-group count/sum/etc
before aggregating — e.g. "compare revenue between new and returning customers" needs a derived
label (count of orders per customer > 1 -> "Returning" else "New") that group_by/metrics then
reference by its alias, exactly like a real column.

CRITICAL DISAMBIGUATION — "new vs returning customers" is NEVER a real column, even if the
dataset happens to have a similarly-worded column like "CustomerType" (e.g. Retail/Wholesale) or
"CustomerCategory". Those are unrelated business categories, not purchase-frequency labels. If
the user's wording is about NEW vs RETURNING, FIRST-TIME vs REPEAT, or ONE-TIME vs LOYAL
customers, you MUST use derived_columns with a count-based condition — do not take the shortcut
of matching to an existing categorical column just because a plausible-sounding one exists. Only
use an existing column directly when the user names an attribute the dataset actually tracks as
such (e.g. "revenue by customer type" or "revenue by retail vs wholesale" -> group_by that real
column, no derived_columns needed).

Respond with ONLY a single JSON object — no markdown fences, no commentary — matching EXACTLY:

{
  "route": "sentiment" | "sql" | "operation",
  "plan": { ... as above ... } | null,
  "confidence": <float 0 to 1>,
  "message": "<one short sentence confirming what you understood>"
}

confidence reflects how well you matched real column names — never how complex the request is.

EXAMPLES (illustrative only — always use the real "Available columns" given to you):

User request: show me the customer satisfaction or sentiment based on ReviewText
-> {"route":"sentiment","plan":{"review_column":"ReviewText","restaurant_column":"Restaurant Name"},"confidence":0.98,"message":"Analyzed customer sentiment from ReviewText and summarized restaurant satisfaction."}

User request: total revenue by region, ranked highest to lowest
-> {"route":"sql","plan":{"group_by":["region"],
     "metrics":[{"column":"revenue","function":"sum","alias":"total_revenue"}],
     "order_by":[{"column":"total_revenue","direction":"desc"}]},
    "confidence":0.9,"message":"Grouped revenue by region, ordered highest to lowest."}

User request: best-selling product in each region
-> {"route":"sql","plan":{"group_by":["region","product"],
     "metrics":[{"column":"quantity","function":"sum","alias":"total_qty"}],
     "window":{"type":"rank","partition_by":["region"],
               "order_by":[{"column":"total_qty","direction":"desc"}]},
     "keep_top_n_per_partition":1},
    "confidence":0.85,"message":"Found the top product by quantity within each region."}

User request: build a pivot table showing total sales by region and product
-> {"route":"operation","plan":null,"confidence":0.9,
    "message":"This reshapes the sheet, so it is a pivot operation, not a SQL question."}

User request: add a column called Customer_Status that marks returning customers
-> {"route":"operation","plan":null,"confidence":0.9,
    "message":"This adds a persistent column to the sheet, so it is an add_column operation, not a SQL question."}

User request: compare revenue between new and returning customers
-> {"route":"sql","plan":{
     "derived_columns":[{"alias":"customer_category","case":{
        "condition":{"window_function":"count","column":"customername",
                      "partition_by":["customername"],"operator":"greater_than","value":"1"},
        "then":"Returning","else":"New"}}],
     "group_by":["customer_category"],
     "metrics":[{"column":"totalprice","function":"sum","alias":"total_rev"}]},
    "confidence":0.85,
    "message":"Classified customers as new/returning by order count, then summed revenue per group."}

User request: compare revenue by customer type
-> {"route":"sql","plan":{
     "group_by":["customertype"],
     "metrics":[{"column":"totalprice","function":"sum","alias":"total_rev"}]},
    "confidence":0.9,
    "message":"Grouped revenue by the existing customer_type column (e.g. Retail vs Wholesale) — not a derived new/returning label, since the user asked about type, not purchase frequency."}
"""


def _extract_json(text: str) -> str:
    """Pulls a JSON object out of arbitrary model output (fenced or with stray prose)."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]
    return text


# ── Deterministic SQL builder (no LLM involved) ───────────────────────────────

class PlanError(Exception):
    """Raised when a plan can't be safely resolved against the actual dataset."""


def _resolve_column(name: str, available_columns: list) -> str:
    """Matches a model-provided column name to a REAL column in the current
    dataset, case-insensitively, with a loose substring fallback. Raises
    PlanError if nothing reasonable matches — the plan must never silently
    reference a column that doesn't exist.
    """
    if not name:
        raise PlanError("Empty column name in plan.")
    name_l = str(name).strip().lower()
    for col in available_columns:
        if col.lower() == name_l:
            return col
    candidates = [c for c in available_columns if name_l in c.lower() or c.lower() in name_l]
    if candidates:
        return candidates[0]
    raise PlanError(f"Column '{name}' does not match any column in the current dataset.")


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value) -> str:
    """Renders a value for use in a SQL literal position, numeric when possible."""
    if value is None:
        return "NULL"
    s = str(value)
    try:
        float(s)
        return s  # numeric, no quoting needed
    except ValueError:
        return "'" + s.replace("'", "''") + "'"



def build_parameterized_filter_sql(
    filters: list[dict],
    available_columns: list[str],
    table_name: str = TABLE_NAME,
) -> tuple[str, list]:
    """Build a safe generic SELECT for row-level filters.

    Column identifiers are resolved/quoted from the real schema; user values
    are returned separately as DuckDB parameters. This is the backend SQL
    structure the agent fills with values — the agent never writes SQL.
    """
    if not isinstance(filters, list) or not filters:
        raise PlanError("At least one filter is required.")
    clauses = []
    params = []
    for f in filters:
        if not isinstance(f, dict):
            raise PlanError("Each filter must be an object.")
        col = _quote_ident(_resolve_column(str(f.get("column") or ""), available_columns))
        op = str(f.get("operator") or "").strip()
        if op not in FILTER_OPERATORS:
            raise PlanError(f"Unsupported filter operator '{op}'.")
        if op == "contains":
            clauses.append(f"CAST({col} AS VARCHAR) ILIKE '%' || ? || '%'")
            params.append(f.get("value"))
        elif op == "equals":
            clauses.append(f"{col} = ?")
            params.append(f.get("value"))
        elif op == "not_equals":
            clauses.append(f"{col} != ?")
            params.append(f.get("value"))
        elif op == "greater_than":
            clauses.append(f"{col} > ?")
            params.append(f.get("value"))
        elif op == "less_than":
            clauses.append(f"{col} < ?")
            params.append(f.get("value"))
        elif op == "greater_than_equal":
            clauses.append(f"{col} >= ?")
            params.append(f.get("value"))
        elif op == "less_than_equal":
            clauses.append(f"{col} <= ?")
            params.append(f.get("value"))
        elif op == "between":
            clauses.append(f"{col} BETWEEN ? AND ?")
            params.extend([f.get("value"), f.get("value2")])
        else:
            raise PlanError(f"Operator '{op}' is not supported by the parameterized filter builder.")
    return f"SELECT * FROM {_quote_ident(table_name)} WHERE " + " AND ".join(clauses), params


def _build_filter_clause(f: dict, available_columns: list, table_name: str) -> str:
    col = _quote_ident(_resolve_column(f.get("column"), available_columns))
    op = f.get("operator")
    if op not in FILTER_OPERATORS:
        raise PlanError(f"Unsupported filter operator '{op}'.")
    value = f.get("value")

    if op == "equals":
        return f"{col} = {_sql_literal(value)}"
    if op == "not_equals":
        return f"{col} != {_sql_literal(value)}"
    if op == "contains":
        return f"{col} ILIKE '%' || {_sql_literal(value)} || '%'"
    if op == "greater_than":
        return f"{col} > {_sql_literal(value)}"
    if op == "less_than":
        return f"{col} < {_sql_literal(value)}"
    if op == "greater_than_equal":
        return f"{col} >= {_sql_literal(value)}"
    if op == "less_than_equal":
        return f"{col} <= {_sql_literal(value)}"
    if op == "between":
        value2 = f.get("value2")
        return f"{col} BETWEEN {_sql_literal(value)} AND {_sql_literal(value2)}"
    if op == "above_average":
        return f"{col} > (SELECT AVG({col}) FROM {table_name})"
    if op == "below_average":
        return f"{col} < (SELECT AVG({col}) FROM {table_name})"
    raise PlanError(f"Unhandled filter operator '{op}'.")


def _build_comparison(expr: str, operator: str, value) -> str:
    mapping = {
        "equals": "=", "not_equals": "!=",
        "greater_than": ">", "less_than": "<",
        "greater_than_equal": ">=", "less_than_equal": "<=",
    }
    if operator not in mapping:
        raise PlanError(f"Unsupported derived-column condition operator '{operator}'.")
    return f"{expr} {mapping[operator]} {_sql_literal(value)}"


def _build_derived_column(dc: dict, available_columns: list) -> tuple:
    """Builds a `CASE WHEN <window aggregate condition> THEN x ELSE y END AS alias`
    expression, e.g. for classifying rows by a per-group count/sum/etc before
    the main aggregation. Returns (alias, select_expr_sql).
    """
    alias = re.sub(r"\W+", "_", (dc.get("alias") or "derived_col").strip().lower()).strip("_")
    case = dc.get("case") or {}
    cond = case.get("condition") or {}

    wf = (cond.get("window_function") or "count").lower()
    if wf not in AGG_FUNCTIONS:
        raise PlanError(f"Unsupported derived-column window_function '{wf}'.")
    partition_by = [_resolve_column(c, available_columns) for c in cond.get("partition_by", []) or []]
    if not partition_by:
        raise PlanError("derived_columns condition requires at least one partition_by column.")
    partition_sql = f"PARTITION BY {', '.join(_quote_ident(c) for c in partition_by)}"

    if wf == "count":
        col = cond.get("column")
        col_expr = _quote_ident(_resolve_column(col, available_columns)) if col else "*"
        window_expr = f"COUNT({col_expr}) OVER ({partition_sql})"
    else:
        col = _resolve_column(cond.get("column"), available_columns)
        window_expr = f"{wf.upper()}({_quote_ident(col)}) OVER ({partition_sql})"

    condition_sql = _build_comparison(window_expr, cond.get("operator"), cond.get("value"))
    then_val = _sql_literal(case.get("then"))
    else_val = _sql_literal(case.get("else"))
    expr_sql = f"CASE WHEN {condition_sql} THEN {then_val} ELSE {else_val} END AS {_quote_ident(alias)}"
    return alias, expr_sql



def build_sql_from_plan(plan: dict, available_columns: list, table_name: str = TABLE_NAME) -> str:
    """Deterministically builds a single DuckDB SELECT statement from a
    structured plan. Every column reference is resolved against
    available_columns — nothing is trusted verbatim from the LLM.
    """
    if not plan:
        raise PlanError("Empty plan.")

    derived_columns = plan.get("derived_columns", []) or []
    source = table_name
    resolvable_columns = list(available_columns)

    if derived_columns:
        derived_select_parts = ["*"]
        for dc in derived_columns:
            alias, expr_sql = _build_derived_column(dc, resolvable_columns)
            derived_select_parts.append(expr_sql)
            resolvable_columns.append(alias)  # so group_by/metrics/order_by can reference it
        inner = f"SELECT {', '.join(derived_select_parts)} FROM {table_name}"
        source = f"({inner})"

    group_by = [_resolve_column(c, resolvable_columns) for c in plan.get("group_by", []) or []]
    metrics = plan.get("metrics", []) or []
    filters = plan.get("filters", []) or []
    window = plan.get("window")
    keep_top_n_per_partition = plan.get("keep_top_n_per_partition")
    order_by = plan.get("order_by", []) or []
    limit = plan.get("limit")

    select_parts = []
    alias_lookup = {}  # metric alias (lowercase) -> select expression, for order/window resolution

    for gc in group_by:
        select_parts.append(_quote_ident(gc))

    for m in metrics:
        func = (m.get("function") or "").lower()
        if func not in AGG_FUNCTIONS:
            raise PlanError(f"Unsupported aggregation function '{func}'.")
        col = _resolve_column(m.get("column"), resolvable_columns)
        alias = re.sub(r"\W+", "_", (m.get("alias") or f"{func}_{col}").strip().lower()).strip("_")
        expr = f"COUNT({_quote_ident(col)})" if func == "count" else f"{func.upper()}({_quote_ident(col)})"
        select_parts.append(f"{expr} AS {_quote_ident(alias)}")
        alias_lookup[alias.lower()] = expr

    if not group_by and not metrics:
        # No aggregation requested — plain row-level query.
        select_parts = ["*"]

    window_alias = None
    if window:
        wtype = (window.get("type") or "").lower()
        if wtype not in WINDOW_TYPES:
            raise PlanError(f"Unsupported window type '{wtype}'.")
        partition_by = [_resolve_column(c, resolvable_columns) for c in window.get("partition_by", []) or []]
        w_order = window.get("order_by", []) or []
        order_clauses = []
        for o in w_order:
            oc = o.get("column", "")
            direction = "DESC" if (o.get("direction") or "desc").lower() == "desc" else "ASC"
            expr = alias_lookup.get(str(oc).lower())
            if expr is None:
                expr = _quote_ident(_resolve_column(oc, resolvable_columns))
            order_clauses.append(f"{expr} {direction}")

        partition_sql = f"PARTITION BY {', '.join(_quote_ident(c) for c in partition_by)}" if partition_by else ""
        order_sql = f"ORDER BY {', '.join(order_clauses)}" if order_clauses else ""
        over_clause = " ".join(p for p in [partition_sql, order_sql] if p)

        if wtype == "rank":
            window_alias = "rnk"
            select_parts.append(f"RANK() OVER ({over_clause}) AS {window_alias}")
        elif wtype == "dense_rank":
            window_alias = "rnk"
            select_parts.append(f"DENSE_RANK() OVER ({over_clause}) AS {window_alias}")
        elif wtype == "running_total":
            if not metrics:
                raise PlanError("running_total requires at least one metric.")
            metric_expr = list(alias_lookup.values())[0]
            window_alias = "running_total"
            select_parts.append(
                f"SUM({metric_expr}) OVER ({order_sql} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {window_alias}"
            )
        elif wtype == "moving_average":
            if not metrics:
                raise PlanError("moving_average requires at least one metric.")
            size = int(window.get("window_size") or 3)
            metric_expr = list(alias_lookup.values())[0]
            window_alias = "moving_avg"
            select_parts.append(
                f"AVG({metric_expr}) OVER ({order_sql} ROWS BETWEEN {max(size - 1, 0)} PRECEDING AND CURRENT ROW) AS {window_alias}"
            )

    where_sql = ""
    if filters:
        clauses = [_build_filter_clause(f, resolvable_columns, source) for f in filters]
        where_sql = "WHERE " + " AND ".join(clauses)

    group_sql = f"GROUP BY {', '.join(_quote_ident(c) for c in group_by)}" if group_by and metrics else ""

    inner_sql = f"SELECT {', '.join(select_parts)} FROM {source} {where_sql} {group_sql}".strip()

    sql = inner_sql
    if keep_top_n_per_partition:
        if not window_alias:
            raise PlanError("keep_top_n_per_partition requires a window (rank/dense_rank) to filter on.")
        sql = f"SELECT * FROM ({inner_sql}) t WHERE {window_alias} <= {int(keep_top_n_per_partition)}"

    outer_order_clauses = []
    for o in order_by:
        oc = str(o.get("column", ""))
        direction = "DESC" if (o.get("direction") or "desc").lower() == "desc" else "ASC"
        if oc.lower() in alias_lookup:
            ref = _quote_ident(re.sub(r"\W+", "_", oc.strip().lower()).strip("_"))
        elif window_alias and oc.lower() == window_alias:
            ref = _quote_ident(window_alias)
        else:
            ref = _quote_ident(_resolve_column(oc, resolvable_columns))
        outer_order_clauses.append(f"{ref} {direction}")

    if outer_order_clauses:
        sql += f" ORDER BY {', '.join(outer_order_clauses)}"

    if limit:
        sql += f" LIMIT {int(limit)}"

    return sql


# ── Deterministic sentiment intent fast-path ─────────────────────────────────
#
# Sentiment requests are a well-defined product capability. Do not make the
# entire request depend on the general-purpose router LLM being available.
# This also prevents a temporary router/API failure from producing the
# misleading "Internal error while routing" message for a sentiment query.
def _detect_sentiment_intent(user_text: str, available_columns: list[str]) -> dict | None:
    text = (user_text or "").lower()
    review_cols = [
        c for c in available_columns
        if any(k in c.lower().replace("_", " ") for k in (
            "review text", "review", "comment", "feedback", "customer feedback", "content"
        ))
    ]
    if not review_cols:
        return None

    sentiment_terms = (
        "sentiment", "sentement", "customer satisfaction", "satisfaction",
        "satisfied", "dissatisfied", "happy customers", "unhappy customers",
        "review sentiment", "review tone", "analyze sentiment", "analyse sentiment",
        "how restaurants are performing", "restaurant performance based on review",
        "based on review", "based on reviews", "from reviews", "from review"
    )
    if not any(term in text for term in sentiment_terms):
        return None

    # Prefer an explicitly named review column; otherwise use the strongest
    # review-text candidate.
    explicit = [
        c for c in available_columns
        if c.lower() in text or c.lower().replace("_", " ") in text
    ]
    review_col = next((c for c in explicit if c in review_cols), None)
    if review_col is None:
        review_col = next((c for c in review_cols if "review" in c.lower()), review_cols[0])

    restaurant_cols = [
        c for c in available_columns
        if "restaurant" in c.lower() and "name" in c.lower()
    ]
    restaurant_col = restaurant_cols[0] if restaurant_cols else None
    return {
        "route": "sentiment",
        "plan": {"review_column": review_col, "restaurant_column": restaurant_col},
        "confidence": 0.98,
        "message": f"Analyzing customer sentiment from {review_col} and summarizing restaurant satisfaction."
    }


# ── Router agent call ─────────────────────────────────────────────────────────

async def _run_router_agent(user_text: str, available_columns: list, df: pd.DataFrame | None = None) -> dict:
    agent = LlmAgent(
        name="query_router_agent",
        model=MODEL,
        instruction=ROUTER_SYSTEM_INSTRUCTION,
        description="Routes a request to SQL or a spreadsheet operation, planning the SQL structurally.",
    )

    app_name = "query_router_app"
    user_id = "api_user"
    session_id = str(uuid.uuid4())

    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    safe_cols, _, reverse_cols = safe_columns(available_columns) if strict_enabled() else (available_columns, {}, {})
    value_forward, reverse_values = value_aliases(df) if strict_enabled() and df is not None else ({}, {})
    safe_text = sanitize_user_text(user_text, available_columns, df=df if strict_enabled() else None) if strict_enabled() else user_text
    prompt = (
        f"Available columns: {json.dumps(safe_cols)}\n"
        f"User request: {safe_text}"
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_text = None
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    final_text = part.text

    print(f"[query_router] raw model output: {final_text!r}")

    if not final_text:
        return {"route": "operation", "plan": None, "confidence": 0.0,
                "message": "No response from the router agent."}

    try:
        cleaned = _extract_json(final_text)
        parsed = json.loads(cleaned)
        return remap_plan(remap_plan(parsed, reverse_cols), reverse_values) if strict_enabled() else parsed
    except json.JSONDecodeError as e:
        print(f"[query_router] JSON parse failed: {e}. Cleaned text was: {cleaned!r}")
        return {"route": "operation", "plan": None, "confidence": 0.0,
                "message": "Could not parse the router's response as JSON."}


# ── SQL execution ──────────────────────────────────────────────────────────────

def _execute_sql(sql: str, df: pd.DataFrame, table_name: str = TABLE_NAME) -> dict:
    con = duckdb.connect(database=":memory:")
    try:
        con.register(table_name, df)
        result_df = con.execute(sql).df()
        result_df = result_df.where(pd.notnull(result_df), None)
        return {
            "columns": list(result_df.columns),
            "rows": result_df.to_dict(orient="records"),
            "row_count": len(result_df),
        }
    finally:
        con.close()


# ── Deterministic entity/filter resolver ─────────────────────────────────────
#
# Local-first: when a user asks to show/filter a named entity (e.g.
# "show me [restaurant] restaurant data"), do NOT ask the external router to
# guess a column or invent column_001.  The workbook itself is the source of
# truth, so resolve the entity locally against the actual dataframe and only
# then execute the filter.  This also guarantees that a value which does not
# exist is reported as not found rather than producing a misleading result.

def _entity_filter_intent(text: str) -> bool:
    return bool(re.search(
        r"\b(?:show|display|find|filter|give|list|fetch|view)\b.*\b(?:data|rows?|records?|restaurants?|customers?|products?|entries?|where)\b|"
        r"\bfilter\b|\bshow\s+me\b",
        text or "", re.I,
    ))


def _local_entity_filter_plan(user_text: str, df: pd.DataFrame):
    """Return (plan, message) for a value-driven show/filter request.

    The plan is built entirely from the local dataframe.  It never assumes a
    synthetic column name and uses the workbook data available to the analysis agent.
    """
    text = str(user_text or "").strip()
    if not text or not _entity_filter_intent(text):
        return None, None

    # Candidate phrases are contiguous n-grams from the user's request.  Long
    # phrases win, which makes values such as "Spice Restaurant 282" beat the
    # shorter "Restaurant" token.
    tokens = re.findall(r"[\w$€£₹₽৳.-]+", text, flags=re.UNICODE)
    stop = {"show", "me", "display", "find", "filter", "give", "list", "fetch", "view",
            "the", "a", "an", "data", "rows", "row", "records", "record", "entries",
            "entry", "where", "that", "with", "for", "only", "please", "from", "in"}
    # Keep domain words such as restaurant because they may be part of the
    # actual value ("Spice Restaurant 282").
    meaningful = [t for t in tokens if t.lower() not in stop]
    if not meaningful:
        return None, None

    phrases = []
    nmax = min(6, len(meaningful))
    for n in range(nmax, 0, -1):
        for i in range(len(meaningful) - n + 1):
            phrases.append(" ".join(meaningful[i:i+n]))

    # Explicit quoted text should be considered first and as a whole.
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
    phrases = quoted + phrases

    best = None
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            # Numeric columns can still contain IDs, but don't match arbitrary
            # natural-language phrases against them.
            continue
        vals = series.dropna().astype(str).str.strip()
        if vals.empty:
            continue
        # unique() avoids repeatedly scanning duplicated restaurant names.
        unique_vals = vals.unique().tolist()
        col_name = str(col)
        col_hint = 0
        if re.search(r"restaurant|name|title|customer|product|city|country|region", col_name, re.I):
            col_hint = 1

        for phrase in phrases:
            phrase = str(phrase).strip()
            if len(phrase) < 2:
                continue
            low = phrase.casefold()
            exact = [v for v in unique_vals if v.casefold() == low]
            if exact:
                value = exact[0]
                score = (3, len(low), col_hint)
                if best is None or score > best[0]:
                    best = (score, col_name, value, "equals")
                continue
            contains = [v for v in unique_vals if low in v.casefold()]
            if contains:
                # Prefer the shortest matching cell value when the user's
                # phrase is contained in a longer cell, unless the phrase
                # itself exactly describes the entity.
                value = min(contains, key=len)
                score = (2, len(low), col_hint)
                if best is None or score > best[0]:
                    best = (score, col_name, phrase, "contains")

    if best is None:
        return None, f"No matching data was found for the requested value in the current worksheet."

    _, col, value, operator = best
    plan = {
        "group_by": [],
        "metrics": [],
        "filters": [{"column": col, "operator": operator, "value": value}],
        "limit": None,
        "order_by": [],
    }
    return plan, f"Found '{value}' in '{col}' and filtered the matching rows."


# ── Main entry point ───────────────────────────────────────────────────────────

async def handle_smart_query(
    user_text: str,
    df: pd.DataFrame,
    available_sheets: list | None = None,
) -> dict:
    """Decides SQL vs spreadsheet-operation for a natural-language request.
    For "sql", the LLM only produces a structured plan; build_sql_from_plan
    (pure Python, no LLM) turns that into the actual query against whichever
    dataset is currently loaded — so the same plan format works regardless
    of what columns the dataset happens to have.
    """
    available_columns = list(df.columns)
    available_sheets = available_sheets or []

    # ── Deterministic entity/filter fast-path ───────────────────────────────
    # Resolve named values against the real worksheet before the LLM router.
    # This prevents hallucinated column_001 selections and ensures we only
    # filter when the requested entity actually exists.
    try:
        entity_plan, entity_message = _local_entity_filter_plan(user_text, df)
        if entity_plan is not None:
            sql = build_sql_from_plan(entity_plan, available_columns, TABLE_NAME)
            result = _execute_sql(sql, df)
            return to_json_safe(smart_query_envelope(
                success=True,
                route="sql",
                confidence=1.0,
                message=entity_message,
                plan=entity_plan,
                sql=sql,
                result=result,
            ))
        # A show/filter request that contained an entity-like phrase but no
        # actual match must not fall through to Gemini and invent a column.
        if entity_message and _entity_filter_intent(user_text):
            return to_json_safe(smart_query_envelope(
                success=True,
                route="sql",
                confidence=1.0,
                message=entity_message,
                plan=None,
                result={"columns": list(df.columns), "rows": [], "row_count": 0},
                warnings=[entity_message],
            ))
    except Exception as e:
        logger.exception("[query_router] local entity/filter resolution failed")
        return _operation_error_response(
            "Could not safely resolve the requested filter against the current worksheet.",
            error_type="LOCAL_FILTER_RESOLUTION_FAILED",
            exception=str(e),
        )

    # ── Deterministic fast-path: ALL registered transformations ─────────────
    # Rule-based (no LLM call) — routes through the SAME centralized
    # TransformationEngine main.py's /transform/apply uses (see
    # common/transformations/). Checked BEFORE the LLM router so requests
    # like "create salary bands", "group age into 0-18,...,60+", "rename
    # Sales to Revenue", "remove duplicate rows", etc. never round-trip
    # through Gemini and never get misclassified as a generic SQL ask.
    #
    # HARDENING: this was the one call in this whole function with no
    # try/except — every other branch below (the LLM router, SQL execution,
    # parse_agentic_command) already guards against exceptions, but a bug
    # here (already substantially reduced by the fixes now inside
    # TransformationEngine.run() itself) would have skipped straight past
    # all of those and out of this function entirely. Matches the same
    # print(...) + traceback.print_exc() pattern used by every other except
    # block in this file, so server logs stay consistent either way.
    try:
        transform_result = _transformation_engine.run(df, query=user_text, value_column=None)
    except Exception as e:
        logger.exception("[query_router] Exception in transformation engine fast-path")
        return _operation_error_response(
            "Transformation failed.", error_type="TRANSFORMATION_ENGINE_EXCEPTION", exception=str(e),
        )

    if transform_result.success:
        # HARDENING: transform_result.success being True only means apply()
        # itself worked — building the response dict below (dataframe ->
        # records, reading fields off ai_report) is separate code that can
        # still fail (e.g. ai_report degraded to {"error": ...} upstream in
        # TransformationEngine.run(), making ai_report.get(...) calls fine,
        # but a future change to what's stored there might not be). This
        # must not throw away an otherwise-successful transformation.
        try:
            new_df = transform_result.dataframe
            new_df_json = new_df.where(pd.notnull(new_df), None)
            ai_report = transform_result.updated_ai_report or {}
            envelope = smart_query_envelope(
                success=True,
                route="operation",
                confidence=transform_result.transformation.get("confidence", 0.8),
                message=transform_result.message,
                operation={
                    "action": transform_result.transformation.get("name"),
                    "metadata": transform_result.metadata,
                    "transformation": transform_result.transformation,
                    "preview": transform_result.preview,
                    "explanation": transform_result.metadata.get("explanation"),
                    "data": {
                        "columns": list(new_df.columns),
                        "rows": new_df_json.to_dict(orient="records"),
                        "row_count": len(new_df),
                    },
                    # Everything Flutter needs to refresh statistics/KPIs/charts/
                    # AI report/executive summary in this SAME response — no
                    # second dataset scan required.
                    "ai_report": ai_report,
                    "chart_recommendation": ai_report.get("chart_recommendation"),
                    "updated_schema": transform_result.updated_schema,
                },
                metadata=transform_result.metadata,
                preview=transform_result.preview,
                statistics=transform_result.updated_statistics,
                schema=transform_result.updated_schema,
                ai_report=ai_report,
            )
            # TASK 8 — global response validation: run every response through
            # to_json_safe() here, at the source, rather than trusting every
            # caller of handle_smart_query() to remember to do it. main.py's
            # /smart_query route also runs the final result through
            # to_json_safe() before handing it to JSONResponse; doing it here
            # too is intentionally redundant (to_json_safe() is idempotent on
            # already-safe values) so this function is safe to call from
            # anywhere, not just from behind that one route.
            return to_json_safe(envelope)
        except Exception as e:
            logger.exception("[query_router] Exception building the response for a successful transformation")
            return _operation_error_response(
                "The transformation succeeded, but the response could not be built.",
                error_type="RESPONSE_BUILD_FAILED",
                exception=str(e),
            )
    elif transform_result.error != "Could not locate a matching transformation for this request.":
        # A transformation WAS confidently detected but failed validation/
        # apply (bad column, unparsable ranges, etc) — surface that error
        # directly instead of silently falling through to the LLM router,
        # which is what the old range_binning-only fast path did too.
        return _operation_error_response(
            transform_result.error,
            error_type="TRANSFORMATION_VALIDATION_FAILED",
            confidence=0.5,
        )
    # else: nothing was confidently detected at all — continue with the
    # deterministic sentiment capability before invoking the general router.
    # This is intentionally before Gemini so a router outage cannot break
    # sentiment analysis.
    sentiment_decision = _detect_sentiment_intent(user_text, available_columns)
    if sentiment_decision is not None:
        decision = sentiment_decision
    else:
        try:
            decision = await _run_router_agent(user_text, available_columns, df)
        except Exception as e:
            logger.exception("[query_router] Exception during routing")
            return _operation_error_response(
                "Internal error while routing the query — check server logs.",
                error_type="ROUTING_FAILED", exception=str(e),
            )

    route = decision.get("route")
    confidence = decision.get("confidence", 0.0)
    message = decision.get("message", "")

    if route == "sentiment":
        try:
            review_col = None
            restaurant_col = None
            # The router may optionally provide explicit columns for sentiment.
            plan = decision.get("plan") or {}
            review_col = plan.get("review_column")
            restaurant_col = plan.get("restaurant_column")
            sentiment = await analyze_sentiment(df, review_column=review_col, restaurant_column=restaurant_col)
            overall = sentiment["overall"]
            return to_json_safe(smart_query_envelope(
                success=True,
                route="sentiment",
                confidence=confidence,
                message=message or "Analyzed customer sentiment from the review text in batches.",
                sentiment=sentiment,
            ))
        except Exception as e:
            logger.exception("[query_router] Exception during sentiment analysis")
            return to_json_safe(smart_query_envelope(
                success=False, route="sentiment", confidence=confidence,
                message=f"Sentiment analysis failed: {e}",
                errors=[{"error_type":"SENTIMENT_ANALYSIS_FAILED","message":str(e)}],
            ))

    if route == "sql":
        plan = decision.get("plan")
        try:
            sql = build_sql_from_plan(plan, available_columns)
        except PlanError as e:
            return to_json_safe(smart_query_envelope(
                success=False,
                route="sql",
                confidence=confidence,
                message=f"Could not build a valid query from the plan: {e}",
                plan=plan,
                errors=[{"error_type": "PLAN_BUILD_FAILED", "message": str(e)}],
            ))
        try:
            result = _execute_sql(sql, df)
        except Exception as e:
            logger.exception("[query_router] Exception during SQL execution")
            return to_json_safe(smart_query_envelope(
                success=False,
                route="sql",
                confidence=confidence,
                message=f"SQL execution failed: {e}",
                plan=plan,
                sql=sql,
                errors=[{"error_type": "SQL_EXECUTION_FAILED", "message": str(e)}],
            ))
        return to_json_safe(smart_query_envelope(
            success=True,
            route="sql",
            confidence=confidence,
            message=message,
            plan=plan,
            sql=sql,
            result=result,
        ))

    # route == "operation" (also the default fallback for anything unexpected)
    try:
        op_result = await parse_agentic_command(user_text, available_columns, available_sheets)
    except Exception as e:
        logger.exception("[query_router] Exception during operation parsing")
        return to_json_safe(smart_query_envelope(
            success=False,
            route="operation",
            confidence=confidence,
            message=f"Operation parsing failed: {e}",
            operation={"action": "transformation_error", "error_type": "OPERATION_PARSE_FAILED", "error": str(e)},
            errors=[{"error_type": "OPERATION_PARSE_FAILED", "message": str(e)}],
        ))

    return to_json_safe(smart_query_envelope(
        success=True,
        route="operation",
        confidence=confidence,
        message=message,
        operation=op_result,
    ))
