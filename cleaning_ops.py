# cleaning_ops.py
# ─────────────────────────────────────────────────────────────────────────────
# Sequential, agent-driven data-cleaning executor.
#
# Wraps the reusable functions in data_cleaning_utils.py (fill_nulls,
# clean_headers, convert_currency) plus two extra structural ops
# (remove_duplicates, filter_rows) behind a single ORDERED "steps" list:
#
#   steps = [
#       {"op": "fill_nulls", "columns": ["age"], "method": "mean"},
#       {"op": "clean_headers", "case": "lower", "replace_spaces_with": "_"},
#       {"op": "convert_currency", "column": "salary",
#        "from_currency": "USD", "to_currency": "INR", "rate": 83.5},
#   ]
#
# This is the EXACT shape already anticipated by `CleaningConfig.steps` in
# data_cleaning_service.dart — it's what cleaning_agent.py produces from a
# natural-language query. run_steps() executes them ONE AT A TIME against a
# running DataFrame, feeding each step's output into the next, so a
# multi-part instruction ("fill nulls in age with mean, then lowercase
# headers, then convert salary usd to inr") is applied — and reported — in
# the exact order the user described it.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

from data_cleaning_utils import fill_nulls, clean_headers, convert_currency


class StepError(Exception):
    """Raised for a malformed/unresolvable step. Caught per-step by
    run_steps so one bad step never aborts the whole pipeline."""


# ── Column resolution ─────────────────────────────────────────────────────
# Columns are re-resolved against the CURRENT dataframe at each step (not
# the original one the agent saw), so a header-casing step earlier in the
# pipeline can't break a later step that references the old casing.

def _resolve_column(name, available_columns):
    if not name:
        return None
    name_l = str(name).strip().lower()
    for col in available_columns:
        if str(col).strip().lower() == name_l:
            return col
    candidates = [c for c in available_columns
                  if name_l in str(c).lower() or str(c).lower() in name_l]
    return candidates[0] if candidates else None


def _resolve_columns(names, available_columns):
    if names is None:
        return None, []
    if isinstance(names, str):
        names = [names]
    resolved, missing = [], []
    for n in names:
        r = _resolve_column(n, available_columns)
        if r:
            resolved.append(r)
        else:
            missing.append(n)
    return resolved, missing


# ── Individual step handlers ────────────────────────────────────────────────
# Each takes (df, step_params) -> (new_df, step_report_dict)

def _op_clean_headers(df, p):
    before = list(df.columns)
    out = clean_headers(
        df,
        case=p.get("case", "lower"),
        trim=p.get("trim", True),
        replace_spaces_with=p.get("replace_spaces_with"),
    )
    return out, {
        "op": "clean_headers",
        "before_columns": before,
        "after_columns": list(out.columns),
    }


def _op_fill_nulls(df, p):
    cols = p.get("columns")
    resolved, missing = _resolve_columns(cols, df.columns) if cols else (None, [])
    target_cols = resolved if resolved else cols
    method = p.get("method", "mean")

    scope = [c for c in (target_cols or df.columns) if c in df.columns]
    nulls_before = {c: int(df[c].isnull().sum()) for c in scope}

    out = fill_nulls(df, columns=target_cols, method=method,
                      custom_value=p.get("custom_value"))

    nulls_after = {c: int(out[c].isnull().sum()) for c in scope if c in out.columns}
    cells_filled = sum(nulls_before.get(c, 0) - nulls_after.get(c, 0) for c in scope)

    return out, {
        "op": "fill_nulls",
        "columns": target_cols or "all columns with nulls",
        "method": method,
        "missing_columns_skipped": missing,
        "cells_filled": cells_filled,
        "nulls_before": nulls_before,
        "nulls_after": nulls_after,
    }


def _op_convert_currency(df, p):
    requested = p.get("column")
    col = _resolve_column(requested, df.columns)
    if not col:
        raise StepError(f"convert_currency: column '{requested}' not found.")
    from_currency = p.get("from_currency", "USD")
    to_currency = p.get("to_currency", "INR")
    out = convert_currency(
        df,
        column=col,
        from_currency=from_currency,
        to_currency=to_currency,
        rate=p.get("rate"),
        new_column=p.get("new_column"),
        use_live_rate=p.get("use_live_rate", False),
    )
    out_col = p.get("new_column") or f"{col}_{to_currency.upper()}"
    return out, {
        "op": "convert_currency",
        "column": col,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "rate_used": p.get("rate"),
        "output_column": out_col,
    }


def _op_remove_duplicates(df, p):
    subset = p.get("subset")
    resolved = None
    if subset:
        resolved, _missing = _resolve_columns(subset, df.columns)
        resolved = resolved or None
    before = len(df)
    out = df.drop_duplicates(subset=resolved, keep=p.get("keep", "first")).reset_index(drop=True)
    return out, {
        "op": "remove_duplicates",
        "subset": resolved or "all columns",
        "rows_before": before,
        "rows_after": len(out),
        "rows_removed": before - len(out),
    }


def _op_filter_rows(df, p):
    requested = p.get("column")
    col = _resolve_column(requested, df.columns)
    if not col:
        raise StepError(f"filter_rows: column '{requested}' not found.")

    operator = p.get("operator", "equals")
    value = p.get("value")
    series = df[col]
    before = len(df)

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if operator == "equals":
        mask = series.astype(str).str.strip() == str(value)
    elif operator == "not_equals":
        mask = series.astype(str).str.strip() != str(value)
    elif operator in ("greater_than", "less_than", "greater_than_equal", "less_than_equal"):
        num = _num(value)
        if num is None:
            raise StepError(f"filter_rows: value '{value}' is not numeric for '{operator}'.")
        numeric_series = pd.to_numeric(series, errors="coerce")
        mask = {
            "greater_than": numeric_series > num,
            "less_than": numeric_series < num,
            "greater_than_equal": numeric_series >= num,
            "less_than_equal": numeric_series <= num,
        }[operator]
    elif operator == "contains":
        mask = series.astype(str).str.contains(str(value), case=False, na=False)
    else:
        raise StepError(f"filter_rows: unsupported operator '{operator}'.")

    out = df[mask].reset_index(drop=True)
    return out, {
        "op": "filter_rows",
        "column": col,
        "operator": operator,
        "value": value,
        "rows_before": before,
        "rows_after": len(out),
        "rows_removed": before - len(out),
    }


# Aliases match the naming already used elsewhere in the codebase
# (data_cleaning_service.dart's `steps` doc-comment, data_cleaner.py's config
# keys) so either naming convention works without the agent needing to know
# which backend module ends up handling it.
_OP_HANDLERS = {
    "clean_headers": _op_clean_headers,
    "standardize_columns": _op_clean_headers,
    "standardize_cols": _op_clean_headers,
    "fill_nulls": _op_fill_nulls,
    "handle_missing_values": _op_fill_nulls,
    "convert_currency": _op_convert_currency,
    "remove_duplicates": _op_remove_duplicates,
    "filter_rows": _op_filter_rows,
}


def run_steps(df: pd.DataFrame, steps: list) -> tuple:
    """Executes an ordered list of cleaning steps ONE AT A TIME, feeding each
    step's output dataframe into the next.

    Returns (final_df, report). report["steps"] lists each step's outcome in
    the order it ran. A step that's malformed or references a column that
    can't be resolved is SKIPPED (status "skipped") rather than aborting the
    remaining steps — so "fill nulls in age, then dedupe, then fix a typo'd
    column name" still completes the two valid steps and reports the third
    was skipped, with why.
    """
    current = df.copy()
    step_reports = []

    for i, raw_step in enumerate(steps):
        op = raw_step.get("op")
        handler = _OP_HANDLERS.get(op)
        if handler is None:
            step_reports.append({
                "step": i + 1, "op": op, "status": "skipped",
                "error": f"Unknown op '{op}'.",
            })
            continue
        try:
            current, info = handler(current, raw_step)
            info["step"] = i + 1
            info["status"] = "done"
            step_reports.append(info)
        except StepError as e:
            step_reports.append({"step": i + 1, "op": op, "status": "skipped", "error": str(e)})
        except Exception as e:
            step_reports.append({"step": i + 1, "op": op, "status": "error", "error": str(e)})

    done = sum(1 for s in step_reports if s["status"] == "done")
    report = {
        "summary": f"Ran {len(steps)} step(s): {done} succeeded, {len(steps) - done} skipped/errored.",
        "rows_before": len(df),
        "rows_after": len(current),
        "columns_before": list(df.columns),
        "columns_after": list(current.columns),
        "cells_filled": sum(s.get("cells_filled", 0) for s in step_reports if s["status"] == "done"),
        "rows_removed": len(df) - len(current) if len(current) <= len(df) else 0,
        "steps": step_reports,
    }
    return current, report
