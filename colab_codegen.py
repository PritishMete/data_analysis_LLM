# colab_codegen.py
# ─────────────────────────────────────────────────────────────────────────────
# Turns the SAME structured outputs your Excel add-in already gets from
# query_router.py (a SQL plan) and command_agent.py (a spreadsheet-action
# JSON) into literal, runnable Python source instead of executing anything
# server-side. Excel has no Python kernel, so those modules do the work and
# hand back a result/JSON. Colab already HAS a live kernel with the user's
# DataFrame in memory, so instead the right move is to hand back code the
# notebook can run itself, against the DataFrame that's already there.
#
# Nothing in query_router.py, command_agent.py, or cleaning_ops.py is
# modified — this module only reads their output shapes and renders text.
# ─────────────────────────────────────────────────────────────────────────────

import re


# ── small shared helpers ──────────────────────────────────────────────────

def _col(df_name: str, column: str) -> str:
    return f'{df_name}[{column!r}]'


def _num(df_name: str, column: str) -> str:
    return f'pd.to_numeric({_col(df_name, column)}, errors="coerce")'


def _lit(value) -> str:
    """Render a plan/action 'value' (always a string from the LLM layer) as
    a Python literal — numeric if it looks numeric, quoted string otherwise.
    """
    if value is None:
        return "None"
    s = str(value)
    try:
        if re.fullmatch(r"-?\d+", s):
            return s
        if re.fullmatch(r"-?\d+\.\d+", s):
            return s
    except Exception:
        pass
    return repr(s)


def _expr_to_pandas(expression: str, df_name: str, available_columns: list) -> str:
    """Rewrites an arithmetic expression written in real column names (as
    produced by command_agent's add_column.formula.rightExpression, e.g.
    "UnitPrice * (1 - DiscountPct/100)") into a pandas expression, e.g.
    df["UnitPrice"] * (1 - df["DiscountPct"]/100).

    Column names are matched case-insensitively against available_columns
    (same matching contract command_agent's prompt already asks the LLM to
    respect), so this stays correct regardless of the dataset's casing.
    """
    lookup = {c.lower(): c for c in available_columns}
    # Tokenize on identifiers vs everything else (operators/numbers/parens).
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^A-Za-z_]+", expression)
    out = []
    for tok in tokens:
        key = tok.strip().lower()
        if key in lookup:
            out.append(_col(df_name, lookup[key]))
        else:
            out.append(tok)
    return "".join(out).strip()


_FILTER_COMPARATORS = {
    "greater_than": ">",
    "less_than": "<",
    "greater_than_equal": ">=",
    "less_than_equal": "<=",
}


# ── FILTER ───────────────────────────────────────────────────────────────

def _gen_filter(action: dict, df_name: str) -> str:
    f = action.get("filter") or {}
    col = f.get("columnName")
    ftype = f.get("type")
    value = f.get("value")
    value2 = f.get("value2")

    if ftype == "equals":
        line = f'{df_name} = {df_name}[{_col(df_name, col)}.astype(str) == {str(value)!r}]'
    elif ftype == "not_equals":
        line = f'{df_name} = {df_name}[{_col(df_name, col)}.astype(str) != {str(value)!r}]'
    elif ftype == "contains":
        line = (f'{df_name} = {df_name}[{_col(df_name, col)}.astype(str)'
                 f'.str.contains({_lit(value)}, case=False, na=False)]')
    elif ftype in _FILTER_COMPARATORS:
        op = _FILTER_COMPARATORS[ftype]
        line = f'{df_name} = {df_name}[{_num(df_name, col)} {op} {_lit(value)}]'
    elif ftype == "between":
        line = (f'{df_name} = {df_name}[{_num(df_name, col)}.between'
                 f'({_lit(value)}, {_lit(value2)})]')
    elif ftype == "above_average":
        line = (f'{df_name} = {df_name}[{_num(df_name, col)} > '
                 f'{_num(df_name, col)}.mean()]')
    elif ftype == "below_average":
        line = (f'{df_name} = {df_name}[{_num(df_name, col)} < '
                 f'{_num(df_name, col)}.mean()]')
    elif ftype == "top_n":
        line = f'{df_name} = {df_name}.nlargest({int(float(value or 10))}, {col!r})'
    elif ftype == "bottom_n":
        line = f'{df_name} = {df_name}.nsmallest({int(float(value or 10))}, {col!r})'
    else:
        line = f'# Unrecognized filter type "{ftype}" — edit this manually.'

    return f'{line}\n{df_name}.reset_index(drop=True, inplace=True)\n{df_name}'


# ── DEDUPLICATE ─────────────────────────────────────────────────────────

def _gen_deduplicate(action: dict, df_name: str) -> str:
    d = action.get("deduplicate") or {}
    subset = d.get("columns")
    subset_arg = repr(subset) if subset else "None"
    return (
        f'{df_name} = {df_name}.drop_duplicates(subset={subset_arg}, keep="first")'
        f'.reset_index(drop=True)\n{df_name}'
    )


# ── COLOR SCALE ──────────────────────────────────────────────────────────

def _gen_color_scale(action: dict, df_name: str) -> str:
    cs = action.get("color_scale") or {}
    col = cs.get("column")
    return (
        f'# Excel conditional-colour-scale formatting has no direct pandas equivalent —\n'
        f'# closest analog in a notebook is a styled DataFrame with a colour gradient:\n'
        f'{df_name}.style.background_gradient(subset=[{col!r}], cmap="RdYlGn")'
    )


# ── ADD_COLUMN ───────────────────────────────────────────────────────────

_WINDOW_FUNC_MAP = {"count": "count", "sum": "sum", "avg": "mean", "min": "min", "max": "max"}
_COND_COMPARATORS = {
    "equals": "==", "not_equals": "!=",
    "greater_than": ">", "less_than": "<",
    "greater_than_equal": ">=", "less_than_equal": "<=",
}


def _gen_add_column(action: dict, df_name: str, available_columns: list) -> str:
    ac = action.get("add_column") or {}
    new_col = ac.get("newColumnName", "New_Column")
    condition = ac.get("condition")
    formula = ac.get("formula")

    if condition:
        window_fn = _WINDOW_FUNC_MAP.get(condition.get("windowFunction", "count"), "count")
        agg_col = condition.get("column")
        partition_by = condition.get("partitionBy") or [agg_col]
        op = _COND_COMPARATORS.get(condition.get("operator", "greater_than"), ">")
        value = condition.get("value")
        then_label = ac.get("thenLabel", "Yes")
        else_label = ac.get("elseLabel", "No")

        return (
            f'import numpy as np\n\n'
            f'_grp = {df_name}.groupby({partition_by!r})[{agg_col!r}].transform({window_fn!r})\n'
            f'{df_name}[{new_col!r}] = np.where(_grp {op} {_lit(value)}, '
            f'{then_label!r}, {else_label!r})\n{df_name}'
        )

    if formula:
        right_expr = _expr_to_pandas(formula.get("rightExpression", ""), df_name, available_columns)
        mode = formula.get("mode", "compute")

        if mode == "compute":
            return f'{df_name}[{new_col!r}] = {right_expr}\n{df_name}'

        left_expr_raw = formula.get("leftExpression")
        left_expr = (_expr_to_pandas(left_expr_raw, df_name, available_columns)
                     if left_expr_raw else None)
        tolerance = formula.get("tolerance")
        then_label = ac.get("thenLabel", "Match")
        else_label = ac.get("elseLabel", "Mismatch")

        if left_expr is None:
            return f'# add_column formula in "compare" mode is missing leftExpression — edit manually.'

        if tolerance:
            comparison = f'np.isclose({left_expr}, ({right_expr}), atol={tolerance})'
        else:
            comparison = f'({left_expr}) == ({right_expr})'

        return (
            f'import numpy as np\n\n'
            f'{df_name}[{new_col!r}] = np.where({comparison}, '
            f'{then_label!r}, {else_label!r})\n{df_name}'
        )

    return '# add_column had neither "condition" nor "formula" set — edit manually.'


# ── FILL_MISSING ─────────────────────────────────────────────────────────

def _gen_fill_missing(action: dict, df_name: str) -> str:
    fm = action.get("fill_missing") or {}
    col = fm.get("column")
    strategy = (fm.get("strategy") or "mean").lower()

    if strategy == "mean":
        return f'{df_name}[{col!r}] = {df_name}[{col!r}].fillna({df_name}[{col!r}].mean())\n{df_name}'
    if strategy == "median":
        return f'{df_name}[{col!r}] = {df_name}[{col!r}].fillna({df_name}[{col!r}].median())\n{df_name}'
    if strategy in ("mode", "auto"):
        return (
            f'_mode = {df_name}[{col!r}].mode(dropna=True)\n'
            f'if not _mode.empty:\n'
            f'    {df_name}[{col!r}] = {df_name}[{col!r}].fillna(_mode.iloc[0])\n{df_name}'
        )
    if strategy == "backtrack":
        src = fm.get("sourceFormulaColumn")
        return (
            f'# "backtrack" asks to re-derive missing {col!r} from the formula stored in\n'
            f'# {src!r} — that formula is specific to your data, so it isn\'t auto-generated.\n'
            f'# Fill in the inverse calculation here, e.g.:\n'
            f'# {df_name}.loc[{df_name}[{col!r}].isna(), {col!r}] = <expression using {src!r}>'
        )
    return f'# Unrecognized fill strategy "{strategy}" — edit this manually.'


# ── PIVOT ────────────────────────────────────────────────────────────────

_PIVOT_OP_MAP = {"sum": "sum", "average": "mean", "count": "count", "min": "min", "max": "max"}


def _gen_pivot(action: dict, df_name: str) -> str:
    p = action.get("pivot") or {}
    row_fields = p.get("rowFields") or []
    value_fields = p.get("valueFields") or []

    agg_map = {vf["field"]: _PIVOT_OP_MAP.get(vf.get("op", "sum"), "sum") for vf in value_fields}
    values = list(agg_map.keys())

    return (
        f'pivot_df = pd.pivot_table(\n'
        f'    {df_name},\n'
        f'    index={row_fields!r},\n'
        f'    values={values!r},\n'
        f'    aggfunc={agg_map!r},\n'
        f').reset_index()\n'
        f'pivot_df'
    )


# ── MULTI_STEP ───────────────────────────────────────────────────────────

_STRATEGY_METHOD = {"mean": "mean()", "median": "median()", "forward_fill": None}


def _gen_multi_step(action: dict, df_name: str) -> str:
    ms = action.get("multi_step") or {}
    steps = ms.get("steps") or []
    lines = []

    for i, step in enumerate(steps, start=1):
        op = step.get("op")
        lines.append(f'# Step {i}: {op}')

        if op == "standardize_columns":
            lines.append(
                f'{df_name}.columns = [str(c).strip().lower().replace(" ", "_") '
                f'for c in {df_name}.columns]'
            )

        elif op == "filter_rows":
            col, operator, value = step.get("column"), step.get("operator"), step.get("value")
            if operator in _FILTER_COMPARATORS:
                py_op = _FILTER_COMPARATORS[operator]
                lines.append(f'{df_name} = {df_name}[{_num(df_name, col)} {py_op} {_lit(value)}]')
            elif operator == "equals":
                lines.append(f'{df_name} = {df_name}[{_col(df_name, col)}.astype(str) == {str(value)!r}]')
            elif operator == "not_equals":
                lines.append(f'{df_name} = {df_name}[{_col(df_name, col)}.astype(str) != {str(value)!r}]')
            elif operator == "contains":
                lines.append(
                    f'{df_name} = {df_name}[{_col(df_name, col)}.astype(str)'
                    f'.str.contains({_lit(value)}, case=False, na=False)]'
                )
            else:
                lines.append(f'# Unrecognized filter operator "{operator}" — edit manually.')

        elif op == "handle_missing_values":
            strategy = (step.get("strategy") or "smart").lower()
            cols = step.get("columns")
            target = f'{df_name}[{cols!r}]' if cols else df_name
            if strategy == "mean":
                lines.append(f'{target} = {target}.fillna({target}.mean(numeric_only=True))')
            elif strategy == "median":
                lines.append(f'{target} = {target}.fillna({target}.median(numeric_only=True))')
            elif strategy == "mode":
                lines.append(f'{target} = {target}.apply(lambda s: s.fillna(s.mode().iloc[0]) '
                              f'if not s.mode().empty else s)')
            elif strategy == "forward_fill":
                lines.append(f'{target} = {target}.ffill()')
            elif strategy == "smart":
                lines.append(
                    f'{target} = {target}.apply(lambda s: s.fillna(s.median()) '
                    f'if pd.api.types.is_numeric_dtype(s) '
                    f'else (s.fillna(s.mode().iloc[0]) if not s.mode().empty else s))'
                )
            else:  # "drop"
                lines.append(f'{df_name} = {df_name}.dropna(subset={cols!r})' if cols
                              else f'{df_name} = {df_name}.dropna()')

        elif op == "remove_duplicates":
            subset = step.get("subset")
            subset_arg = repr(subset) if subset else "None"
            lines.append(f'{df_name} = {df_name}.drop_duplicates(subset={subset_arg}, keep="first")')

        elif op == "normalize_text":
            lines.append(
                f'for _c in {df_name}.select_dtypes(include="object").columns:\n'
                f'    {df_name}[_c] = {df_name}[_c].astype(str).str.strip()'
            )

        elif op == "handle_outliers":
            method = step.get("method", "cap")
            lines.append(
                f'for _c in {df_name}.select_dtypes(include="number").columns:\n'
                f'    q1, q3 = {df_name}[_c].quantile([0.25, 0.75])\n'
                f'    iqr = q3 - q1\n'
                f'    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr\n'
            )
            if method == "remove":
                lines.append(f'    {df_name} = {df_name}[({df_name}[_c] >= lo) & ({df_name}[_c] <= hi)]')
            else:  # cap or mark both shown as capping here; "mark" would need a flag column
                lines.append(f'    {df_name}[_c] = {df_name}[_c].clip(lower=lo, upper=hi)')

        elif op == "infer_types":
            lines.append(
                f'for _c in {df_name}.columns:\n'
                f'    {df_name}[_c] = pd.to_numeric({df_name}[_c], errors="ignore")'
            )

        elif op == "remove_empty_rows":
            lines.append(f'{df_name} = {df_name}.dropna(how="all")')

        else:
            lines.append(f'# Unknown step op "{op}" — edit manually.')

        lines.append("")

    lines.append(f'{df_name}.reset_index(drop=True, inplace=True)')
    lines.append(df_name)
    return "\n".join(lines)


# ── SQL plan (query_router.py) ────────────────────────────────────────────

def gen_sql_code(sql: str, df_name: str) -> str:
    """query_router.build_sql_from_plan() always emits SQL against a table
    literally named "data" (query_router.TABLE_NAME). DuckDB's Python API
    resolves bare table names in a query against local/global variables of
    that name, so aliasing the real DataFrame to `data` right before the
    query lets the generated SQL run unmodified against Colab's DataFrame.
    """
    return (
        f'import duckdb\n\n'
        f'data = {df_name}  # alias so the query below can reference it as "data"\n\n'
        f'query = """\n{sql}\n"""\n\n'
        f'result = duckdb.query(query).df()\n'
        f'result'
    )


# ── operation dispatch (command_agent.py) ─────────────────────────────────

_ACTION_GENERATORS = {
    "filter": _gen_filter,
    "deduplicate": _gen_deduplicate,
    "color_scale": _gen_color_scale,
    "fill_missing": _gen_fill_missing,
    "pivot": _gen_pivot,
}


def gen_operation_code(action: dict, df_name: str, available_columns: list) -> str:
    kind = action.get("action")
    if kind == "add_column":
        return _gen_add_column(action, df_name, available_columns)
    if kind == "multi_step":
        return _gen_multi_step(action, df_name)
    gen = _ACTION_GENERATORS.get(kind)
    if gen is None:
        return (
            f'# Could not confidently match this request to a known operation.\n'
            f'# {action.get("message", "")}'
        )
    return gen(action, df_name)
