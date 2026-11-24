"""Validate the small JSON language emitted by the personal finance adapter."""
from math import isfinite


def normalize_finance_plan(raw: dict) -> dict:
    plan = dict(raw)
    condition = plan.get("where")
    if condition not in (
        {"column": "Transaction Type", "operator": "eq", "value": "Income"},
        {"column": "Transaction Type", "operator": "eq", "value": "Expense"},
    ) or plan.get("column") != "Amount":
        raise ValueError("Expected an income or expense measure on Amount")

    op = plan.get("operation")
    if op == "aggregate" and plan.get("frequency") == "month" and plan.get("group_by") == "Date":
        op = plan["operation"] = "trend"
    keys = {"operation", "column", "where"}
    if op == "filter":
        keys.update(("operator", "value"))
        value = plan.get("value")
        if plan.get("operator") not in ("gt", "lt") or type(value) not in (int, float) or not isfinite(value):
            raise ValueError("Invalid finance threshold")
    elif op in ("aggregate", "rank", "trend"):
        keys.update(("group_by", "function"))
        if plan.get("function") not in ("sum", "mean"):
            raise ValueError("Unsupported finance statistic")
        if op == "trend":
            keys.add("frequency")
            if plan.get("frequency") != "month" or plan.get("group_by") != "Date":
                raise ValueError("Expected a calendar month trend")
        elif plan.get("group_by") not in ("Category", "Sub category"):
            raise ValueError("Unsupported grouping")
        if op == "rank":
            keys.update(("direction", "limit"))
            if plan.get("direction") not in ("asc", "desc") or type(plan.get("limit")) is not int or plan["limit"] <= 0:
                raise ValueError("Invalid ranking")
    else:
        raise ValueError("Unsupported finance operation")
    if keys != plan.keys():
        raise ValueError("Unexpected finance plan fields")
    return plan
