"""Offline tests for the generic agentic filter-plan architecture.

These tests never call the LLM. They prove the common Pizza Hut / Domino's
request shape produces the same generic plan shape with only the entity value
changed, and that the backend parameterized SQL builder does not interpolate
user values into SQL syntax.
"""

import pandas as pd

from command_agent import _deterministic_filter_plan
from query_router import build_parameterized_filter_sql


COLUMNS = [
    "Restaurant Name",
    "Has Online delivery",
    "Has Table booking",
    "Aggregate rating",
    "City",
]


def test_equivalent_restaurant_queries_have_identical_structure():
    pizza = _deterministic_filter_plan(
        "show me Pizza Hut restaurants having online delivery and online table booking having rating more than 3.5",
        COLUMNS,
    )
    dominos = _deterministic_filter_plan(
        "show me Domino's Pizza restaurants having online delivery and online table booking having rating more than 3.5",
        COLUMNS,
    )

    assert pizza is not None and dominos is not None
    assert [(f["column"], f["operator"]) for f in pizza["filters"]] == [
        (f["column"], f["operator"]) for f in dominos["filters"]
    ]
    assert pizza["filters"][0]["value"] == "Pizza Hut"
    assert dominos["filters"][0]["value"] == "Domino's Pizza"
    assert pizza["filters"][1]["value"] is True
    assert pizza["filters"][2]["value"] is True
    assert pizza["filters"][3]["value"] == "3.5"


def test_parameterized_filter_sql_separates_values():
    filters = [
        {"column": "Restaurant Name", "operator": "contains", "value": "Domino's Pizza"},
        {"column": "Has Online delivery", "operator": "equals", "value": True},
        {"column": "Aggregate rating", "operator": "greater_than", "value": 3.5},
    ]
    sql, params = build_parameterized_filter_sql(filters, COLUMNS)

    assert "Domino's Pizza" not in sql
    assert "?" in sql
    assert params == ["Domino's Pizza", True, 3.5]


if __name__ == "__main__":
    test_equivalent_restaurant_queries_have_identical_structure()
    test_parameterized_filter_sql_separates_values()
    print("ALL GENERIC FILTER PLAN TESTS PASSED")
