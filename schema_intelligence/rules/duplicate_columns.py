# schema_intelligence/rules/duplicate_columns.py
# ─────────────────────────────────────────────────────────────────────────────
# The one DatasetRule in this delivery — everything else here judges a
# single column; this compares every PAIR of columns against each other,
# which is why it needs the DatasetRule contract (whole DataFrame in, not
# one Series) rather than ColumnRule.
# ─────────────────────────────────────────────────────────────────────────────

from itertools import combinations

import pandas as pd

from ..contracts import DatasetRule, RuleResult


def _row_aligned_equality_ratio(a: pd.Series, b: pd.Series) -> float:
    """Fraction of rows where a[i] == b[i] (as strings, so int 5 and float
    5.0 and "5" all compare equal — cross-dtype duplicates from a messy
    upload are still real duplicates). Returns 0.0 for mismatched lengths
    (shouldn't happen within one DataFrame, but stay defensive)."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    equal = a.reset_index(drop=True).astype(str) == b.reset_index(drop=True).astype(str)
    return float(equal.mean())


class DuplicateColumnsRule(DatasetRule):
    """Flags column pairs that are effectively the same data under a
    different header — e.g. an export that includes both "CustomerID" and
    an accidentally-copied "Customer_ID_2". `min_confidence` is the minimum
    row-aligned equality ratio required to report a pair at all; 1.0 would
    mean "only report EXACT duplicates," so the default below intentionally
    allows for a little dirty data (a handful of rows where one copy got a
    typo/reformat) while still requiring near-total agreement.
    """

    name = "duplicate_columns"

    def __init__(self, min_confidence: float = 0.98):
        self.min_confidence = min_confidence

    def evaluate(self, dataset_id: str, dataframe: pd.DataFrame) -> list[RuleResult]:
        results: list[RuleResult] = []
        columns = list(dataframe.columns)

        for col_a, col_b in combinations(columns, 2):
            ratio = _row_aligned_equality_ratio(dataframe[col_a], dataframe[col_b])
            if ratio < self.min_confidence:
                continue
            results.append(
                RuleResult(
                    role="duplicate_column",
                    confidence=round(ratio, 4),
                    rule_name=self.name,
                    evidence={
                        "column_a": str(col_a),
                        "column_b": str(col_b),
                        "row_equality_ratio": round(ratio, 4),
                    },
                )
            )

        return results
