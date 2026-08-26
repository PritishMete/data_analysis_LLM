# schema_intelligence/rules/percentage_column.py
import pandas as pd

from ..contracts import ColumnContext, ColumnRule, RuleResult
from ._shared import name_hint_score, non_null_sample

_NAME_HINTS = ("pct", "percent", "percentage", "%", "rate", "ratio", "discount")


class PercentageColumnRule(ColumnRule):
    name = "percentage"

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        series = context.series
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return None

        name_score = name_hint_score(context.column_name, _NAME_HINTS)
        if name_score == 0.0:
            return None  # numeric range alone is too ambiguous without a name hint

        sample = non_null_sample(series)
        if len(sample) == 0:
            return None

        in_fraction_range = float(((sample >= 0) & (sample <= 1)).mean())
        in_percent_range = float(((sample >= 0) & (sample <= 100)).mean())
        range_conformance = max(in_fraction_range, in_percent_range)

        confidence = round(name_score * range_conformance, 4)
        if confidence < self.min_confidence:
            return None

        return RuleResult(
            role="percentage",
            confidence=confidence,
            rule_name=self.name,
            evidence={
                "range_conformance": round(range_conformance, 4),
                "scale": "0-1" if in_fraction_range >= in_percent_range else "0-100",
            },
        )
