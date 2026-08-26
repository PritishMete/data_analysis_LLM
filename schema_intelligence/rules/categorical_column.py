# schema_intelligence/rules/categorical_column.py
import pandas as pd

from ..contracts import ColumnContext, ColumnRule, RuleResult


class CategoricalColumnRule(ColumnRule):
    """Confidence is driven by how LOW the cardinality ratio is (few
    distinct values relative to row count = strongly categorical), capped by
    an absolute cardinality ceiling so a huge table with a genuinely huge
    number of small-ish categories doesn't get scored as confidently as a
    tiny, tight category set.
    """

    name = "category"

    def __init__(self, max_absolute_cardinality: int = 50, min_confidence: float = 0.3):
        self.max_absolute_cardinality = max_absolute_cardinality
        self.min_confidence = min_confidence

    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        series = context.series
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return None
        if context.row_count == 0:
            return None

        unique_count = int(series.nunique(dropna=True))
        if unique_count > self.max_absolute_cardinality:
            return None

        cardinality_ratio = unique_count / context.row_count
        confidence = round(max(0.0, 1.0 - cardinality_ratio), 4)
        if confidence < self.min_confidence:
            return None

        return RuleResult(
            role="category",
            confidence=confidence,
            rule_name=self.name,
            evidence={"unique_count": unique_count, "cardinality_ratio": round(cardinality_ratio, 4)},
        )
