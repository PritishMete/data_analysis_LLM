# schema_intelligence/rules/primary_key.py
from ..contracts import ColumnContext, ColumnRule, RuleResult
from ._shared import name_hint_score

_IDENTITY_NAME_HINTS = ("id", "key", "uuid", "identifier", "pk")


class PrimaryKeyRule(ColumnRule):
    """A column is a strong primary-key candidate when it's fully unique and
    non-null. Confidence degrades gracefully rather than being a hard
    yes/no: a column that's 97% unique with a couple of dirty duplicate rows
    still deserves a high (not zero) score.

    Uniqueness alone is a WEAK signal on a small sample — a handful of
    unrelated numeric values (e.g. 3 different prices) will very often be
    coincidentally unique without meaning anything about identity. Full
    confidence therefore also requires identity-shaped naming ("id", "key",
    "uuid", ...); without it, confidence is halved — enough to still surface
    as a candidate, but not enough to out-rank a rule with genuine positive
    evidence (e.g. CurrencyColumnRule's name-hint match) for the same column.
    """

    name = "primary_key"

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence  # floor below which we report nothing at all

    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        if context.row_count <= 1:
            return None  # a 1-row table makes every column trivially "unique"

        series = context.series
        missing_count = int(series.isnull().sum())
        non_null_ratio = 1.0 - (missing_count / context.row_count)
        unique_count = int(series.nunique(dropna=True))
        uniqueness_ratio = unique_count / context.row_count

        has_identity_name = bool(name_hint_score(context.column_name, _IDENTITY_NAME_HINTS))
        base_confidence = uniqueness_ratio * non_null_ratio
        confidence = round(base_confidence if has_identity_name else base_confidence * 0.5, 4)

        if confidence < self.min_confidence:
            return None

        return RuleResult(
            role="primary_key",
            confidence=confidence,
            rule_name=self.name,
            evidence={
                "unique_count": unique_count,
                "row_count": context.row_count,
                "uniqueness_ratio": round(uniqueness_ratio, 4),
                "non_null_ratio": round(non_null_ratio, 4),
                "identity_shaped_name": has_identity_name,
            },
        )
