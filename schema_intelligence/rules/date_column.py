# schema_intelligence/rules/date_column.py
import pandas as pd

from ..contracts import ColumnContext, ColumnRule, RuleResult
from ._shared import name_hint_score, non_null_sample

_NAME_HINTS = ("date", "time", "created", "updated", "timestamp", "dob", "birthday")


class DateColumnRule(ColumnRule):
    name = "date"

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        series = context.series

        if pd.api.types.is_datetime64_any_dtype(series):
            return RuleResult(
                role="date",
                confidence=1.0,
                rule_name=self.name,
                evidence={"reason": "native datetime dtype"},
            )

        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return None

        sample = non_null_sample(series, max_n=200)
        if len(sample) == 0:
            return None

        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        parse_ratio = float(parsed.notna().mean())
        name_score = name_hint_score(context.column_name, _NAME_HINTS)

        # A name hint is corroborating evidence, not a requirement — a
        # column plainly full of parseable dates is still a date column
        # even if it's oddly named, just reported at a slightly lower
        # confidence than a name-confirmed one.
        confidence = round(parse_ratio * (1.0 if name_score else 0.85), 4)
        if confidence < self.min_confidence:
            return None

        return RuleResult(
            role="date",
            confidence=confidence,
            rule_name=self.name,
            evidence={"parse_ratio": round(parse_ratio, 4), "name_hint": bool(name_score), "sample_size": len(sample)},
        )
