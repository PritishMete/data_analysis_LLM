# schema_intelligence/rules/currency_column.py
import pandas as pd

from ..contracts import ColumnContext, ColumnRule, RuleResult
from ._shared import name_hint_score

_STRONG_HINTS = ("price", "amount", "cost", "revenue", "salary", "total", "payment")
_WEAK_HINTS = ("fee", "wage", "income", "expense", "balance", "currency")


class CurrencyColumnRule(ColumnRule):
    """Numeric dtype is a hard gate (currency values that survived upstream
    cleaning are numeric, never strings with symbols still attached — see
    data_cleaning_utils.py). Confidence itself comes entirely from how
    strongly the column NAME suggests a monetary quantity, since numeric
    values alone can't be distinguished from any other numeric measure."""

    name = "currency"

    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        series = context.series
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return None

        strong = name_hint_score(context.column_name, _STRONG_HINTS)
        weak = name_hint_score(context.column_name, _WEAK_HINTS)
        if strong == 0.0 and weak == 0.0:
            return None

        confidence = 0.85 if strong else 0.55
        return RuleResult(
            role="currency",
            confidence=confidence,
            rule_name=self.name,
            evidence={"name_hint_strength": "strong" if strong else "weak"},
        )
