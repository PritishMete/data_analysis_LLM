# schema_intelligence/rules/email_column.py
import re

import pandas as pd

from ..contracts import ColumnContext, ColumnRule, RuleResult
from ._shared import match_ratio

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class EmailColumnRule(ColumnRule):
    name = "email"

    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        series = context.series
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return None

        ratio, sample_size = match_ratio(series, _EMAIL_RE)
        if sample_size == 0 or ratio == 0.0:
            return None

        return RuleResult(
            role="email",
            confidence=round(ratio, 4),
            rule_name=self.name,
            evidence={"match_ratio": round(ratio, 4), "sample_size": sample_size},
        )
