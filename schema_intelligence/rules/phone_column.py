# schema_intelligence/rules/phone_column.py
import re

import pandas as pd

from ..contracts import ColumnContext, ColumnRule, RuleResult
from ._shared import match_ratio, name_hint_score

_PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,20}$")
_NAME_HINTS = ("phone", "mobile", "contact_number", "tel")


class PhoneColumnRule(ColumnRule):
    """Requires BOTH a plausible column name AND a pattern match — a bare
    digit-pattern regex alone would false-positive against things like zip
    codes or numeric IDs formatted with dashes."""

    name = "phone"

    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        series = context.series
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return None

        name_score = name_hint_score(context.column_name, _NAME_HINTS)
        if name_score == 0.0:
            return None

        ratio, sample_size = match_ratio(series, _PHONE_RE)
        if sample_size == 0 or ratio == 0.0:
            return None

        confidence = round(ratio * name_score, 4)
        return RuleResult(
            role="phone",
            confidence=confidence,
            rule_name=self.name,
            evidence={"match_ratio": round(ratio, 4), "name_hint": True, "sample_size": sample_size},
        )
