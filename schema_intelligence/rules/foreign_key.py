# schema_intelligence/rules/foreign_key.py
# ─────────────────────────────────────────────────────────────────────────────
# This rule only produces a CANDIDATE score from naming convention — it
# never sees other datasets, so it can't confirm an actual reference.
# Confirmed foreign keys still come from relationship_detector.py's
# value-overlap comparison against other real datasets (see service.py,
# which upgrades a column from "foreign_key_candidate" to "foreign_key" only
# once a relationship is actually detected). Keeping these as two separate,
# honestly-labeled signals is more useful than pretending name-matching
# alone proves a reference exists.
# ─────────────────────────────────────────────────────────────────────────────

from ..contracts import ColumnContext, ColumnRule, RuleResult

_ID_SUFFIXES = ("_id", "_key", "_code")


class ForeignKeyCandidateRule(ColumnRule):
    name = "foreign_key_candidate"

    def __init__(self, base_confidence: float = 0.6):
        self.base_confidence = base_confidence

    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        col_name = str(context.column_name).strip().lower()

        if col_name == "id":
            return None  # bare "id" is almost always THIS table's own key

        # The table's own leading id-like column is conventionally its own
        # primary key, not a reference to something else (see
        # PrimaryKeyRule/relationship_detector for the fuller reasoning).
        first_column = context.dataframe.columns[0] if len(context.dataframe.columns) > 0 else None
        if context.column_name == first_column and (col_name.endswith("id") or col_name.endswith(_ID_SUFFIXES)):
            return None

        matches_suffix = col_name.endswith(_ID_SUFFIXES) or (col_name.endswith("id") and len(col_name) > 2)
        if not matches_suffix:
            return None

        return RuleResult(
            role="foreign_key_candidate",
            confidence=self.base_confidence,
            rule_name=self.name,
            evidence={"column_name": context.column_name, "reason": "id-like naming convention"},
        )
