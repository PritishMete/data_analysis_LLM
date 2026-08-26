# schema_intelligence/rules/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
# TO ADD A NEW DETECTION CAPABILITY:
#   1. Write a new file in this folder implementing ColumnRule or
#      DatasetRule (see contracts.py).
#   2. Import it below and register an instance.
# Nothing in registry.py, service.py, or any OTHER rule needs to change.
# That's the entire extensibility story for this engine.
# ─────────────────────────────────────────────────────────────────────────────

from ..registry import register_column_rule, register_dataset_rule
from .categorical_column import CategoricalColumnRule
from .currency_column import CurrencyColumnRule
from .date_column import DateColumnRule
from .duplicate_columns import DuplicateColumnsRule
from .email_column import EmailColumnRule
from .foreign_key import ForeignKeyCandidateRule
from .percentage_column import PercentageColumnRule
from .phone_column import PhoneColumnRule
from .primary_key import PrimaryKeyRule

register_column_rule(EmailColumnRule())
register_column_rule(PhoneColumnRule())
register_column_rule(DateColumnRule())
register_column_rule(CurrencyColumnRule())
register_column_rule(PercentageColumnRule())
register_column_rule(PrimaryKeyRule())
register_column_rule(ForeignKeyCandidateRule())
register_column_rule(CategoricalColumnRule())

register_dataset_rule(DuplicateColumnsRule())
