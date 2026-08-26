# schema_intelligence/contracts.py
# ─────────────────────────────────────────────────────────────────────────────
# The engine's extension contract. Every detection capability — existing or
# future — implements ONE of these two interfaces and registers itself (see
# registry.py). Nothing that dispatches between rules needs to change when a
# new rule is added: that's what "do not hardcode rules" means in concrete
# terms here — there is no central if/elif chain deciding "is this an email
# column, else is it a phone column, else...". Each rule independently
# decides whether it applies and how confident it is; the engine just runs
# every registered rule and keeps what scored high enough.
# ─────────────────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ColumnContext:
    """Everything a per-column rule might need. Deliberately includes the
    full DataFrame (not just the one Series) so a rule CAN look at sibling
    columns if it needs to (e.g. a future "this looks like a currency
    because there's already a sibling '<name>_currency' column" rule) —
    without needing a contract change later.
    """
    dataset_id: str
    column_name: str
    series: pd.Series
    dataframe: pd.DataFrame
    row_count: int


@dataclass(frozen=True)
class RuleResult:
    """One rule's verdict. `role` is a free-form string label (e.g. "email",
    "primary_key", "duplicate_column") — NOT a fixed enum — so a new rule can
    introduce a new role without touching any shared type definition.
    `evidence` holds whatever numbers justified the score, for auditability
    (this is what makes the engine's decisions inspectable rather than a
    black box, which matters a lot given there's no LLM here to "explain
    itself" — the evidence dict IS the explanation).
    """
    role: str
    confidence: float  # 0.0-1.0
    rule_name: str
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be within [0.0, 1.0], got {self.confidence}")


class ColumnRule(ABC):
    """A rule that evaluates ONE column at a time (email, phone, date,
    currency, percentage, primary key, foreign-key-candidate, categorical,
    ...). Implementations live in schema_intelligence/rules/.
    """

    name: str = "unnamed_column_rule"

    @abstractmethod
    def evaluate(self, context: ColumnContext) -> RuleResult | None:
        """Return a RuleResult if this rule has an opinion about the column,
        or None if it plainly doesn't apply (e.g. a numeric-only rule sees a
        text column). Returning None is different from returning a
        RuleResult with confidence=0.0 — None means "not applicable",
        whereas a low confidence score means "applicable, but weak evidence"."""
        raise NotImplementedError


class DatasetRule(ABC):
    """A rule that evaluates the WHOLE dataset at once — i.e. anything that
    compares multiple columns against each other rather than judging one
    column in isolation. Duplicate-column detection is the first of these;
    a future "these two columns are functionally dependent" rule would be
    another.
    """

    name: str = "unnamed_dataset_rule"

    @abstractmethod
    def evaluate(self, dataset_id: str, dataframe: pd.DataFrame) -> list[RuleResult]:
        raise NotImplementedError
