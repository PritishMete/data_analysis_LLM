# schema_intelligence/registry.py
# ─────────────────────────────────────────────────────────────────────────────
# The registry is deliberately dumb: it holds two lists (column rules,
# dataset rules) and knows how to run every rule in each list. Adding a new
# detection capability means writing a new rule class and calling
# register_column_rule()/register_dataset_rule() once — see the bottom of
# rules/__init__.py, which is the ONLY file that imports every individual
# rule module. This file itself never changes when a rule is added or
# removed.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd

from .contracts import ColumnContext, ColumnRule, DatasetRule, RuleResult

_column_rules: list[ColumnRule] = []
_dataset_rules: list[DatasetRule] = []


def register_column_rule(rule: ColumnRule) -> ColumnRule:
    """Also usable as a decorator on a class, e.g.:
        @register_column_rule
        class MyRule(ColumnRule): ...
    (register_column_rule is called with an INSTANCE, so pair it with a
    module-level `register_column_rule(MyRule())` call instead if the rule
    needs constructor arguments — see rules/*.py for the pattern used.)
    """
    _column_rules.append(rule)
    return rule


def register_dataset_rule(rule: DatasetRule) -> DatasetRule:
    _dataset_rules.append(rule)
    return rule


def registered_column_rules() -> list[ColumnRule]:
    return list(_column_rules)


def registered_dataset_rules() -> list[DatasetRule]:
    return list(_dataset_rules)


def run_column_rules(context: ColumnContext) -> list[RuleResult]:
    """Runs every registered column rule against one column, returning every
    non-None result sorted by confidence (highest first). Callers decide
    what to do with ties/multiple candidates (SchemaIntelligenceService
    keeps only the top one for the authoritative `inferred_role`, but the
    full ranked list is persisted for audit — see service.py).
    """
    results = []
    for rule in _column_rules:
        result = rule.evaluate(context)
        if result is not None:
            results.append(result)
    return sorted(results, key=lambda r: r.confidence, reverse=True)


def run_dataset_rules(dataset_id: str, dataframe: pd.DataFrame) -> list[RuleResult]:
    results: list[RuleResult] = []
    for rule in _dataset_rules:
        results.extend(rule.evaluate(dataset_id, dataframe))
    return sorted(results, key=lambda r: r.confidence, reverse=True)
