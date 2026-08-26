# schema_intelligence/rules/_shared.py
# ─────────────────────────────────────────────────────────────────────────────
# Small pure helpers shared by several rules. Kept here (rather than copy-
# pasted into each rule file) so there's exactly one implementation of
# "sample a column" and "compute a regex match ratio" — every rule that needs
# either imports from here.
# ─────────────────────────────────────────────────────────────────────────────

import re

import pandas as pd


def non_null_sample(series: pd.Series, max_n: int = 500) -> pd.Series:
    """Caps detector cost on huge columns. 500 non-null values is already
    far more than enough to settle a match-ratio question deterministically."""
    non_null = series.dropna()
    if len(non_null) > max_n:
        return non_null.sample(n=max_n, random_state=0)
    return non_null


def match_ratio(series: pd.Series, pattern: re.Pattern, max_n: int = 500) -> tuple[float, int]:
    """Returns (ratio of sampled non-null values matching `pattern`, sample size)."""
    sample = non_null_sample(series, max_n=max_n)
    if len(sample) == 0:
        return 0.0, 0
    matches = sample.astype(str).str.strip().str.match(pattern)
    return float(matches.mean()), len(sample)


def name_hint_score(column_name: str, hints: tuple[str, ...]) -> float:
    """1.0 if the column name contains one of `hints` as a whole word-ish
    substring, else 0.0. Kept as a simple binary signal deliberately — this
    is ONE piece of evidence a rule combines with others, not the whole
    decision by itself."""
    name = str(column_name).strip().lower()
    return 1.0 if any(h in name for h in hints) else 0.0
