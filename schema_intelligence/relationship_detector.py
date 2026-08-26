# schema_intelligence/relationship_detector.py
# ─────────────────────────────────────────────────────────────────────────────
# Deterministic foreign-key/relationship candidate detection via VALUE-SET
# OVERLAP — no AI involved. A candidate column (name shaped like an FK, e.g.
# "customer_id") is compared against columns in OTHER datasets that look like
# real keys there; if enough of the candidate's values are found in that
# other column's value set, it's recorded as a relationship.
#
# IMPORTANT, HONEST LIMITATION: the Dataset Registry (datasets/service.py)
# persists only METADATA for each upload (row/column counts, dtypes, hashes)
# — not a snapshot of the actual row values — because this architecture's
# DuckDB/pandas usage is intentionally request-scoped/ephemeral, not a data
# warehouse. That means true value-overlap comparison only works when the
# CALLER has real DataFrames for both sides in hand at the same time —
# e.g. a multi-sheet Excel upload, or a batch-ingest endpoint that receives
# several files in one request. Comparing against a dataset from a
# completely separate, earlier upload session would need a persisted raw-data
# snapshot (e.g. Parquet in object storage) added later; this module doesn't
# pretend to do that with metadata alone.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass

import pandas as pd

from .contracts import ColumnContext
from .rules.primary_key import PrimaryKeyRule

_primary_key_rule = PrimaryKeyRule(min_confidence=0.9)  # only match against a STRONG key candidate


def _looks_like_primary_key(series: pd.Series, row_count: int) -> bool:
    """Thin wrapper around PrimaryKeyRule so this module doesn't duplicate
    the uniqueness/non-null scoring logic — one implementation, reused here
    and in the rule engine proper."""
    context = ColumnContext(
        dataset_id="",  # not used by PrimaryKeyRule
        column_name="",  # not used by PrimaryKeyRule
        series=series,
        dataframe=pd.DataFrame(),  # not used by PrimaryKeyRule
        row_count=row_count,
    )
    return _primary_key_rule.evaluate(context) is not None


@dataclass
class RelationshipCandidate:
    source_column: str
    target_dataset_id: str
    target_column: str
    confidence: float  # fraction (0.0-1.0) of source values found in target's value set


def compute_value_overlap(source_series: pd.Series, target_series: pd.Series) -> float:
    """Fraction of `source_series`'s distinct non-null values that also
    appear in `target_series`'s distinct non-null value set. 0.0 if either
    side is empty."""
    source_vals = set(source_series.dropna().astype(str))
    if not source_vals:
        return 0.0
    target_vals = set(target_series.dropna().astype(str))
    if not target_vals:
        return 0.0
    return len(source_vals & target_vals) / len(source_vals)


def find_relationship_candidates(
    *,
    source_dataset_id: str,
    source_df: pd.DataFrame,
    candidate_columns: list[str],
    other_datasets: list[tuple[str, pd.DataFrame]],
    min_confidence: float = 0.8,
) -> list[RelationshipCandidate]:
    """`other_datasets` must be explicitly supplied as real (dataset_id,
    DataFrame) pairs by the caller — see the module docstring above for why.
    Matches a candidate column against another dataset's column only if that
    column looks like a real key there (_looks_like_primary_key) OR shares the
    exact same name (case-insensitive) — avoids matching against arbitrary
    unrelated columns that happen to overlap by coincidence.
    """
    candidates: list[RelationshipCandidate] = []

    for col in candidate_columns:
        if col not in source_df.columns:
            continue
        source_series = source_df[col]

        for other_dataset_id, other_df in other_datasets:
            if other_dataset_id == source_dataset_id:
                continue
            for other_col in other_df.columns:
                same_name = str(other_col).strip().lower() == str(col).strip().lower()
                other_series = other_df[other_col]
                if not same_name and not _looks_like_primary_key(other_series, len(other_df)):
                    continue

                confidence = compute_value_overlap(source_series, other_series)
                if confidence >= min_confidence:
                    candidates.append(
                        RelationshipCandidate(
                            source_column=col,
                            target_dataset_id=other_dataset_id,
                            target_column=str(other_col),
                            confidence=round(confidence, 4),
                        )
                    )

    return candidates
