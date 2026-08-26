# schema_intelligence/service.py
# ─────────────────────────────────────────────────────────────────────────────
# Orchestrates the rule engine (registry.py + rules/) against a specific
# dataset and persists the results:
#   - the single winning role per column -> datasets/dataset_columns.
#     inferred_role, via DatasetRepository.update_column_role() — an
#     EXISTING method on an EXISTING repository, called as-is. The Dataset
#     Registry package itself (models/repository/service) is not modified
#     by one line as part of this feature.
#   - the FULL confidence-scored candidate list per column -> the new
#     column_role_detections table (ColumnRoleDetectionRepository)
#   - detected duplicate-column pairs -> the new duplicate_column_pairs
#     table (DuplicateColumnRepository)
#   - confirmed cross-dataset relationships -> dataset_relationships
#     (RelationshipRepository, unchanged from before)
#
# No detection LOGIC lives in this file — it's pure composition + persistence,
# same principle as before this rewrite. `rules` is imported (not just its
# registry) specifically for its import-time SIDE EFFECT of registering every
# built-in rule; see rules/__init__.py.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field

import pandas as pd

from datasets.repository import DatasetRepository

from . import rules  # noqa: F401 — import side effect: registers all built-in rules
from .contracts import ColumnContext, RuleResult
from .models import ColumnRoleDetection, DatasetRelationship, DuplicateColumnPair
from .registry import run_column_rules, run_dataset_rules
from .relationship_detector import RelationshipCandidate, find_relationship_candidates
from .repository import ColumnRoleDetectionRepository, DuplicateColumnRepository, RelationshipRepository


@dataclass
class SchemaAnalysisResult:
    dataset_id: str
    column_roles: dict[str, str | None] = field(default_factory=dict)  # winning role per column
    column_candidates: dict[str, list[RuleResult]] = field(default_factory=dict)  # ALL scored candidates
    duplicate_columns: list[RuleResult] = field(default_factory=list)
    relationships: list[RelationshipCandidate] = field(default_factory=list)


class SchemaIntelligenceService:
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        relationship_repository: RelationshipRepository,
        column_detection_repository: ColumnRoleDetectionRepository | None = None,
        duplicate_repository: DuplicateColumnRepository | None = None,
    ):
        self.dataset_repository = dataset_repository
        self.relationship_repository = relationship_repository
        # Optional so existing call sites constructed with just the first two
        # arguments keep working (role detection + relationship confirmation
        # still run; only the richer audit-trail persistence is skipped).
        self.column_detection_repository = column_detection_repository
        self.duplicate_repository = duplicate_repository

    def analyze_dataset(self, dataset_id: str, df: pd.DataFrame) -> SchemaAnalysisResult:
        """Runs every registered ColumnRule against every column, plus every
        registered DatasetRule against the whole DataFrame (duplicate-column
        detection today; whatever's added to rules/ tomorrow). Persists:
          - the single top-confidence role per column, via the EXISTING
            DatasetRepository.update_column_role()
          - the full ranked candidate list per column (audit trail)
          - detected duplicate-column pairs
        """
        row_count = len(df)
        column_roles: dict[str, str | None] = {}
        column_candidates: dict[str, list[RuleResult]] = {}
        all_detection_rows: list[ColumnRoleDetection] = []

        for column_name in df.columns:
            context = ColumnContext(
                dataset_id=dataset_id,
                column_name=str(column_name),
                series=df[column_name],
                dataframe=df,
                row_count=row_count,
            )
            candidates = run_column_rules(context)
            column_candidates[str(column_name)] = candidates

            winning_role = candidates[0].role if candidates else None
            column_roles[str(column_name)] = winning_role
            self.dataset_repository.update_column_role(dataset_id, str(column_name), winning_role)

            # Extension point: persist the confidence/evidence behind that
            # SAME winning role directly into the Dataset Registry too (see
            # DatasetColumn.inferred_role_confidence and
            # update_column_role_metadata's docstrings for why this is a
            # separate call rather than a change to update_column_role
            # above). None/None when there were no candidates at all, so a
            # column that plainly matched no rule doesn't keep a stale
            # confidence score from a previous analysis run.
            winning_confidence = candidates[0].confidence if candidates else None
            winning_evidence = candidates[0].evidence if candidates else None
            self.dataset_repository.update_column_role_metadata(
                dataset_id, str(column_name), confidence=winning_confidence, evidence=winning_evidence
            )

            all_detection_rows.extend(
                ColumnRoleDetection(
                    dataset_id=dataset_id,
                    column_name=str(column_name),
                    role=c.role,
                    confidence=c.confidence,
                    rule_name=c.rule_name,
                    evidence=c.evidence,
                )
                for c in candidates
            )

        if self.column_detection_repository is not None:
            self.column_detection_repository.replace_for_dataset(dataset_id, all_detection_rows)

        duplicate_results = run_dataset_rules(dataset_id, df)
        if self.duplicate_repository is not None:
            duplicate_rows = [
                DuplicateColumnPair(
                    dataset_id=dataset_id,
                    column_a=r.evidence["column_a"],
                    column_b=r.evidence["column_b"],
                    confidence=r.confidence,
                )
                for r in duplicate_results
                if r.role == "duplicate_column"
            ]
            self.duplicate_repository.replace_for_dataset(dataset_id, duplicate_rows)

        return SchemaAnalysisResult(
            dataset_id=dataset_id,
            column_roles=column_roles,
            column_candidates=column_candidates,
            duplicate_columns=duplicate_results,
        )

    def detect_relationships(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        other_datasets: list[tuple[str, pd.DataFrame]] | None = None,
        min_confidence: float = 0.8,
    ) -> list[RelationshipCandidate]:
        """Optional second pass — only produces results when `other_datasets`
        is supplied (real DataFrames for other datasets available in the SAME
        request). See relationship_detector.py's module docstring for why
        this can't reach back into a previous, separate upload session's raw
        data. Candidate columns now come from the rule engine's
        "foreign_key_candidate" role (ForeignKeyCandidateRule) instead of a
        hardcoded name-matching function.
        """
        if not other_datasets:
            return []

        row_count = len(df)
        candidate_columns = []
        for column_name in df.columns:
            context = ColumnContext(
                dataset_id=dataset_id,
                column_name=str(column_name),
                series=df[column_name],
                dataframe=df,
                row_count=row_count,
            )
            top_candidates = run_column_rules(context)
            if any(c.role == "foreign_key_candidate" for c in top_candidates):
                candidate_columns.append(str(column_name))

        if not candidate_columns:
            return []

        candidates = find_relationship_candidates(
            source_dataset_id=dataset_id,
            source_df=df,
            candidate_columns=candidate_columns,
            other_datasets=other_datasets,
            min_confidence=min_confidence,
        )

        if candidates:
            rows = [
                DatasetRelationship(
                    source_dataset_id=dataset_id,
                    source_column=c.source_column,
                    target_dataset_id=c.target_dataset_id,
                    target_column=c.target_column,
                    confidence=c.confidence,
                )
                for c in candidates
            ]
            self.relationship_repository.add_relationships(rows)
            # A confirmed relationship makes the source column a real
            # foreign_key, not just an untested candidate — upgrade its role
            # from "foreign_key_candidate" to "foreign_key".
            for c in candidates:
                self.dataset_repository.update_column_role(dataset_id, c.source_column, "foreign_key")
                # Extension point (same pattern as analyze_dataset above):
                # the relationship's own value-overlap confidence IS the
                # confidence behind this "foreign_key" role, so persist it
                # into the Dataset Registry too, with the confirmed target
                # as evidence.
                self.dataset_repository.update_column_role_metadata(
                    dataset_id,
                    c.source_column,
                    confidence=c.confidence,
                    evidence={"target_dataset_id": c.target_dataset_id, "target_column": c.target_column},
                )

        return candidates
