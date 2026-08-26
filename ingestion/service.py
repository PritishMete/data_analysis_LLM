# ingestion/service.py
# ─────────────────────────────────────────────────────────────────────────────
# This is the ONLY module that imports both `datasets` and `schema_intelligence`
# services. Neither of those packages imports the other directly — composing
# them here (rather than having schema_intelligence reach into datasets, or
# vice versa) keeps each package independently testable/replaceable, which is
# what "modular, own package, clean architecture" means in practice: the
# dependency direction is ingestion -> {datasets, schema_intelligence}, never
# sideways between the two feature packages.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass

import pandas as pd

from datasets.service import DatasetRegistration, DatasetRegistryService
from schema_intelligence.relationship_detector import RelationshipCandidate
from schema_intelligence.service import SchemaIntelligenceService


@dataclass
class IngestionResult:
    registration: DatasetRegistration
    column_roles: dict[str, str | None]
    relationships: list[RelationshipCandidate]


class DatasetIngestionOrchestrator:
    def __init__(self, registry_service: DatasetRegistryService, intelligence_service: SchemaIntelligenceService):
        self.registry_service = registry_service
        self.intelligence_service = intelligence_service

    def ingest(
        self,
        *,
        df: pd.DataFrame,
        raw_bytes: bytes,
        organization_id: str,
        dataset_name: str,
        uploaded_by: str | None,
        source_type: str,
        other_datasets: list[tuple[str, pd.DataFrame]] | None = None,
    ) -> IngestionResult:
        """The full "a dataset just got uploaded" use case:
          1. Register it (idempotent on exact re-upload — see datasets/service.py).
          2. Run Schema Intelligence over it (skipped for an exact duplicate,
             since the original upload was already analyzed).
          3. Optionally detect cross-dataset relationships, IF the caller
             supplied other real DataFrames to compare against (see
             schema_intelligence/relationship_detector.py for why this can't
             reach into a prior, separate upload session on its own).
        """
        registration = self.registry_service.register_dataset(
            df=df,
            raw_bytes=raw_bytes,
            organization_id=organization_id,
            dataset_name=dataset_name,
            uploaded_by=uploaded_by,
            source_type=source_type,
        )

        if registration.was_duplicate:
            existing_roles = {c.column_name: c.inferred_role for c in registration.columns}
            return IngestionResult(registration=registration, column_roles=existing_roles, relationships=[])

        analysis = self.intelligence_service.analyze_dataset(registration.dataset.dataset_id, df)
        relationships = self.intelligence_service.detect_relationships(
            registration.dataset.dataset_id, df, other_datasets=other_datasets
        )

        return IngestionResult(
            registration=registration,
            column_roles=analysis.column_roles,
            relationships=relationships,
        )
