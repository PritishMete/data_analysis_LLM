# tests/test_ingestion.py
import pandas as pd

from datasets.repository import DatasetRepository
from datasets.service import DatasetRegistryService
from ingestion.service import DatasetIngestionOrchestrator
from schema_intelligence.repository import RelationshipRepository
from schema_intelligence.service import SchemaIntelligenceService


def _build_orchestrator(db_session) -> DatasetIngestionOrchestrator:
    dataset_repo = DatasetRepository(db_session)
    relationship_repo = RelationshipRepository(db_session)
    registry_service = DatasetRegistryService(dataset_repo)
    intelligence_service = SchemaIntelligenceService(dataset_repo, relationship_repo)
    return DatasetIngestionOrchestrator(registry_service, intelligence_service)


def test_ingest_registers_and_analyzes_in_one_call(db_session):
    orchestrator = _build_orchestrator(db_session)
    df = pd.DataFrame({
        "OrderID": [1, 2, 3],
        "CustomerEmail": ["a@x.com", "b@x.com", "c@x.com"],
        "TotalPrice": [100.0, 200.0, 300.0],
    })

    result = orchestrator.ingest(
        df=df,
        raw_bytes=b"orders-file",
        organization_id="org_1",
        dataset_name="orders.csv",
        uploaded_by="priyanka",
        source_type="csv",
    )

    assert result.registration.was_duplicate is False
    assert result.column_roles["CustomerEmail"] == "email"
    assert result.column_roles["OrderID"] == "primary_key"
    assert result.column_roles["TotalPrice"] == "currency"
    assert result.relationships == []  # no other_datasets supplied


def test_ingest_duplicate_upload_skips_reanalysis(db_session):
    orchestrator = _build_orchestrator(db_session)
    df = pd.DataFrame({"OrderID": [1, 2, 3], "TotalPrice": [1.0, 2.0, 3.0]})

    first = orchestrator.ingest(
        df=df, raw_bytes=b"same-bytes", organization_id="org_1",
        dataset_name="orders.csv", uploaded_by="p", source_type="csv",
    )
    second = orchestrator.ingest(
        df=df, raw_bytes=b"same-bytes", organization_id="org_1",
        dataset_name="orders.csv", uploaded_by="p", source_type="csv",
    )

    assert first.registration.was_duplicate is False
    assert second.registration.was_duplicate is True
    assert second.registration.dataset.dataset_id == first.registration.dataset.dataset_id


def test_ingest_with_other_datasets_detects_relationship(db_session):
    orchestrator = _build_orchestrator(db_session)

    customers_df = pd.DataFrame({"CustomerID": [1, 2, 3], "Name": ["A", "B", "C"]})
    customers_result = orchestrator.ingest(
        df=customers_df, raw_bytes=b"customers-file", organization_id="org_1",
        dataset_name="customers.csv", uploaded_by="p", source_type="csv",
    )

    orders_df = pd.DataFrame({"OrderID": [1, 2, 3], "CustomerID": [1, 2, 3]})
    orders_result = orchestrator.ingest(
        df=orders_df,
        raw_bytes=b"orders-file",
        organization_id="org_1",
        dataset_name="orders.csv",
        uploaded_by="p",
        source_type="csv",
        other_datasets=[(customers_result.registration.dataset.dataset_id, customers_df)],
    )

    assert len(orders_result.relationships) == 1
    assert orders_result.relationships[0].source_column == "CustomerID"
    assert orders_result.relationships[0].target_dataset_id == customers_result.registration.dataset.dataset_id
    assert orders_result.column_roles["CustomerID"] in ("primary_key", "numeric")
    # ^ reflects analyze_dataset's FIRST pass (before relationships are known),
    # which is why "primary_key" is expected here rather than "foreign_key" —
    # see test_schema_intelligence_service_persists_roles_and_relationships
    # for confirmation that the PERSISTED row is what actually gets upgraded.
