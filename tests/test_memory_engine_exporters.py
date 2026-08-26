# tests/test_memory_engine_exporters.py
import io
import json

import pandas as pd
import pyarrow.parquet as pq

from datasets.repository import DatasetRepository
from datasets.service import DatasetRegistryService
from memory_engine.exporters import TRAINING_EXPORT_COLUMNS, TrainingDatasetExporter
from query_history.repository import QueryHistoryRepository
from query_history.service import QueryHistoryService


def _sales_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Region": ["North", "South"],
        "Product": ["Widget", "Gadget"],
        "Quantity": [10, 5],
        "Revenue": [100.0, 200.0],
    })


def _build(db_session):
    dataset_repo = DatasetRepository(db_session)
    registry_service = DatasetRegistryService(dataset_repo)
    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    exporter = TrainingDatasetExporter(QueryHistoryRepository(db_session), dataset_repo)
    return registry_service, history_service, exporter


def test_to_records_has_exactly_the_seven_spec_columns(db_session):
    registry_service, history_service, exporter = _build(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="total revenue by region",
        intent="revenue_by_region",
        dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        python_pipeline={"group_by": ["region"]},
        execution_time_ms=42.5,
        success=True,
    )
    entry = history_service.get_history(organization_id="org_1")[0]
    history_service.repository.set_feedback(entry.id, 1)

    entries = exporter.collect(organization_id="org_1")
    records = exporter.to_records(entries)

    assert len(records) == 1
    record = records[0]
    assert set(record.keys()) == set(TRAINING_EXPORT_COLUMNS)
    assert record["intent"] == "revenue_by_region"
    assert record["question"] == "total revenue by region"
    assert record["sql"] == "SELECT region, SUM(revenue) FROM data GROUP BY region"
    assert json.loads(record["pipeline"]) == {"group_by": ["region"]}
    assert record["execution_time_ms"] == 42.5
    assert record["feedback"] == 1
    assert record["dataset_type"] == "csv"


def test_to_records_handles_missing_dataset_and_missing_pipeline(db_session):
    _, history_service, exporter = _build(db_session)
    history_service.log_execution(user_query="ad hoc question", success=True)

    entries = exporter.collect(limit=10)
    records = exporter.to_records(entries)

    assert len(records) == 1
    assert records[0]["dataset_type"] is None
    assert records[0]["pipeline"] is None
    assert records[0]["feedback"] is None


def test_collect_includes_failures_by_default(db_session):
    registry_service, history_service, exporter = _build(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="good query", dataset_id=reg.dataset.dataset_id, success=True,
    )
    history_service.log_execution(
        user_query="bad query", dataset_id=reg.dataset.dataset_id, success=False,
        error_message="syntax error",
    )

    all_entries = exporter.collect(organization_id="org_1")
    assert len(all_entries) == 2

    successful_only = exporter.collect(organization_id="org_1", only_successful=True)
    assert len(successful_only) == 1
    assert successful_only[0].user_query == "good query"


def test_export_csv_round_trips(db_session):
    registry_service, history_service, exporter = _build(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="xlsx",
    )
    history_service.log_execution(
        user_query="total revenue by region", intent="revenue_by_region",
        dataset_id=reg.dataset.dataset_id, generated_sql="SELECT 1",
        execution_time_ms=10.0, success=True,
    )

    entries = exporter.collect(organization_id="org_1")
    csv_bytes = exporter.export_csv(entries)

    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert list(df.columns) == TRAINING_EXPORT_COLUMNS
    assert df.iloc[0]["dataset_type"] == "xlsx"
    assert df.iloc[0]["sql"] == "SELECT 1"


def test_export_parquet_round_trips_and_preserves_schema(db_session):
    registry_service, history_service, exporter = _build(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="json",
    )
    history_service.log_execution(
        user_query="total revenue by region", intent="revenue_by_region",
        dataset_id=reg.dataset.dataset_id, generated_sql="SELECT 1",
        python_pipeline=[{"op": "filter"}], execution_time_ms=10.0, success=True,
    )

    entries = exporter.collect(organization_id="org_1")
    parquet_bytes = exporter.export_parquet(entries)

    table = pq.read_table(io.BytesIO(parquet_bytes))
    assert table.column_names == TRAINING_EXPORT_COLUMNS
    df = table.to_pandas()
    assert df.iloc[0]["dataset_type"] == "json"
    assert json.loads(df.iloc[0]["pipeline"]) == [{"op": "filter"}]


def test_export_rejects_unsupported_format(db_session):
    import pytest
    _, _, exporter = _build(db_session)
    with pytest.raises(ValueError):
        exporter.export([], fmt="xlsx")


def test_csv_and_parquet_produce_the_same_columns(db_session):
    registry_service, history_service, exporter = _build(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="q", dataset_id=reg.dataset.dataset_id, generated_sql="SELECT 1", success=True,
    )
    entries = exporter.collect(organization_id="org_1")

    csv_df = pd.read_csv(io.BytesIO(exporter.export_csv(entries)))
    parquet_df = pd.read_parquet(io.BytesIO(exporter.export_parquet(entries)))

    assert list(csv_df.columns) == list(parquet_df.columns) == TRAINING_EXPORT_COLUMNS
