# tests/test_query_history.py
import pandas as pd
import pytest

from datasets.repository import DatasetRepository
from datasets.service import DatasetRegistryService
from query_history.repository import QueryHistoryRepository
from query_history.service import QueryHistoryService


def _build_services(db_session):
    dataset_repo = DatasetRepository(db_session)
    history_repo = QueryHistoryRepository(db_session)
    history_service = QueryHistoryService(history_repo, dataset_repo)
    registry_service = DatasetRegistryService(dataset_repo)
    return registry_service, history_service


# ── log_execution: full field coverage ──────────────────────────────────────

def test_log_execution_persists_every_field(db_session):
    _, service = _build_services(db_session)

    entry = service.log_execution(
        user_query="total revenue by region",
        intent="aggregate",
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        python_pipeline={"group_by": ["region"], "metrics": [{"column": "revenue", "function": "sum"}]},
        execution_time_ms=42.5,
        rows_returned=7,
        dataset_id=None,
        organization_id="org_1",
        success=True,
    )

    assert entry.id is not None
    assert entry.intent == "aggregate"
    assert entry.rows_returned == 7
    assert entry.success is True
    assert entry.error_message is None
    assert entry.feedback_score is None


def test_log_execution_auto_resolves_schema_hash_from_dataset_id(db_session):
    registry_service, history_service = _build_services(db_session)

    df = pd.DataFrame({"Region": ["North"], "Revenue": [100.0]})
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )

    entry = history_service.log_execution(user_query="q1", dataset_id=reg.dataset.dataset_id)
    assert entry.schema_hash == reg.dataset.schema_hash


def test_log_execution_without_dataset_repository_skips_schema_hash(db_session):
    # Constructing with only the repository (old call shape) must still work.
    service = QueryHistoryService(QueryHistoryRepository(db_session))
    entry = service.log_execution(user_query="q1", dataset_id="some-id-that-may-not-exist")
    assert entry.schema_hash is None


def test_log_execution_records_failure_with_error_message(db_session):
    _, service = _build_services(db_session)
    entry = service.log_execution(
        user_query="broken query", success=False, error_message="ValueError: column not found",
    )
    assert entry.success is False
    assert entry.error_message == "ValueError: column not found"


# ── planner_version ──────────────────────────────────────────────────────

def test_log_execution_persists_planner_version(db_session):
    _, service = _build_services(db_session)
    entry = service.log_execution(user_query="q1", organization_id="org_1", planner_version="gemini-1.5-flash")
    assert entry.planner_version == "gemini-1.5-flash"


def test_log_execution_planner_version_defaults_to_none(db_session):
    # Existing callers that don't know about planner_version yet must still work.
    _, service = _build_services(db_session)
    entry = service.log_execution(user_query="q1")
    assert entry.planner_version is None


def test_get_history_filters_by_planner_version(db_session):
    _, service = _build_services(db_session)
    service.log_execution(user_query="q1", organization_id="org_1", planner_version="v1")
    service.log_execution(user_query="q2", organization_id="org_1", planner_version="v2")

    assert len(service.get_history(organization_id="org_1", planner_version="v1")) == 1
    assert service.get_history(organization_id="org_1", planner_version="v1")[0].user_query == "q1"


def test_tracker_logs_planner_version_set_at_track_time(db_session):
    _, service = _build_services(db_session)
    with service.track(user_query="q", organization_id="org_1", planner_version="rules-engine-v2") as tracker:
        tracker.set_result(generated_sql="SELECT 1")

    entry = service.get_history(organization_id="org_1")[0]
    assert entry.planner_version == "rules-engine-v2"


def test_tracker_logs_planner_version_set_via_set_result(db_session):
    # Covers the "which planner actually handled it" only known mid-execution case.
    _, service = _build_services(db_session)
    with service.track(user_query="q", organization_id="org_1") as tracker:
        tracker.set_result(generated_sql="SELECT 1", planner_version="fallback-planner")

    entry = service.get_history(organization_id="org_1")[0]
    assert entry.planner_version == "fallback-planner"


# ── record_feedback / get_history (existing behavior, unchanged) ───────────

def test_record_feedback_updates_existing_entry(db_session):
    _, service = _build_services(db_session)
    entry = service.log_execution(user_query="q1", organization_id="org_1")
    updated = service.record_feedback(entry.id, feedback_score=1)
    assert updated is not None
    assert updated.feedback_score == 1


def test_record_feedback_returns_none_for_missing_entry(db_session):
    _, service = _build_services(db_session)
    assert service.record_feedback(99999, feedback_score=1) is None


def test_get_history_filters_by_organization_dataset_and_success(db_session):
    _, service = _build_services(db_session)

    service.log_execution(user_query="q1", organization_id="org_1", dataset_id="ds_a", success=True)
    service.log_execution(user_query="q2", organization_id="org_1", dataset_id="ds_b", success=False)
    service.log_execution(user_query="q3", organization_id="org_2", dataset_id="ds_a", success=True)

    assert len(service.get_history(organization_id="org_1")) == 2
    assert len(service.get_history(dataset_id="ds_a")) == 2
    assert len(service.get_history(organization_id="org_1", dataset_id="ds_a")) == 1
    assert len(service.get_history(success=False)) == 1
    assert len(service.get_history(organization_id="org_1", success=True)) == 1


def test_find_reusable_plan_only_matches_successful_exact_text(db_session):
    _, service = _build_services(db_session)

    service.log_execution(user_query="total revenue by region", dataset_id="ds_a", success=False)
    service.log_execution(user_query="total revenue by region", dataset_id="ds_a", success=True, generated_sql="SELECT 1")
    service.log_execution(user_query="different query", dataset_id="ds_a", success=True)

    reusable = service.find_reusable_plan(user_query="total revenue by region", dataset_id="ds_a")
    assert reusable is not None
    assert reusable.generated_sql == "SELECT 1"
    assert service.find_reusable_plan(user_query="totally unseen query", dataset_id="ds_a") is None


# ── get_training_examples ───────────────────────────────────────────────────

def test_get_training_examples_scoped_by_schema_hash(db_session):
    registry_service, history_service = _build_services(db_session)

    sales_df = pd.DataFrame({"Region": ["North"], "Revenue": [100.0]})
    other_df = pd.DataFrame({"City": ["X"], "Cost": [5]})

    sales_reg = registry_service.register_dataset(
        df=sales_df, raw_bytes=b"a", organization_id="org_1", dataset_name="a.csv", uploaded_by="p", source_type="csv"
    )
    other_reg = registry_service.register_dataset(
        df=other_df, raw_bytes=b"b", organization_id="org_1", dataset_name="b.csv", uploaded_by="p", source_type="csv"
    )

    history_service.log_execution(user_query="q1", dataset_id=sales_reg.dataset.dataset_id, success=True)
    history_service.log_execution(user_query="q2", dataset_id=other_reg.dataset.dataset_id, success=True)
    history_service.log_execution(user_query="q3", dataset_id=sales_reg.dataset.dataset_id, success=False)

    sales_examples = history_service.get_training_examples(schema_hash=sales_reg.dataset.schema_hash)
    assert len(sales_examples) == 1  # only the successful one, since only_successful defaults True
    assert sales_examples[0].user_query == "q1"

    all_sales_examples = history_service.get_training_examples(
        schema_hash=sales_reg.dataset.schema_hash, only_successful=False
    )
    assert len(all_sales_examples) == 2


# ── QueryExecutionTracker — the reusable automatic-logging context manager ─

def test_tracker_logs_success_with_measured_timing_and_result(db_session):
    _, service = _build_services(db_session)

    with service.track(user_query="total revenue by region", dataset_id=None, organization_id="org_1", intent="aggregate") as tracker:
        rows = [{"region": "North", "revenue": 100}, {"region": "South", "revenue": 200}]
        tracker.set_result(generated_sql="SELECT region, revenue FROM data", rows_returned=len(rows))

    history = service.get_history(organization_id="org_1")
    assert len(history) == 1
    entry = history[0]
    assert entry.success is True
    assert entry.rows_returned == 2
    assert entry.generated_sql == "SELECT region, revenue FROM data"
    assert entry.execution_time_ms is not None
    assert entry.execution_time_ms >= 0


def test_tracker_logs_failure_and_still_reraises_the_exception(db_session):
    _, service = _build_services(db_session)

    with pytest.raises(ValueError, match="boom"):
        with service.track(user_query="a query that blows up", organization_id="org_1") as tracker:
            raise ValueError("boom")

    history = service.get_history(organization_id="org_1")
    assert len(history) == 1
    entry = history[0]
    assert entry.success is False
    assert "boom" in entry.error_message


def test_tracker_set_result_can_be_called_multiple_times_without_clobbering(db_session):
    _, service = _build_services(db_session)

    with service.track(user_query="q", organization_id="org_1") as tracker:
        tracker.set_result(generated_sql="SELECT 1")
        tracker.set_result(rows_returned=5)  # should NOT wipe out generated_sql set above

    entry = service.get_history(organization_id="org_1")[0]
    assert entry.generated_sql == "SELECT 1"
    assert entry.rows_returned == 5


def test_tracker_auto_resolves_schema_hash_via_dataset_id(db_session):
    registry_service, history_service = _build_services(db_session)
    df = pd.DataFrame({"Region": ["North"], "Revenue": [100.0]})
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"sales", organization_id="org_1", dataset_name="sales.csv", uploaded_by="p", source_type="csv"
    )

    with history_service.track(user_query="q", dataset_id=reg.dataset.dataset_id) as tracker:
        tracker.set_result(rows_returned=1)

    entry = history_service.get_history(dataset_id=reg.dataset.dataset_id)[0]
    assert entry.schema_hash == reg.dataset.schema_hash
