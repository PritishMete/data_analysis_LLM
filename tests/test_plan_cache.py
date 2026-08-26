# tests/test_plan_cache.py
import pandas as pd

from datasets.repository import DatasetRepository
from datasets.service import DatasetRegistryService
from plan_cache.repository import PlanCacheRepository
from plan_cache.service import PlanCacheService
from query_history.repository import QueryHistoryRepository
from query_history.service import QueryHistoryService


def _build_services(db_session):
    dataset_repo = DatasetRepository(db_session)
    registry_service = DatasetRegistryService(dataset_repo)
    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    plan_cache_service = PlanCacheService(dataset_repo, PlanCacheRepository(db_session))
    return registry_service, history_service, plan_cache_service


def _sales_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Region": ["North", "South"],
        "Product": ["Widget", "Gadget"],
        "Quantity": [10, 5],
        "Revenue": [100.0, 200.0],
    })


def test_no_hit_when_nothing_logged_yet(db_session):
    registry_service, _, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales-1", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )

    hit = plan_cache_service.find_cached_plan(
        dataset_id=reg.dataset.dataset_id, user_query="total revenue by region"
    )
    assert hit is None


def test_reuses_plan_for_same_dataset(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales-1", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )

    history_service.log_execution(
        user_query="total revenue by region",
        dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        success=True,
    )

    hit = plan_cache_service.find_cached_plan(
        dataset_id=reg.dataset.dataset_id, user_query="total revenue by region"
    )
    assert hit is not None
    assert hit.matched_on == "same_dataset"
    assert hit.generated_sql == "SELECT region, SUM(revenue) FROM data GROUP BY region"


def test_reuses_plan_across_different_datasets_with_same_schema_shape(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)

    # Two DIFFERENT organizations, two DIFFERENT dataset rows, but the exact
    # same column names + dtypes -> identical schema_hash.
    org_a = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"org-a-sales", organization_id="org_a",
        dataset_name="q1_sales.csv", uploaded_by="alice", source_type="csv",
    )
    org_b = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"org-b-sales", organization_id="org_b",
        dataset_name="q2_sales.csv", uploaded_by="bob", source_type="csv",
    )
    assert org_a.dataset.dataset_id != org_b.dataset.dataset_id
    assert org_a.dataset.schema_hash == org_b.dataset.schema_hash

    # org_a already asked this question successfully...
    history_service.log_execution(
        user_query="total revenue by region",
        dataset_id=org_a.dataset.dataset_id,
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        success=True,
    )

    # ...org_b asks the SAME question for the FIRST time on a dataset
    # that's never been queried before, but shares the schema shape.
    hit = plan_cache_service.find_cached_plan(
        dataset_id=org_b.dataset.dataset_id, user_query="total revenue by region"
    )
    assert hit is not None
    assert hit.matched_on == "same_schema_shape"
    assert hit.source_dataset_id == org_a.dataset.dataset_id
    assert hit.generated_sql == "SELECT region, SUM(revenue) FROM data GROUP BY region"


def test_does_not_reuse_plan_across_different_schema_shapes(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)

    sales_reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="total revenue by region",
        dataset_id=sales_reg.dataset.dataset_id,
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        success=True,
    )

    different_shape_df = pd.DataFrame({"City": ["A"], "Cuisine": ["Indian"], "Cost": [500]})
    restaurants_reg = registry_service.register_dataset(
        df=different_shape_df, raw_bytes=b"restaurants", organization_id="org_1",
        dataset_name="restaurants.csv", uploaded_by="p", source_type="csv",
    )

    hit = plan_cache_service.find_cached_plan(
        dataset_id=restaurants_reg.dataset.dataset_id, user_query="total revenue by region"
    )
    assert hit is None


def test_does_not_reuse_a_failed_execution(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )

    history_service.log_execution(
        user_query="total revenue by region",
        dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT this is broken sql",
        success=False,
    )

    hit = plan_cache_service.find_cached_plan(
        dataset_id=reg.dataset.dataset_id, user_query="total revenue by region"
    )
    assert hit is None


def test_returns_none_for_unregistered_dataset(db_session):
    _, _, plan_cache_service = _build_services(db_session)
    hit = plan_cache_service.find_cached_plan(dataset_id="does-not-exist", user_query="anything")
    assert hit is None


# ── New dimensions: intent / planner_version / confidence / expiration /
# invalidation. evaluate() is exercised directly here since it's the one
# that reports WHY a lookup didn't hit; find_cached_plan() (tested above,
# unchanged) is just evaluate(...).hit.

from datetime import datetime, timedelta, timezone

from plan_cache.service import PlanCacheOutcome, PlanCacheService


def test_evaluate_reports_hit_for_exact_match(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="total revenue by region", dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT 1", success=True,
    )

    result = plan_cache_service.evaluate(dataset_id=reg.dataset.dataset_id, user_query="total revenue by region")
    assert result.outcome == PlanCacheOutcome.HIT
    assert result.hit.confidence == 0.9  # exact-query tier, no feedback, brand new


def test_evaluate_reports_miss_when_nothing_matches(db_session):
    registry_service, _, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    result = plan_cache_service.evaluate(dataset_id=reg.dataset.dataset_id, user_query="never asked before")
    assert result.outcome == PlanCacheOutcome.MISS
    assert result.hit is None


def test_intent_tier_reuses_plan_despite_different_wording(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="total revenue by region",
        intent="revenue_by_region",
        planner_version="gemini-1.5-flash",
        dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        success=True,
    )

    # Completely different phrasing -> tier 1 (exact match) can't find it,
    # but tier 2 (intent + schema + planner_version) can.
    result = plan_cache_service.evaluate(
        dataset_id=reg.dataset.dataset_id,
        user_query="what's the revenue breakdown per region?",
        intent="revenue_by_region",
        planner_version="gemini-1.5-flash",
    )
    assert result.outcome == PlanCacheOutcome.HIT
    assert result.hit.matched_on == "same_intent"
    assert result.hit.generated_sql == "SELECT region, SUM(revenue) FROM data GROUP BY region"


def test_intent_tier_does_not_cross_planner_versions(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="total revenue by region", intent="revenue_by_region", planner_version="v1",
        dataset_id=reg.dataset.dataset_id, generated_sql="SELECT 1", success=True,
    )

    result = plan_cache_service.evaluate(
        dataset_id=reg.dataset.dataset_id,
        user_query="different phrasing",
        intent="revenue_by_region",
        planner_version="v2",  # different planner version -> must not reuse v1's plan
    )
    assert result.outcome == PlanCacheOutcome.MISS


def test_negative_feedback_drags_confidence_below_threshold(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    entry = history_service.log_execution(
        user_query="total revenue by region", dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT 1", success=True,
    )
    history_service.repository.set_feedback(entry.id, -1)

    result = plan_cache_service.evaluate(dataset_id=reg.dataset.dataset_id, user_query="total revenue by region")
    assert result.outcome != PlanCacheOutcome.HIT


def test_expired_plan_is_reported_as_expired_not_miss(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    entry = history_service.log_execution(
        user_query="total revenue by region", dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT 1", success=True,
    )
    # Force the row to look old enough to have aged out of the TTL window.
    from query_history.models import QueryHistory
    row = db_session.get(QueryHistory, entry.id)
    row.created_at = datetime.now(timezone.utc) - timedelta(days=999)
    db_session.commit()

    result = plan_cache_service.evaluate(dataset_id=reg.dataset.dataset_id, user_query="total revenue by region")
    assert result.outcome == PlanCacheOutcome.EXPIRED
    assert result.hit is None


def test_invalidate_plan_by_query_history_id(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    entry = history_service.log_execution(
        user_query="total revenue by region", dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT 1", success=True,
    )

    before = plan_cache_service.evaluate(dataset_id=reg.dataset.dataset_id, user_query="total revenue by region")
    assert before.outcome == PlanCacheOutcome.HIT

    plan_cache_service.invalidate_plan(query_history_id=entry.id, reason="turned out to be wrong")

    after = plan_cache_service.evaluate(dataset_id=reg.dataset.dataset_id, user_query="total revenue by region")
    assert after.outcome == PlanCacheOutcome.INVALIDATED
    assert after.hit is None


def test_invalidate_scope_does_not_affect_plans_logged_afterward(db_session):
    registry_service, history_service, plan_cache_service = _build_services(db_session)
    reg = registry_service.register_dataset(
        df=_sales_df(), raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="total revenue by region", dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT old", success=True,
    )

    plan_cache_service.invalidate_scope(dataset_id=reg.dataset.dataset_id, reason="schema semantics changed")

    stale = plan_cache_service.evaluate(dataset_id=reg.dataset.dataset_id, user_query="total revenue by region")
    assert stale.outcome == PlanCacheOutcome.INVALIDATED

    # A fresh, successful execution AFTER the scope invalidation is fair game again.
    history_service.log_execution(
        user_query="total revenue by region", dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT fixed", success=True,
    )
    fresh = plan_cache_service.evaluate(dataset_id=reg.dataset.dataset_id, user_query="total revenue by region")
    assert fresh.outcome == PlanCacheOutcome.HIT
    assert fresh.hit.generated_sql == "SELECT fixed"


def test_invalidate_scope_raises_for_unregistered_dataset(db_session):
    import pytest
    _, _, plan_cache_service = _build_services(db_session)
    with pytest.raises(ValueError):
        plan_cache_service.invalidate_scope(dataset_id="does-not-exist")
