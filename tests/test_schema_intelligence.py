# tests/test_schema_intelligence.py
import pandas as pd

from datasets.repository import DatasetRepository
from datasets.service import DatasetRegistryService
from schema_intelligence.contracts import ColumnContext, RuleResult
from schema_intelligence.registry import registered_column_rules, registered_dataset_rules
from schema_intelligence.relationship_detector import compute_value_overlap, find_relationship_candidates
from schema_intelligence.repository import ColumnRoleDetectionRepository, DuplicateColumnRepository, RelationshipRepository
from schema_intelligence.rules.categorical_column import CategoricalColumnRule
from schema_intelligence.rules.currency_column import CurrencyColumnRule
from schema_intelligence.rules.duplicate_columns import DuplicateColumnsRule
from schema_intelligence.rules.email_column import EmailColumnRule
from schema_intelligence.rules.percentage_column import PercentageColumnRule
from schema_intelligence.rules.primary_key import PrimaryKeyRule
from schema_intelligence.service import SchemaIntelligenceService


def _context(series: pd.Series, column_name: str, dataframe: pd.DataFrame | None = None) -> ColumnContext:
    df = dataframe if dataframe is not None else pd.DataFrame({column_name: series})
    return ColumnContext(
        dataset_id="ds_test", column_name=column_name, series=series, dataframe=df, row_count=len(series)
    )


# ── Engine plumbing ──────────────────────────────────────────────────────────

def test_registry_has_every_built_in_rule_registered():
    # Importing schema_intelligence.service triggers `from . import rules`,
    # which registers everything — this just confirms nothing silently
    # failed to register.
    import schema_intelligence.service  # noqa: F401

    column_rule_names = {r.name for r in registered_column_rules()}
    assert {"email", "phone", "date", "currency", "percentage", "primary_key",
            "foreign_key_candidate", "category"}.issubset(column_rule_names)

    dataset_rule_names = {r.name for r in registered_dataset_rules()}
    assert "duplicate_columns" in dataset_rule_names


def test_rule_result_rejects_out_of_range_confidence():
    import pytest
    with pytest.raises(ValueError):
        RuleResult(role="x", confidence=1.5, rule_name="test")


# ── Individual rule tests (pure — no DB) ────────────────────────────────────

def test_primary_key_rule_full_confidence_for_perfectly_unique_column():
    rule = PrimaryKeyRule()
    result = rule.evaluate(_context(pd.Series([1, 2, 3, 4]), "id"))
    assert result is not None
    assert result.role == "primary_key"
    assert result.confidence == 1.0


def test_primary_key_rule_graded_confidence_with_some_duplicates():
    rule = PrimaryKeyRule(min_confidence=0.5)
    # 3 unique out of 4 rows -> uniqueness_ratio 0.75, no nulls -> confidence 0.75
    result = rule.evaluate(_context(pd.Series([1, 1, 2, 3]), "id"))
    assert result is not None
    assert result.confidence == 0.75


def test_primary_key_rule_returns_none_below_floor():
    rule = PrimaryKeyRule(min_confidence=0.9)
    result = rule.evaluate(_context(pd.Series([1, 1, 1, 2]), "id"))  # uniqueness_ratio 0.5
    assert result is None


def test_email_rule_confidence_equals_match_ratio():
    rule = EmailColumnRule()
    series = pd.Series(["a@x.com", "b@x.com", "not-an-email", "c@x.com"])
    result = rule.evaluate(_context(series, "CustomerEmail"))
    assert result is not None
    assert result.role == "email"
    assert result.confidence == 0.75  # 3 of 4 match


def test_email_rule_returns_none_for_non_text_dtype():
    rule = EmailColumnRule()
    result = rule.evaluate(_context(pd.Series([1, 2, 3]), "SomeNumber"))
    assert result is None


def test_currency_rule_requires_numeric_and_name_hint():
    rule = CurrencyColumnRule()
    assert rule.evaluate(_context(pd.Series([10.0, 20.0]), "TotalPrice")) is not None
    assert rule.evaluate(_context(pd.Series([10.0, 20.0]), "Quantity")) is None
    assert rule.evaluate(_context(pd.Series(["10", "20"]), "TotalPrice")) is None  # not numeric dtype


def test_percentage_rule_scores_by_name_and_range():
    rule = PercentageColumnRule()
    good = rule.evaluate(_context(pd.Series([0.1, 0.5, 0.9]), "DiscountPct"))
    assert good is not None
    assert good.confidence == 1.0  # name hint + full range conformance

    out_of_range = rule.evaluate(_context(pd.Series([150, 200]), "DiscountPct"))
    assert out_of_range is None  # confidence 0 -> filtered by min_confidence

    no_hint = rule.evaluate(_context(pd.Series([0.1, 0.5]), "Quantity"))
    assert no_hint is None


def test_categorical_rule_confidence_reflects_cardinality():
    rule = CategoricalColumnRule()
    low_cardinality = pd.Series(["Retail", "Online", "Retail", "Partner"] * 25)  # 100 rows, 3 uniques
    result = rule.evaluate(_context(low_cardinality, "Channel"))
    assert result is not None
    assert result.confidence > 0.9  # very few uniques relative to row count


def test_duplicate_columns_rule_detects_identical_columns():
    df = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Customer_ID_Copy": [1, 2, 3],
        "TotalPrice": [100.0, 200.0, 300.0],
    })
    rule = DuplicateColumnsRule()
    results = rule.evaluate("ds_test", df)
    assert len(results) == 1
    assert results[0].role == "duplicate_column"
    assert results[0].confidence == 1.0
    assert {results[0].evidence["column_a"], results[0].evidence["column_b"]} == {"CustomerID", "Customer_ID_Copy"}


def test_duplicate_columns_rule_tolerates_minor_dirty_data():
    # 9 of 10 rows identical -> 0.9 equality ratio, below the 0.98 default
    # floor, so this should NOT be reported as a duplicate.
    df = pd.DataFrame({
        "A": list(range(10)),
        "B": [0, 1, 2, 3, 4, 5, 6, 7, 8, 999],  # last row differs
    })
    rule = DuplicateColumnsRule(min_confidence=0.98)
    assert rule.evaluate("ds_test", df) == []

    lenient_rule = DuplicateColumnsRule(min_confidence=0.85)
    results = lenient_rule.evaluate("ds_test", df)
    assert len(results) == 1
    assert results[0].confidence == 0.9


# ── Relationship / value-overlap tests (unchanged behavior, new plumbing) ──

def test_compute_value_overlap_full_and_partial():
    source = pd.Series([1, 2, 3])
    target_full = pd.Series([1, 2, 3, 4, 5])
    target_partial = pd.Series([1, 2, 99])
    assert compute_value_overlap(source, target_full) == 1.0
    assert round(compute_value_overlap(source, target_partial), 4) == round(2 / 3, 4)


def test_find_relationship_candidates_matches_fk_against_pk():
    orders_df = pd.DataFrame({"OrderID": [1, 2, 3], "CustomerID": [10, 20, 30]})
    customers_df = pd.DataFrame({"CustomerID": [10, 20, 30, 40], "Name": ["A", "B", "C", "D"]})

    candidates = find_relationship_candidates(
        source_dataset_id="orders", source_df=orders_df, candidate_columns=["CustomerID"],
        other_datasets=[("customers", customers_df)], min_confidence=0.8,
    )
    assert len(candidates) == 1
    assert candidates[0].target_column == "CustomerID"
    assert candidates[0].confidence == 1.0


# ── Service-level tests (real DB via db_session fixture) ──────────────────

def _build_service(db_session):
    dataset_repo = DatasetRepository(db_session)
    return (
        dataset_repo,
        SchemaIntelligenceService(
            dataset_repo,
            RelationshipRepository(db_session),
            ColumnRoleDetectionRepository(db_session),
            DuplicateColumnRepository(db_session),
        ),
    )


def test_analyze_dataset_persists_winning_roles_and_full_candidate_audit_trail(db_session):
    dataset_repo, service = _build_service(db_session)
    registry_service = DatasetRegistryService(dataset_repo)

    df = pd.DataFrame({
        "OrderID": [1, 2, 3],
        "CustomerEmail": ["a@x.com", "b@x.com", "c@x.com"],
        "TotalPrice": [100.0, 200.0, 300.0],
    })
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"orders", organization_id="org_1",
        dataset_name="orders.csv", uploaded_by="p", source_type="csv",
    )

    analysis = service.analyze_dataset(reg.dataset.dataset_id, df)

    assert analysis.column_roles["CustomerEmail"] == "email"
    assert analysis.column_roles["OrderID"] == "primary_key"
    assert analysis.column_roles["TotalPrice"] == "currency"

    # The winning role also landed on the Dataset Registry's OWN column, via
    # its existing update_column_role() method — proving reuse, not duplication.
    persisted_columns = dataset_repo.get_columns(reg.dataset.dataset_id)
    email_col = next(c for c in persisted_columns if c.column_name == "CustomerEmail")
    assert email_col.inferred_role == "email"

    # And the full scored candidate list is queryable independently.
    email_candidates = analysis.column_candidates["CustomerEmail"]
    assert email_candidates[0].role == "email"
    assert email_candidates[0].confidence == 1.0

    detection_repo = ColumnRoleDetectionRepository(db_session)
    stored = detection_repo.list_for_dataset(reg.dataset.dataset_id)
    assert any(d.column_name == "CustomerEmail" and d.role == "email" for d in stored)


def test_analyze_dataset_persists_confidence_and_evidence_on_dataset_registry(db_session):
    # "Persist all metadata inside Dataset Registry": the winning role's
    # confidence/evidence must land on DatasetColumn itself, not just in
    # schema_intelligence's own column_role_detections audit table.
    dataset_repo, service = _build_service(db_session)
    registry_service = DatasetRegistryService(dataset_repo)

    df = pd.DataFrame({
        "OrderID": [1, 2, 3],
        "CustomerEmail": ["a@x.com", "b@x.com", "c@x.com"],
    })
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"orders-conf", organization_id="org_1",
        dataset_name="orders.csv", uploaded_by="p", source_type="csv",
    )

    service.analyze_dataset(reg.dataset.dataset_id, df)

    persisted_columns = dataset_repo.get_columns(reg.dataset.dataset_id)
    email_col = next(c for c in persisted_columns if c.column_name == "CustomerEmail")
    assert email_col.inferred_role == "email"
    assert email_col.inferred_role_confidence == 1.0
    assert email_col.inferred_role_evidence is not None
    assert email_col.role_detected_at is not None

    pk_col = next(c for c in persisted_columns if c.column_name == "OrderID")
    assert pk_col.inferred_role == "primary_key"
    assert pk_col.inferred_role_confidence is not None
    assert pk_col.inferred_role_confidence > 0.0


def test_reanalyzing_clears_confidence_for_columns_that_stop_matching(db_session):
    dataset_repo, service = _build_service(db_session)
    registry_service = DatasetRegistryService(dataset_repo)
    df = pd.DataFrame({"CustomerEmail": ["a@x.com", "b@x.com"]})
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"reanalyze-conf", organization_id="org_1",
        dataset_name="x.csv", uploaded_by="p", source_type="csv",
    )
    service.analyze_dataset(reg.dataset.dataset_id, df)

    non_matching_df = pd.DataFrame({"CustomerEmail": ["z"]})  # 1 row: not email-shaped,
    # and PrimaryKeyRule explicitly declines on a single-row table (every
    # column is trivially "unique" there), so this genuinely produces zero
    # candidates rather than accidentally tripping a different rule.
    service.analyze_dataset(reg.dataset.dataset_id, non_matching_df)

    columns = dataset_repo.get_columns(reg.dataset.dataset_id)
    email_col = next(c for c in columns if c.column_name == "CustomerEmail")
    assert email_col.inferred_role is None
    assert email_col.inferred_role_confidence is None


def test_analyze_dataset_persists_duplicate_columns(db_session):
    dataset_repo, service = _build_service(db_session)
    registry_service = DatasetRegistryService(dataset_repo)

    df = pd.DataFrame({
        "CustomerID": [1, 2, 3],
        "Customer_ID_Copy": [1, 2, 3],
        "TotalPrice": [100.0, 200.0, 300.0],
    })
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"dupe-test", organization_id="org_1",
        dataset_name="dupes.csv", uploaded_by="p", source_type="csv",
    )

    analysis = service.analyze_dataset(reg.dataset.dataset_id, df)
    assert len(analysis.duplicate_columns) == 1

    dup_repo = DuplicateColumnRepository(db_session)
    persisted = dup_repo.list_for_dataset(reg.dataset.dataset_id)
    assert len(persisted) == 1
    assert {persisted[0].column_a, persisted[0].column_b} == {"CustomerID", "Customer_ID_Copy"}


def test_reanalyzing_a_dataset_replaces_prior_detections_not_duplicates(db_session):
    dataset_repo, service = _build_service(db_session)
    registry_service = DatasetRegistryService(dataset_repo)
    df = pd.DataFrame({"CustomerEmail": ["a@x.com", "b@x.com"]})
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"reanalyze", organization_id="org_1",
        dataset_name="x.csv", uploaded_by="p", source_type="csv",
    )

    service.analyze_dataset(reg.dataset.dataset_id, df)
    service.analyze_dataset(reg.dataset.dataset_id, df)  # run twice

    detection_repo = ColumnRoleDetectionRepository(db_session)
    stored = detection_repo.list_for_column(reg.dataset.dataset_id, "CustomerEmail")
    # Exactly one "email" detection row for this column, not two.
    email_rows = [d for d in stored if d.role == "email"]
    assert len(email_rows) == 1


def test_service_works_without_optional_audit_repositories(db_session):
    """Backward-compat: constructing with only the first two arguments
    (pre-rewrite call shape) still runs detection and updates inferred_role
    — it just skips the richer audit-trail persistence."""
    dataset_repo = DatasetRepository(db_session)
    registry_service = DatasetRegistryService(dataset_repo)
    service = SchemaIntelligenceService(dataset_repo, RelationshipRepository(db_session))

    df = pd.DataFrame({"CustomerEmail": ["a@x.com", "b@x.com"]})
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"no-audit-repo", organization_id="org_1",
        dataset_name="x.csv", uploaded_by="p", source_type="csv",
    )
    analysis = service.analyze_dataset(reg.dataset.dataset_id, df)
    assert analysis.column_roles["CustomerEmail"] == "email"


def test_detect_relationships_persists_and_upgrades_role(db_session):
    dataset_repo, service = _build_service(db_session)
    registry_service = DatasetRegistryService(dataset_repo)

    orders_df = pd.DataFrame({
        "OrderID": [1, 2, 3],
        "CustomerID": [10, 20, 30],
        "CustomerEmail": ["a@x.com", "b@x.com", "c@x.com"],
    })
    customers_df = pd.DataFrame({"CustomerID": [10, 20, 30], "Name": ["A", "B", "C"]})

    orders_reg = registry_service.register_dataset(
        df=orders_df, raw_bytes=b"orders", organization_id="org_1",
        dataset_name="orders.csv", uploaded_by="p", source_type="csv",
    )
    customers_reg = registry_service.register_dataset(
        df=customers_df, raw_bytes=b"customers", organization_id="org_1",
        dataset_name="customers.csv", uploaded_by="p", source_type="csv",
    )

    service.analyze_dataset(orders_reg.dataset.dataset_id, orders_df)
    relationships = service.detect_relationships(
        orders_reg.dataset.dataset_id, orders_df,
        other_datasets=[(customers_reg.dataset.dataset_id, customers_df)],
    )
    assert len(relationships) == 1

    updated_columns = dataset_repo.get_columns(orders_reg.dataset.dataset_id)
    customer_id_col = next(c for c in updated_columns if c.column_name == "CustomerID")
    assert customer_id_col.inferred_role == "foreign_key"


def test_detect_relationships_persists_confidence_for_confirmed_foreign_key(db_session):
    dataset_repo, service = _build_service(db_session)
    registry_service = DatasetRegistryService(dataset_repo)

    orders_df = pd.DataFrame({
        "OrderID": [1, 2, 3],
        "CustomerID": [10, 20, 30],
    })
    customers_df = pd.DataFrame({"CustomerID": [10, 20, 30], "Name": ["A", "B", "C"]})

    orders_reg = registry_service.register_dataset(
        df=orders_df, raw_bytes=b"orders-fk-conf", organization_id="org_1",
        dataset_name="orders.csv", uploaded_by="p", source_type="csv",
    )
    customers_reg = registry_service.register_dataset(
        df=customers_df, raw_bytes=b"customers-fk-conf", organization_id="org_1",
        dataset_name="customers.csv", uploaded_by="p", source_type="csv",
    )

    service.analyze_dataset(orders_reg.dataset.dataset_id, orders_df)
    service.detect_relationships(
        orders_reg.dataset.dataset_id, orders_df,
        other_datasets=[(customers_reg.dataset.dataset_id, customers_df)],
    )

    updated_columns = dataset_repo.get_columns(orders_reg.dataset.dataset_id)
    customer_id_col = next(c for c in updated_columns if c.column_name == "CustomerID")
    assert customer_id_col.inferred_role == "foreign_key"
    assert customer_id_col.inferred_role_confidence == 1.0
    assert customer_id_col.inferred_role_evidence["target_dataset_id"] == customers_reg.dataset.dataset_id
    assert customer_id_col.inferred_role_evidence["target_column"] == "CustomerID"
