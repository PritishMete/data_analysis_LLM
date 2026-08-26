# tests/test_dataset_registry.py
import pandas as pd

from datasets.hashing import compute_file_hash, compute_schema_hash
from datasets.repository import DatasetRepository
from datasets.service import DatasetRegistryService


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "OrderID": [1, 2, 3, 4],
        "CustomerEmail": ["a@x.com", "b@x.com", "c@x.com", None],
        "TotalPrice": [100.0, 200.0, None, 400.0],
    })


def test_register_dataset_creates_dataset_and_columns(db_session):
    repo = DatasetRepository(db_session)
    service = DatasetRegistryService(repo)
    df = _sample_df()
    raw_bytes = b"OrderID,CustomerEmail,TotalPrice\n1,a@x.com,100\n"

    result = service.register_dataset(
        df=df,
        raw_bytes=raw_bytes,
        organization_id="org_1",
        dataset_name="orders.csv",
        uploaded_by="priyanka",
        source_type="csv",
    )

    assert result.was_duplicate is False
    assert result.dataset.row_count == 4
    assert result.dataset.column_count == 3
    assert result.dataset.organization_id == "org_1"
    assert len(result.columns) == 3

    total_price_col = next(c for c in result.columns if c.column_name == "TotalPrice")
    assert total_price_col.nullable is True
    assert total_price_col.missing_percentage == 25.0  # 1 of 4 rows missing

    order_id_col = next(c for c in result.columns if c.column_name == "OrderID")
    assert order_id_col.nullable is False
    assert order_id_col.unique_count == 4


def test_register_dataset_is_idempotent_on_exact_reupload(db_session):
    repo = DatasetRepository(db_session)
    service = DatasetRegistryService(repo)
    df = _sample_df()
    raw_bytes = b"identical-bytes"

    first = service.register_dataset(
        df=df, raw_bytes=raw_bytes, organization_id="org_1",
        dataset_name="orders.csv", uploaded_by="priyanka", source_type="csv",
    )
    second = service.register_dataset(
        df=df, raw_bytes=raw_bytes, organization_id="org_1",
        dataset_name="orders.csv", uploaded_by="priyanka", source_type="csv",
    )

    assert second.was_duplicate is True
    assert second.dataset.dataset_id == first.dataset.dataset_id

    # No duplicate row was inserted.
    all_for_org = repo.list_by_organization("org_1")
    assert len(all_for_org) == 1


def test_same_bytes_different_organization_creates_separate_dataset(db_session):
    repo = DatasetRepository(db_session)
    service = DatasetRegistryService(repo)
    df = _sample_df()
    raw_bytes = b"shared-bytes-across-orgs"

    org_a = service.register_dataset(
        df=df, raw_bytes=raw_bytes, organization_id="org_a",
        dataset_name="orders.csv", uploaded_by="alice", source_type="csv",
    )
    org_b = service.register_dataset(
        df=df, raw_bytes=raw_bytes, organization_id="org_b",
        dataset_name="orders.csv", uploaded_by="bob", source_type="csv",
    )

    assert org_a.dataset.dataset_id != org_b.dataset.dataset_id
    assert org_a.was_duplicate is False
    assert org_b.was_duplicate is False


def test_compute_file_hash_is_deterministic():
    assert compute_file_hash(b"abc") == compute_file_hash(b"abc")
    assert compute_file_hash(b"abc") != compute_file_hash(b"abd")


def test_compute_schema_hash_ignores_column_order_and_case():
    a = compute_schema_hash([("OrderID", "int64"), ("Total Price", "float64")])
    b = compute_schema_hash([("total price", "float64"), ("orderid", "int64")])
    c = compute_schema_hash([("OrderID", "int64"), ("Total Price", "object")])

    assert a == b  # order + case-insensitive name shouldn't matter
    assert a != c  # a genuinely different dtype should
