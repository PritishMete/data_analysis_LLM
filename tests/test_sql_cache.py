# tests/test_sql_cache.py
import pandas as pd

from datasets.repository import DatasetRepository
from datasets.service import DatasetRegistryService
from query_history.repository import QueryHistoryRepository
from query_history.service import QueryHistoryService
from sql_cache.service import SqlCacheService
from sql_cache.strategies import ExactMatchStrategy, TextSimilarityStrategy


# ── SimilarityStrategy now canonically lives in memory_engine/ — sql_cache
# just re-exports it (see sql_cache/contracts.py, sql_cache/strategies.py).
# Lock in that these are re-exports (identical objects), not copies that
# could silently drift apart. ─────────────────────────────────────────────

def test_sql_cache_contracts_reexports_memory_engine_contract():
    from memory_engine.contracts import SimilarityStrategy as CanonicalSimilarityStrategy
    from sql_cache.contracts import SimilarityStrategy as ReexportedSimilarityStrategy

    assert ReexportedSimilarityStrategy is CanonicalSimilarityStrategy


def test_sql_cache_strategies_reexports_memory_engine_strategies():
    from memory_engine.strategies import ExactMatchStrategy as CanonicalExact
    from memory_engine.strategies import TextSimilarityStrategy as CanonicalText

    assert ExactMatchStrategy is CanonicalExact
    assert TextSimilarityStrategy is CanonicalText


# ── TextSimilarityStrategy: pure, no DB ─────────────────────────────────────

def test_identical_queries_score_1():
    strategy = TextSimilarityStrategy()
    assert strategy.score("total revenue by region", "total revenue by region") == 1.0


def test_case_and_whitespace_differences_still_score_near_1():
    strategy = TextSimilarityStrategy()
    score = strategy.score("Total Revenue By Region", "total   revenue by region  ")
    assert score >= 0.99


def test_single_word_typo_lowers_confidence_via_token_mismatch():
    # "revenue" -> "revenu" is a single missing character, but it changes an
    # ENTIRE TOKEN — the token-overlap half of the blend correctly treats
    # that as a genuinely different word (a typo could just as easily be a
    # different word entirely), pulling the combined score down from the
    # near-1.0 a purely character-level metric alone would give it. This is
    # intentional, conservative behavior for a 95%-threshold cache: a typo
    # that lands on a semantically-loaded word should NOT sail through as
    # "safe to reuse" just because it looks close character-by-character.
    strategy = TextSimilarityStrategy()
    score = strategy.score("total revenue by region", "total revenu by region")
    assert 0.6 < score < 0.9
    assert score < 0.95  # correctly falls below the default reuse threshold


def test_reordered_words_score_reasonably_via_token_overlap():
    strategy = TextSimilarityStrategy()
    score = strategy.score("revenue total by region", "region by total revenue")
    assert score > 0.5  # same tokens, different order


def test_unrelated_queries_score_low():
    strategy = TextSimilarityStrategy()
    score = strategy.score("total revenue by region", "list all customers with overdue invoices")
    assert score < 0.4


def test_empty_query_scores_zero():
    strategy = TextSimilarityStrategy()
    assert strategy.score("", "total revenue by region") == 0.0
    assert strategy.score("total revenue by region", "") == 0.0


def test_invalid_weights_raise():
    import pytest
    with pytest.raises(ValueError):
        TextSimilarityStrategy(char_weight=0.7, token_weight=0.7)  # doesn't sum to 1.0
    with pytest.raises(ValueError):
        TextSimilarityStrategy(char_weight=1.5, token_weight=-0.5)


def test_exact_match_strategy_is_binary():
    strategy = ExactMatchStrategy()
    assert strategy.score("total revenue by region", "Total Revenue By Region ") == 1.0
    assert strategy.score("total revenue by region", "total revenu by region") == 0.0  # even a tiny typo -> 0


# ── SqlCacheService: real DB via db_session fixture ────────────────────────

def _seed_history(db_session, **kwargs) -> QueryHistoryService:
    service = QueryHistoryService(QueryHistoryRepository(db_session))
    service.log_execution(**kwargs)
    return service


def test_finds_near_duplicate_above_default_threshold(db_session):
    history_service = _seed_history(
        db_session,
        user_query="total revenue by region",
        dataset_id="ds_a",
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        success=True,
    )
    cache_service = SqlCacheService(QueryHistoryRepository(db_session))

    hit = cache_service.find_similar_cached_query(
        user_query="total revenue by region ", dataset_id="ds_a"  # trailing space only
    )
    assert hit is not None
    assert hit.similarity_score >= 0.95
    assert hit.generated_sql == "SELECT region, SUM(revenue) FROM data GROUP BY region"


def test_cache_hit_carries_planner_version(db_session):
    _seed_history(
        db_session,
        user_query="total revenue by region",
        dataset_id="ds_a",
        generated_sql="SELECT 1",
        success=True,
        planner_version="gemini-1.5-flash",
    )
    cache_service = SqlCacheService(QueryHistoryRepository(db_session))

    hit = cache_service.find_similar_cached_query(user_query="total revenue by region", dataset_id="ds_a")
    assert hit is not None
    assert hit.planner_version == "gemini-1.5-flash"


def test_rejects_match_below_confidence_threshold(db_session):
    _seed_history(
        db_session, user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT 1", success=True,
    )
    cache_service = SqlCacheService(QueryHistoryRepository(db_session), min_confidence=0.95)

    # Meaningfully different phrasing — should NOT clear 95%.
    hit = cache_service.find_similar_cached_query(
        user_query="show me total sales grouped by customer segment", dataset_id="ds_a"
    )
    assert hit is None


def test_threshold_is_configurable_and_respected(db_session):
    _seed_history(
        db_session, user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT 1", success=True,
    )
    lenient_service = SqlCacheService(QueryHistoryRepository(db_session), min_confidence=0.5)
    strict_service = SqlCacheService(QueryHistoryRepository(db_session), min_confidence=0.99)

    query = "revenue total by region"  # reordered — moderate similarity, not near-identical
    lenient_hit = lenient_service.find_similar_cached_query(user_query=query, dataset_id="ds_a")
    strict_hit = strict_service.find_similar_cached_query(user_query=query, dataset_id="ds_a")

    assert lenient_hit is not None
    assert strict_hit is None


def test_ignores_failed_executions_as_candidates(db_session):
    _seed_history(
        db_session, user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT broken sql", success=False,
    )
    cache_service = SqlCacheService(QueryHistoryRepository(db_session))
    hit = cache_service.find_similar_cached_query(user_query="total revenue by region", dataset_id="ds_a")
    assert hit is None  # only a FAILED prior run exists — nothing safe to reuse


def test_scoped_to_dataset_when_provided(db_session):
    _seed_history(
        db_session, user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT 1", success=True,
    )
    cache_service = SqlCacheService(QueryHistoryRepository(db_session))

    # Same exact text, but scoped to a DIFFERENT dataset -> no candidates at all.
    hit = cache_service.find_similar_cached_query(user_query="total revenue by region", dataset_id="ds_b")
    assert hit is None


def test_swapping_strategy_changes_behavior_without_touching_service(db_session):
    _seed_history(
        db_session, user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT 1", success=True,
    )
    # A typo that TextSimilarityStrategy would happily match, but
    # ExactMatchStrategy — injected instead, nothing else changed — rejects.
    exact_only_service = SqlCacheService(
        QueryHistoryRepository(db_session), similarity_strategy=ExactMatchStrategy(), min_confidence=0.95
    )
    hit = exact_only_service.find_similar_cached_query(user_query="total revenu by region", dataset_id="ds_a")
    assert hit is None


def test_picks_best_match_among_multiple_candidates(db_session):
    service = QueryHistoryService(QueryHistoryRepository(db_session))
    service.log_execution(user_query="total revenue by region", dataset_id="ds_a", generated_sql="SQL_A", success=True)
    service.log_execution(user_query="count of orders by region", dataset_id="ds_a", generated_sql="SQL_B", success=True)

    cache_service = SqlCacheService(QueryHistoryRepository(db_session), min_confidence=0.0)
    hit = cache_service.find_similar_cached_query(user_query="total revenue by region", dataset_id="ds_a")
    assert hit.generated_sql == "SQL_A"  # the closer match, not just the first/last logged


def test_real_dataset_scoping_end_to_end(db_session):
    """Full path: register a real dataset, log a real execution against it,
    confirm the cache finds it scoped by the real dataset_id."""
    dataset_repo = DatasetRepository(db_session)
    registry_service = DatasetRegistryService(dataset_repo)
    history_service = QueryHistoryService(QueryHistoryRepository(db_session), dataset_repo)

    df = pd.DataFrame({"Region": ["North"], "Revenue": [100.0]})
    reg = registry_service.register_dataset(
        df=df, raw_bytes=b"sales", organization_id="org_1",
        dataset_name="sales.csv", uploaded_by="p", source_type="csv",
    )
    history_service.log_execution(
        user_query="total revenue by region",
        dataset_id=reg.dataset.dataset_id,
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        success=True,
    )

    cache_service = SqlCacheService(QueryHistoryRepository(db_session))
    hit = cache_service.find_similar_cached_query(
        user_query="total revenue by region", dataset_id=reg.dataset.dataset_id
    )
    assert hit is not None
    assert hit.generated_sql == "SELECT region, SUM(revenue) FROM data GROUP BY region"
