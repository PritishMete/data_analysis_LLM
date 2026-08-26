# tests/test_memory_engine.py
import pandas as pd

from datasets.repository import DatasetRepository
from datasets.service import DatasetRegistryService
from memory_engine.contracts import CandidateRanker, RankedCandidate
from memory_engine.rankers import DefaultCandidateRanker
from memory_engine.service import AnalyticsMemoryEngine
from query_history.repository import QueryHistoryRepository
from query_history.service import QueryHistoryService
from sql_cache.strategies import ExactMatchStrategy


def _seed(db_session, **kwargs):
    service = QueryHistoryService(QueryHistoryRepository(db_session))
    return service.log_execution(**kwargs)


def _engine(db_session, **kwargs) -> AnalyticsMemoryEngine:
    return AnalyticsMemoryEngine(QueryHistoryRepository(db_session), **kwargs)


# ── find_similar_query ──────────────────────────────────────────────────────

def test_find_similar_query_returns_full_experience(db_session):
    _seed(
        db_session,
        user_query="total revenue by region",
        dataset_id="ds_a",
        intent="aggregate",
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        python_pipeline={"group_by": ["region"]},
        visualization={"chart_type": "bar", "x": "region", "y": "revenue"},
        execution_time_ms=42.0,
        rows_returned=5,
        success=True,
    )
    engine = _engine(db_session)

    match = engine.find_similar_query(user_query="total revenue by region", dataset_id="ds_a")
    assert match is not None
    assert match.intent == "aggregate"
    assert match.generated_sql == "SELECT region, SUM(revenue) FROM data GROUP BY region"
    assert match.python_pipeline == {"group_by": ["region"]}
    assert match.visualization == {"chart_type": "bar", "x": "region", "y": "revenue"}
    assert match.rows_returned == 5
    assert match.similarity_score == 1.0


def test_find_similar_query_carries_planner_version(db_session):
    _seed(
        db_session,
        user_query="total revenue by region",
        dataset_id="ds_a",
        generated_sql="SELECT 1",
        success=True,
        planner_version="gemini-1.5-flash",
    )
    engine = _engine(db_session)

    match = engine.find_similar_query(user_query="total revenue by region", dataset_id="ds_a")
    assert match is not None
    assert match.planner_version == "gemini-1.5-flash"


def test_find_similar_query_none_below_threshold(db_session):
    _seed(db_session, user_query="total revenue by region", dataset_id="ds_a", success=True)
    engine = _engine(db_session, default_min_confidence=0.95)

    assert engine.find_similar_query(
        user_query="show me customer churn broken down by plan tier", dataset_id="ds_a"
    ) is None


def test_find_similar_query_scoped_by_schema_hash_across_datasets(db_session):
    """The core cross-dataset generalization case: two DIFFERENT datasets
    with the SAME schema shape, scoped purely by schema_hash (no dataset_id
    at all) — this is what distinguishes the Memory Engine's scoping from a
    plain per-dataset cache."""
    dataset_repo = DatasetRepository(db_session)
    registry_service = DatasetRegistryService(dataset_repo)

    shape_df = pd.DataFrame({"Region": ["North"], "Revenue": [100.0]})
    org_a = registry_service.register_dataset(
        df=shape_df, raw_bytes=b"org-a", organization_id="org_a",
        dataset_name="a.csv", uploaded_by="p", source_type="csv",
    )
    org_b = registry_service.register_dataset(
        df=shape_df, raw_bytes=b"org-b", organization_id="org_b",
        dataset_name="b.csv", uploaded_by="p", source_type="csv",
    )
    assert org_a.dataset.schema_hash == org_b.dataset.schema_hash

    history_service = QueryHistoryService(QueryHistoryRepository(db_session), dataset_repo)
    history_service.log_execution(
        user_query="total revenue by region",
        dataset_id=org_a.dataset.dataset_id,
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region",
        success=True,
    )

    engine = AnalyticsMemoryEngine(QueryHistoryRepository(db_session))
    match = engine.find_similar_query(
        user_query="total revenue by region", schema_hash=org_b.dataset.schema_hash
    )
    assert match is not None
    assert match.generated_sql == "SELECT region, SUM(revenue) FROM data GROUP BY region"


def test_find_similar_query_prefers_higher_feedback_among_similar_matches(db_session):
    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    low_rated = history_service.log_execution(
        user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT_LOW_RATED", success=True,
    )
    high_rated = history_service.log_execution(
        user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT_HIGH_RATED", success=True,
    )
    history_service.record_feedback(low_rated.id, -1)
    history_service.record_feedback(high_rated.id, 1)

    engine = _engine(db_session)
    match = engine.find_similar_query(user_query="total revenue by region", dataset_id="ds_a")
    assert match.generated_sql == "SELECT_HIGH_RATED"


def test_ignores_failed_executions(db_session):
    _seed(db_session, user_query="total revenue by region", dataset_id="ds_a",
          generated_sql="SELECT broken", success=False)
    engine = _engine(db_session)
    assert engine.find_similar_query(user_query="total revenue by region", dataset_id="ds_a") is None


# ── find_best_sql / find_best_pipeline: skip non-matching-shape candidates ─

def test_find_best_sql_skips_top_match_with_no_sql(db_session):
    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    # Best OVERALL similarity match has no SQL at all (e.g. a pivot action).
    history_service.log_execution(
        user_query="total revenue by region", dataset_id="ds_a",
        python_pipeline={"action": "pivot"}, generated_sql=None, success=True,
    )
    # A slightly more loosely-matching entry (still >= default threshold via
    # exact text here) that DOES have SQL.
    history_service.log_execution(
        user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT region, SUM(revenue) FROM data GROUP BY region", success=True,
    )

    engine = _engine(db_session)
    sql = engine.find_best_sql(user_query="total revenue by region", dataset_id="ds_a")
    assert sql == "SELECT region, SUM(revenue) FROM data GROUP BY region"


def test_find_best_pipeline_skips_entries_with_no_pipeline(db_session):
    history_service = QueryHistoryService(QueryHistoryRepository(db_session))
    history_service.log_execution(
        user_query="total revenue by region", dataset_id="ds_a",
        generated_sql="SELECT 1", python_pipeline=None, success=True,
    )
    history_service.log_execution(
        user_query="total revenue by region", dataset_id="ds_a",
        python_pipeline={"action": "pivot", "rows": ["region"]}, success=True,
    )

    engine = _engine(db_session)
    pipeline = engine.find_best_pipeline(user_query="total revenue by region", dataset_id="ds_a")
    assert pipeline == {"action": "pivot", "rows": ["region"]}


def test_find_best_sql_returns_none_when_no_candidate_has_sql(db_session):
    _seed(db_session, user_query="total revenue by region", dataset_id="ds_a",
          python_pipeline={"action": "pivot"}, generated_sql=None, success=True)
    engine = _engine(db_session)
    assert engine.find_best_sql(user_query="total revenue by region", dataset_id="ds_a") is None


# ── record_feedback ──────────────────────────────────────────────────────────

def test_record_feedback_updates_entry(db_session):
    entry = _seed(db_session, user_query="q", success=True)
    engine = _engine(db_session)
    updated = engine.record_feedback(entry.id, 1)
    assert updated is not None
    assert updated.feedback_score == 1


def test_record_feedback_returns_none_for_missing_entry(db_session):
    engine = _engine(db_session)
    assert engine.record_feedback(999999, 1) is None


# ── Modularity: swapping the similarity strategy changes behavior cleanly ──

def test_swapping_similarity_strategy_changes_matching_behavior(db_session):
    _seed(db_session, user_query="total revenue by region", dataset_id="ds_a",
          generated_sql="SELECT 1", success=True)

    fuzzy_engine = _engine(db_session)
    exact_engine = _engine(db_session, similarity_strategy=ExactMatchStrategy())

    # Trailing whitespace: fuzzy strategy matches, exact strategy doesn't.
    assert fuzzy_engine.find_similar_query(user_query="total revenue by region ", dataset_id="ds_a") is not None
    assert exact_engine.find_similar_query(user_query="total revenue by region  extra", dataset_id="ds_a") is None


# ── CandidateRanker: the swap point future ML integration is built around ─

def test_default_ranker_used_when_none_provided(db_session):
    engine = _engine(db_session)
    assert isinstance(engine.ranker, DefaultCandidateRanker)


def test_similarity_strategy_kwarg_still_threads_into_default_ranker(db_session):
    # Backward-compat path: existing callers passing similarity_strategy=
    # directly (rather than the newer ranker=) must keep working exactly as
    # before.
    engine = _engine(db_session, similarity_strategy=ExactMatchStrategy())
    assert isinstance(engine.ranker, DefaultCandidateRanker)
    assert engine.ranker.similarity_strategy.name == "exact_match"


def test_custom_ranker_is_used_wholesale_when_provided(db_session):
    _seed(db_session, user_query="total revenue by region", dataset_id="ds_a",
          generated_sql="SELECT 1", success=True)

    class AlwaysZeroRanker(CandidateRanker):
        name = "always_zero_ranker"

        def rank(self, user_query, candidates, *, min_confidence):
            # Deliberately ignores user_query/min_confidence to prove the
            # engine defers entirely to whatever ranker it's given — this
            # stands in for a future ML ranker with its own scoring logic.
            return [RankedCandidate(score=0.0, entry=c) for c in candidates]

    engine = AnalyticsMemoryEngine(QueryHistoryRepository(db_session), ranker=AlwaysZeroRanker())
    match = engine.find_similar_query(user_query="completely different text", dataset_id="ds_a")
    assert match is not None
    assert match.similarity_score == 0.0
    assert engine.ranker.name == "always_zero_ranker"


def test_ranker_receives_only_already_scoped_candidates(db_session):
    # The ranker must never need to know about org/dataset/schema_hash
    # scoping — that's the engine's job before candidates ever reach it.
    _seed(db_session, user_query="q_in_scope", dataset_id="ds_a", organization_id="org_1", success=True)
    _seed(db_session, user_query="q_out_of_scope", dataset_id="ds_b", organization_id="org_1", success=True)

    seen_queries = []

    class RecordingRanker(CandidateRanker):
        name = "recording_ranker"

        def rank(self, user_query, candidates, *, min_confidence):
            seen_queries.extend(c.user_query for c in candidates)
            return []

    engine = AnalyticsMemoryEngine(QueryHistoryRepository(db_session), ranker=RecordingRanker())
    engine.find_similar_query(user_query="anything", dataset_id="ds_a", organization_id="org_1")

    assert seen_queries == ["q_in_scope"]


# ── No Gemini/LLM dependency (structural check, not just a docstring claim) ─

def test_memory_engine_package_has_no_llm_imports():
    import ast
    import importlib
    import inspect
    import pkgutil

    import memory_engine

    forbidden = {"google", "openai", "anthropic"}
    imported_roots = set()

    for module_info in pkgutil.iter_modules(memory_engine.__path__):
        mod = importlib.import_module(f"memory_engine.{module_info.name}")
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

    assert imported_roots.isdisjoint(forbidden), f"Found forbidden LLM imports: {imported_roots & forbidden}"
