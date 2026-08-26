# query_history/service.py
# ─────────────────────────────────────────────────────────────────────────────
# Two ways to use this service:
#
#   1. log_execution(...) — call directly when you already have every field
#      in hand (e.g. logging a parsed-but-not-yet-run command, as main.py's
#      /agentic_command hook does today).
#
#   2. track(...) — a context manager that IS the "reusable logging service"
#      requested: wrap the actual execution of a query/pipeline in a `with`
#      block and it automatically times it, detects success/failure
#      (including from a raised exception), and persists the row on exit —
#      without the caller computing timing or remembering to call
#      log_execution() itself. Critically, it NEVER swallows an exception:
#      logging a failure and then re-raising means dropping this into an
#      existing endpoint changes nothing about that endpoint's behavior or
#      error handling — pure additive observability, which is the whole
#      point given "do not modify existing APIs".
# ─────────────────────────────────────────────────────────────────────────────

import time
import traceback
from types import TracebackType

from datasets.repository import DatasetRepository

from .models import QueryHistory
from .repository import QueryHistoryRepository


class QueryHistoryService:
    def __init__(self, repository: QueryHistoryRepository, dataset_repository: DatasetRepository | None = None):
        self.repository = repository
        # Optional — only needed to auto-resolve `schema_hash` from
        # `dataset_id` (see _resolve_schema_hash). Reuses the Dataset
        # Registry's existing repository/read path; nothing about that
        # package is touched.
        self.dataset_repository = dataset_repository

    def _resolve_schema_hash(self, dataset_id: str | None) -> str | None:
        if dataset_id is None or self.dataset_repository is None:
            return None
        dataset = self.dataset_repository.get_by_id(dataset_id)
        return dataset.schema_hash if dataset is not None else None

    def log_execution(
        self,
        *,
        user_query: str,
        intent: str | None = None,
        generated_sql: str | None = None,
        python_pipeline: dict | list | None = None,
        visualization: dict | None = None,
        execution_time_ms: float | None = None,
        rows_returned: int | None = None,
        dataset_id: str | None = None,
        organization_id: str | None = None,
        success: bool = True,
        error_message: str | None = None,
        planner_version: str | None = None,
    ) -> QueryHistory:
        entry = QueryHistory(
            dataset_id=dataset_id,
            organization_id=organization_id,
            schema_hash=self._resolve_schema_hash(dataset_id),
            user_query=user_query,
            intent=intent,
            generated_sql=generated_sql,
            python_pipeline=python_pipeline,
            visualization=visualization,
            execution_time_ms=execution_time_ms,
            rows_returned=rows_returned,
            success=success,
            error_message=error_message,
            planner_version=planner_version,
        )
        return self.repository.create(entry)

    def track(
        self,
        *,
        user_query: str,
        dataset_id: str | None = None,
        organization_id: str | None = None,
        intent: str | None = None,
        planner_version: str | None = None,
    ) -> "QueryExecutionTracker":
        """Returns a context manager — see module docstring and
        QueryExecutionTracker below for the actual behavior."""
        return QueryExecutionTracker(
            self,
            user_query=user_query,
            dataset_id=dataset_id,
            organization_id=organization_id,
            intent=intent,
            planner_version=planner_version,
        )

    def record_feedback(self, entry_id: int, feedback_score: int) -> QueryHistory | None:
        return self.repository.set_feedback(entry_id, feedback_score)

    def get_history(
        self,
        *,
        organization_id: str | None = None,
        dataset_id: str | None = None,
        success: bool | None = None,
        planner_version: str | None = None,
        limit: int = 50,
    ) -> list[QueryHistory]:
        return self.repository.list_recent(
            organization_id=organization_id,
            dataset_id=dataset_id,
            success=success,
            planner_version=planner_version,
            limit=limit,
        )

    def get_training_examples(self, *, schema_hash: str | None = None, only_successful: bool = True, limit: int = 5000) -> list[QueryHistory]:
        """See QueryHistoryRepository.list_for_training — the entry point a
        future ML training job would call."""
        return self.repository.list_for_training(schema_hash=schema_hash, only_successful=only_successful, limit=limit)

    def find_reusable_plan(self, *, user_query: str, dataset_id: str | None) -> QueryHistory | None:
        """Deterministic exact-match lookup against past SUCCESSFUL runs —
        see repository docstring. Retained here for direct same-dataset
        lookups; plan_cache/service.py builds on top of this same table for
        the cross-dataset (same schema_hash, different dataset) case.
        """
        matches = self.repository.find_similar_successful(user_query=user_query, dataset_id=dataset_id, limit=1)
        return matches[0] if matches else None


class QueryExecutionTracker:
    """Reusable instrumentation for ONE query execution. Usage:

        with query_history_service.track(user_query=text, dataset_id=ds_id, intent="pivot") as tracker:
            result = actually_run_the_query(...)
            tracker.set_result(generated_sql=sql, rows_returned=len(result.rows))

    On successful exit: logs execution_time_ms (measured automatically),
    success=True, and whatever set_result() provided.
    On an exception inside the `with` block: logs success=False with the
    exception's message as error_message, THEN RE-RAISES — this never
    changes what the wrapped code does, only observes it.
    """

    def __init__(
        self,
        service: QueryHistoryService,
        *,
        user_query: str,
        dataset_id: str | None,
        organization_id: str | None,
        intent: str | None,
        planner_version: str | None = None,
    ):
        self.service = service
        self.user_query = user_query
        self.dataset_id = dataset_id
        self.organization_id = organization_id
        self.intent = intent
        self.planner_version = planner_version

        self.generated_sql: str | None = None
        self.python_pipeline: dict | list | None = None
        self.visualization: dict | None = None
        self.rows_returned: int | None = None

        self._start_time: float | None = None

    def set_result(
        self,
        *,
        generated_sql: str | None = None,
        python_pipeline: dict | list | None = None,
        visualization: dict | None = None,
        rows_returned: int | None = None,
        planner_version: str | None = None,
    ) -> None:
        """Call this from inside the `with` block once the actual
        query/pipeline has run, to attach what got produced. Every argument
        is optional and only overwrites when explicitly provided, so you can
        call it more than once (e.g. once the SQL is known, again once row
        count is known) without clobbering earlier values with None.
        `planner_version` can also be set here (rather than only at
        `track()` time) for cases where which planner actually handled the
        query — e.g. a fallback planner after a primary one declined — is
        only known once execution is underway."""
        if generated_sql is not None:
            self.generated_sql = generated_sql
        if python_pipeline is not None:
            self.python_pipeline = python_pipeline
        if visualization is not None:
            self.visualization = visualization
        if rows_returned is not None:
            self.rows_returned = rows_returned
        if planner_version is not None:
            self.planner_version = planner_version

    def __enter__(self) -> "QueryExecutionTracker":
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        elapsed_ms = (time.perf_counter() - (self._start_time or time.perf_counter())) * 1000
        success = exc_type is None
        error_message = None
        if not success:
            error_message = "".join(traceback.format_exception_only(exc_type, exc_value)).strip()

        try:
            self.service.log_execution(
                user_query=self.user_query,
                intent=self.intent,
                generated_sql=self.generated_sql,
                python_pipeline=self.python_pipeline,
                visualization=self.visualization,
                execution_time_ms=round(elapsed_ms, 3),
                rows_returned=self.rows_returned,
                dataset_id=self.dataset_id,
                organization_id=self.organization_id,
                success=success,
                error_message=error_message,
                planner_version=self.planner_version,
            )
        except Exception:
            # Logging itself must NEVER be able to break the caller — if
            # persistence fails for any reason, swallow that failure (it's
            # printed for visibility) rather than let an observability
            # problem masquerade as, or compound, the real error.
            traceback.print_exc()

        return False  # never suppress the original exception, if any
