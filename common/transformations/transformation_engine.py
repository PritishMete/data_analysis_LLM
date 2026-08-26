# common/transformations/transformation_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# THE single place every transformation (range binning, rename, drop, fill
# missing, dedupe, merge/split columns, type conversion, date features, ...)
# gets located, validated, previewed, applied, and pushed through the SAME
# downstream analytics pipeline every other route in this project already
# uses: common/report/orchestrator.generate_structured_report_data().
#
# This does NOT introduce a second analytics pipeline. It calls the exact
# same orchestrator function main.py's /transform/range_binning and
# query_router.py's fast-path already call — see common/report/orchestrator.py.
# Nothing in kpi_detector.py, trend_detector.py, outlier_detector.py,
# recommendation_engine.py, chart_recommender.py, or executive_summary.py is
# reimplemented here.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_history import TransformationHistory
from common.transformations.transformation_registry import (
    all_transformations,
    detect_transformation,
    get as registry_get,
)
from common.transformations.transformation_result import TransformationResult
from common.report.orchestrator import ALL_ANALYSIS_IDS, generate_structured_report_data


def _detect_value_column(df: pd.DataFrame, exclude: set[str]) -> str | None:
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    return numeric[0] if numeric else None


def _compute_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Lightweight schema summary — column names/dtypes/counts — computed
    directly from the dataframe. This mirrors the shape of main.py's
    analyze_dataframe()['summary'] without importing main.py (avoiding a
    main.py <-> engine circular import); main.py's analyze_dataframe remains
    the canonical full endpoint-level schema/statistics generator and is
    unchanged by this module.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
    return {
        "columns": list(df.columns),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


def diff_schema(before_df: pd.DataFrame, after_df: pd.DataFrame) -> dict[str, Any]:
    """Added/removed columns + dtype changes between two dataframes — the
    "Refresh Schema" step: added/removed columns, changed datatypes,
    updated numeric/categorical columns.
    """
    before_cols, after_cols = set(before_df.columns), set(after_df.columns)
    added = sorted(after_cols - before_cols)
    removed = sorted(before_cols - after_cols)
    changed_dtypes = {
        str(c): {"before": str(before_df[c].dtype), "after": str(after_df[c].dtype)}
        for c in before_cols & after_cols
        if str(before_df[c].dtype) != str(after_df[c].dtype)
    }
    return {
        "added_columns": added,
        "removed_columns": removed,
        "changed_dtypes": changed_dtypes,
        "before": _compute_schema(before_df),
        "after": _compute_schema(after_df),
    }


class TransformationEngine:
    """Receives a DataFrame + (user query OR explicit transformation name/
    params), locates the right registered transformation, validates,
    previews, applies, refreshes ALL downstream analytics via the existing
    orchestrator, and returns one TransformationResult.
    """

    def __init__(self, registry: dict[str, BaseTransformation] | None = None):
        # `registry` override exists purely for isolated unit testing —
        # production code always uses the real global registry.
        self._registry = registry

    def registry_get(self, name: str) -> BaseTransformation | None:
        if self._registry is not None:
            return self._registry.get(name)
        return registry_get(name)

    def _locate(
        self, df: pd.DataFrame, query: str | None, transformation_name: str | None
    ) -> tuple[BaseTransformation, dict] | None:
        if transformation_name:
            transformation = self.registry_get(transformation_name)
            if transformation is None:
                return None
            return transformation, {"detected": True, "params": {}, "confidence": 1.0}
        if query:
            if self._registry is not None:
                best, best_conf = None, 0.0
                for t in self._registry.values():
                    d = t.detect(query, df)
                    if d.get("detected") and float(d.get("confidence", 0)) > best_conf:
                        best, best_conf = t, float(d.get("confidence", 0))
                return (best, {}) if best else None
            return detect_transformation(query, df)
        return None

    # ── Preview (no mutation, no history entry) ─────────────────────────
    def preview(
        self,
        df: pd.DataFrame,
        transformation_name: str | None = None,
        query: str | None = None,
        params: dict[str, Any] | None = None,
        sample_rows: int = 10,
    ) -> TransformationResult:
        start = time.perf_counter()
        located = self._locate(df, query, transformation_name)
        if located is None:
            return TransformationResult.failure(
                "Could not locate a matching transformation for this request.",
                execution_time=time.perf_counter() - start,
            )
        transformation, detection = located
        resolved_params = {**detection.get("params", {}), **(params or {})}

        try:
            transformation.validate(df, resolved_params)
            preview_data = transformation.preview(df, resolved_params, sample_rows=sample_rows)
        except TransformationError as e:
            return TransformationResult.failure(str(e), execution_time=time.perf_counter() - start)

        return TransformationResult(
            success=True,
            transformation={"name": transformation.name, "display_name": transformation.display_name},
            preview=preview_data,
            execution_time=time.perf_counter() - start,
            message=f"Preview generated for {transformation.display_name}.",
        )

    # ── Apply (mutates the working dataframe, refreshes the pipeline) ───
    def run(
        self,
        df: pd.DataFrame,
        transformation_name: str | None = None,
        query: str | None = None,
        params: dict[str, Any] | None = None,
        history: TransformationHistory | None = None,
        value_column: str | None = None,
        refresh_analytics: bool = True,
    ) -> TransformationResult:
        start = time.perf_counter()

        # HARDENING: _locate() calls .detect() on every candidate
        # transformation in the registry (not just the one that ends up
        # matching) to find the best one — so a bug in ANY registered
        # transformation's detect() (a bad regex match, an unexpected dtype,
        # etc.) previously crashed the request for EVERY query, not just
        # ones related to that transformation. This must never propagate:
        # the engine's whole contract is "always return a TransformationResult,
        # never raise."
        try:
            located = self._locate(df, query, transformation_name)
        except Exception as e:
            return TransformationResult.failure(
                f"Could not evaluate this request against the transformation registry: {e}",
                execution_time=time.perf_counter() - start,
            )
        if located is None:
            return TransformationResult.failure(
                "Could not locate a matching transformation for this request.",
                execution_time=time.perf_counter() - start,
            )
        transformation, detection = located
        resolved_params = {**detection.get("params", {}), **(params or {})}

        try:
            transformation.validate(df, resolved_params)
        except TransformationError as e:
            return TransformationResult.failure(str(e), execution_time=time.perf_counter() - start)

        # HARDENING: preview() runs the transformation's logic again (e.g.
        # range_binning's preview calls apply_range_binning directly, with no
        # try/except of its own) purely to build a before/after sample. A bug
        # here must degrade to "no preview available", not crash a request
        # that would otherwise have succeeded.
        try:
            preview_data = transformation.preview(df, resolved_params, sample_rows=10)
        except Exception as e:
            preview_data = {"error": f"Preview unavailable: {e}"}

        try:
            apply_result = transformation.apply(df, resolved_params)
        except TransformationError as e:
            return TransformationResult.failure(str(e), execution_time=time.perf_counter() - start)
        except Exception as e:  # transformation bugs must not crash the engine
            return TransformationResult.failure(
                f"{transformation.display_name} failed: {e}",
                execution_time=time.perf_counter() - start,
            )

        new_df = apply_result["dataframe"]
        # HARDENING: apply() already succeeded — new_df is a real, valid
        # transformed dataframe. A bug in metadata()/diff_schema() (e.g. an
        # adapter's metadata() referencing a key apply_result doesn't have)
        # must not throw away a transformation that actually worked.
        try:
            metadata = transformation.metadata(apply_result)
        except Exception as e:
            metadata = {"error": f"Could not build transformation metadata: {e}"}
        try:
            schema_diff = diff_schema(df, new_df)
        except Exception as e:
            schema_diff = {"error": f"Could not compute schema diff: {e}"}

        # ── Refresh downstream analytics — REUSES the existing orchestrator,
        # never a second/duplicate pipeline. Never rescans the dataset from
        # disk: `new_df` is passed by reference, already in memory. ────────
        ai_report: dict[str, Any] = {}
        if refresh_analytics:
            resolved_value_column = value_column or metadata.get("source_column") or _detect_value_column(
                new_df, exclude={metadata.get("new_column")} if metadata.get("new_column") else set()
            )
            derived_columns = []
            if metadata.get("new_column"):
                derived_columns.append({
                    "new_column": metadata.get("new_column"),
                    "source_column": metadata.get("source_column"),
                    "method": transformation.display_name,
                    "category_count": metadata.get("category_count")
                    or (len(metadata["ranges"]) if metadata.get("ranges") else None),
                })
            try:
                ai_report = generate_structured_report_data(
                    new_df,
                    list(ALL_ANALYSIS_IDS),
                    value_column=resolved_value_column,
                    derived_column=metadata.get("new_column"),
                    derived_source_column=metadata.get("source_column"),
                    derived_columns=derived_columns or None,
                )
            except Exception as e:
                ai_report = {"error": f"Analytics refresh failed: {e}"}

        execution_time = time.perf_counter() - start

        entry = None
        if history is not None:
            try:
                entry = history.record(
                    transformation_name=transformation.name,
                    before_df=df,
                    after_df=new_df,
                    params=resolved_params,
                    metadata=metadata,
                    query=query,
                    execution_time=execution_time,
                )
            except Exception:
                entry = None  # undo/redo for this step won't be available, but the transformation itself still succeeded

        result = TransformationResult(
            success=True,
            transformation={
                "applied": True,
                "name": transformation.name,
                "display_name": transformation.display_name,
                "history_id": entry.id if entry else None,
                **metadata,
            },
            preview=preview_data,
            metadata=metadata,
            updated_schema=schema_diff,
            updated_statistics=ai_report.get("statistics", {}),
            updated_kpis=ai_report.get("detected_kpis", []),
            updated_charts=ai_report.get("chart_recommendation", {}),
            updated_ai_report=ai_report,
            execution_time=execution_time,
            message=f"{transformation.display_name} applied successfully.",
        )
        # Not a dataclass field (kept out of to_dict()/JSON) — routes that
        # need the actual transformed dataframe (e.g. to build an `export`
        # payload) read this directly rather than re-deriving it.
        result.dataframe = new_df
        return result

    # ── Undo / Redo — restore a snapshot, then re-run the SAME analytics
    #    refresh so Flutter gets a fully consistent result either way. ────
    def undo(self, history: TransformationHistory, value_column: str | None = None) -> TransformationResult:
        start = time.perf_counter()
        try:
            result = history.undo()
        except Exception as e:
            return TransformationResult.failure(f"Undo failed: {e}", execution_time=time.perf_counter() - start)
        if result is None:
            return TransformationResult.failure("Nothing to undo.", execution_time=time.perf_counter() - start)
        restored_df, entry = result
        return self._post_undo_redo_result(restored_df, entry, "Undo", start, value_column)

    def redo(self, history: TransformationHistory, value_column: str | None = None) -> TransformationResult:
        start = time.perf_counter()
        try:
            result = history.redo()
        except Exception as e:
            return TransformationResult.failure(f"Redo failed: {e}", execution_time=time.perf_counter() - start)
        if result is None:
            return TransformationResult.failure("Nothing to redo.", execution_time=time.perf_counter() - start)
        restored_df, entry = result
        return self._post_undo_redo_result(restored_df, entry, "Redo", start, value_column)

    def _post_undo_redo_result(self, restored_df, entry, action, start, value_column):
        ai_report: dict[str, Any] = {}
        try:
            ai_report = generate_structured_report_data(
                restored_df, list(ALL_ANALYSIS_IDS), value_column=value_column,
            )
        except Exception as e:
            ai_report = {"error": f"Analytics refresh failed: {e}"}

        try:
            updated_schema = _compute_schema(restored_df)
        except Exception as e:
            updated_schema = {"error": f"Could not compute schema: {e}"}

        result = TransformationResult(
            success=True,
            transformation={
                "applied": action == "Redo",
                "action": action.lower(),
                "name": entry.transformation_name,
                "history_id": entry.id,
            },
            metadata=entry.metadata,
            updated_schema=updated_schema,
            updated_statistics=ai_report.get("statistics", {}),
            updated_kpis=ai_report.get("detected_kpis", []),
            updated_charts=ai_report.get("chart_recommendation", {}),
            updated_ai_report=ai_report,
            execution_time=time.perf_counter() - start,
            message=f"{action} applied: reverted '{entry.transformation_name}'.",
        )
        result.dataframe = restored_df
        return result
