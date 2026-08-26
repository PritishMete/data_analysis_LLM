from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from learning.training_export import TrainingDatasetExporter

from .app import get_service

router = APIRouter()


@router.get("/v1/export/training-dataset")
def export_training_dataset(
    format: str = Query("report", pattern="^(json|jsonl|csv|report)$"),
    include_candidate_strategies: bool = Query(True),
    persist: bool = Query(False),
    limit: int = Query(1000, ge=1, le=10_000),
):
    service = get_service()
    exporter = TrainingDatasetExporter(service.store, service.training_export_policy)
    bundle, preview = exporter.build_bundle(
        limit=limit,
        include_candidate_strategies=include_candidate_strategies,
    )

    if persist:
        service.export_training_dataset_files(
            include_candidate_strategies=include_candidate_strategies,
            limit=limit,
        )

    if format == "jsonl":
        return Response(content=exporter.export_bytes(bundle.records, fmt="jsonl"), media_type="application/x-ndjson")
    if format == "csv":
        return Response(content=exporter.export_bytes(bundle.records, fmt="csv"), media_type="text/csv")
    if format == "report":
        return {
            "exported": bool(bundle.records),
            "format": "report",
            "report": bundle.report(),
        }
    return {
        "exported": bool(bundle.records),
        "format": "json",
        "preview_count": len(preview),
        "eligible_examples": len(bundle.records),
        "records": preview,
        "report": bundle.report(),
    }
