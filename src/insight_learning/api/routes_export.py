from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from learning.training_export import TrainingDatasetExporter

from .app import get_service
from .schemas import TrainingCandidateInvalidationRequest, TrainingDatasetCreateRequest

router = APIRouter()


@router.get("/v1/export/training-dataset")
def export_training_dataset(
    format: str = Query("report", pattern="^(json|jsonl|csv|report|manifest|readiness)$"),
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

    if format == "readiness":
        return {
            "exported": bool(bundle.records),
            "format": "readiness",
            "readiness": service.evaluate_training_dataset_readiness(
                include_candidate_strategies=include_candidate_strategies,
                limit=limit,
            ),
        }
    if format == "manifest":
        return {
            "exported": bool(bundle.records),
            "format": "manifest",
            "manifest": service.build_training_dataset_manifest(
                include_candidate_strategies=include_candidate_strategies,
                limit=limit,
            ),
        }
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


@router.post("/v1/export/training-dataset/create")
def create_training_dataset(payload: TrainingDatasetCreateRequest):
    service = get_service()
    created = service.create_training_dataset(
        include_candidate_strategies=payload.include_candidate_strategies,
        limit=payload.limit,
    )
    return {
        "created": True,
        "manifest": created["manifest"],
        "paths": created["paths"],
    }


@router.post("/v1/export/training-dataset/invalidate")
def invalidate_training_candidate(payload: TrainingCandidateInvalidationRequest):
    service = get_service()
    invalidation = service.invalidate_training_candidate(
        source_id=payload.source_id,
        family_fingerprint=payload.family_fingerprint,
        reason=payload.reason,
    )
    return {
        "invalidated": True,
        "invalidation": invalidation,
    }
