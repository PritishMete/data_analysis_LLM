from __future__ import annotations

from fastapi import APIRouter

from .app import get_service

router = APIRouter()


@router.get("/v1/learning/status")
def learning_status() -> dict:
    return {"status": "ok", "learning": get_service().learning_status()}
