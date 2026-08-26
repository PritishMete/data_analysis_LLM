from __future__ import annotations

from fastapi import APIRouter

from .app import get_service

router = APIRouter()


@router.get("/v1/skills")
def skills() -> dict:
    return {"skills": get_service().skills()}

