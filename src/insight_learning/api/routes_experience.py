from __future__ import annotations

from fastapi import APIRouter

from .app import get_service
from .schemas import ExperienceRequest

router = APIRouter()


@router.post("/v1/experience")
def experience(request: ExperienceRequest) -> dict:
    return get_service().experience(request)

