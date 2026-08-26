from __future__ import annotations

from fastapi import APIRouter

from .app import get_service
from .schemas import PlanRequest

router = APIRouter()


@router.post("/v1/plan")
def plan(request: PlanRequest) -> dict:
    return get_service().plan(request)

