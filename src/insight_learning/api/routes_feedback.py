from __future__ import annotations

from fastapi import APIRouter

from .app import get_service
from .schemas import FeedbackRequest

router = APIRouter()


@router.post("/v1/feedback")
def feedback(request: FeedbackRequest) -> dict:
    return get_service().feedback(request)

