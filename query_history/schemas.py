# query_history/schemas.py
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class QueryHistoryCreate(BaseModel):
    dataset_id: str | None = None
    organization_id: str | None = None
    user_query: str
    intent: str | None = None
    generated_sql: str | None = None
    python_pipeline: Any | None = None
    visualization: Any | None = None
    execution_time_ms: float | None = None
    rows_returned: int | None = None
    success: bool = True
    error_message: str | None = None
    planner_version: str | None = None


class QueryHistoryOut(BaseModel):
    id: int
    dataset_id: str | None
    organization_id: str | None
    schema_hash: str | None
    user_query: str
    intent: str | None
    generated_sql: str | None
    python_pipeline: Any | None
    visualization: Any | None
    execution_time_ms: float | None
    rows_returned: int | None
    success: bool
    error_message: str | None
    feedback_score: int | None
    planner_version: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class FeedbackUpdate(BaseModel):
    feedback_score: int
