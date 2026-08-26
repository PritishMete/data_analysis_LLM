# common/transformations/transformation_result.py
# ─────────────────────────────────────────────────────────────────────────────
# The single response shape returned by TransformationEngine.run() /
# .preview() / .undo() / .redo(). Every route (main.py's REST endpoints,
# query_router.py's fast-path) builds its Flutter-facing response FROM this,
# instead of each hand-assembling its own dict — one result shape everywhere.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TransformationResult:
    success: bool
    transformation: dict[str, Any] = field(default_factory=dict)
    preview: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_schema: dict[str, Any] = field(default_factory=dict)
    updated_statistics: dict[str, Any] = field(default_factory=dict)
    updated_kpis: list = field(default_factory=list)
    updated_charts: dict[str, Any] = field(default_factory=dict)
    updated_ai_report: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def failure(cls, error: str, execution_time: float = 0.0) -> "TransformationResult":
        return cls(success=False, error=error, message=error, execution_time=execution_time)
