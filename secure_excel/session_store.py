"""In-memory workbook/session storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4
from typing import Any

import pandas as pd


@dataclass
class ExcelSession:
    session_id: str
    dataframe: pd.DataFrame
    schema: dict[str, Any]
    workbook_context: dict[str, Any] = field(default_factory=dict)


class ExcelSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ExcelSession] = {}

    def create(self, dataframe: pd.DataFrame, schema: dict[str, Any], workbook_context: dict[str, Any] | None = None) -> ExcelSession:
        session = ExcelSession(
            session_id=str(uuid4()),
            dataframe=dataframe.copy(),
            schema=schema,
            workbook_context=workbook_context or {},
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ExcelSession:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session_id {session_id!r}")
        return self._sessions[session_id]

    def all_sessions(self) -> list[ExcelSession]:
        return list(self._sessions.values())


SESSION_STORE = ExcelSessionStore()

