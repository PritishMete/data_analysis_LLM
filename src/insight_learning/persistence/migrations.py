from __future__ import annotations

from pathlib import Path

from .database import initialise_runtime_db


def migrate(path: Path | None = None) -> Path:
    return initialise_runtime_db(path)

