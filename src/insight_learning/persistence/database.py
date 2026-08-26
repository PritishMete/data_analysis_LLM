from __future__ import annotations

from pathlib import Path
import os
import sqlite3
from typing import Iterable


def get_runtime_db_path() -> Path:
    override = os.environ.get("INSIGHT_LEARNING_DB_PATH")
    if override:
        return Path(override)
    runtime_dir = Path(os.environ.get("INSIGHT_LEARNING_RUNTIME_DIR", "runtime"))
    return runtime_dir / "learning.db"


def initialise_runtime_db(path: Path | None = None) -> Path:
    db_path = path or get_runtime_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics (metric_key TEXT PRIMARY KEY, metric_value TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS experiences (experience_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS skills (skill_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS plan_templates (template_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS candidate_strategies (strategy_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS failure_lessons (lesson_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS corrections (correction_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.commit()
    return db_path

