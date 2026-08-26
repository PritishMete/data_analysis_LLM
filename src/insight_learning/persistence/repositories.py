from __future__ import annotations

from pathlib import Path
from typing import Any

from learning.experience_store import LearningExperienceStore

from .database import initialise_runtime_db


class LearningRepository:
    def __init__(self, root: Path | None = None):
        self.root = root
        self.store = LearningExperienceStore(root=root)
        initialise_runtime_db()

    def summary(self) -> dict[str, Any]:
        return {
            "experiences": len(self.store.load_recent(limit=10_000)),
            "skills": len(self.store.load_candidate_strategies(limit=10_000)),
            "templates": len(self.store.load_plan_templates(limit=10_000)),
            "failure_lessons": len(self.store.load_failure_lessons(limit=10_000)),
            "corrections": len(self.store.load_corrections(limit=10_000)),
        }
