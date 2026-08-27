from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class TrainingRunRecord:
    run_id: str
    base_model: str
    dataset_version: str
    method: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingRunRegistry:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, runs: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(runs, indent=2, sort_keys=True), encoding="utf-8")
