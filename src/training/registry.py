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


@dataclass(slots=True)
class ModelRegistryEntry:
    model_id: str
    base_model: str
    license: str
    context_length: int
    recommended_sequence_length: int
    qlora_enabled: bool
    status: str = "prototype_ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_model": self.base_model,
            "license": self.license,
            "context_length": self.context_length,
            "recommended_sequence_length": self.recommended_sequence_length,
            "qlora_enabled": self.qlora_enabled,
            "status": self.status,
        }


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
