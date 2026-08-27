from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .profiles import PLANNER_PROFILE_STANDARD

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PLANNER_PROFILE = PLANNER_PROFILE_STANDARD
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_TRAINING_DIR = Path("runtime") / "fine_tuning"
DEFAULT_DATASET_DIR = Path("runtime") / "training"
DEFAULT_OUTPUT_DIR = Path("runtime") / "fine_tuning" / "runs"
DEFAULT_MANIFEST_NAME = "dataset_manifest.json"
DEFAULT_MANIFEST_SHA256_NAME = "dataset_manifest.sha256"


@dataclass(slots=True)
class TrainingConfig:
    base_model: str = DEFAULT_BASE_MODEL
    method: str = "qlora"
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    dataset_dir: Path = DEFAULT_DATASET_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    training_dir: Path = DEFAULT_TRAINING_DIR
    python_target: str = "3.11-3.12"
    planner_profile: str = DEFAULT_PLANNER_PROFILE
    planner_backend: str = "auto"
    smoke_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / DEFAULT_MANIFEST_NAME

    @property
    def manifest_sha256_path(self) -> Path:
        return self.dataset_dir / DEFAULT_MANIFEST_SHA256_NAME


def load_default_config() -> TrainingConfig:
    return TrainingConfig()
