from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingPlan:
    base_model: str
    method: str
    dataset_dir: Path
    output_dir: Path
    max_seq_len: int
    smoke_only: bool = False


def build_training_plan(config: Any) -> TrainingPlan:
    return TrainingPlan(
        base_model=config.base_model,
        method=config.method,
        dataset_dir=config.dataset_dir,
        output_dir=config.output_dir,
        max_seq_len=config.max_seq_len,
        smoke_only=config.smoke_only,
    )
