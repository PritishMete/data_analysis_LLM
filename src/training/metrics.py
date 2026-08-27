from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingMetrics:
    train_examples: int
    validation_examples: int
    test_examples: int
    ready_for_prototype: bool
