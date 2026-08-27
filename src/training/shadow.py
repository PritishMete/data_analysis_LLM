from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ShadowComparison:
    total: int = 0
    matches: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, predicted: dict[str, Any], reference: dict[str, Any]) -> None:
        self.total += 1
        if predicted == reference:
            self.matches += 1
