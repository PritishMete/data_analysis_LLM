from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CheckpointLayout:
    root: Path
    best_checkpoint: Path
    final_checkpoint: Path
