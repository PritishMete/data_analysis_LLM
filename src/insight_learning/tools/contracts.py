from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class ToolContract:
    tool_name: str
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]
    privacy_level: str = "local-only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

