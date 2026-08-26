from __future__ import annotations

from pathlib import Path


def scan_teacher_repository(root: str | Path) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(str(path.relative_to(base)) for path in base.rglob("*.py"))
