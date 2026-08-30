from __future__ import annotations

import json
import os
import random
import string
import time
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("/kaggle/working/smoke_runs")


def utc_run_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def generate_run_id(*, git_commit: str | None = None, timestamp: str | None = None, suffix_length: int = 4) -> str:
    commit = (git_commit or "unknown").strip()[:7] or "unknown"
    stamp = timestamp or utc_run_timestamp()
    alphabet = string.ascii_lowercase + string.digits
    suffix = "".join(random.choice(alphabet) for _ in range(suffix_length))
    return f"{commit}-{stamp}-{suffix}"


def run_root_for(run_id: str, *, base_root: Path = DEFAULT_RUN_ROOT) -> Path:
    return base_root / run_id


def ensure_run_root(run_id: str, *, base_root: Path = DEFAULT_RUN_ROOT) -> Path:
    root = run_root_for(run_id, base_root=base_root)
    root.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoints", "adapters", "metrics", "manifests"):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_current_run_id(explicit_run_id: str | None = None, *, base_root: Path = DEFAULT_RUN_ROOT) -> str | None:
    if explicit_run_id:
        return explicit_run_id
    env = os.environ.get("KAGGLE_SMOKE_RUN_ID") or os.environ.get("KAGGLE_RUN_ID")
    if env:
        return env
    if not base_root.exists():
        return None
    candidates = [path.name for path in base_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]
