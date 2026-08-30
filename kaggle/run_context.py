from __future__ import annotations

import json
import os
import random
import string
import time
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path("/kaggle/working/smoke_runs")
SOURCE_IDENTITY_NAME = "source_identity.json"
SOURCE_IDENTITY_RESOLVED_NAME = "source_identity_resolved.json"


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


def _is_full_sha(value: str | None) -> bool:
    return bool(value) and len(value.strip()) == 40 and all(ch in "0123456789abcdefABCDEF" for ch in value.strip())


def write_source_identity(
    run_root: Path,
    *,
    run_id: str,
    expected_git_commit: str | None,
    executed_source_commit: str | None,
    source_identity_method: str,
    source_identity_verified: bool,
    timestamp: float | None = None,
) -> Path:
    payload = {
        "run_id": run_id,
        "expected_git_commit": expected_git_commit,
        "executed_source_commit": executed_source_commit,
        "source_identity_method": source_identity_method,
        "source_identity_verified": bool(source_identity_verified),
        "timestamp": timestamp or time.time(),
    }
    return write_json(run_root / SOURCE_IDENTITY_RESOLVED_NAME, payload)


def source_identity_paths(run_root: Path, repo_root: Path | None = None) -> list[Path]:
    paths = [run_root / SOURCE_IDENTITY_NAME, run_root / SOURCE_IDENTITY_RESOLVED_NAME]
    if repo_root is not None:
        paths.append(repo_root / SOURCE_IDENTITY_NAME)
        paths.append(repo_root / SOURCE_IDENTITY_RESOLVED_NAME)
    return paths


def resolve_executed_source_commit(
    *,
    run_root: Path,
    repo_root: Path | None = None,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    explicit = os.environ.get("KAGGLE_EXECUTED_SOURCE_COMMIT") or os.environ.get("KAGGLE_SOURCE_COMMIT")
    if _is_full_sha(explicit):
        return {
            "executed_source_commit": explicit.strip(),
            "source_identity_method": "environment",
            "source_identity_verified": True,
        }
    for path in source_identity_paths(run_root, repo_root):
        payload = read_json(path)
        if not payload:
            continue
        executed = payload.get("executed_source_commit") or payload.get("executed_git_commit") or payload.get("source_commit") or payload.get("commit")
        method = str(payload.get("source_identity_method") or "source_identity_json")
        verified = bool(payload.get("source_identity_verified", True))
        if _is_full_sha(executed):
            return {
                "executed_source_commit": str(executed).strip(),
                "source_identity_method": method,
                "source_identity_verified": verified,
            }
    if repo_root is not None and (repo_root / ".git").exists():
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        commit = (result.stdout or "").strip()
        if result.returncode == 0 and _is_full_sha(commit):
            return {
                "executed_source_commit": commit,
                "source_identity_method": "git_rev_parse",
                "source_identity_verified": True,
            }
    return {
        "executed_source_commit": None,
        "source_identity_method": None,
        "source_identity_verified": False,
        "reason": "SOURCE_IDENTITY_MISSING" if expected_git_commit else "SOURCE_IDENTITY_MISSING",
    }


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
