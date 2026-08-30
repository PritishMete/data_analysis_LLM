from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kaggle.bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths  # type: ignore[no-redef]
    from kaggle.import_trace import write_import_trace  # type: ignore[no-redef]
else:
    from .bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths
    from .import_trace import write_import_trace


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(KAGGLE_WORKING_ROOT))
    parser.add_argument("--bootstrap-pid", type=int, default=None)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    paths = ensure_kaggle_paths(output_root)
    report_root = paths.reports
    training_pid = os.getpid()
    heartbeat_path = report_root / "smoke_heartbeat.json"
    write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="bootstrap_started")
    git_commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    ).stdout.strip() or None
    git_commit = git_commit_result
    heartbeat_path.write_text(
        json.dumps(
            {
                "stage": "training_started",
                "timestamp": time.time(),
                "git_commit": git_commit,
                "smoke_mode": True,
                "bootstrap_pid": args.bootstrap_pid,
                "training_pid": training_pid,
                "fresh_process_verified": args.bootstrap_pid is not None and args.bootstrap_pid != training_pid,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.environ["KAGGLE_SKIP_DEP_INSTALL"] = "1"
    write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="before_project_training_import")
    from kaggle.run_semantic_training import run_notebook_flow
    write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")

    result = run_notebook_flow(output_root=output_root)
    result["bootstrap_pid"] = args.bootstrap_pid
    result["training_pid"] = training_pid
    result["fresh_process_verified"] = args.bootstrap_pid is not None and args.bootstrap_pid != training_pid
    _write_json(report_root / "smoke_training_report.json", result.get("smoke_training_report", {}))
    _write_json(report_root / "final_report.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
