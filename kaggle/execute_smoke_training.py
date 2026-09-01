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
    from kaggle.run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity  # type: ignore[no-redef]
    from kaggle.import_trace import write_import_trace  # type: ignore[no-redef]
else:
    from .bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths
    from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity
    from .import_trace import write_import_trace


def _workflow_mode() -> str:
    return str(os.environ.get("KAGGLE_WORKFLOW_MODE") or "smoke").strip().lower()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(KAGGLE_WORKING_ROOT))
    parser.add_argument("--bootstrap-pid", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expected-git-commit", default=None)
    parser.add_argument("--source-root", default=None)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    resolved_run_id = args.run_id or resolve_current_run_id(base_root=output_root / "smoke_runs") or generate_run_id()
    run_root = ensure_run_root(resolved_run_id, base_root=output_root / "smoke_runs")
    paths = ensure_kaggle_paths(run_root)
    report_root = run_root
    training_pid = os.getpid()
    heartbeat_path = report_root / "smoke_heartbeat.json"
    write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="bootstrap_started")
    repo_root = Path(args.source_root) if args.source_root else (output_root / "data_analysis_LLM")
    resolved_source = resolve_executed_source_commit(run_root=run_root, repo_root=repo_root, expected_git_commit=args.expected_git_commit)
    git_commit = resolved_source.get("executed_source_commit")
    write_source_identity(
        run_root,
        run_id=resolved_run_id,
        expected_git_commit=args.expected_git_commit,
        executed_source_commit=git_commit,
        source_identity_method=str(resolved_source.get("source_identity_method") or "unknown"),
        source_identity_verified=bool(resolved_source.get("source_identity_verified")),
    )
    heartbeat_path.write_text(
        json.dumps(
            {
                "stage": "runtime_process_started",
                "timestamp": time.time(),
                "git_commit": git_commit,
                "expected_git_commit": args.expected_git_commit,
                "executed_source_commit": git_commit,
                "smoke_mode": True,
                "run_id": resolved_run_id,
                "bootstrap_pid": args.bootstrap_pid,
                "training_pid": training_pid,
                "fresh_process_verified": args.bootstrap_pid is not None and args.bootstrap_pid != training_pid,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.environ["KAGGLE_SMOKE_RUN_ID"] = resolved_run_id
    if args.expected_git_commit:
        os.environ["KAGGLE_EXPECTED_GIT_COMMIT"] = args.expected_git_commit
    if git_commit:
        os.environ["KAGGLE_EXECUTED_SOURCE_COMMIT"] = git_commit
    write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="before_project_training_import")
    workflow_mode = _workflow_mode()
    if workflow_mode == "torch_compat":
        from kaggle.torch_compat_cycle import run_torch_compat_cycle

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_torch_compat_cycle(output_root=run_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root, bootstrap_pid=args.bootstrap_pid)
    elif workflow_mode == "bnb_native_diagnose":
        from kaggle.bnb_native_diagnose import run_bnb_native_diagnose

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_bnb_native_diagnose(output_root=run_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root)
    elif workflow_mode == "bnb_compat":
        from kaggle.bnb_compat_cycle import run_bnb_compat_cycle

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_bnb_compat_cycle(output_root=output_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root, bootstrap_pid=args.bootstrap_pid)
    elif workflow_mode == "qwen_nf4_load":
        from kaggle.qwen_nf4_load_cycle import run_qwen_nf4_load_cycle

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_qwen_nf4_load_cycle(output_root=output_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root)
    elif workflow_mode in {"qwen_qlora_backward", "qwen_qlora_training_smoke"}:
        from kaggle.qwen_qlora_backward_cycle import run_qwen_qlora_backward_cycle

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_qwen_qlora_backward_cycle(output_root=output_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root)
    elif workflow_mode == "qwen_qlora_learning_experiment":
        from kaggle.qwen_qlora_learning_experiment import run_qwen_qlora_learning_experiment

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_qwen_qlora_learning_experiment(output_root=output_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root)
    elif workflow_mode == "qwen_semantic_memorization":
        from kaggle.qwen_qlora_learning_experiment import run_qwen_qlora_learning_experiment

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_qwen_qlora_learning_experiment(output_root=output_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root, memorization=True)
    elif workflow_mode == "qwen_semantic_generation_diagnostic":
        from kaggle.semantic_generation_diagnostic import run_semantic_generation_diagnostic

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_semantic_generation_diagnostic(output_root=output_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root)
    elif workflow_mode == "semantic_corpus_audit":
        from kaggle.semantic_corpus_audit_cycle import run_semantic_corpus_audit

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_semantic_corpus_audit(output_root=output_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root)
    else:
        from kaggle.run_semantic_training import run_notebook_flow

        write_import_trace(report_root / "import_trace.jsonl", module="kaggle.execute_smoke_training", event="after_project_training_import")
        result = run_notebook_flow(output_root=run_root, run_id=resolved_run_id, expected_git_commit=args.expected_git_commit, source_root=repo_root)
    result["bootstrap_pid"] = args.bootstrap_pid
    result["training_pid"] = training_pid
    result["fresh_process_verified"] = args.bootstrap_pid is not None and args.bootstrap_pid != training_pid
    _write_json(report_root / "smoke_training_report.json", result.get("smoke_training_report", {}))
    _write_json(report_root / "final_report.json", result)
    legacy_report_root = output_root / "reports"
    legacy_report_root.mkdir(parents=True, exist_ok=True)
    _write_json(legacy_report_root / "smoke_heartbeat.json", json.loads(heartbeat_path.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
