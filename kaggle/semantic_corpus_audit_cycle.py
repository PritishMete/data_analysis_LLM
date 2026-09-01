"""Kaggle-safe canonical semantic corpus audit; never loads a model or test rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from learning.semantic_corpus_audit import audit_dataset

from .bootstrap import KAGGLE_INPUT_ROOT, _load_json_if_exists, resolve_canonical_dataset_root
from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


def _marker(name: str, payload: dict[str, Any]) -> None:
    print(f"{name}={json.dumps(payload, sort_keys=True)}", flush=True)


def run_semantic_corpus_audit(*, output_root: Path, run_id: str, expected_git_commit: str | None, source_root: Path) -> dict[str, Any]:
    run_root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    executed: str | None = None
    try:
        resolved_source = resolve_executed_source_commit(run_root=run_root, repo_root=source_root, expected_git_commit=expected_git_commit)
        executed = resolved_source.get("executed_source_commit")
        write_source_identity(run_root, run_id=run_id, expected_git_commit=expected_git_commit, executed_source_commit=executed, source_identity_method=str(resolved_source.get("source_identity_method") or "unknown"), source_identity_verified=bool(resolved_source.get("source_identity_verified")))
        if expected_git_commit and executed != expected_git_commit:
            raise RuntimeError("stale_kaggle_checkout")
        resolved = resolve_canonical_dataset_root(KAGGLE_INPUT_ROOT)
        if not resolved.get("root"):
            raise RuntimeError(str(resolved.get("reason") or "canonical_dataset_missing"))
        root = Path(str(resolved["root"]))
        audit = audit_dataset(root, output_path=run_root / "semantic_corpus_audit.json")
        result = {"run_id": run_id, "expected_git_commit": expected_git_commit, "executed_git_commit": executed, "canonical_dataset_root": str(root), **audit}
        _marker("SEMANTIC_CORPUS_AUDIT_JSON", result)
        return result
    except Exception as exc:
        (run_root / "smoke_failure.json").write_text(json.dumps({"run_id": run_id, "stage": "semantic_corpus_audit", "exception_type": type(exc).__name__, "sanitized_message": str(exc).replace("\n", " ")[:1000]}, indent=2, sort_keys=True), encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/kaggle/working")
    parser.add_argument("--run-id")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--source-root")
    args = parser.parse_args(argv)
    root = Path(args.output_root)
    run_id = args.run_id or resolve_current_run_id(base_root=root / "smoke_runs") or generate_run_id()
    result = run_semantic_corpus_audit(output_root=root, run_id=run_id, expected_git_commit=args.expected_git_commit, source_root=Path(args.source_root) if args.source_root else root / "data_analysis_LLM")
    return 0 if result.get("classification") in {"SEMANTIC_SUBSET_SELECTION_BUG", "INSUFFICIENT_SEMANTIC_TRAINING_EXAMPLES", "SEMANTIC_CORPUS_TOO_SMALL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
