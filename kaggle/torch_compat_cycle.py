from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from kaggle.bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths, inspect_kaggle_gpu_identity  # type: ignore[no-redef]
    from kaggle.import_trace import write_import_trace  # type: ignore[no-redef]
    from kaggle.p100_torch_runtime import run_shared_p100_torch_bootstrap  # type: ignore[no-redef]
    from kaggle.run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity  # type: ignore[no-redef]
else:
    from .bootstrap import KAGGLE_WORKING_ROOT, ensure_kaggle_paths, inspect_kaggle_gpu_identity
    from .import_trace import write_import_trace
    from .p100_torch_runtime import run_shared_p100_torch_bootstrap
    from .run_context import ensure_run_root, generate_run_id, resolve_current_run_id, resolve_executed_source_commit, write_source_identity


WORKFLOW_MODE = "torch_compat"


@dataclass(slots=True)
class TorchCompatCycleReport:
    run_id: str
    expected_git_commit: str | None
    executed_git_commit: str | None
    workflow_mode: str
    gpu: dict[str, Any]
    shared_torch_bootstrap: dict[str, Any]
    artifact_paths: dict[str, str]
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _write_heartbeat(report_root: Path, *, stage: str, run_id: str, expected_git_commit: str | None, executed_git_commit: str | None, safe_message: str) -> None:
    _write_json(
        report_root / "smoke_heartbeat.json",
        {
            "stage": stage,
            "timestamp": time.time(),
            "run_id": run_id,
            "expected_git_commit": expected_git_commit,
            "executed_git_commit": executed_git_commit,
            "workflow_mode": WORKFLOW_MODE,
            "smoke_mode": False,
            "safe_message": safe_message,
        },
    )


def _emit_breadcrumb(path: Path, *, stage: str, success: bool, safe_message: str, run_id: str) -> None:
    _append_jsonl(
        path,
        {
            "run_id": run_id,
            "stage": stage,
            "success": success,
            "safe_message": safe_message,
            "timestamp": time.time(),
        },
    )


def run_torch_compat_cycle(
    *,
    output_root: Path,
    run_id: str,
    expected_git_commit: str | None,
    source_root: Path,
    bootstrap_pid: int | None = None,
) -> dict[str, Any]:
    report_root = ensure_run_root(run_id, base_root=output_root / "smoke_runs")
    ensure_kaggle_paths(report_root)
    breadcrumbs_path = report_root / "smoke_breadcrumbs.jsonl"
    _write_heartbeat(report_root, stage="notebook_started", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=None, safe_message="torch compat start")
    _emit_breadcrumb(breadcrumbs_path, stage="notebook_started", success=True, safe_message="notebook started", run_id=run_id)
    _emit_breadcrumb(breadcrumbs_path, stage="bootstrap_started", success=True, safe_message="bootstrap start", run_id=run_id)
    _write_json(report_root / "runner_metadata.json", {"run_id": run_id, "expected_git_commit": expected_git_commit, "bootstrap_pid": bootstrap_pid, "workflow_mode": WORKFLOW_MODE, "timestamp": time.time()})
    repo_root = Path(source_root)
    resolved = resolve_executed_source_commit(run_root=report_root, repo_root=repo_root, expected_git_commit=expected_git_commit)
    executed_git_commit = resolved.get("executed_source_commit")
    write_source_identity(
        report_root,
        run_id=run_id,
        expected_git_commit=expected_git_commit,
        executed_source_commit=executed_git_commit,
        source_identity_method=str(resolved.get("source_identity_method") or "unknown"),
        source_identity_verified=bool(resolved.get("source_identity_verified")),
    )
    _write_heartbeat(report_root, stage="source_identity_verified", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="source identity verified")
    if expected_git_commit and executed_git_commit != expected_git_commit:
        raise RuntimeError("stale_kaggle_checkout")

    _emit_breadcrumb(breadcrumbs_path, stage="repo_checkout_started", success=True, safe_message="repo checkout start", run_id=run_id)
    _emit_breadcrumb(breadcrumbs_path, stage="repo_checkout_complete", success=True, safe_message="repo checkout complete", run_id=run_id)
    _write_heartbeat(report_root, stage="repo_checkout_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message="repo checkout complete")

    shared_torch = run_shared_p100_torch_bootstrap(
        report_root=report_root,
        repo_root=repo_root,
        phase_prefix="torch_compat",
        write_markers=True,
    )
    _write_json(report_root / "torch_compat_report.json", {
        "run_id": run_id,
        "expected_git_commit": expected_git_commit,
        "executed_git_commit": executed_git_commit,
        "workflow_mode": WORKFLOW_MODE,
        "gpu": inspect_kaggle_gpu_identity(),
        "shared_torch_bootstrap": shared_torch,
        "artifact_paths": {
            "probe_torch_preinstall.json": str(report_root / "probe_torch_preinstall.json"),
            "probe_torch_install.json": str(report_root / "probe_torch_install.json"),
            "probe_torch_runtime.json": str(report_root / "probe_torch_runtime.json"),
            "probe_torch_cuda_runtime.json": str(report_root / "probe_torch_cuda_runtime.json"),
            "shared_torch_bootstrap_result.json": str(report_root / "shared_torch_bootstrap_result.json"),
            "shared_torch_runtime_result.json": str(report_root / "shared_torch_runtime_result.json"),
        },
        "verdict": shared_torch.get("verdict"),
    })
    _emit_breadcrumb(breadcrumbs_path, stage="torch_compat_complete", success=True, safe_message=str(shared_torch.get("verdict") or "unknown"), run_id=run_id)
    _write_heartbeat(report_root, stage="torch_compat_complete", run_id=run_id, expected_git_commit=expected_git_commit, executed_git_commit=executed_git_commit, safe_message=str(shared_torch.get("verdict") or "unknown"))
    return {
        "run_id": run_id,
        "expected_git_commit": expected_git_commit,
        "executed_git_commit": executed_git_commit,
        "workflow_mode": WORKFLOW_MODE,
        "gpu": inspect_kaggle_gpu_identity(),
        "shared_torch_bootstrap": shared_torch,
        "artifact_paths": {
            "probe_torch_preinstall.json": str(report_root / "probe_torch_preinstall.json"),
            "probe_torch_install.json": str(report_root / "probe_torch_install.json"),
            "probe_torch_runtime.json": str(report_root / "probe_torch_runtime.json"),
            "probe_torch_cuda_runtime.json": str(report_root / "probe_torch_cuda_runtime.json"),
            "shared_torch_bootstrap_result.json": str(report_root / "shared_torch_bootstrap_result.json"),
            "shared_torch_runtime_result.json": str(report_root / "shared_torch_runtime_result.json"),
        },
        "verdict": shared_torch.get("verdict"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(KAGGLE_WORKING_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expected-git-commit", default=None)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--bootstrap-pid", type=int, default=None)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    resolved_run_id = args.run_id or resolve_current_run_id(base_root=output_root / "smoke_runs") or generate_run_id()
    report = run_torch_compat_cycle(
        output_root=output_root,
        run_id=resolved_run_id,
        expected_git_commit=args.expected_git_commit,
        source_root=Path(args.source_root) if args.source_root else (output_root / "data_analysis_LLM"),
        bootstrap_pid=args.bootstrap_pid,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("verdict") == "P100_QLORA_RUNTIME_COMPATIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
