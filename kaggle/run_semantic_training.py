from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bootstrap import (
    KAGGLE_WORKING_ROOT,
    build_artifact_manifest,
    build_semantic_kaggle_report,
    create_final_zip,
    detect_resume_checkpoint,
    discover_semantic_dataset,
    ensure_kaggle_paths,
    inspect_kaggle_environment,
    load_semantic_config,
    semantic_verdict,
    verify_attached_dataset,
)


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _safe_runtime_report(*, dataset_dir: Path, output_root: Path) -> dict[str, Any]:
    env = inspect_kaggle_environment().to_dict()
    paths = ensure_kaggle_paths(output_root)
    config = load_semantic_config()
    verification = verify_attached_dataset(dataset_dir)
    resume_checkpoint = detect_resume_checkpoint(paths.checkpoints)
    report = build_semantic_kaggle_report(dataset_dir=dataset_dir, output_root=output_root)
    return {
        "environment": env,
        "paths": paths.to_dict(),
        "config": config,
        "dataset_dir": str(dataset_dir),
        "dataset_verification": verification,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "semantic_report": report,
    }


def build_training_plan(*, dataset_dir: Path, output_root: Path = KAGGLE_WORKING_ROOT, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> dict[str, Any]:
    paths = ensure_kaggle_paths(output_root)
    config = load_semantic_config()
    return {
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "base_model": base_model,
        "config_path": "configs/qwen25_0_5b_semantic_qlora.yaml",
        "resume_checkpoint": str(detect_resume_checkpoint(paths.checkpoints)) if detect_resume_checkpoint(paths.checkpoints) else None,
        "steps": [
            "inspect_environment",
            "verify_cuda_gpu",
            "install_missing_training_dependencies",
            "discover_semantic_dataset",
            "verify_sha256_manifest",
            "validate_dataset_readiness",
            "run_untouched_base_model_semantic_benchmark",
            "run_semantic_qlora_training",
            "evaluate_validation",
            "select_best_checkpoint",
            "freeze_checkpoint_choice",
            "run_holdout_test_once",
            "run_ood_fallback_evaluation",
            "apply_semantic_success_gates",
            "export_lora_adapter",
            "generate_artifact_manifest",
            "create_final_zip",
        ],
        "config": config,
        "paths": paths.to_dict(),
    }


def run_notebook_flow(*, output_root: Path = KAGGLE_WORKING_ROOT, resume_from: str | None = None) -> dict[str, Any]:
    paths = ensure_kaggle_paths(output_root)
    dataset_dir = discover_semantic_dataset()
    if dataset_dir is None:
        return {
            "verdict": "KAGGLE_GPU_NOT_AVAILABLE",
            "reason": "no_attached_dataset_found",
            "paths": paths.to_dict(),
        }
    resume_checkpoint = detect_resume_checkpoint(paths.checkpoints, resume_from=resume_from)
    runtime = _safe_runtime_report(dataset_dir=dataset_dir, output_root=output_root)
    training_plan = build_training_plan(dataset_dir=dataset_dir, output_root=output_root)
    safe_report_path = paths.reports / "final_report.json"
    safe_metrics_path = paths.metrics / "semantic_metrics.json"
    artifact_manifest_path = paths.manifests / "artifact_manifest.json"
    _write_json(safe_report_path, {"runtime": runtime, "training_plan": training_plan})
    _write_json(safe_metrics_path, runtime["semantic_report"]["semantic_readiness"])
    artifact_manifest = build_artifact_manifest([safe_report_path, safe_metrics_path])
    _write_json(artifact_manifest_path, artifact_manifest)
    final_zip = create_final_zip(paths.root, [safe_report_path, safe_metrics_path, artifact_manifest_path], zip_name="semantic_extractor_artifacts.zip")
    result = {
        "environment": runtime["environment"],
        "paths": paths.to_dict(),
        "dataset_dir": str(dataset_dir),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "dataset_verification": runtime["dataset_verification"],
        "semantic_readiness": runtime["semantic_report"]["semantic_readiness"],
        "training_plan": training_plan,
        "artifact_manifest": artifact_manifest,
        "final_zip": str(final_zip),
        "verdict": semantic_verdict(
            gate_results={
                "intent_accuracy": 0.0,
                "binding_accuracy": 0.0,
                "predicate_coverage": 0.0,
                "logical_structure_accuracy": 0.0,
                "semantic_schema_valid_rate": 0.0,
                "fallback_accuracy": 0.0,
            },
            readiness=bool(runtime["semantic_report"]["semantic_readiness"].get("ready")),
            fallback_rate=1.0,
        ),
    }
    return result


def main() -> int:
    result = run_notebook_flow()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
