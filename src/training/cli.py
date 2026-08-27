from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_default_config
from .dataset import validate_dataset, verify_dataset_manifest, write_dataset_manifest
from .hardware import detect_hardware
from .model_loader import DEFAULT_PROTOTYPE_MODEL, DEFAULT_MODEL_REGISTRY_ENTRY
from .qlora import QLoRAConfig
from .promotion import evaluate_promotion_gates
from .metrics import evaluate_training_metrics


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m training.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hardware", help="Inspect local training hardware")

    validate = sub.add_parser("validate-dataset", help="Validate the canonical fine-tuning dataset")
    validate.add_argument("--dataset-dir", type=Path, default=load_default_config().dataset_dir)

    manifest_create = sub.add_parser("manifest-create", help="Create a SHA-256 dataset manifest")
    manifest_create.add_argument("--dataset-dir", type=Path, default=load_default_config().dataset_dir)
    manifest_create.add_argument("--manifest-path", type=Path, default=None)

    manifest_verify = sub.add_parser("manifest-verify", help="Verify a SHA-256 dataset manifest")
    manifest_verify.add_argument("--dataset-dir", type=Path, default=load_default_config().dataset_dir)
    manifest_verify.add_argument("--manifest-path", type=Path, default=None)

    dry_run = sub.add_parser("dry-run", help="Validate config, dataset, and hardware without training")
    dry_run.add_argument("--dataset-dir", type=Path, default=load_default_config().dataset_dir)
    dry_run.add_argument("--base-model", default=load_default_config().base_model)
    dry_run.add_argument("--max-seq-len", type=int, default=load_default_config().max_seq_len)

    train = sub.add_parser("train", help="Prepare a reproducible GPU training run")
    train.add_argument("--dataset-dir", type=Path, default=load_default_config().dataset_dir)
    train.add_argument("--output-dir", type=Path, default=load_default_config().output_dir)
    train.add_argument("--base-model", default=load_default_config().base_model)
    train.add_argument("--method", choices=["qlora", "lora"], default=load_default_config().method)
    train.add_argument("--allow-smoke-only", action="store_true")
    train.add_argument("--max-seq-len", type=int, default=load_default_config().max_seq_len)

    return parser


def _dry_run_payload(dataset_dir: Path, base_model: str, max_seq_len: int) -> dict[str, Any]:
    hardware = detect_hardware()
    validation = validate_dataset(dataset_dir)
    config = load_default_config()
    qlora = QLoRAConfig(model_id=base_model)
    promotion = evaluate_promotion_gates(
        readiness=validation.readiness,
        metrics={
            "json_validity": 1.0,
            "plan_validity": 1.0,
            "tool_selection_f1": 1.0,
            "invalid_tool_rate": 0.0,
        },
    )
    return {
        "status": "dry_run",
        "hardware": hardware.to_dict(),
        "dataset": validation.to_dict(),
        "promotion_gate": promotion.to_dict(),
        "config": {
            "base_model": base_model,
            "max_seq_len": max_seq_len,
            "dataset_dir": str(dataset_dir),
            "output_dir": str(config.output_dir),
            "qlora": qlora.to_dict(),
            "prototype_model": DEFAULT_PROTOTYPE_MODEL.to_dict(),
            "model_registry_entry": DEFAULT_MODEL_REGISTRY_ENTRY.to_dict(),
        },
        "metrics_schema": evaluate_training_metrics(
            predicted=[{"plan_valid": True, "predicate_keys": ["a"], "logical_structure": "AND", "semantic_roles": ["x"], "tool_graph": ["tool.a"]}],
            expected=[{"plan_valid": True, "predicate_keys": ["a"], "logical_structure": "AND", "semantic_roles": ["x"], "tool_graph": ["tool.a"]}],
        ).to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hardware":
        _print_json(detect_hardware().to_dict())
        return 0
    if args.command == "validate-dataset":
        result = validate_dataset(args.dataset_dir)
        _print_json(result.to_dict())
        return 0 if result.ready_for_prototype else 2
    if args.command == "manifest-create":
        manifest_path = write_dataset_manifest(args.dataset_dir, args.manifest_path)
        _print_json({"created": True, "manifest_path": str(manifest_path), "dataset_dir": str(args.dataset_dir)})
        return 0
    if args.command == "manifest-verify":
        verification = verify_dataset_manifest(args.dataset_dir, args.manifest_path)
        _print_json(verification)
        return 0 if verification["verified"] else 2
    if args.command == "dry-run":
        payload = _dry_run_payload(args.dataset_dir, args.base_model, args.max_seq_len)
        _print_json(payload)
        return 0 if payload["dataset"]["ready_for_prototype"] else 2
    if args.command == "train":
        hw = detect_hardware()
        validation = validate_dataset(args.dataset_dir)
        if not validation.ready_for_prototype:
            _print_json({"status": "blocked", "reason": "dataset_validation_failed", "validation": validation.to_dict()})
            return 2
        if not hw.cuda_available and not args.allow_smoke_only:
            _print_json({"status": "blocked", "reason": "cuda_unavailable", "hardware": hw.to_dict()})
            return 3
        _print_json({
            "status": "prepared",
            "mode": "smoke_only" if not hw.cuda_available else "cuda_ready",
            "hardware": hw.to_dict(),
            "dataset": validation.to_dict(),
            "training": {
                "base_model": args.base_model,
                "method": args.method,
                "max_seq_len": args.max_seq_len,
                "output_dir": str(args.output_dir),
                "python_target": "3.11-3.12",
            },
        })
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
