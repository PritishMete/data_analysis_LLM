from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import load_default_config
from .dataset import validate_dataset
from .hardware import detect_hardware


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m training.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hardware", help="Inspect local training hardware")

    validate = sub.add_parser("validate-dataset", help="Validate the canonical fine-tuning dataset")
    validate.add_argument("--dataset-dir", type=Path, default=load_default_config().dataset_dir)

    train = sub.add_parser("train", help="Prepare a reproducible GPU training run")
    train.add_argument("--dataset-dir", type=Path, default=load_default_config().dataset_dir)
    train.add_argument("--output-dir", type=Path, default=load_default_config().output_dir)
    train.add_argument("--base-model", default=load_default_config().base_model)
    train.add_argument("--method", choices=["qlora", "lora"], default=load_default_config().method)
    train.add_argument("--allow-smoke-only", action="store_true")
    train.add_argument("--max-seq-len", type=int, default=load_default_config().max_seq_len)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hardware":
        _print_json(detect_hardware().to_dict())
        return 0
    if args.command == "validate-dataset":
        result = validate_dataset(args.dataset_dir)
        _print_json(result.to_dict())
        return 0 if result.ready_for_prototype else 2
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
