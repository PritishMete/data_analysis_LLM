"""Controlled 128-train / 16-validation semantic QLoRA experiment.

This entry point intentionally reuses the proven semantic learning experiment
without changing the runtime, model, quantization, LoRA topology, prompt/target
contract, generation path, privacy gates, or sealed test handling.

It exists to make the post-memorization milestone explicit and to fail closed
if the agreed 128/16 baseline drifts before submission.
"""
from __future__ import annotations

from collections.abc import Sequence

from . import qwen_qlora_learning_experiment as experiment


EXPECTED_TRAIN_EXAMPLES = 128
EXPECTED_VALIDATION_EXAMPLES = 16
EXPECTED_LEARNING_RATE = 1e-5
EXPECTED_OPTIMIZER_STEPS = 16
EXPECTED_VALIDATION_STEPS = (0, 4, 8, 12, 16)
EXPECTED_GRADIENT_ACCUMULATION = 8
EXPECTED_MAX_SEQUENCE_LENGTH = 768


def verify_128_16_contract() -> dict[str, object]:
    """Fail closed if the controlled 128/16 baseline has drifted."""
    expected = {
        "TRAIN_EXAMPLES": EXPECTED_TRAIN_EXAMPLES,
        "VALIDATION_EXAMPLES": EXPECTED_VALIDATION_EXAMPLES,
        "LEARNING_RATE": EXPECTED_LEARNING_RATE,
        "OPTIMIZER_STEPS": EXPECTED_OPTIMIZER_STEPS,
        "VALIDATION_STEPS": EXPECTED_VALIDATION_STEPS,
        "GRADIENT_ACCUMULATION": EXPECTED_GRADIENT_ACCUMULATION,
        "MAX_SEQUENCE_LENGTH": EXPECTED_MAX_SEQUENCE_LENGTH,
    }
    actual = {
        "TRAIN_EXAMPLES": experiment.TRAIN_EXAMPLES,
        "VALIDATION_EXAMPLES": experiment.VALIDATION_EXAMPLES,
        "LEARNING_RATE": experiment.LEARNING_RATE,
        "OPTIMIZER_STEPS": experiment.OPTIMIZER_STEPS,
        "VALIDATION_STEPS": experiment.VALIDATION_STEPS,
        "GRADIENT_ACCUMULATION": experiment.GRADIENT_ACCUMULATION,
        "MAX_SEQUENCE_LENGTH": experiment.MAX_SEQUENCE_LENGTH,
    }
    drift = {key: {"expected": expected[key], "actual": actual[key]} for key in expected if actual[key] != expected[key]}
    if drift:
        raise RuntimeError(f"CONTROLLED_128_16_BASELINE_DRIFTED:{drift}")
    return {
        "experiment": "128_TRAIN_16_VALIDATION",
        "train_examples": EXPECTED_TRAIN_EXAMPLES,
        "validation_examples": EXPECTED_VALIDATION_EXAMPLES,
        "learning_rate": EXPECTED_LEARNING_RATE,
        "optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        "validation_steps": list(EXPECTED_VALIDATION_STEPS),
        "gradient_accumulation": EXPECTED_GRADIENT_ACCUMULATION,
        "max_sequence_length": EXPECTED_MAX_SEQUENCE_LENGTH,
        "test_split_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or [])
    if "--memorization" in args:
        raise RuntimeError("CONTROLLED_128_16_REFUSES_MEMORIZATION_FLAG")
    verify_128_16_contract()
    return experiment.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
