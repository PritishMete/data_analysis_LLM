"""Second-stage 8-example memorization convergence diagnostic.

This entry point intentionally reuses the proven semantic QLoRA experiment and
changes only the memorization learning rate.  Runtime, quantization, model,
LoRA topology, prompt/target contract, generation path, privacy gates, train
cohort, validation behavior, and sealed test handling remain unchanged.

The purpose is narrow: determine whether the already-observed learning signal
can converge on the same eight TRAIN examples before spending GPU time on the
128/16 experiment.
"""
from __future__ import annotations

from collections.abc import Sequence

from . import qwen_qlora_learning_experiment as experiment


# The first diagnostic used 1e-5 for 50 optimizer steps and showed a real but
# weak learning signal.  Use a memorization-only LoRA LR here; do not change the
# standard 128/16 learning rate in qwen_qlora_learning_experiment.py.
MEMORIZATION_CONVERGENCE_LR = 1e-4
EXPECTED_BASE_LEARNING_RATE = 1e-5


def configure_memorization_convergence() -> None:
    """Apply only the targeted memorization LR override."""
    if experiment.LEARNING_RATE != EXPECTED_BASE_LEARNING_RATE:
        raise RuntimeError(
            "BASE_LEARNING_RATE_DRIFTED: expected the proven 128/16 baseline "
            f"{EXPECTED_BASE_LEARNING_RATE}, found {experiment.LEARNING_RATE}"
        )
    experiment.LEARNING_RATE = MEMORIZATION_CONVERGENCE_LR


def _with_memorization_flag(argv: Sequence[str] | None) -> list[str]:
    args = list(argv or [])
    if "--memorization" not in args:
        args.append("--memorization")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    configure_memorization_convergence()
    return experiment.main(_with_memorization_flag(argv))


if __name__ == "__main__":
    raise SystemExit(main())
