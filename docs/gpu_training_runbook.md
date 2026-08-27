# GPU Training Runbook

This repository prepares a prototype planner fine-tuning pipeline for
`Qwen/Qwen2.5-1.5B-Instruct` without running real training on CPU-only
hardware.

## Goals

- Verify the canonical training dataset is ready.
- Verify SHA-256 dataset manifests before export.
- Refuse real training when CUDA is unavailable.
- Support dry-run validation of config and dataset without downloading the
  model.
- Keep all exported fine-tuning data privacy-safe and deduplicated.

## Local checks

```bash
python -m training.cli hardware
python -m training.cli validate-dataset
python -m training.cli manifest-create
python -m training.cli manifest-verify
python -m training.cli dry-run
```

## GPU training gate

Real training is blocked unless:

- CUDA is available.
- The dataset is ready for prototype promotion.
- The manifest verification passes.
- The QLoRA configuration matches the prototype model.

If CUDA is unavailable, use dry-run mode only:

```bash
python -m training.cli dry-run
```

## Prototype model metadata

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- QLoRA: 4-bit NF4, `q_proj/k_proj/v_proj/o_proj`
- Recommended sequence length: `2048`
- Recommended GPU class: `RTX 3060 12GB or better`

## Promotion gates

Promotion is blocked unless the prototype dataset and metrics clear the
configured gates for:

- JSON validity
- plan validity
- predicate coverage
- logical structure accuracy
- semantic-role coverage
- tool-selection F1
- tool-sequence accuracy
- invalid-tool rate

## Shadow mode

Future shadow-mode integration should compare live planner behavior against a
frozen prototype model. That path should remain read-only to the training
pipeline until it is explicitly enabled.
