# GPU Training Runbook

This repository prepares prototype planner fine-tuning pipelines for both the
existing `Qwen/Qwen2.5-1.5B-Instruct` path and the new low-spec
`Qwen/Qwen2.5-0.5B-Instruct` profile without running real training on CPU-only
hardware.

## Goals

- Verify the canonical training dataset is ready.
- Verify SHA-256 dataset manifests before export.
- Refuse real training when CUDA is unavailable.
- Support dry-run validation of config and dataset without downloading the
  model.
- Support profile-aware hardware checks for both `standard` and `low_spec`
  planner targets.
- Keep all exported fine-tuning data privacy-safe and deduplicated.

## Local checks

```bash
export DATASET_DIR="$PWD/runtime/training"
export MODEL_OUTPUT_DIR="$PWD/runtime/models"
export HF_HOME="$HOME/.cache/huggingface"
export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export PLANNER_PROFILE="standard"
export PLANNER_BACKEND="auto"

python -m training.cli gpu-preflight \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$MODEL_OUTPUT_DIR" \
  --planner-profile "$PLANNER_PROFILE"

python -m training.cli manifest-verify \
  --dataset-dir "$DATASET_DIR"

python -m training.cli dry-run \
  --dataset-dir "$DATASET_DIR" \
  --base-model "$BASE_MODEL" \
  --planner-profile "$PLANNER_PROFILE" \
  --planner-backend "$PLANNER_BACKEND"

bash scripts/run_qwen_qlora.sh
```

For the low-spec planner preparation path, set:

```bash
export BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export PLANNER_PROFILE="low_spec"
```

## GPU training gate

Real training is blocked unless:

- CUDA is available.
- The dataset is ready for prototype promotion.
- The manifest verification passes.
- The QLoRA configuration matches the prototype model.
- The selected planner profile's training VRAM requirement is satisfied.

If CUDA is unavailable, use dry-run mode only:

```bash
python -m training.cli dry-run
```

## Prototype model metadata

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- QLoRA: 4-bit NF4, `q_proj/k_proj/v_proj/o_proj`
- Recommended sequence length: `2048`
- Recommended GPU class: `RTX 3060 12GB or better`

## Low-spec profile metadata

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- QLoRA: 4-bit NF4, LoRA rank `8`, alpha `16`, dropout `0.05`
- Recommended sequence length: `1024`
- Runtime targets: CPU-only or `GTX 1650 4GB` with safe offload-aware inference
- Training remains conservative and may still require an external GPU

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

## Resume support

If the run is interrupted, restart from the latest checkpoint by reusing the
same `MODEL_OUTPUT_DIR` and passing `--resume-from-checkpoint` to the launcher
or by exporting the checkpoint path into the shell wrapper before rerunning the
script.
