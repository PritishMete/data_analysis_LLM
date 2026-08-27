#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-runtime/training}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-runtime/models}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
SEED="${SEED:-}"

export HF_HOME

python -m training.cli gpu-preflight \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$MODEL_OUTPUT_DIR"

python -m training.cli manifest-verify \
  --dataset-dir "$DATASET_DIR"

python -m training.cli dry-run \
  --dataset-dir "$DATASET_DIR" \
  --base-model "$BASE_MODEL"

python -m training.cli gpu-launch \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$MODEL_OUTPUT_DIR" \
  --base-model "$BASE_MODEL" \
  --hf-home "$HF_HOME" \
  ${SEED:+--seed "$SEED"}

echo "Training launcher prepared. Start the actual CUDA training loop from the generated experiment directory."
