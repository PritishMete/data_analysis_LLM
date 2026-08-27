# GPU Training Runbook

This runbook covers the external-GPU specialization workflow for
`Qwen/Qwen2.5-0.5B-Instruct`.

## Inputs

- `DATASET_DIR`
- `MODEL_OUTPUT_DIR`
- `HF_HOME`
- `BASE_MODEL`
- `RESUME_FROM_CHECKPOINT` optionally

## Before training

```bash
export DATASET_DIR="$PWD/runtime/training"
export MODEL_OUTPUT_DIR="$PWD/runtime/models"
export HF_HOME="$HOME/.cache/huggingface"
export BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

python -m training.cli gpu-preflight \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$MODEL_OUTPUT_DIR" \
  --manifest-path "$DATASET_DIR/dataset_manifest.sha256.json" \
  --planner-profile low_spec

python -m training.cli manifest-verify \
  --dataset-dir "$DATASET_DIR" \
  --manifest-path "$DATASET_DIR/dataset_manifest.sha256.json"

python -m training.cli dry-run \
  --dataset-dir "$DATASET_DIR" \
  --base-model "$BASE_MODEL" \
  --planner-profile low_spec \
  --planner-backend auto
```

## 0.5B training launch

The first real specialization run should be launched on the CUDA machine after
preflight passes.

```bash
python -m training.cli gpu-launch \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$MODEL_OUTPUT_DIR" \
  --base-model "$BASE_MODEL" \
  --planner-profile low_spec \
  --planner-backend auto \
  --hf-home "$HF_HOME"
```

If the run is interrupted, restart from the latest checkpoint:

```bash
python -m training.cli gpu-launch \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$MODEL_OUTPUT_DIR" \
  --base-model "$BASE_MODEL" \
  --planner-profile low_spec \
  --planner-backend auto \
  --resume-from-checkpoint "$MODEL_OUTPUT_DIR/experiment-001/checkpoint-<step>" \
  --hf-home "$HF_HOME"
```

## External training workflow

1. Verify dataset hashes.
2. Validate corpus schema and split integrity.
3. Run the untouched baseline benchmark.
4. Train QLoRA with the V2 0.5B config.
5. Evaluate validation metrics using structural gates, not loss alone.
6. Freeze checkpoint selection.
7. Run a single holdout test evaluation.
8. Export the adapter and optional merged model.
9. Convert to GGUF and quantize to `Q4_K_M` if needed for local inference.
10. Write SHA-256 manifests for every returned artifact.

## Return artifacts

Return the trained assets to the low-spec machine as files plus a manifest:

- LoRA adapter
- merged model, if produced
- GGUF model, if produced
- artifact manifest with SHA-256
- model-registry metadata

## Post-training benchmark

Benchmark the returned model on the GTX 1650 machine:

```bash
python -m training.cli inference-benchmark \
  --profile low_spec \
  --backend auto \
  --device cuda \
  --benchmark builtin \
  --output-dir runtime/benchmark_gpu

python -m training.cli inference-benchmark \
  --profile low_spec \
  --backend llama_cpp \
  --device cpu \
  --benchmark builtin \
  --output-dir runtime/benchmark_cpu
```

Measure:

- plan quality
- median latency
- p95 latency
- tokens per second
- peak RAM
- peak VRAM
- critic acceptance
- Gemini fallback rate

## Routing policy after success

If the specialized 0.5B model passes the quality gates:

1. trusted strategy
2. 0.5B local planner
3. Gemini

If it fails:

1. trusted strategy
2. Gemini

## Shadow mode

If promotion gates pass, register the model only as `shadow`. Do not mark the
model production-ready from this workflow.

