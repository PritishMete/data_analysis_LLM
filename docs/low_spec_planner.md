# Low-Spec Planner

This repository now supports two planner profiles:

- `low_spec` uses `Qwen/Qwen2.5-0.5B-Instruct`
- `standard` keeps the existing `Qwen/Qwen2.5-1.5B-Instruct` path

The low-spec profile is intended for narrow analytics planning only:

- input: intent, semantic roles, predicate structure, available tools
- output: structured plan or tool graph
- no workbook execution

## Why the 0.5B model exists

The 1.5B prototype model is still the preferred training target when a larger CUDA
machine is available. The 0.5B planner exists to make inference practical on:

- CPU-only laptops
- 8 to 16 GB system RAM devices
- 4 GB GPUs such as the GTX 1650

Training the 0.5B model can still require an external GPU, but inference is
designed to stay lightweight.

## Profile requirements

### `low_spec`

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- training minimum VRAM: `8 GB`
- training recommended VRAM: `12 GB`
- inference minimum RAM: `4 GB`
- inference recommended RAM: `8 GB`
- inference GPU VRAM: `4 GB`
- QLoRA: 4-bit NF4, LoRA rank 8, alpha 16, dropout 0.05, batch size 1
- sequence length: `1024`
- gradient checkpointing: enabled

### `standard`

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- training minimum VRAM: `12 GB`
- training recommended VRAM: `16 GB`
- inference minimum RAM: `8 GB`
- inference recommended RAM: `12 GB`
- inference GPU VRAM: `8 GB`
- QLoRA: 4-bit NF4, LoRA rank 16, alpha 32, dropout 0.05, batch size 1
- sequence length: `2048`

## Runtime profiles

- `cpu_low_spec`: quantized CPU-first runtime, no CUDA dependency
- `gpu_4gb`: 4 GB GPU-aware runtime with safe CPU fallback

The runtime chooses between Transformers and `llama_cpp` style backends using the
same planner interface.

## Backend routing

Routing priority stays conservative:

1. trusted strategy
2. low-spec planner
3. Gemini fallback

If a trusted strategy already covers the request, the tiny model should not be
called.

## Training and inference are separate

Training may happen on a larger external GPU. Inference should still work on
low-spec hardware after the trained adapter or quantized model is brought back.

The intended deployment path is:

1. train or adapt externally
2. merge or export the adapter if appropriate
3. convert to GGUF when useful
4. quantize for CPU-friendly inference
5. serve through the planner interface

## GGUF plan

The target quantization for the low-spec runtime is `Q4_K_M` unless a later
benchmark suggests a different 4-bit variant is safer.

## Example commands

```bash
export PLANNER_PROFILE=low_spec
export PLANNER_BACKEND=auto

python -m training.cli gpu-preflight \
  --planner-profile "$PLANNER_PROFILE" \
  --dataset-dir runtime/training \
  --output-dir runtime/models

python -m training.cli dry-run \
  --planner-profile "$PLANNER_PROFILE" \
  --planner-backend "$PLANNER_BACKEND" \
  --dataset-dir runtime/training \
  --base-model Qwen/Qwen2.5-0.5B-Instruct
```

## Current hardware note

The GTX 1650 4 GB machine is useful for inference validation, but it is still
below the conservative training threshold for this profile. That means the local
machine is expected to support low-spec inference, not safe local QLoRA training.
