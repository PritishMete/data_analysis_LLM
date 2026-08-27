# Low-Spec Inference Benchmark

This document records the local benchmark path for the low-spec planner profile.
The benchmark is inference-only. It does not fine-tune or train any model.

## Target profile

- base model: `Qwen/Qwen2.5-0.5B-Instruct`
- profile: `low_spec`
- expected deployment modes:
  - CPU-only, quantized runtime
  - GTX 1650 4 GB, when a safe backend fits

## Backend options

- `transformers`
- `llama_cpp`
- `heuristic` fallback when no local model backend is available

## CLI

```bash
export PLANNER_PROFILE=low_spec
export HF_HOME="$PWD/runtime/model_cache"

python -m training.cli inference-benchmark \
  --profile low_spec \
  --backend auto \
  --device cpu \
  --benchmark builtin \
  --output-dir runtime/benchmark
```

For a GPU attempt:

```bash
python -m training.cli inference-benchmark \
  --profile low_spec \
  --backend auto \
  --device cuda \
  --benchmark builtin \
  --output-dir runtime/benchmark
```

## Measured fields

The benchmark report captures:

- valid JSON rate
- schema validity rate
- plan validity rate
- intent accuracy
- tool-selection F1
- tool-sequence accuracy
- predicate coverage
- logical-structure accuracy
- semantic-role coverage
- invalid-tool rate
- fallback accuracy
- median latency
- p95 latency

## Safety

- no training artifacts are required
- raw workbook data is never used
- benchmark output stays under `runtime/`
- model caches stay under `runtime/model_cache/` or `HF_HOME`

## Practical verdicts

The benchmark is intended to answer two questions:

1. Is CPU inference practically usable for planner-only work?
2. Can a GTX 1650 4 GB safely run the low-spec planner without exhausting VRAM?

If a real model backend cannot load safely, the command should still complete and
report the fallback path rather than crashing.
