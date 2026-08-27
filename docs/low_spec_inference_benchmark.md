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
  --case-limit 3 \
  --case-timeout-seconds 120 \
  --progress \
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

## Real GTX 1650 subset result

The first real CUDA benchmark completed successfully on the local NVIDIA GTX 1650
using the untouched `Qwen/Qwen2.5-0.5B-Instruct` model.

Observed subset metrics from 3 representative queries:

- model load time: `6811.802 ms`
- median inference latency: `13991.405 ms`
- p95 inference latency: `14196.066 ms`
- median end-to-end latency: `27013.227 ms`
- peak VRAM: `15.201 MB`
- valid JSON rate: `1.0`
- schema validity rate: `1.0`
- plan validity rate: `0.0`
- tool-selection F1: `0.0`
- tool-sequence accuracy: `0.0`
- predicate coverage: `0.0`
- logical-structure accuracy: `0.3333333333333333`
- semantic-role coverage: `0.0`
- invalid-tool rate: `0.6666666666666666`
- critic pass rate: `0.0`
- fallback rate: `1.0`

Interpretation:

- The GTX 1650 path is usable enough to run and measure.
- Planner quality is not acceptable for direct use because the model emitted
  unrelated tool plans and failed the structural quality gates.
- Latency is high enough that interactive planner use would feel slow.

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

## Shadow planner update

The recommended low-spec architecture now treats the 0.5B model as a semantic
planner in shadow mode:

1. trusted strategies route first
2. the local model emits semantic bindings, predicate structure, aggregation,
   and ranking hints
3. the executable tool graph is composed deterministically from the detected
   intent and allowlisted tools
4. critic and validators gate acceptance
5. Gemini remains the fallback when JSON, schema, predicate parity, tool
   allowlisting, ambiguity checks, or critic checks fail

This path is benchmarked against the older full-plan generation path so the
quality/latency tradeoff can be compared without executing shadow plans on user
data.
