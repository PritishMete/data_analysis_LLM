# Semantic Extractor Runbook

This runbook prepares the new `Qwen/Qwen2.5-0.5B-Instruct` specialization target
for semantic extraction only.

The semantic extractor does not generate tool IDs or SQL. It emits only:

- intent
- semantic bindings
- predicate graph
- aggregation
- ranking
- limit
- requires_fallback
- confidence

## Dataset preparation

Build the semantic-only corpus from the canonical planner corpus:

```bash
python -m training.cli semantic-dataset \
  --output-dir runtime/semantic_training \
  --include-candidate-strategies \
  --persist
```

This writes:

- `runtime/semantic_training/train.jsonl`
- `runtime/semantic_training/validation.jsonl`
- `runtime/semantic_training/test.jsonl`
- `runtime/semantic_training/readiness.json`

The conversion is semantic-only. It does not include raw query text, workbook
values, tool IDs, SQL, filenames, or private identifiers.

## Readiness gate

The semantic extractor dataset is considered ready when the readiness report
meets the configured gates:

- intent diversity is sufficient
- predicate-structure diversity is sufficient
- role coverage is sufficient
- ambiguity rate is low
- average quality is high enough

## External QLoRA training

Use the dedicated semantic extractor config:

```bash
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_DIR="$PWD/runtime/semantic_training"
MODEL_OUTPUT_DIR="$PWD/runtime/models/semantic_extractor"
HF_HOME="$HOME/.cache/huggingface"

python -m training.cli gpu-preflight \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$MODEL_OUTPUT_DIR" \
  --manifest-path "$DATASET_DIR/dataset_manifest.sha256.json" \
  --planner-profile low_spec
```

The external trainer should then load `configs/qwen25_0_5b_semantic_qlora.yaml`
and train the model off-machine. Do not fine-tune locally on the low-spec host.

## Post-training benchmark

After receiving artifacts back from the external machine, benchmark on the local
GTX 1650 and CPU:

```bash
python -m training.cli inference-benchmark \
  --profile low_spec \
  --backend semantic_extraction \
  --device cuda \
  --benchmark builtin \
  --case-limit 3 \
  --case-timeout-seconds 120 \
  --progress \
  --output-dir runtime/benchmark_semantic_extract_cuda

python -m training.cli inference-benchmark \
  --profile low_spec \
  --backend semantic_composed \
  --device cpu \
  --benchmark builtin \
  --case-limit 3 \
  --case-timeout-seconds 120 \
  --progress \
  --output-dir runtime/benchmark_semantic_composed_cpu
```

## Artifact return workflow

Return the following artifacts from the external training machine:

- LoRA adapter
- optional merged model
- optional GGUF `Q4_K_M`
- SHA-256 manifest
- registry metadata

## Routing order

The runtime should remain:

1. trusted strategy
2. semantic extractor
3. deterministic composer
4. critic
5. Gemini fallback

Shadow mode only. Do not promote the local model to production from this path.

