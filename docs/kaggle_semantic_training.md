# Kaggle Semantic Training

This notebook workflow prepares and runs the semantic-extractor specialization
for `Qwen/Qwen2.5-0.5B-Instruct` on Kaggle with minimal manual setup.

## What to upload

Create a **private Kaggle dataset** that contains only:

- the semantic training dataset directory
- `dataset_manifest.sha256.json` if available
- the repository notebook and helper files if you want the Kaggle notebook to
  reference them directly

Do **not** upload:

- `kaggle.json`
- secrets or API keys
- raw customer data
- model weights
- checkpoints
- logs
- runtime output

## Notebook

Open `kaggle/semantic_extractor_training.ipynb`.

In Kaggle, only:

1. enable GPU
2. attach the private semantic training dataset
3. run all cells

The notebook discovers the attached dataset automatically under `/kaggle/input/`
and writes all outputs to `/kaggle/working/`.

## Output locations

The workflow stores safe artifacts in:

- `/kaggle/working/checkpoints/`
- `/kaggle/working/adapters/`
- `/kaggle/working/reports/`
- `/kaggle/working/metrics/`
- `/kaggle/working/manifests/`
- `/kaggle/working/semantic_extractor_artifacts.zip`

The ZIP contains only safe artifacts:

- LoRA adapter metadata or adapter files
- safe metrics
- training config
- artifact manifest
- final report
- optional registry metadata

The ZIP must not contain the training dataset.

## SHA-256 verification

If the attached dataset includes `dataset_manifest.sha256.json`, the notebook
verifies it before any training step.

You can also verify the manifest locally with:

```bash
python -m training.cli manifest-verify \
  --dataset-dir /path/to/dataset \
  --manifest-path /path/to/dataset/dataset_manifest.sha256.json
```

## Resume support

If Kaggle reuses a working directory or you pass a prior checkpoint path, the
workflow reuses the latest checkpoint instead of restarting from zero.

## Model and benchmark

The notebook targets only:

- `Qwen/Qwen2.5-0.5B-Instruct`
- semantic extraction only
- no SQL
- no tool IDs
- no executable plan generation

## Post-run benchmark on GTX 1650

After downloading the returned artifacts, benchmark the model on the local GTX
1650 machine with the existing low-spec inference path:

```bash
.venv\Scripts\python.exe -m training.cli inference-benchmark \
  --profile low_spec \
  --backend semantic_extraction \
  --device cuda \
  --benchmark builtin
```

## Optional later upload

If you want to publish the returned model later, upload the safe artifacts to a
private Hugging Face repository only after reviewing the metrics and registry
metadata.
