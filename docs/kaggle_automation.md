# Kaggle Automation Workflow

This repository includes a developer workflow for pushing and running the Kaggle
semantic-extractor notebook from the local development environment using Kaggle's
official CLI.

## One-time setup

1. Install the Kaggle CLI if it is not already available.
2. Authenticate once using Kaggle's normal credential storage mechanism.
3. Do **not** commit `kaggle.json`, OAuth tokens, or any secret files.

The workflow looks for credentials in the standard locations:

- `~/.kaggle/kaggle.json`
- `KAGGLE_CONFIG_DIR/kaggle.json`
- `KAGGLE_USERNAME` and `KAGGLE_KEY`

## Daily workflow

Run the commands from the repository root:

```bash
python scripts/kaggle_runner.py preflight
python scripts/kaggle_runner.py push
python scripts/kaggle_runner.py run
python scripts/kaggle_runner.py status
python scripts/kaggle_runner.py outputs
python scripts/kaggle_runner.py full-cycle
```

## What the workflow does

- checks Kaggle authentication
- verifies the target notebook ref can be created or already exists
- checks access to the private dataset `jaistudio/data-analysis-llm`
- syncs the local semantic-extractor notebook into a Kaggle staging folder
- configures GPU and internet metadata for notebook execution
- pushes the notebook with the official Kaggle CLI
- polls execution status safely
- downloads only safe generated artifacts
- summarizes failures without printing secrets

## Safe artifacts

Only the following artifact names are intended for download:

- `final_report.json`
- `semantic_metrics.json`
- `artifact_manifest.json`
- `semantic_extractor_artifacts.zip`

The workflow does not automatically download private canonical dataset files.

## Failure handling

The runner reports these failures clearly:

- authentication failure
- missing Kaggle CLI
- dataset access failure
- notebook execution failure
- timeout
- artifact missing
- local pytest failure before full-cycle execution

## Notes

- The workflow never makes the dataset public.
- The workflow never prints tokens or raw credential contents.
- The notebook slug used by default is `data-analysis-llm-semantic-extractor`.
