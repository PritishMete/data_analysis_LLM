# Learner adapters

Each domain directory is a durable copy of a Kaggle PEFT adapter. Keep the
base model out of Git; the runtime downloads or mounts it separately.

Expected files per adapter: `adapter_config.json`, `adapter_model.safetensors`
or `adapter_model.bin`, and tokenizer files. Configure the root with
`LEARNER_ADAPTER_ROOT` and enable loading with `LEARNER_ADAPTERS_ENABLED=true`.

The current Kaggle outputs to copy are:

- `myntra_plan_adapter` → `models/adapters/myntra/`
- `finance_candidate` → `models/adapters/finance/`
- `restaurant_plan_adapter_v2` → `models/adapters/restaurant/`

Do not commit the Qwen base model or API keys.
