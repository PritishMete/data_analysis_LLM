"""Fail-fast validation for durable adapter packages before deployment."""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    root = Path(os.getenv("LEARNER_ADAPTER_ROOT", "models/adapters"))
    failures = []
    for domain in ("myntra", "finance", "restaurant"):
        path = root / domain
        manifest = path / "adapter.manifest.json"
        config = path / "adapter_config.json"
        weights = list(path.glob("adapter_model.*"))
        if not manifest.exists():
            failures.append(f"{domain}: missing adapter.manifest.json")
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("status") != "checkpoint_copied_from_kaggle":
            failures.append(f"{domain}: checkpoint not copied ({data.get('status')})")
        if not config.exists() or not weights:
            failures.append(f"{domain}: missing adapter_config.json or adapter weights")
    if failures:
        print("ADAPTER_VALIDATION_FAILED")
        print("\n".join(failures))
        return 1
    print("ADAPTER_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
