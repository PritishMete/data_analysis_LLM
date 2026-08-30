from __future__ import annotations

import json
import os
import time
import sys
from pathlib import Path
from typing import Any

DEFAULT_IMPORT_TRACE_PATH = Path("/kaggle/working/reports/import_trace.jsonl")


def runtime_import_state(*, compatibility_patch_ran: bool = False) -> dict[str, bool]:
    return {
        "torch_loaded": "torch" in globals() or "torch" in globals().get("__builtins__", {}),
        "transformers_loaded": "transformers" in globals() or "transformers" in globals().get("__builtins__", {}),
        "peft_loaded": "peft" in globals() or "peft" in globals().get("__builtins__", {}),
        "compatibility_patch_ran": bool(compatibility_patch_ran),
    }


def _module_loaded(name: str) -> bool:
    return name in sys.modules


def write_import_trace(
    path: Path | str = DEFAULT_IMPORT_TRACE_PATH,
    *,
    module: str,
    event: str,
    compatibility_patch_ran: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "pid": os.getpid(),
        "module": module,
        "event": event,
        "torch_loaded": _module_loaded("torch"),
        "transformers_loaded": _module_loaded("transformers"),
        "peft_loaded": _module_loaded("peft"),
        "compatibility_patch_ran": bool(compatibility_patch_ran),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except Exception:
            pass
    return payload
