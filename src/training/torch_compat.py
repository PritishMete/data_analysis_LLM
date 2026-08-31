from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TorchCompatibilityReport:
    torch_imported: bool
    skip_code_present_before: bool | None
    skip_code_patch_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_imported": self.torch_imported,
            "skip_code_present_before": self.skip_code_present_before,
            "skip_code_patch_applied": self.skip_code_patch_applied,
        }


def ensure_torch_dynamo_compatibility() -> TorchCompatibilityReport:
    try:
        import torch
    except Exception:
        return TorchCompatibilityReport(torch_imported=False, skip_code_present_before=None, skip_code_patch_applied=False)

    try:
        dynamo_eval_frame = getattr(getattr(torch, "_C", None), "_dynamo", None)
        eval_frame = getattr(dynamo_eval_frame, "eval_frame", None) if dynamo_eval_frame is not None else None
        if eval_frame is None:
            return TorchCompatibilityReport(torch_imported=True, skip_code_present_before=None, skip_code_patch_applied=False)
        has_skip_code = hasattr(eval_frame, "skip_code")
        if has_skip_code:
            try:
                python_eval_frame = getattr(getattr(torch, "_dynamo", None), "eval_frame", None)
                if python_eval_frame is not None and not hasattr(python_eval_frame, "skip_code"):
                    setattr(python_eval_frame, "skip_code", getattr(eval_frame, "skip_code"))
            except Exception:
                pass
            return TorchCompatibilityReport(torch_imported=True, skip_code_present_before=True, skip_code_patch_applied=False)

        def _skip_code(*args: Any, **kwargs: Any) -> None:
            return None

        setattr(eval_frame, "skip_code", _skip_code)
        try:
            python_eval_frame = getattr(getattr(torch, "_dynamo", None), "eval_frame", None)
            if python_eval_frame is not None and not hasattr(python_eval_frame, "skip_code"):
                setattr(python_eval_frame, "skip_code", _skip_code)
        except Exception:
            pass
        return TorchCompatibilityReport(torch_imported=True, skip_code_present_before=False, skip_code_patch_applied=True)
    except Exception:
        return TorchCompatibilityReport(torch_imported=True, skip_code_present_before=None, skip_code_patch_applied=False)
