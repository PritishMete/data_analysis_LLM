from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HardwareReport:
    python_version: str
    platform: str
    machine: str
    processor: str
    ram_gb: float | None
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    vram_gb: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "processor": self.processor,
            "ram_gb": self.ram_gb,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
        }


def detect_hardware() -> HardwareReport:
    ram_gb = None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        pass

    torch_version = None
    cuda_available = False
    cuda_version = None
    gpu_name = None
    vram_gb = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = round(props.total_memory / 1024**3, 2)
    except Exception:
        pass

    return HardwareReport(
        python_version=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        ram_gb=ram_gb,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
    )
