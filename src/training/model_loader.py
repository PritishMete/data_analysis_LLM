from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PrototypeModelSpec:
    model_id: str
    parameter_count: str
    license: str
    context_length: int
    recommended_sequence_length: int
    estimated_qlora_vram_gb: str
    recommended_gpu_class: str


DEFAULT_PROTOTYPE_MODEL = PrototypeModelSpec(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    parameter_count="1.5B",
    license="Apache-2.0",
    context_length=32768,
    recommended_sequence_length=2048,
    estimated_qlora_vram_gb="8-12",
    recommended_gpu_class="RTX 3060 12GB or better",
)
