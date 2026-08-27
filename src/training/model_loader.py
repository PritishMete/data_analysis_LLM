from __future__ import annotations

from dataclasses import dataclass

from .profiles import LOW_SPEC_MODEL_PROFILE, STANDARD_MODEL_PROFILE

@dataclass(slots=True)
class PrototypeModelSpec:
    model_id: str
    parameter_count: str
    license: str
    context_length: int
    recommended_sequence_length: int
    estimated_qlora_vram_gb: str
    recommended_gpu_class: str
    qlora_config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "parameter_count": self.parameter_count,
            "license": self.license,
            "context_length": self.context_length,
            "recommended_sequence_length": self.recommended_sequence_length,
            "estimated_qlora_vram_gb": self.estimated_qlora_vram_gb,
            "recommended_gpu_class": self.recommended_gpu_class,
            "qlora_config": dict(self.qlora_config),
        }


DEFAULT_PROTOTYPE_MODEL = PrototypeModelSpec(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    parameter_count="1.5B",
    license="Apache-2.0",
    context_length=32768,
    recommended_sequence_length=2048,
    estimated_qlora_vram_gb="8-12",
    recommended_gpu_class="RTX 3060 12GB or better",
    qlora_config={
        "enabled": True,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
)

LOW_SPEC_PLANNER_MODEL = PrototypeModelSpec(
    model_id=LOW_SPEC_MODEL_PROFILE.model_id,
    parameter_count=LOW_SPEC_MODEL_PROFILE.parameter_count,
    license="Apache-2.0",
    context_length=LOW_SPEC_MODEL_PROFILE.context_length,
    recommended_sequence_length=LOW_SPEC_MODEL_PROFILE.recommended_sequence_length,
    estimated_qlora_vram_gb="4-8",
    recommended_gpu_class="CPU-only, 4GB GPU optional",
    qlora_config=dict(LOW_SPEC_MODEL_PROFILE.qlora_config),
)

STANDARD_PLANNER_MODEL = PrototypeModelSpec(
    model_id=STANDARD_MODEL_PROFILE.model_id,
    parameter_count=STANDARD_MODEL_PROFILE.parameter_count,
    license="Apache-2.0",
    context_length=STANDARD_MODEL_PROFILE.context_length,
    recommended_sequence_length=STANDARD_MODEL_PROFILE.recommended_sequence_length,
    estimated_qlora_vram_gb="8-12",
    recommended_gpu_class="RTX 3060 12GB or better",
    qlora_config=dict(STANDARD_MODEL_PROFILE.qlora_config),
)


@dataclass(slots=True)
class PrototypeModelRegistryEntry:
    model_id: str
    license: str
    context_length: int
    recommended_sequence_length: int
    qlora_enabled: bool
    minimum_cuda: str = "12.1"
    status: str = "prototype_ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "license": self.license,
            "context_length": self.context_length,
            "recommended_sequence_length": self.recommended_sequence_length,
            "qlora_enabled": self.qlora_enabled,
            "minimum_cuda": self.minimum_cuda,
            "status": self.status,
        }


DEFAULT_MODEL_REGISTRY_ENTRY = PrototypeModelRegistryEntry(
    model_id=DEFAULT_PROTOTYPE_MODEL.model_id,
    license=DEFAULT_PROTOTYPE_MODEL.license,
    context_length=DEFAULT_PROTOTYPE_MODEL.context_length,
    recommended_sequence_length=DEFAULT_PROTOTYPE_MODEL.recommended_sequence_length,
    qlora_enabled=True,
)
