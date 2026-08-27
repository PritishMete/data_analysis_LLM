from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PLANNER_PROFILE_LOW_SPEC = "low_spec"
PLANNER_PROFILE_STANDARD = "standard"
PLANNER_RUNTIME_CPU_LOW_SPEC = "cpu_low_spec"
PLANNER_RUNTIME_GPU_4GB = "gpu_4gb"

PLANNER_BACKEND_AUTO = "auto"
PLANNER_BACKEND_TRANSFORMERS = "transformers"
PLANNER_BACKEND_LLAMA_CPP = "llama_cpp"


@dataclass(slots=True)
class PlannerModelProfile:
    profile_name: str
    model_id: str
    parameter_count: str
    context_length: int
    recommended_sequence_length: int
    training_min_vram_gb: float
    training_recommended_vram_gb: float
    inference_min_ram_gb: float
    inference_recommended_ram_gb: float
    inference_gpu_vram_gb: float | None
    qlora_config: dict[str, Any]
    gguf_quantization: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "model_id": self.model_id,
            "parameter_count": self.parameter_count,
            "context_length": self.context_length,
            "recommended_sequence_length": self.recommended_sequence_length,
            "training_min_vram_gb": self.training_min_vram_gb,
            "training_recommended_vram_gb": self.training_recommended_vram_gb,
            "inference_min_ram_gb": self.inference_min_ram_gb,
            "inference_recommended_ram_gb": self.inference_recommended_ram_gb,
            "inference_gpu_vram_gb": self.inference_gpu_vram_gb,
            "qlora_config": dict(self.qlora_config),
            "gguf_quantization": self.gguf_quantization,
            "notes": list(self.notes or []),
        }


@dataclass(slots=True)
class PlannerRuntimeProfile:
    profile_name: str
    backend_preference: tuple[str, ...]
    require_cuda: bool
    quantized_runtime: str
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "backend_preference": list(self.backend_preference),
            "require_cuda": self.require_cuda,
            "quantized_runtime": self.quantized_runtime,
            "notes": list(self.notes or []),
        }


LOW_SPEC_MODEL_PROFILE = PlannerModelProfile(
    profile_name=PLANNER_PROFILE_LOW_SPEC,
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    parameter_count="0.5B",
    context_length=32768,
    recommended_sequence_length=1024,
    training_min_vram_gb=8.0,
    training_recommended_vram_gb=12.0,
    inference_min_ram_gb=4.0,
    inference_recommended_ram_gb=8.0,
    inference_gpu_vram_gb=4.0,
    qlora_config={
        "enabled": True,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "gradient_checkpointing": True,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_len": 1024,
    },
    gguf_quantization="Q4_K_M",
    notes=[
        "optimized for structured analytics planning only",
        "supports CPU-only inference via quantized runtime",
        "supports 4GB GPU inference with offload-aware settings",
    ],
)

STANDARD_MODEL_PROFILE = PlannerModelProfile(
    profile_name=PLANNER_PROFILE_STANDARD,
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    parameter_count="1.5B",
    context_length=32768,
    recommended_sequence_length=2048,
    training_min_vram_gb=12.0,
    training_recommended_vram_gb=16.0,
    inference_min_ram_gb=8.0,
    inference_recommended_ram_gb=12.0,
    inference_gpu_vram_gb=8.0,
    qlora_config={
        "enabled": True,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "gradient_checkpointing": True,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_len": 2048,
    },
    gguf_quantization="Q4_K_M",
    notes=[
        "existing prototype training path remains intact",
        "recommended for 12GB+ GPU training",
    ],
)

CPU_LOW_SPEC_RUNTIME = PlannerRuntimeProfile(
    profile_name=PLANNER_RUNTIME_CPU_LOW_SPEC,
    backend_preference=(PLANNER_BACKEND_LLAMA_CPP, PLANNER_BACKEND_TRANSFORMERS),
    require_cuda=False,
    quantized_runtime="gguf",
    notes=["no CUDA dependency", "structured JSON output only", "critic validation required"],
)

GPU_4GB_RUNTIME = PlannerRuntimeProfile(
    profile_name=PLANNER_RUNTIME_GPU_4GB,
    backend_preference=(PLANNER_BACKEND_TRANSFORMERS, PLANNER_BACKEND_LLAMA_CPP),
    require_cuda=False,
    quantized_runtime="transformers",
    notes=["partial GPU offload only", "safe fallback to CPU when CUDA is unavailable"],
)


MODEL_PROFILES = {
    PLANNER_PROFILE_LOW_SPEC: LOW_SPEC_MODEL_PROFILE,
    PLANNER_PROFILE_STANDARD: STANDARD_MODEL_PROFILE,
}

RUNTIME_PROFILES = {
    PLANNER_RUNTIME_CPU_LOW_SPEC: CPU_LOW_SPEC_RUNTIME,
    PLANNER_RUNTIME_GPU_4GB: GPU_4GB_RUNTIME,
}


def select_model_profile(name: str | None = None) -> PlannerModelProfile:
    profile_name = (name or PLANNER_PROFILE_STANDARD).strip().lower()
    return MODEL_PROFILES.get(profile_name, STANDARD_MODEL_PROFILE)


def select_runtime_profile(name: str | None = None) -> PlannerRuntimeProfile:
    profile_name = (name or PLANNER_RUNTIME_CPU_LOW_SPEC).strip().lower()
    return RUNTIME_PROFILES.get(profile_name, CPU_LOW_SPEC_RUNTIME)


def choose_backend(*, backend: str | None, runtime_profile: PlannerRuntimeProfile, cuda_available: bool, llama_cpp_available: bool) -> str:
    requested = (backend or PLANNER_BACKEND_AUTO).strip().lower()
    if requested in {PLANNER_BACKEND_TRANSFORMERS, PLANNER_BACKEND_LLAMA_CPP}:
        return requested
    for candidate in runtime_profile.backend_preference:
        if candidate == PLANNER_BACKEND_TRANSFORMERS and cuda_available:
            return candidate
        if candidate == PLANNER_BACKEND_LLAMA_CPP and llama_cpp_available:
            return candidate
        if candidate == PLANNER_BACKEND_TRANSFORMERS and not cuda_available:
            return candidate
    return PLANNER_BACKEND_TRANSFORMERS if cuda_available else PLANNER_BACKEND_LLAMA_CPP
