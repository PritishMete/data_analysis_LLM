from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QLoRAConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "enabled": self.enabled,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
        }
