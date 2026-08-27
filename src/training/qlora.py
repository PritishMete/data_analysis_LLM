from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QLoRAConfig:
    enabled: bool = True
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
