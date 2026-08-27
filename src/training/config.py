from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .profiles import PLANNER_PROFILE_STANDARD
from .qlora import QLoRAConfig

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PLANNER_PROFILE = PLANNER_PROFILE_STANDARD
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_TRAINING_DIR = Path("runtime") / "fine_tuning"
DEFAULT_DATASET_DIR = Path("runtime") / "training"
DEFAULT_OUTPUT_DIR = Path("runtime") / "fine_tuning" / "runs"
DEFAULT_MANIFEST_NAME = "dataset_manifest.json"
DEFAULT_MANIFEST_SHA256_NAME = "dataset_manifest.sha256"


@dataclass(slots=True)
class TrainingConfig:
    base_model: str = DEFAULT_BASE_MODEL
    method: str = "qlora"
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    dataset_dir: Path = DEFAULT_DATASET_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    training_dir: Path = DEFAULT_TRAINING_DIR
    python_target: str = "3.11-3.12"
    planner_profile: str = DEFAULT_PLANNER_PROFILE
    planner_backend: str = "auto"
    smoke_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / DEFAULT_MANIFEST_NAME

    @property
    def manifest_sha256_path(self) -> Path:
        return self.dataset_dir / DEFAULT_MANIFEST_SHA256_NAME


@dataclass(slots=True)
class LowSpecTrainingConfigV2:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    method: str = "qlora"
    qlora: QLoRAConfig = field(default_factory=lambda: QLoRAConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    ))
    epochs: int = 4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_seq_len: int = 1024
    validation_metrics: tuple[str, ...] = (
        "plan_validity",
        "tool_selection_f1",
        "predicate_coverage",
        "logical_structure_accuracy",
        "semantic_role_coverage",
    )
    success_gates: dict[str, float] = field(
        default_factory=lambda: {
            "valid_json_rate": 0.99,
            "schema_valid_rate": 0.99,
            "plan_validity_rate": 0.90,
            "tool_selection_f1": 0.90,
            "predicate_coverage": 0.90,
            "logical_structure_accuracy": 0.90,
            "semantic_role_coverage": 0.90,
            "invalid_tool_rate": 0.01,
        }
    )
    tool_descriptions: dict[str, str] = field(
        default_factory=lambda: {
            "sql.filter": "filter rows by a boolean predicate",
            "sql.group_by": "group rows and aggregate a measure",
            "analytics.summary": "summarize dataset shape and safe stats",
            "common.transformations.range_binning": "bin numeric values into labeled ranges",
            "categorization_agent._deterministic_special_mapping": "categorize or normalize values deterministically",
            "data_cleaning_utils.fill_nulls": "fill missing values with a fixed strategy",
            "secure_excel.executor": "run workbook logic locally and safely",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "base_model": self.base_model,
            "method": self.method,
            "qlora": self.qlora.to_dict(),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_seq_len": self.max_seq_len,
            "validation_metrics": list(self.validation_metrics),
            "success_gates": dict(self.success_gates),
            "tool_descriptions": dict(self.tool_descriptions),
        }
        return payload


def load_default_config() -> TrainingConfig:
    return TrainingConfig()
