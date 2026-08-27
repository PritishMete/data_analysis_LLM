from .config import TrainingConfig, load_default_config
from .cli import main
from .dataset import (
    DatasetValidationResult,
    build_manifest_fingerprint,
    create_dataset_manifest,
    validate_dataset,
    verify_dataset_manifest,
    write_dataset_manifest,
)
from .execution import (
    ExperimentRunMetadata,
    ExperimentSummary,
    PreflightResult,
    SeedBundle,
    create_experiment_id,
    decide_final_promotion,
    derive_seed_bundle,
    init_experiment,
    preflight_gpu_training,
    recommended_oom_actions,
    record_safe_metrics,
    update_shadow_registry_status,
    write_experiment_summary,
)
from .hardware import HardwareReport, detect_hardware
from .metrics import TrainingMetrics, evaluate_training_metrics
from .model_loader import (
    DEFAULT_MODEL_REGISTRY_ENTRY,
    DEFAULT_PROTOTYPE_MODEL,
    PrototypeModelRegistryEntry,
    PrototypeModelSpec,
)
from .promotion import PromotionGateResult, evaluate_promotion_gates

__all__ = [
    "DatasetValidationResult",
    "ExperimentRunMetadata",
    "ExperimentSummary",
    "HardwareReport",
    "PromotionGateResult",
    "PrototypeModelRegistryEntry",
    "PrototypeModelSpec",
    "PreflightResult",
    "TrainingConfig",
    "TrainingMetrics",
    "SeedBundle",
    "build_manifest_fingerprint",
    "create_dataset_manifest",
    "create_experiment_id",
    "decide_final_promotion",
    "derive_seed_bundle",
    "detect_hardware",
    "evaluate_promotion_gates",
    "evaluate_training_metrics",
    "init_experiment",
    "load_default_config",
    "main",
    "DEFAULT_MODEL_REGISTRY_ENTRY",
    "DEFAULT_PROTOTYPE_MODEL",
    "preflight_gpu_training",
    "recommended_oom_actions",
    "record_safe_metrics",
    "validate_dataset",
    "verify_dataset_manifest",
    "update_shadow_registry_status",
    "write_experiment_summary",
    "write_dataset_manifest",
]
