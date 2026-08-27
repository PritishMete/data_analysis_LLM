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
    "HardwareReport",
    "PromotionGateResult",
    "PrototypeModelRegistryEntry",
    "PrototypeModelSpec",
    "TrainingConfig",
    "TrainingMetrics",
    "build_manifest_fingerprint",
    "create_dataset_manifest",
    "detect_hardware",
    "evaluate_promotion_gates",
    "evaluate_training_metrics",
    "load_default_config",
    "main",
    "DEFAULT_MODEL_REGISTRY_ENTRY",
    "DEFAULT_PROTOTYPE_MODEL",
    "validate_dataset",
    "verify_dataset_manifest",
    "write_dataset_manifest",
]
