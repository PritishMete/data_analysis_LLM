# common/transformations/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
# Importing this package registers every built-in transformation exactly
# once (each adapters/*.py module calls register(...) at import time) and
# exposes the centralized engine's public API.
#
# To add a new transformation:
#   1. Write common/transformations/adapters/your_thing.py implementing
#      BaseTransformation, ending with `register(YourThing())`.
#   2. Add one `from common.transformations.adapters import your_thing` line
#      below.
# Nothing else changes.
# ─────────────────────────────────────────────────────────────────────────────

from common.transformations.base_transformation import BaseTransformation, TransformationError
from common.transformations.transformation_result import TransformationResult
from common.transformations.transformation_history import TransformationHistory, TransformationHistoryEntry
from common.transformations.transformation_engine import TransformationEngine, diff_schema
from common.transformations.transformation_registry import (
    DuplicateTransformationError,
    all_transformations,
    detect_transformation,
    get as get_transformation,
    names as transformation_names,
    register,
)

# Import every adapter so it self-registers. Order doesn't matter.
from common.transformations.adapters import range_binning_transformation  # noqa: F401
from common.transformations.adapters import rename_columns  # noqa: F401
from common.transformations.adapters import drop_columns  # noqa: F401
from common.transformations.adapters import fill_missing  # noqa: F401
from common.transformations.adapters import remove_duplicates  # noqa: F401
from common.transformations.adapters import merge_columns  # noqa: F401
from common.transformations.adapters import split_column  # noqa: F401
from common.transformations.adapters import type_conversion  # noqa: F401
from common.transformations.adapters import date_features  # noqa: F401

__all__ = [
    "BaseTransformation",
    "TransformationError",
    "DuplicateTransformationError",
    "TransformationResult",
    "TransformationHistory",
    "TransformationHistoryEntry",
    "TransformationEngine",
    "diff_schema",
    "all_transformations",
    "detect_transformation",
    "get_transformation",
    "transformation_names",
    "register",
]
