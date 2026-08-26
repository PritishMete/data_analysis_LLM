# common/transformations/transformation_registry.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for "which transformations exist". Concrete
# transformations register themselves by calling `register(...)` at import
# time (see common/transformations/adapters/*.py and the bottom of this
# package's __init__.py, which imports every adapter module exactly once).
#
# Adding a new transformation requires ONLY:
#   1. Write a class implementing BaseTransformation in adapters/your_thing.py
#   2. Call register(YourThing()) at the bottom of that file
#   3. Import that module from common/transformations/__init__.py
# Nothing else changes — not the engine, not the routes, not the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import inspect

import pandas as pd

from common.transformations.base_transformation import BaseTransformation

_REGISTRY: dict[str, BaseTransformation] = {}


class DuplicateTransformationError(ValueError):
    """Raised when register() is called with a `.name` that's already
    registered to a *different* transformation instance/class. Prevents two
    adapters from silently shadowing each other (whichever imported last
    would otherwise win with no warning)."""

    def __init__(self, name: str, existing: BaseTransformation, new: BaseTransformation):
        self.name = name
        self.existing = existing
        self.new = new

        def _describe(t: BaseTransformation) -> str:
            cls = type(t)
            try:
                file = inspect.getfile(cls)
            except (TypeError, OSError):
                file = "<unknown file>"
            return f"class={cls.__module__}.{cls.__qualname__}, file={file}"

        super().__init__(
            f"Transformation name '{name}' is already registered — refusing to overwrite it.\n"
            f"  Existing: {_describe(existing)}\n"
            f"  New:      {_describe(new)}\n"
            "Rename one of these transformations (each `.name` must be globally unique) "
            "or, if this is intentional (e.g. a test fixture replacing a built-in), "
            "call register(transformation, allow_replace=True)."
        )


def register(transformation: BaseTransformation, *, allow_replace: bool = False) -> BaseTransformation:
    """Registers a transformation under its `.name`. Returns the
    transformation so it can be used as `THING = register(MyThing())`.

    Raises DuplicateTransformationError if `.name` is already taken by a
    different instance and `allow_replace` is not explicitly set — this is
    what stops a copy-pasted adapter (or a rename typo landing on an
    existing name) from silently shadowing an existing transformation with
    no error at import time.
    """
    if not transformation.name or transformation.name == "unnamed_transformation":
        raise ValueError(f"{type(transformation).__name__} must set a real `.name` before registering.")

    existing = _REGISTRY.get(transformation.name)
    if existing is not None and existing is not transformation and not allow_replace:
        raise DuplicateTransformationError(transformation.name, existing, transformation)

    _REGISTRY[transformation.name] = transformation
    return transformation


def get(name: str) -> BaseTransformation | None:
    return _REGISTRY.get(name)


def all_transformations() -> dict[str, BaseTransformation]:
    """Read-only-by-convention snapshot of the registry (name -> instance)."""
    return dict(_REGISTRY)


def names() -> list[str]:
    return list(_REGISTRY.keys())


def clear() -> None:
    """Test-only: wipes the registry. Never called from application code."""
    _REGISTRY.clear()


def detect_transformation(text: str, df: pd.DataFrame) -> tuple[BaseTransformation, dict] | None:
    """Rule-based (no LLM) intent routing across ALL registered
    transformations: asks every one of them `.detect(text, df)` and returns
    the highest-confidence match above a minimal threshold, or None if
    nothing plausible fired.

    This is the single place "which transformation did the user mean"
    gets decided — callers (TransformationEngine, query_router.py) should
    use this instead of hardcoding a specific transformation's own
    detect() the way the range_binning fast-path used to.
    """
    best: tuple[BaseTransformation, dict] | None = None
    best_confidence = 0.0
    for transformation in _REGISTRY.values():
        try:
            detection = transformation.detect(text, df)
        except Exception:
            # A single misbehaving transformation's detect() must never
            # break intent routing for every other transformation.
            continue
        if not detection or not detection.get("detected"):
            continue
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        if confidence > best_confidence:
            best = (transformation, detection)
            best_confidence = confidence
    if best is None or best_confidence < 0.4:
        return None
    return best
