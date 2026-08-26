# test_transformation_registry_duplicates.py
# ─────────────────────────────────────────────────────────────────────────────
# Part 7 (Transformation Registry): duplicate-registration protection.
#
# Registering two different transformation instances under the same `.name`
# must fail loudly (DuplicateTransformationError) instead of the second one
# silently shadowing the first with no warning — that's the exact bug class
# this guards against (e.g. a copy-pasted adapter, or a rename typo that
# happens to collide with an existing name).
#
# Every test snapshots and restores the real registry so these tests never
# leak state into (or depend on) the actual built-in transformations.
#
# Usage:
#   python test_transformation_registry_duplicates.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Any

import pandas as pd

from common.transformations import transformation_registry
from common.transformations.base_transformation import BaseTransformation
from common.transformations.transformation_registry import DuplicateTransformationError


class _DummyTransformation(BaseTransformation):
    """Minimal concrete BaseTransformation for registry tests only — no
    real transformation logic, just enough to satisfy the abstract contract."""

    def __init__(self, name: str, display_name: str | None = None):
        self.name = name
        self.display_name = display_name or name

    def detect(self, text: str, df: pd.DataFrame) -> dict[str, Any]:
        return {"detected": False, "params": {}, "confidence": 0.0}

    def validate(self, df: pd.DataFrame, params: dict[str, Any]) -> None:
        return None

    def preview(self, df: pd.DataFrame, params: dict[str, Any], sample_rows: int = 10) -> dict[str, Any]:
        return {"affected_columns": [], "affected_rows": 0, "before": [], "after": []}

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        return {"dataframe": df.copy(), "metadata": {}}


class _OtherDummyTransformation(_DummyTransformation):
    """A distinct class, used to simulate a *different* implementation
    colliding on the same registry name."""


def _snapshot_registry() -> dict[str, BaseTransformation]:
    return dict(transformation_registry._REGISTRY)


def _restore_registry(snapshot: dict[str, BaseTransformation]) -> None:
    transformation_registry._REGISTRY.clear()
    transformation_registry._REGISTRY.update(snapshot)


def test_duplicate_registration_raises():
    snapshot = _snapshot_registry()
    try:
        first = _DummyTransformation("test_dummy_transform")
        second = _OtherDummyTransformation("test_dummy_transform")
        transformation_registry.register(first)
        try:
            transformation_registry.register(second)
            assert False, "expected DuplicateTransformationError"
        except DuplicateTransformationError as e:
            assert e.name == "test_dummy_transform"
            assert e.existing is first
            assert e.new is second
    finally:
        _restore_registry(snapshot)
    print("test_duplicate_registration_raises: PASS")


def test_duplicate_error_message_includes_existing_and_new_details():
    snapshot = _snapshot_registry()
    try:
        first = _DummyTransformation("test_dummy_transform")
        second = _OtherDummyTransformation("test_dummy_transform")
        transformation_registry.register(first)
        try:
            transformation_registry.register(second)
            assert False, "expected DuplicateTransformationError"
        except DuplicateTransformationError as e:
            message = str(e)
            assert "_DummyTransformation" in message
            assert "_OtherDummyTransformation" in message
            assert "test_dummy_transform" in message
    finally:
        _restore_registry(snapshot)
    print("test_duplicate_error_message_includes_existing_and_new_details: PASS")


def test_registering_same_instance_twice_is_not_a_duplicate():
    """Re-registering the exact same instance (e.g. a module re-imported in
    a test harness) must not raise — only a genuinely different instance
    colliding on the name should."""
    snapshot = _snapshot_registry()
    try:
        only = _DummyTransformation("test_dummy_transform")
        transformation_registry.register(only)
        transformation_registry.register(only)  # should not raise
    finally:
        _restore_registry(snapshot)
    print("test_registering_same_instance_twice_is_not_a_duplicate: PASS")


def test_allow_replace_bypasses_duplicate_check():
    snapshot = _snapshot_registry()
    try:
        first = _DummyTransformation("test_dummy_transform")
        second = _OtherDummyTransformation("test_dummy_transform")
        transformation_registry.register(first)
        transformation_registry.register(second, allow_replace=True)  # should not raise
        assert transformation_registry.get("test_dummy_transform") is second
    finally:
        _restore_registry(snapshot)
    print("test_allow_replace_bypasses_duplicate_check: PASS")


def test_built_in_transformations_do_not_collide_with_each_other():
    """Sanity check on the real registry: every currently-registered
    built-in transformation must have a unique name (this would already be
    true by construction, but re-registering each one now must not raise)."""
    for name, transformation in transformation_registry.all_transformations().items():
        # Re-registering the exact same live instance must be a no-op, not
        # a DuplicateTransformationError.
        transformation_registry.register(transformation)
        assert transformation_registry.get(name) is transformation
    print("test_built_in_transformations_do_not_collide_with_each_other: PASS")


def test_missing_name_still_raises_value_error_not_duplicate_error():
    """A transformation that never set `.name` should still fail with the
    existing ValueError, not be mistaken for a duplicate-name collision."""
    unnamed = _DummyTransformation("unnamed_transformation")
    try:
        transformation_registry.register(unnamed)
        assert False, "expected ValueError"
    except DuplicateTransformationError:
        assert False, "missing-name should raise plain ValueError, not DuplicateTransformationError"
    except ValueError:
        pass
    print("test_missing_name_still_raises_value_error_not_duplicate_error: PASS")


if __name__ == "__main__":
    test_duplicate_registration_raises()
    test_duplicate_error_message_includes_existing_and_new_details()
    test_registering_same_instance_twice_is_not_a_duplicate()
    test_allow_replace_bypasses_duplicate_check()
    test_built_in_transformations_do_not_collide_with_each_other()
    test_missing_name_still_raises_value_error_not_duplicate_error()
    print("\nAll transformation registry duplicate-protection tests passed.")
