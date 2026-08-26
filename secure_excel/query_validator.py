"""Validation for structured Excel commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .query_types import SUPPORTED_OPERATIONS, SUPPORTED_OPERATORS


NUMERIC_ROLES = {"numeric_metric", "currency_metric", "rating_metric", "count", "percentage"}
TEXT_ROLES = {
    "entity_name",
    "restaurant_entity",
    "customer_entity",
    "product_entity",
    "supplier_entity",
    "employee_entity",
    "geographic_area",
    "category",
    "description",
    "status",
    "identifier",
    "email",
    "phone",
    "url",
    "address",
}
BOOLEAN_ROLES = {"boolean_capability", "delivery_capability", "table_booking_capability"}


@dataclass
class ValidationError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


def _column_profile(schema: dict[str, Any], column_id: str) -> dict[str, Any] | None:
    for column in schema.get("columns", []):
        if column["column_id"] == column_id:
            return column
    return None


def _coerce_value(column_role: str, value: Any) -> Any:
    if value is None:
        return None
    if column_role in BOOLEAN_ROLES:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "available", "open", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "not available", "closed", "off"}:
                return False
        raise ValidationError(f"Value {value!r} is not valid for a boolean-capability column.")
    if column_role in NUMERIC_ROLES:
        if value is None:
            return value
        try:
            return float(value)
        except Exception as exc:
            raise ValidationError(f"Value {value!r} is not numeric.") from exc
    return value


def validate_structured_query(query: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    operation = query.get("operation")
    if operation not in SUPPORTED_OPERATIONS:
        raise ValidationError(f"Unsupported operation: {operation!r}")

    normalized_conditions = []
    for condition in query.get("conditions", []):
        column_id = condition.get("column_id")
        operator = condition.get("operator")
        if not column_id:
            raise ValidationError("Condition is missing a column_id.")
        if operator not in SUPPORTED_OPERATORS:
            raise ValidationError(f"Unsupported operator: {operator!r}")
        column = _column_profile(schema, column_id)
        if column is None:
            raise ValidationError(f"Unknown column_id: {column_id!r}")
        role = column["role"]
        value = _coerce_value(role, condition.get("value"))
        value2 = _coerce_value(role, condition.get("value2"))

        if operator in {"greater_than", "less_than", "greater_equal", "less_equal", "between"} and role not in NUMERIC_ROLES | {"date"}:
            raise ValidationError(f"Operator {operator!r} is not compatible with role {role!r}.")
        if operator in {"contains", "starts_with", "ends_with"} and role in BOOLEAN_ROLES | NUMERIC_ROLES:
            raise ValidationError(f"Operator {operator!r} is not compatible with role {role!r}.")

        normalized_conditions.append({
            "column_id": column_id,
            "operator": operator,
            "value": value,
            "value2": value2,
        })

    normalized = dict(query)
    normalized["conditions"] = normalized_conditions
    return normalized
