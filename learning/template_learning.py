from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any
import re

from learning.models import (
    BoundPlan,
    DatasetSemanticProfile,
    LogicalGroup,
    PlanTemplate,
    PredicateNode,
    QueryFeatures,
    stable_hash,
)


_NUMBER_RE = re.compile(r"(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?|between)\s+(?P<value>-?\d+(?:\.\d+)?)", re.I)
_BOOLEAN_HINTS = {"true", "false", "yes", "no", "available", "unavailable", "open", "closed"}


def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _infer_value_kind(value: Any, operator: str | None = None) -> str:
    if isinstance(value, bool):
        return "boolean_true" if value else "boolean_false"
    if operator in {"greater_than", "less_than", "greater_than_equal", "less_than_equal", "between"}:
        return "numeric_comparison"
    if isinstance(value, (int, float)):
        return "numeric_literal"
    if value is None:
        return "unknown"
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "available", "open", "on"}:
        return "boolean_true"
    if lowered in {"false", "no", "unavailable", "closed", "off"}:
        return "boolean_false"
    return "literal"


def _template_kind(plan: dict[str, Any]) -> str:
    if plan.get("derived_columns") or plan.get("group_by") or plan.get("metrics") or plan.get("window") or plan.get("keep_top_n_per_partition"):
        return "workflow"
    return "filter"


def _tool_sequence_for_plan(plan: dict[str, Any], kind: str) -> list[str]:
    if kind == "workflow":
        sequence = ["resolve_semantic_targets"]
        if plan.get("filters"):
            sequence.append("filter_rows")
        if plan.get("derived_columns"):
            sequence.append("derive_columns")
        if plan.get("group_by") or plan.get("metrics"):
            sequence.append("group_and_aggregate")
        if plan.get("window") or plan.get("keep_top_n_per_partition"):
            sequence.append("rank_and_top_n")
        if plan.get("order_by"):
            sequence.append("sort_results")
        sequence.append("validate_result")
        return sequence
    return ["resolve_semantic_targets", "filter_rows", "validate_filter_result"]


def _predicate_structure_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for idx, predicate in enumerate(plan.get("filters") or []):
        nodes.append(
            PredicateNode(
                kind="predicate",
                role=f"role_{idx}",
                operator=str(predicate.get("operator") or "equals"),
                value_kind=_infer_value_kind(predicate.get("value"), str(predicate.get("operator") or "")),
            ).to_dict()
        )
    if plan.get("derived_columns"):
        for derived in plan.get("derived_columns") or []:
            nodes.append(
                PredicateNode(
                    kind="predicate",
                    role="derived_role",
                    operator="derived",
                    value_kind="derived_label",
                ).to_dict()
            )
    return nodes


def extract_template_from_experience(
    *,
    decision: Any,
    features: QueryFeatures,
    dataset_profile: dict[str, Any],
    result_summary: dict[str, Any] | None = None,
) -> PlanTemplate | None:
    plan = decision.plan or {}
    if not plan:
        return None

    plan_kind = _template_kind(plan)
    column_roles = dataset_profile.get("column_roles") or {}
    required_roles: list[str] = []
    for predicate in plan.get("filters") or []:
        column = str(predicate.get("column") or "")
        role = str(column_roles.get(column) or "unknown")
        if role != "unknown":
            required_roles.append(role)
    for group_col in plan.get("group_by") or []:
        role = str(column_roles.get(str(group_col)) or "unknown")
        if role != "unknown":
            required_roles.append(role)
    if not required_roles:
        required_roles = list(features.semantic_roles)

    predicate_structure = _predicate_structure_from_plan(plan)
    tool_sequence = list(decision.tool_sequence or []) or _tool_sequence_for_plan(plan, plan_kind)
    output_contract = {
        "result_kind": (result_summary or {}).get("result_kind") or ("table" if plan_kind == "workflow" else "filtered_rows"),
        "shape": (result_summary or {}).get("shape"),
        "column_count": (result_summary or {}).get("column_count"),
    }

    template_signature = stable_hash(
        {
            "intent": features.intent,
            "plan_kind": plan_kind,
            "required_roles": required_roles,
            "predicate_structure": predicate_structure,
            "logical_structure": features.logical_structure,
            "tool_sequence": tool_sequence,
        }
    )[:16]
    template_id = f"template.{template_signature}.v1"
    average_quality = float((result_summary or {}).get("quality", 0.0) or 0.0)
    if average_quality <= 0:
        average_quality = 0.0
    return PlanTemplate(
        id=template_id,
        intent=features.intent,
        plan_kind=plan_kind,
        required_roles=required_roles,
        predicate_structure=predicate_structure,
        logical_structure=features.logical_structure,
        tool_sequence=tool_sequence,
        output_contract=output_contract,
        dependencies=[],
        source_experience_signature=features.semantic_signature,
        support_count=1,
        average_quality=average_quality,
        state="observed",
        last_seen_at=None,
    )


class ExperienceTemplateExtractor:
    def extract(
        self,
        *,
        decision: Any,
        features: QueryFeatures,
        dataset_profile: dict[str, Any],
        result_summary: dict[str, Any] | None = None,
    ) -> PlanTemplate | None:
        return extract_template_from_experience(
            decision=decision,
            features=features,
            dataset_profile=dataset_profile,
            result_summary=result_summary,
        )


def _parse_query_thresholds(user_text: str) -> list[dict[str, Any]]:
    clauses: list[dict[str, Any]] = []
    text = _strip(user_text)
    for match in _NUMBER_RE.finditer(text):
        op = match.group("op").lower()
        value = match.group("value")
        operator_map = {
            "above": "greater_than",
            "over": "greater_than",
            "greater than": "greater_than",
            "more than": "greater_than",
            "at least": "greater_than_equal",
            "below": "less_than",
            "under": "less_than",
            "less than": "less_than",
            "equal to": "equals",
            "equals": "equals",
            "equal": "equals",
            "between": "between",
        }
        clauses.append({"operator": operator_map.get(op, "greater_than"), "value": value})
    return clauses


def _choose_column_for_role(
    role: str,
    dataset_profile: DatasetSemanticProfile,
    user_text: str | None = None,
    correction_candidates: dict[str, str] | None = None,
) -> str | None:
    available = dataset_profile.available_columns
    column_roles = dataset_profile.column_roles
    if correction_candidates:
        preferred = correction_candidates.get(role)
        if preferred and preferred in available:
            return preferred

    candidates = [column for column in available if column_roles.get(column) == role]
    if candidates:
        if user_text:
            normalized = re.sub(r"[^a-z0-9]+", "", user_text.lower())
            for candidate in candidates:
                if re.sub(r"[^a-z0-9]+", "", candidate.lower()) in normalized:
                    return candidate
        return candidates[0]

    alias_sets = {
        "numeric_measure": {"revenue", "profit", "score", "rating", "amount", "defect", "margin", "sales", "quantity"},
        "dimension": {"category", "region", "country", "city", "type", "segment", "group", "status", "name"},
        "boolean_capability": {"verified", "approved", "express", "active", "delivery", "booking", "available"},
    }
    aliases = alias_sets.get(role, set())
    for candidate in available:
        normalized = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if any(alias in normalized for alias in aliases):
            return candidate
    return None


def _extract_corrections(corrections: list[dict[str, Any]], dataset_signature: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for correction in corrections:
        if correction.get("dataset_semantic_signature") != dataset_signature:
            continue
        requested_role = correction.get("requested_role")
        preferred = correction.get("preferred_semantic_candidate") or correction.get("resolution_preference")
        if requested_role and preferred:
            mapping[str(requested_role)] = str(preferred)
    return mapping


def bind_template(
    template: PlanTemplate,
    *,
    features: QueryFeatures,
    dataset_profile: DatasetSemanticProfile,
    user_text: str | None = None,
    corrections: list[dict[str, Any]] | None = None,
) -> BoundPlan | None:
    correction_map = _extract_corrections(corrections or [], dataset_profile.dataset_semantic_signature)
    resolved_roles: dict[str, str] = {}
    unresolved_roles: list[str] = []
    binding_confidence = 0.5

    for idx, role in enumerate(template.required_roles):
        column = _choose_column_for_role(role, dataset_profile, user_text=user_text, correction_candidates=correction_map)
        if column is None:
            unresolved_roles.append(role)
            continue
        resolved_roles[f"role_{idx}:{role}"] = column
        binding_confidence += 0.08

    if unresolved_roles:
        binding_confidence -= 0.18 * len(unresolved_roles)

    if template.plan_kind == "filter":
        threshold_clauses = _parse_query_thresholds(user_text or "")
        filters: list[dict[str, Any]] = []
        role_columns = list(resolved_roles.values())
        for idx, predicate in enumerate(template.predicate_structure):
            column = role_columns[idx] if idx < len(role_columns) else None
            if column is None:
                continue
            op = predicate.get("operator") or "equals"
            value_kind = predicate.get("value_kind") or "literal"
            value: Any = True if value_kind == "boolean_true" else False if value_kind == "boolean_false" else None
            if op in {"greater_than", "less_than", "greater_than_equal", "less_than_equal", "between"}:
                if threshold_clauses:
                    chosen = threshold_clauses[min(idx, len(threshold_clauses) - 1)]
                    value = chosen.get("value")
                    op = chosen.get("operator") or op
                else:
                    unresolved_roles.append(f"threshold:{idx}")
                    continue
            filters.append({"column": column, "operator": op, "value": value})
        if unresolved_roles and len(filters) < len(template.predicate_structure):
            binding_confidence -= 0.1
        if not filters:
            return None
        plan = {
            "filters": filters,
            "group_by": [],
            "metrics": [],
            "order_by": [],
            "limit": None,
            "logic": template.logical_structure,
            "predicate_graph": [node for node in template.predicate_structure],
        }
    else:
        resolved_columns = list(resolved_roles.values())
        if len(resolved_columns) < 2:
            unresolved_roles.append("workflow_columns")
            return None
        dimension = resolved_columns[0]
        metric = resolved_columns[1]
        metric_operator = "avg" if "average" in " ".join(template.tool_sequence).lower() or "avg" in template.output_contract.get("result_kind", "").lower() else "sum"
        if any(token in template.output_contract.get("result_kind", "").lower() for token in {"top", "rank"}):
            metric_operator = "avg"
        plan = {
            "group_by": [dimension],
            "metrics": [{"column": metric, "function": metric_operator, "alias": f"{metric_operator}_{re.sub(r'\\W+', '_', metric.lower()).strip('_')}"}],
            "order_by": [{"column": f"{metric_operator}_{re.sub(r'\\W+', '_', metric.lower()).strip('_')}", "direction": "desc"}],
            "limit": 1 if "top" in template.tool_sequence or "rank" in template.tool_sequence or "return_top" in template.tool_sequence else None,
            "logic": template.logical_structure,
            "predicate_graph": [node for node in template.predicate_structure],
        }

    if binding_confidence < 0.55:
        return None
    binding_confidence = max(0.0, min(0.99, binding_confidence))
    provenance = {
        "plan_source": "validated_template" if template.state in {"validated", "trusted"} else "experience_transfer",
        "template_id": template.id,
        "experience_support": template.support_count,
        "binding_confidence": round(binding_confidence, 4),
        "skill_ids": [],
    }
    return BoundPlan(
        template_id=template.id,
        plan=plan,
        binding_confidence=round(binding_confidence, 4),
        resolved_roles=resolved_roles,
        unresolved_roles=unresolved_roles,
        provenance=provenance,
    )


class TemplateBinder:
    def bind(
        self,
        template: PlanTemplate,
        *,
        features: QueryFeatures,
        dataset_profile: DatasetSemanticProfile,
        user_text: str | None = None,
        corrections: list[dict[str, Any]] | None = None,
    ) -> BoundPlan | None:
        return bind_template(
            template,
            features=features,
            dataset_profile=dataset_profile,
            user_text=user_text,
            corrections=corrections,
        )
