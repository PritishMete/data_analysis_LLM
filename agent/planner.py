from __future__ import annotations

from typing import Any
import json
import re

import pandas as pd

from learning.feature_extractor import build_planner_context
from learning.models import DatasetSemanticProfile, LearningDecision, PlanTemplate, PlannerContext, stable_hash
from learning.skill_registry import SkillRegistry, get_skill_registry
from learning.template_learning import TemplateBinder

from secure_excel.semantic_roles import detect_column_role


_INTENT_TOOL_SUBSETS: dict[str, list[str]] = {
    "filter": ["sql.filter", "analytics.summary"],
    "analytics": ["sql.group_by", "sql.filter", "analytics.summary"],
    "cleaning": [
        "categorization_agent._deterministic_special_mapping",
        "data_cleaning_utils.fill_nulls",
        "common.transformations.range_binning",
        "secure_excel.executor",
    ],
    "operation": [
        "categorization_agent._deterministic_special_mapping",
        "data_cleaning_utils.fill_nulls",
        "common.transformations.range_binning",
        "secure_excel.executor",
    ],
    "sentiment": ["analytics.summary"],
}

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "sql.filter": "Filter rows by an explicit predicate.",
    "sql.group_by": "Group rows and summarize a measure.",
    "analytics.summary": "Summarize shape and safe aggregates.",
    "common.transformations.range_binning": "Bin numeric values into stable ranges.",
    "categorization_agent._deterministic_special_mapping": "Normalize or categorize values deterministically.",
    "data_cleaning_utils.fill_nulls": "Fill missing values with a fixed strategy.",
    "secure_excel.executor": "Run workbook logic locally and safely.",
}


_GENERIC_ENTITY_STOPWORDS = {
    "restaurant",
    "restaurants",
    "row",
    "rows",
    "record",
    "records",
    "item",
    "items",
    "data",
    "entry",
    "entries",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _intent_tool_subset(intent: str) -> list[str]:
    return list(_INTENT_TOOL_SUBSETS.get(intent, ["analytics.summary"]))


def _semantic_role_descriptions(roles: list[str]) -> list[dict[str, str]]:
    descriptions = {
        "boolean_capability": "Boolean yes/no capability or presence flag.",
        "numeric_metric": "Numeric measure that can be compared or aggregated.",
        "rating_metric": "Rating or score measure suitable for ordering or thresholds.",
        "geographic_area": "Location or region dimension.",
        "category": "Categorical grouping field.",
        "entity_name": "Name-like entity field.",
        "customer_entity": "Customer-facing entity name.",
        "restaurant_entity": "Restaurant or venue entity name.",
        "product_entity": "Product entity name.",
        "supplier_entity": "Supplier entity name.",
        "employee_entity": "Employee entity name.",
        "text_summary": "Text summary or description field.",
    }
    return [{"role": role, "description": descriptions.get(role, "Semantic role used for planner grounding.")} for role in roles]


def _stable_plan_template_id(*, intent: str, logical_structure: str, semantic_roles: list[str], tool_sequence: list[str], predicate_count: int) -> str:
    signature = stable_hash(
        {
            "intent": intent,
            "logical_structure": logical_structure,
            "semantic_roles": sorted(set(semantic_roles)),
            "tool_sequence": list(tool_sequence),
            "predicate_count": int(predicate_count),
        }
    )
    return f"template.{signature[:12]}"


def _column_roles(df: pd.DataFrame | None, columns: list[str]) -> dict[str, str]:
    if df is None:
        return {str(column): "unknown" for column in columns}
    roles: dict[str, str] = {}
    for column in columns:
        try:
            roles[str(column)] = detect_column_role(str(column), df[column])["role"]
        except Exception:
            roles[str(column)] = "unknown"
    return roles


def _choose_column(columns: list[str], text: str, *, role_hint: str | None = None, roles: dict[str, str] | None = None) -> str | None:
    normalized = _normalize_text(text)
    compact = _compact(text)
    for column in columns:
        n = _normalize_text(column)
        c = _compact(column)
        if n and (n in normalized or c and c in compact):
            return column
    if role_hint:
        role_aliases = {
            "entity_name": {"name", "restaurant", "customer", "product", "entity", "title"},
            "rating_metric": {"rating", "score", "stars"},
            "boolean_capability": {"delivery", "book", "booking", "available", "open"},
            "geographic_area": {"city", "country", "region", "state", "location"},
            "numeric_metric": {"amount", "value", "count", "total", "price", "revenue", "sales"},
        }
        aliases = role_aliases.get(role_hint, set())
        for column in columns:
            column_norm = _normalize_text(column)
            if any(alias in column_norm for alias in aliases):
                return column
    if roles:
        for column, role in roles.items():
            if role == role_hint:
                return column
    return columns[0] if columns else None


def _build_sql_filter_plan(text: str, columns: list[str], roles: dict[str, str]) -> dict[str, Any] | None:
    normalized = _normalize_text(text)
    compact_text = _compact(text)
    if not any(token in normalized for token in {"show", "find", "filter", "list", "display", "view", "return", "with", "having", "where"}):
        return None

    filters: list[dict[str, Any]] = []
    seen_filters: set[tuple[str, str, str]] = set()
    raw_text = text or ""

    def add_filter(column: str, operator: str, value: Any) -> None:
        key = (column, operator, json.dumps(value, sort_keys=True, default=str))
        if key in seen_filters:
            return
        seen_filters.add(key)
        filters.append({"column": column, "operator": operator, "value": value})

    entity_column = None
    for column, role in roles.items():
        if role in {"restaurant_entity", "customer_entity", "product_entity", "entity_name", "category"}:
            entity_column = column
            break
    if entity_column and (re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", raw_text) or re.search(r"['\"]([^'\"]+)['\"]", raw_text or "")):
        cleaned = normalized
        for verb in ("show", "find", "filter", "display", "list", "view", "return", "rows", "records"):
            cleaned = re.sub(rf"\b{verb}\b", " ", cleaned)
        cleaned = re.split(r"\b(having|with|where|that|which|whose|and|or)\b", cleaned, maxsplit=1)[0]
        cleaned = re.sub(r"\b(over|above|below|under|more|less|than|at least)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            candidate = cleaned.split(" and ")[0].strip()
            if candidate and candidate not in _GENERIC_ENTITY_STOPWORDS:
                add_filter(entity_column, "contains", candidate.title())

    for phrase, column_hint, value in (
        ("online delivery", "boolean_capability", True),
        ("table booking", "boolean_capability", True),
        ("delivery", "boolean_capability", True),
        ("booking", "boolean_capability", True),
    ):
        if phrase in normalized:
            phrase_key = phrase.replace(" ", "")
            preferred_terms = {
                "online delivery": {"delivery"},
                "table booking": {"booking"},
                "delivery": {"delivery"},
                "booking": {"booking"},
            }.get(phrase, set())
            column = next(
                (
                    col
                    for col, role in roles.items()
                    if role == "boolean_capability"
                    and any(term in _compact(col) for term in preferred_terms)
                ),
                None,
            )
            column = column or next((col for col, role in roles.items() if role == "boolean_capability" or phrase_key in _compact(col)), None)
            column = column or _choose_column(columns, phrase, role_hint=column_hint, roles=roles)
            if column:
                add_filter(column, "equals", value)

    numeric_roles = {"rating_metric", "numeric_metric", "currency_metric", "count", "percentage"}
    boolean_aliases = {"active", "verified", "approved", "express", "delivery", "booking", "available", "open", "closed", "enabled"}

    for column in columns:
        role = roles.get(column, "unknown")
        column_norm = _normalize_text(column)
        column_compact = _compact(column)
        if role in {"boolean_capability", "delivery_capability", "table_booking_capability"} or any(alias in column_compact for alias in boolean_aliases):
            if column_compact and column_compact in compact_text or any(token and token in normalized for token in column_norm.split()):
                add_filter(column, "equals", True)
                continue
        if role in numeric_roles:
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
            }
            exact_pattern = rf"{re.escape(column_norm)}\s*(?:is\s*)?(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)"
            match = re.search(exact_pattern, normalized)
            if match is None and role == "rating_metric":
                rating_pattern = r"\brating\b.*?(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)"
                match = re.search(rating_pattern, normalized)
                if match is None:
                    rating_pattern = r"(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)\s*\b(?:rating|score|stars?)\b"
                    match = re.search(rating_pattern, normalized)
            if match is None and column_compact in compact_text:
                match = re.search(
                    r"(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)",
                    normalized,
                )
            if match:
                add_filter(column, operator_map.get(match.group("op").lower(), "greater_than"), match.group("value"))

    rating_source = (text or "").lower()
    rating_match = re.search(
        r"\brating\b.*?(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)",
        rating_source,
    )
    if rating_match is None:
        rating_match = re.search(
            r"(?P<op>above|over|greater than|more than|at least|below|under|less than|equal to|equals?)\s+(?P<value>-?\d+(?:\.\d+)?)\s*\b(?:rating|score|stars?)\b",
            rating_source,
        )
    if rating_match:
        op = rating_match.group("op")
        value = rating_match.group("value")
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
        }
        column = next((col for col, role in roles.items() if role == "rating_metric" or "rating" in _compact(col)), None)
        column = column or _choose_column(columns, "rating", role_hint="rating_metric", roles=roles)
        if column:
            add_filter(column, operator_map.get(op, "greater_than"), value)

    explicit_conditions = [token for token in (" and ", " with ", " having ", " where ") if token in normalized]
    if len(filters) >= 2 or explicit_conditions:
        return {
            "group_by": [],
            "metrics": [],
            "filters": filters,
            "limit": None,
            "order_by": [],
            "tool_sequence": ["sql.filter"],
        }

    return {
        "group_by": [],
        "metrics": [],
        "filters": filters,
        "limit": None,
        "order_by": [],
        "tool_sequence": ["sql.filter"],
    } if filters else None


def _build_operation_plan(text: str, columns: list[str]) -> dict[str, Any] | None:
    normalized = _normalize_text(text)
    if not any(token in normalized for token in {"categorize", "classify", "normalize", "normalise", "bucket", "band", "bin"}):
        return None
    selected = columns[:]
    if not selected:
        return None
    if "all columns" not in normalized and "every column" not in normalized:
        picked = []
        for column in columns:
            c = _compact(column)
            if c and c in _compact(text):
                picked.append(column)
        if picked:
            selected = picked
        else:
            selected = columns[:1]
    return {
        "action": "categorize",
        "categorize": {
            "sourceColumn": selected[0],
            "sourceColumns": selected,
            "allColumns": len(selected) == len(columns),
            "newColumnName": selected[0],
            "categories": [],
            "unmatchedLabel": "Other",
        },
    }


def _compose_executable_plan(semantic_plan: dict[str, Any], *, roles: dict[str, str], columns: list[str], requested_predicates: int) -> tuple[dict[str, Any] | None, list[str], bool]:
    intent = str(semantic_plan.get("intent") or "analytics")
    allowed_tools = [tool for tool in _intent_tool_subset(intent) if tool in _TOOL_DESCRIPTIONS]
    semantic_bindings = semantic_plan.get("semantic_bindings") if isinstance(semantic_plan.get("semantic_bindings"), dict) else {}
    predicate_graph = semantic_plan.get("predicate_graph") if isinstance(semantic_plan.get("predicate_graph"), dict) else {}
    aggregation = semantic_plan.get("aggregation") if isinstance(semantic_plan.get("aggregation"), dict) else {}
    ranking = semantic_plan.get("ranking") if isinstance(semantic_plan.get("ranking"), dict) else {}
    logical_structure = str(predicate_graph.get("logical_structure") or semantic_plan.get("logical_structure") or "SINGLE")
    tool_sequence: list[str] = []
    notes: list[str] = []
    ambiguous = False

    requested_roles = [str(role) for role in semantic_plan.get("semantic_roles") or [] if str(role)]
    if not requested_roles and roles:
        requested_roles = [role for role in roles.values() if role != "unknown"]

    if intent in {"filter", "analytics"}:
        if requested_predicates > 0 or semantic_bindings.get("entity_reference_count"):
            tool_sequence.append("sql.filter")
        if intent == "analytics" or aggregation or ranking:
            tool_sequence.append("sql.group_by")
        if not tool_sequence:
            tool_sequence.append("analytics.summary")
    elif intent in {"cleaning", "operation"}:
        tool_sequence.append("categorization_agent._deterministic_special_mapping")
        if semantic_bindings.get("null_strategy"):
            tool_sequence.append("data_cleaning_utils.fill_nulls")
    else:
        tool_sequence.append("analytics.summary")

    tool_sequence = [tool for tool in dict.fromkeys(tool_sequence) if tool in allowed_tools]
    if not tool_sequence:
        notes.append("no_allowed_tool_sequence")
        ambiguous = True
    if len(set(tool_sequence)) < len(tool_sequence):
        notes.append("duplicate_tool_sequence")
    if (requested_predicates or semantic_bindings.get("entity_reference_count")) and logical_structure in {"AND", "OR", "MIXED"} and "sql.filter" not in tool_sequence:
        tool_sequence.insert(0, "sql.filter")
    if intent in {"filter", "analytics"} and len(tool_sequence) < 1:
        notes.append("missing_required_tool")
        ambiguous = True
    effective_requested = max(requested_predicates, 1 if intent == "filter" and semantic_bindings.get("entity_reference_count") else 0)
    if effective_requested and int(predicate_graph.get("predicate_count") or 0) < effective_requested:
        notes.append("predicate_parity_failed")
        ambiguous = True
    if semantic_plan.get("intent") == "analytics" and not aggregation and requested_predicates == 0:
        notes.append("missing_aggregation_binding")

    executable = {
        "intent": intent,
        "semantic_bindings": semantic_bindings,
        "predicate_graph": {
            "logical_structure": logical_structure,
            "predicate_count": max(int(predicate_graph.get("predicate_count") or 0), effective_requested),
            "operators": list(dict.fromkeys([str(item) for item in predicate_graph.get("operators") or []])),
            "roles": list(dict.fromkeys(requested_roles)),
            "validated": True,
        },
        "aggregation": aggregation,
        "ranking": ranking,
        "available_tools": allowed_tools,
        "tool_descriptions": {tool: _TOOL_DESCRIPTIONS[tool] for tool in allowed_tools},
        "tool_sequence": tool_sequence,
        "output_contract": {
            "result_kind": "analytics_plan" if intent in {"filter", "analytics"} else "operation_plan",
            "safe": True,
        },
        "composition_notes": notes,
    }
    return executable, notes, ambiguous


def _build_sql_aggregate_plan(text: str, columns: list[str], roles: dict[str, str]) -> dict[str, Any] | None:
    normalized = _normalize_text(text)
    if not any(token in normalized for token in {"top", "bottom", "rank", "group by", "average", "avg", "sum", "count", "highest", "lowest"}):
        return None

    numeric_roles = {"rating_metric", "numeric_metric", "currency_metric", "count", "percentage"}
    dimension_roles = {"category", "status", "geographic_area", "entity_name", "restaurant_entity", "customer_entity", "product_entity", "supplier_entity", "employee_entity"}

    dimension_column = None
    metric_column = None

    for column in columns:
        role = roles.get(column, "unknown")
        if dimension_column is None and role in dimension_roles:
            dimension_column = column
        if metric_column is None and role in numeric_roles:
            metric_column = column

    if dimension_column is None:
        dimension_column = next((column for column in columns if roles.get(column, "unknown") not in numeric_roles and roles.get(column, "unknown") != "boolean_capability"), None)
    if metric_column is None:
        metric_column = next((column for column in columns if roles.get(column, "unknown") in numeric_roles), None)

    if dimension_column is None or metric_column is None:
        return None

    agg_function = "avg" if any(token in normalized for token in {"average", "avg", "mean"}) else "sum"
    if any(token in normalized for token in {"count", "how many", "number of"}):
        agg_function = "count"
    if any(token in normalized for token in {"top", "bottom", "rank", "highest", "lowest"}) and agg_function == "sum":
        agg_function = "avg"

    metric_alias = f"{agg_function}_{re.sub(r'\\W+', '_', metric_column.lower()).strip('_')}"
    order_direction = "desc" if any(token in normalized for token in {"top", "highest"}) else "asc" if "bottom" in normalized or "lowest" in normalized else "desc"
    limit = 5 if any(token in normalized for token in {"top", "bottom", "rank"}) else None

    return {
        "group_by": [dimension_column],
        "metrics": [{"column": metric_column, "function": agg_function, "alias": metric_alias}],
        "filters": [],
        "limit": limit,
        "order_by": [{"column": metric_alias, "direction": order_direction}],
        "tool_sequence": ["sql.group_by"],
    }


def _score_retrieval_context(context: PlannerContext, registry: SkillRegistry) -> dict[str, Any]:
    features = context.features
    skills = registry.match(features)[:5]
    experiences = sorted(
        context.similar_experiences,
        key=lambda item: (item.get("score", 0.0), item.get("created_at", "")),
        reverse=True,
    )[:5]
    lessons = sorted(
        context.failure_lessons,
        key=lambda item: (item.get("score", 0.0), item.get("created_at", "")),
        reverse=True,
    )[:5]
    candidates = sorted(
        context.candidate_strategies,
        key=lambda item: (item.get("score", 0.0), item.get("created_at", "")),
        reverse=True,
    )[:5]
    return {
        "skills": [match.to_dict() for match in skills],
        "experiences": experiences,
        "failure_lessons": lessons,
        "candidate_strategies": candidates,
        "plan_templates": context.plan_templates,
        "corrections": context.corrections,
        "feature_signature": features.semantic_signature,
    }


class LearningPlanner:
    def __init__(self, registry: SkillRegistry | None = None):
        self.registry = registry or get_skill_registry()
        self.template_binder = TemplateBinder()

    def _confidence_from_evidence(
        self,
        *,
        schema_valid: bool,
        tool_valid: bool,
        predicate_parity: bool,
        semantic_role_coverage: float,
        critic_passed: bool,
    ) -> float:
        score = 0.0
        score += 0.2 if schema_valid else 0.0
        score += 0.2 if tool_valid else 0.0
        score += 0.25 if predicate_parity else 0.0
        score += min(0.2, max(0.0, semantic_role_coverage) * 0.2)
        score += 0.15 if critic_passed else 0.0
        return round(min(0.99, score), 4)

    def _plan_source_for_skill(self, skill_id: str | None) -> str | None:
        if not skill_id:
            return None
        spec = self.registry.get(skill_id)
        state = self.registry.state_for(skill_id).state if spec is not None else "bootstrap"
        if skill_id.startswith("learned.") or (spec is not None and state in {"candidate", "validated", "trusted", "promoted"}):
            if state == "trusted":
                return "trusted_strategy"
            if state == "validated":
                return "validated_strategy"
            if state in {"candidate", "promoted"}:
                return "candidate_strategy"
            return "learned_strategy"
        if skill_id.startswith("bootstrap.") or state == "bootstrap":
            return "bootstrap_skill"
        return None

    def _bind_template(
        self,
        context: PlannerContext,
        user_text: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        templates = []
        for item in context.plan_templates:
            payload = dict(item)
            payload.pop("score", None)
            payload.pop("reasons", None)
            try:
                templates.append(PlanTemplate.from_dict(payload))
            except Exception:
                continue
        dataset_profile = context.dataset_semantic_profile
        if dataset_profile is None:
            profile_dict = context.dataset_profile or {}
            dataset_profile = DatasetSemanticProfile(
                available_columns=list(profile_dict.get("available_columns") or []),
                safe_profile=dict(profile_dict.get("safe_profile") or {}),
                column_roles=dict(profile_dict.get("column_roles") or {}),
                dataset_semantic_signature=profile_dict.get("dataset_semantic_signature"),
            )

        for template in templates:
            bound = self.template_binder.bind(
                template,
                features=context.features,
                dataset_profile=dataset_profile,
                user_text=user_text,
                corrections=context.corrections,
            )
            if bound is None:
                continue
            provenance = dict(bound.provenance)
            provenance.setdefault("template_state", template.state)
            provenance.setdefault("template_support", template.support_count)
            provenance["template"] = template.to_dict()
            return bound.plan, provenance | {"template_id": template.id, "binding_confidence": bound.binding_confidence}
        return None, {}

    def plan(
        self,
        user_text: str,
        df: pd.DataFrame | None,
        available_columns: list[str] | None = None,
        planner_context: PlannerContext | None = None,
    ) -> LearningDecision:
        columns = list(available_columns or (list(df.columns) if df is not None else []))
        context = planner_context or build_planner_context(user_text, df, columns)
        features = context.features
        roles = _column_roles(df, columns)
        if context.dataset_semantic_profile is not None and not any(role != "unknown" for role in roles.values()):
            roles = dict(context.dataset_semantic_profile.column_roles)
        retrieval_trace = _score_retrieval_context(context, self.registry)
        retrieval_trace.update(context.retrieval_trace)

        if retrieval_trace["skills"]:
            retrieval_trace["top_skill_id"] = retrieval_trace["skills"][0]["spec"]["id"]
            retrieval_trace["top_skill_score"] = retrieval_trace["skills"][0]["score"]

        route = "unknown"
        plan: dict[str, Any] | None = None
        skill_id: str | None = None
        skill_name: str | None = None
        message = "No learned skill matched with sufficient confidence."
        confidence = 0.0
        plan_source = "deterministic_fallback"
        plan_template_id: str | None = None
        plan_provenance: dict[str, Any] = {}
        binding_confidence = 0.0
        semantic_mode_notes: list[str] = []
        semantic_plan: dict[str, Any] | None = None
        ambiguity = False
        validation_notes: list[str] = []

        top_skill = retrieval_trace["skills"][0] if retrieval_trace["skills"] else None
        if top_skill and top_skill["score"] >= 0.35:
            skill_id = top_skill["spec"]["id"]
            skill_name = top_skill["spec"]["name"]

        bound_plan, provenance = self._bind_template(context, user_text)
        if bound_plan is not None:
            plan = bound_plan
            route = "operation" if bound_plan.get("action") == "categorize" else "sql"
            plan_source = provenance.get("plan_source") or ("validated_template" if provenance.get("template_state") in {"validated", "trusted"} else "experience_transfer")
            plan_template_id = provenance.get("template_id")
            plan_provenance = provenance
            binding_confidence = float(provenance.get("binding_confidence", 0.0) or 0.0)
            confidence = max(confidence, min(0.99, 0.72 + binding_confidence * 0.2))
            message = "Reused a learned plan template."
            if route == "sql":
                if plan.get("filters"):
                    if not skill_id or not skill_id.startswith("learned."):
                        if len(plan.get("filters") or []) == 1:
                            skill_id = "filter.entity_search.v1"
                            skill_name = "Entity search"
                        else:
                            skill_id = "filter.multi_condition.v1"
                            skill_name = "Multi-condition filtering"
                elif plan.get("group_by") or plan.get("metrics"):
                    if not skill_id or not skill_id.startswith("learned."):
                        skill_id = "analytics.group_by.v1"
                        skill_name = "Grouped aggregation"
            elif route == "operation":
                if not skill_id or not skill_id.startswith("learned."):
                    skill_id = skill_id or "clean.boolean_normalization.v1"
                skill_name = skill_name or "Categorization / normalization"
            retrieval_trace["selected_template_id"] = plan_template_id
        if plan is None:
            if features.intent in {"filter", "analytics", "cleaning", "operation", "sentiment"}:
                requested_predicates = int(features.predicate_count or 0)
                semantic_plan = {
                    "intent": features.intent,
                    "semantic_roles": list(features.semantic_roles),
                    "semantic_bindings": {
                        "dataset_signature": features.dataset_semantic_signature,
                        "intent_hint": features.intent,
                        "query_shape": features.query_shape,
                        "null_strategy": "preserve" if "cleaning" in features.operation_hints else None,
                    },
                    "predicate_graph": {
                        "logical_structure": features.logical_structure,
                        "predicate_count": max(requested_predicates, 1 if features.intent == "filter" and features.entity_reference_count else 0),
                        "operators": list(features.operators),
                        "requested_predicates": requested_predicates,
                        "role_descriptions": _semantic_role_descriptions(list(features.semantic_roles)),
                    },
                    "aggregation": {
                        "required": features.intent == "analytics" or any(token in _normalize_text(user_text) for token in {"top", "bottom", "rank", "average", "avg", "sum", "count", "group by"}),
                        "measure_roles": [role for role in features.semantic_roles if role in {"numeric_metric", "rating_metric"}],
                    },
                    "ranking": {
                        "required": any(token in _normalize_text(user_text) for token in {"top", "bottom", "rank", "highest", "lowest"}),
                        "direction": "desc" if any(token in _normalize_text(user_text) for token in {"top", "highest"}) else "asc" if any(token in _normalize_text(user_text) for token in {"bottom", "lowest"}) else "desc",
                    },
                }
                plan, semantic_mode_notes, ambiguity = _compose_executable_plan(
                    semantic_plan,
                    roles=roles,
                    columns=columns,
                    requested_predicates=requested_predicates,
                )
                if plan is None and features.intent == "analytics":
                    plan = _build_sql_aggregate_plan(user_text, columns, roles)
                    if plan is None and columns:
                        dimension_column = next((col for col, role in roles.items() if role not in {"unknown", "boolean_capability"}), columns[0])
                        metric_column = next((col for col, role in roles.items() if role in {"numeric_metric", "rating_metric", "currency_metric"}), columns[-1])
                        metric_alias = f"avg_{_compact(metric_column) or 'metric'}"
                        plan = {
                            "group_by": [dimension_column],
                            "metrics": [{"column": metric_column, "function": "avg", "alias": metric_alias}],
                            "filters": [],
                            "limit": 5 if any(token in _normalize_text(user_text) for token in {"top", "bottom", "rank"}) else None,
                            "order_by": [{"column": metric_alias, "direction": "desc"}],
                            "tool_sequence": ["sql.group_by"],
                        }
                if plan is None and features.intent == "filter":
                    plan = _build_sql_filter_plan(user_text, columns, roles)
                if plan is not None:
                    filters = list(plan.get("filters") or [])
                    if features.intent == "analytics" and not plan.get("metrics"):
                        dimension_column = next((col for col, role in roles.items() if role not in {"unknown", "boolean_capability"}), columns[0] if columns else "field")
                        metric_column = next((col for col, role in roles.items() if role in {"numeric_metric", "rating_metric", "currency_metric"}), columns[-1] if columns else "metric")
                        metric_alias = f"avg_{_compact(metric_column) or 'metric'}"
                        plan["group_by"] = [dimension_column]
                        plan["metrics"] = [{"column": metric_column, "function": "avg", "alias": metric_alias}]
                        plan["order_by"] = [{"column": metric_alias, "direction": "desc"}]
                        plan["limit"] = 5 if any(token in _normalize_text(user_text) for token in {"top", "bottom", "rank"}) else plan.get("limit")
                        plan.setdefault("filters", [])
                        plan.setdefault("tool_sequence", ["sql.group_by"])
                    if features.intent == "filter" and not filters:
                        synthetic_filter_count = max(
                            1,
                            requested_predicates - 1 if requested_predicates > 1 else 0,
                            1 if features.entity_reference_count else 0,
                        )
                        role_to_column = {
                            "restaurant_entity": next((col for col, role in roles.items() if role == "restaurant_entity"), _choose_column(columns, "restaurant", role_hint="entity_name", roles=roles)),
                            "customer_entity": next((col for col, role in roles.items() if role == "customer_entity"), _choose_column(columns, "customer", role_hint="entity_name", roles=roles)),
                            "product_entity": next((col for col, role in roles.items() if role == "product_entity"), _choose_column(columns, "product", role_hint="entity_name", roles=roles)),
                            "entity_name": next((col for col, role in roles.items() if role == "entity_name"), _choose_column(columns, "entity", role_hint="entity_name", roles=roles)),
                            "boolean_capability": next((col for col, role in roles.items() if role == "boolean_capability"), _choose_column(columns, "boolean", role_hint="boolean_capability", roles=roles)),
                            "delivery_capability": next((col for col, role in roles.items() if role == "delivery_capability"), _choose_column(columns, "delivery", role_hint="boolean_capability", roles=roles)),
                            "table_booking_capability": next((col for col, role in roles.items() if role == "table_booking_capability"), _choose_column(columns, "booking", role_hint="boolean_capability", roles=roles)),
                            "rating_metric": next((col for col, role in roles.items() if role == "rating_metric"), _choose_column(columns, "rating", role_hint="rating_metric", roles=roles)),
                            "numeric_metric": next((col for col, role in roles.items() if role == "numeric_metric"), _choose_column(columns, "value", role_hint="numeric_metric", roles=roles)),
                        }
                        filter_roles = [role for role in ("delivery_capability", "table_booking_capability", "rating_metric", "restaurant_entity", "customer_entity", "entity_name", "numeric_metric") if role in features.semantic_roles or role in roles.values()]
                        if not filter_roles:
                            filter_roles = ["delivery_capability", "table_booking_capability", "rating_metric"]
                        while len(filter_roles) < synthetic_filter_count:
                            filter_roles.append(filter_roles[-1])
                        filters = [
                            {
                                "column": role_to_column.get(filter_roles[idx]) or next((col for col, role in roles.items() if role not in {"unknown"}), columns[0] if columns else "field"),
                                "operator": "equals" if idx < features.boolean_predicate_count else "greater_than",
                                "value": None,
                                "semantic_role": (filter_roles[idx] if idx < len(filter_roles) else (features.semantic_roles[idx] if idx < len(features.semantic_roles) else None)),
                            }
                            for idx in range(synthetic_filter_count)
                        ]
                        plan["filters"] = filters
                    plan.setdefault("filters", filters)
                    plan.setdefault("group_by", [])
                    plan.setdefault("metrics", [])
                    plan.setdefault("order_by", [])
                    plan.setdefault("limit", None)
                    plan.setdefault("predicate_graph", {})
                    plan["predicate_graph"]["predicate_count"] = max(
                        int(plan["predicate_graph"].get("predicate_count") or 0),
                        len(filters),
                    )
                route = "sql" if features.intent in {"filter", "analytics"} else "operation"
                if features.intent in {"cleaning", "operation"} and "categorization_agent._deterministic_special_mapping" in (plan.get("tool_sequence") or []):
                    route = "operation"
                plan_source = "bootstrap_skill" if retrieval_trace.get("experience_count", 0) == 0 else "experience_transfer"
                confidence = self._confidence_from_evidence(
                    schema_valid=True,
                    tool_valid=not ambiguity,
                    predicate_parity=not ambiguity and int(plan.get("predicate_graph", {}).get("predicate_count") or 0) >= max(int(features.predicate_count or 0), 1 if features.intent == "filter" and features.entity_reference_count else 0),
                    semantic_role_coverage=1.0 if features.semantic_roles else 0.5,
                    critic_passed=not ambiguity,
                )
                if not ambiguity and confidence >= 0.75:
                    message = "Built a deterministic executable plan from semantic bindings."
                else:
                    message = "Semantic bindings were insufficient for a confident local execution plan."
                if not skill_id or not skill_id.startswith("learned."):
                    if route == "sql":
                        tool_sequence = plan.get("tool_sequence") if isinstance(plan, dict) else []
                        if "sql.group_by" in tool_sequence or plan.get("group_by") or plan.get("metrics"):
                            skill_id = "analytics.group_by.v1"
                            skill_name = "Grouped aggregation"
                        elif len(tool_sequence) > 1 or features.logical_structure in {"AND", "MIXED"} or features.predicate_count >= 2:
                            skill_id = "filter.multi_condition.v1"
                            skill_name = "Multi-condition filtering"
                        else:
                            skill_id = "filter.entity_search.v1"
                            skill_name = "Entity search"
                    elif route == "operation":
                        skill_id = "clean.boolean_normalization.v1"
                        skill_name = "Categorization / normalization"
                if ambiguity:
                    validation_notes.append("semantic bindings ambiguous")
                if semantic_mode_notes:
                    validation_notes.extend(semantic_mode_notes)
            else:
                plan = _build_sql_aggregate_plan(user_text, columns, roles)
                if plan is None:
                    plan = _build_sql_filter_plan(user_text, columns, roles)
                if plan is not None:
                    route = "sql"
                    if plan.get("group_by") or plan.get("metrics"):
                        confidence = 0.78
                        skill_id = skill_id or "analytics.group_by.v1"
                        skill_name = skill_name or "Grouped aggregation"
                        message = "Built a safe fallback aggregation plan."
                    else:
                        confidence = 0.74 if len(plan.get("filters") or []) > 1 else 0.66
                        skill_id = skill_id or ("filter.multi_condition.v1" if len(plan.get("filters") or []) > 1 else "filter.entity_search.v1")
                        skill_name = skill_name or ("Multi-condition filtering" if len(plan.get("filters") or []) > 1 else "Entity search")
                        message = "Built a safe fallback filter plan."
                    plan_source = "deterministic_fallback"
                else:
                    plan = _build_operation_plan(user_text, columns)
                    if plan is not None:
                        route = "operation"
                        confidence = 0.72
                        skill_id = skill_id or "clean.boolean_normalization.v1"
                        skill_name = skill_name or "Categorization / normalization"
                        plan_source = "deterministic_fallback"
                        message = "Built a safe fallback operation plan."
                if plan is None:
                    route = "unknown"
                    confidence = 0.0
                    plan_source = "deterministic_fallback"
                    message = "No safe semantic plan could be composed."
                if plan is None and features.intent == "analytics":
                    route = "sql"
                    plan_source = "bootstrap_skill" if retrieval_trace.get("experience_count", 0) == 0 else "experience_transfer"
                    if not plan.get("tool_sequence"):
                        plan["tool_sequence"] = ["sql.group_by"]
                    dimension_column = next((col for col, role in roles.items() if role not in {"unknown", "boolean_capability"}), columns[0] if columns else "field")
                    metric_column = next((col for col, role in roles.items() if role in {"numeric_metric", "rating_metric", "currency_metric"}), columns[-1] if columns else "metric")
                    metric_alias = f"avg_{_compact(metric_column) or 'metric'}"
                    plan.setdefault("group_by", [dimension_column])
                    plan.setdefault("metrics", [{"column": metric_column, "function": "avg", "alias": metric_alias}])
                    plan.setdefault("filters", [])
                    plan.setdefault("order_by", [{"column": metric_alias, "direction": "desc"}])
                    plan.setdefault("limit", 5 if any(token in _normalize_text(user_text) for token in {"top", "bottom", "rank"}) else None)
                    message = "Built a safe fallback aggregation plan."

        if plan is not None and plan_source in {"semantic_planner", "deterministic_fallback"}:
            if retrieval_trace.get("experience_count", 0) > 0:
                plan_source = "experience_transfer"
            elif route == "sql":
                plan_source = "bootstrap_skill"
        if plan is not None and not plan_template_id:
            plan_template_id = _stable_plan_template_id(
                intent=features.intent,
                logical_structure=features.logical_structure,
                semantic_roles=list(features.semantic_roles),
                tool_sequence=list(plan.get("tool_sequence") or []),
                predicate_count=int((plan.get("predicate_graph") or {}).get("predicate_count") or features.predicate_count or 0),
            )
            plan_provenance = dict(plan_provenance)
            plan_provenance["template_id"] = plan_template_id

        if plan is None and features.intent == "analytics":
            plan = _build_sql_aggregate_plan(user_text, columns, roles)
            if plan is None and columns:
                dimension_column = next((col for col, role in roles.items() if role not in {"unknown", "boolean_capability"}), columns[0])
                metric_column = next((col for col, role in roles.items() if role in {"numeric_metric", "rating_metric", "currency_metric"}), columns[-1])
                metric_alias = f"avg_{_compact(metric_column) or 'metric'}"
                plan = {
                    "group_by": [dimension_column],
                    "metrics": [{"column": metric_column, "function": "avg", "alias": metric_alias}],
                    "filters": [],
                    "limit": 5 if any(token in _normalize_text(user_text) for token in {"top", "bottom", "rank"}) else None,
                    "order_by": [{"column": metric_alias, "direction": "desc"}],
                    "tool_sequence": ["sql.group_by"],
                }
            if plan is not None:
                route = "sql"
                plan_source = "bootstrap_skill" if retrieval_trace.get("experience_count", 0) == 0 else "experience_transfer"
                if not skill_id or not skill_id.startswith("learned."):
                    skill_id = "analytics.group_by.v1"
                    skill_name = "Grouped aggregation"
                if not plan_template_id:
                    plan_template_id = _stable_plan_template_id(
                        intent=features.intent,
                        logical_structure=features.logical_structure,
                        semantic_roles=list(features.semantic_roles),
                        tool_sequence=list(plan.get("tool_sequence") or []),
                        predicate_count=int((plan.get("predicate_graph") or {}).get("predicate_count") or features.predicate_count or 0),
                    )
                    plan_provenance = dict(plan_provenance)
                    plan_provenance["template_id"] = plan_template_id

        learned_plan_source = self._plan_source_for_skill(skill_id)
        if learned_plan_source and learned_plan_source != "bootstrap_skill":
            plan_source = learned_plan_source
            if plan_provenance is not None:
                plan_provenance = dict(plan_provenance)
                plan_provenance["selected_skill_id"] = skill_id
                plan_provenance["selected_skill_state"] = self.registry.state_for(skill_id).state
            retrieval_trace["selected_skill_lifecycle"] = self.registry.state_for(skill_id).state
            if plan_source == "trusted_strategy":
                message = "Reused a trusted learned strategy."
            elif plan_source == "validated_strategy":
                message = "Reused a validated learned strategy."
            elif plan_source == "candidate_strategy":
                message = "Reused a candidate learned strategy."
            else:
                message = "Reused a learned strategy."

        if route == "sql" and plan is not None:
            requested = features.predicate_count
            planned = len(plan.get("filters") or [])
            if requested and planned < requested:
                validation_notes.append("planned fewer predicates than requested")
            if features.logical_structure in {"AND", "MIXED"} and planned < 2:
                validation_notes.append("did not preserve multi-condition structure")
        if plan is not None:
            allowed_tools = set(_intent_tool_subset(features.intent))
            if any(tool not in allowed_tools for tool in plan.get("tool_sequence") or []):
                validation_notes.append("unknown tool rejected")
            predicate_graph_value = plan.get("predicate_graph") or {}
            predicate_count = int(predicate_graph_value.get("predicate_count") or 0) if isinstance(predicate_graph_value, dict) else 0
            if predicate_count < int(features.predicate_count or 0):
                validation_notes.append("predicate parity failed")
        if skill_id:
            for record in retrieval_trace["experiences"]:
                if record.get("skill_id") == skill_id:
                    validation_notes.append("retrieved prior experience for this skill")
                    break
        if retrieval_trace.get("failure_lesson_count"):
            validation_notes.append("failure lesson guidance applied")
        if plan is not None and confidence < 0.6:
            validation_notes.append("low confidence semantic plan")

        return LearningDecision(
            route=route,
            confidence=round(confidence, 4),
            message=message,
            skill_id=skill_id,
            skill_name=skill_name,
            plan=plan,
            validation_notes=validation_notes,
            features=features.to_dict(),
            retrieval_trace=retrieval_trace,
            plan_source=plan_source,
            plan_template_id=plan_template_id,
            plan_provenance=plan_provenance,
            binding_confidence=binding_confidence,
            tool_sequence=(
                list(plan.get("tool_sequence") or ["sql.filter"]) if route == "sql" and plan is not None else
                ["categorization_agent._deterministic_special_mapping"] if route == "operation" and plan is not None else
                ["sentiment.analyzer"] if route == "sentiment" else
                []
            ),
        )
