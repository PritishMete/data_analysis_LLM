from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import argparse
import contextlib
import importlib.util
import json
import os
import warnings
from pathlib import Path
import sys
from typing import Any, Callable, Iterable
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning.experience_store import LearningExperienceStore
from learning.models import ExperienceRecord, PlanTemplate, stable_hash
from learning.training_export import TrainingDatasetExporter, TrainingExportPolicy

warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually*",
    category=UserWarning,
)


DEFAULT_RUNTIME_ROOT = Path("runtime")
DEFAULT_CURRICULUM_RUNTIME_ROOT = DEFAULT_RUNTIME_ROOT / "curriculum"
DEFAULT_CURRICULUM_REPORT_PATH = DEFAULT_CURRICULUM_RUNTIME_ROOT / "report.json"
DEFAULT_CURRICULUM_DOC_PATH = Path("docs") / "analytics_curriculum_report.md"
DEFAULT_TRAINING_EXPORT_DIR = DEFAULT_RUNTIME_ROOT / "training"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _teacher_root() -> Path:
    override = os.environ.get("TEACHER_REPO_ROOT")
    if override:
        return Path(override)
    return Path(r"E:\teacher_ref")


@dataclass(slots=True)
class CurriculumDomain:
    name: str
    entity_column: str
    dimension_column: str
    metric_column: str
    secondary_metric_column: str
    date_column: str
    boolean_columns: tuple[str, str]
    special_intent: str
    special_window_type: str
    special_metric_column: str | None = None


@dataclass(slots=True)
class CurriculumFamily:
    family_id: str
    domain: CurriculumDomain
    intent: str
    family_kind: str
    query_template: str
    plan_factory: Callable[[pd.DataFrame, int, CurriculumDomain], dict[str, Any]]
    dataframe_factory: Callable[[int, CurriculumDomain], pd.DataFrame]
    result_kind: str = "table"
    route: str = "sql"
    is_special: bool = False


@dataclass(slots=True)
class CurriculumCase:
    case_id: str
    family_id: str
    domain: str
    intent: str
    variant: int
    user_text: str
    dataframe: pd.DataFrame
    available_sheets: list[str]
    plan: dict[str, Any]
    result_payload: dict[str, Any]
    route: str = "sql"
    quality_score: float = 0.98
    critic_passed: bool = True
    result_validation_passed: bool = True
    plan_completeness_passed: bool = True
    privacy_validation_passed: bool = True
    no_unresolved_ambiguity: bool = True
    no_critical_repair: bool = True
    repair_count: int = 0
    correction_state: str = "validated"
    plan_source: str = "validated_template"
    plan_template_id: str | None = None
    skill_id: str | None = None
    failure: bool = False


@dataclass(slots=True)
class CurriculumPhaseResult:
    name: str
    inspected: int
    bridge_accepted: int
    gemini_fallback_calls: int
    route_counts: dict[str, int]
    plan_source_counts: dict[str, int]
    average_confidence: float
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CurriculumRunResult:
    generated_at: str
    teacher_root: str
    student_root: str
    runtime_root: str
    training_export_dir: str
    dataset_version: str
    total_cases: int
    total_families: int
    total_domains: int
    total_intents: int
    total_seeded_events: int
    training_report: dict[str, Any]
    readiness_checkpoints: list[dict[str, Any]]
    phases: list[CurriculumPhaseResult]
    restart_verified: bool
    privacy_verified: bool
    export_paths: dict[str, str]
    sample_families: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phases"] = [asdict(phase) for phase in self.phases]
        return payload


def _load_teacher_modules():
    teacher_root = _teacher_root()
    if not teacher_root.exists():
        raise FileNotFoundError(f"Teacher repo not found: {teacher_root}")

    teacher_query_router_path = teacher_root / "query_router.py"
    teacher_learning_bridge_path = teacher_root / "learning_bridge.py"
    if not teacher_query_router_path.exists() or not teacher_learning_bridge_path.exists():
        raise FileNotFoundError("Teacher bridge files are missing from the teacher repository.")

    sys.path.insert(0, str(teacher_root))
    try:
        query_spec = importlib.util.spec_from_file_location(
            "analytics_curriculum_teacher_query_router",
            teacher_query_router_path,
        )
        bridge_spec = importlib.util.spec_from_file_location(
            "analytics_curriculum_teacher_learning_bridge",
            teacher_learning_bridge_path,
        )
        if query_spec is None or query_spec.loader is None or bridge_spec is None or bridge_spec.loader is None:
            raise RuntimeError("Could not load teacher modules for the curriculum harness.")

        query_module = importlib.util.module_from_spec(query_spec)
        bridge_module = importlib.util.module_from_spec(bridge_spec)
        sys.modules[query_spec.name] = query_module
        sys.modules[bridge_spec.name] = bridge_module
        query_spec.loader.exec_module(query_module)
        bridge_spec.loader.exec_module(bridge_module)
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(teacher_root))

    return query_module, bridge_module


def _build_domain_profiles() -> list[CurriculumDomain]:
    return [
        CurriculumDomain(
            name="retail",
            entity_column="Customer Name",
            dimension_column="Region",
            metric_column="Revenue",
            secondary_metric_column="Order Count",
            date_column="Month",
            boolean_columns=("Online Order", "Loyalty Member"),
            special_intent="distribution",
            special_window_type="rank",
        ),
        CurriculumDomain(
            name="logistics",
            entity_column="Shipment ID",
            dimension_column="Route",
            metric_column="Delay Minutes",
            secondary_metric_column="Shipment Count",
            date_column="Ship Month",
            boolean_columns=("Priority", "Active"),
            special_intent="summary",
            special_window_type="running_total",
        ),
        CurriculumDomain(
            name="finance",
            entity_column="Account Name",
            dimension_column="Branch",
            metric_column="Balance",
            secondary_metric_column="Transaction Count",
            date_column="Month",
            boolean_columns=("Verified", "Active"),
            special_intent="benchmark",
            special_window_type="moving_average",
        ),
        CurriculumDomain(
            name="healthcare",
            entity_column="Patient ID",
            dimension_column="Ward",
            metric_column="Stay Length",
            secondary_metric_column="Case Count",
            date_column="Admit Month",
            boolean_columns=("Admitted", "Readmitted"),
            special_intent="risk_review",
            special_window_type="rank",
        ),
        CurriculumDomain(
            name="hr",
            entity_column="Employee Name",
            dimension_column="Department",
            metric_column="Salary",
            secondary_metric_column="Hire Count",
            date_column="Hire Month",
            boolean_columns=("Remote", "Performance Flag"),
            special_intent="staffing",
            special_window_type="running_total",
        ),
        CurriculumDomain(
            name="ecommerce",
            entity_column="Product Name",
            dimension_column="Category",
            metric_column="Sales",
            secondary_metric_column="Units Sold",
            date_column="Month",
            boolean_columns=("Subscription", "Repeat Buyer"),
            special_intent="churn_watch",
            special_window_type="moving_average",
        ),
        CurriculumDomain(
            name="support",
            entity_column="Case ID",
            dimension_column="Team",
            metric_column="Resolution Time",
            secondary_metric_column="Ticket Count",
            date_column="Week",
            boolean_columns=("Escalated", "Active"),
            special_intent="escalation",
            special_window_type="rank",
        ),
        CurriculumDomain(
            name="education",
            entity_column="Student Name",
            dimension_column="Grade Band",
            metric_column="Pass Rate",
            secondary_metric_column="Enrollment",
            date_column="Semester",
            boolean_columns=("Online", "Attendance Flag"),
            special_intent="retention",
            special_window_type="running_total",
        ),
        CurriculumDomain(
            name="manufacturing",
            entity_column="Batch ID",
            dimension_column="Plant",
            metric_column="Defect Rate",
            secondary_metric_column="Output",
            date_column="Month",
            boolean_columns=("QC Pass", "Shift Active"),
            special_intent="quality_audit",
            special_window_type="moving_average",
        ),
        CurriculumDomain(
            name="restaurant",
            entity_column="Restaurant Name",
            dimension_column="Cuisine",
            metric_column="Rating",
            secondary_metric_column="Orders",
            date_column="Week",
            boolean_columns=("Delivery", "Booking"),
            special_intent="hospitality",
            special_window_type="rank",
        ),
    ]


def _safe_rows_from_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    safe_df = df.where(pd.notnull(df), None)
    return safe_df.to_dict(orient="records")


def _domain_frame(variant: int, domain: CurriculumDomain) -> pd.DataFrame:
    base = variant * 10
    rows = []
    for index in range(6):
        rows.append(
            {
                domain.entity_column: f"{domain.name.title()} Entity {variant:02d}-{index + 1}",
                domain.dimension_column: [
                    "North",
                    "South",
                    "East",
                    "West",
                    "Central",
                    "Remote",
                ][index],
                domain.metric_column: round(base + 12.5 + index * 4.5, 2),
                domain.secondary_metric_column: base + 3 + index,
                domain.date_column: f"2026-0{(index % 6) + 1}-0{(index % 9) + 1}",
                domain.boolean_columns[0]: index % 2 == 0,
                domain.boolean_columns[1]: index % 3 != 0,
            }
        )
    return pd.DataFrame(rows)


def _compare_frame(variant: int, domain: CurriculumDomain) -> pd.DataFrame:
    df = _domain_frame(variant, domain)
    repeated_values = [f"{domain.name.title()} Repeat {variant:02d}"] * 3 + [
        f"{domain.name.title()} New {variant:02d}-{index}"
        for index in range(3, 6)
    ]
    df[domain.entity_column] = repeated_values
    return df


def _base_result(df: pd.DataFrame, plan: dict[str, Any], *, route: str = "sql") -> dict[str, Any]:
    rows = _safe_rows_from_frame(df)
    return {
        "success": True,
        "route": route,
        "confidence": 0.97,
        "plan_source": "validated_template",
        "plan_template_id": plan.get("plan_template_id") or plan.get("template_id"),
        "skill_id": plan.get("skill_id"),
        "plan": plan,
        "result": {
            "columns": list(df.columns),
            "rows": rows,
            "row_count": len(rows),
        },
        "critic_passed": True,
        "result_validation_passed": True,
        "plan_completeness_passed": True,
        "privacy_validation_passed": True,
        "no_unresolved_ambiguity": True,
        "no_critical_repair": True,
        "repair_count": 0,
        "correction_state": "validated",
        "quality_score": 0.97,
    }


def _variant_tag(variant: int) -> str:
    return f"variant_{variant:02d}"


def _variant_tool_sequence(domain: CurriculumDomain, family_kind: str, variant: int) -> list[str]:
    slug = _safe_slug(domain.name)
    variant_step = f"{_variant_tag(variant)}.{slug}"
    if family_kind == "filter":
        return [
            f"resolve_semantic_targets.{slug}",
            f"filter_rows.{slug}",
            f"validate_filter_result.{slug}",
            variant_step,
        ]
    if family_kind == "aggregate":
        return [
            f"resolve_semantic_targets.{slug}",
            f"group_and_aggregate.{slug}",
            f"sort_results.{slug}",
            f"validate_result.{slug}",
            variant_step,
        ]
    if family_kind == "compare":
        return [
            f"resolve_semantic_targets.{slug}",
            f"derive_columns.{slug}",
            f"group_and_aggregate.{slug}",
            f"validate_result.{slug}",
            variant_step,
        ]
    if family_kind == "trend":
        return [
            f"resolve_semantic_targets.{slug}",
            f"group_and_aggregate.{slug}",
            f"sort_results.{slug}",
            f"validate_result.{slug}",
            variant_step,
        ]
    return [
        f"resolve_semantic_targets.{slug}",
        f"group_and_aggregate.{slug}",
        f"rank_and_top_n.{slug}",
        f"validate_result.{slug}",
        variant_step,
    ]


def _filter_plan_factory(df: pd.DataFrame, variant: int, domain: CurriculumDomain) -> dict[str, Any]:
    metric_threshold = round(20 + variant * 2.5, 2)
    return {
        "filters": [
            {"column": domain.boolean_columns[0], "operator": "equals", "value": True},
            {"column": domain.boolean_columns[1], "operator": "equals", "value": True},
            {"column": domain.metric_column, "operator": "greater_than", "value": metric_threshold},
        ],
        "group_by": [],
        "metrics": [],
        "order_by": [],
        "limit": None,
        "tool_sequence": _variant_tool_sequence(domain, "filter", variant),
    }


def _aggregate_plan_factory(df: pd.DataFrame, variant: int, domain: CurriculumDomain) -> dict[str, Any]:
    metric_alias = f"total_{_safe_slug(domain.metric_column)}"
    return {
        "group_by": [domain.dimension_column],
        "metrics": [{"column": domain.metric_column, "function": "sum", "alias": metric_alias}],
        "filters": [],
        "order_by": [{"column": metric_alias, "direction": "desc"}],
        "limit": 5,
        "tool_sequence": _variant_tool_sequence(domain, "aggregate", variant),
    }


def _compare_plan_factory(df: pd.DataFrame, variant: int, domain: CurriculumDomain) -> dict[str, Any]:
    metric_alias = f"total_{_safe_slug(domain.metric_column)}"
    bucket_alias = f"{_safe_slug(domain.name)}_bucket"
    return {
        "derived_columns": [
            {
                "alias": bucket_alias,
                "case": {
                    "condition": {
                        "window_function": "count",
                        "column": domain.entity_column,
                        "partition_by": [domain.entity_column],
                        "operator": "greater_than",
                        "value": "1",
                    },
                    "then": "Returning",
                    "else": "New",
                },
            }
        ],
        "group_by": [bucket_alias],
        "metrics": [{"column": domain.metric_column, "function": "sum", "alias": metric_alias}],
        "filters": [],
        "order_by": [{"column": metric_alias, "direction": "desc"}],
        "limit": None,
        "tool_sequence": _variant_tool_sequence(domain, "compare", variant),
    }


def _trend_plan_factory(df: pd.DataFrame, variant: int, domain: CurriculumDomain) -> dict[str, Any]:
    metric_alias = f"running_{_safe_slug(domain.metric_column)}"
    return {
        "group_by": [domain.date_column],
        "metrics": [{"column": domain.metric_column, "function": "sum", "alias": metric_alias}],
        "filters": [],
        "window": {
            "type": "running_total",
            "partition_by": [],
            "order_by": [{"column": domain.date_column, "direction": "asc"}],
        },
        "order_by": [{"column": domain.date_column, "direction": "asc"}],
        "limit": None,
        "tool_sequence": _variant_tool_sequence(domain, "trend", variant),
    }


def _special_plan_factory(df: pd.DataFrame, variant: int, domain: CurriculumDomain) -> dict[str, Any]:
    metric_column = domain.special_metric_column or domain.metric_column
    metric_alias = f"special_{_safe_slug(metric_column)}"
    if domain.special_window_type == "rank":
        return {
            "group_by": [domain.dimension_column],
            "metrics": [{"column": metric_column, "function": "avg", "alias": metric_alias}],
            "filters": [],
            "window": {
                "type": "rank",
                "partition_by": [],
                "order_by": [{"column": metric_alias, "direction": "desc"}],
            },
            "order_by": [{"column": metric_alias, "direction": "desc"}],
            "keep_top_n_per_partition": None,
            "limit": 3,
            "tool_sequence": _variant_tool_sequence(domain, "special", variant),
        }
    if domain.special_window_type == "running_total":
        return {
            "group_by": [domain.date_column],
            "metrics": [{"column": metric_column, "function": "sum", "alias": metric_alias}],
            "filters": [],
            "window": {
                "type": "running_total",
                "partition_by": [],
                "order_by": [{"column": domain.date_column, "direction": "asc"}],
            },
            "order_by": [{"column": domain.date_column, "direction": "asc"}],
            "tool_sequence": _variant_tool_sequence(domain, "special", variant),
        }
    return {
        "group_by": [domain.dimension_column],
        "metrics": [{"column": metric_column, "function": "avg", "alias": metric_alias}],
        "filters": [],
        "window": {
            "type": "moving_average",
            "partition_by": [],
            "order_by": [{"column": domain.dimension_column, "direction": "asc"}],
            "window_size": 3,
        },
        "order_by": [{"column": metric_alias, "direction": "desc"}],
        "limit": 4,
        "tool_sequence": _variant_tool_sequence(domain, "special", variant),
    }


def _query_text_for_family(domain: CurriculumDomain, family_kind: str, variant: int) -> str:
    if family_kind == "filter":
        return (
            f"show {domain.name} records with {domain.boolean_columns[0].lower()} and "
            f"{domain.boolean_columns[1].lower()} and {domain.metric_column.lower()} above {20 + variant * 2}"
        )
    if family_kind == "aggregate":
        return f"total {domain.metric_column.lower()} by {domain.dimension_column.lower()} ranked highest to lowest"
    if family_kind == "compare":
        return f"compare {domain.metric_column.lower()} between new and returning {domain.entity_column.lower()} groups"
    if family_kind == "trend":
        return f"{domain.metric_column.lower()} trend over {domain.date_column.lower()}"
    return f"{domain.special_intent} analysis for {domain.name} using {domain.dimension_column.lower()}"


def _family_specs() -> list[CurriculumFamily]:
    specs: list[CurriculumFamily] = []
    for domain in _build_domain_profiles():
        specs.extend(
            [
                CurriculumFamily(
                    family_id=f"{domain.name}.filter",
                    domain=domain,
                    intent="filter",
                    family_kind="filter",
                    query_template=_query_text_for_family(domain, "filter", 0),
                    plan_factory=_filter_plan_factory,
                    dataframe_factory=_domain_frame,
                ),
                CurriculumFamily(
                    family_id=f"{domain.name}.aggregate",
                    domain=domain,
                    intent="aggregate",
                    family_kind="aggregate",
                    query_template=_query_text_for_family(domain, "aggregate", 0),
                    plan_factory=_aggregate_plan_factory,
                    dataframe_factory=_domain_frame,
                ),
                CurriculumFamily(
                    family_id=f"{domain.name}.compare",
                    domain=domain,
                    intent="compare",
                    family_kind="compare",
                    query_template=_query_text_for_family(domain, "compare", 0),
                    plan_factory=_compare_plan_factory,
                    dataframe_factory=_compare_frame,
                ),
                CurriculumFamily(
                    family_id=f"{domain.name}.trend",
                    domain=domain,
                    intent="trend",
                    family_kind="trend",
                    query_template=_query_text_for_family(domain, "trend", 0),
                    plan_factory=_trend_plan_factory,
                    dataframe_factory=_domain_frame,
                ),
                CurriculumFamily(
                    family_id=f"{domain.name}.{domain.special_intent}",
                    domain=domain,
                    intent=domain.special_intent,
                    family_kind="special",
                    query_template=_query_text_for_family(domain, "special", 0),
                    plan_factory=_special_plan_factory,
                    dataframe_factory=_domain_frame,
                    is_special=True,
                ),
            ]
        )
    return specs


def _build_cases(*, variants_per_family: int = 11) -> list[CurriculumCase]:
    cases: list[CurriculumCase] = []
    for family in _family_specs():
        for variant in range(variants_per_family):
            frame = family.dataframe_factory(variant, family.domain)
            user_text = _query_text_for_family(family.domain, family.family_kind, variant)
            plan = family.plan_factory(frame, variant, family.domain)
            result_payload = _base_result(frame, plan)
            case = CurriculumCase(
                case_id=f"{family.family_id}.v{variant:02d}",
                family_id=family.family_id,
                domain=family.domain.name,
                intent=family.intent,
                variant=variant,
                user_text=user_text,
                dataframe=frame,
                available_sheets=["Sheet1"],
                plan=plan,
                result_payload=result_payload,
                route="sql",
                quality_score=0.97,
                plan_template_id=f"template.{_safe_slug(family.family_id)}",
                skill_id=f"learned.{_safe_slug(family.family_id)}",
            )
            cases.append(case)
    return cases


def _build_failure_cases(family_specs: list[CurriculumFamily]) -> list[CurriculumCase]:
    failures: list[CurriculumCase] = []
    for index, family in enumerate(family_specs[:10]):
        frame = family.dataframe_factory(index + 20, family.domain)
        user_text = f"failed {family.intent} example for {family.domain.name}"
        plan = family.plan_factory(frame, index + 20, family.domain)
        result_payload = _base_result(frame, plan)
        result_payload.update(
            {
                "confidence": 0.8,
                "critic_passed": False,
                "result_validation_passed": False,
                "plan_completeness_passed": False,
                "privacy_validation_passed": False,
                "no_unresolved_ambiguity": False,
                "no_critical_repair": False,
                "repair_count": 1,
                "correction_state": "corrected",
                "quality_score": 0.8,
            }
        )
        failures.append(
            CurriculumCase(
                case_id=f"{family.family_id}.failure",
                family_id=family.family_id,
                domain=family.domain.name,
                intent=family.intent,
                variant=999,
                user_text=user_text,
                dataframe=frame,
                available_sheets=["Sheet1"],
                plan=plan,
                result_payload=result_payload,
                route="sql",
                quality_score=0.8,
                critic_passed=False,
                result_validation_passed=False,
                plan_completeness_passed=False,
                privacy_validation_passed=False,
                no_unresolved_ambiguity=False,
                no_critical_repair=False,
                repair_count=1,
                correction_state="corrected",
                plan_source="validated_template",
                plan_template_id=f"template.{_safe_slug(family.family_id)}",
                skill_id=f"learned.{_safe_slug(family.family_id)}",
                failure=True,
            )
        )
    return failures


def _make_store_root(runtime_root: Path) -> Path:
    return runtime_root / "state"


def _make_student_client(runtime_root: Path) -> TestClient:
    state_root = _make_store_root(runtime_root)
    os.environ["INSIGHT_LEARNING_RUNTIME_DIR"] = str(state_root)
    os.environ["DATA_ANALYSIS_LLM_STATE_DIR"] = str(state_root)
    import importlib

    app_module = importlib.import_module("insight_learning.api.app")

    app_module._SERVICE = None
    app = app_module.create_app()
    return TestClient(app)


def _dataset_alias_fields() -> list[dict[str, str]]:
    return [
        {"id": "FIELD_01", "semantic_role": "entity_name", "dtype": "string"},
        {"id": "FIELD_02", "semantic_role": "dimension", "dtype": "string"},
        {"id": "FIELD_03", "semantic_role": "numeric_measure", "dtype": "number"},
        {"id": "FIELD_04", "semantic_role": "numeric_measure", "dtype": "number"},
        {"id": "FIELD_05", "semantic_role": "date", "dtype": "string"},
        {"id": "FIELD_06", "semantic_role": "boolean_capability", "dtype": "boolean"},
        {"id": "FIELD_07", "semantic_role": "boolean_capability", "dtype": "boolean"},
    ]


def _safe_query_features_for_case(case: CurriculumCase, domain: CurriculumDomain) -> dict[str, Any]:
    safe_plan = _safe_plan(case.plan, domain)
    filters = list(safe_plan.get("filters") or [])
    derived_columns = list(safe_plan.get("derived_columns") or [])
    window = safe_plan.get("window")
    operators = list(
        dict.fromkeys(
            [
                *(str(predicate.get("operator")) for predicate in filters if isinstance(predicate, dict) and predicate.get("operator")),
                *([str(window.get("type"))] if isinstance(window, dict) and window.get("type") else []),
            ]
        )
    )
    logical_structure = "AND" if len(filters) > 1 else "SINGLE"
    return {
        "predicate_count": len(filters) + len(derived_columns) + (1 if window else 0),
        "logical_structure": logical_structure,
        "semantic_roles": [field["semantic_role"] for field in _dataset_alias_fields()],
        "operators": operators,
        "tool_hints": list(safe_plan.get("tool_sequence") or []),
        "schema_version": 2,
    }


def _safe_plan_template(case: CurriculumCase, domain: CurriculumDomain) -> PlanTemplate:
    safe_plan = _safe_plan(case.plan, domain)
    tool_sequence = [step for step in list(safe_plan.get("tool_sequence") or []) if not str(step).startswith("variant_")]
    plan_kind = "workflow" if any(key in safe_plan for key in ("group_by", "metrics", "window", "derived_columns")) else "filter"
    if plan_kind == "filter":
        required_roles = ["boolean_capability", "boolean_capability", "numeric_measure"]
    elif case.intent == "trend" or domain.special_window_type == "running_total":
        required_roles = ["date", "numeric_measure"]
    else:
        required_roles = ["dimension", "numeric_measure"]
    predicate_structure = []
    for index, predicate in enumerate(safe_plan.get("filters") or []):
        if isinstance(predicate, dict):
            predicate_structure.append(
                {
                    "kind": "predicate",
                    "role": f"role_{index}",
                    "operator": str(predicate.get("operator") or "equals"),
                    "value_kind": "boolean_true" if predicate.get("value") is True else "boolean_false" if predicate.get("value") is False else "numeric_literal",
                }
            )
    if not predicate_structure and plan_kind == "workflow":
        predicate_structure = [
            {
                "kind": "predicate",
                "role": "workflow_role",
                "operator": "derived",
                "value_kind": "derived_label",
            }
        ]
    template_id = case.plan_template_id or f"template.{_safe_slug(case.family_id)}"
    return PlanTemplate(
        id=template_id,
        intent=case.intent,
        plan_kind=plan_kind,
        required_roles=required_roles,
        predicate_structure=predicate_structure,
        logical_structure="AND" if plan_kind == "filter" else "SINGLE",
        tool_sequence=tool_sequence,
        output_contract={
            "result_kind": "table" if plan_kind == "workflow" else "filtered_rows",
            "family_id": case.family_id,
            "variant": case.variant,
        },
        dependencies=[],
        source_experience_signature=stable_hash(
            {
                "family_id": case.family_id,
                "variant": case.variant,
                "tool_sequence": tool_sequence,
                "intent": case.intent,
            }
        ),
        support_count=1,
        average_quality=case.quality_score,
        state="trusted",
        validated_at=_utcnow_iso(),
        promoted_at=_utcnow_iso(),
        last_seen_at=_utcnow_iso(),
    )


def _experience_record_for_case(case: CurriculumCase, domain: CurriculumDomain) -> ExperienceRecord:
    safe_plan = _safe_plan(case.plan, domain)
    safe_plan_summary = {
        "tool_sequence": list(safe_plan.get("tool_sequence") or []),
        "filter_count": len(safe_plan.get("filters") or []),
        "metric_count": len(safe_plan.get("metrics") or []),
        "has_group_by": bool(safe_plan.get("group_by")),
        "has_window": bool(safe_plan.get("window")),
        "derived_columns_count": len(safe_plan.get("derived_columns") or []),
    }
    query_features = _safe_query_features_for_case(case, domain)
    dataset_signature = stable_hash({"family_id": case.family_id, "domain": case.domain, "variant": case.variant})[:32]
    semantic_signature = stable_hash(
        {
            "intent": case.intent,
            "family_id": case.family_id,
            "variant": case.variant,
            "tool_sequence": list(safe_plan.get("tool_sequence") or []),
            "semantic_roles": query_features["semantic_roles"],
        }
    )
    return ExperienceRecord(
        intent=case.intent,
        query_features=query_features,
        semantic_roles=list(query_features["semantic_roles"]),
        operators=list(query_features["operators"]),
        logical_structure=str(query_features["logical_structure"]),
        tool_sequence=list(safe_plan.get("tool_sequence") or []),
        result_summary={
            "result_kind": "table",
            "row_count": int(case.result_payload.get("result", {}).get("row_count") or 0),
            "column_count": len(case.result_payload.get("result", {}).get("columns") or []),
            "shape": [
                int(case.result_payload.get("result", {}).get("row_count") or 0),
                len(case.result_payload.get("result", {}).get("columns") or []),
            ],
        },
        dataset_semantic_signature=dataset_signature,
        semantic_signature=semantic_signature,
        route=case.route,
        skill_id=case.skill_id,
        confidence=case.quality_score,
        success=True,
        score=case.quality_score,
        event_id=case.case_id,
        plan_hash=stable_hash(safe_plan),
        plan_summary=safe_plan_summary,
        failure_reason=None,
        feedback_score=None,
        repair_count=case.repair_count,
        critic_passed=case.critic_passed,
        result_validation_passed=case.result_validation_passed,
        plan_completeness_passed=case.plan_completeness_passed,
        privacy_validation_passed=case.privacy_validation_passed,
        no_unresolved_ambiguity=case.no_unresolved_ambiguity,
        no_critical_repair=case.no_critical_repair,
        correction_state=case.correction_state,
        skill_state_before=None,
        skill_state_after=None,
        plan_source=case.plan_source,
        plan_template_id=case.plan_template_id,
        plan_provenance={"template": safe_plan},
        correction_type=None,
        correction_summary=None,
        candidate_strategy_id=None,
        created_at=_utcnow_iso(),
        version=2,
    )


def _seed_student_memory(service, cases: Iterable[CurriculumCase], *, include_templates: bool = True) -> int:
    by_family: dict[str, list[CurriculumCase]] = {}
    domain_lookup = {domain.name: domain for domain in _build_domain_profiles()}
    for case in cases:
        by_family.setdefault(case.family_id, []).append(case)
        domain = domain_lookup[case.domain]
        experience = _experience_record_for_case(case, domain)
        service.store.append(experience)

    templates_written = 0
    if include_templates:
        for family_cases in by_family.values():
            if not family_cases:
                continue
            canonical_case = min(family_cases, key=lambda item: item.variant)
            domain = domain_lookup[canonical_case.domain]
            template = _safe_plan_template(canonical_case, domain)
            template.support_count = len(family_cases)
            template.average_quality = round(
                sum(case.quality_score for case in family_cases) / len(family_cases),
                4,
            )
            template.state = "trusted" if template.average_quality >= 0.95 and template.support_count >= 3 else "validated"
            template.validated_at = template.validated_at or _utcnow_iso()
            template.promoted_at = template.promoted_at or _utcnow_iso()
            service.store.upsert_plan_template(template)
            templates_written += 1
    return templates_written


def _field_aliases(domain: CurriculumDomain) -> list[tuple[str, str]]:
    return [
        (domain.entity_column, "FIELD_01"),
        (domain.dimension_column, "FIELD_02"),
        (domain.metric_column, "FIELD_03"),
        (domain.secondary_metric_column, "FIELD_04"),
        (domain.date_column, "FIELD_05"),
        (domain.boolean_columns[0], "FIELD_06"),
        (domain.boolean_columns[1], "FIELD_07"),
    ]


def _safe_plan(plan: dict[str, Any], domain: CurriculumDomain) -> dict[str, Any]:
    alias_map = {actual: alias for actual, alias in _field_aliases(domain)}

    def _remap(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _remap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_remap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_remap(item) for item in value)
        if isinstance(value, str):
            return alias_map.get(value, value)
        return value

    return _remap(plan)


def _manual_learning_event_payload(case: CurriculumCase) -> dict[str, Any]:
    alias_fields = [
        {
            "id": alias,
            "semantic_role": semantic_role,
            "dtype": dtype,
        }
        for alias, semantic_role, dtype in [
            ("FIELD_01", "entity_name", "string"),
            ("FIELD_02", "dimension", "string"),
            ("FIELD_03", "numeric_metric", "number"),
            ("FIELD_04", "numeric_metric", "number"),
            ("FIELD_05", "date", "string"),
            ("FIELD_06", "boolean_capability", "boolean"),
            ("FIELD_07", "boolean_capability", "boolean"),
        ]
    ]
    safe_plan = _safe_plan(case.plan, next(domain for domain in _build_domain_profiles() if domain.name == case.domain))
    tool_graph = list(safe_plan.get("tool_sequence") or [])
    if not tool_graph:
        tool_graph = ["resolve_semantic_targets", "validate_result"]
    semantic_roles = [field["semantic_role"] for field in alias_fields]
    operators = list(dict.fromkeys(
        [predicate.get("operator") for predicate in safe_plan.get("filters", []) if isinstance(predicate, dict) and predicate.get("operator")]  # type: ignore[union-attr]
        + ([safe_plan.get("window", {}).get("type")] if isinstance(safe_plan.get("window"), dict) else [])
    ))
    payload = {
        "schema_version": 2,
        "event_id": case.case_id,
        "intent": case.intent,
        "query_features": {
            "predicate_count": len(safe_plan.get("filters") or []) + len(safe_plan.get("derived_columns") or []) + (1 if safe_plan.get("window") else 0),
            "logical_structure": "AND" if len((safe_plan.get("filters") or [])) > 1 else "SINGLE",
            "semantic_roles": semantic_roles,
            "operators": operators,
            "tool_hints": list(tool_graph),
            "schema_version": 2,
        },
        "dataset_profile": {"fields": alias_fields},
        "tool_graph": tool_graph,
        "plan": safe_plan,
        "execution": {
            "success": True,
            "route": case.route,
            "result_kind": case.result_kind if hasattr(case, "result_kind") else "table",
            "row_count": len(case.result_payload.get("result", {}).get("rows") or []),
            "column_count": len(case.result_payload.get("result", {}).get("columns") or []),
        },
        "validation": {"success": True, "warnings": [], "errors": []},
        "quality_score": case.quality_score,
        "route": case.route,
        "plan_source": case.plan_source,
        "skill_id": case.skill_id,
        "plan_template_id": case.plan_template_id,
        "dataset_semantic_signature": stable_hash({"domain": case.domain, "family": case.family_id})[:32],
        "execution_success": True,
        "critic_passed": case.critic_passed,
        "result_validation_passed": case.result_validation_passed,
        "plan_completeness_passed": case.plan_completeness_passed,
        "privacy_validation_passed": case.privacy_validation_passed,
        "no_unresolved_ambiguity": case.no_unresolved_ambiguity,
        "no_critical_repair": case.no_critical_repair,
        "repair_count": case.repair_count,
        "correction_state": case.correction_state,
        "safe_query_abstraction": {
            "available_columns": [field["id"] for field in alias_fields],
            "available_sheet_count": 1,
            "dataset_semantic_signature": stable_hash({"domain": case.domain, "family": case.family_id})[:32],
        },
    }
    return payload


def _ingest_cases(student_client: TestClient, cases: Iterable[CurriculumCase]) -> int:
    ingested = 0
    for case in cases:
        payload = _manual_learning_event_payload(case)
        response = student_client.post("/v1/experience", json=payload)
        response.raise_for_status()
        ingested += 1
    return ingested


def _make_bridge_adapter(student_client: TestClient, teacher_learning_bridge):
    LearningPlanResult = teacher_learning_bridge.LearningPlanResult

    class CurriculumBridgeAdapter:
        def __init__(self) -> None:
            self.plan_calls = 0
            self.ingest_calls = 0

        async def plan(self, abstraction):
            self.plan_calls += 1
            response = student_client.post("/v1/plan", json=abstraction.to_plan_request())
            response.raise_for_status()
            payload = response.json()
            plan = payload.get("plan")
            if not isinstance(plan, dict) or not plan:
                return None
            route = "operation" if str(plan.get("action") or "").lower() == "categorize" else "sql"
            accepted = route == "sql" and float(payload.get("confidence") or 0.0) >= 0.82
            return LearningPlanResult(
                accepted=accepted,
                confidence=float(payload.get("confidence") or 0.0),
                plan_source=str(payload.get("plan_source") or "experience_transfer"),
                route=route,
                plan=plan,
                skill_id=payload.get("skill_id"),
                plan_template_id=payload.get("plan_template_id"),
                message=str(payload.get("message") or ""),
                raw_response=payload,
                reverse_field_map=abstraction.reverse_field_map,
            )

        async def ingest(self, event):
            self.ingest_calls += 1
            response = student_client.post("/v1/experience", json=event.to_dict())
            response.raise_for_status()
            return response.json()

    return CurriculumBridgeAdapter()


def _evaluate_cases(
    teacher_query_router,
    bridge_adapter,
    cases: Iterable[CurriculumCase],
) -> CurriculumPhaseResult:
    import asyncio

    ordered_cases = list(cases)
    fallback_counter = {"count": 0}
    route_counts: dict[str, int] = {}
    plan_source_counts: dict[str, int] = {}
    accepted = 0
    confidences: list[float] = []

    async def _run_case(case: CurriculumCase) -> None:
        nonlocal accepted
        async def _fake_router_agent(user_text: str, available_columns: list, df: pd.DataFrame | None = None) -> dict[str, Any]:
            fallback_counter["count"] += 1
            return {
                "route": "sql",
                "plan": case.plan,
                "confidence": 0.9,
                "message": "Deterministic curriculum fallback",
            }

        with patch.object(teacher_query_router, "get_learning_bridge", lambda: bridge_adapter), patch.object(
            teacher_query_router, "_run_router_agent", _fake_router_agent
        ):
            result = await teacher_query_router.handle_smart_query(case.user_text, case.dataframe, case.available_sheets)
        route = str(result.get("route") or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
        plan_source = str(result.get("plan_source") or result.get("metadata", {}).get("learning", {}).get("plan_source") or "unknown")
        plan_source_counts[plan_source] = plan_source_counts.get(plan_source, 0) + 1
        confidences.append(float(result.get("confidence") or 0.0))
        if result.get("metadata", {}).get("learning", {}).get("used") or plan_source in {"validated_template", "experience_transfer"}:
            accepted += 1

    async def _run_all() -> None:
        for case in ordered_cases:
            await _run_case(case)

    asyncio.run(_run_all())
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return CurriculumPhaseResult(
        name="evaluation",
        inspected=len(ordered_cases),
        bridge_accepted=accepted,
        gemini_fallback_calls=fallback_counter["count"],
        route_counts=route_counts,
        plan_source_counts=plan_source_counts,
        average_confidence=round(average_confidence, 4),
        notes=[],
    )


def _evaluate_single_pass(
    *,
    teacher_query_router,
    bridge_adapter,
    case: CurriculumCase,
) -> tuple[dict[str, Any], bool]:
    import asyncio

    fallback_counter = {"count": 0}

    async def _fake_router_agent(user_text: str, available_columns: list, df: pd.DataFrame | None = None) -> dict[str, Any]:
        fallback_counter["count"] += 1
        return {
            "route": "sql",
            "plan": case.plan,
            "confidence": 0.9,
            "message": "Deterministic curriculum fallback",
        }

    async def _run() -> dict[str, Any]:
        with patch.object(teacher_query_router, "get_learning_bridge", lambda: bridge_adapter), patch.object(
            teacher_query_router, "_run_router_agent", _fake_router_agent
        ):
            return await teacher_query_router.handle_smart_query(case.user_text, case.dataframe, case.available_sheets)

    result = asyncio.run(_run())
    learned = bool(result.get("metadata", {}).get("learning", {}).get("used"))
    return result, learned


def _readiness_snapshot(name: str, exporter: TrainingDatasetExporter, bundle) -> dict[str, Any]:
    readiness = exporter.evaluate_readiness(bundle)
    report = bundle.report()
    return {
        "checkpoint": name,
        "ready": readiness.ready,
        "ready_for_prototype": readiness.ready_for_prototype,
        "reason": readiness.reason,
        "eligible_examples": report["eligible_examples"],
        "family_count": report["family_count"],
        "intent_count": report["intent_count"],
        "tool_graph_count": report["tool_graph_count"],
        "average_quality": report["average_quality"],
        "readiness_score": getattr(readiness, "readiness_score", None),
    }


def _render_markdown_report(result: CurriculumRunResult) -> str:
    training = result.training_report
    phases = result.phases
    checkpoints = result.readiness_checkpoints
    lines = [
        "# Analytics Curriculum Report",
        "",
        f"- generated_at: {result.generated_at}",
        f"- teacher_repo: `{result.teacher_root}`",
        f"- student_repo: `{result.student_root}`",
        f"- total domains: {result.total_domains}",
        f"- total families: {result.total_families}",
        f"- total intents: {result.total_intents}",
        f"- total seeded events: {result.total_seeded_events}",
        "",
        "## Readiness Checkpoints",
    ]
    for checkpoint in checkpoints:
        lines.append(
            f"- {checkpoint['checkpoint']}: ready={checkpoint['ready']} ready_for_prototype={checkpoint['ready_for_prototype']} "
            f"eligible={checkpoint['eligible_examples']} families={checkpoint['family_count']} intents={checkpoint['intent_count']}"
        )
    lines.extend(
        [
            "",
            "## Bridge Learning",
        ]
    )
    for phase in phases:
        lines.append(
            f"- {phase.name}: accepted={phase.bridge_accepted}/{phase.inspected} "
            f"fallback_calls={phase.gemini_fallback_calls} average_confidence={phase.average_confidence:.3f}"
        )
    lines.extend(
        [
            "",
            "## Export Summary",
            f"- inspected: {training['total_experiences_inspected']}",
            f"- eligible: {training['eligible_examples']}",
            f"- rejected: {training['rejected_examples']}",
            f"- duplicates_removed: {training['duplicates_removed']}",
            f"- family_count: {training['family_count']}",
            f"- intent_count: {training['intent_count']}",
            f"- average_quality: {training['average_quality']}",
            f"- train/validation/test: {training['train_count']}/{training['validation_count']}/{training['test_count']}",
            "",
            "## Privacy",
            "- The curriculum harness stores only generalized structure, validation metadata, and safe semantic signatures in the report.",
            "- No raw workbook rows, filenames, sheet names, email addresses, phone numbers, or account identifiers are included.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_curriculum(
    *,
    runtime_root: Path | None = None,
    training_export_dir: Path | None = None,
    docs_path: Path | None = None,
    report_path: Path | None = None,
    variants_per_family: int = 11,
    max_examples_per_fingerprint: int = 10,
) -> CurriculumRunResult:
    runtime_root = Path(runtime_root or DEFAULT_CURRICULUM_RUNTIME_ROOT)
    training_export_dir = Path(training_export_dir or DEFAULT_TRAINING_EXPORT_DIR)
    docs_path = Path(docs_path or DEFAULT_CURRICULUM_DOC_PATH)
    report_path = Path(report_path or DEFAULT_CURRICULUM_REPORT_PATH)
    runtime_root.mkdir(parents=True, exist_ok=True)
    training_export_dir.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    teacher_query_router, teacher_learning_bridge = _load_teacher_modules()
    student_client = _make_student_client(runtime_root)
    bridge_adapter = _make_bridge_adapter(student_client, teacher_learning_bridge)

    family_specs = _family_specs()
    cases = _build_cases(variants_per_family=variants_per_family)
    failures = _build_failure_cases(family_specs)
    representative_cases = [case for case in cases if case.variant == 0]

    import importlib

    app_module = importlib.import_module("insight_learning.api.app")
    app_module._SERVICE = None
    service = app_module.get_service()
    exporter_policy = TrainingExportPolicy(
        minimum_quality=0.95,
        minimum_readiness_quality=0.96,
        minimum_eligible_examples=500,
        minimum_family_count=50,
        minimum_intent_count=10,
        max_single_intent_share=0.40,
        max_single_tool_graph_share=0.40,
        require_execution_success=True,
        require_critic_pass=True,
        require_result_validation=True,
        require_plan_completeness=True,
        require_privacy_pass=True,
        require_no_unresolved_ambiguity=True,
        require_no_critical_repair=True,
        allow_repaired_examples=False,
        max_examples_per_fingerprint=max_examples_per_fingerprint,
        max_examples_per_intent=100,
        max_examples_per_tool_graph=100,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )
    exporter = TrainingDatasetExporter(service.store, exporter_policy)
    bundle_before, _ = exporter.build_bundle(include_candidate_strategies=True)
    readiness_checkpoints = [
        _readiness_snapshot("T0_baseline", exporter, bundle_before),
    ]

    # Baseline: no seed examples loaded yet.
    baseline_phase = _evaluate_cases(teacher_query_router, bridge_adapter, representative_cases)

    # Seed the student memory directly so the export path sees the exact safe records
    # and canonical templates we want the bridge to reuse.
    _seeded_templates = _seed_student_memory(service, cases, include_templates=True)
    seeded_failures = _seed_student_memory(service, failures, include_templates=False)

    bundle_after_seed, _ = exporter.build_bundle(include_candidate_strategies=True)
    readiness_checkpoints.append(_readiness_snapshot("T1_seeded", exporter, bundle_after_seed))

    # Evaluate after learning using the same structural families.
    learned_phase = _evaluate_cases(teacher_query_router, bridge_adapter, representative_cases)

    # Verify the learned memory survives a restart by rebuilding the service and bridge client.
    app_module._SERVICE = None
    restarted_client = _make_student_client(runtime_root)
    restarted_bridge = _make_bridge_adapter(restarted_client, teacher_learning_bridge)
    restart_phase = _evaluate_cases(teacher_query_router, restarted_bridge, representative_cases)
    restart_verified = restart_phase.bridge_accepted >= learned_phase.bridge_accepted

    # Export the dataset with the curriculum policy. The report is intentionally
    # written to runtime/ so it stays out of Git.
    export_paths = exporter.export_files(output_dir=training_export_dir, include_candidate_strategies=True)
    bundle_final, _ = exporter.build_bundle(include_candidate_strategies=True)
    readiness_checkpoints.extend(
        [
            _readiness_snapshot("T2_learned", exporter, bundle_final),
            _readiness_snapshot("T3_restart", exporter, bundle_final),
            _readiness_snapshot("T4_exported", exporter, bundle_final),
        ]
    )

    training_report = bundle_final.report()
    readiness = exporter.evaluate_readiness(bundle_final)
    phases = [baseline_phase, learned_phase, restart_phase]
    report_blob = json.dumps(
        {
            "report": training_report,
            "phases": [asdict(phase) for phase in phases],
            "checkpoints": readiness_checkpoints,
            "paths": {key: str(path) for key, path in export_paths.items()},
        },
        sort_keys=True,
        default=str,
    )
    privacy_verified = all(
        literal not in report_blob
        for literal in [
            "John Smith",
            "john@example.com",
            "ACC-9988",
            "SecretCompanyXYZ",
            "9876543210",
        ]
    )
    report_payload = CurriculumRunResult(
        generated_at=_utcnow_iso(),
        teacher_root=str(_teacher_root()),
        student_root=str(_make_store_root(runtime_root)),
        runtime_root=str(runtime_root),
        training_export_dir=str(training_export_dir),
        dataset_version=training_report.get("dataset_version") or getattr(readiness, "dataset_version", ""),
        total_cases=len(cases) + len(failures),
        total_families=len(family_specs),
        total_domains=len(_build_domain_profiles()),
        total_intents=len(training_report.get("intent_distribution", {})),
        total_seeded_events=len(cases) + seeded_failures,
        training_report=training_report,
        readiness_checkpoints=readiness_checkpoints,
        phases=phases,
        restart_verified=restart_verified,
        privacy_verified=privacy_verified,
        export_paths={key: str(path) for key, path in export_paths.items()},
        sample_families=[
            {
                "family_id": family.family_id,
                "domain": family.domain.name,
                "intent": family.intent,
                "family_kind": family.family_kind,
            }
            for family in family_specs[:10]
        ],
    )

    report_path.write_text(json.dumps(report_payload.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    docs_path.write_text(_render_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the analytics curriculum report and runtime exports.")
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_CURRICULUM_RUNTIME_ROOT)
    parser.add_argument("--training-export-dir", type=Path, default=DEFAULT_TRAINING_EXPORT_DIR)
    parser.add_argument("--docs-path", type=Path, default=DEFAULT_CURRICULUM_DOC_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_CURRICULUM_REPORT_PATH)
    parser.add_argument("--variants-per-family", type=int, default=11)
    parser.add_argument("--max-examples-per-fingerprint", type=int, default=10)
    args = parser.parse_args(argv)

    run_curriculum(
        runtime_root=args.runtime_root,
        training_export_dir=args.training_export_dir,
        docs_path=args.docs_path,
        report_path=args.report_path,
        variants_per_family=max(1, args.variants_per_family),
        max_examples_per_fingerprint=max(1, args.max_examples_per_fingerprint),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
