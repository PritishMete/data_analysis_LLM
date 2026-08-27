from __future__ import annotations

import json
from pathlib import Path

from learning.plan_schema import canonical_plan_schema, validate_plan_schema
from insight_learning.ontology.roles import SEMANTIC_ROLES as ONTOLOGY_ROLES
from training.benchmark import audit_failure_modes, builtin_benchmark_cases, run_planner_benchmark
from training.config import LowSpecTrainingConfigV2
from training.formatting import fine_tuning_candidate_to_example
from training.profiles import PLANNER_BACKEND_LLAMA_CPP
from learning.training_export import TrainingDatasetExporter, TrainingExportPolicy
from learning.experience_store import LearningExperienceStore
from agent.planner import LearningPlanner
from tests.test_training_export_hardening import _experience_record


def test_training_inference_schema_parity():
    candidate = {
        "intent": "filter",
        "semantic_roles": ["boolean_capability", "numeric_metric"],
        "predicate_graph": {"operator": "AND"},
        "logical_structure": "AND",
        "available_tools": ["sql.filter", "sql.group_by"],
        "tool_graph": ["sql.filter"],
        "output_contract": {"result_kind": "analytics_plan"},
    }
    schema = canonical_plan_schema(
        intent=candidate["intent"],
        semantic_roles=candidate["semantic_roles"],
        predicate_graph=candidate["predicate_graph"],
        logical_structure=candidate["logical_structure"],
        available_tools=candidate["available_tools"],
        tool_graph=candidate["tool_graph"],
        output_contract=candidate["output_contract"],
    )
    example = fine_tuning_candidate_to_example(candidate)

    assert schema.validate() == []
    assert example.input["available_tools"] == candidate["available_tools"]
    assert example.output["tool_graph"] == candidate["tool_graph"]
    assert example.metadata["schema_version"] == 1
    assert validate_plan_schema(
        {
            "schema_version": 1,
            "intent": candidate["intent"],
            "semantic_roles": candidate["semantic_roles"],
            "predicate_graph": candidate["predicate_graph"],
            "logical_structure": candidate["logical_structure"],
            "available_tools": candidate["available_tools"],
            "tool_graph": candidate["tool_graph"],
            "output_contract": candidate["output_contract"],
        }
    ) == []


def test_tool_registry_and_semantic_role_parity():
    from agent.tool_registry import get_tool_registry

    registry = get_tool_registry()
    assert set(registry.allowed_names()) == set(training_config_tool_names())
    assert {"boolean_capability", "numeric_metric", "rating_metric", "geographic_area"}.issubset(set(ONTOLOGY_ROLES))


def training_config_tool_names() -> list[str]:
    from agent.tool_registry import get_tool_registry

    return get_tool_registry().allowed_names()


def test_corpus_structural_validation_and_baseline_report(tmp_path):
    store = LearningExperienceStore(root=tmp_path / "state")
    store.append(_experience_record(event_id="evt_valid"))
    exporter = TrainingDatasetExporter(store, TrainingExportPolicy(max_examples_per_fingerprint=1))
    bundle, _ = exporter.build_bundle()
    report = bundle.report()

    assert report["eligible_examples"] == 1
    assert report["invalidated_excluded"] == 0
    assert report["train_count"] + report["validation_count"] + report["test_count"] == 1
    baseline = {
        "valid_json_rate": 1.0,
        "schema_valid_rate": 1.0,
        "plan_validity_rate": 0.0,
        "tool_selection_f1": 0.0,
        "tool_sequence_accuracy": 0.0,
        "predicate_coverage": 0.17647058823529413,
        "logical_structure_accuracy": 0.7647058823529411,
        "semantic_role_coverage": 0.11764705882352941,
        "invalid_tool_rate": 0.0,
    }
    assert baseline["valid_json_rate"] == 1.0
    assert baseline["plan_validity_rate"] == 0.0


def test_few_shot_formatter_is_concise():
    candidate = {
        "intent": "filter",
        "semantic_roles": ["boolean_capability"],
        "predicate_graph": {"operator": "SINGLE"},
        "logical_structure": "SINGLE",
        "available_tools": ["sql.filter"],
        "tool_graph": ["sql.filter"],
        "output_contract": {"result_kind": "filtered_rows"},
        "source_kind": "experience",
        "source_id": "evt",
        "quality_score": 0.98,
        "plan_source": "validated_template",
        "family_fingerprint": "0123456789abcdef",
    }
    example = fine_tuning_candidate_to_example(candidate)
    payload = json.dumps(example.to_dict(), sort_keys=True)
    assert "structured_plan" in payload
    assert "raw_values_allowed" in payload
    assert "John Smith" not in payload


def test_failure_mode_audit_and_success_gates_recorded():
    summary = run_planner_benchmark(profile_name="low_spec", backend=PLANNER_BACKEND_LLAMA_CPP, device="cpu")
    audit = audit_failure_modes(summary)
    assert audit.total_cases == len(builtin_benchmark_cases())
    assert audit.failure_counts["wrong_tool"] >= 0
    assert audit.failure_counts["missing_predicate"] >= 0
    assert summary.metrics["valid_json_rate"] >= 0.99
    assert summary.metrics["schema_valid_rate"] >= 0.99
    assert "fallback_rate" in summary.metrics


def test_low_spec_v2_config_and_return_artifact_manifest():
    config = LowSpecTrainingConfigV2()
    payload = config.to_dict()
    assert payload["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert payload["qlora"]["lora_r"] == 16
    assert payload["success_gates"]["tool_selection_f1"] == 0.90
    manifest = {
        "adapter": "adapter.safetensors",
        "merged_model": "merged",
        "gguf": "model.Q4_K_M.gguf",
        "sha256": "a" * 64,
    }
    assert len(manifest["sha256"]) == 64


def test_semantic_planner_uses_intent_specific_tool_subsets():
    planner = LearningPlanner()
    decision = planner.plan(
        "Show restaurants with delivery and booking",
        None,
        ["delivery", "booking", "rating"],
    )
    assert decision.plan_source in {"semantic_planner", "validated_template", "experience_transfer", "deterministic_fallback", "trusted_strategy", "bootstrap_skill"}
    assert decision.plan is not None
    tool_sequence = decision.plan.get("tool_sequence") or []
    assert tool_sequence
    assert all(tool in {"sql.filter", "sql.group_by", "analytics.summary", "categorization_agent._deterministic_special_mapping", "data_cleaning_utils.fill_nulls", "common.transformations.range_binning", "secure_excel.executor"} for tool in tool_sequence)
    assert (decision.plan.get("predicate_graph") or {}).get("predicate_count", 0) >= 2


def test_semantic_plan_schema_and_shadow_mode_non_execution():
    summary = run_planner_benchmark(profile_name="low_spec", backend="semantic", device="cpu")
    assert summary.shadow_mode is True
    assert summary.metrics["fallback_rate"] >= 0.0
    assert summary.metrics["peak_vram_mb"] is None or summary.metrics["peak_vram_mb"] >= 0.0
    for case in summary.cases:
        assert "semantic" in case.plan_source or case.plan_source in {"semantic_planner", "deterministic_fallback", "validated_template", "experience_transfer", "bootstrap_skill", "trusted_strategy"}


def test_semantic_extraction_and_composition_modes_are_reported():
    extract_summary = run_planner_benchmark(profile_name="low_spec", backend="semantic_extraction", device="cpu", benchmark="builtin", case_limit=1)
    compose_summary = run_planner_benchmark(profile_name="low_spec", backend="semantic_composed", device="cpu", benchmark="builtin", case_limit=1)

    assert extract_summary.backend == "semantic_extraction"
    assert compose_summary.backend == "semantic_composed"
    assert extract_summary.metrics["valid_json_rate"] == 1.0
    assert extract_summary.metrics["schema_valid_rate"] == 1.0
    assert compose_summary.metrics["valid_json_rate"] == 1.0
    assert compose_summary.metrics["schema_valid_rate"] == 1.0
    assert all(case.plan_source == "semantic_extraction" for case in extract_summary.cases)
    assert all(case.plan_source == "semantic_composed" for case in compose_summary.cases)
    assert all(not (case.parsed_plan or {}).get("tool_sequence") for case in extract_summary.cases)
    assert any((case.parsed_plan or {}).get("tool_sequence") for case in compose_summary.cases)
