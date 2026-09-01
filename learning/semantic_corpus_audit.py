"""Privacy-safe audit of the canonical semantic-training conversion funnel."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from kaggle.bootstrap import _load_canonical_record, _load_json_if_exists, _split_row_count
from learning.semantic_extractor_training import (
    ALLOWED_SEMANTIC_INTENTS,
    build_semantic_extractor_targets,
    validate_semantic_target,
)
from learning.training_export import TrainingExportBundle, TrainingExportPolicy


MIN_TRAIN_EXAMPLES = 32
MIN_VALIDATION_EXAMPLES = 8
UNSAFE_MARKERS = ("query_text", "normalized_query", "workbook", "sheet_name", "filename", "customer name", "email", "phone", "address", "account", "sql")


def _safe_read_rows(path: Path, split: str) -> list[dict[str, Any]]:
    if split == "test":
        raise RuntimeError("TEST_SPLIT_ACCESS_FORBIDDEN")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            rows.append(value)
    return rows


def _metadata_eligible(item: dict[str, Any]) -> bool:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict) or float(metadata.get("quality") or 0.0) < 0.95:
        return False
    required_true = ("execution_success", "critic_passed", "result_validation_passed", "plan_completeness_passed", "privacy_validation_passed", "no_unresolved_ambiguity", "no_critical_repair")
    return all(metadata.get(key) is True for key in required_true) and int(metadata.get("repair_count") or 0) == 0 and str(metadata.get("correction_state") or "none").lower() not in {"corrected", "critical_repair", "repaired"}


def _privacy_valid(item: dict[str, Any]) -> bool:
    # Only inspect the already-safe canonical abstraction and metadata, never raw user text.
    safe = json.dumps({"input": item.get("input"), "output": item.get("output"), "metadata": item.get("metadata")}, sort_keys=True).lower()
    return not any(marker in safe for marker in UNSAFE_MARKERS)


def _bundle_for(item: dict[str, Any], split: str) -> TrainingExportBundle:
    record = _load_canonical_record(item, split)
    return TrainingExportBundle(
        records=[record],
        rejected_count=0,
        rejected_reasons=Counter(),
        duplicates_removed=0,
        inspected_count=1,
        policy=TrainingExportPolicy.from_env(),
        source_distribution=Counter([record.plan_source or record.source_kind]),
        intent_distribution=Counter([record.intent]),
        tool_graph_distribution=Counter(["|".join(record.tool_graph) or "<empty>"]),
        step_distribution=Counter([len(record.tool_graph)]),
        predicate_complexity_distribution=Counter([int(record.predicate_graph.get("predicate_count") or 0)]),
        average_quality=record.quality_score,
        dataset_version="",
    )


def _structure_label(item: dict[str, Any]) -> str:
    input_payload = item.get("input") if isinstance(item.get("input"), dict) else {}
    logical = str(input_payload.get("logical_structure") or "SINGLE").upper()
    predicate_count = int(input_payload.get("predicate_count") or len(input_payload.get("operators") or []) or 0)
    labels = ["multiple_predicates" if predicate_count > 1 else "single_predicate", logical]
    output = item.get("output") if isinstance(item.get("output"), dict) else {}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    plan_shape = metadata.get("plan_shape") if isinstance(metadata.get("plan_shape"), dict) else {}
    if plan_shape.get("metric_count"):
        labels.append("aggregation")
    if plan_shape.get("limit") is not None:
        labels.append("limit")
    if plan_shape.get("tool_sequence"):
        labels.append("ranking_or_multi_operation")
    if output.get("requires_fallback") or metadata.get("requires_fallback"):
        labels.append("fallback_required")
    return "|".join(sorted(set(labels)))


def _reason(item: dict[str, Any], *, structurally_readable: bool, eligible: bool, privacy_valid: bool, conversion_success: bool, target_valid: bool) -> str | None:
    if not structurally_readable:
        return "OTHER_SAFE_REASON"
    if not eligible:
        return "INELIGIBLE_EVENT"
    if not privacy_valid:
        return "PRIVACY_GATE_FAILED"
    input_payload = item.get("input") if isinstance(item.get("input"), dict) else {}
    if str(input_payload.get("intent") or "") not in ALLOWED_SEMANTIC_INTENTS:
        return "UNSUPPORTED_INTENT"
    if not conversion_success:
        return "CONVERSION_EXCEPTION"
    if not target_valid:
        return "INVALID_TARGET_SCHEMA"
    return None


def _audit_split(root: Path, split: str) -> dict[str, Any]:
    path = root / f"{split}.jsonl"
    rows = _safe_read_rows(path, split)
    counters = Counter()
    reasons = Counter()
    intents_usable = Counter()
    intents_rejected = Counter()
    structures = Counter()
    for item in rows:
        structurally_readable = isinstance(item, dict) and isinstance(item.get("input"), dict) and isinstance(item.get("output"), dict) and isinstance(item.get("metadata"), dict)
        eligible = structurally_readable and _metadata_eligible(item)
        privacy_valid = eligible and _privacy_valid(item)
        conversion_attempted = privacy_valid
        conversion_success = False
        target_valid = False
        if conversion_attempted:
            try:
                input_payload = item.get("input") if isinstance(item.get("input"), dict) else {}
                if str(input_payload.get("intent") or "") not in ALLOWED_SEMANTIC_INTENTS:
                    raise ValueError("unsupported_intent")
                targets = build_semantic_extractor_targets(_bundle_for(item, split))
                conversion_success = len(targets) == 1
                if conversion_success:
                    target_valid = validate_semantic_target(targets[0].to_dict())[0]
            except ValueError:
                conversion_success = False
            except Exception:
                conversion_success = False
        counters["structurally_readable"] += int(structurally_readable)
        counters["eligible"] += int(eligible)
        counters["privacy_valid"] += int(privacy_valid)
        counters["conversion_attempted"] += int(conversion_attempted)
        counters["conversion_success"] += int(conversion_success)
        counters["target_valid"] += int(target_valid)
        usable = target_valid
        if usable:
            counters["usable"] += 1
            output = targets[0].output
            intents_usable[str(output.get("intent") or "unknown")] += 1
            structures[_structure_label(item)] += 1
        else:
            reason = _reason(item, structurally_readable=structurally_readable, eligible=eligible, privacy_valid=privacy_valid, conversion_success=conversion_success, target_valid=target_valid) or "OTHER_SAFE_REASON"
            reasons[reason] += 1
            input_payload = item.get("input") if isinstance(item.get("input"), dict) else {}
            intents_rejected[str(input_payload.get("intent") or "unknown")] += 1
            structures[f"rejected:{reason}"] += 1
    return {
        "total": len(rows),
        "structurally_readable": counters["structurally_readable"],
        "eligible": counters["eligible"],
        "privacy_valid": counters["privacy_valid"],
        "conversion_attempted": counters["conversion_attempted"],
        "conversion_success": counters["conversion_success"],
        "target_valid": counters["target_valid"],
        "usable": counters["usable"],
        "rejection_reasons": dict(sorted(reasons.items())),
        "usable_intent_coverage": dict(sorted(intents_usable.items())),
        "rejected_intent_coverage": dict(sorted(intents_rejected.items())),
        "structure_coverage": dict(sorted(structures.items())),
    }


def classify_learning_failure(*, stage: str, training_completed: bool) -> str:
    """Keep pre-training funnel failures distinct from learning outcomes."""
    if not training_completed and stage in {"dataset", "subset_selection", "conversion", "privacy", "eligibility"}:
        return "INSUFFICIENT_SEMANTIC_TRAINING_EXAMPLES"
    if training_completed:
        return "MODEL_NOT_LEARNING"
    return "SEMANTIC_CORPUS_TOO_SMALL"


def audit_dataset(root: Path, *, output_path: Path | None = None) -> dict[str, Any]:
    train = _audit_split(root, "train")
    validation = _audit_split(root, "validation")
    # Counts come from metadata only; this intentionally does not read test.jsonl.
    manifest = _load_json_if_exists(root / "dataset_manifest.json") or _load_json_if_exists(root / "manifest.json") or {}
    report = _load_json_if_exists(root / "dataset_report.json") or _load_json_if_exists(root / "report.json") or {}
    test_total = int(manifest.get("test_count", report.get("test_count", 0)) or 0)
    if train["usable"] >= 128 and validation["usable"] >= 16:
        classification = "SEMANTIC_SUBSET_SELECTION_BUG"
    elif train["usable"] < MIN_TRAIN_EXAMPLES or validation["usable"] < MIN_VALIDATION_EXAMPLES:
        classification = "SEMANTIC_CORPUS_TOO_SMALL"
    else:
        classification = "INSUFFICIENT_SEMANTIC_TRAINING_EXAMPLES"
    result = {
        "train": train,
        "validation": validation,
        "train_total": train["total"],
        "validation_total": validation["total"],
        "test_total_from_metadata": test_total,
        "recommended_next_train_size": min(128, train["usable"]),
        "recommended_next_validation_size": min(16, validation["usable"]),
        "minimum_train_size": MIN_TRAIN_EXAMPLES,
        "minimum_validation_size": MIN_VALIDATION_EXAMPLES,
        "target_parser_evaluator_contract": "canonical_semantic_output_keys",
        "test_split_accessed": False,
        "classification": classification,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_dataset(args.dataset_root, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
