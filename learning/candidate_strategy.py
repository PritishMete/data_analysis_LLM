from __future__ import annotations

from collections import defaultdict

from learning.config import PROMOTION_THRESHOLDS
from learning.models import CandidateStrategy, ExperienceRecord, PlanTemplate, SkillSpec, stable_hash
from learning.template_learning import extract_template_from_experience


def _strategy_key(record: ExperienceRecord) -> tuple[str, str, str, tuple[str, ...]]:
    return (
        record.intent,
        record.logical_structure,
        record.plan_template_id or record.semantic_signature,
        tuple(str(item) for item in record.tool_sequence),
    )


def discover_candidate_strategy(records: list[ExperienceRecord]) -> CandidateStrategy | None:
    successful = [record for record in records if record.success]
    if len(successful) < 2:
        return None

    groups: dict[tuple[str, str, str, tuple[str, ...]], list[ExperienceRecord]] = defaultdict(list)
    for record in successful:
        groups[_strategy_key(record)].append(record)

    best_records: list[ExperienceRecord] = []
    best_key: tuple[str, str, str, tuple[str, ...]] | None = None
    for key, grouped in groups.items():
        if len(grouped) > len(best_records):
            best_records = grouped
            best_key = key

    if not best_records or best_key is None or len(best_records) < 2:
        return None

    intent, logical_structure, signature, tool_sequence = best_key
    semantic_roles = sorted({role for record in best_records for role in record.semantic_roles})
    average_quality = round(sum(record.score for record in best_records) / len(best_records), 4)
    template = best_records[0].plan_provenance.get("template") if best_records[0].plan_provenance else None
    template_id = best_records[0].plan_template_id or None
    if isinstance(template, dict) and not template_id:
        template_id = str(template.get("id") or "")
    plan_template = template if isinstance(template, dict) else {}
    return CandidateStrategy(
        strategy_id=f"strategy.{stable_hash({'intent': intent, 'signature': signature, 'tools': list(tool_sequence)})[:12]}",
        intent=intent,
        semantic_signature=signature,
        tool_sequence=list(tool_sequence),
        semantic_roles=semantic_roles,
        evidence_count=len(best_records),
        average_quality=average_quality,
        logical_structure=logical_structure,
        plan_template_id=template_id,
        output_contract=plan_template.get("output_contract") if isinstance(plan_template, dict) else {},
        dependencies=plan_template.get("dependencies") if isinstance(plan_template, dict) else [],
        plan_template=plan_template if isinstance(plan_template, dict) else {},
        state="candidate",
        lifecycle_state="observed"
        if len(best_records) < PROMOTION_THRESHOLDS["candidate_successes"]
        else ("validated" if len(best_records) < PROMOTION_THRESHOLDS["trusted_successes"] else "trusted"),
        last_seen_at=max(record.created_at for record in best_records),
    )


def promote_candidate_strategy(strategy: CandidateStrategy) -> tuple[bool, SkillSpec | None]:
    if strategy.evidence_count < PROMOTION_THRESHOLDS["validated_successes"] or strategy.average_quality < PROMOTION_THRESHOLDS["validated_quality"]:
        return False, None

    tool_sequence = list(strategy.tool_sequence)
    spec = SkillSpec(
        id=strategy.strategy_id.replace("strategy.", "learned."),
        name=f"Learned {strategy.intent.replace('_', ' ').title()} Strategy",
        description="A promoted strategy discovered from repeated successful experiences.",
        intents=[strategy.intent],
        examples=[],
        required_semantic_roles=list(strategy.semantic_roles),
        required_input_types=["table"],
        preconditions=["observed repeated success in the local learning store"],
        supported_parameters=[],
        tool=tool_sequence[0] if tool_sequence else "learned.strategy",
        validation_rules=["replay against safe historical experiences"],
        failure_conditions=["evidence insufficient", "unsafe replay"],
        expected_result="A promoted learned strategy.",
        post_execution_checks=["quality remains high"],
        confidence=min(0.99, max(0.85, strategy.average_quality)),
        source_implementation="learning.candidate_strategy",
        quality=strategy.average_quality,
        success_count=strategy.evidence_count,
        failure_count=0,
        lifecycle="trusted" if strategy.evidence_count >= PROMOTION_THRESHOLDS["trusted_successes"] and strategy.average_quality >= PROMOTION_THRESHOLDS["trusted_quality"] else "validated",
        provenance={
            "source": "candidate_strategy",
            "strategy_id": strategy.strategy_id,
            "evidence_count": strategy.evidence_count,
            "average_quality": strategy.average_quality,
        },
        version=1,
        tags=["learned", "candidate", "promoted"],
        tool_sequence=tool_sequence,
        plan_template_id=strategy.plan_template_id,
        required_roles=list(strategy.semantic_roles),
        output_contract=dict(strategy.output_contract or {}),
        dependencies=list(strategy.dependencies or []),
    )
    return True, spec


def update_candidate_memory(
    *,
    experience: ExperienceRecord,
    recent_experiences: list[ExperienceRecord],
) -> CandidateStrategy | None:
    if not experience.success:
        return None
    pool = [record for record in recent_experiences if record.success]
    pool.append(experience)
    strategy = discover_candidate_strategy(pool)
    if strategy is None:
        return None
    return strategy


def enrich_experience_with_template(
    experience: ExperienceRecord,
    *,
    decision,
    features,
    dataset_profile,
    result_summary: dict | None,
) -> ExperienceRecord:
    template = extract_template_from_experience(
        decision=decision,
        features=features,
        dataset_profile=dataset_profile,
        result_summary=result_summary,
    )
    if template is None:
        return experience
    experience.plan_template_id = template.id
    experience.plan_source = "experience_transfer" if template.state == "observed" else "validated_template"
    experience.plan_provenance = {
        "plan_source": experience.plan_source,
        "template_id": template.id,
        "experience_support": template.support_count,
        "skill_ids": [],
        "binding_confidence": 0.0,
        "template": template.to_dict(),
    }
    return experience
