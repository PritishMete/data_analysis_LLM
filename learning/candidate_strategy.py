from __future__ import annotations

from collections import defaultdict
from typing import Any

from learning.models import CandidateStrategy, ExperienceRecord, SkillSpec, stable_hash


def _strategy_key(record: ExperienceRecord) -> tuple[str, str, tuple[str, ...]]:
    return (
        record.intent,
        record.semantic_signature,
        tuple(str(item) for item in record.tool_sequence),
    )


def discover_candidate_strategy(records: list[ExperienceRecord]) -> CandidateStrategy | None:
    successful = [record for record in records if record.success]
    if len(successful) < 2:
        return None

    groups: dict[tuple[str, str, tuple[str, ...]], list[ExperienceRecord]] = defaultdict(list)
    for record in successful:
        groups[_strategy_key(record)].append(record)

    best_records: list[ExperienceRecord] = []
    best_key: tuple[str, str, tuple[str, ...]] | None = None
    for key, grouped in groups.items():
        if len(grouped) > len(best_records):
            best_records = grouped
            best_key = key

    if not best_records or best_key is None or len(best_records) < 2:
        return None

    intent, semantic_signature, tool_sequence = best_key
    semantic_roles = sorted({role for record in best_records for role in record.semantic_roles})
    average_quality = round(sum(record.score for record in best_records) / len(best_records), 4)
    return CandidateStrategy(
        strategy_id=f"strategy.{stable_hash({'intent': intent, 'semantic_signature': semantic_signature, 'tools': list(tool_sequence)})[:12]}",
        intent=intent,
        semantic_signature=semantic_signature,
        tool_sequence=list(tool_sequence),
        semantic_roles=semantic_roles,
        evidence_count=len(best_records),
        average_quality=average_quality,
        state="candidate",
        last_seen_at=max(record.created_at for record in best_records),
    )


def promote_candidate_strategy(strategy: CandidateStrategy) -> tuple[bool, SkillSpec | None]:
    if strategy.evidence_count < 3 or strategy.average_quality < 0.8:
        return False, None

    skill_id = strategy.strategy_id.replace("strategy.", "learned.")
    spec = SkillSpec(
        id=skill_id,
        name=f"Learned {strategy.intent.replace('_', ' ').title()} Strategy",
        description="A promoted strategy discovered from repeated successful experiences.",
        intents=[strategy.intent],
        examples=[],
        required_semantic_roles=list(strategy.semantic_roles),
        required_input_types=["table"],
        preconditions=["observed repeated success in the local learning store"],
        supported_parameters=[],
        tool=strategy.tool_sequence[0] if strategy.tool_sequence else "learned.strategy",
        validation_rules=["replay against safe historical experiences"],
        failure_conditions=["evidence insufficient", "unsafe replay"],
        expected_result="A promoted learned strategy.",
        post_execution_checks=["quality remains high"],
        confidence=min(0.99, max(0.8, strategy.average_quality)),
        source_implementation="learning.candidate_strategy",
        version=1,
        tags=["learned", "candidate", "promoted"],
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

