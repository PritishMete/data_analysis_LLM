from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any, Callable

from learning.models import (
    CandidateStrategy,
    CorrectionRecord,
    ExperienceRecord,
    FailureLesson,
    PlanTemplate,
    SCHEMA_VERSION,
    SkillState,
    stable_hash,
)
from learning.canonical_training import TrainingCandidateInvalidation


def _default_root() -> Path:
    override = os.environ.get("DATA_ANALYSIS_LLM_STATE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".data_analysis_llm"


class LearningExperienceStore:
    def __init__(self, root: Path | None = None):
        self.root = root or _default_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiences_path = self.root / "experiences.jsonl"
        self.failure_lessons_path = self.root / "failure_lessons.jsonl"
        self.plan_templates_path = self.root / "plan_templates.jsonl"
        self.corrections_path = self.root / "corrections.jsonl"
        self.candidate_strategies_path = self.root / "candidate_strategies.jsonl"
        self.training_invalidations_path = self.root / "training_invalidations.jsonl"
        self.summary_path = self.root / "experience_state.json"
        self._migrate_experiences()

    @staticmethod
    def _safe_experience_keys() -> set[str]:
        return {
            "event_id",
            "intent",
            "query_features",
            "semantic_roles",
            "operators",
            "logical_structure",
            "tool_sequence",
            "result_summary",
            "dataset_semantic_signature",
            "semantic_signature",
            "route",
            "skill_id",
            "confidence",
            "success",
            "score",
            "plan_hash",
            "plan_summary",
            "failure_reason",
            "feedback_score",
            "repair_count",
            "critic_passed",
            "result_validation_passed",
            "plan_completeness_passed",
            "privacy_validation_passed",
            "no_unresolved_ambiguity",
            "no_critical_repair",
            "correction_state",
            "skill_state_before",
            "skill_state_after",
            "correction_type",
            "correction_summary",
            "candidate_strategy_id",
            "plan_source",
            "plan_template_id",
            "plan_provenance",
            "created_at",
            "version",
        }

    @staticmethod
    def _safe_invalidation_keys() -> set[str]:
        return {
            "source_id",
            "family_fingerprint",
            "reason",
            "created_at",
            "corpus_version",
        }

    @staticmethod
    def _safe_query_feature_keys() -> set[str]:
        return {
            "intent",
            "predicate_count",
            "boolean_predicate_count",
            "numeric_comparison_count",
            "entity_reference_count",
            "logical_structure",
            "semantic_roles",
            "operators",
            "operation_hints",
            "tool_hints",
            "query_shape",
            "dataset_semantic_signature",
            "semantic_signature",
            "confidence",
            "predicate_graph",
            "role_candidates",
            "step_count",
            "has_multiple_steps",
            "schema_version",
        }

    def _sanitize_experience_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        if "dataset_semantic_signature" not in data and data.get("schema_signature") is not None:
            data["dataset_semantic_signature"] = data.get("schema_signature")
        data.pop("query_text", None)
        data.pop("normalized_query", None)
        data.pop("schema_signature", None)
        data.pop("tokens", None)
        data.pop("available_columns", None)

        query_features = data.get("query_features")
        if not isinstance(query_features, dict):
            query_features = {}
        query_features = {key: query_features.get(key) for key in self._safe_query_feature_keys() if key in query_features}
        query_features.setdefault("intent", data.get("intent") or data.get("route") or "unknown")
        query_features.setdefault("logical_structure", data.get("logical_structure") or "SINGLE")
        query_features.setdefault("predicate_count", len(data.get("operators") or []))
        query_features.setdefault(
            "semantic_signature",
            stable_hash({
                "intent": query_features.get("intent"),
                "route": data.get("route"),
                "skill_id": data.get("skill_id"),
                "logical_structure": query_features.get("logical_structure"),
            }),
        )
        data["query_features"] = query_features

        data.setdefault("intent", query_features.get("intent", "unknown"))
        data.setdefault("semantic_roles", [])
        data.setdefault("operators", [])
        data.setdefault("logical_structure", query_features.get("logical_structure", "SINGLE"))
        data.setdefault("tool_sequence", [])
        data.setdefault("result_summary", {})
        data.setdefault("dataset_semantic_signature", None)
        data.setdefault("event_id", None)
        data.setdefault("semantic_signature", query_features.get("semantic_signature"))
        data.setdefault("route", "unknown")
        data.setdefault("skill_id", None)
        data.setdefault("confidence", 0.0)
        data.setdefault("success", False)
        data.setdefault("score", 0.0)
        data.setdefault("plan_hash", None)
        data.setdefault("plan_summary", {})
        data.setdefault("failure_reason", None)
        data.setdefault("feedback_score", None)
        data.setdefault("repair_count", 0)
        data.setdefault("critic_passed", None)
        data.setdefault("result_validation_passed", None)
        data.setdefault("plan_completeness_passed", None)
        data.setdefault("privacy_validation_passed", None)
        data.setdefault("no_unresolved_ambiguity", None)
        data.setdefault("no_critical_repair", None)
        data.setdefault("correction_state", None)
        data.setdefault("skill_state_before", None)
        data.setdefault("skill_state_after", None)
        data.setdefault("correction_type", None)
        data.setdefault("correction_summary", None)
        data.setdefault("candidate_strategy_id", None)
        data.setdefault("created_at", data.get("created_at") or data.get("timestamp") or "")
        data.setdefault("version", SCHEMA_VERSION)
        clean = {key: data.get(key) for key in self._safe_experience_keys()}
        clean["query_features"] = query_features
        return clean

    def _sanitize_training_invalidation_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        source_id = data.get("source_id")
        family_fingerprint = data.get("family_fingerprint")
        reason = data.get("reason") or "manual"
        if source_id is not None and not isinstance(source_id, str):
            source_id = str(source_id)
        if family_fingerprint is not None and not isinstance(family_fingerprint, str):
            family_fingerprint = str(family_fingerprint)
        if source_id is not None and not source_id:
            source_id = None
        if family_fingerprint is not None and not family_fingerprint:
            family_fingerprint = None
        clean = TrainingCandidateInvalidation(
            source_id=source_id,
            family_fingerprint=family_fingerprint,
            reason=str(reason),
        ).to_dict()
        return {key: clean.get(key) for key in self._safe_invalidation_keys()}

    def _migrate_experiences(self) -> None:
        if not self.experiences_path.exists():
            return
        try:
            raw_lines = self.experiences_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        migrated: list[str] = []
        changed = False
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            clean = self._sanitize_experience_payload(payload)
            if clean != payload:
                changed = True
            migrated.append(json.dumps(clean, sort_keys=True))
        if changed:
            tmp = self.experiences_path.with_suffix(".tmp")
            tmp.write_text("\n".join(migrated) + ("\n" if migrated else ""), encoding="utf-8")
            tmp.replace(self.experiences_path)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def append(self, record: ExperienceRecord) -> ExperienceRecord:
        payload = self._sanitize_experience_payload(record.to_dict())
        event_id = payload.get("event_id")
        if event_id:
            existing = self._find_experience_by_event_id(str(event_id))
            if existing is not None:
                return ExperienceRecord.from_dict(existing)
        self._append_jsonl(self.experiences_path, payload)
        return ExperienceRecord.from_dict(payload)

    def _find_experience_by_event_id(self, event_id: str) -> dict[str, Any] | None:
        if not event_id or not self.experiences_path.exists():
            return None
        lines = self.experiences_path.read_text(encoding="utf-8").splitlines()
        for raw in reversed(lines):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict) and str(payload.get("event_id") or "") == event_id:
                return self._sanitize_experience_payload(payload)
        return None

    def load_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.experiences_path.exists():
            return []
        lines = self.experiences_path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for raw in reversed(lines):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            records.append(self._sanitize_experience_payload(payload))
            if len(records) >= limit:
                break
        return records

    def append_failure_lesson(self, lesson: FailureLesson) -> FailureLesson:
        payload = lesson.to_dict()
        self._append_jsonl(self.failure_lessons_path, payload)
        return FailureLesson.from_dict(payload)

    def load_failure_lessons(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._load_jsonl(self.failure_lessons_path, limit, FailureLesson.from_dict)

    def upsert_plan_template(self, template: PlanTemplate) -> PlanTemplate:
        existing = self.load_plan_templates(limit=None)
        updated: dict[str, dict[str, Any]] = {}
        for item in existing:
            if item.get("id") and item["id"] != template.id:
                updated[item["id"]] = item
        payload = template.to_dict()
        updated[template.id] = payload
        tmp = self.plan_templates_path.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(item, sort_keys=True) for item in updated.values()) + "\n", encoding="utf-8")
        tmp.replace(self.plan_templates_path)
        return PlanTemplate.from_dict(payload)

    def load_plan_templates(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._load_jsonl(self.plan_templates_path, limit, PlanTemplate.from_dict)

    def append_correction(self, correction: CorrectionRecord) -> CorrectionRecord:
        payload = correction.to_dict()
        self._append_jsonl(self.corrections_path, payload)
        return CorrectionRecord.from_dict(payload)

    def load_corrections(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._load_jsonl(self.corrections_path, limit, CorrectionRecord.from_dict)

    def append_candidate_strategy(self, strategy: CandidateStrategy) -> CandidateStrategy:
        payload = strategy.to_dict()
        self._append_jsonl(self.candidate_strategies_path, payload)
        return CandidateStrategy.from_dict(payload)

    def load_candidate_strategies(self, limit: int = 50) -> list[dict[str, Any]]:
        records = self._load_jsonl(self.candidate_strategies_path, limit=None, factory=CandidateStrategy.from_dict)
        latest: dict[str, dict[str, Any]] = {}
        for record in records:
            strategy_id = str(record.get("strategy_id") or "")
            if strategy_id and strategy_id not in latest:
                latest[strategy_id] = record
        values = list(latest.values())
        values.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return values[:limit]

    def append_training_invalidation(self, invalidation: TrainingCandidateInvalidation | dict[str, Any]) -> dict[str, Any]:
        payload = invalidation.to_dict() if isinstance(invalidation, TrainingCandidateInvalidation) else dict(invalidation)
        clean = self._sanitize_training_invalidation_payload(payload)
        self._append_jsonl(self.training_invalidations_path, clean)
        return clean

    def invalidate_training_candidate(
        self,
        *,
        source_id: str | None = None,
        family_fingerprint: str | None = None,
        reason: str = "manual",
    ) -> dict[str, Any]:
        return self.append_training_invalidation(
            {
                "source_id": source_id,
                "family_fingerprint": family_fingerprint,
                "reason": reason,
            }
        )

    def load_training_invalidations(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._load_jsonl(self.training_invalidations_path, limit, TrainingCandidateInvalidation.from_dict)

    def _load_jsonl(
        self,
        path: Path,
        limit: int | None,
        factory: Callable[[dict[str, Any]], Any],
    ) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for raw in reversed(lines):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            try:
                record = factory(payload).to_dict()
            except Exception:
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        return records

    def load_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {"schema_version": 1, "skills": {}, "candidate_strategies": {}}
        try:
            payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema_version": 1, "skills": {}, "candidate_strategies": {}}
        if not isinstance(payload, dict):
            return {"schema_version": 1, "skills": {}, "candidate_strategies": {}}
        payload.setdefault("schema_version", 1)
        payload.setdefault("skills", {})
        payload.setdefault("candidate_strategies", {})
        return payload

    def save_summary(self, summary: dict[str, Any]) -> None:
        tmp = self.summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.summary_path)

    def update_skill_state(self, skill_id: str, before: SkillState | None, after: SkillState | None) -> None:
        summary = self.load_summary()
        skills = summary.setdefault("skills", {})
        if after is None:
            return
        skills[skill_id] = {
            **after.to_dict(),
            "previous_state": before.state if before is not None else None,
        }
        self.save_summary(summary)

    def update_candidate_strategy(self, strategy: CandidateStrategy) -> None:
        summary = self.load_summary()
        strategies = summary.setdefault("candidate_strategies", {})
        strategies[strategy.strategy_id] = strategy.to_dict()
        self.save_summary(summary)
