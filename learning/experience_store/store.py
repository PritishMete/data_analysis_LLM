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
    SCHEMA_VERSION,
    SkillState,
    stable_hash,
)


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
        self.corrections_path = self.root / "corrections.jsonl"
        self.candidate_strategies_path = self.root / "candidate_strategies.jsonl"
        self.summary_path = self.root / "experience_state.json"
        self._migrate_experiences()

    @staticmethod
    def _safe_experience_keys() -> set[str]:
        return {
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
            "skill_state_before",
            "skill_state_after",
            "correction_type",
            "correction_summary",
            "candidate_strategy_id",
            "created_at",
            "version",
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
        self._append_jsonl(self.experiences_path, payload)
        return ExperienceRecord.from_dict(payload)

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
