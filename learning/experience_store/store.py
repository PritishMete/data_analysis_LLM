from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any

from learning.models import ExperienceRecord, SkillState


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
        self.summary_path = self.root / "experience_state.json"

    def append(self, record: ExperienceRecord) -> ExperienceRecord:
        self.experiences_path.parent.mkdir(parents=True, exist_ok=True)
        with self.experiences_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return record

    def load_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.experiences_path.exists():
            return []
        lines = self.experiences_path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for raw in reversed(lines):
            if not raw.strip():
                continue
            try:
                records.append(json.loads(raw))
            except Exception:
                continue
            if len(records) >= limit:
                break
        return records

    def load_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {"schema_version": 1, "skills": {}}
        try:
            payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {"schema_version": 1, "skills": {}}
        if not isinstance(payload, dict):
            return {"schema_version": 1, "skills": {}}
        payload.setdefault("schema_version", 1)
        payload.setdefault("skills", {})
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
