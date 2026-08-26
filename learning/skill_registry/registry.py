from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import Any

from learning.bootstrap.skills import bootstrap_skill_specs
from learning.config import PROMOTION_THRESHOLDS
from learning.models import QueryFeatures, SkillMatch, SkillSpec, SkillState


def _default_state_dir() -> Path:
    root = os.environ.get("DATA_ANALYSIS_LLM_STATE_DIR")
    if root:
        return Path(root)
    return Path.home() / ".data_analysis_llm"


@dataclass(slots=True)
class RegistrySnapshot:
    specs: dict[str, SkillSpec]
    states: dict[str, SkillState]


class SkillRegistry:
    def __init__(self, state_path: Path | None = None):
        self.state_path = state_path or (_default_state_dir() / "skills_state.json")
        self._specs = {spec.id: spec for spec in bootstrap_skill_specs()}
        self._dynamic_specs: dict[str, SkillSpec] = {}
        self._states = self._load_states()
        self._load_dynamic_specs()
        for skill_id, spec in self._specs.items():
            self._states.setdefault(skill_id, SkillState(skill_id=skill_id, confidence=spec.confidence))

    def snapshot(self) -> RegistrySnapshot:
        return RegistrySnapshot(specs=dict(self._specs), states=dict(self._states))

    def all(self) -> list[SkillSpec]:
        return list(self._specs.values()) + list(self._dynamic_specs.values())

    def get(self, skill_id: str) -> SkillSpec | None:
        return self._specs.get(skill_id) or self._dynamic_specs.get(skill_id)

    def state_for(self, skill_id: str) -> SkillState:
        spec = self.get(skill_id)
        return self._states.setdefault(skill_id, SkillState(skill_id=skill_id, confidence=spec.confidence if spec else 0.5))

    def effective_confidence(self, skill_id: str) -> float:
        spec = self.get(skill_id)
        if spec is None:
            return 0.0
        state = self.state_for(skill_id)
        return round(min(0.99, max(0.0, (spec.confidence * 0.7) + (state.confidence * 0.3))), 4)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "skills": {skill_id: state.to_dict() for skill_id, state in self._states.items()},
            "dynamic_skills": {skill_id: spec.to_dict() for skill_id, spec in self._dynamic_specs.items()},
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load_states(self) -> dict[str, SkillState]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw = payload.get("skills") if isinstance(payload, dict) else {}
        if not isinstance(raw, dict):
            return {}
        states: dict[str, SkillState] = {}
        for skill_id, state_payload in raw.items():
            if isinstance(state_payload, dict):
                try:
                    states[skill_id] = SkillState.from_dict({"skill_id": skill_id, **state_payload})
                except Exception:
                    continue
        return states

    def _load_dynamic_specs(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        raw = payload.get("dynamic_skills") if isinstance(payload, dict) else {}
        if not isinstance(raw, dict):
            return
        for skill_id, spec_payload in raw.items():
            if isinstance(spec_payload, dict):
                try:
                    spec = SkillSpec(**spec_payload)
                except Exception:
                    continue
                self._dynamic_specs[skill_id] = spec
                self._states.setdefault(skill_id, SkillState(skill_id=skill_id, confidence=spec.confidence, state=spec.lifecycle or "candidate"))

    def match(self, features: QueryFeatures) -> list[SkillMatch]:
        from learning.retriever import score_skill

        matches: list[SkillMatch] = []
        for spec in self._specs.values():
            score, reasons = score_skill(spec, features)
            if score <= 0:
                continue
            matches.append(SkillMatch(spec=spec, score=score, reasons=reasons))
        matches.sort(key=lambda item: (item.score, self.effective_confidence(item.spec.id)), reverse=True)
        return matches

    def register_dynamic_skill(self, spec: SkillSpec) -> None:
        self._dynamic_specs[spec.id] = spec
        self._states.setdefault(
            spec.id,
            SkillState(skill_id=spec.id, confidence=spec.confidence, state=spec.lifecycle or "candidate"),
        )
        self.save()

    def update_from_experience(self, skill_id: str | None, *, success: bool, score: float, now_iso: str | None = None) -> tuple[SkillState | None, SkillState | None]:
        if not skill_id or self.get(skill_id) is None:
            return None, None

        state_before = SkillState.from_dict(self.state_for(skill_id).to_dict())
        state = self.state_for(skill_id)
        if now_iso is not None:
            state.last_seen_at = now_iso
        state.confidence = self._adjust_confidence(state.confidence, success=success, score=score)
        if success:
            state.success_count += 1
        else:
            state.failure_count += 1
        total = state.success_count + state.failure_count
        if total:
            observed = score if score > 0 else (1.0 if success else 0.0)
            if total == 1:
                state.average_quality_score = observed
            else:
                state.average_quality_score = round(
                    ((state.average_quality_score * (total - 1)) + observed) / total,
                    4,
                )
        state.state = self._state_label(state)
        if state.state == "promoted" and state.promoted_at is None:
            state.promoted_at = now_iso
        if state.state in {"candidate", "promoted"}:
            state.candidate_promotions += 1
        self.save()
        return state_before, SkillState.from_dict(state.to_dict())

    @staticmethod
    def _adjust_confidence(confidence: float, *, success: bool, score: float) -> float:
        delta = 0.04 if success else -0.06
        delta += (score - 0.5) * 0.08 if score is not None else 0.0
        return round(min(0.99, max(0.05, confidence + delta)), 4)

    @staticmethod
    def _state_label(state: SkillState) -> str:
        if state.failure_count >= 3 and state.failure_count > state.success_count:
            return "demoted"
        if state.success_count >= PROMOTION_THRESHOLDS["trusted_successes"] and state.average_quality_score >= PROMOTION_THRESHOLDS["trusted_quality"]:
            return "trusted"
        if state.success_count >= PROMOTION_THRESHOLDS["validated_successes"] and state.average_quality_score >= PROMOTION_THRESHOLDS["validated_quality"]:
            return "validated"
        if state.success_count >= PROMOTION_THRESHOLDS["candidate_successes"]:
            return "candidate"
        return "bootstrap"


_REGISTRY: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SkillRegistry()
    return _REGISTRY
