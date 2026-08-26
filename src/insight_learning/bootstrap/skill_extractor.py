from __future__ import annotations

from learning.bootstrap.skills import bootstrap_skill_specs


def extract_bootstrap_skills() -> list[dict]:
    return [spec.to_dict() for spec in bootstrap_skill_specs()]

