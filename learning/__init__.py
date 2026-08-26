"""Self-learning analytics primitives.

The orchestration layer lives in `agent/`. This package intentionally stays
light so importing any of the storage or model modules does not pull the
full request-time planner into memory.
"""

from .bootstrap.skills import bootstrap_skill_specs
from .models import ExperienceRecord, LearningDecision, QueryFeatures, SkillMatch, SkillSpec, SkillState

__all__ = [
    "bootstrap_skill_specs",
    "ExperienceRecord",
    "LearningDecision",
    "QueryFeatures",
    "SkillMatch",
    "SkillSpec",
    "SkillState",
]
