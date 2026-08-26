"""Self-learning analytics primitives.

The orchestration layer lives in `agent/`. This package intentionally stays
light so importing any of the storage or model modules does not pull the
full request-time planner into memory.
"""

from .bootstrap.skills import bootstrap_skill_specs
from .models import (
    AgentPlan,
    BoundPlan,
    CandidateStrategy,
    CorrectionRecord,
    DatasetSemanticProfile,
    ExperienceRecord,
    FailureLesson,
    LogicalGroup,
    LearningDecision,
    PlanTemplate,
    PredicateNode,
    PlannerContext,
    QueryFeatures,
    SkillMatch,
    SkillSpec,
    SkillState,
    PlanStep,
)
from .canonical_training import (
    CANONICAL_CORPUS_VERSION,
    DEFAULT_MINIMUM_QUALITY,
    FineTuningCandidate,
    PlannerModelVersion,
    PlannerTrainingBackend,
    StructuralFamily,
    TrainingBenchmarkResult,
    TrainingCandidateInvalidation,
    TrainingDatasetManifest,
    TrainingReadinessAssessment,
    build_training_dataset_manifest,
    evaluate_training_readiness,
)

__all__ = [
    "bootstrap_skill_specs",
    "BoundPlan",
    "ExperienceRecord",
    "AgentPlan",
    "CandidateStrategy",
    "CorrectionRecord",
    "DatasetSemanticProfile",
    "FailureLesson",
    "LearningDecision",
    "LogicalGroup",
    "PlanTemplate",
    "PredicateNode",
    "PlannerContext",
    "QueryFeatures",
    "SkillMatch",
    "SkillSpec",
    "SkillState",
    "PlanStep",
    "CANONICAL_CORPUS_VERSION",
    "DEFAULT_MINIMUM_QUALITY",
    "FineTuningCandidate",
    "PlannerModelVersion",
    "PlannerTrainingBackend",
    "StructuralFamily",
    "TrainingBenchmarkResult",
    "TrainingCandidateInvalidation",
    "TrainingDatasetManifest",
    "TrainingReadinessAssessment",
    "build_training_dataset_manifest",
    "evaluate_training_readiness",
]
