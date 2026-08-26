from __future__ import annotations


PROMOTION_THRESHOLDS = {
    "candidate_successes": 2,
    "validated_successes": 4,
    "validated_quality": 0.85,
    "trusted_successes": 7,
    "trusted_quality": 0.90,
}


EVALUATION_WEIGHTS = {
    "plan_completeness": 0.25,
    "result_validation": 0.25,
    "semantic_binding": 0.15,
    "critic": 0.15,
    "execution": 0.10,
    "output_contract": 0.05,
    "feedback": 0.05,
}


MAX_REPAIR_STEPS = 2
