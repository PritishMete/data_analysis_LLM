from __future__ import annotations

from learning.config import EVALUATION_WEIGHTS


class ResultEvaluator:
    """Produces a privacy-safe quality score from a response summary."""

    def score(
        self,
        *,
        success: bool,
        result_summary: dict | None = None,
        feedback_score: int | None = None,
        plan_completeness: float | None = None,
        result_validation: float | None = None,
        semantic_binding: float | None = None,
        critic_score: float | None = None,
        execution_score: float | None = None,
        output_contract: float | None = None,
        repair_penalty: float | None = None,
        fallback_penalty: float | None = None,
        ambiguity_penalty: float | None = None,
    ) -> float:
        if not success:
            return 0.0

        summary = result_summary or {}
        row_count = summary.get("row_count")
        column_count = summary.get("column_count")

        plan_score = plan_completeness if plan_completeness is not None else (1.0 if summary else 0.5)
        validation_score = result_validation if result_validation is not None else (1.0 if summary else 0.5)
        binding_score = semantic_binding if semantic_binding is not None else (0.8 if success else 0.5)
        critic_component = critic_score if critic_score is not None else (0.9 if success else 0.5)
        execution_component = execution_score if execution_score is not None else 1.0
        output_component = output_contract if output_contract is not None else (0.8 if success else 0.5)
        feedback_component = 0.5
        if feedback_score is not None:
            feedback_component = 0.8 if feedback_score > 0 else (0.2 if feedback_score < 0 else 0.5)

        score = (
            plan_score * EVALUATION_WEIGHTS["plan_completeness"]
            + validation_score * EVALUATION_WEIGHTS["result_validation"]
            + binding_score * EVALUATION_WEIGHTS["semantic_binding"]
            + critic_component * EVALUATION_WEIGHTS["critic"]
            + execution_component * EVALUATION_WEIGHTS["execution"]
            + output_component * EVALUATION_WEIGHTS["output_contract"]
            + feedback_component * EVALUATION_WEIGHTS["feedback"]
        )

        if isinstance(row_count, int):
            if row_count > 0:
                score += 0.03
            elif row_count == 0:
                score -= 0.05
        if isinstance(column_count, int) and column_count > 0:
            score += 0.02

        for penalty in (repair_penalty, fallback_penalty, ambiguity_penalty):
            if penalty is not None:
                score -= abs(penalty)

        return round(max(0.0, min(1.0, score)), 4)
