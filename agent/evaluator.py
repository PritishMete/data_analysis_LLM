from __future__ import annotations


class ResultEvaluator:
    """Produces a privacy-safe quality score from a response summary."""

    def score(self, *, success: bool, result_summary: dict | None = None, feedback_score: int | None = None) -> float:
        if not success:
            base = 0.0
        else:
            base = 0.75
            if result_summary:
                row_count = result_summary.get("row_count")
                if isinstance(row_count, int):
                    base += 0.05 if row_count > 0 else -0.05
                if result_summary.get("result_kind") == "table":
                    base += 0.05
        if feedback_score is not None:
            if feedback_score > 0:
                base += 0.1
            elif feedback_score < 0:
                base -= 0.15
        return round(max(0.0, min(1.0, base)), 4)
