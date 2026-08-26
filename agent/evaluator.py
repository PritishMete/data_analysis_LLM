from __future__ import annotations


class ResultEvaluator:
    """Produces a privacy-safe quality score from a response summary."""

    def score(self, *, success: bool, result_summary: dict | None = None, feedback_score: int | None = None) -> float:
        if not success:
            base = 0.0
        else:
            base = 0.72
            summary = result_summary or {}
            row_count = summary.get("row_count")
            if isinstance(row_count, int):
                if row_count > 0:
                    base += 0.08
                elif row_count == 0:
                    base -= 0.04
            column_count = summary.get("column_count")
            if isinstance(column_count, int) and column_count > 0:
                base += 0.03
            if summary.get("result_kind") == "table":
                base += 0.05
            if summary.get("result_kind") == "operation":
                base += 0.04
        if feedback_score is not None:
            if feedback_score > 0:
                base += 0.12
            elif feedback_score < 0:
                base -= 0.18
        return round(max(0.0, min(1.0, base)), 4)
