import pandas as pd

from common.statistics.service import calculate_data_quality_score


def test_clean_dataset_scores_100():
    df = pd.DataFrame({"id": [1, 2, 3], "date": ["2026-01-01", "2026-01-02", "2026-01-03"]})
    result = calculate_data_quality_score(df)
    assert result["quality_score"] == 100.0
    assert result["quality_grade"] == "Excellent"


def test_quality_score_accounts_for_missing_duplicates_invalid_dates_and_empty_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
            "date": ["2026-01-01", "bad-date", "bad-date", "2026-01-04"],
            "amount": [100, 200, 200, 200],
            "empty": [None, None, None, None],
        }
    )
    result = calculate_data_quality_score(df)
    assert 0 <= result["quality_score"] <= 100
    assert result["quality_grade"] in {"Excellent", "Good", "Fair", "Poor", "Critical"}
    assert "missing" in result["quality_summary"].lower()
    assert "duplicate" in result["quality_summary"].lower()
    assert "invalid date" in result["quality_summary"].lower()
    assert "empty column" in result["quality_summary"].lower()


def test_missing_blank_strings_are_counted():
    df = pd.DataFrame({"name": ["Alice", "  ", None]})
    result = calculate_data_quality_score(df)
    assert result["quality_score"] < 100
    assert "missing" in result["quality_summary"].lower()


def test_empty_dataset_is_safe():
    result = calculate_data_quality_score(pd.DataFrame())
    assert result == {
        "quality_score": 100.0,
        "quality_grade": "Excellent",
        "quality_summary": "Dataset is empty, so no quality issues were detected.",
    }
