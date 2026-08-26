# tests/test_analyze_dataframe.py
import pandas as pd

from main import analyze_dataframe

EXPECTED_TOP_LEVEL_KEYS = {
    "summary",
    "distribution",
    "quality",
    "duplicates",
    "missing_values",
    "numeric_statistics",
    "categorical_statistics",
}


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 4],
        "amount": [10.5, 20.0, None, 15.5, 15.5],
        "category": ["a", "b", "a", "a", "a"],  # last two rows are exact duplicates
    })


def test_returns_exactly_the_seven_grouped_keys():
    result = analyze_dataframe(_sample_df())
    assert set(result.keys()) == EXPECTED_TOP_LEVEL_KEYS


def test_summary_group_has_rows_columns_and_column_names():
    result = analyze_dataframe(_sample_df())
    assert result["summary"] == {"rows": 5, "columns": 3, "column_names": ["id", "amount", "category"]}


def test_distribution_group_has_unique_value_counts():
    result = analyze_dataframe(_sample_df())
    assert result["distribution"]["unique_values"]["category"] == 2
    assert result["distribution"]["unique_values"]["id"] == 4


def test_duplicates_group_matches_actual_duplicate_row_count():
    result = analyze_dataframe(_sample_df())
    assert result["duplicates"]["count"] == 1


def test_missing_values_group_matches_actual_missing_counts():
    result = analyze_dataframe(_sample_df())
    assert result["missing_values"]["amount"] == 1
    assert result["missing_values"]["id"] == 0


def test_quality_group_reuses_duplicates_and_missing_values_exactly():
    result = analyze_dataframe(_sample_df())
    assert result["quality"]["duplicate_rows"] == result["duplicates"]["count"]
    assert result["quality"]["missing_values"] == result["missing_values"]


def test_numeric_and_categorical_statistics_are_correctly_split():
    result = analyze_dataframe(_sample_df())

    assert "amount" in result["numeric_statistics"]
    assert "id" in result["numeric_statistics"]
    assert "category" not in result["numeric_statistics"]

    assert "category" in result["categorical_statistics"]
    assert "amount" not in result["categorical_statistics"]

    assert result["numeric_statistics"]["amount"]["mean"] == 15.375
    assert result["categorical_statistics"]["category"]["top"] == "a"
    assert result["categorical_statistics"]["category"]["freq"] == 3


def test_no_business_insight_or_recommendation_keys_present():
    result = analyze_dataframe(_sample_df())
    for forbidden in ("insight", "insights", "recommendation", "recommendations", "summary_text"):
        assert forbidden not in result
    assert isinstance(result["summary"], dict)  # "summary" here is structural, never narrative text


def test_no_raw_row_previews_or_free_text_in_output():
    result = analyze_dataframe(_sample_df())
    for legacy_key in ("preview", "sample", "info", "describe"):
        assert legacy_key not in result


def test_handles_empty_dataframe():
    result = analyze_dataframe(pd.DataFrame())
    assert result["summary"]["rows"] == 0
    assert result["summary"]["columns"] == 0
    assert result["numeric_statistics"] == {}
    assert result["categorical_statistics"] == {}


def test_handles_dataframe_with_only_numeric_columns():
    result = analyze_dataframe(pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}))
    assert result["categorical_statistics"] == {}
    assert "x" in result["numeric_statistics"]
    assert "y" in result["numeric_statistics"]


def test_handles_dataframe_with_only_categorical_columns():
    result = analyze_dataframe(pd.DataFrame({"label": ["a", "b", "c"]}))
    assert result["numeric_statistics"] == {}
    assert "label" in result["categorical_statistics"]
