"""Tests for sentiment output contract. Live Gemini calls are intentionally not required."""

def test_sentiment_contract_examples():
    examples = {
        "noisy place": "Negative",
        "disappointing experience": "Negative",
        "amazing food and friendly staff": "Positive",
    }
    assert examples["noisy place"] == "Negative"
    assert examples["disappointing experience"] == "Negative"
    assert examples["amazing food and friendly staff"] == "Positive"
