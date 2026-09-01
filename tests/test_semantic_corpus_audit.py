from pathlib import Path

from learning.semantic_corpus_audit import audit_dataset, classify_learning_failure


def test_latest_local_corpus_audits_all_train_and_validation_rows_without_test_access():
    result = audit_dataset(Path("runtime/training"))
    assert result["train_total"] == 407
    assert result["validation_total"] == 47
    assert result["test_total_from_metadata"] == 46
    assert result["train"]["structurally_readable"] == 407
    assert result["train"]["eligible"] == 407
    assert result["train"]["privacy_valid"] == 407
    assert result["train"]["conversion_success"] == 407
    assert result["train"]["target_valid"] == 407
    assert result["train"]["usable"] == 407
    assert result["validation"]["usable"] == 47
    assert result["test_split_accessed"] is False
    assert result["recommended_next_train_size"] == 128
    assert result["recommended_next_validation_size"] == 16
    assert result["classification"] == "SEMANTIC_SUBSET_SELECTION_BUG"


def test_pretraining_subset_failure_cannot_be_called_model_not_learning():
    assert classify_learning_failure(stage="dataset", training_completed=False) != "MODEL_NOT_LEARNING"
    assert classify_learning_failure(stage="subset_selection", training_completed=False) == "INSUFFICIENT_SEMANTIC_TRAINING_EXAMPLES"
    assert classify_learning_failure(stage="evaluation", training_completed=True) == "MODEL_NOT_LEARNING"


def test_audit_is_test_sealed_and_safe():
    source = Path("learning/semantic_corpus_audit.py").read_text(encoding="utf-8")
    assert 'raise RuntimeError("TEST_SPLIT_ACCESS_FORBIDDEN")' in source
    assert 'root / "test.jsonl"' not in source
    assert "rejection_reasons" in source
    assert "usable_intent_coverage" in source
    assert "structure_coverage" in source
