from pathlib import Path

from kaggle import semantic_generation_diagnostic as diagnostic


SOURCE = Path(diagnostic.__file__).read_text(encoding="utf-8")


def test_diagnostic_uses_only_safe_synthetic_rows_and_fixed_budgets():
    rows = diagnostic._synthetic_rows()
    assert len(rows) == 4
    assert diagnostic.DIAGNOSTIC_BUDGETS == (192, 256, 384)
    assert all("query_text" not in row["input"] and "workbook" not in row["input"] for row in rows)
    assert all("tool_graph" not in row["output"] and "sql" not in row["output"] for row in rows)


def test_diagnostic_has_no_training_operations_or_test_access():
    assert "torch.optim" not in SOURCE
    assert ".backward(" not in SOURCE
    assert "test.jsonl" not in SOURCE
    assert '"training_performed": False' in SOURCE
    assert '"test_split_accessed": False' in SOURCE


def test_template_and_eos_audits_are_structural_only():
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(max(1, len(text) // 8)))}

    row = diagnostic._synthetic_rows()[0]
    template = diagnostic._template_audit(Tokenizer(), row)
    eos = diagnostic._target_eos_audit(Tokenizer(), row)
    assert template["training_inference_prefix_match"] is True
    assert template["assistant_boundary"] == template["user_range"][1]
    assert eos["target_eos_present"] is True
    assert eos["target_eos_supervised"] is True


def test_runner_exposes_generation_diagnostic_command():
    from scripts import kaggle_runner

    assert "qwen-semantic-generation-diagnostic-cycle" in kaggle_runner.build_parser().format_help()
    assert kaggle_runner.qwen_semantic_generation_diagnostic_cycle.__name__ == "qwen_semantic_generation_diagnostic_cycle"
