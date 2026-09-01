from pathlib import Path
from types import SimpleNamespace

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


def test_dependency_install_preserves_package_failure_and_safe_tails(monkeypatch):
    def fake_run(*args, **kwargs):
        assert kwargs["timeout"] == diagnostic.DEPENDENCY_TIMEOUT_SECONDS
        return SimpleNamespace(
            returncode=1,
            stdout="normal output\naccess_token=do-not-leak",
            stderr="pip failed\nsecret=do-not-leak",
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    result = diagnostic._run_dependency_install("transformers", "transformers==4.46.3")
    assert result["classification"] == "TRANSFORMERS_INSTALL_FAILED"
    assert result["returncode"] == 1
    assert "do-not-leak" not in (result["stdout_tail"] or "")
    assert "do-not-leak" not in (result["stderr_tail"] or "")
    assert result["command"][0] == "<python>"


def test_dependency_version_drift_is_explicit(monkeypatch):
    versions = {name: expected for name, expected in diagnostic.EXPECTED_DEPENDENCY_VERSIONS.items()}
    versions["peft"] = "0.0.0"
    monkeypatch.setattr(diagnostic, "_version", lambda name: versions.get(name))
    result = diagnostic._install_generation_dependencies(import_checker=lambda name: object())
    assert result["ok"] is False
    assert result["classification"] == "DEPENDENCY_VERSION_MISMATCH"
    assert result["versions"]["peft"] == "0.0.0"


def test_dependency_specs_are_exact_and_no_deps():
    assert all("==" in requirement for _, requirement in diagnostic.GENERATION_DEPENDENCIES)
    assert diagnostic.EXPECTED_DEPENDENCY_VERSIONS["accelerate"] == "1.13.0"
    assert diagnostic.EXPECTED_DEPENDENCY_VERSIONS["peft"] == "0.13.2"
    assert "--no-deps" in Path(diagnostic.__file__).read_text(encoding="utf-8")


def test_failure_redacts_credentials():
    safe = diagnostic._redact_safe_text("token=abc secret: xyz authorization=Bearer-value")
    assert "abc" not in safe and "xyz" not in safe and "Bearer-value" not in safe
