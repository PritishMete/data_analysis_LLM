from __future__ import annotations

from pathlib import Path

from training.benchmark import LlamaCppPlannerAdapter, run_planner_benchmark, write_benchmark_report
from training.cli import main
from training.profiles import PLANNER_BACKEND_AUTO, PLANNER_BACKEND_LLAMA_CPP, PLANNER_BACKEND_TRANSFORMERS


def test_inference_benchmark_cli_and_report(tmp_path, monkeypatch):
    report_dir = tmp_path / "benchmark"
    assert main([
        "inference-benchmark",
        "--profile",
        "low_spec",
        "--backend",
        "llama_cpp",
        "--device",
        "cpu",
        "--benchmark",
        "builtin",
        "--output-dir",
        str(report_dir),
    ]) == 0
    path = report_dir / "low_spec_inference_benchmark.json"
    assert path.exists()


def test_backend_selection_and_adapter_fallback(monkeypatch):
    adapter = LlamaCppPlannerAdapter(profile_name="low_spec")
    health = adapter.health()
    assert "available" in health
    assert adapter.metadata()["backend"] == "llama_cpp"

    summary = run_planner_benchmark(profile_name="low_spec", backend=PLANNER_BACKEND_LLAMA_CPP, device="cpu")
    assert summary.profile["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "valid_json_rate" in summary.metrics
    assert 0.0 <= summary.metrics["valid_json_rate"] <= 1.0
    assert summary.backend in {PLANNER_BACKEND_TRANSFORMERS, PLANNER_BACKEND_LLAMA_CPP, "heuristic"}


def test_shadow_mode_and_critic_metrics_are_recorded():
    summary = run_planner_benchmark(profile_name="low_spec", backend=PLANNER_BACKEND_LLAMA_CPP, device="cpu")
    assert summary.model_health["backend"] in {"heuristic", "transformers", "llama_cpp"}
    assert "tool_selection_f1" in summary.metrics
    assert "fallback_accuracy" in summary.metrics
    assert summary.cases
