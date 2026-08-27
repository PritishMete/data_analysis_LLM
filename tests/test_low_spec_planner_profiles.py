from __future__ import annotations

from pathlib import Path

from agent.planner import LearningPlanner
from learning.models import SkillState
from training.execution import preflight_gpu_training, select_model_and_runtime_profile
from training.hardware import HardwareReport
from training.profiles import (
    CPU_LOW_SPEC_RUNTIME,
    GPU_4GB_RUNTIME,
    LOW_SPEC_MODEL_PROFILE,
    STANDARD_MODEL_PROFILE,
    choose_backend,
    select_model_profile,
    select_runtime_profile,
)


def test_model_profile_selection():
    low_spec = select_model_profile("low_spec")
    standard = select_model_profile("standard")

    assert low_spec.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert low_spec.parameter_count == "0.5B"
    assert standard.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert standard.parameter_count == "1.5B"
    assert LOW_SPEC_MODEL_PROFILE.training_min_vram_gb == 8.0
    assert STANDARD_MODEL_PROFILE.training_min_vram_gb == 12.0


def test_runtime_profile_selection_and_backend_priority():
    cpu = select_runtime_profile("cpu_low_spec")
    gpu = select_runtime_profile("gpu_4gb")

    assert cpu.profile_name == CPU_LOW_SPEC_RUNTIME.profile_name
    assert gpu.profile_name == GPU_4GB_RUNTIME.profile_name
    assert choose_backend(backend="auto", runtime_profile=cpu, cuda_available=False, llama_cpp_available=True) == "llama_cpp"
    assert choose_backend(backend="auto", runtime_profile=gpu, cuda_available=True, llama_cpp_available=False) == "transformers"


def test_profile_metadata_for_existing_execution_path():
    payload = select_model_and_runtime_profile(
        planner_profile="low_spec",
        runtime_profile="gpu_4gb",
        backend="auto",
        cuda_available=False,
        llama_cpp_available=False,
    )
    assert payload["planner_profile"]["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert payload["runtime_profile"]["profile_name"] == "gpu_4gb"
    assert payload["backend"] == "transformers"


def test_preflight_uses_profile_specific_training_minimum(tmp_path, monkeypatch):
    from training import execution as execution_module

    monkeypatch.setattr(execution_module, "detect_hardware", lambda: HardwareReport(
        python_version="3.12",
        platform="Linux",
        machine="x86_64",
        processor="x86_64",
        ram_gb=16.0,
        torch_version="2.6.0+cu124",
        cuda_available=True,
        cuda_version="12.4",
        gpu_name="NVIDIA GeForce GTX 1650",
        vram_gb=4.0,
    ))
    monkeypatch.setattr(execution_module, "validate_dataset", lambda _: type("R", (), {"ready_for_prototype": True, "blockers": [], "to_dict": lambda self: {"ready_for_prototype": True}})())
    monkeypatch.setattr(execution_module, "verify_dataset_manifest", lambda *args, **kwargs: {"verified": True, "mismatches": [], "manifest_path": "x", "dataset_version": "v1"})

    result = preflight_gpu_training(
        dataset_dir=Path("runtime") / "training",
        output_dir=tmp_path / "models",
        planner_profile="low_spec",
        minimum_vram_gb=4.0,
    )
    assert result.ready is False
    assert "vram_below_threshold" in result.blockers
    assert result.hardware["planner_profile"] == "low_spec"
    assert result.hardware["training_min_vram_gb"] == 8.0


def test_trusted_strategy_priority_path():
    class DummyRegistry:
        def get(self, skill_id: str):
            return object()

        def state_for(self, skill_id: str):
            return SkillState(
                skill_id=skill_id,
                confidence=0.99,
                success_count=10,
                failure_count=0,
                average_quality_score=0.98,
                state="trusted",
            )

    planner = LearningPlanner(registry=DummyRegistry())
    assert planner._plan_source_for_skill("learned.analytics.strategy") == "trusted_strategy"
