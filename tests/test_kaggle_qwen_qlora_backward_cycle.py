from pathlib import Path

from kaggle import qwen_qlora_backward_cycle as cycle


SOURCE = Path(cycle.__file__).read_text(encoding="utf-8")


def test_backward_cycle_preserves_proven_runtime_and_pins_peft():
    assert cycle.MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cycle.PEFT_SPEC == "peft==0.13.2"
    assert "run_qwen_nf4_load_cycle" in SOURCE
    assert "BNB_NF4_P100_RUNTIME_PASSED" in SOURCE
    assert "QWEN_0_5B_NF4_P100_RUNTIME_PASSED" in SOURCE
    assert "--no-deps" in SOURCE
    assert "torch" in SOURCE and "bitsandbytes" in SOURCE


def test_lora_configuration_is_explicit_and_targets_qwen_projections():
    assert "r=16" in SOURCE
    assert "lora_alpha=32" in SOURCE
    assert "lora_dropout=0.05" in SOURCE
    assert 'bias="none"' in SOURCE
    assert 'task_type="CAUSAL_LM"' in SOURCE
    for target in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert f'"{target}"' in SOURCE
    assert "missing_targets" in SOURCE
    assert "LORA_TARGET_MODULE_NOT_FOUND" in SOURCE


def test_backward_cycle_proves_trainable_lora_and_frozen_base():
    assert '"trainable_parameter_names_are_lora"' in SOURCE
    assert '"base_model_frozen"' in SOURCE
    assert "NO_TRAINABLE_LORA_PARAMETERS" in SOURCE
    assert "BASE_MODEL_UNEXPECTEDLY_TRAINABLE" in SOURCE
    assert "trainable_parameters" in SOURCE


def test_smoke_is_synthetic_short_and_does_not_export_artifacts():
    assert cycle.MAX_SMOKE_SEQUENCE_LENGTH <= 128
    assert "_synthetic_example" in SOURCE
    assert '"train_data_used": False' in SOURCE
    assert '"validation_data_used": False' in SOURCE
    assert '"test_data_used": False' in SOURCE
    assert "save_pretrained" not in SOURCE
    assert "save_pretrained" not in SOURCE
    assert "torch.save" not in SOURCE
    assert "Trainer(" not in SOURCE
    assert "adapter_model" not in SOURCE


def test_exactly_one_backward_and_optimizer_step_are_required():
    assert SOURCE.count("loss.backward()") == 1
    assert SOURCE.count("optimizer.step()") == 1
    assert '"optimizer_step_count": 1' in SOURCE
    assert "LORA_GRADIENT_MISSING" in SOURCE
    assert "LORA_GRADIENT_NONFINITE" in SOURCE
    assert "LORA_GRADIENT_ZERO" in SOURCE
    assert "LORA_PARAMETER_UNCHANGED" in SOURCE


def test_required_observability_markers_exist():
    for marker in (
        "PEFT_DEPENDENCY_RESULT_JSON",
        "KBIT_PREPARATION_RESULT_JSON",
        "LORA_ATTACHMENT_RESULT_JSON",
        "LORA_PARAMETER_RESULT_JSON",
        "QLORA_FORWARD_RESULT_JSON",
        "QLORA_BACKWARD_RESULT_JSON",
        "QLORA_OPTIMIZER_RESULT_JSON",
        "QLORA_MEMORY_RESULT_JSON",
        "QLORA_FINAL_RESULT_JSON",
    ):
        assert marker in SOURCE


def test_synthetic_example_enforces_sequence_limit():
    class FakeTensor:
        shape = (1, 7)

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": FakeTensor()}

    encoded, length = cycle._synthetic_example(FakeTokenizer())
    assert encoded["input_ids"].shape[-1] == 7
    assert length == 7
