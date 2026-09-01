from pathlib import Path

from kaggle import qwen_nf4_load_cycle


def test_qwen_cycle_uses_exact_model_and_nf4_settings():
    source = Path(qwen_nf4_load_cycle.__file__).read_text(encoding="utf-8")
    assert qwen_nf4_load_cycle.MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "load_in_4bit=True" in source
    assert 'bnb_4bit_quant_type="nf4"' in source
    assert "bnb_4bit_use_double_quant=True" in source
    assert "torch.float16" in source
    assert "transformers==4.46.3" in source
    assert "tokenizers==0.20.3" in source
    assert "huggingface_hub==0.26.2" in source


def test_qwen_cycle_is_training_free_and_peft_free():
    source = Path(qwen_nf4_load_cycle.__file__).read_text(encoding="utf-8")
    assert "import peft" not in source
    assert "optimizer" not in source
    assert "backward(" not in source
    assert "train.jsonl" not in source
    assert "validation.jsonl" not in source
    assert "test.jsonl" not in source


def test_qwen_cycle_requires_bnb_nf4_gate_before_model_load():
    source = Path(qwen_nf4_load_cycle.__file__).read_text(encoding="utf-8")
    gate = 'if bnb_report.get("verdict") != "BNB_NF4_P100_RUNTIME_PASSED":'
    assert gate in source
    assert source.index(gate) < source.index("AutoTokenizer.from_pretrained")
    assert source.index("run_bnb_compat_cycle") < source.index("_install_missing_dependencies")


def test_qwen_cycle_emits_memory_and_model_markers():
    source = Path(qwen_nf4_load_cycle.__file__).read_text(encoding="utf-8")
    for marker in (
        "MODEL_DEPENDENCY_RESULT_JSON",
        "TOKENIZER_RESULT_JSON",
        "MODEL_LOAD_RESULT_JSON",
        "MODEL_DEVICE_RESULT_JSON",
        "MODEL_MEMORY_RESULT_JSON",
        "MODEL_FORWARD_RESULT_JSON",
        "MODEL_FINAL_RESULT_JSON",
    ):
        assert marker in source
