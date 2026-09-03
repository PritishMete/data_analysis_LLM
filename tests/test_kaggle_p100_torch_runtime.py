from __future__ import annotations

from pathlib import Path

from kaggle import bnb_compat_cycle, torch_compat_cycle
from kaggle import bootstrap, bootstrap_environment
from kaggle import p100_torch_runtime


def test_torch_and_bnb_cycles_share_the_same_torch_bootstrap_helper():
    torch_source = Path(torch_compat_cycle.__file__).read_text(encoding="utf-8")
    bnb_source = Path(bnb_compat_cycle.__file__).read_text(encoding="utf-8")
    assert "run_shared_p100_torch_bootstrap" in torch_source
    assert "run_shared_p100_torch_bootstrap" in bnb_source
    assert "run_shared_p100_torch_validation" in bnb_source


def test_no_duplicate_torch_installer_implementation_remains():
    torch_source = Path(torch_compat_cycle.__file__).read_text(encoding="utf-8")
    bnb_source = Path(bnb_compat_cycle.__file__).read_text(encoding="utf-8")
    shared_source = Path(p100_torch_runtime.__file__).read_text(encoding="utf-8")

    assert "def _installer_snippet" not in torch_source
    assert "def _probe_snippet" not in torch_source
    assert "def _validate_runtime_snippet" not in torch_source
    assert "def _torch_install_snippet" not in torch_source
    assert "def _torch_probe_snippet" not in torch_source
    assert "def _torch_cuda_validation_snippet" not in torch_source
    assert "def _torch_install_snippet" not in bnb_source
    assert "def _torch_probe_snippet" not in bnb_source
    assert "def _torch_cuda_validation_snippet" not in bnb_source
    assert "def run_shared_p100_torch_bootstrap" in shared_source
    assert "def run_shared_p100_torch_validation" in shared_source


def test_skip_code_probe_failure_maps_to_torch_dynamo_binary_mismatch():
    assert p100_torch_runtime._classify_torch_probe_failure({"stdout": "skip_code missing"}) == "TORCH_DYNAMO_BINARY_MISMATCH"


def test_preinstall_inspection_is_non_terminal_for_incompatible_default_torch():
    payload = {"ok": True, "json": {"default_torch_appears_p100_incompatible": True, "inspect_only": True}}

    assert p100_torch_runtime._classify_preinstall_inspection(payload) is None


def test_pip_success_does_not_certify_an_incompatible_torch_runtime():
    runtime = {"ok": True, "json": {"torch_version": "2.10.0+cu128", "torch_cuda_version": "12.8", "compute_capability": [6, 0], "arch_list": ["sm_70"]}}
    cuda = {"ok": True, "json": {"basic_cuda_tensor_test": True}}

    assert p100_torch_runtime._torch_profile_failure(runtime, cuda) == "TORCH_VERSION_MISMATCH"


def test_generation_bootstrap_reuses_shared_p100_path_before_runtime_packages():
    source = Path(bootstrap_environment.__file__).read_text(encoding="utf-8")

    assert "run_shared_p100_torch_bootstrap" in source
    assert source.index("run_shared_p100_torch_bootstrap") < source.index("_install_packages")


def test_canonical_bnb_probe_has_subprocess_dependency():
    source = Path(bootstrap.__file__).read_text(encoding="utf-8")

    assert "import subprocess" in source
    assert "def _probe_bitsandbytes_runtime" in source


def test_bnb_install_snippet_uses_no_deps():
    assert "--no-deps" in bnb_compat_cycle._installer_snippet()


def test_cuda_requirements_come_from_torch_metadata_without_cuda12(monkeypatch):
    class Distribution:
        requires = [
            "nvidia-cuda-runtime-cu11 (==11.8.89)",
            "nvidia-cublas-cu11 (==11.11.3.6) ; platform_system == 'Windows'",
            "nvidia-cuda-runtime-cu12==12.8.0",
            "typing-extensions>=4.8",
        ]

    monkeypatch.setattr(p100_torch_runtime.metadata, "distribution", lambda name: Distribution())
    assert p100_torch_runtime._torch_cuda_requirements() == [
        "nvidia-cublas-cu11==11.11.3.6",
        "nvidia-cuda-runtime-cu11==11.8.89",
    ]


def test_cuda_library_paths_distinguish_required_sonames(monkeypatch, tmp_path):
    class Distribution:
        def locate_file(self, _name):
            return tmp_path

    (tmp_path / "libcudart.so.11.0").write_bytes(b"")
    (tmp_path / "libcublas.so.11").write_bytes(b"")
    (tmp_path / "libcusparse.so.11").write_bytes(b"")
    monkeypatch.setattr(p100_torch_runtime, "_torch_cuda_requirements", lambda: ["nvidia-cuda-runtime-cu11==11.8.89"])
    monkeypatch.setattr(p100_torch_runtime.metadata, "distribution", lambda _name: Distribution())
    paths = p100_torch_runtime._cuda_library_paths()
    assert paths["libcudart"] and paths["libcublas"] and paths["libcusparse"]


def test_shared_torch_bootstrap_writes_runtime_markers_without_bnb(monkeypatch, tmp_path):
    monkeypatch.setattr(p100_torch_runtime, "_run_json_probe", lambda *args, **kwargs: {"ok": True, "json": {"torch_version": "2.5.1+cu118", "torch_cuda_version": "11.8", "gpu_available": True, "gpu_name": "Tesla P100-PCIE-16GB", "compute_capability": [6, 0], "arch_list": ["sm_60"], "skip_code_available": True, "basic_cuda_tensor_test": True}, "stdout": "{}"})
    monkeypatch.setattr(p100_torch_runtime, "_run_command", lambda *args, **kwargs: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(p100_torch_runtime, "_prepare_cuda_runtime", lambda *args, **kwargs: {"classification": "CUDA_RUNTIME_READY", "torch_version_before": "2.5.1+cu118", "torch_version_after": "2.5.1+cu118"})

    report = p100_torch_runtime.run_shared_p100_torch_bootstrap(report_root=tmp_path, repo_root=tmp_path, phase_prefix="test", write_markers=True)

    assert report["verdict"] == "P100_TORCH_RUNTIME_PASSED"
    assert (tmp_path / "shared_torch_bootstrap_result.json").exists()
    assert (tmp_path / "shared_torch_runtime_result.json").exists()
    assert (tmp_path / p100_torch_runtime.TORCH_PREINSTALL_INSPECTION_JSON).exists()
    assert (tmp_path / p100_torch_runtime.TORCH_INSTALL_RESULT_JSON).exists()
    assert (tmp_path / p100_torch_runtime.TORCH_POSTINSTALL_RUNTIME_JSON).exists()
    assert (tmp_path / p100_torch_runtime.TORCH_POSTINSTALL_CUDA_JSON).exists()
