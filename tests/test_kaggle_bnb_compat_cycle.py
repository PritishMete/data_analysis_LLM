from __future__ import annotations

from pathlib import Path

from kaggle import bnb_compat_cycle
from kaggle import bnb_native_diagnose


def test_native_diagnose_derives_cuda_library_name():
    assert bnb_native_diagnose._cuda_tag("11.8") == "cuda118"
    assert bnb_native_diagnose._cuda_tag("12.8") == "cuda128"
    assert bnb_native_diagnose._cuda_tag(None) is None


def test_native_diagnose_distinguishes_missing_library_and_native_failure():
    assert bnb_native_diagnose._classify(
        expected_exists=False, native_load={"passed": False}, dependency={"resolved": False}, selected=None, backend_active=False
    ) == "BITSANDBYTES_CUDA_LIBRARY_MISSING"
    assert bnb_native_diagnose._classify(
        expected_exists=True, native_load={"passed": False}, dependency={"resolved": True}, selected=None, backend_active=False
    ) == "BITSANDBYTES_NATIVE_LIBRARY_LOAD_FAILED"


def test_native_diagnose_classifies_dependency_failure():
    assert bnb_native_diagnose._classify(
        expected_exists=True, native_load={"passed": True}, dependency={"resolved": False}, selected="lib.so", backend_active=True
    ) == "BITSANDBYTES_CUDA_DEPENDENCY_MISSING"


def test_native_diagnose_enumerates_only_native_files(tmp_path):
    (tmp_path / "libbitsandbytes_cpu.so").write_bytes(b"")
    (tmp_path / "libbitsandbytes_cuda118.so").write_bytes(b"")
    (tmp_path / "README.txt").write_bytes(b"")
    assert bnb_native_diagnose._native_libraries(tmp_path) == ["libbitsandbytes_cpu.so", "libbitsandbytes_cuda118.so"]


def test_torch_install_snippet_targets_cu118_and_forces_reinstall():
    snippet = bnb_compat_cycle._torch_install_snippet()

    assert "torch==2.5.1" in snippet
    assert "torchvision==0.20.1" in snippet
    assert "torchaudio==2.5.1" in snippet
    assert "https://download.pytorch.org/whl/cu118" in snippet
    assert "--force-reinstall" in snippet


def test_bnb_cycle_writes_torch_then_bnb_artifacts_and_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(bnb_compat_cycle, "inspect_kaggle_gpu_identity", lambda: {"gpu_name": "Tesla P100-PCIE-16GB", "compute_capability": [6, 0]})
    monkeypatch.setattr(
        bnb_compat_cycle,
        "resolve_executed_source_commit",
        lambda **kwargs: {"executed_source_commit": "a" * 40, "source_identity_method": "git_rev_parse", "source_identity_verified": True},
    )
    monkeypatch.setattr(bnb_compat_cycle, "write_source_identity", lambda *args, **kwargs: Path(tmp_path / "source_identity_resolved.json"))
    monkeypatch.setattr(bnb_compat_cycle, "_run_command", lambda *args, **kwargs: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(
        bnb_compat_cycle,
        "_torch_state_probe",
        lambda: {
            "ok": True,
            "json": {
                "torch_version": "2.5.1+cu118",
                "torch_cuda_version": "11.8",
                "cuda_available": True,
                "gpu_name": "Tesla P100-PCIE-16GB",
                "capability": [6, 0],
                "arch_list": ["sm_60"],
                "sm_60_supported": True,
                "basic_cuda_tensor_test": True,
                "torch_dynamo_import_passed": True,
                "skip_code_available": True,
            },
        },
    )
    monkeypatch.setattr(
        bnb_compat_cycle,
        "_run_json_probe",
        lambda command, *, timeout, label: {
            "ok": True,
            "json": {
                "torch_version": "2.5.1+cu118",
                "torch_cuda_version": "11.8",
                "bnb_version": "0.43.3",
                "bnb_file": "/tmp/bnb.py",
                "available_cuda_versions": ["11.8"],
                "cuda_backend_active": True,
                "selected_native_cuda_library": "/tmp/libbitsandbytes_cuda118.so",
                "cuda_available": True,
                "device_name": "Tesla P100-PCIE-16GB",
                "capability": [6, 0],
                "arch_list": ["sm_60"],
                "basic_cuda_tensor_test": True,
                "nf4_initialization": True,
                "nf4_quantization": True,
                "nf4_dequantization": True,
                "nf4_cuda": True,
                "nf4_capability_available": True,
            },
            "stdout": "{}",
        },
    )

    report = bnb_compat_cycle.run_bnb_compat_cycle(
        output_root=tmp_path,
        run_id="abc123-20260831T000000Z-test",
        expected_git_commit="a" * 40,
        source_root=tmp_path,
        bootstrap_pid=11,
    )

    assert report["verdict"] == "BNB_NF4_P100_RUNTIME_PASSED"
    assert report["torch_version"] == "2.5.1+cu118"
    assert report["torch_cuda_version"] == "11.8"
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_torch_preinstall.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_torch_install.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_torch_runtime.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_torch_runtime_post_bnb.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_bnb_install.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_bnb_import.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_bnb_cuda.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_nf4.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "bnb_compat_report.json").exists()


def test_bnb_cycle_marks_cpu_fallback_when_backend_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr(bnb_compat_cycle, "inspect_kaggle_gpu_identity", lambda: {"gpu_name": "Tesla P100-PCIE-16GB", "compute_capability": [6, 0]})
    monkeypatch.setattr(
        bnb_compat_cycle,
        "resolve_executed_source_commit",
        lambda **kwargs: {"executed_source_commit": "a" * 40, "source_identity_method": "git_rev_parse", "source_identity_verified": True},
    )
    monkeypatch.setattr(bnb_compat_cycle, "write_source_identity", lambda *args, **kwargs: Path(tmp_path / "source_identity_resolved.json"))
    monkeypatch.setattr(bnb_compat_cycle, "_run_command", lambda *args, **kwargs: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(
        bnb_compat_cycle,
        "_torch_state_probe",
        lambda: {
            "ok": True,
            "json": {
                "torch_version": "2.5.1+cu118",
                "torch_cuda_version": "11.8",
                "cuda_available": True,
                "gpu_name": "Tesla P100-PCIE-16GB",
                "capability": [6, 0],
                "arch_list": ["sm_60"],
                "sm_60_supported": True,
                "basic_cuda_tensor_test": True,
                "torch_dynamo_import_passed": True,
                "skip_code_available": True,
            },
        },
    )
    monkeypatch.setattr(
        bnb_compat_cycle,
        "_run_json_probe",
        lambda command, *, timeout, label: {
            "ok": True,
            "json": {
                "torch_version": "2.5.1+cu118",
                "torch_cuda_version": "11.8",
                "bnb_version": "0.43.3",
                "bnb_file": "/tmp/bnb.py",
                "available_cuda_versions": ["11.8"],
                "cuda_backend_active": False,
                "selected_native_cuda_library": None,
            },
            "stdout": "{}",
        },
    )

    report = bnb_compat_cycle.run_bnb_compat_cycle(
        output_root=tmp_path,
        run_id="abc123-20260831T000000Z-test",
        expected_git_commit="a" * 40,
        source_root=tmp_path,
        bootstrap_pid=11,
    )

    assert report["verdict"] == "BITSANDBYTES_CPU_FALLBACK"
