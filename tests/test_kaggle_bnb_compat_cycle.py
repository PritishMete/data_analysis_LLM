from __future__ import annotations

import json
from pathlib import Path

from kaggle import bnb_compat_cycle


def test_bnb_installer_snippet_is_pinned_and_isolated():
    snippet = bnb_compat_cycle._installer_snippet()

    assert "bitsandbytes==0.43.3" in snippet
    assert "--no-deps" in snippet
    assert "-m\", \"pip\", \"install\"" in snippet


def test_bnb_cycle_writes_isolated_artifacts_and_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(bnb_compat_cycle, "inspect_kaggle_gpu_identity", lambda: {"gpu_name": "Tesla P100-PCIE-16GB", "compute_capability": [6, 0]})
    monkeypatch.setattr(bnb_compat_cycle, "resolve_executed_source_commit", lambda **kwargs: {"executed_source_commit": "a" * 40, "source_identity_method": "git_rev_parse", "source_identity_verified": True})
    monkeypatch.setattr(bnb_compat_cycle, "write_source_identity", lambda *args, **kwargs: Path(tmp_path / "source_identity_resolved.json"))

    def fake_run_command(command, *, timeout, cwd=None):
        if command[:3] == [bnb_compat_cycle.sys.executable, "-c",]:
            if "bitsandbytes==0.43.3" in command[2]:
                return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(bnb_compat_cycle, "_run_command", fake_run_command)
    monkeypatch.setattr(bnb_compat_cycle, "_torch_state_probe", lambda: {"ok": True, "json": {"torch_version": "2.5.1+cu118", "torch_cuda_version": "11.8"}})
    probes = {
        "bnb_import": {"ok": True, "json": {"torch_version": "2.5.1+cu118", "torch_cuda_version": "11.8", "bnb_version": "0.43.3", "bnb_file": "/tmp/bnb.py", "available_cuda_versions": ["11.8"], "cuda_backend_active": True, "selected_native_cuda_library": "/tmp/libbitsandbytes_cuda118.so"}, "stdout": "{}"},
        "bnb_cuda": {"ok": True, "json": {"cuda_available": True, "device_name": "Tesla P100-PCIE-16GB", "capability": [6, 0], "arch_list": ["sm_60"], "basic_cuda_tensor_test": True, "cuda_backend_active": True}, "stdout": "{}"},
        "nf4": {"ok": True, "json": {"nf4_initialization": True, "nf4_quantization": True, "nf4_dequantization": True, "nf4_cuda": True, "nf4_capability_available": True}, "stdout": "{}"},
    }
    monkeypatch.setattr(bnb_compat_cycle, "_run_json_probe", lambda command, *, timeout, label: probes[label])

    report = bnb_compat_cycle.run_bnb_compat_cycle(
        output_root=tmp_path,
        run_id="abc123-20260831T000000Z-test",
        expected_git_commit="a" * 40,
        source_root=tmp_path,
        bootstrap_pid=11,
    )

    assert report["workflow_mode"] == "bnb_compat"
    assert report["verdict"] == "BNB_NF4_P100_RUNTIME_PASSED"
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_bnb_install.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_bnb_import.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_bnb_cuda.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "probe_nf4.json").exists()
    assert (tmp_path / "smoke_runs" / "abc123-20260831T000000Z-test" / "bnb_compat_report.json").exists()


def test_bnb_cycle_marks_cpu_fallback_when_backend_inactive(tmp_path, monkeypatch):
    monkeypatch.setattr(bnb_compat_cycle, "inspect_kaggle_gpu_identity", lambda: {"gpu_name": "Tesla P100-PCIE-16GB", "compute_capability": [6, 0]})
    monkeypatch.setattr(bnb_compat_cycle, "resolve_executed_source_commit", lambda **kwargs: {"executed_source_commit": "a" * 40, "source_identity_method": "git_rev_parse", "source_identity_verified": True})
    monkeypatch.setattr(bnb_compat_cycle, "write_source_identity", lambda *args, **kwargs: Path(tmp_path / "source_identity_resolved.json"))
    monkeypatch.setattr(bnb_compat_cycle, "_run_command", lambda *args, **kwargs: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(bnb_compat_cycle, "_torch_state_probe", lambda: {"ok": True, "json": {"torch_version": "2.5.1+cu118", "torch_cuda_version": "11.8"}})
    monkeypatch.setattr(
        bnb_compat_cycle,
        "_run_json_probe",
        lambda command, *, timeout, label: {"ok": True, "json": {"torch_version": "2.5.1+cu118", "torch_cuda_version": "11.8", "bnb_version": "0.43.3", "bnb_file": "/tmp/bnb.py", "available_cuda_versions": ["11.8"], "cuda_backend_active": False, "selected_native_cuda_library": None}},
    )

    report = bnb_compat_cycle.run_bnb_compat_cycle(
        output_root=tmp_path,
        run_id="abc123-20260831T000000Z-test",
        expected_git_commit="a" * 40,
        source_root=tmp_path,
        bootstrap_pid=11,
    )

    assert report["verdict"] == "BITSANDBYTES_CPU_FALLBACK"
