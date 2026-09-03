from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

from kaggle.bootstrap import (
    build_artifact_manifest,
    build_kaggle_dependency_plan,
    build_semantic_dataset_from_canonical,
    create_final_zip,
    discover_semantic_dataset,
    load_semantic_config,
    resolve_canonical_dataset_root,
    semantic_verdict,
    verify_attached_dataset,
    write_sha_manifest,
    write_dependency_preflight_report,
)
from kaggle.run_context import resolve_executed_source_commit, write_source_identity
from kaggle.run_semantic_training import _build_smoke_corpus, _dependency_probe_snippets, _patch_torch_dynamo_compatibility, _smoke_split_targets
from kaggle.run_semantic_training import _safe_commit_hash, _write_smoke_failure, _write_smoke_heartbeat, run_notebook_flow


def _canonical_record(
    *,
    source_id: str,
    intent: str,
    family_fingerprint: str,
    split: str,
    logical_structure: str = "AND",
    quality: float = 0.99,
    semantic_roles: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source_kind": "experience",
        "source_id": source_id,
        "split": split,
        "family_fingerprint": family_fingerprint,
        "input": {
            "intent": intent,
            "semantic_roles": semantic_roles or ["numeric_metric", "filter_value"],
            "operators": ["equals", "greater_than"],
            "logical_structure": logical_structure,
            "predicate_graph": {"predicate_count": 2, "shape": "safe"},
        },
        "output": {
            "tool_graph": ["sql.filter"],
            "plan_source": "validated_template",
            "plan_template_id": "plan.template.semantic",
            "source_kind": "experience",
            "candidate_state": None,
        },
        "metadata": {
            "quality": quality,
            "execution_success": True,
            "critic_passed": True,
            "result_validation_passed": True,
            "plan_completeness_passed": True,
            "privacy_validation_passed": True,
            "no_unresolved_ambiguity": True,
            "no_critical_repair": True,
            "repair_count": 0,
            "correction_state": "validated",
            "candidate_state": None,
            "candidate_evidence_count": None,
            "candidate_average_quality": None,
            "dataset_semantic_signature": "0123456789abcdef",
            "family_fingerprint": family_fingerprint,
            "split": split,
            "family_size": 1,
            "created_at": "2026-08-27T00:00:00+00:00",
            "plan_shape": {"limit": 5, "metric_count": 1, "tool_sequence": ["sql.filter"]},
        },
    }


def _write_canonical_dataset(root: Path, *, with_sha_manifest: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    records = [
        _canonical_record(source_id="evt_a", intent="filter", family_fingerprint="a" * 64, split="train", semantic_roles=["numeric_metric", "filter_value", "dimension_label"]),
        _canonical_record(source_id="evt_b", intent="analytics", family_fingerprint="b" * 64, split="train", logical_structure="OR", semantic_roles=["trend_metric", "time_window", "dimension_label"]),
        _canonical_record(source_id="evt_c", intent="operation", family_fingerprint="c" * 64, split="train", logical_structure="MIXED", semantic_roles=["action", "constraint", "status_flag"]),
        _canonical_record(source_id="evt_d", intent="cleaning", family_fingerprint="d" * 64, split="train", logical_structure="NOT", semantic_roles=["column_name", "null_check", "threshold"]),
        _canonical_record(source_id="evt_e", intent="sentiment", family_fingerprint="e" * 64, split="validation", semantic_roles=["text_span", "sentiment_label", "source_field"]),
        _canonical_record(source_id="evt_f", intent="filter", family_fingerprint="f" * 64, split="validation", semantic_roles=["numeric_metric", "range_bound", "dimension_label"]),
        _canonical_record(source_id="evt_g", intent="analytics", family_fingerprint="g" * 64, split="validation", logical_structure="OR", semantic_roles=["trend_metric", "aggregation_target", "time_window"]),
        _canonical_record(source_id="evt_h", intent="operation", family_fingerprint="h" * 64, split="test", logical_structure="MIXED", semantic_roles=["action", "workflow_step", "status_flag"]),
        _canonical_record(source_id="evt_i", intent="cleaning", family_fingerprint="i" * 64, split="test", logical_structure="NOT", semantic_roles=["column_name", "duplicate_check", "null_check"]),
        _canonical_record(source_id="evt_j", intent="sentiment", family_fingerprint="j" * 64, split="test", semantic_roles=["text_span", "sentiment_label", "channel_source"]),
    ]
    for split in ("train", "validation", "test"):
        lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records if record["split"] == split]
        (root / f"{split}.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    split_counts = {split: sum(1 for record in records if record["split"] == split) for split in ("train", "validation", "test")}
    manifest = {
        "dataset_version": "canonical-test",
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "eligible_examples": sum(split_counts.values()),
        "readiness": {"ready_for_prototype": True},
    }
    report = {
        "dataset_version": "canonical-test",
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "eligible_examples": sum(split_counts.values()),
        "readiness": {"ready_for_prototype": True},
        "split_integrity_passed": True,
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / "dataset_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if with_sha_manifest:
        write_sha_manifest(root, root)
    return root


def test_recursive_discovery_resolves_nested_kaggle_root(tmp_path):
    kaggle_input = tmp_path / "kaggle" / "input"
    canonical_root = _write_canonical_dataset(kaggle_input / "datasets" / "jaistudio" / "data-analysis-llm")

    resolved = resolve_canonical_dataset_root(kaggle_input)
    discovered = discover_semantic_dataset(kaggle_input)
    verification = verify_attached_dataset(canonical_root)

    assert resolved["root"] == str(canonical_root)
    assert discovered == canonical_root
    assert verification["verified"] is True
    assert verification["mismatches"] == []


def test_multiple_candidate_roots_fail_clearly(tmp_path):
    kaggle_input = tmp_path / "input"
    root_a = _write_canonical_dataset(kaggle_input / "vendor-a" / "dataset-a")
    root_b = _write_canonical_dataset(kaggle_input / "vendor-b" / "dataset-b")

    resolved = resolve_canonical_dataset_root(kaggle_input)

    assert resolved["root"] is None
    assert resolved["reason"] == "ambiguous_dataset_root"
    candidate_roots = {item["root"] for item in resolved["candidates"]}
    assert candidate_roots == {str(root_a), str(root_b)}


def test_missing_sha_manifest_falls_back_to_consistency_and_generates_sha(tmp_path):
    canonical_root = _write_canonical_dataset(tmp_path / "canonical")
    verification = verify_attached_dataset(canonical_root)
    semantic_root = tmp_path / "semantic_training"

    assert verification["verified"] is True
    assert verification["mismatches"] == []
    assert "consistency" in verification

    sha_manifest_path = write_sha_manifest(canonical_root, semantic_root)
    payload = json.loads(sha_manifest_path.read_text(encoding="utf-8"))
    names = [item["name"] for item in payload["files"]]

    assert "train.jsonl" in names
    assert "validation.jsonl" in names
    assert "test.jsonl" in names
    assert "dataset_manifest.json" in names
    assert "manifest.json" in names
    assert "dataset_report.json" in names
    assert "report.json" in names
    assert payload["dataset_root"] == str(canonical_root)


def test_sha_manifest_verifier_accepts_list_format(tmp_path):
    canonical_root = _write_canonical_dataset(tmp_path / "canonical")
    sha_manifest_path = write_sha_manifest(canonical_root, canonical_root)

    verification = verify_attached_dataset(canonical_root)

    assert sha_manifest_path.exists()
    assert verification["verified"] is True
    assert verification["mismatches"] == []


def test_canonical_to_semantic_conversion_preserves_privacy_and_rows(tmp_path):
    canonical_root = _write_canonical_dataset(tmp_path / "canonical")
    semantic_report = build_semantic_dataset_from_canonical(canonical_root, tmp_path / "semantic_training")
    semantic_root = Path(semantic_report["semantic_output_root"])

    train = (semantic_root / "train.jsonl").read_text(encoding="utf-8")
    validation = (semantic_root / "validation.jsonl").read_text(encoding="utf-8")
    test = (semantic_root / "test.jsonl").read_text(encoding="utf-8")
    combined = train + validation + test

    assert semantic_report["semantic_row_count"] > 0
    assert semantic_report["readiness"]["ready"] is True
    assert sum(semantic_report["split_counts"].values()) == semantic_report["semantic_row_count"]
    assert "intent" in combined
    assert "semantic_bindings" in combined
    assert "predicate_graph" in combined
    assert "tool_graph" not in combined
    assert "sql" not in combined.lower()
    for needle in ["John Smith", "john@example.com", "ACC-9988", "SecretCompanyXYZ", "9876543210"]:
        assert needle not in combined


def test_artifact_manifest_and_zip_exclusions(tmp_path):
    safe = tmp_path / "training_config"
    unsafe = tmp_path / "dataset.jsonl"
    safe.write_text("safe", encoding="utf-8")
    unsafe.write_text("secret", encoding="utf-8")

    manifest = build_artifact_manifest([safe, unsafe])
    zip_path = create_final_zip(tmp_path, [safe, unsafe], zip_name="bundle.zip")

    assert manifest["manifest_version"] == 1
    assert {item["name"] for item in manifest["artifacts"]} == {"dataset.jsonl", "training_config"}
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as archive:
        assert archive.namelist() == ["training_config"]


def test_semantic_verdict_logic():
    promotable = semantic_verdict(
        gate_results={
            "intent_accuracy": 0.96,
            "binding_accuracy": 0.91,
            "predicate_coverage": 0.92,
            "logical_structure_accuracy": 0.92,
            "semantic_schema_valid_rate": 0.99,
            "fallback_accuracy": 0.95,
        },
        readiness=True,
        fallback_rate=0.0,
    )
    rejected = semantic_verdict(
        gate_results={"intent_accuracy": 0.2},
        readiness=True,
        fallback_rate=1.0,
    )
    failed = semantic_verdict(
        gate_results={},
        readiness=False,
        fallback_rate=0.0,
    )

    assert promotable == "PROMOTE_SEMANTIC_EXTRACTOR_TO_SHADOW"
    assert rejected == "REJECT_SEMANTIC_EXTRACTOR"
    assert failed == "TRAINING_FAILED"


def test_semantic_config_loader_resolves_repo_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_semantic_config()

    assert config["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["training"]["max_seq_len"] == 768


def test_smoke_split_targets_and_report_require_validation(tmp_path):
    canonical_root = _write_canonical_dataset(tmp_path / "canonical")
    semantic_root = tmp_path / "semantic_training"
    semantic_report = build_semantic_dataset_from_canonical(canonical_root, semantic_root)
    smoke = _build_smoke_corpus(Path(semantic_report["semantic_output_root"]), tmp_path / "smoke")

    assert smoke["report"]["train_count"] >= 1
    assert smoke["report"]["validation_count"] >= 5
    assert smoke["report"]["test_count"] == 0
    assert len(smoke["splits"]["validation"]) >= 5
    assert len(smoke["splits"]["train"]) <= 100


def test_smoke_heartbeat_and_failure_artifacts_are_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_EXPECTED_GIT_COMMIT", "a" * 40)
    heartbeat = _write_smoke_heartbeat(tmp_path / "reports", stage="notebook_started")
    failure = _write_smoke_failure(report_root=tmp_path / "reports", stage="model_load_started", exc=RuntimeError("boom"), torch_module=None)

    heartbeat_payload = json.loads(heartbeat.read_text(encoding="utf-8"))
    failure_payload = json.loads(failure.read_text(encoding="utf-8"))

    assert heartbeat_payload["stage"] == "notebook_started"
    assert heartbeat_payload["smoke_mode"] is True
    assert heartbeat_payload["git_commit"] is not None
    assert len(str(heartbeat_payload["git_commit"])) == 40
    assert failure_payload["stage"] == "model_load_started"
    assert failure_payload["exception_type"] == "RuntimeError"
    assert "boom" in failure_payload["sanitized_exception_message"]


def test_dependency_preflight_selects_p100_cu118_stack(tmp_path):
    gpu_identity = {
        "gpu_name": "Tesla P100-PCIE-16GB",
        "driver_version": "535.54.03",
        "memory_total_mb": 16280,
    }
    torch_probe = {
        "ok": True,
        "json": {
            "version": "2.10.0+cu128",
            "cuda": "12.8",
            "available": True,
            "device_name": "Tesla P100-PCIE-16GB",
            "capability": [6, 0],
        },
    }
    bitsandbytes_probe = {
        "ok": True,
        "json": {
            "version": "0.50.2",
                "available_cuda_versions": ["11.8", "12.1", "12.4", "12.6"],
            "cuda_specs": {
                "highest_compute_capability": [7, 5],
                "cuda_version_string": "124",
                "cuda_version_tuple": [12, 4],
            },
        },
    }

    preflight = build_kaggle_dependency_plan(
        gpu_identity=gpu_identity,
        torch_probe=torch_probe,
        bitsandbytes_probe=bitsandbytes_probe,
    )
    report_path = write_dependency_preflight_report(tmp_path, preflight)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert preflight.requires_torch_cu126 is True
    assert preflight.requires_bitsandbytes_upgrade is False
    assert preflight.compatibility_passed is True
    assert payload["gpu_name"] == "Tesla P100-PCIE-16GB"
    assert payload["install_plan"]["pip_groups"][0]["index_url"].endswith("/cu118")
    assert any("torch==2.5.1+cu118" in package for package in payload["install_plan"]["pip_groups"][0]["packages"])
    assert any(group["name"] == "model_runtime" for group in payload["install_plan"]["pip_groups"])
    model_group = next(group for group in payload["install_plan"]["pip_groups"] if group["name"] == "model_runtime")
    assert "huggingface_hub==0.26.2" in model_group["packages"]


def test_dependency_preflight_passes_after_cu118_verification():
    gpu_identity = {
        "gpu_name": "Tesla P100-PCIE-16GB",
        "driver_version": "535.54.03",
        "memory_total_mb": 16280,
    }
    torch_probe = {
        "ok": True,
        "json": {
            "version": "2.10.0+cu128",
            "cuda": "12.8",
            "available": True,
            "device_name": "Tesla P100-PCIE-16GB",
            "capability": [6, 0],
        },
    }
    bitsandbytes_probe = {
        "ok": True,
        "json": {
            "version": "0.50.2",
            "available_cuda_versions": ["11.8", "12.1", "12.4"],
            "cuda_specs": {
                "highest_compute_capability": [6, 0],
                "cuda_version_string": "124",
                "cuda_version_tuple": [12, 4],
            },
        },
    }

    preflight = build_kaggle_dependency_plan(
        gpu_identity=gpu_identity,
        torch_probe=torch_probe,
        bitsandbytes_probe=bitsandbytes_probe,
    )

    assert preflight.compatibility_passed is True
    assert preflight.reason is None
    assert any(group["name"] == "torch_cu126" for group in preflight.install_plan["pip_groups"])
    assert preflight.install_plan["pip_groups"][0]["index_url"].endswith("/cu118")


def test_bootstrap_uses_canonical_bnb_probe_and_nf4_gate():
    bootstrap_source = Path("kaggle/bootstrap.py").read_text(encoding="utf-8")
    environment_source = Path("kaggle/bootstrap_environment.py").read_text(encoding="utf-8")
    assert "def _probe_bitsandbytes_runtime" in bootstrap_source
    assert "_probe_bitsandbytes_runtime," in environment_source
    assert "postinstall_bnb = _probe_bitsandbytes_runtime()" in environment_source
    assert '"real_bnb_cuda_operation"' in bootstrap_source
    assert '"BNB_IMPORT_FAILED"' in environment_source
    assert '"BNB_CUDA_RUNTIME_FAILED"' in environment_source
    assert '"BNB_NF4_RUNTIME_FAILED"' in environment_source
    assert 'and bool((bnb_probe.get("json") or {}).get("real_bnb_cuda_operation"))' in environment_source


def test_stale_kaggle_checkout_fails_fast(tmp_path, monkeypatch):
    canonical_root = _write_canonical_dataset(tmp_path / "canonical")
    repo_root = tmp_path / "archive"
    (repo_root / "kaggle").mkdir(parents=True, exist_ok=True)
    (repo_root / "src").mkdir(parents=True, exist_ok=True)
    for name in ("bootstrap_environment.py", "execute_smoke_training.py", "run_semantic_training.py"):
        (repo_root / "kaggle" / name).write_text("# stub\n", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_EXPECTED_GIT_COMMIT", "expected123")
    monkeypatch.setattr("kaggle.run_semantic_training.resolve_canonical_dataset_root", lambda: {"root": str(canonical_root)})
    monkeypatch.setattr("kaggle.run_semantic_training.verify_attached_dataset", lambda dataset_dir: {"verified": True, "mismatches": []})
    monkeypatch.setattr("kaggle.run_semantic_training.resolve_executed_source_commit", lambda **kwargs: {"executed_source_commit": "actual456", "source_identity_method": "source_identity_json", "source_identity_verified": True})
    monkeypatch.setattr("kaggle.run_semantic_training.build_semantic_dataset_from_canonical", lambda dataset_dir, output_dir: {"semantic_output_root": str(tmp_path / "semantic_training"), "bundle_report": {"train_count": 1, "validation_count": 1, "test_count": 1}, "semantic_row_count": 3, "split_counts": {"train": 1, "validation": 1, "test": 1}, "readiness": {"ready": True}, "sha_manifest_path": str(tmp_path / "sha.json")})
    monkeypatch.setattr("kaggle.run_semantic_training.build_training_plan", lambda **kwargs: {"base_model": "Qwen/Qwen2.5-0.5B-Instruct"})
    monkeypatch.setattr("kaggle.run_semantic_training._build_smoke_corpus", lambda semantic_root, output_root: {"root": tmp_path / "smoke_training", "report": {}, "splits": {"train": [], "validation": [], "test": []}})
    monkeypatch.setattr("kaggle.run_semantic_training.detect_resume_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr("kaggle.run_semantic_training.ensure_kaggle_paths", lambda output_root: type("P", (), {"reports": tmp_path / "reports", "metrics": tmp_path / "metrics", "manifests": tmp_path / "manifests", "root": tmp_path, "checkpoints": tmp_path / "checkpoints", "adapters": tmp_path / "adapters", "to_dict": lambda self: {"root": str(tmp_path)}})())
    monkeypatch.setattr("kaggle.run_semantic_training._run_real_smoke_training", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr("kaggle.run_semantic_training._safe_commit_hash", lambda repo_root=None: "expected123")
    monkeypatch.setattr("kaggle.run_semantic_training._validate_archive_root", lambda repo_root: None)

    try:
        run_notebook_flow(output_root=tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "STALE_SOURCE_SNAPSHOT"
    else:
        raise AssertionError("stale checkout should fail fast")


def test_resolve_executed_source_commit_supports_archive_without_git(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    source_identity = write_source_identity(
        run_root,
        run_id="run-123",
        expected_git_commit="a" * 40,
        executed_source_commit="b" * 40,
        source_identity_method="explicit_runner_metadata",
        source_identity_verified=True,
    )
    (tmp_path / "archive").mkdir()
    resolved = resolve_executed_source_commit(run_root=run_root, repo_root=tmp_path / "archive", expected_git_commit="a" * 40)

    assert source_identity.exists()
    assert resolved["executed_source_commit"] == "b" * 40
    assert resolved["source_identity_method"] == "explicit_runner_metadata"


def test_torch_dynamo_compatibility_patch_adds_missing_skip_code():
    class DummyEvalFrame:
        pass

    class DummyDynamo:
        eval_frame = DummyEvalFrame()

    class DummyC:
        _dynamo = DummyDynamo()

    class DummyTorch:
        _C = DummyC()

    patched = _patch_torch_dynamo_compatibility(DummyTorch())

    assert patched is True
    assert callable(DummyTorch._C._dynamo.eval_frame.skip_code)


def test_dependency_probe_snippets_add_repo_root_to_sys_path():
    snippets = _dependency_probe_snippets()
    for snippet in snippets.values():
        assert "pathlib.Path.cwd()" in snippet
        assert "sys.path.insert(0, str(repo_root))" in snippet
        assert "from src.training.torch_compat import ensure_torch_dynamo_compatibility" in snippet


def test_dependency_compatibility_preflight_writes_isolated_probe_artifacts(tmp_path, monkeypatch):
    probes = {
        "compat": {"ok": True, "parent_pid": 11, "child_pid": 22, "returncode": 0, "signal": None, "timed_out": False, "stdout": '{"torch_imported": true}', "stderr": "", "json": {"torch_imported": True}},
        "torch_import": {"ok": True, "parent_pid": 11, "child_pid": 23, "returncode": 0, "signal": None, "timed_out": False, "stdout": '{"version": "2.6.0+cu124", "cuda": "12.4", "available": true}', "stderr": "", "json": {"version": "2.6.0+cu124", "cuda": "12.4", "available": True}},
        "torch_cuda": {"ok": True, "parent_pid": 11, "child_pid": 24, "returncode": 0, "signal": None, "timed_out": False, "stdout": '{"available": true, "device_name": "Tesla P100-PCIE-16GB", "capability": [6, 0], "arch_list": ["sm_60"], "basic_cuda_tensor_test": true}', "stderr": "", "json": {"available": True, "device_name": "Tesla P100-PCIE-16GB", "capability": [6, 0], "arch_list": ["sm_60"], "basic_cuda_tensor_test": True}},
        "bitsandbytes": {"ok": True, "parent_pid": 11, "child_pid": 25, "returncode": 0, "signal": None, "timed_out": False, "stdout": '{"version": "0.43.3", "available_cuda_versions": ["12.4"], "cuda_specs": {"highest_compute_capability": [6, 0], "cuda_version_string": "12.4", "cuda_version_tuple": [12, 4]}}', "stderr": "", "json": {"version": "0.43.3", "available_cuda_versions": ["12.4"], "cuda_specs": {"highest_compute_capability": [6, 0], "cuda_version_string": "12.4", "cuda_version_tuple": [12, 4]}}},
        "nf4": {"ok": True, "parent_pid": 11, "child_pid": 26, "returncode": 0, "signal": None, "timed_out": False, "stdout": '{"nf4_capability_available": true, "cuda_specs": {"highest_compute_capability": [6, 0], "cuda_version_string": "12.4", "cuda_version_tuple": [12, 4]}}', "stderr": "", "json": {"nf4_capability_available": True, "cuda_specs": {"highest_compute_capability": [6, 0], "cuda_version_string": "12.4", "cuda_version_tuple": [12, 4]}}},
    }

    monkeypatch.setattr("kaggle.run_semantic_training._dependency_probe_snippets", lambda: {
        "compat": "compat",
        "torch_import": "torch_import",
        "torch_cuda": "torch_cuda",
        "bitsandbytes": "bitsandbytes",
        "nf4": "nf4",
    })
    monkeypatch.setattr("kaggle.run_semantic_training._run_python_probe", lambda snippet, *, timeout, phase, label: probes[label])

    result = __import__("kaggle.run_semantic_training", fromlist=["_run_dependency_compatibility_preflight"])._run_dependency_compatibility_preflight(  # type: ignore[attr-defined]
        report_root=tmp_path,
        breadcrumbs_path=tmp_path / "smoke_breadcrumbs.jsonl",
    )

    for name in [
        "probe_compat_shim.json",
        "probe_torch_import_runtime.json",
        "probe_torch_cuda_runtime.json",
        "probe_bitsandbytes_runtime.json",
        "probe_nf4_runtime.json",
    ]:
        assert (tmp_path / "reports" / name).exists()
    assert result["preflight"]["compatibility_passed"] is True


def test_kaggle_run_semantic_training_import_is_transformers_lazy(monkeypatch):
    import importlib
    import sys

    sys.modules.pop("transformers", None)
    sys.modules.pop("kaggle.run_semantic_training", None)
    sys.modules.pop("src.training.benchmark", None)

    module = importlib.import_module("kaggle.run_semantic_training")

    assert "transformers" not in sys.modules
    assert "src.training.benchmark" not in sys.modules
    assert hasattr(module, "COMPATIBILITY_REPORT")
    assert module.COMPATIBILITY_REPORT is None


def test_bootstrap_environment_import_is_transformers_lazy(monkeypatch):
    import importlib
    import sys

    sys.modules.pop("transformers", None)
    sys.modules.pop("kaggle.bootstrap_environment", None)

    module = importlib.import_module("kaggle.bootstrap_environment")

    assert "transformers" not in sys.modules
    assert hasattr(module, "main")


def test_bootstrap_environment_supports_standalone_script_import():
    module_path = Path("kaggle/bootstrap_environment.py").resolve()
    result = runpy.run_path(str(module_path), run_name="bootstrap_environment_script")
    assert result["KAGGLE_WORKING_ROOT"].name == "working"


def test_import_trace_order_records_ml_boundaries(tmp_path, monkeypatch):
    import importlib

    trace_path = tmp_path / "import_trace.jsonl"
    trace_module = importlib.import_module("kaggle.import_trace")
    monkeypatch.setattr(trace_module, "DEFAULT_IMPORT_TRACE_PATH", trace_path)

    trace_module.write_import_trace(trace_path, module="bootstrap", event="bootstrap_started")
    trace_module.write_import_trace(trace_path, module="training", event="before_torch_import")
    trace_module.write_import_trace(trace_path, module="training", event="after_torch_import", compatibility_patch_ran=True)

    lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [line["event"] for line in lines] == ["bootstrap_started", "before_torch_import", "after_torch_import"]
    assert all("pid" in line and "timestamp" in line for line in lines)


def test_execute_smoke_training_uses_fresh_process_metadata(monkeypatch, tmp_path):
    import importlib

    module = importlib.import_module("kaggle.execute_smoke_training")
    monkeypatch.setattr("kaggle.run_semantic_training.run_notebook_flow", lambda **kwargs: {"smoke_training_report": {"ok": True}, "result": "ok"})
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: __import__("subprocess").CompletedProcess(args=args[0], returncode=0, stdout="0309f6e824127a1ebab2bf13a87cb7ab12ff3a61\n", stderr=""))
    monkeypatch.delenv("KAGGLE_SKIP_DEP_INSTALL", raising=False)

    result_code = module.main(["--output-root", str(tmp_path), "--bootstrap-pid", "12345"])

    payload = json.loads((tmp_path / "reports" / "smoke_heartbeat.json").read_text(encoding="utf-8"))
    assert result_code == 0
    assert payload["bootstrap_pid"] == 12345
    assert payload["training_pid"] != 12345
    assert payload["fresh_process_verified"] is True
    assert "KAGGLE_SKIP_DEP_INSTALL" not in os.environ


def test_execute_smoke_training_supports_standalone_script_import():
    result = runpy.run_path(str(Path("kaggle/execute_smoke_training.py").resolve()), run_name="execute_smoke_training_script")
    assert result["__name__"] == "execute_smoke_training_script"


def test_torch_compat_cycle_supports_standalone_script_import():
    result = runpy.run_path(str(Path("kaggle/torch_compat_cycle.py").resolve()), run_name="torch_compat_cycle_script")
    assert result["__name__"] == "torch_compat_cycle_script"
