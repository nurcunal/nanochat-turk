from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.strict_checkpoint import (
    CheckpointIntegrityError,
    build_strict_checkpoint_identity,
    finalize_strict_checkpoint,
    save_strict_rank_state,
)
from nanochat.training_log import CanonicalTrainingLog, checkpoint_curve_log_state
from nanochat.turkish_backend import (
    MIXTURE_QUALITY_APPROVAL_KIND,
    RESOURCE_APPROVAL_KIND,
    RESOURCE_REPORT_KIND,
)
from nanochat.turkish_corpus import load_corpus_policy
from scripts import d32_family_workflow as workflow
from scripts import upload_base_checkpoint_to_hf as uploader


ROOT = Path(__file__).resolve().parents[1]
RECIPE_V3 = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_wsd_v3.json"
RECIPE_V4 = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_wsd_v4.json"


@pytest.mark.parametrize("recipe_path", (RECIPE_V3, RECIPE_V4))
def test_repeat_capacity_families_retain_all_final_transactions(
    recipe_path: Path,
) -> None:
    recipe, _ = workflow.load_recipe(recipe_path)
    assert {item["retention"] for item in recipe["checkpoints"]["finals"]} == {
        "full_resumable"
    }
    storage = recipe["storage"]
    assert storage["cooled_final_model_bundles_retained"] == 0
    assert storage["full_cooled_final_transactions_at_peak"] == 3
    assert storage["required_free_bytes_at_training_preflight"] == 560 * 1024**3
    assert storage["estimated_end_to_end_peak_project_bytes"] == 792 * 1024**3


def test_family_metric_roots_are_exact_and_remote_categories_are_stable() -> None:
    assert workflow.family_metric_roots(workflow.FAMILY_ID_V3) == {
        "family": Path("metrics/d32_family"),
        "smoke": Path("metrics/d32_smoke"),
        "proxy": Path("metrics/d32_proxy"),
        "signal_smoke": Path("metrics/d32_signal_smoke"),
    }
    assert workflow.family_metric_roots(workflow.FAMILY_ID_V4) == {
        "family": Path("metrics/d32_v4/family"),
        "smoke": Path("metrics/d32_v4/smoke"),
        "proxy": Path("metrics/d32_v4/proxy"),
        "signal_smoke": Path("metrics/d32_v4/signal_smoke"),
    }
    with pytest.raises(workflow.FamilyWorkflowError, match="metric namespace"):
        workflow.family_metric_roots("tr_d32_general_bpe32k_unknown")


class _AtomicHubFixture:
    def __init__(
        self,
        files: dict[str, bytes],
        *,
        fail_commit: bool = False,
        parent_sha: str | None = "a" * 40,
        mismatch_path: str | None = None,
        unexpected_remote_path: str | None = None,
    ) -> None:
        self.files = dict(files)
        self.fail_commit = fail_commit
        self.parent_sha = parent_sha
        self.mismatch_path = mismatch_path
        self.unexpected_remote_path = unexpected_remote_path
        self.create_repo_calls = []
        self.commit_calls = []
        self.list_repo_files_calls = []

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.parent_sha)

    def list_repo_files(self, **kwargs):
        self.list_repo_files_calls.append(kwargs)
        assert kwargs["revision"] == self.parent_sha
        return sorted(self.files)

    def upload_file(self, **_kwargs):
        raise AssertionError("closed publication must never commit one file at a time")

    def create_commit(self, **kwargs):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        self.commit_calls.append(kwargs)
        if self.fail_commit:
            raise RuntimeError("simulated interruption before ref update")
        replacement = dict(self.files)
        for operation in kwargs["operations"]:
            if isinstance(operation, CommitOperationDelete):
                replacement.pop(operation.path_in_repo, None)
            elif isinstance(operation, CommitOperationAdd):
                replacement[operation.path_in_repo] = Path(
                    operation.path_or_fileobj
                ).read_bytes()
            else:  # pragma: no cover - protects the fixture from API drift
                raise AssertionError(type(operation))
        self.files = replacement
        return SimpleNamespace(oid="b" * 40)

    def list_repo_tree(self, **kwargs):
        assert kwargs == {
            "repo_id": "fixture/d32",
            "repo_type": "model",
            "revision": "b" * 40,
            "recursive": True,
            "expand": False,
        }
        files = dict(self.files)
        if self.unexpected_remote_path:
            files[self.unexpected_remote_path] = b"unexpected"
        records = []
        for path, payload in sorted(files.items()):
            git_blob = hashlib.sha1(
                f"blob {len(payload)}\0".encode("ascii") + payload
            ).hexdigest()
            lfs = None
            if path.endswith(".pt"):
                lfs = SimpleNamespace(
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size=len(payload),
                )
            if path == self.mismatch_path:
                if lfs is None:
                    git_blob = "0" * 40
                else:
                    lfs.sha256 = "0" * 64
            records.append(
                SimpleNamespace(
                    path=path,
                    size=len(payload),
                    blob_id=git_blob,
                    lfs=lfs,
                )
            )
        return records


def test_closed_hf_publication_replaces_stale_payload_in_one_commit(
    tmp_path: Path,
) -> None:
    readme = _touch(tmp_path / "README.md", b"new card")
    manifest = _touch(tmp_path / "upload_manifest.json", b"new manifest")
    model = _touch(tmp_path / "model.pt", b"new model")
    api = _AtomicHubFixture(
        {
            ".gitattributes": b"hub metadata",
            "README.md": b"old card",
            "finals/s12/stale_optimizer.pt": b"stale",
        }
    )

    uploader._publish_closed_model_snapshot(
        api,
        repo_id="fixture/d32",
        files=[
            (readme, "README.md"),
            (manifest, "provenance/upload_manifest.json"),
            (model, "finals/s12/model.pt"),
        ],
        private=True,
        commit_message="publish fixture",
    )

    assert api.create_repo_calls == [
        {
            "repo_id": "fixture/d32",
            "repo_type": "model",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert len(api.commit_calls) == 1
    assert api.commit_calls[0]["parent_commit"] == "a" * 40
    assert api.list_repo_files_calls == [
        {
            "repo_id": "fixture/d32",
            "repo_type": "model",
            "revision": "a" * 40,
        }
    ]
    assert api.files == {
        ".gitattributes": b"hub metadata",
        "README.md": b"new card",
        "provenance/upload_manifest.json": b"new manifest",
        "finals/s12/model.pt": b"new model",
    }


def test_interrupted_closed_hf_publication_leaves_previous_commit_intact(
    tmp_path: Path,
) -> None:
    replacement = _touch(tmp_path / "README.md", b"new card")
    previous = {
        ".gitattributes": b"hub metadata",
        "README.md": b"previous complete card",
        "finals/s12/model.pt": b"previous complete model",
    }
    api = _AtomicHubFixture(previous, fail_commit=True)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        uploader._publish_closed_model_snapshot(
            api,
            repo_id="fixture/d32",
            files=[(replacement, "README.md")],
            private=True,
            commit_message="publish fixture",
        )

    assert len(api.commit_calls) == 1
    assert api.files == previous


def test_closed_hf_publication_handles_new_empty_repo_without_parent(
    tmp_path: Path,
) -> None:
    readme = _touch(tmp_path / "README.md", b"first card")
    api = _AtomicHubFixture({}, parent_sha=None)

    uploader._publish_closed_model_snapshot(
        api,
        repo_id="fixture/d32",
        files=[(readme, "README.md")],
        private=True,
        commit_message="first publication",
    )

    assert api.list_repo_files_calls == []
    assert api.commit_calls[0]["parent_commit"] is None
    assert api.files == {"README.md": b"first card"}


def test_closed_hf_publication_limits_legacy_cleanup_to_managed_prefix(
    tmp_path: Path,
) -> None:
    readme = _touch(tmp_path / "README.md", b"replacement")
    api = _AtomicHubFixture(
        {
            ".gitattributes": b"hub metadata",
            "other-release/model.pt": b"keep",
            "release-v1/stale.log": b"delete",
        }
    )

    uploader._publish_closed_model_snapshot(
        api,
        repo_id="fixture/d32",
        files=[(readme, "release-v1/README.md")],
        private=True,
        commit_message="replace release v1",
        managed_prefix="release-v1",
    )

    assert api.files == {
        ".gitattributes": b"hub metadata",
        "other-release/model.pt": b"keep",
        "release-v1/README.md": b"replacement",
    }


@pytest.mark.parametrize(
    ("remote_path", "message"),
    (
        ("README.md", "published Git blob mismatch"),
        ("finals/s12/model.pt", "published LFS SHA-256 mismatch"),
    ),
)
def test_closed_hf_publication_rejects_remote_content_mismatch(
    tmp_path: Path, remote_path: str, message: str
) -> None:
    payload = _touch(tmp_path / Path(remote_path).name, b"payload")
    api = _AtomicHubFixture({}, mismatch_path=remote_path)

    with pytest.raises(RuntimeError, match=message):
        uploader._publish_closed_model_snapshot(
            api,
            repo_id="fixture/d32",
            files=[(payload, remote_path)],
            private=True,
            commit_message="publish fixture",
        )


def test_closed_hf_publication_rejects_unexpected_remote_path(
    tmp_path: Path,
) -> None:
    readme = _touch(tmp_path / "README.md", b"card")
    api = _AtomicHubFixture({}, unexpected_remote_path="stale.bin")

    with pytest.raises(RuntimeError, match="unexpected=.*stale.bin"):
        uploader._publish_closed_model_snapshot(
            api,
            repo_id="fixture/d32",
            files=[(readme, "README.md")],
            private=True,
            commit_message="publish fixture",
        )


def _write_receipt(path: Path, kind: str, **fields) -> tuple[dict, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = seal_manifest(
        {
            "schema_version": fields.pop("schema_version", "1.0"),
            "kind": kind,
            **fields,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(path, value)
    return value, value["canonical_sha256"]


def _touch(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _evidence_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


@pytest.mark.parametrize(
    ("recipe_path", "data_version"),
    ((RECIPE_V3, "v3"), (RECIPE_V4, "v4")),
)
def test_main_family_dry_run_verifies_complete_publication_inventory(
    tmp_path: Path, monkeypatch, capsys, recipe_path: Path, data_version: str
) -> None:
    recipe, recipe_sha = workflow.load_recipe(recipe_path)
    base = tmp_path / "base"
    corpus_root = base / recipe["artifacts"]["corpus_root"]
    tokenizer_root = base / recipe["artifacts"]["tokenizer_root"]
    control = base / "control" / "d32"
    data_control = base / "control" / f"data_{data_version}"
    lineage_dir = control / "lineage"
    for directory in (corpus_root, tokenizer_root, control, data_control, lineage_dir):
        directory.mkdir(parents=True, exist_ok=True)

    policy = load_corpus_policy(ROOT / recipe["artifacts"]["mixture_config"])
    policy_sha = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
    source_plan, source_plan_sha = _write_receipt(
        data_control / "source_plan.json",
        "turkish_backend_source_plan",
        policy_sha256=policy_sha,
        objects=[{"rank": 0, "size_bytes": 1}],
    )
    calibration, calibration_sha = _write_receipt(
        data_control / "backend_calibration.json",
        "turkish_backend_calibration",
        policy_sha256=policy_sha,
    )
    audit_root = data_control / "sample_quality_audit"
    copied_receipts = {
        name: _touch(audit_root / f"{name}.json", name.encode("utf-8"))
        for name in (
            "cluster_receipt",
            "object_launch_receipt",
            "bucket_launch_receipt",
        )
    }
    reviewed_examples = {
        (decision, representation): _touch(
            audit_root / f"{decision}_examples.{extension}",
            f"{decision}-{representation}\n".encode("utf-8"),
        )
        for decision in ("accepted", "rejected")
        for representation, extension in (("jsonl", "jsonl"), ("plaintext", "txt"))
    }
    audit_report, sample_audit_sha = _write_receipt(
        audit_root / "sample_quality_audit_report.json",
        "turkish_bounded_backend_sample_quality_audit",
        schema_version="2.0",
        input_artifacts={
            name: _evidence_record(path, audit_root)
            for name, path in copied_receipts.items()
        },
        example_sampling={
            "files": {
                decision: {
                    representation: _evidence_record(
                        reviewed_examples[(decision, representation)], audit_root
                    )
                    for representation in ("jsonl", "plaintext")
                }
                for decision in ("accepted", "rejected")
            }
        },
    )
    mixture, mixture_sha = _write_receipt(
        data_control / "mixture_quality_approval.json",
        MIXTURE_QUALITY_APPROVAL_KIND,
        decision="accepted",
        sample_quality_audit_sha256=sample_audit_sha,
        sample_cluster_receipt_sha256="b" * 64,
        evidence_bundle={
            "schema_version": "1.0",
            "root": audit_root.relative_to(data_control).as_posix(),
            "report": _evidence_record(
                audit_root / "sample_quality_audit_report.json", audit_root
            ),
        },
    )
    backend_report, backend_report_sha = _write_receipt(
        data_control / "backend_resource_report.json",
        RESOURCE_REPORT_KIND,
        schema_version="2.0",
        policy_sha256=policy_sha,
        source_plan_sha256=source_plan_sha,
        calibration_sha256=calibration_sha,
        sample_cluster_receipt_sha256="b" * 64,
    )
    resource, resource_sha = _write_receipt(
        data_control / "resource_approval.json",
        RESOURCE_APPROVAL_KIND,
        decision="accepted",
        resource_report_sha256=backend_report_sha,
        mixture_quality_approval_sha256=mixture_sha,
        evidence_bundle={
            "schema_version": "1.0",
            "resource_report": _evidence_record(
                data_control / "backend_resource_report.json", data_control
            ),
            "mixture_quality_approval": _evidence_record(
                data_control / "mixture_quality_approval.json", data_control
            ),
        },
    )
    pack_plan, pack_sha = _write_receipt(
        data_control / "production_source_pack_plan.json",
        workflow.DATA_PREP_PACK_PLAN_KIND,
        family_id=recipe["family_id"],
        recipe_sha256=recipe_sha,
        policy_sha256=policy_sha,
        source_plan_sha256=source_plan_sha,
    )
    writer_probe, writer_sha = _write_receipt(
        data_control / "post_cluster_writer_probe.json",
        workflow.DATA_PREP_WRITER_PROBE_KIND,
        family_id=recipe["family_id"],
        recipe_sha256=recipe_sha,
        policy_sha256=policy_sha,
        source_plan_sha256=source_plan_sha,
        calibration_sha256=calibration_sha,
        backend_resource_report_sha256=backend_report_sha,
        cluster_receipt_sha256="b" * 64,
    )
    approval_paths = {
        "source_plan": data_control / "source_plan.json",
        "calibration": data_control / "backend_calibration.json",
        "backend_resource_report": data_control / "backend_resource_report.json",
        "resource_approval": data_control / "resource_approval.json",
        "mixture_quality_approval": data_control / "mixture_quality_approval.json",
    }
    storage_sample, storage_sample_sha = _write_receipt(
        data_control / "d32_data_prep_storage_sample.json",
        workflow.DATA_PREP_STORAGE_SAMPLE_KIND,
        schema_version="3.0",
        family_id=recipe["family_id"],
        recipe_sha256=recipe_sha,
        policy_sha256=policy_sha,
        source_plan_sha256=source_plan_sha,
        calibration_sha256=calibration_sha,
        backend_resource_report_sha256=backend_report_sha,
        resource_approval_sha256=resource_sha,
        mixture_quality_approval_sha256=mixture_sha,
        sample_quality_audit_sha256=sample_audit_sha,
        sample_cluster_receipt_sha256="b" * 64,
        production_pack_plan_sha256=pack_sha,
        writer_probe_sha256=writer_sha,
        sample_documents=100,
        estimated_total_documents=1_000,
        approval_evidence={
            "schema_version": "1.0",
            **{
                name: _evidence_record(path, data_control)
                for name, path in approval_paths.items()
            },
        },
    )
    storage_gate, storage_sha = _write_receipt(
        control / "data_prep_storage_gate.json",
        "d32_data_prep_storage_gate",
        schema_version="3.0",
        family_id=recipe["family_id"],
        recipe_sha256=recipe_sha,
        policy_sha256=policy_sha,
        source_plan_sha256=source_plan_sha,
        calibration_sha256=calibration_sha,
        backend_resource_report_sha256=backend_report_sha,
        resource_approval_sha256=resource_sha,
        mixture_quality_approval_sha256=mixture_sha,
        sample_quality_audit_sha256=sample_audit_sha,
        sample_cluster_receipt_sha256="b" * 64,
        production_pack_plan_sha256=pack_sha,
        sample_measurement_sha256=storage_sample_sha,
        writer_probe_sha256=writer_sha,
    )
    cluster, cluster_sha = _write_receipt(
        control / "cluster_launch.json",
        "turkish_packed_production_cluster_launch_receipt",
        cluster_completed=True,
        sample_cluster_receipt_sha256="b" * 64,
    )
    production_chain = {
        "cluster_launch_receipt_sha256": cluster_sha,
        "production_pack_plan_sha256": pack_sha,
        "resource_approval_sha256": resource_sha,
        "mixture_quality_approval_sha256": mixture_sha,
        "data_prep_storage_gate_sha256": storage_sha,
        "sample_cluster_receipt_sha256": "b" * 64,
    }

    qa_example = _touch(corpus_root / "qa" / "reviewed.jsonl", b'{"text":"iyi"}\n')
    qa_report, qa_report_sha = _write_receipt(
        corpus_root / "qa" / "qa_report.json",
        "turkish_pretrain_stratified_qa_report",
        examples={
            "accepted": {
                "path": "reviewed.jsonl",
                "size_bytes": qa_example.stat().st_size,
                "sha256": file_sha256(qa_example),
            }
        },
    )
    qa_approval, qa_approval_sha = _write_receipt(
        corpus_root / "qa" / "qa_approval.json",
        "turkish_pretrain_qa_approval",
        decision="accepted",
        qa_report_sha256=qa_report_sha,
    )
    parent_pool, parent_pool_sha = _write_receipt(
        corpus_root / "parent_pool_manifest.json",
        "turkish_pretrain_corpus",
        production_chain=production_chain,
    )
    _write_receipt(
        corpus_root / "parent_pool_ownership.json",
        "turkish_run_owned_filtered_pool",
        pool_manifest_sha256=parent_pool_sha,
    )
    packing_report, packing_report_sha = _write_receipt(
        corpus_root / "packing_preflight" / "packing_preflight_report.json",
        "turkish_packing_preflight_report",
    )
    packing_approval, packing_approval_sha = _write_receipt(
        corpus_root / "packing_preflight" / "packing_preflight_approval.json",
        "turkish_packing_preflight_approval",
        packing_report_sha256=packing_report_sha,
    )
    source_receipt_path = corpus_root / recipe["artifacts"]["source_receipt"]

    preparation_records = {}
    derived_sources = {}
    for source_id, artifact_key, kind in (
        ("macocu_genre_tr", "macocu_preparation_manifest", "turkish_macocu_genre_preparation"),
        ("mot_tr_v1_11", "mot_preparation_manifest", "turkish_high_trust_anchor_preparation"),
        ("parlamint_tr_v5_0", "parlamint_preparation_manifest", "turkish_high_trust_anchor_preparation"),
    ):
        path = base / recipe["artifacts"][artifact_key]
        manifest, manifest_sha = _write_receipt(path, kind, source_id=source_id)
        record = {"path": str(path.resolve()), "sha256": manifest_sha}
        preparation_records[artifact_key] = record
        derived_sources[source_id] = {"manifest_sha256": manifest_sha}
    source_receipt, source_receipt_sha = _write_receipt(
        source_receipt_path,
        "turkish_pretrain_source_receipt",
        derived_sources=derived_sources,
    )

    dataset, dataset_sha = _write_receipt(
        corpus_root / recipe["artifacts"]["nanochat_dataset_manifest"],
        "nanochat_dataset_manifest",
        metadata={
            "parent_pool_manifest_sha256": parent_pool_sha,
            "qa_approval_sha256": qa_approval_sha,
            "production_chain": production_chain,
        },
    )
    corpus, corpus_sha = _write_receipt(
        corpus_root / recipe["artifacts"]["corpus_manifest"],
        "turkish_pretrain_corpus",
        production_chain=production_chain,
        parent_pool_manifest_sha256=parent_pool_sha,
        quality_assurance={
            "report_sha256": qa_report_sha,
            "approval_sha256": qa_approval_sha,
        },
        packing_preflight={
            "report_sha256": packing_report_sha,
            "approval_sha256": packing_approval_sha,
        },
    )
    for name in {
        recipe["artifacts"]["validation_exposure_manifest"],
        recipe["artifacts"]["exposure_plan_index"],
        recipe["artifacts"]["packing_capacity_receipt"],
        *recipe["artifacts"]["training_exposure_manifests"].values(),
    }:
        _write_receipt(corpus_root / name, "fixture_provenance")

    tokenizer_package, tokenizer_sha = _write_receipt(
        tokenizer_root / "package_manifest.json",
        "turkish_raw_bpe_tokenizer_package",
        training_receipt_sha256="d" * 64,
        production_chain=production_chain,
        parent_corpus_manifest_sha256=parent_pool_sha,
        qa_approval_sha256=qa_approval_sha,
    )
    training_receipt, training_sha = _write_receipt(
        tokenizer_root / "training_receipt.json",
        "turkish_raw_bpe_training_receipt",
        sample_manifest_sha256="e" * 64,
        production_chain=production_chain,
        parent_corpus_manifest_sha256=parent_pool_sha,
        qa_approval_sha256=qa_approval_sha,
    )
    sample_root = corpus_root / "tokenizer_quality"
    sample_dataset, sample_dataset_sha = _write_receipt(
        sample_root / "fineweb2_manifest.json",
        "nanochat_dataset_manifest",
        metadata={
            "production_chain": production_chain,
            "parent_corpus_manifest_sha256": parent_pool_sha,
            "qa_approval_sha256": qa_approval_sha,
        },
    )
    sample_manifest, sample_sha = _write_receipt(
        sample_root / "tokenizer_sample_manifest.json",
        "turkish_raw_bpe_training_sample",
        nanochat_dataset_manifest_sha256=sample_dataset_sha,
        production_chain=production_chain,
        parent_corpus_manifest_sha256=parent_pool_sha,
        qa_approval_sha256=qa_approval_sha,
    )
    training_receipt["sample_manifest_sha256"] = sample_sha
    training_receipt["dataset_manifest_sha256"] = sample_dataset_sha
    training_receipt["name"] = recipe["artifacts"]["tokenizer_name"]
    training_receipt["vocab_size"] = recipe["model"]["vocab_size"]
    training_receipt["policy_sha256"] = policy_sha
    training_receipt["canonical_sha256"] = None
    training_receipt = seal_manifest(training_receipt)
    training_sha = training_receipt["canonical_sha256"]
    write_json_atomic(tokenizer_root / "training_receipt.json", training_receipt)
    tokenizer_package["training_receipt_sha256"] = training_sha
    tokenizer_package["canonical_sha256"] = None
    tokenizer_package = seal_manifest(tokenizer_package)
    tokenizer_sha = tokenizer_package["canonical_sha256"]
    write_json_atomic(tokenizer_root / "package_manifest.json", tokenizer_package)
    quality_root = sample_root
    sample_evidence_files = [
        {
            **_evidence_record(
                sample_root / "tokenizer_sample_manifest.json", sample_root
            ),
            "role": "tokenizer_training_sample_manifest",
        },
        {
            **_evidence_record(sample_root / "fineweb2_manifest.json", sample_root),
            "role": "tokenizer_sample_dataset_manifest",
        },
    ]
    sample_evidence = {
        "kind": "turkish_tokenizer_sample_evidence_v1",
        "sample_manifest_sha256": sample_sha,
        "dataset_manifest_sha256": sample_dataset_sha,
        "files": sample_evidence_files,
    }
    quality_report, _ = _write_receipt(
        quality_root / "quality_report.json",
        "fixture_tokenizer_quality_report",
        production_chain=production_chain,
        parent_corpus_manifest_sha256=parent_pool_sha,
        qa_approval_sha256=qa_approval_sha,
        training_receipt_sha256=training_sha,
        sample_evidence=sample_evidence,
    )
    quality_approval, _ = _write_receipt(
        quality_root / "quality_approval.json",
        "fixture_tokenizer_quality_approval",
        production_chain=production_chain,
        parent_corpus_manifest_sha256=parent_pool_sha,
        qa_approval_sha256=qa_approval_sha,
    )
    dataset["metadata"].update(
        {
            "tokenizer_quality_report_sha256": quality_report[
                "canonical_sha256"
            ],
            "tokenizer_quality_approval_sha256": quality_approval[
                "canonical_sha256"
            ],
            "tokenizer_sample_manifest_sha256": sample_sha,
            "tokenizer_sample_dataset_manifest_sha256": sample_dataset_sha,
        }
    )
    dataset["canonical_sha256"] = None
    dataset = seal_manifest(dataset)
    dataset_sha = dataset["canonical_sha256"]
    write_json_atomic(
        corpus_root / recipe["artifacts"]["nanochat_dataset_manifest"], dataset
    )
    corpus["tokenizer"] = {
        "name": recipe["artifacts"]["tokenizer_name"],
        "vocab_size": recipe["model"]["vocab_size"],
        "package_sha256": tokenizer_sha,
        "training_receipt_sha256": training_sha,
        "quality_report_sha256": quality_report["canonical_sha256"],
        "quality_approval_sha256": quality_approval["canonical_sha256"],
        "sample_evidence": {
            "relative_path": "tokenizer_quality",
            **sample_evidence,
        },
    }
    corpus["canonical_sha256"] = None
    corpus = seal_manifest(corpus)
    corpus_sha = corpus["canonical_sha256"]
    write_json_atomic(corpus_root / recipe["artifacts"]["corpus_manifest"], corpus)

    validation_sha = verify_manifest_hash(
        json.loads(
            (corpus_root / recipe["artifacts"]["validation_exposure_manifest"]).read_text()
        )
    )
    exposure_plans = {
        key: {"sha256": verify_manifest_hash(json.loads((corpus_root / name).read_text()))}
        for key, name in recipe["artifacts"]["training_exposure_manifests"].items()
    }
    preflight, preflight_sha = _write_receipt(
        control / "preflight.json",
        "d32_family_preflight_receipt",
        recipe={"canonical_sha256": recipe_sha},
        base_dir=str(base.resolve()),
        data_preparation_storage_gate_sha256=storage_sha,
        production_cluster_launch_receipt_sha256=cluster_sha,
        data_preparation_provenance={
            "policy_sha256": policy_sha,
            "source_plan_sha256": source_plan_sha,
            "calibration_sha256": calibration_sha,
            "backend_resource_report_sha256": resource["resource_report_sha256"],
            "resource_approval_sha256": resource_sha,
            "mixture_quality_approval_sha256": mixture_sha,
            "sample_quality_audit_sha256": sample_audit_sha,
            "production_pack_plan_sha256": pack_sha,
        },
        corpus={
            "root": str(corpus_root.resolve()),
            "production_chain": production_chain,
            "manifest_sha256": corpus_sha,
            "parent_pool_manifest_sha256": parent_pool_sha,
            "qa_approval_sha256": qa_approval_sha,
            "dataset_manifest_sha256": dataset_sha,
            "source_receipt_sha256": source_receipt_sha,
            "validation_exposure_manifest_sha256": validation_sha,
            "validation_payload_bytes": 100,
            "validation_documents": 1,
            "training_exposure_plans": exposure_plans,
            **preparation_records,
        },
        tokenizer={
            "root": str(tokenizer_root.resolve()),
            "name": recipe["artifacts"]["tokenizer_name"],
            "package_manifest_sha256": tokenizer_sha,
            "training_receipt_sha256": training_sha,
            "evidence_archive_relative_path": "tokenizer_quality",
            "quality_report_sha256": quality_report["canonical_sha256"],
            "quality_approval_sha256": quality_approval["canonical_sha256"],
            "sample_manifest_sha256": sample_sha,
            "sample_dataset_manifest_sha256": sample_dataset_sha,
            "sample_evidence_files": sample_evidence_files,
        },
        code={"git_commit": "1" * 40},
    )
    attention_probe, attention_probe_sha = _write_receipt(
        control / "attention_probe.json", "d32_attention_backend_probe"
    )
    static_gate, static_sha = _write_receipt(
        control / "static_launcher_gate_ws4.json", "d32_static_launcher_gate"
    )
    signal_gate, signal_sha = _write_receipt(
        control / "signal_resume_gate_ws4.json", "d32_signal_resume_gate"
    )
    wd_approval, wd_sha = _write_receipt(
        control / "wd_proxy_approval.json", "wsd_proxy_approval"
    )
    smoke8, smoke8_sha = _write_receipt(
        control / "smoke_ws8.json",
        "d32_distributed_smoke_receipt",
        world_size=8,
    )
    gate, gate_sha = _write_receipt(
        control / "production_topology_gate.json",
        "d32_production_topology_gate",
        recipe_sha256=recipe_sha,
        preflight_receipt_sha256=preflight_sha,
        passed=True,
        authorized_production_world_size=8,
        authorized_production_nodes=2,
        signal_resume_gate_sha256=signal_sha,
        smoke_8gpu_sha256=smoke8_sha,
        smoke_16gpu_sha256=None,
    )

    validation_protocol = {
        "manifest_sha256": validation_sha,
        "payload_bytes": 100,
        "documents": 1,
        "full_manifest": True,
        "packing_policy": "whole_document_no_crop_rows_before_rank_sharding",
        "bos_boundary_targets_masked": True,
        "padding_targets_masked": True,
        "eval_every_updates": 250,
        "eval_tokens_cli_unused": -1,
        "target_tokens": 10,
        "logical_rows": 1,
        "row_layout_sha256": "f" * 64,
        "padded_token_positions_world1": 2048,
        "padded_token_positions_runtime_world": 16384,
    }
    checkpoint_records = {}
    stage_by_target = {
        (str(stage["model_tag"]), int(stage["target_step"])): stage
        for stage in recipe["stages"]
    }
    for model_tag, step in {
        (recipe["checkpoints"]["trunk_model_tag"], int(item["step"]))
        for item in recipe["checkpoints"]["stable_forks"]
    } | {
        (str(item["model_tag"]), int(item["final_step"]))
        for item in recipe["checkpoints"]["finals"]
    }:
        stage = stage_by_target[(model_tag, step)]
        run_id = (
            f"{recipe['family_id']}_trunk"
            if stage["kind"] == "trunk"
            else f"{recipe['family_id']}_{stage['id']}"
        )
        bpb = 2.0
        if stage["kind"] == "cooldown_fork":
            curve_dir = (
                base
                / workflow.family_metric_roots(recipe["family_id"])["family"]
                / run_id
            )
            curve_dir.mkdir(parents=True, exist_ok=True)
            curve = CanonicalTrainingLog(
                curve_dir / "training_curve.jsonl",
                study_id=recipe["family_id"],
                run_id=run_id,
                resume=False,
            )
            curve.append(
                event_type="validation",
                updates_completed=step,
                metrics={
                    "val/all_target_nats": math.log(2.0) * 100 * bpb,
                    "val/all_target_count": 10,
                    "val/payload_nats": math.log(2.0) * 100 * bpb,
                    "val/payload_target_count": 10,
                    "val/payload_bytes": 100,
                    "val/bpb": bpb,
                },
                identities={
                    "validation_manifest_sha256": validation_sha,
                    "run_sha256": "9" * 64,
                },
            )
            curve_state = checkpoint_curve_log_state(
                curve_dir / "training_curve.jsonl", curve.state
            )
        else:
            curve_state = {
                "event_count": 1,
                "last_event_sha256": "8" * 64,
                "last_updates_completed": step,
                "recovered_truncated_bytes": 0,
                "file_sha256": "7" * 64,
            }
        protocol = {
            "protocol_version": "d32_wsd_strict_v1",
            "run_kind": "production",
            "recipe_scope": (
                "production_trunk" if stage["kind"] == "trunk" else stage["id"]
            ),
            "model_tag": model_tag,
            "num_iterations": int(stage["num_iterations"]),
            "validation": validation_protocol,
        }
        identity = build_strict_checkpoint_identity(
            study_id=recipe["family_id"],
            run_id=run_id,
            study_manifest_sha256=recipe_sha,
            run_sha256="9" * 64,
            tokenizer_artifact_sha256=tokenizer_sha,
            exposure_plan_sha256=exposure_plans[
                f"{stage['exposure_plan_family']}_ws8_seed42"
            ]["sha256"],
            optimizer_audit={"fixture": True},
            curve_log_state=curve_state,
            extra={"protocol": protocol},
        )
        checkpoint_dir = base / "base_checkpoints" / model_tag
        meta = {
            "step": step,
            "updates_completed": step,
            "strict_run_contract_sha256": "9" * 64,
            "val_bpb": bpb,
            "validation_coverage": {
                key: validation_protocol[key]
                for key in (
                    "target_tokens",
                    "payload_bytes",
                    "documents",
                    "logical_rows",
                    "padded_token_positions_world1",
                    "row_layout_sha256",
                )
            },
            "loop_state": {"min_val_bpb": bpb - 0.1},
        }
        rank_records = [
            save_strict_rank_state(
                checkpoint_dir,
                step,
                rank=rank,
                expected_world_size=8,
                optimizer_data={"rank": rank, "role": "optimizer"},
                loader_state={"rank": rank, "role": "loader"},
                rng_state={"rank": rank, "role": "rng"},
            )
            for rank in range(8)
        ]
        manifest = finalize_strict_checkpoint(
            checkpoint_dir,
            step,
            {"fixture_model": model_tag},
            meta,
            rank_records=rank_records,
            expected_world_size=8,
            identity=identity,
            updates_completed=step,
        )
        checkpoint_records[(model_tag, step)] = (manifest, manifest["canonical_sha256"])

    lineage_records = {}
    for stage in recipe["stages"]:
        target_key = (str(stage["model_tag"]), int(stage["target_step"]))
        target_manifest, target_sha = checkpoint_records[target_key]
        source = None
        if stage.get("source_step") is not None:
            source_key = (
                str(stage.get("source_model_tag", stage["model_tag"])),
                int(stage["source_step"]),
            )
            source_manifest, source_sha = checkpoint_records[source_key]
            source = {
                "model_tag": source_key[0],
                "step": source_key[1],
                "checkpoint_sha256": source_sha,
                "run_id": source_manifest["identity"]["run_id"],
            }
        receipt, digest = _write_receipt(
            lineage_dir / f"{stage['id']}.json",
            "d32_checkpoint_lineage_receipt",
            family_id=recipe["family_id"],
            stage_id=stage["id"],
            recipe_sha256=recipe_sha,
            preflight_receipt_sha256=preflight_sha,
            production_gate_sha256=gate_sha,
            wsd_proxy_approval_sha256=wd_sha,
            production_world_size=8,
            exposure_plan_key=f"{stage['exposure_plan_family']}_ws8_seed42",
            source=source,
            target={
                "model_tag": target_key[0],
                "step": target_key[1],
                "checkpoint_sha256": target_sha,
                "retention_class": (
                    "full_resumable_stable_fork"
                    if stage["kind"] == "trunk"
                    else "cooled_final_full_resumable_retained"
                ),
            },
        )
        lineage_records[stage["id"]] = (receipt, digest)

    final_evidence = workflow.collect_final_evaluation_evidence(
        recipe=recipe,
        recipe_sha=recipe_sha,
        preflight=preflight,
        preflight_sha=preflight_sha,
        gate=gate,
        gate_sha=gate_sha,
        base_dir=base,
        lineage_dir=lineage_dir,
        checkpoint_records=checkpoint_records,
        lineage_records=lineage_records,
    )
    final_quality, final_quality_sha = _write_receipt(
        control / "final_quality_approval.json",
        workflow.FINAL_MODEL_PUBLICATION_APPROVAL_KIND,
        family_id=recipe["family_id"],
        recipe_sha256=recipe_sha,
        preflight_receipt_sha256=preflight_sha,
        production_gate_sha256=gate_sha,
        final_evaluations=final_evidence,
        required_final_labels=sorted(final_evidence),
        automatic_evidence_validation_passed=True,
        manual_acceptance_required=True,
        automatic_decision=False,
        quality_decision_policy="manual_review_no_automatic_numeric_threshold",
        review_confirmation=(
            "all_three_final_fixed_validation_results_and_checkpoint_lineage_reviewed"
        ),
        reviewer="Fixture Reviewer",
        reviewed_at_utc="2026-08-21T12:00:00Z",
        decision="accepted",
        notes="fixture",
    )

    import nanochat.tokenizer_quality as tokenizer_quality
    import nanochat.turkish_backend as backend

    for name in (
        "validate_source_plan",
        "validate_backend_calibration",
        "validate_mixture_quality_approval",
        "validate_resource_approval",
        "validate_resource_projection",
    ):
        monkeypatch.setattr(backend, name, lambda *_args, **_kwargs: None)
    macocu_verifications = []
    monkeypatch.setattr(
        backend,
        "validate_macocu_preparation_manifest",
        lambda manifest, policy, root, *, verify_files: macocu_verifications.append(
            (manifest, Path(root), verify_files)
        ),
    )
    monkeypatch.setattr(workflow, "_validate_production_pack_plan", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        workflow, "_validate_data_prep_storage_sample", lambda *_args, **_kwargs: 1
    )
    monkeypatch.setattr(
        workflow, "_validate_storage_approval_evidence", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(workflow, "_validate_writer_probe", lambda *_args, **_kwargs: ({}, 1))
    monkeypatch.setattr(
        workflow, "_validate_data_prep_storage_gate_receipt", lambda *_args, **_kwargs: 1
    )
    monkeypatch.setattr(
        workflow,
        "_verify_anchor_preparation_binding",
        lambda path, *, source_id, derived_sources: preparation_records[
            "mot_preparation_manifest"
            if source_id == "mot_tr_v1_11"
            else "parlamint_preparation_manifest"
        ],
    )
    monkeypatch.setattr(
        tokenizer_quality,
        "validate_tokenizer_quality_gate",
        lambda *_args, **_kwargs: (quality_report, quality_approval),
    )
    monkeypatch.setattr(
        tokenizer_quality,
        "validate_tokenizer_sample_evidence_archive",
        lambda *_args, **_kwargs: (
            sample_manifest,
            sample_dataset,
            sample_evidence,
        ),
    )
    monkeypatch.setattr(
        uploader,
        "_select_verified_tokenizer_uploads",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                canonical_sha256=tokenizer_sha,
                manifest={"training_receipt_sha256": tokenizer_package["training_receipt_sha256"]},
            ),
            [(tokenizer_root / "package_manifest.json", "tokenizer/package_manifest.json")],
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_verify_attention_probe",
        lambda *_args, **_kwargs: (attention_probe, attention_probe_sha),
    )
    monkeypatch.setattr(
        workflow,
        "_verify_static_launcher_gate",
        lambda *_args, **_kwargs: (static_gate, static_sha),
    )
    monkeypatch.setattr(
        workflow,
        "_verify_signal_resume_gate",
        lambda *_args, **_kwargs: (signal_gate, signal_sha),
    )
    monkeypatch.setattr(
        workflow,
        "_verify_gate_and_preflight",
        lambda *_args, **_kwargs: (
            preflight,
            preflight_sha,
            gate,
            gate_sha,
            wd_approval,
            wd_sha,
        ),
    )
    monkeypatch.setattr(
        uploader,
        "run_command",
        lambda command: "1" * 40 if command[1:] == ["rev-parse", "HEAD"] else "",
    )

    active_metric_files = {}
    for category, relative_root in workflow.family_metric_roots(
        recipe["family_id"]
    ).items():
        metric_path = _touch(
            base / relative_root / f"active_{data_version}_{category}.json",
            f"active-{data_version}-{category}\n".encode("utf-8"),
        )
        active_metric_files[category] = metric_path.name
    if data_version == "v4":
        for category, relative_root in workflow.family_metric_roots(
            workflow.FAMILY_ID_V3
        ).items():
            _touch(
                base / relative_root / f"stale_v3_{category}.json",
                b"stale-v3\n",
            )
            _touch(
                base / "metrics" / category / f"stale_neutral_{category}.json",
                b"stale-neutral\n",
            )

    args = argparse.Namespace(
        repo_id="fixture/d32",
        private=True,
        dry_run=True,
        base_dir=str(base),
        family_recipe=str(recipe_path),
        preflight_receipt=str(control / "preflight.json"),
        attention_probe=str(control / "attention_probe.json"),
        wd_proxy_approval=str(control / "wd_proxy_approval.json"),
        static_launcher_gate=str(control / "static_launcher_gate_ws4.json"),
        signal_resume_gate=str(control / "signal_resume_gate_ws4.json"),
        production_gate=str(control / "production_topology_gate.json"),
        cluster_launch_receipt=str(control / "cluster_launch.json"),
        lineage_dir=str(lineage_dir),
        source_plan=str(data_control / "source_plan.json"),
        backend_calibration=str(data_control / "backend_calibration.json"),
        backend_resource_report=str(data_control / "backend_resource_report.json"),
        mixture_quality_approval=str(data_control / "mixture_quality_approval.json"),
        resource_approval=str(data_control / "resource_approval.json"),
        production_pack_plan=str(data_control / "production_source_pack_plan.json"),
        data_prep_storage_sample=str(
            data_control / "d32_data_prep_storage_sample.json"
        ),
        writer_probe=str(data_control / "post_cluster_writer_probe.json"),
        data_prep_storage_gate=str(control / "data_prep_storage_gate.json"),
        final_quality_approval=str(control / "final_quality_approval.json"),
        family_final_optimizer_policy="omit",
    )
    uploader.main_family(args)
    output = capsys.readouterr().out
    for remote in (
        "provenance/data/macocu_preparation_manifest.json",
        "provenance/data/mot_preparation_manifest.json",
        "provenance/data/parlamint_preparation_manifest.json",
        "provenance/data_controls/source_plan.json",
        "provenance/data_controls/backend_calibration.json",
        "provenance/data_controls/backend_resource_report.json",
        "provenance/data_controls/mixture_quality_approval.json",
        "provenance/data_controls/resource_approval.json",
        "provenance/data_controls/production_source_pack_plan.json",
        "provenance/data_controls/d32_data_prep_storage_sample.json",
        "provenance/data_controls/post_cluster_writer_probe.json",
        "provenance/data_controls/data_prep_storage_gate.json",
        "provenance/data_controls/sample_quality_audit/sample_quality_audit_report.json",
        "provenance/data_controls/sample_quality_audit/accepted_examples.jsonl",
        "provenance/data_controls/sample_quality_audit/accepted_examples.txt",
        "provenance/data_controls/sample_quality_audit/rejected_examples.jsonl",
        "provenance/data_controls/sample_quality_audit/rejected_examples.txt",
        "provenance/control/final_quality_approval.json",
        "provenance/control/smoke_ws8.json",
        "provenance/data/qa/reviewed.jsonl",
        "provenance/tokenizer/quality/quality_report.json",
        "provenance/tokenizer/quality/quality_approval.json",
        "provenance/tokenizer/sample/tokenizer_sample_manifest.json",
        "provenance/tokenizer/sample/fineweb2_manifest.json",
        "provenance/source/nanochat/turkish_anchor_preparation.py",
        "provenance/source/runs/uhem_d32_require_clean_code.sh",
        "provenance/source/runs/uhem_slurm_submit.sh",
        "provenance/source/scripts/prepare_turkish_anchors.py",
    ):
        assert f"-> {remote}" in output
    assert (
        f"-> provenance/source/runs/uhem_d32_{data_version}_paths.sh" in output
    )
    for category, filename in active_metric_files.items():
        assert f"-> metrics/{category}/{filename}" in output
    if data_version == "v4":
        assert "stale_v3_" not in output
        assert "stale_neutral_" not in output
    for fork in recipe["checkpoints"]["stable_forks"]:
        root = f"stable_forks/step_{int(fork['step']):06d}"
        assert f"-> {root}/completion.json" in output
        for rank in range(8):
            for role in ("optimizer", "loader", "rng"):
                assert f"-> {root}/rank_{rank:05d}_{role}.pt" in output
    for final in recipe["checkpoints"]["finals"]:
        root = f"finals/{final['label']}"
        assert f"-> {root}/model.pt" in output
        assert f"-> {root}/meta.json" in output
        assert f"-> {root}/provenance/source_completion.json" in output
        assert f"-> {root}/rank_00000_optimizer.pt" not in output
    assert final_quality_sha == final_quality["canonical_sha256"]
    assert "Dry run: remote state was not changed." in output
    assert macocu_verifications and all(item[2] is True for item in macocu_verifications)

    args.family_final_optimizer_policy = "include"
    uploader.main_family(args)
    include_output = capsys.readouterr().out
    for final in recipe["checkpoints"]["finals"]:
        root = f"finals/{final['label']}"
        assert f"-> {root}/completion.json" in include_output
        for rank in range(8):
            for role in ("optimizer", "loader", "rng"):
                assert f"-> {root}/rank_{rank:05d}_{role}.pt" in include_output

    first_final = recipe["checkpoints"]["finals"][0]
    model_path = (
        base
        / "base_checkpoints"
        / first_final["model_tag"]
        / f"strict_{int(first_final['final_step']):06d}"
        / "model.pt"
    )
    original_model = model_path.read_bytes()
    model_path.write_bytes(original_model + b"tampered")
    with pytest.raises(CheckpointIntegrityError, match="size mismatch|hash mismatch"):
        uploader.main_family(args)
    model_path.write_bytes(original_model)

    qa_example.write_bytes(qa_example.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="archived QA reviewed example differs"):
        uploader.main_family(args)


def test_final_model_publication_approval_is_manual_and_fail_closed() -> None:
    recipe = {"family_id": "fixture_family"}
    evidence = {"s12": {}, "s20": {}, "s40": {}}
    rejected = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": workflow.FINAL_MODEL_PUBLICATION_APPROVAL_KIND,
            "family_id": recipe["family_id"],
            "recipe_sha256": "1" * 64,
            "preflight_receipt_sha256": "2" * 64,
            "production_gate_sha256": "3" * 64,
            "final_evaluations": evidence,
            "required_final_labels": sorted(evidence),
            "automatic_evidence_validation_passed": True,
            "manual_acceptance_required": True,
            "automatic_decision": False,
            "quality_decision_policy": "manual_review_no_automatic_numeric_threshold",
            "review_confirmation": (
                "all_three_final_fixed_validation_results_and_checkpoint_lineage_reviewed"
            ),
            "reviewer": "Reviewer",
            "reviewed_at_utc": "2026-08-21T12:00:00Z",
            "decision": "rejected",
            "notes": "quality was not accepted",
            "canonical_sha256": None,
        }
    )
    workflow.validate_final_model_publication_approval(
        rejected,
        recipe=recipe,
        recipe_sha="1" * 64,
        preflight_sha="2" * 64,
        gate_sha="3" * 64,
        expected_evidence=evidence,
        require_accepted=False,
    )
    with pytest.raises(workflow.FamilyWorkflowError, match="requires an accepted"):
        workflow.validate_final_model_publication_approval(
            rejected,
            recipe=recipe,
            recipe_sha="1" * 64,
            preflight_sha="2" * 64,
            gate_sha="3" * 64,
            expected_evidence=evidence,
            require_accepted=True,
        )

    accepted = dict(rejected)
    accepted["decision"] = "accepted"
    accepted["notes"] = "reviewed and accepted"
    accepted["canonical_sha256"] = None
    accepted = seal_manifest(accepted)
    workflow.validate_final_model_publication_approval(
        accepted,
        recipe=recipe,
        recipe_sha="1" * 64,
        preflight_sha="2" * 64,
        gate_sha="3" * 64,
        expected_evidence=evidence,
        require_accepted=True,
    )
    stale_evidence = {**evidence, "s40": {"final_validation_bpb": 1.99}}
    with pytest.raises(workflow.FamilyWorkflowError, match="malformed or stale"):
        workflow.validate_final_model_publication_approval(
            accepted,
            recipe=recipe,
            recipe_sha="1" * 64,
            preflight_sha="2" * 64,
            gate_sha="3" * 64,
            expected_evidence=stale_evidence,
            require_accepted=True,
        )
