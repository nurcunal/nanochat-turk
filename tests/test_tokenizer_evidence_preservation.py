from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nanochat.experiment_manifest import file_sha256, seal_manifest, write_json_atomic
from nanochat.tokenizer_quality import (
    QUALITY_REPORT_KIND,
    TOKENIZER_SAMPLE_DATASET_MANIFEST_FILE,
    TOKENIZER_SAMPLE_EVIDENCE_KIND,
    TOKENIZER_SAMPLE_MANIFEST_FILE,
    validate_tokenizer_sample_evidence_archive,
)
from nanochat.turkish_corpus import TurkishCorpusError


SHA = "a" * 64


def _record(path: Path, role: str) -> dict:
    return {
        "path": path.name,
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _fixture(root: Path) -> tuple[dict, dict]:
    root.mkdir()
    chain = {"data_prep_storage_gate_sha256": "b" * 64}
    dataset = seal_manifest(
        {
            "schema_version": "1.0",
            "manifest_type": "dataset",
            "profile": "strict",
            "dataset": {
                "repo_id": "local-composite/test",
                "path": "tokenizer_sample",
                "requested_revision": "c" * 40,
                "resolved_revision": "c" * 40,
                "repo_type": "dataset",
            },
            "text_column": "text",
            "ordered_files": [
                {"path": "train.parquet", "size_bytes": 1, "sha256": "d" * 64},
                {
                    "path": "validation.parquet",
                    "size_bytes": 1,
                    "sha256": "e" * 64,
                },
            ],
            "validation_file": "validation.parquet",
            "created_by": {"git_commit": "f" * 40, "tool": "test"},
            "metadata": {
                "policy_sha256": SHA,
                "production_chain": chain,
                "parent_corpus_manifest_sha256": "1" * 64,
                "qa_approval_sha256": "2" * 64,
            },
            "canonical_sha256": None,
        }
    )
    sample = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_raw_bpe_training_sample",
            "name": "tr_test_bpe_32k",
            "vocab_size": 32_768,
            "policy_sha256": SHA,
            "production_chain": chain,
            "parent_corpus_manifest_sha256": "1" * 64,
            "qa_approval_sha256": "2" * 64,
            "nanochat_dataset_manifest_sha256": dataset["canonical_sha256"],
            "canonical_sha256": None,
        }
    )
    training = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_raw_bpe_training_receipt",
            "name": sample["name"],
            "vocab_size": sample["vocab_size"],
            "policy_sha256": SHA,
            "production_chain": chain,
            "parent_corpus_manifest_sha256": "1" * 64,
            "qa_approval_sha256": "2" * 64,
            "sample_manifest_sha256": sample["canonical_sha256"],
            "dataset_manifest_sha256": dataset["canonical_sha256"],
            "canonical_sha256": None,
        }
    )
    sample_path = root / TOKENIZER_SAMPLE_MANIFEST_FILE
    dataset_path = root / TOKENIZER_SAMPLE_DATASET_MANIFEST_FILE
    write_json_atomic(sample_path, sample)
    write_json_atomic(dataset_path, dataset)
    evidence = {
        "kind": TOKENIZER_SAMPLE_EVIDENCE_KIND,
        "sample_manifest_sha256": sample["canonical_sha256"],
        "dataset_manifest_sha256": dataset["canonical_sha256"],
        "files": [
            _record(sample_path, "tokenizer_training_sample_manifest"),
            _record(dataset_path, "tokenizer_sample_dataset_manifest"),
        ],
    }
    report = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": QUALITY_REPORT_KIND,
            "training_receipt_sha256": training["canonical_sha256"],
            "policy_sha256": SHA,
            "production_chain": chain,
            "parent_corpus_manifest_sha256": "1" * 64,
            "qa_approval_sha256": "2" * 64,
            "heldout_validation": {
                "dataset_manifest_sha256": dataset["canonical_sha256"],
                "sample_manifest_sha256": sample["canonical_sha256"],
                "path": "validation.parquet",
                "size_bytes": 1,
                "sha256": "e" * 64,
            },
            "sample_evidence": evidence,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(root / "quality_report.json", report)
    return training, report


def _validate(root: Path, training: dict, report: dict) -> None:
    validate_tokenizer_sample_evidence_archive(
        root,
        training_receipt=training,
        expected_quality_report_sha256=report["canonical_sha256"],
        expected_production_chain=training["production_chain"],
        expected_parent_corpus_manifest_sha256=training[
            "parent_corpus_manifest_sha256"
        ],
        expected_qa_approval_sha256=training["qa_approval_sha256"],
    )


def test_valid_archive_is_bound_to_training_receipt(tmp_path: Path) -> None:
    root = tmp_path / "quality"
    training, report = _fixture(root)
    sample, dataset, evidence = validate_tokenizer_sample_evidence_archive(
        root,
        training_receipt=training,
        expected_quality_report_sha256=report["canonical_sha256"],
    )
    assert sample["canonical_sha256"] == training["sample_manifest_sha256"]
    assert dataset["canonical_sha256"] == training["dataset_manifest_sha256"]
    assert [item["path"] for item in evidence["files"]] == [
        TOKENIZER_SAMPLE_MANIFEST_FILE,
        TOKENIZER_SAMPLE_DATASET_MANIFEST_FILE,
    ]


def test_archive_rejects_stale_file_bytes(tmp_path: Path) -> None:
    root = tmp_path / "quality"
    training, report = _fixture(root)
    path = root / TOKENIZER_SAMPLE_MANIFEST_FILE
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(TurkishCorpusError, match="failed verification"):
        _validate(root, training, report)


def test_archive_rejects_self_consistent_manifest_substitution(tmp_path: Path) -> None:
    root = tmp_path / "quality"
    training, report = _fixture(root)
    sample_path = root / TOKENIZER_SAMPLE_MANIFEST_FILE
    replacement = copy.deepcopy(json.loads(sample_path.read_text(encoding="utf-8")))
    replacement["name"] = "substituted_bpe_32k"
    replacement["canonical_sha256"] = None
    replacement = seal_manifest(replacement)
    write_json_atomic(sample_path, replacement)
    changed_report = copy.deepcopy(report)
    changed_report["sample_evidence"]["sample_manifest_sha256"] = replacement[
        "canonical_sha256"
    ]
    changed_report["sample_evidence"]["files"][0] = _record(
        sample_path, "tokenizer_training_sample_manifest"
    )
    changed_report["canonical_sha256"] = None
    changed_report = seal_manifest(changed_report)
    write_json_atomic(root / "quality_report.json", changed_report)
    with pytest.raises(TurkishCorpusError, match="lineage drift"):
        _validate(root, training, changed_report)


def test_archive_rejects_evidence_path_substitution(tmp_path: Path) -> None:
    root = tmp_path / "quality"
    training, report = _fixture(root)
    changed_report = copy.deepcopy(report)
    changed_report["sample_evidence"]["files"][0]["path"] = "../sample.json"
    changed_report["canonical_sha256"] = None
    changed_report = seal_manifest(changed_report)
    write_json_atomic(root / "quality_report.json", changed_report)
    with pytest.raises(TurkishCorpusError, match="inventory contract drift"):
        _validate(root, training, changed_report)


def test_archive_rejects_symlink_substitution(tmp_path: Path) -> None:
    root = tmp_path / "quality"
    training, report = _fixture(root)
    evidence_path = root / TOKENIZER_SAMPLE_MANIFEST_FILE
    outside = tmp_path / "outside.json"
    outside.write_bytes(evidence_path.read_bytes())
    evidence_path.unlink()
    evidence_path.symlink_to(outside)
    with pytest.raises(TurkishCorpusError, match="failed verification"):
        _validate(root, training, report)
