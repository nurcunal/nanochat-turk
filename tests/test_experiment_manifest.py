from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nanochat.experiment_manifest import (
    LEGACY_PROFILE,
    ManifestValidationError,
    canonical_json,
    canonical_json_bytes,
    file_sha256,
    load_json_strict,
    manifest_sha256,
    seal_manifest,
    validate_artifact_inventory,
    validate_artifact_manifest,
    validate_dataset_manifest,
    validate_file_record,
    validate_ordered_file_records,
    verify_file_inventory,
    verify_manifest_hash,
)


SHA_A = "a" * 64


def dataset_manifest() -> dict:
    return {
        "schema_version": "1.0",
        "manifest_type": "dataset",
        "profile": "strict",
        "dataset": {
            "repo_id": "HuggingFaceFW/fineweb-2",
            "repo_type": "dataset",
            "path": "data/tur_Latn/train",
            "requested_revision": "0123456789abcdef0123456789abcdef01234567",
            "resolved_revision": "0123456789abcdef0123456789abcdef01234567",
        },
        "text_column": "text",
        "ordered_files": [
            {"path": "000_00000.parquet", "size_bytes": 12, "sha256": SHA_A},
            {"path": "000_00001.parquet", "size_bytes": 34, "sha256": "b" * 64},
        ],
        "validation_file": "000_00001.parquet",
        "created_by": {"git_commit": "f" * 40},
    }


def artifact_manifest(records: list[dict]) -> dict:
    return {
        "schema_version": "1.0",
        "manifest_type": "artifact",
        "profile": "strict",
        "artifact_origin": {"artifact_id": "bpe_32768", "kind": "tokenizer"},
        "packaging_provenance": {
            "git_commit": "f" * 40,
            "source_date_epoch": 1_787_750_400,
        },
        "data_provenance": {"dataset_manifest_sha256": SHA_A},
        "segmenter_provenance": None,
        "metric_environment": {"python": "3.10"},
        "publication_target": {"repo_id": "nurcunal/nanochat-turk-tokenizers"},
        "files": records,
    }


def test_canonical_json_is_compact_sorted_utf8_and_preserves_array_order() -> None:
    value = {"z": [2, 1], "a": "Türkçe"}
    assert canonical_json(value) == '{"a":"Türkçe","z":[2,1]}\n'


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"profile":"strict","profile":"legacy-partial"}', "duplicate JSON key"),
        ('{"value":NaN}', "non-finite JSON number"),
    ],
)
def test_strict_json_loader_rejects_ambiguous_values(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ManifestValidationError, match=message):
        load_json_strict(path)


def test_manifest_hash_excludes_self_hash_and_is_key_order_independent() -> None:
    first = {"z": 1, "a": {"second": 2, "first": 1}}
    reordered = {"a": {"first": 1, "second": 2}, "z": 1}
    digest = manifest_sha256(first)

    assert digest == manifest_sha256(reordered)
    assert digest == manifest_sha256({**first, "canonical_sha256": SHA_A})
    assert digest != manifest_sha256({"z": 1, "a": [1, 2]})
    assert digest != manifest_sha256({"z": 1, "a": [2, 1]})


def test_canonical_json_can_materialize_the_self_hash_exclusion_rule() -> None:
    value = {"payload": 1, "canonical_sha256": SHA_A}
    assert canonical_json_bytes(
        value, self_hash_field="canonical_sha256"
    ) == b'{"canonical_sha256":null,"payload":1}\n'


def test_seal_and_verify_manifest_hash_without_mutating_input() -> None:
    manifest = dataset_manifest()
    sealed = seal_manifest(manifest)

    assert "canonical_sha256" not in manifest
    assert sealed["canonical_sha256"] == manifest_sha256(sealed)
    assert verify_manifest_hash(sealed) == sealed["canonical_sha256"]
    validate_dataset_manifest(sealed)

    sealed["text_column"] = "changed"
    with pytest.raises(ManifestValidationError, match="does not match"):
        verify_manifest_hash(sealed)


def test_strict_dataset_manifest_validates() -> None:
    validate_dataset_manifest(dataset_manifest())


def test_strict_dataset_requires_validation_identity_and_full_creator_commit() -> None:
    missing_validation = dataset_manifest()
    missing_validation.pop("validation_file")
    with pytest.raises(ManifestValidationError, match="validation_file"):
        validate_dataset_manifest(missing_validation)

    short_commit = dataset_manifest()
    short_commit["created_by"]["git_commit"] = "fedcba9876543210"
    with pytest.raises(ManifestValidationError, match="full lowercase Git commit"):
        validate_dataset_manifest(short_commit)


@pytest.mark.parametrize(
    "revision",
    [None, "", "main", "MASTER", "Latest", "HEAD", "null"],
)
@pytest.mark.parametrize("field", ["requested_revision", "resolved_revision"])
def test_strict_dataset_rejects_null_empty_and_mutable_revisions(
    field: str, revision: str | None
) -> None:
    manifest = dataset_manifest()
    manifest["dataset"][field] = revision

    with pytest.raises(ManifestValidationError, match=field):
        validate_dataset_manifest(manifest)


def test_legacy_dataset_allows_honest_mutable_and_unresolved_revision() -> None:
    manifest = dataset_manifest()
    manifest["profile"] = LEGACY_PROFILE
    manifest["dataset"]["requested_revision"] = "main"
    manifest["dataset"]["resolved_revision"] = None
    manifest["created_by"]["git_commit"] = None

    validate_dataset_manifest(manifest)

    with pytest.raises(ManifestValidationError, match="disagrees"):
        validate_dataset_manifest(manifest, profile="strict")


def test_legacy_profile_must_be_declared_in_the_manifest() -> None:
    manifest = dataset_manifest()
    manifest.pop("profile")
    manifest["dataset"]["requested_revision"] = "main"
    manifest["dataset"]["resolved_revision"] = None

    with pytest.raises(ManifestValidationError, match="explicitly declare"):
        validate_dataset_manifest(manifest, profile="legacy")


def test_manifest_cannot_use_legacy_alias_as_its_declared_profile() -> None:
    manifest = dataset_manifest()
    manifest["profile"] = "legacy"

    with pytest.raises(ManifestValidationError, match="legacy-partial"):
        validate_dataset_manifest(manifest)


def test_explicit_null_profile_is_not_treated_as_an_omitted_strict_default() -> None:
    manifest = dataset_manifest()
    manifest["profile"] = None

    with pytest.raises(ManifestValidationError, match="profile"):
        validate_dataset_manifest(manifest)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/file.bin",
        "C:/absolute/file.bin",
        "../outside.bin",
        "nested/../../outside.bin",
        "./file.bin",
        "nested/./file.bin",
        "nested//file.bin",
        "nested\\file.bin",
        "file.bin\n",
    ],
)
def test_file_record_rejects_unsafe_or_non_normal_paths(path: str) -> None:
    with pytest.raises(ManifestValidationError, match="path"):
        validate_file_record({"path": path, "size_bytes": 1, "sha256": SHA_A})


@pytest.mark.parametrize("size", [-1, 1.2, True, "1"])
def test_file_record_rejects_invalid_sizes(size: object) -> None:
    with pytest.raises(ManifestValidationError, match="size_bytes"):
        validate_file_record({"path": "file.bin", "size_bytes": size, "sha256": SHA_A})


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "g" * 64, None])
def test_file_record_rejects_noncanonical_sha256(digest: object) -> None:
    with pytest.raises(ManifestValidationError, match="sha256"):
        validate_file_record({"path": "file.bin", "size_bytes": 1, "sha256": digest})


def test_ordered_file_records_reject_duplicate_paths() -> None:
    records = [
        {"path": "same.bin", "size_bytes": 1, "sha256": SHA_A},
        {"path": "same.bin", "size_bytes": 1, "sha256": SHA_A},
    ]
    with pytest.raises(ManifestValidationError, match="duplicate path"):
        validate_ordered_file_records(records)


def test_artifact_inventory_verifies_size_hash_and_exact_file_set(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    payload = nested / "tokenizer.bin"
    payload.write_bytes(b"abc")
    records = [
        {
            "path": "nested/tokenizer.bin",
            "role": "tokenizer_model",
            "size_bytes": 3,
            "sha256": file_sha256(payload),
        }
    ]
    manifest = artifact_manifest(records)

    validate_artifact_manifest(manifest)
    validate_artifact_inventory(manifest, tmp_path, require_exact=True)

    extra = tmp_path / "local-note.txt"
    extra.write_text("not published", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="unrecorded files"):
        verify_file_inventory(tmp_path, records, require_exact=True)
    verify_file_inventory(
        tmp_path,
        records,
        require_exact=True,
        ignored_paths=["local-note.txt"],
    )

    payload.write_bytes(b"xyz")
    with pytest.raises(ManifestValidationError, match="sha256"):
        verify_file_inventory(tmp_path, records)


def test_strict_artifact_requires_full_packaging_git_commit() -> None:
    manifest = artifact_manifest(
        [{"path": "a.bin", "role": "payload", "size_bytes": 1, "sha256": SHA_A}]
    )
    manifest["packaging_provenance"]["git_commit"] = "main"
    with pytest.raises(ManifestValidationError, match="full lowercase Git commit"):
        validate_artifact_manifest(manifest)


def test_artifact_manifest_hash_is_deterministic_but_inventory_order_is_significant() -> None:
    first_record = {
        "path": "a.bin",
        "role": "payload",
        "size_bytes": 1,
        "sha256": SHA_A,
    }
    second_record = {
        "path": "b.bin",
        "role": "metadata",
        "size_bytes": 2,
        "sha256": "b" * 64,
    }
    forward = artifact_manifest([first_record, second_record])
    reverse = artifact_manifest([second_record, first_record])

    assert manifest_sha256(forward) == manifest_sha256(copy.deepcopy(forward))
    assert manifest_sha256(forward) != manifest_sha256(reverse)


def test_schema_documents_are_valid_json_and_expose_the_same_core_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for filename, manifest_type in (
        ("dataset-manifest.schema.json", "dataset"),
        ("artifact-manifest.schema.json", "artifact"),
    ):
        schema = json.loads((repo_root / "schemas" / filename).read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["properties"]["manifest_type"]["const"] == manifest_type
        if manifest_type == "dataset":
            file_record = schema["$defs"]["fileRecord"]
            assert file_record["required"] == ["path", "size_bytes", "sha256"]
        else:
            file_record = schema["$defs"]["artifactFileRecord"]
            assert file_record["required"] == ["path", "role", "size_bytes", "sha256"]
