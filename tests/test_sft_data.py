import hashlib
import json

import pytest

from nanochat.sft_data import build_custom_sft_data_identity


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('[{"role":"user","content":"s"},{"role":"assistant","content":"c"}]\n', encoding="utf-8")
    validation.write_text('[{"role":"user","content":"v"},{"role":"assistant","content":"y"}]\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": {"sha256": "source-digest", "rows": 2},
                "outputs": {
                    "train": {
                        "path": train.name,
                        "rows": 1,
                        "bytes": train.stat().st_size,
                        "sha256": _sha256(train),
                    },
                    "validation": {
                        "path": validation.name,
                        "rows": 1,
                        "bytes": validation.stat().st_size,
                        "sha256": _sha256(validation),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return train, validation, manifest


def test_build_custom_sft_data_identity_verifies_inferred_manifest(tmp_path):
    train, validation, manifest = _write_fixture(tmp_path)

    identity = build_custom_sft_data_identity(train, validation)

    assert identity["train"]["rows"] == 1
    assert identity["validation"]["sha256"] == _sha256(validation)
    assert identity["manifest"]["sha256"] == _sha256(manifest)
    assert identity["source"]["sha256"] == "source-digest"


def test_build_custom_sft_data_identity_rejects_changed_data(tmp_path):
    train, validation, manifest = _write_fixture(tmp_path)
    train.write_text(train.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outputs.train.bytes mismatch"):
        build_custom_sft_data_identity(train, validation, manifest)


def test_build_custom_sft_data_identity_requires_explicit_manifest(tmp_path):
    train, validation, _ = _write_fixture(tmp_path)
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="manifest not found"):
        build_custom_sft_data_identity(train, validation, missing)
