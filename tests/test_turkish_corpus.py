from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat.experiment_manifest import canonical_json, file_sha256, load_json_strict, seal_manifest
from nanochat.turkish_corpus import (
    D32_EVAL_ROW_CAPACITY,
    FragmentWriter,
    MACOCU_MD5,
    MACOCU_SIZE_BYTES,
    MACOCU_SOURCE_ID,
    MACOCU_SOURCE_URL,
    SOURCE_RECEIPT_KIND,
    TurkishCorpusError,
    _write_eval_split,
    allocate_fallback_quotas,
    archive_source_receipt,
    iter_input_records,
    load_corpus_policy,
)


class _CharacterTokenizer:
    @staticmethod
    def encode(texts, *, num_threads):
        assert num_threads >= 1
        return [[2] * len(text) for text in texts]


def test_jsonl_zstd_input_is_decompressed_exactly_once(tmp_path: Path):
    source = tmp_path / "sample.jsonl.zst"
    with pa.output_stream(str(source), compression="zstd") as stream:
        stream.write(b'{"id":"bir","text":"Merhaba dunya"}\n')

    assert list(iter_input_records(source)) == [
        {
            "id": "bir",
            "text": "Merhaba dunya",
            "_input_path": "sample.jsonl.zst",
        }
    ]


def _row(text: str, index: int) -> dict:
    return {
        "text": text,
        "source_id": "fineweb2_hq_tr",
        "mixture_id": "fineweb2_hq_general",
        "document_id": f"d{index}",
        "url": "",
        "cluster_id": f"{index:064x}",
        "shuffle_key": f"{index:064x}",
        "quality_score": 1.0,
        "register_bucket": "not_applicable",
    }


def test_eval_materialization_seals_no_crop_and_counts_long_rejections(tmp_path: Path):
    pool = tmp_path / "pool"
    pool.mkdir()
    source = pool / "val.parquet"
    rows = [_row("kısa", 1), _row("x" * D32_EVAL_ROW_CAPACITY, 2)]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=FragmentWriter._schema),
        source,
        row_group_size=2,
    )
    manifest = {
        "files": [
            {
                "path": source.name,
                "split": "val",
                "mixture_id": "fineweb2_hq_general",
                "size_bytes": source.stat().st_size,
                "sha256": file_sha256(source),
            }
        ]
    }
    destination = tmp_path / "validation.parquet"
    record, tokens, documents, _peak, policy = _write_eval_split(
        pool,
        manifest,
        {},
        _CharacterTokenizer(),
        destination,
        "val",
    )
    assert record["rows"] == 1
    assert tokens == {"fineweb2_hq_general": 5}
    assert documents == {"fineweb2_hq_general": 1}
    assert policy["policy"] == "whole_document_no_crop"
    assert policy["max_encoded_tokens_with_bos"] == 2049
    assert policy["rejected_long_documents"] == 1
    assert policy["rejected_long_encoded_tokens_with_bos"] == 2050


def test_fallback_allocation_uses_approved_measured_source_weights():
    policy = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v1.json")
    measured = {
        "cosmos_everyday": 0.10,
        "finepdfs_edu_general": 0.08,
        "fineweb2_general": 0.15,
        "fineweb2_hq_general": 0.20,
        "finewiki_reference": 0.02,
        "hplt_conversation": 0.20,
        "hplt_general": 0.25,
    }
    capacity = {key: 10_000 for key in measured}
    effective, ledger, desired = allocate_fallback_quotas(
        policy,
        target_tokens=1_000,
        safe_capacity=capacity,
        source_weights=measured,
    )
    assert desired["hplt_conversation"] == 200
    assert desired["hplt_general"] == 250
    assert effective == desired
    assert ledger == []


def _source_receipt_for_policy(policy: dict) -> dict:
    derived_sources = {}
    if any(source["id"] == MACOCU_SOURCE_ID for source in policy["sources"]):
        derived_sources[MACOCU_SOURCE_ID] = {
            "manifest_sha256": "d" * 64,
            "upstream": {
                "uri": MACOCU_SOURCE_URL,
                "md5": MACOCU_MD5,
                "size_bytes": MACOCU_SIZE_BYTES,
            },
        }
    return seal_manifest(
        {
            "schema_version": "1.0",
            "kind": SOURCE_RECEIPT_KIND,
            "policy_sha256": hashlib.sha256(
                canonical_json(policy).encode("utf-8")
            ).hexdigest(),
            "derived_sources": derived_sources,
            "sources": [
                {
                    "id": source["id"],
                    "repo_id": source["repo_id"],
                    "resolved_revision": source["resolved_revision"],
                    "license_id": source["license_id"],
                    "files": [
                        {
                            "uri": f"https://example.invalid/{source['id']}.parquet",
                            "checksum": {"algorithm": "sha256", "value": "a" * 64},
                            "size_bytes": 1,
                        }
                    ],
                }
                for source in policy["sources"]
            ],
            "canonical_sha256": None,
        }
    )


def test_source_receipt_archive_preserves_and_verifies_sealed_identity(tmp_path: Path):
    policy = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v2.json")
    receipt = _source_receipt_for_policy(policy)

    archived = archive_source_receipt(
        tmp_path,
        receipt,
        policy,
        expected_sha256=receipt["canonical_sha256"],
    )

    assert archived == receipt
    assert load_json_strict(tmp_path / "source_receipt.json") == receipt


def test_source_receipt_archive_rejects_parent_hash_mismatch(tmp_path: Path):
    policy = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v2.json")
    receipt = _source_receipt_for_policy(policy)

    with pytest.raises(TurkishCorpusError, match="hash binding differs"):
        archive_source_receipt(
            tmp_path,
            receipt,
            policy,
            expected_sha256="b" * 64,
        )

    assert not (tmp_path / "source_receipt.json").exists()
