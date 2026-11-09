from __future__ import annotations

import copy
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
    TOKENIZER_SAMPLE_SEED_V2,
    TOKENIZER_SAMPLE_SEED_V3,
    TurkishCorpusError,
    _tokenizer_sample_seed_for_policy,
    _write_eval_split,
    allocate_fallback_quotas,
    archive_source_receipt,
    iter_input_records,
    load_corpus_policy,
    validate_corpus_policy,
    validate_source_receipt,
)


def test_historical_v1_tokenizer_sample_fallback_seed_is_stable():
    v1 = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v1.json")
    v2 = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v2.json")
    v3 = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v3.json")

    assert "sampling_seed" not in v1["tokenizer_training"]
    assert _tokenizer_sample_seed_for_policy(v1) == TOKENIZER_SAMPLE_SEED_V2
    assert _tokenizer_sample_seed_for_policy(v2) == TOKENIZER_SAMPLE_SEED_V2
    assert _tokenizer_sample_seed_for_policy(v3) == TOKENIZER_SAMPLE_SEED_V3


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


def test_v3_policy_freezes_pdf_ocr_free_sources_mix_and_tokenizer():
    policy = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v3.json")
    sources = {source["id"]: source for source in policy["sources"]}
    assert set(sources) == {
        "hplt3_tr",
        "fineweb2_hq_tr",
        "fineweb2_strict_tr_v3",
        "macocu_genre_tr",
        "finewiki_tr",
        "mot_tr_v1_11",
        "parlamint_tr_v5_0",
    }
    assert "fineweb2_tr" not in sources
    assert "finepdfs_edu_tr" not in sources
    assert {source["text_origin"] for source in sources.values()} <= {
        "born_digital_text",
        "structured_text",
    }
    assert sources["hplt3_tr"]["selected_wds_bins"] == [8, 9]
    hplt_selectors = {
        bucket["id"]: bucket["selector"]
        for bucket in policy["mixture"]
        if bucket["source_id"] == "hplt3_tr"
    }
    assert hplt_selectors == {
        "hplt_wds8_general": {
            "wds_bins": [8],
            "register_any": ["IN", "NA", "HI", "IP"],
            "register_min_probability": 0.4,
            "max_machine_translated_probability": 0.1,
            "max_lyrical_probability": 0.1,
        },
        "hplt_wds9_general": {
            "wds_bins": [9],
            "register_any": ["IN", "NA", "HI", "IP"],
            "register_min_probability": 0.4,
            "max_machine_translated_probability": 0.1,
            "max_lyrical_probability": 0.1,
        },
    }
    strict = sources["fineweb2_strict_tr_v3"]["derivation"]
    assert strict["expected_object_count"] == 30
    assert strict["expected_total_bytes"] == 134_789_283_815
    assert strict["raw_fallback_allowed"] is False
    assert policy["tokenizer_training"]["name"] == "tr_general_raw_bpe_32k_v3"
    assert sum(bucket["weight"] for bucket in policy["mixture"]) == pytest.approx(1.0)
    capacity = policy["materialization"]["packing_capacity_gate"]
    assert (
        capacity["horizon_optimizer_steps"]["s40_margin"]
        * capacity["global_batch_tokens"]
        == 68_451_041_280
    )
    assert capacity["preferred_min_first_epoch_packed_positions"] == 34_225_520_640
    assert capacity["hard_min_first_epoch_packed_positions"] == 17_112_760_320


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda policy: policy["sources"][0].__setitem__("text_origin", "ocr_text"), "text_origin"),
        (
            lambda policy: next(
                source
                for source in policy["sources"]
                if source["id"] == "hplt3_tr"
            ).__setitem__("selected_wds_bins", [8, 9, 10]),
            "WDS 8 and 9",
        ),
        (
            lambda policy: next(
                source
                for source in policy["sources"]
                if source["id"] == "fineweb2_strict_tr_v3"
            )["derivation"].__setitem__("raw_fallback_allowed", True),
            "derivation contract",
        ),
        (
            lambda policy: policy["content_policy"].__setitem__(
                "max_mojibake_sequence_hits", 1
            ),
            "integer zero",
        ),
        (
            lambda policy: policy["language_policy"]["independent_audit"].__setitem__(
                "document_min_probability", 0.0
            ),
            "independent_audit.document_min_probability",
        ),
        (
            lambda policy: next(
                source
                for source in policy["sources"]
                if source["id"] == "hplt3_tr"
            )["adapter"].__setitem__("source_lid_min_probability", 0.0),
            "source-LID contract",
        ),
        (
            lambda policy: next(
                source
                for source in policy["sources"]
                if source["id"] == "fineweb2_strict_tr_v3"
            )["adapter"].__setitem__("source_lid_min_probability", 0.0),
            "source-LID contract",
        ),
        (
            lambda policy: policy["content_policy"].__setitem__(
                "max_code_line_fraction", 1.0
            ),
            "content_policy.max_code_line_fraction",
        ),
        (
            lambda policy: next(
                bucket
                for bucket in policy["mixture"]
                if bucket["id"] == "hplt_wds8_general"
            )["selector"].__setitem__("register_any", ["SP"]),
            "HPLT selector contract",
        ),
        (
            lambda policy: next(
                bucket
                for bucket in policy["mixture"]
                if bucket["id"] == "hplt_wds9_general"
            )["selector"].__setitem__("max_lyrical_probability", 1.0),
            "HPLT selector contract",
        ),
    ],
)
def test_v3_policy_safety_contracts_fail_closed(mutation, message):
    policy = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v3.json")
    mutated = copy.deepcopy(policy)
    mutation(mutated)
    with pytest.raises(TurkishCorpusError, match=message):
        validate_corpus_policy(mutated)


def test_v3_source_receipt_binds_strict_fineweb_and_native_anchor_admission(
    tmp_path: Path, monkeypatch
):
    policy = load_corpus_policy("configs/pretrain/tr_d32_turkish_general_v3.json")
    strict_source = next(
        source
        for source in policy["sources"]
        if source["id"] == "fineweb2_strict_tr_v3"
    )
    derived = {
        MACOCU_SOURCE_ID: {
            "manifest_sha256": "d" * 64,
            "upstream": {
                "uri": MACOCU_SOURCE_URL,
                "md5": MACOCU_MD5,
                "size_bytes": MACOCU_SIZE_BYTES,
            },
        },
        "fineweb2_strict_tr_v3": {
            "contract": strict_source["derivation"],
            "resolved_inventory": {
                "object_count": 30,
                "total_bytes": 134_789_283_815,
                "sha256": "e2f10096b18e2329ddad230e99fbcf77e0294ebe6d1f9f652b077b57fb04adca",
                "hash_semantics": "canonical_json_uri_sorted_uri_size_bytes_expected_checksums_v1",
            },
            "admission": {
                "candidate_source_id": "fineweb2_strict_tr_v3",
                "raw_source_id": "fineweb2_tr",
                "only_passing_rows_enter_candidates": True,
                "direct_raw_fallback": False,
                "processing_binding_sha256": strict_source["derivation"][
                    "processing_binding_sha256"
                ],
                "audit_policy_binding_sha256": strict_source["derivation"][
                    "audit_policy_binding_sha256"
                ],
            },
        },
    }
    manifests = {}
    for index, anchor_id in enumerate(("mot_tr_v1_11", "parlamint_tr_v5_0"), 1):
        root = tmp_path / anchor_id
        root.mkdir()
        manifest_path = root / "manifest.json"
        manifest_path.write_text("{}\n")
        manifest = {
            "source_id": anchor_id,
            "canonical_sha256": str(index) * 64,
            "production_acceptance": {
                "stage": "accepted_production",
                "eligible_for_production": True,
            },
        }
        manifests[root] = manifest
        derived[anchor_id] = {
            "manifest_uri": manifest_path.resolve().as_uri(),
            "manifest_sha256": manifest["canonical_sha256"],
            "downstream_admission": {
                "preparer_automatically_admits_training": False,
                "backend_turkish_no_code_audit_required": True,
            },
        }

    import nanochat.turkish_anchor_preparation as anchors

    monkeypatch.setattr(
        anchors,
        "validate_anchor_preparation",
        lambda root, **_kwargs: manifests[Path(root)],
    )
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": SOURCE_RECEIPT_KIND,
            "policy_sha256": hashlib.sha256(
                canonical_json(policy).encode("utf-8")
            ).hexdigest(),
            "derived_sources": derived,
            "sources": [
                {
                    "id": source["id"],
                    "repo_id": source["repo_id"],
                    "resolved_revision": source["resolved_revision"],
                    "license_id": source["license_id"],
                    "files": [
                        {
                            "uri": f"https://example.invalid/{source['id']}",
                            "checksum": {
                                "algorithm": "sha256",
                                "value": "f" * 64,
                            },
                            "size_bytes": 1,
                        }
                    ],
                }
                for source in policy["sources"]
            ],
            "canonical_sha256": None,
        }
    )
    validate_source_receipt(receipt, policy)
    tampered = copy.deepcopy(receipt)
    tampered["derived_sources"]["fineweb2_strict_tr_v3"]["admission"][
        "direct_raw_fallback"
    ] = True
    tampered = seal_manifest(tampered)
    with pytest.raises(TurkishCorpusError, match="admission drift"):
        validate_source_receipt(tampered, policy)
