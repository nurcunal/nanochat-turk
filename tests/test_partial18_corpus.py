from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat.experiment_manifest import (
    canonical_json,
    load_json_strict,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.strict_dataloader import verify_strict_dataset
from nanochat.turkish_backend import _INTERNAL_SCHEMA
from nanochat.turkish_corpus import assign_split, canonical_text_hash
from scripts.materialize_partial18_turkish_corpus import (
    build_input_inventory,
    materialize_partial18,
    verify_source_evidence,
    _validate_candidate_lid_fields,
)


class CharacterTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def encode(self, text, *, prepend=None, num_threads=8):
        del num_threads

        def one(value: str) -> list[int]:
            result = [1] * len(value)
            return result if prepend is None else [int(prepend), *result]

        return [one(value) for value in text] if isinstance(text, list) else one(text)


def _policy() -> dict:
    return {
        "name": "partial18-test",
        "sources": [
            {
                "id": "source_a",
                "adapter": {
                    "turkish_values": ["tur"],
                    "source_lid_min_probability": 0.9,
                },
            }
        ],
        "language_policy": {
            "independent_audit": {
                "required_top_label": "tur_Latn",
                "document_min_probability": 0.8,
                "document_min_margin": 0.2,
                "paragraph_min_probability": 0.65,
                "paragraph_min_margin": 0.1,
                "max_failed_long_paragraph_fraction": 0.1,
            }
        },
        "splits": {
            "unit": "dedup_cluster",
            "algorithm": "sha256_threshold_v1",
            "seed": "partial18-test-split",
            "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
        },
        "mixture": [
            {
                "id": "general",
                "source_id": "source_a",
                "weight": 1.0,
                "selector": {},
            }
        ],
    }


def _texts_for_split(
    policy: dict, split: str, count: int, *, long: bool = False
) -> list[str]:
    result: list[str] = []
    index = 0
    while len(result) < count:
        prefix = ("uzun-" + "ğ" * 10_050) if long else "kısa"
        text = f"{prefix}-{split}-{index}"
        if assign_split(canonical_text_hash(text), policy["splits"]) == split:
            result.append(text)
        index += 1
    return result


def _row(text: str, rank: int, index: int) -> dict:
    return {
        "text": text,
        "source_id": "source_a",
        "document_id": f"doc-{rank}-{index}",
        "url": f"https://example.test/{rank}/{index}",
        "source_lid_label": "tur",
        "source_lid_probability": 0.99,
        "lid_label": "tur_Latn",
        "lid_probability": 0.98,
        "lid_margin": 0.8,
        "paragraph_min_probability": 0.9,
        "paragraph_min_margin": 0.7,
        "failed_long_paragraph_fraction": 0.0,
        "dedup_cluster_id": canonical_text_hash(text),
        # These deliberately look non-passing. The salvage materializer must
        # preserve and count them, not treat placeholders as gate evidence.
        "dedup_keep": False,
        "quality_score": 0.0,
        "wds_bin": None,
        "web-register": "{}",
        "genre": "",
        "pii_replacements": 7,
        "harmful_signal_hits": 3,
        "quality_filter_flags": '["placeholder-not-evidence"]',
        "formatting_changes": '{"placeholder":true}',
        "candidate_rank": rank,
        "candidate_doc_index": index,
    }


def _write_candidates(root: Path, policy: dict) -> dict[tuple[int, int], dict]:
    texts = [
        *_texts_for_split(policy, "train", 16),
        *_texts_for_split(policy, "train", 1, long=True),
        *_texts_for_split(policy, "val", 2),
        *_texts_for_split(policy, "val", 1, long=True),
        *_texts_for_split(policy, "test", 2),
        *_texts_for_split(policy, "test", 1, long=True),
    ]
    original: dict[tuple[int, int], dict] = {}
    midpoint = len(texts) // 2
    for rank, rank_texts in ((0, texts[:midpoint]), (1, texts[midpoint:])):
        rows = [_row(text, rank, index) for index, text in enumerate(rank_texts)]
        original.update({(rank, index): row for index, row in enumerate(rows)})
        path = root / "objects" / f"{rank:05d}" / "candidates.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=_INTERNAL_SCHEMA),
            path,
            row_group_size=3,
        )
    return original


def test_partial18_materializer_preserves_train_text_and_is_honest(tmp_path: Path) -> None:
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    input_root = tmp_path / "input"
    originals = _write_candidates(input_root, policy)
    inventory, _schema = build_input_inventory(input_root, (0, 1))
    for item in inventory:
        item["source_id"] = "source_a"
    inventory_sha = hashlib.sha256(
        canonical_json(inventory).encode("utf-8")
    ).hexdigest()
    output = tmp_path / "output"
    result = materialize_partial18(
        input_root=input_root,
        output_dir=output,
        policy=policy,
        policy_path=policy_path,
        tokenizer=CharacterTokenizer(),
        tokenizer_lineage={
            "tokenizer_package_sha256": "a" * 64,
            "tokenizer_training_receipt_sha256": "b" * 64,
            "tokenizer_name": "fixture",
            "tokenizer_job_id": 2,
            "tokenizer_input_inventory_sha256": inventory_sha,
            "tokenizer_input_inventory": inventory,
            "tokenizer_input_root": str(input_root.resolve()),
            "tokenizer_policy_sha256": hashlib.sha256(
                canonical_json(policy).encode("utf-8")
            ).hexdigest(),
            "tokenizer_split_policy": policy["splits"],
            "tokenizer_producer_git_commit": "e" * 40,
            "tokenizer_production_eligible": False,
        },
        source_evidence={
            "source_plan_sha256": "f" * 64,
            "source_plan_file_sha256": "1" * 64,
            "sample_ranks_file_sha256": "2" * 64,
            "sample_ranks_canonical_payload_sha256": "3" * 64,
            "selected_rank_records": [
                {"rank": 0, "source_id": "source_a", "uri": "https://a/0"},
                {"rank": 1, "source_id": "source_a", "uri": "https://a/1"},
            ],
        },
        expected_ranks=(0, 1),
        git_commit="d" * 40,
        source_job_provenance={
            "job_id": 1,
            "state": "CANCELLED",
            "partition": "cpu2dq",
            "allocated_cpus": 128,
        },
        rows_per_train_shard=32,
        row_group_rows=2,
        tokenizer_batch_size=2,
        tokenizer_threads=1,
        eval_max_tokens_with_bos=64,
        capacity_world_sizes=(1,),
        capacity_B=1,
        capacity_T=16,
        capacity_buffer_size=2,
        capacity_global_batch_tokens=16,
        verify_capacity_parity=False,
    )

    assert result["production_eligible"] is False
    contract = verify_strict_dataset(output, verify_bytes=True)
    ordered = contract["ordered_relative"]
    assert ordered == sorted(ordered)
    assert ordered[-1] == "validation.parquet"
    assert "test/test.parquet" not in ordered

    train_keys = []
    for relative in contract["train_relative"]:
        train_keys.extend(pq.read_table(output / relative, columns=["shuffle_key"])["shuffle_key"].to_pylist())
    assert train_keys == sorted(train_keys)
    assert len(train_keys) == len(set(train_keys))
    assert all(train_keys)

    observed: dict[tuple[int, int], dict] = {}
    for relative in [*contract["train_relative"], "validation.parquet", "test/test.parquet"]:
        for row in pq.read_table(output / relative).to_pylist():
            key = (row["candidate_rank"], row["candidate_doc_index"])
            observed[key] = row
            original = originals[key]
            for field in _INTERNAL_SCHEMA.names:
                assert row[field] == original[field]
            assert row["canonical_text_sha256"] == canonical_text_hash(row["text"])
            assert row["dedup_cluster_id"] == row["canonical_text_sha256"]
            assert row["split"] == assign_split(
                row["canonical_text_sha256"], policy["splits"]
            )
            assert row["encoded_tokens_with_bos"] == len(row["text"]) + 1

    assigned = {
        key: row
        for key, row in originals.items()
        if assign_split(canonical_text_hash(row["text"]), policy["splits"])
        == "train"
    }
    assert assigned.keys() <= observed.keys()
    long_train = max(assigned.values(), key=lambda row: len(row["text"]))
    assert len(long_train["text"]) > 10_000
    assert observed[(long_train["candidate_rank"], long_train["candidate_doc_index"])][
        "text"
    ] == long_train["text"]

    corpus = load_json_strict(output / "partial18_corpus_manifest.json")
    verify_manifest_hash(corpus)
    assert corpus["production_eligible"] is False
    assert corpus["row_policy"]["document_character_cap"] is None
    assert corpus["row_policy"]["text_transform"] == "none"
    gates = corpus["candidate_gate_status"]
    assert gates["source_language_id_end_to_end_attested"] is False
    assert gates["independent_glotlid_end_to_end_attested"] is False
    assert gates["global_minhash_completed"] is False
    assert gates["official_gopher_filters_completed"] is False
    assert gates["official_fineweb_filters_completed"] is False
    assert gates["local_quality_filters_completed"] is False
    assert gates["pii_filter_completed"] is False
    assert gates["code_filter_completed"] is False
    assert gates["manual_qa_completed"] is False
    row_evidence = corpus["candidate_stage_row_field_evidence"]
    assert row_evidence["rows_validated"] == len(originals)
    assert row_evidence[
        "all_rows_match_frozen_source_lid_labels_and_thresholds"
    ] is True
    assert row_evidence[
        "all_rows_match_frozen_independent_glotlid_labels_and_thresholds"
    ] is True
    placeholders = corpus["candidate_placeholder_fields"]
    assert placeholders["dedup_keep"]["used_for_admission_or_dedup"] is False
    assert placeholders["quality_filter_flags"]["used_as_quality_evidence_or_filter"] is False

    counts = corpus["counts"]
    assert counts["assigned"]["totals"]["documents"] == len(originals)
    assert counts["excluded_eval_oversize"]["totals"]["documents"] == 2
    assert counts["excluded_eval_oversize"]["by_split"]["train"]["documents"] == 0
    assert counts["written"]["by_split"]["train"] == counts["assigned"]["by_split"][
        "train"
    ]
    assert len(observed) + 2 == len(originals)

    capacity = load_json_strict(output / "partial18_capacity_receipt.json")
    verify_manifest_hash(capacity)
    world = capacity["simulation"]["worlds"]["1"]
    assert world["first_epoch_common_complete_microbatches_per_rank"] > 0
    assert world["first_epoch_optimizer_aligned_packed_positions"] > 0
    assert world["documents_cropped"] > 0

    materialization = load_json_strict(output / "partial18_materialization_receipt.json")
    verify_manifest_hash(materialization)
    assert materialization["input_documents"] == len(originals)
    assert materialization["excluded_eval_oversize_documents"] == 2
    assert materialization["production_eligible"] is False


def test_partial18_rejects_train_shard_topology_before_materialization(
    tmp_path: Path,
) -> None:
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    input_root = tmp_path / "input"
    _write_candidates(input_root, policy)
    inventory, _schema = build_input_inventory(input_root, (0, 1))
    for item in inventory:
        item["source_id"] = "source_a"
    inventory_sha = hashlib.sha256(
        canonical_json(inventory).encode("utf-8")
    ).hexdigest()
    lineage = {
        "tokenizer_package_sha256": "a" * 64,
        "tokenizer_input_inventory": inventory,
        "tokenizer_input_inventory_sha256": inventory_sha,
        "tokenizer_input_root": str(input_root.resolve()),
        "tokenizer_policy_sha256": hashlib.sha256(
            canonical_json(policy).encode("utf-8")
        ).hexdigest(),
        "tokenizer_split_policy": policy["splits"],
    }
    try:
        materialize_partial18(
            input_root=input_root,
            output_dir=tmp_path / "bad-output",
            policy=policy,
            policy_path=policy_path,
            tokenizer=CharacterTokenizer(),
            tokenizer_lineage=lineage,
            source_evidence={
                "selected_rank_records": [
                    {"rank": 0, "source_id": "source_a"},
                    {"rank": 1, "source_id": "source_a"},
                ]
            },
            expected_ranks=(0, 1),
            git_commit="d" * 40,
            source_job_provenance={},
            rows_per_train_shard=30,
            row_group_rows=2,
        )
    except ValueError as exc:
        assert "row-group count divisible by 16" in str(exc)
    else:  # pragma: no cover - explicit adversarial assertion
        raise AssertionError("invalid ws16 shard topology was accepted")
    assert not (tmp_path / "bad-output").exists()


def test_source_plan_and_sample_rank_binding_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "fixture",
            "objects": [
                {"rank": 0, "source_id": "source_a", "uri": "https://a/0"},
                {"rank": 1, "source_id": "source_a", "uri": "https://a/1"},
            ],
            "canonical_sha256": None,
        }
    )
    plan_path = tmp_path / "source_plan.json"
    ranks_path = tmp_path / "resource_sample_ranks.json"
    write_json_atomic(plan_path, plan)
    write_json_atomic(ranks_path, {"ranks": [0, 1], "slurm_array": "0,1"})
    monkeypatch.setattr(
        "scripts.materialize_partial18_turkish_corpus.validate_source_plan",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.materialize_partial18_turkish_corpus.select_resource_sample_ranks",
        lambda _plan: [0, 1],
    )
    evidence = verify_source_evidence(
        plan_path,
        ranks_path,
        policy=_policy(),
        expected_ranks=(0, 1),
        expected_source_plan_sha256=plan["canonical_sha256"],
    )
    assert evidence["source_plan_sha256"] == plan["canonical_sha256"]
    assert [item["rank"] for item in evidence["selected_rank_records"]] == [0, 1]

    write_json_atomic(ranks_path, {"ranks": [0, 1], "slurm_array": "1,0"})
    with pytest.raises(ValueError, match="sample ranks differ"):
        verify_source_evidence(
            plan_path,
            ranks_path,
            policy=_policy(),
            expected_ranks=(0, 1),
            expected_source_plan_sha256=plan["canonical_sha256"],
        )


def test_candidate_lid_row_evidence_fails_closed() -> None:
    row = _row("Türkçe deneme", 0, 0)
    _validate_candidate_lid_fields(row, "source_a", _policy())
    row["lid_probability"] = 0.79
    with pytest.raises(ValueError, match="GlotLID fields fail"):
        _validate_candidate_lid_fields(row, "source_a", _policy())
