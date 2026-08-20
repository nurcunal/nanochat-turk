from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.experiment_manifest import file_sha256
from nanochat.turkish_corpus import (
    D32_EVAL_ROW_CAPACITY,
    FragmentWriter,
    _write_eval_split,
    allocate_fallback_quotas,
    load_corpus_policy,
)


class _CharacterTokenizer:
    @staticmethod
    def encode(texts, *, num_threads):
        assert num_threads >= 1
        return [[2] * len(text) for text in texts]


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
