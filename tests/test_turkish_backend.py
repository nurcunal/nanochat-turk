from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nanochat.turkish_backend import _source_lid, _stage_source_object
from nanochat.turkish_corpus import (
    TurkishCorpusError,
    load_corpus_policy,
    select_mixture_bucket,
    source_lid_result,
    strict_hplt_register_scores,
)


POLICY = Path("configs/pretrain/tr_d32_turkish_general_v1.json")


def _hplt_adapter():
    policy = load_corpus_policy(POLICY)
    return next(item for item in policy["sources"] if item["id"] == "hplt3_tr")[
        "adapter"
    ]


def test_hplt_parallel_language_probability_pairing_is_index_exact():
    adapter = _hplt_adapter()
    record = {
        "lang": ["eng_Latn", "tur_Latn", "deu_Latn"],
        "prob": [0.999, 0.91, 0.95],
    }
    assert source_lid_result(record, adapter, strict_schema=True) == (
        "tur_Latn",
        0.91,
        True,
    )
    assert _source_lid(record, adapter) == ("tur_Latn", 0.91, True)


@pytest.mark.parametrize(
    "record",
    [
        {"lang": ["tur_Latn", "eng_Latn"], "prob": [0.99]},
        {"lang": ["tur_Latn"], "prob": "0.99"},
        {"lang": ["tur_Latn"], "prob": [float("nan")]},
    ],
)
def test_hplt_parallel_language_schema_fails_closed(record):
    with pytest.raises(TurkishCorpusError):
        source_lid_result(record, _hplt_adapter(), strict_schema=True)


def test_literal_web_register_and_mt_gate_are_enforced():
    policy = load_corpus_policy(POLICY)
    accepted = {
        "wds_bin": 10,
        "web-register": {"SP": 0.70, "MT": 0.10, "IN": 0.05},
    }
    assert strict_hplt_register_scores(accepted) == {
        "SP": 0.70,
        "MT": 0.10,
        "IN": 0.05,
    }
    assert select_mixture_bucket("hplt3_tr", accepted, policy)[0] == (
        "hplt_conversation"
    )
    rejected_mt = {
        "wds_bin": 10,
        "web-register": {"SP": 0.70, "MT": 0.20},
    }
    assert select_mixture_bucket("hplt3_tr", rejected_mt, policy) is None
    with pytest.raises(TurkishCorpusError, match="literal 'web-register'"):
        strict_hplt_register_scores({"web_register": {"SP": 0.9}})


def test_local_source_checksum_drift_fails_closed(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"bounded fixture")
    destination = tmp_path / "staged.bin"
    item = {
        "uri": source.as_uri(),
        "size_bytes": source.stat().st_size,
        "expected_checksums": [
            {"algorithm": "sha256", "value": hashlib.sha256(b"wrong").hexdigest()}
        ],
    }
    with pytest.raises(TurkishCorpusError, match="checksum drift"):
        _stage_source_object(item, destination)
