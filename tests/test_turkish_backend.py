from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from nanochat.turkish_backend import (
    _source_lid,
    _stage_source_object,
    select_resource_sample_ranks,
)
from nanochat.turkish_corpus import (
    TurkishCorpusError,
    audit_document,
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


def test_resource_sample_covers_every_source_and_hplt_quality_bin():
    plan = {
        "objects": [
            {"rank": 0, "source_id": "web", "size_bytes": 20, "uri": "web-b"},
            {"rank": 1, "source_id": "web", "size_bytes": 10, "uri": "web-a"},
            {"rank": 2, "source_id": "hplt3_tr", "wds_bin": 10, "size_bytes": 8, "uri": "10"},
            {"rank": 3, "source_id": "hplt3_tr", "wds_bin": 9, "size_bytes": 9, "uri": "9"},
            {"rank": 4, "source_id": "hplt3_tr", "wds_bin": 8, "size_bytes": 7, "uri": "8"},
            {"rank": 5, "source_id": "hplt3_tr", "wds_bin": 8, "size_bytes": 6, "uri": "8-small"},
        ]
    }

    assert select_resource_sample_ranks(plan) == [1, 2, 3, 5]


def test_no_code_gate_rejects_assignment_and_builtin_call_snippet():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = """Bugün küçük bir sayı listesinin nasıl sıralanacağını anlatan bir yazı okudum.
Yazar önce değerleri hazırlıyor, ardından sonucu ekranda gösteriyor ve her adımı ayrıntılı biçimde açıklıyor.
sayilar = [3, 1, 2]
sonuc = sorted(sayilar)
print(sonuc)
Bu örnek ilk bakışta kısa görünse de yazının geri kalanı kullanılan yöntemi ve beklenen çıktıyı uzun uzun tartışıyor."""

    decision = audit_document(
        text,
        source_lid_ok=True,
        content_policy=content_policy,
    )

    assert decision.accepted is False
    assert decision.reason == "code_content"
    assert decision.metrics["code_line_fraction"] > 0.02


def test_no_code_gate_keeps_ordinary_turkish_conversation():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = (
        "Dün akşam arkadaşlarımla mahalledeki küçük lokantaya gittik. "
        "Yemekleri beklerken gün içinde yaşadıklarımızı anlattık, hafta sonu için "
        "plan yaptık ve uzun zamandır görmediğimiz komşularımızdan söz ettik. "
        "Daha sonra çay içip eve yürüdük; sohbet doğal, sıcak ve oldukça keyifliydi."
    )

    decision = audit_document(
        text,
        source_lid_ok=True,
        content_policy=content_policy,
    )

    assert decision.accepted is True
    assert decision.reason == "accepted"
