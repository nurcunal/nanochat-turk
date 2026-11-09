from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nanochat.turkish_corpus import (
    CORPUS_NAME_V3,
    CORPUS_NAME_V4,
    TOKENIZER_NAME_V4,
    TurkishCorpusError,
    load_corpus_policy,
    validate_corpus_policy,
)
from scripts import turkish_packed_sample


ROOT = Path(__file__).resolve().parents[1]
POLICY_V3 = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_v3.json"
POLICY_V4 = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_v4.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v4_is_exact_fineweb_backbone_delta_from_v3() -> None:
    v3 = _load_json(POLICY_V3)
    v4 = load_corpus_policy(POLICY_V4)

    assert v3["name"] == CORPUS_NAME_V3
    assert v4["name"] == CORPUS_NAME_V4
    assert v4["schema_version"] == v3["schema_version"] == "3.0"
    assert v4["tokenizer_training"]["name"] == TOKENIZER_NAME_V4

    expected_sources = [
        source for source in v3["sources"] if source["id"] != "hplt3_tr"
    ]
    assert v4["sources"] == expected_sources
    assert {source["id"] for source in v4["sources"]} == {
        "fineweb2_hq_tr",
        "fineweb2_strict_tr_v3",
        "macocu_genre_tr",
        "finewiki_tr",
        "mot_tr_v1_11",
        "parlamint_tr_v5_0",
    }
    assert v4["deduplication"]["source_priority"] == [
        source_id
        for source_id in v3["deduplication"]["source_priority"]
        if source_id != "hplt3_tr"
    ]

    expected_mixture = copy.deepcopy(
        [bucket for bucket in v3["mixture"] if bucket["source_id"] != "hplt3_tr"]
    )
    strict = next(
        bucket
        for bucket in expected_mixture
        if bucket["id"] == "fineweb2_strict_general"
    )
    strict["weight"] = 0.7365
    strict["source_cap"] = 0.7365
    assert v4["mixture"] == expected_mixture
    assert sum(bucket["weight"] for bucket in v4["mixture"]) == pytest.approx(1.0)
    assert not any(bucket["source_id"] == "hplt3_tr" for bucket in v4["mixture"])

    for key in (
        "language_policy",
        "content_policy",
        "splits",
        "quality_assurance",
        "materialization",
    ):
        assert v4[key] == v3[key]
    expected_tokenizer = copy.deepcopy(v3["tokenizer_training"])
    expected_tokenizer["name"] = TOKENIZER_NAME_V4
    assert v4["tokenizer_training"] == expected_tokenizer

    strict_source = next(
        source
        for source in v4["sources"]
        if source["id"] == "fineweb2_strict_tr_v3"
    )
    assert strict_source["derivation"]["expected_object_count"] == 30
    assert strict_source["derivation"]["expected_total_bytes"] == 134_789_283_815
    assert strict_source["derivation"]["raw_fallback_allowed"] is False


def test_v4_frozen_source_and_weight_contracts_fail_closed() -> None:
    v3 = _load_json(POLICY_V3)
    v4 = _load_json(POLICY_V4)

    with_hplt = copy.deepcopy(v4)
    with_hplt["sources"].append(
        next(source for source in v3["sources"] if source["id"] == "hplt3_tr")
    )
    with_hplt["deduplication"]["source_priority"].append("hplt3_tr")
    with pytest.raises(TurkishCorpusError, match="source inventory"):
        validate_corpus_policy(with_hplt)

    wrong_weight = copy.deepcopy(v4)
    strict = next(
        bucket
        for bucket in wrong_weight["mixture"]
        if bucket["id"] == "fineweb2_strict_general"
    )
    strict["weight"] = 0.7364
    strict["source_cap"] = 0.7364
    with pytest.raises(TurkishCorpusError, match="mixture ids/weights"):
        validate_corpus_policy(wrong_weight)

    wrong_tokenizer = copy.deepcopy(v4)
    wrong_tokenizer["tokenizer_training"]["name"] = "tr_general_raw_bpe_32k_v3"
    with pytest.raises(TurkishCorpusError, match="tokenizer identity"):
        validate_corpus_policy(wrong_tokenizer)


def test_v3_remains_supported_and_packed_sample_defaults_to_v4() -> None:
    assert load_corpus_policy(POLICY_V3)["name"] == CORPUS_NAME_V3
    assert turkish_packed_sample.DEFAULT_POLICY == Path(
        "configs/pretrain/tr_d32_turkish_general_v4.json"
    )
    assert ("3.0", CORPUS_NAME_V4) in (
        turkish_packed_sample.SUPPORTED_POLICY_IDENTITIES
    )
