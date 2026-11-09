from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
import struct
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import nanochat.turkish_backend as backend
import nanochat.turkish_corpus as corpus
from nanochat.experiment_manifest import (
    canonical_json,
    seal_manifest,
    write_json_atomic,
)
from nanochat.turkish_backend import (
    MIXTURE_QUALITY_APPROVAL_KIND,
    RESOURCE_APPROVAL_KIND,
    RESOURCE_BILLING_CONTRACT,
    RESOURCE_REPORT_KIND,
    _resource_projection_accounting,
    _source_lid,
    _stage_source_object,
    seal_resource_approval,
    select_resource_sample_ranks,
    validate_resource_approval,
    validate_resource_projection,
    validate_mixture_quality_approval,
)
from nanochat.turkish_corpus import (
    HPLT_WEB_REGISTER_KEYS,
    TurkishCorpusError,
    audit_document,
    dominant_register,
    iter_input_records,
    load_corpus_policy,
    select_mixture_bucket,
    source_lid_result,
    strict_hplt_register_scores,
)
from scripts.turkish_data_backend import build_parser


POLICY = Path("configs/pretrain/tr_d32_turkish_general_v1.json")
POLICY_V2 = Path("configs/pretrain/tr_d32_turkish_general_v2.json")
POLICY_V3 = Path("configs/pretrain/tr_d32_turkish_general_v3.json")


def _resource_accounting():
    stage_wall = {
        "download": 10.0,
        "score_lid": 20.0,
        "minhash_signature": 30.0,
        "minhash_buckets": 40.0,
        "priority_cluster_quality_format": 20.0,
    }
    stage_process_cpu = {
        "download": 2.0,
        "score_lid": 8.0,
        "minhash_signature": 10.0,
        "minhash_buckets": 30.0,
        "priority_cluster_quality_format": 10.0,
    }
    return _resource_projection_accounting(
        stage_wall,
        stage_process_cpu,
        safety_factor=1.5,
        billable_cpus_per_job=128,
    )


def _sealed_resource_report(policy: dict, plan_sha256: str) -> dict:
    accounting = _resource_accounting()
    return seal_manifest(
        {
            "schema_version": "2.0",
            "kind": RESOURCE_REPORT_KIND,
            "policy_sha256": hashlib.sha256(
                canonical_json(policy).encode("utf-8")
            ).hexdigest(),
            "source_plan_sha256": plan_sha256,
            "calibration_sha256": "c" * 64,
            "sample_cluster_receipt_sha256": "e" * 64,
            "billing_contract": dict(RESOURCE_BILLING_CONTRACT),
            "sample_selection": {
                "ranks": [],
                "covers_hplt_wds_bins": [],
                "hplt_selected_objects": [],
            },
            "projection": {
                "safety_factor": 1.5,
                "candidate_documents": 200.0,
                "raw_largest_object_bytes": 100.0,
                "candidate_bytes": 100.0,
                "signature_bytes": 100.0,
                "duplicate_edge_bytes": 100.0,
                "backend_output_bytes": 100.0,
                "peak_disk_components_before_safety_factor": {
                    "raw_largest_object_bytes": 100.0,
                    "candidate_bytes": 100.0,
                    "signature_bytes": 100.0,
                    "duplicate_edge_bytes": 100.0,
                    "backend_output_bytes": 100.0,
                },
                "peak_disk_bytes_before_safety_factor": 500.0,
                "peak_disk_bytes_with_safety_factor": 750.0,
                "peak_disk_model": backend.RESOURCE_PEAK_DISK_MODEL,
                "cluster_scaling": {
                    "sample_candidate_documents": 100,
                    "projected_candidate_scale": 2.0,
                    "sample_edge_participating_documents": 10,
                    "projected_edge_participating_documents": 20.0,
                    "sample_peak_rss_bytes": 1024**3,
                    "projected_peak_rss_bytes": 2 * 1024**3,
                    "projected_peak_rss_bytes_with_safety_factor": 3 * 1024**3,
                    "projected_wall_seconds_with_safety_factor": 30.0,
                    "rss_projection_model": (
                        "sample_peak_rss_times_max_one_and_candidate_scale"
                    ),
                },
                **accounting,
            },
            "limits": {
                "effective_peak_limit_bytes": 10_000,
                "cluster_memory_limit_bytes": 192 * 1024**3,
                "cluster_wall_limit_seconds": 172_800,
            },
            "automated_gate_passed": True,
            "manual_approval_required": True,
            "canonical_sha256": None,
        }
    )


def _hplt_adapter():
    policy = load_corpus_policy(POLICY)
    return next(item for item in policy["sources"] if item["id"] == "hplt3_tr")[
        "adapter"
    ]


def _hplt_registers(**overrides: float) -> dict[str, float]:
    scores = dict.fromkeys(HPLT_WEB_REGISTER_KEYS, 0.0)
    scores.update(overrides)
    return scores


def test_v3_hplt_resolver_fetches_only_wds8_and_wds9(monkeypatch):
    policy = load_corpus_policy(POLICY_V3)
    source = next(item for item in policy["sources"] if item["id"] == "hplt3_tr")
    map_payload = b"\n".join(
        f"https://example.invalid/{wds}_0.jsonl.zst".encode()
        for wds in (8, 9, 10)
    ) + b"\n"
    md5_payload = b"\n".join(
        f"{digit * 32}  {wds}_0.jsonl.zst".encode()
        for digit, wds in zip("123", (8, 9, 10), strict=True)
    ) + b"\n"
    monkeypatch.setattr(backend, "HPLT_MAP_SHA256", hashlib.sha256(map_payload).hexdigest())
    monkeypatch.setattr(
        backend, "HPLT_MD5_LIST_SHA256", hashlib.sha256(md5_payload).hexdigest()
    )

    class Response:
        def __init__(self, content=b"", headers=None):
            self.content = content
            self.headers = headers or {}

        def raise_for_status(self):
            return None

    def request_get(uri, **_kwargs):
        return Response(map_payload if uri == source["source_url"] else md5_payload)

    def request_head(_uri, **_kwargs):
        return Response(headers={"Content-Length": "123"})

    objects = backend._parse_hplt_objects(
        source, request_get=request_get, request_head=request_head
    )
    assert [item["wds_bin"] for item in objects] == [8, 9]


def test_strict_fineweb_derivation_binds_complete_inventory(monkeypatch):
    policy = load_corpus_policy(POLICY_V3)
    source = copy.deepcopy(
        next(
            item
            for item in policy["sources"]
            if item["id"] == "fineweb2_strict_tr_v3"
        )
    )
    objects = [
        {
            "uri": f"https://example.invalid/{letter}.parquet",
            "size_bytes": size,
            "expected_checksums": [
                {"algorithm": "sha256", "value": letter * 64}
            ],
        }
        for letter, size in (("a", 11), ("b", 13))
    ]
    projection = backend._fineweb2_inventory_projection(objects)
    inventory_hash = hashlib.sha256(canonical_json(projection).encode()).hexdigest()
    monkeypatch.setattr(backend, "V3_FINEWEB2_OBJECT_COUNT", 2)
    monkeypatch.setattr(backend, "V3_FINEWEB2_TOTAL_BYTES", 24)
    monkeypatch.setattr(backend, "V3_FINEWEB2_INVENTORY_SHA256", inventory_hash)
    source["derivation"].update(
        expected_object_count=2,
        expected_total_bytes=24,
        expected_inventory_sha256=inventory_hash,
    )
    provenance = backend._strict_fineweb_derivation(source, policy, objects)
    assert provenance["resolved_inventory"]["sha256"] == inventory_hash
    assert provenance["admission"]["direct_raw_fallback"] is False

    mutated = copy.deepcopy(objects)
    mutated[0]["size_bytes"] += 1
    with pytest.raises(TurkishCorpusError, match="complete frozen"):
        backend._strict_fineweb_derivation(source, policy, mutated)


def test_anchor_resolver_requires_sealed_accepted_production_manifest(
    tmp_path: Path, monkeypatch
):
    policy = load_corpus_policy(POLICY_V3)
    source = next(
        item for item in policy["sources"] if item["id"] == "mot_tr_v1_11"
    )
    root = tmp_path / "mot"
    shard = root / "data" / "part-00000.jsonl.zst"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"fixture")
    (root / "manifest.json").write_text("{}\n")
    manifest = {
        "kind": "turkish_high_trust_anchor_preparation",
        "preparer_version": "turkish_anchor_preparation_v2",
        "source_id": "mot_tr_v1_11",
        "canonical_sha256": "a" * 64,
        "production_acceptance": {
            "stage": "accepted_production",
            "eligible_for_production": True,
            "receipt": {"canonical_sha256": "b" * 64},
        },
        "acquisition_receipt": {"canonical_sha256": "c" * 64},
        "clean": {"documents": 1, "logical_jsonl_sha256": "d" * 64},
        "artifacts": {
            "data": {
                "logical_jsonl_sha256": "d" * 64,
                "totals": {"shards": 1, "rows": 1, "size_bytes": 7},
                "shards": [
                    {
                        "path": "data/part-00000.jsonl.zst",
                        "size_bytes": 7,
                        "sha256": "e" * 64,
                    }
                ],
            }
        },
    }
    import nanochat.turkish_anchor_preparation as anchors

    monkeypatch.setattr(
        anchors, "validate_anchor_preparation", lambda *_args, **_kwargs: manifest
    )
    objects, provenance = backend._parse_anchor_objects(source, root / "manifest.json")
    assert len(objects) == 1
    assert objects[0]["preparation_manifest_sha256"] == "a" * 64
    assert provenance["downstream_admission"] == {
        "preparer_automatically_admits_training": False,
        "backend_turkish_no_code_audit_required": True,
    }

    manifest["production_acceptance"] = {
        "stage": "discovery_unaccepted",
        "eligible_for_production": False,
    }
    with pytest.raises(TurkishCorpusError, match="accepted-production"):
        backend._parse_anchor_objects(source, root / "manifest.json")


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


def test_source_lid_clamps_only_fasttext_scale_roundoff():
    adapter = {
        "language_field": "language",
        "language_probability_field": "language_score",
        "source_lid_min_probability": 0.9,
        "turkish_values": ["tur"],
    }
    assert source_lid_result(
        {"language": "tur", "language_score": 1.0000100135803223},
        adapter,
        strict_schema=True,
    ) == ("tur", 1.0, True)
    with pytest.raises(TurkishCorpusError, match="fastText tolerance"):
        source_lid_result(
            {"language": "tur", "language_score": 1.001},
            adapter,
            strict_schema=True,
        )


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
        "web-register": _hplt_registers(SP=0.70, MT=0.10, IN=0.05),
    }
    assert strict_hplt_register_scores(accepted) == accepted["web-register"]
    assert select_mixture_bucket("hplt3_tr", accepted, policy)[0] == (
        "hplt_conversation"
    )
    rejected_mt = {
        "wds_bin": 10,
        "web-register": _hplt_registers(SP=0.70, MT=0.20),
    }
    assert select_mixture_bucket("hplt3_tr", rejected_mt, policy) is None
    with pytest.raises(TurkishCorpusError, match="literal 'web-register'"):
        strict_hplt_register_scores({"web_register": {"SP": 0.9}})


def test_hplt_web_register_inventory_is_the_frozen_official_schema():
    assert HPLT_WEB_REGISTER_KEYS == (
        "MT",
        "LY",
        "SP",
        "ID",
        "NA",
        "HI",
        "IN",
        "OP",
        "IP",
        "it",
        "ne",
        "sr",
        "nb",
        "re",
        "en",
        "ra",
        "dtp",
        "fi",
        "lt",
        "rv",
        "ob",
        "rs",
        "av",
        "ds",
        "ed",
    )


@pytest.mark.parametrize("missing_label", ["LY", "MT", "IN"])
def test_hplt_web_register_requires_every_official_label(missing_label):
    scores = _hplt_registers(IN=0.7)
    del scores[missing_label]
    with pytest.raises(TurkishCorpusError, match="official 25-key schema"):
        strict_hplt_register_scores({"web-register": scores})
    with pytest.raises(TurkishCorpusError, match="official 25-key schema"):
        select_mixture_bucket(
            "hplt3_tr",
            {"wds_bin": 8, "web-register": scores},
            load_corpus_policy(POLICY_V3),
        )


def test_hplt_web_register_rejects_unknown_extra_label():
    scores = _hplt_registers(IN=0.7)
    scores["unknown"] = 0.0
    with pytest.raises(TurkishCorpusError, match="official 25-key schema"):
        strict_hplt_register_scores({"web-register": scores})


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.0001, 1.0001])
def test_hplt_web_register_scores_remain_finite_probabilities(invalid):
    scores = _hplt_registers(IN=invalid)
    with pytest.raises(TurkishCorpusError, match=r"outside \[0,1\]"):
        strict_hplt_register_scores({"web-register": scores})


def test_hplt_web_register_rejects_numeric_strings():
    scores = _hplt_registers(IN=0.7)
    scores["IN"] = "0.7"
    with pytest.raises(TurkishCorpusError, match="score is not numeric"):
        strict_hplt_register_scores({"web-register": scores})


@pytest.mark.parametrize(
    ("wds_bin", "expected_bucket"),
    [(8, "hplt_wds8_general"), (9, "hplt_wds9_general")],
)
def test_v3_hplt_selector_enforces_exact_register_mt_and_lyrical_bounds(
    wds_bin, expected_bucket
):
    policy = load_corpus_policy(POLICY_V3)
    accepted = {
        "wds_bin": wds_bin,
        "web-register": _hplt_registers(IN=0.4, MT=0.1, LY=0.1),
    }
    assert select_mixture_bucket("hplt3_tr", accepted, policy) == (
        expected_bucket,
        0.4,
    )

    for scores in (
        _hplt_registers(IN=0.399999, MT=0.0, LY=0.0),
        _hplt_registers(IN=0.4, MT=0.100001, LY=0.0),
        _hplt_registers(IN=0.4, MT=0.0, LY=0.100001),
    ):
        assert (
            select_mixture_bucket(
                "hplt3_tr",
                {"wds_bin": wds_bin, "web-register": scores},
                policy,
            )
            is None
        )


def test_macocu_v2_routes_only_frozen_genres_and_non_macocu_genre_is_safe():
    policy = load_corpus_policy(POLICY_V2)
    assert select_mixture_bucket(
        "macocu_genre_tr", {"genre": "Forum", "quality_score": 0.7}, policy
    ) == ("macocu_conversation", 0.7)
    assert select_mixture_bucket(
        "macocu_genre_tr", {"genre": "News", "quality_score": 0.6}, policy
    ) == ("macocu_general", 0.6)
    assert select_mixture_bucket(
        "macocu_genre_tr", {"genre": "Promotion"}, policy
    ) is None
    assert dominant_register({"source_id": "hplt3_tr", "genre": "", "web-register": {"IN": 0.9}}) == "IN"
    assert dominant_register({"source_id": "fineweb2_tr", "genre": ""}) == "not_applicable"


class _BytesResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def raise_for_status(self):
        return None

    def iter_content(self, *, chunk_size: int):
        yield from (
            self.payload[index : index + chunk_size]
            for index in range(0, len(self.payload), chunk_size)
        )

    def close(self):
        return None


def test_macocu_preparation_is_deterministic_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows = [
        {"id": "1", "title": None, "text": "Bir forum yazısı", "url": "https://a", "domain": "a", "tld": "tr", "genre": "Forum"},
        {"id": "2", "title": "Başlık", "text": "Bir haber yazısı", "url": "https://b", "domain": "b", "tld": "tr", "genre": "News"},
        {"id": "3", "title": "Bilgi", "text": "Bir açıklama", "url": "https://c", "domain": "c", "tld": "tr", "genre": "Information/Explanation"},
    ]
    raw = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )
    compressed_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_buffer, mode="wb", mtime=0) as handle:
        handle.write(raw)
    compressed = compressed_buffer.getvalue()
    md5 = hashlib.md5(compressed).hexdigest()  # noqa: S324 - fixture identity

    for module in (backend, corpus):
        monkeypatch.setattr(module, "MACOCU_SIZE_BYTES", len(compressed))
        monkeypatch.setattr(module, "MACOCU_MD5", md5)
        monkeypatch.setattr(module, "MACOCU_EXPECTED_ROWS", len(rows))
    policy = json.loads(POLICY_V2.read_text())
    source = next(item for item in policy["sources"] if item["id"] == "macocu_genre_tr")
    source["resolved_revision"] = md5
    source["expected_md5"] = md5
    source["expected_size_bytes"] = len(compressed)
    source["expected_rows"] = len(rows)

    request_get = lambda *_args, **_kwargs: _BytesResponse(compressed)
    first = backend.prepare_macocu_genre(
        policy,
        tmp_path / "first",
        target_uncompressed_bytes=10_000,
        request_get=request_get,
    )
    second = backend.prepare_macocu_genre(
        policy,
        tmp_path / "second",
        target_uncompressed_bytes=10_000,
        request_get=request_get,
    )
    local_upstream = tmp_path / "MaCoCu-Genre.tr.jsonl.gz"
    local_upstream.write_bytes(compressed)

    def no_network(*_args, **_kwargs):
        raise AssertionError("local MaCoCu reuse must not access the network")

    local = backend.prepare_macocu_genre(
        policy,
        tmp_path / "local",
        target_uncompressed_bytes=10_000,
        request_get=no_network,
        upstream_path=local_upstream,
    )
    # All fixture records deliberately share one shard so this catches missing
    # JSONL framing between adjacent canonical objects.
    assert len(first["shards"]) == 1
    assert first["shards"][0]["rows"] == len(rows)
    assert [item["sha256"] for item in first["shards"]] == [
        item["sha256"] for item in second["shards"]
    ]
    assert local == first
    assert local["upstream"]["sha256"] == hashlib.sha256(compressed).hexdigest()
    symlink = tmp_path / "macocu-symlink.gz"
    symlink.symlink_to(local_upstream)
    with pytest.raises(TurkishCorpusError, match="symlinked, or unsafe"):
        backend.prepare_macocu_genre(
            policy,
            tmp_path / "unsafe-local",
            target_uncompressed_bytes=10_000,
            request_get=no_network,
            upstream_path=symlink,
        )
    corrupted = tmp_path / "macocu-corrupted.gz"
    corrupted_bytes = bytearray(compressed)
    corrupted_bytes[-1] ^= 1
    corrupted.write_bytes(corrupted_bytes)
    with pytest.raises(TurkishCorpusError, match="MD5 drift"):
        backend.prepare_macocu_genre(
            policy,
            tmp_path / "corrupt-local",
            target_uncompressed_bytes=10_000,
            request_get=no_network,
            upstream_path=corrupted,
        )
    observed = []
    for item in first["shards"]:
        shard_path = tmp_path / "first" / item["path"]
        with pa.input_stream(str(shard_path), compression="zstd") as stream:
            framed = stream.read()
        assert framed.endswith(b"\n")
        assert b"\n\n" not in framed
        assert len(framed.splitlines()) == item["rows"]
        observed.extend(iter_input_records(shard_path))
    assert [{key: row[key] for key in corpus.MACOCU_SCHEMA} for row in observed] == rows


def test_prepare_macocu_cli_accepts_optional_verified_upstream_file(tmp_path: Path):
    upstream = tmp_path / "official.jsonl.gz"
    parsed = build_parser().parse_args(
        [
            "prepare-macocu",
            "--output-dir",
            str(tmp_path / "prepared"),
            "--upstream-file",
            str(upstream),
        ]
    )
    assert parsed.upstream_file == upstream


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


def test_backend_raw_stage_consumes_held_descriptor_during_parent_swap(
    tmp_path: Path,
):
    payload = b'{"value":"original-1"}\n{"value":"original-2"}\n'
    source = tmp_path / "source.jsonl"
    source.write_bytes(payload)
    build = tmp_path / "build"
    build.mkdir()
    record, staged = _stage_source_object(
        {
            "uri": source.as_uri(),
            "size_bytes": len(payload),
            "expected_checksums": [
                {
                    "algorithm": "sha256",
                    "value": hashlib.sha256(payload).hexdigest(),
                }
            ],
        },
        build / "source.jsonl",
    )
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()

    rows = iter_input_records(staged)
    saved_build = tmp_path / "saved-build"
    malicious_build = tmp_path / "malicious-build"
    build.rename(saved_build)
    build.mkdir()
    (build / staged.name).write_text('{"value":"malicious"}\n')
    try:
        first = next(rows)
    finally:
        build.rename(malicious_build)
        saved_build.rename(build)
    assert first["value"] == "original-1"
    assert [row["value"] for row in rows] == ["original-2"]
    staged.close()


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


def test_resource_sample_uses_interior_spread_and_avoids_tiny_tail():
    objects = []
    rank = 0
    expected = set()
    for source_id in ("fineweb2_hq_tr", "finewiki_tr"):
        for index in range(9):
            objects.append(
                {
                    "rank": rank,
                    "source_id": source_id,
                    "size_bytes": 1 if index == 8 else 1_000,
                    "uri": f"https://example.test/{source_id}/part-{index:05d}.parquet",
                }
            )
            if index in (2, 4, 6):
                expected.add(rank)
            rank += 1

    selected = set(select_resource_sample_ranks({"objects": objects}))

    assert selected == expected
    assert all(objects[item]["size_bytes"] != 1 for item in selected)


def test_resource_sample_uses_smallest_complete_hplt_shard_per_wds_bin():
    objects = []
    expected = set()
    rank = 0
    for wds_bin in (8, 9, 10):
        for index in range(9):
            objects.append(
                {
                    "rank": rank,
                    "source_id": "hplt3_tr",
                    "wds_bin": wds_bin,
                    "size_bytes": 1 if index == 8 else 1_000,
                    "uri": (
                        f"https://example.test/hplt/wds-{wds_bin}/"
                        f"part-{index:05d}.jsonl.zst"
                    ),
                }
            )
            if index == 8:
                expected.add(rank)
            rank += 1

    selected = set(select_resource_sample_ranks({"objects": objects}))

    assert selected == expected
    assert all(objects[item]["size_bytes"] == 1 for item in selected)


def test_hplt_sample_bin_coverage_is_derived_from_the_source_plan():
    plan = {
        "hplt_control": {"selected_wds_bins": [8, 9]},
        "objects": [
            {
                "rank": 0,
                "source_id": "hplt3_tr",
                "wds_bin": 8,
                "size_bytes": 100,
                "uri": "https://example.test/8_1.jsonl.zst",
            },
            {
                "rank": 1,
                "source_id": "hplt3_tr",
                "wds_bin": 8,
                "size_bytes": 10,
                "uri": "https://example.test/8_2.jsonl.zst",
            },
            {
                "rank": 2,
                "source_id": "hplt3_tr",
                "wds_bin": 9,
                "size_bytes": 20,
                "uri": "https://example.test/9_1.jsonl.zst",
            },
        ],
    }

    assert backend._hplt_sample_bin_coverage(plan) == [8, 9]

    drifted = copy.deepcopy(plan)
    drifted["hplt_control"]["selected_wds_bins"] = [8, 9, 10]
    with pytest.raises(TurkishCorpusError, match="cover each configured HPLT"):
        backend._hplt_sample_bin_coverage(drifted)


def _sealed_hplt_resource_plan(wds_bins: list[int]) -> dict:
    objects = []
    for rank, wds_bin in enumerate(wds_bins):
        objects.append(
            {
                "rank": rank,
                "source_id": "hplt3_tr",
                "wds_bin": wds_bin,
                "size_bytes": 100 + rank,
                "uri": f"https://example.test/{wds_bin}_{rank}.jsonl.zst",
            }
        )
    return seal_manifest(
        {
            "hplt_control": {"selected_wds_bins": wds_bins},
            "objects": objects,
            "canonical_sha256": None,
        }
    )


@pytest.mark.parametrize("wds_bins", [[8, 9], [8, 9, 10]])
def test_resource_projection_accepts_exact_plan_derived_hplt_sample(wds_bins):
    policy = load_corpus_policy(POLICY)
    plan = _sealed_hplt_resource_plan(wds_bins)
    report = _sealed_resource_report(policy, plan["canonical_sha256"])
    report["sample_selection"] = backend._resource_sample_contract(plan)
    report["canonical_sha256"] = None
    report = seal_manifest(report)

    assert validate_resource_projection(report, plan=plan) == report[
        "canonical_sha256"
    ]


@pytest.mark.parametrize("field", ["rank", "wds_bin", "uri", "size_bytes"])
def test_resource_projection_rejects_hplt_inventory_drift_from_bound_plan(field):
    policy = load_corpus_policy(POLICY_V3)
    plan = _sealed_hplt_resource_plan([8, 9])
    report = _sealed_resource_report(policy, plan["canonical_sha256"])
    report["sample_selection"] = backend._resource_sample_contract(plan)
    selected = report["sample_selection"]["hplt_selected_objects"]
    if field == "rank":
        selected[0][field] = 99
        report["sample_selection"]["ranks"][0] = 99
    elif field == "wds_bin":
        selected[0][field] = 99
        report["sample_selection"]["covers_hplt_wds_bins"] = [9, 99]
    elif field == "uri":
        selected[0][field] = "https://attacker.test/replaced.jsonl.zst"
    else:
        selected[0][field] += 1
    report["canonical_sha256"] = None

    with pytest.raises(TurkishCorpusError, match="bound source plan"):
        validate_resource_projection(seal_manifest(report), plan=plan)


def test_resource_projection_rejects_different_sealed_source_plan():
    policy = load_corpus_policy(POLICY_V3)
    plan = _sealed_hplt_resource_plan([8, 9])
    report = _sealed_resource_report(policy, plan["canonical_sha256"])
    report["sample_selection"] = backend._resource_sample_contract(plan)
    report["canonical_sha256"] = None
    report = seal_manifest(report)
    other_plan = copy.deepcopy(plan)
    other_plan["objects"][0]["uri"] = "https://example.test/replaced.jsonl.zst"
    other_plan["canonical_sha256"] = None

    with pytest.raises(TurkishCorpusError, match="source-plan binding drift"):
        validate_resource_projection(report, plan=seal_manifest(other_plan))


def test_resource_sample_spreads_across_macocu_and_avoids_tiny_tail():
    objects = [
        {
            "rank": index,
            "source_id": "macocu_genre_tr",
            "size_bytes": 100 if index < 8 else 1,
            "uri": f"file:///prepared/part-{index:05d}.jsonl.zst",
            "genre_counts": {
                "Forum": 100,
                "Opinion/Argumentation": 100,
                "Information/Explanation": 100,
                "Instruction": 100,
                "News": 100,
            },
        }
        for index in range(9)
    ]
    selected = select_resource_sample_ranks({"objects": objects})
    assert {2, 4, 6} <= set(selected)
    assert 8 not in selected


def test_resource_accounting_bills_wall_time_at_full_cpu2dq_node_rate():
    accounting = _resource_accounting()

    assert accounting["wall_seconds_with_safety_factor"] == 180.0
    assert accounting["billed_cpu_seconds_with_safety_factor"] == 23_040.0
    assert accounting["billed_cpu_saat_with_safety_factor"] == 6.4
    assert accounting["diagnostic_process_cpu"][
        "process_cpu_seconds_with_safety_factor"
    ] == 90.0
    assert accounting["diagnostic_process_cpu"][
        "process_cpu_efficiency_against_billable_capacity"
    ] == pytest.approx(60.0 / (120.0 * 128))


def test_resource_projection_uses_stage_wall_not_process_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    policy = {
        "sources": [{"id": "source"}],
        "materialization": {"max_peak_disk_bytes": 10_000_000_000},
    }
    plan = seal_manifest(
        {
            "canonical_sha256": None,
            "objects": [
                {
                    "rank": 0,
                    "source_id": "source",
                    "size_bytes": 1_000,
                    "uri": "https://example.test/source",
                }
            ],
        }
    )
    calibration = {"canonical_sha256": "c" * 64}
    objects = [
        {
            "canonical_sha256": "d" * 64,
            "source_id": "source",
            "raw_object": {"size_bytes": 100},
            "candidate_file": {"rows": 10, "size_bytes": 20},
            "telemetry": {
                "download": {"wall_seconds": 2.0, "cpu_seconds": 1.0},
                "score_lid": {"wall_seconds": 3.0, "cpu_seconds": 2.0},
                "minhash_signature": {"wall_seconds": 5.0, "cpu_seconds": 4.0},
            },
        }
    ]
    buckets = [
        {
            "canonical_sha256": "e" * 64,
            "input_signature_bytes": 100,
            "output": {"duplicate_edges": 2},
            "telemetry": {"wall_seconds": 7.0, "cpu_seconds": 6.0},
        }
    ]
    cluster = seal_manifest(
        {
            "canonical_sha256": None,
            "sample_mode": True,
            "telemetry": {
                "wall_seconds": 11.0,
                "cpu_seconds": 10.0,
                "peak_rss_bytes": 1024**3,
                "edge_participating_documents": 4,
            },
            "output_files": [{"size_bytes": 30}],
        }
    )
    monkeypatch.setattr(backend, "validate_corpus_policy", lambda _policy: None)
    monkeypatch.setattr(
        backend, "validate_source_plan", lambda _plan, _policy: None
    )
    monkeypatch.setattr(
        backend, "validate_backend_calibration", lambda _calibration, _policy: None
    )
    monkeypatch.setattr(backend, "_load_object_receipts", lambda *args, **kwargs: objects)
    monkeypatch.setattr(backend, "_load_bucket_receipts", lambda *args, **kwargs: buckets)
    monkeypatch.setattr(backend, "load_json_strict", lambda _path: cluster)

    report = backend.build_resource_projection(
        policy,
        plan,
        calibration,
        tmp_path / "sample",
        tmp_path / "resource-report.json",
        quota_headroom_bytes=10_000_000_000,
        billable_cpus_per_job=128,
        safety_factor=1.5,
    )

    stage_wall = report["projection"]["stage_wall_seconds_before_safety_factor"]
    assert stage_wall == {
        "download": 20.0,
        "score_lid": 30.0,
        "minhash_signature": 50.0,
        "minhash_buckets": 6_664.0,
        "priority_cluster_quality_format": 110.0,
    }
    assert report["projection"]["billed_cpu_saat_with_safety_factor"] == pytest.approx(
        sum(stage_wall.values()) * 1.5 * 128 / 3600
    )
    projection = report["projection"]
    assert projection["signature_bytes"] == 95_200.0
    assert projection["peak_disk_components_before_safety_factor"] == {
        "raw_largest_object_bytes": 1_000,
        "candidate_bytes": 200.0,
        "signature_bytes": 95_200.0,
        "duplicate_edge_bytes": 320.0,
        "backend_output_bytes": 300.0,
    }
    assert projection["peak_disk_bytes_before_safety_factor"] == 97_020.0
    assert projection["peak_disk_bytes_with_safety_factor"] == 145_530.0
    assert projection["peak_disk_model"] == backend.RESOURCE_PEAK_DISK_MODEL
    assert report["sample_selection"]["algorithm"] == (
        "non_hplt_uri_quartiles_plus_hplt_smallest_complete_shard_"
        "per_wds_bin_plus_macocu_genres_v5"
    )
    assert report["sample_selection"]["object_order"] == (
        "source_plan_uri_ascending"
    )
    assert report["sample_selection"]["size_based_selection"] is True
    assert report["sample_selection"]["size_based_selection_scope"] == (
        "hplt3_tr_only"
    )
    assert report["sample_selection"][
        "non_hplt_per_source_stream_spread_quantiles"
    ] == [
        0.25,
        0.5,
        0.75,
    ]
    assert report["sample_selection"]["hplt_per_wds_bin_selection"] == (
        "minimum_size_complete_shard_then_uri_v1"
    )
    assert report["sample_selection"]["hplt_selected_objects"] == []
    assert report["sample_selection"]["covers_hplt_wds_bins"] == []
    stale = copy.deepcopy(report)
    stale["sample_selection"]["covers_hplt_wds_bins"] = [8, 9, 10]
    stale["canonical_sha256"] = None
    with pytest.raises(TurkishCorpusError, match="bound source plan"):
        validate_resource_projection(seal_manifest(stale), plan=plan)
    diagnostic_cpu = report["projection"]["diagnostic_process_cpu"][
        "stage_process_cpu_seconds_before_safety_factor"
    ]
    assert sum(diagnostic_cpu.values()) == 5_882.0
    assert report["billing_contract"] == RESOURCE_BILLING_CONTRACT


def test_resource_report_and_approval_bind_billed_cpu_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    policy = load_corpus_policy(POLICY)
    plan_path = tmp_path / "source-plan.json"
    calibration_path = tmp_path / "calibration.json"
    plan = seal_manifest(
        {"fixture": "plan", "objects": [], "canonical_sha256": None}
    )
    calibration = seal_manifest(
        {"fixture": "calibration", "canonical_sha256": None}
    )
    write_json_atomic(plan_path, plan)
    write_json_atomic(calibration_path, calibration)
    plan_sha256 = plan["canonical_sha256"]
    report = _sealed_resource_report(policy, plan_sha256)
    report["calibration_sha256"] = calibration["canonical_sha256"]
    report = seal_manifest(report)
    assert validate_resource_projection(report, plan=plan) == report[
        "canonical_sha256"
    ]
    report_path = tmp_path / "resource-report.json"
    quality_path = tmp_path / "mixture-quality-approval.json"
    approval_path = tmp_path / "resource-approval.json"
    write_json_atomic(report_path, report)
    quality = seal_manifest(
        {
            "schema_version": "3.0",
            "kind": MIXTURE_QUALITY_APPROVAL_KIND,
            "sample_quality_audit_sha256": "d" * 64,
            "policy_sha256": report["policy_sha256"],
            "source_plan_sha256": plan_sha256,
            "calibration_sha256": calibration["canonical_sha256"],
            "cluster_receipt_sha256": "e" * 64,
            "sample_cluster_receipt_sha256": "e" * 64,
            "reviewed_example_files": {
                decision: {
                    "rows": 1,
                    "jsonl_sha256": "f" * 64,
                    "plaintext_sha256": "a" * 64,
                }
                for decision in ("accepted", "rejected")
            },
            "coverage_complete": True,
            "automatic_decision": False,
            "review_confirmation": (
                "bounded_strata_and_accepted_rejected_examples_reviewed"
            ),
            "reviewer": "quality-reviewer",
            "reviewed_at_utc": "2026-08-20T17:00:00Z",
            "decision": "accepted",
            "notes": "fixture",
            "canonical_sha256": None,
        }
    )
    write_json_atomic(quality_path, quality)
    monkeypatch.setattr(backend, "validate_source_plan", lambda *_args: None)
    monkeypatch.setattr(
        backend, "validate_backend_calibration", lambda *_args: None
    )
    monkeypatch.setattr(
        backend,
        "validate_mixture_quality_approval",
        lambda approval, **_kwargs: approval["canonical_sha256"],
    )

    drifted_report = copy.deepcopy(report)
    drifted_report["sample_selection"] = {
        "ranks": [99],
        "covers_hplt_wds_bins": [99],
        "hplt_selected_objects": [
            {
                "rank": 99,
                "wds_bin": 99,
                "size_bytes": 123,
                "uri": "https://attacker.test/99_0.jsonl.zst",
            }
        ],
    }
    drifted_report["canonical_sha256"] = None
    drifted_report = seal_manifest(drifted_report)
    drifted_report_path = tmp_path / "drifted-resource-report.json"
    write_json_atomic(drifted_report_path, drifted_report)
    with pytest.raises(TurkishCorpusError, match="bound source plan"):
        seal_resource_approval(
            drifted_report_path,
            quality_path,
            tmp_path / "drifted-resource-approval.json",
            policy_path=POLICY,
            source_plan_path=plan_path,
            calibration_path=calibration_path,
            reviewer="resource-reviewer",
            reviewed_at_utc="2026-08-20T18:00:00Z",
            decision="accepted",
        )

    cross_quality = copy.deepcopy(quality)
    cross_quality["cluster_receipt_sha256"] = "9" * 64
    cross_quality["sample_cluster_receipt_sha256"] = "9" * 64
    cross_quality = seal_manifest(cross_quality)
    cross_quality_path = tmp_path / "cross-cluster-quality.json"
    write_json_atomic(cross_quality_path, cross_quality)
    with pytest.raises(TurkishCorpusError, match="sample-cluster lineage drift"):
        seal_resource_approval(
            report_path,
            cross_quality_path,
            tmp_path / "cross-cluster-resource.json",
            policy_path=POLICY,
            source_plan_path=plan_path,
            calibration_path=calibration_path,
            reviewer="resource-reviewer",
            reviewed_at_utc="2026-08-20T18:00:00Z",
            decision="accepted",
        )

    approval = seal_resource_approval(
        report_path,
        quality_path,
        approval_path,
        policy_path=POLICY,
        source_plan_path=plan_path,
        calibration_path=calibration_path,
        reviewer="resource-reviewer",
        reviewed_at_utc="2026-08-20T18:00:00Z",
        decision="accepted",
    )

    assert approval["kind"] == RESOURCE_APPROVAL_KIND
    assert approval["approved_projection"] == {
        "billing_contract": RESOURCE_BILLING_CONTRACT,
        "billed_cpu_saat_with_safety_factor": 6.4,
    }
    validate_resource_approval(
        approval,
        plan=plan,
        policy=policy,
        calibration=calibration,
        approval_path=approval_path,
    )

    drifted_approval = copy.deepcopy(approval)
    drifted_report_raw = drifted_report_path.read_bytes()
    drifted_approval["resource_report_sha256"] = drifted_report[
        "canonical_sha256"
    ]
    drifted_approval["evidence_bundle"]["resource_report"] = {
        "path": drifted_report_path.name,
        "size_bytes": len(drifted_report_raw),
        "sha256": hashlib.sha256(drifted_report_raw).hexdigest(),
    }
    drifted_approval["canonical_sha256"] = None
    drifted_approval = seal_manifest(drifted_approval)
    drifted_approval_path = tmp_path / "drifted-validation-approval.json"
    write_json_atomic(drifted_approval_path, drifted_approval)
    with pytest.raises(TurkishCorpusError, match="bound source plan"):
        validate_resource_approval(
            drifted_approval,
            plan=plan,
            policy=policy,
            calibration=calibration,
            approval_path=drifted_approval_path,
        )

    bad_report = copy.deepcopy(report)
    bad_report["projection"]["billed_cpu_saat_with_safety_factor"] = 0.025
    bad_report = seal_manifest(bad_report)
    with pytest.raises(TurkishCorpusError, match="billed_cpu_saat.*arithmetic drift"):
        validate_resource_projection(bad_report, plan=plan)

    bad_peak = copy.deepcopy(report)
    bad_peak["projection"]["peak_disk_bytes_with_safety_factor"] = 0.0
    bad_peak = seal_manifest(bad_peak)
    with pytest.raises(TurkishCorpusError, match="peak disk with safety factor"):
        validate_resource_projection(bad_peak, plan=plan)

    bad_component = copy.deepcopy(report)
    bad_component["projection"]["candidate_bytes"] = 0.0
    bad_component = seal_manifest(bad_component)
    with pytest.raises(TurkishCorpusError, match="candidate_bytes arithmetic drift"):
        validate_resource_projection(bad_component, plan=plan)

    bad_approval = copy.deepcopy(approval)
    bad_approval["approved_projection"]["billing_contract"][
        "billable_cpus_per_job"
    ] = 8
    bad_approval = seal_manifest(bad_approval)
    with pytest.raises(TurkishCorpusError, match="billing contract drift"):
        validate_resource_approval(
            bad_approval,
            plan=plan,
            policy=policy,
            calibration=calibration,
            approval_path=approval_path,
        )


def test_resource_report_cli_requires_explicit_billable_cpu_count():
    required = [
        "resource-report",
        "--source-plan=plan.json",
        "--calibration=calibration.json",
        "--sample-run-dir=sample",
        "--quota-headroom-bytes=1000000",
        "--output=report.json",
    ]
    with pytest.raises(SystemExit):
        build_parser().parse_args(required)
    parsed = build_parser().parse_args([*required, "--billable-cpus-per-job=128"])
    assert parsed.billable_cpus_per_job == 128


def test_fabricated_sha_only_mixture_approval_is_rejected(tmp_path: Path):
    policy = load_corpus_policy(POLICY)
    plan = seal_manifest({"objects": [], "canonical_sha256": None})
    calibration = seal_manifest({"canonical_sha256": None})
    approval = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": MIXTURE_QUALITY_APPROVAL_KIND,
            "sample_quality_audit_sha256": "a" * 64,
            "policy_sha256": backend._policy_sha256(policy),
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "cluster_receipt_sha256": "b" * 64,
            "reviewed_example_files": {
                decision: {
                    "rows": 1,
                    "jsonl_sha256": "c" * 64,
                    "plaintext_sha256": "d" * 64,
                }
                for decision in ("accepted", "rejected")
            },
            "coverage_complete": True,
            "automatic_decision": False,
            "review_confirmation": (
                "bounded_strata_and_accepted_rejected_examples_reviewed"
            ),
            "reviewer": "fabricator",
            "reviewed_at_utc": "2026-08-20T17:00:00Z",
            "decision": "accepted",
            "canonical_sha256": None,
        }
    )
    path = tmp_path / "fabricated.json"
    write_json_atomic(path, approval)

    with pytest.raises(TurkishCorpusError, match="actual|evidence|accepted manual"):
        validate_mixture_quality_approval(
            approval,
            policy=policy,
            plan=plan,
            calibration=calibration,
            approval_path=path,
        )


def test_fabricated_sha_only_resource_approval_is_rejected(tmp_path: Path):
    policy = load_corpus_policy(POLICY)
    plan = {"canonical_sha256": "1" * 64}
    calibration = {"canonical_sha256": "2" * 64}
    approval = seal_manifest(
        {
            "schema_version": "4.0",
            "kind": RESOURCE_APPROVAL_KIND,
            "resource_report_sha256": "3" * 64,
            "policy_sha256": backend._policy_sha256(policy),
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "mixture_quality_approval_sha256": "4" * 64,
            "sample_cluster_receipt_sha256": "5" * 64,
            "evidence_bundle": {
                "schema_version": "1.0",
                "resource_report": {
                    "path": "missing-report.json",
                    "size_bytes": 1,
                    "sha256": "3" * 64,
                },
                "mixture_quality_approval": {
                    "path": "missing-quality.json",
                    "size_bytes": 1,
                    "sha256": "4" * 64,
                },
            },
            "approved_projection": {
                "billing_contract": RESOURCE_BILLING_CONTRACT,
                "billed_cpu_saat_with_safety_factor": 1.0,
            },
            "reviewer": "fabricator",
            "reviewed_at_utc": "2026-08-20T17:00:00Z",
            "decision": "accepted",
            "notes": "hash-shaped strings only",
            "canonical_sha256": None,
        }
    )
    path = tmp_path / "fabricated-resource.json"
    write_json_atomic(path, approval)

    with pytest.raises(TurkishCorpusError, match="evidence file is missing"):
        validate_resource_approval(
            approval,
            plan=plan,
            policy=policy,
            calibration=calibration,
            approval_path=path,
        )


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


def test_no_code_gate_rejects_ruby_times_do_and_puts():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = """Bir eğitim yazısı, aynı cümleyi birkaç kez göstermeyi anlatıyor ve uzun bir Türkçe açıklamayla örneğin amacını, beklenen sonucu ve günlük kullanımını ayrıntılı biçimde tartışıyor.
10.times do
  puts "Merhaba"
end
Yazının geri kalanında okuyucunun adımları nasıl izleyeceği ve sonucu nasıl değerlendireceği doğal bir dille açıklanıyor."""

    decision = audit_document(text, source_lid_ok=True, content_policy=content_policy)

    assert decision.accepted is False
    assert decision.reason == "code_content"
    assert decision.metrics["code_line_fraction"] >= 0.4


def test_no_code_gate_keeps_four_line_turkish_physics_numeric_equality():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = """Enerji = 5 kilovat saattir.
Bu değer, evde kullanılan küçük bir aygıtın belirli bir süre boyunca tükettiği toplam enerjiyi açık ve anlaşılır biçimde göstermektedir.
Fizik dersinde öğretmen, güç ile zaman arasındaki ilişkiyi gündelik yaşamdan örneklerle anlatarak öğrencilerin kavramı daha kolay öğrenmesini sağlamaktadır.
Sonuç olarak verilen eşitlik bir programlama komutu değil, ölçülen fiziksel büyüklüğün Türkçe bir cümle içinde sade biçimde ifade edilmesidir."""

    decision = audit_document(text, source_lid_ok=True, content_policy=content_policy)

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.metrics["code_line_fraction"] == 0.0


def test_no_code_gate_rejects_consecutive_scalar_assignment_structure():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = """Bu uzun Türkçe açıklama, basit bir hesabın gündelik bir örnek üzerinden nasıl yapıldığını anlatıyor ve okuyucuya sonucu değerlendirmesi için yeterli bağlam sağlıyor.
x = 5
y = 10
z = x + y
Son bölüm, sayıların anlamını doğal bir Türkçe anlatımla açıklıyor; buna rağmen ortadaki ardışık atama satırları çalıştırılabilir kod yapısıdır."""

    decision = audit_document(text, source_lid_ok=True, content_policy=content_policy)

    assert decision.accepted is False
    assert decision.reason == "code_content"
    assert decision.metrics["consecutive_scalar_assignment_lines"] == 3


def test_no_code_gate_rejects_compact_semicolon_assignment_sequence():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = """Bu Türkçe açıklama, küçük bir hesabın gündelik bir örnek içinde nasıl kurulduğunu ve sayıların sonuç üzerinde ne anlama geldiğini ayrıntılı biçimde anlatmaktadır.
x = 5; y = 10; z = x + y
Son bölümde okuyucunun sonucu nasıl değerlendireceği doğal bir dille açıklanır; ortadaki tek satır ise üç ayrı çalıştırılabilir atama ifadesi içerir."""

    decision = audit_document(text, source_lid_ok=True, content_policy=content_policy)

    assert decision.accepted is False
    assert decision.reason == "code_content"
    assert decision.metrics["compact_scalar_assignments"] == 3


def test_no_code_gate_rejects_typed_assignment_control_flow_and_augmented_update():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = """Bu uzun Türkçe metin, bir değerin hangi koşul altında değiştirildiğini gündelik bir örnek üzerinden tanıtıyor ve okuyucuya bağlam hakkında yeterli açıklama sunuyor.
x: int = 5
if x > 3:
    x += 1
Metnin sonunda değişimin anlamı yeniden doğal Türkçe cümlelerle ele alınıyor; buna rağmen ortadaki satırlar doğrudan çalıştırılabilir Python yapısıdır."""

    decision = audit_document(text, source_lid_ok=True, content_policy=content_policy)

    assert decision.accepted is False
    assert decision.reason == "code_content"
    assert decision.metrics["code_line_fraction"] >= 0.6


def test_no_code_gate_keeps_turkish_formula_explanation():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = """Bir veri grubundaki ortalamayı bulmak için iki değerin toplamı öğe sayısına bölünür.
Ortalama = (ilk değer + ikinci değer) / 2 olarak hesaplanır.
Bu eşitlik bir programlama komutu değildir; ders anlatımında kullanılan matematiksel ilişkinin Türkçe açıklamasıdır.
Öğretmen daha sonra aynı yöntemi farklı sayılarla göstererek öğrencilerin kavramı gündelik örneklerle pekiştirmesini sağlar."""

    decision = audit_document(text, source_lid_ok=True, content_policy=content_policy)

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.metrics["code_line_fraction"] == 0.0


def test_no_code_gate_keeps_title_cased_units_glossary():
    content_policy = load_corpus_policy(POLICY)["content_policy"]
    text = """Aşağıdaki kısa sözlük, temel ölçü türlerinin günlük yaşamda kullanılan birimlerini sade biçimde tanıtır.
Uzunluk = metre
Ağırlık = kilogram
Zaman = saniye
Bu satırlar çalıştırılabilir komutlar değildir; fen dersindeki kavramları karşılıklarıyla gösteren doğal bir Türkçe başvuru listesidir.
Öğrenciler listeyi okuduktan sonra her birimin kullanıldığı durumları kendi örnekleriyle açıklar."""

    decision = audit_document(text, source_lid_ok=True, content_policy=content_policy)

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.metrics["consecutive_scalar_assignment_lines"] == 0


def test_dedup_decision_retains_complete_frozen_shape():
    decision = corpus.DedupDecision(
        accepted=False,
        cluster_id="a" * 64,
        duplicate_kind="near_duplicate",
    )

    assert decision.accepted is False
    assert decision.cluster_id == "a" * 64
    assert decision.duplicate_kind == "near_duplicate"


def test_bounded_evidence_snapshot_is_stable_across_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "evidence.json"
    replacement = tmp_path / "replacement.json"
    original = b'{"lineage":"original"}\n'
    swapped = b'{"lineage":"replacement"}\n'
    target.write_bytes(original)
    replacement.write_bytes(swapped)
    real_read = os.read
    did_swap = False

    def swap_then_read(descriptor: int, size: int) -> bytes:
        nonlocal did_swap
        if not did_swap:
            did_swap = True
            os.replace(replacement, target)
        return real_read(descriptor, size)

    monkeypatch.setattr(backend.os, "read", swap_then_read)
    with pytest.raises(TurkishCorpusError, match="changed while read"):
        backend._read_bounded_regular_file_snapshot(
            target, label="racing evidence", max_bytes=1024
        )

    assert target.read_bytes() == swapped


def test_verified_artifact_rejects_same_content_path_replacement(
    tmp_path: Path,
):
    target = tmp_path / "artifact.bin"
    replacement = tmp_path / "replacement.bin"
    content = b"same attested bytes\n"
    target.write_bytes(content)
    replacement.write_bytes(content)

    with pytest.raises(TurkishCorpusError, match="path changed during consumption"):
        with backend._open_verified_regular_artifact(
            target,
            label="test artifact",
            expected_size=len(content),
            expected_sha256=hashlib.sha256(content).hexdigest(),
        ) as handle:
            assert handle.read() == content
            os.replace(replacement, target)


def test_verified_artifact_rejects_in_place_mutation(
    tmp_path: Path,
):
    target = tmp_path / "artifact.bin"
    original = b"attested-content"
    mutated = b"tampered-content"
    assert len(original) == len(mutated)
    target.write_bytes(original)

    with pytest.raises(TurkishCorpusError, match="changed during consumption"):
        with backend._open_verified_regular_artifact(
            target,
            label="test artifact",
            expected_size=len(original),
            expected_sha256=hashlib.sha256(original).hexdigest(),
        ) as handle:
            assert handle.read() == original
            with target.open("r+b") as mutator:
                mutator.write(mutated)
                mutator.flush()
                os.fsync(mutator.fileno())


def test_duplicate_edge_consumer_rejects_path_replacement(
    tmp_path: Path,
):
    target = tmp_path / "bucket_matches" / "00000_00.dups"
    target.parent.mkdir()
    payload = struct.pack("<4I", 0, 1, 2, 3) + struct.pack("<4I", 4, 5, 6, 7)
    target.write_bytes(payload)
    replacement = tmp_path / "replacement.dups"
    replacement.write_bytes(payload)
    record = {
        "path": target.relative_to(tmp_path).as_posix(),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "duplicate_edges": 2,
    }

    edges = backend._iter_duplicate_edges(
        tmp_path, record, label="test duplicate edges"
    )
    assert next(edges) == (0, 1, 2, 3)
    os.replace(replacement, target)
    with pytest.raises(TurkishCorpusError, match="path changed during consumption"):
        list(edges)


def test_candidate_parquet_consumer_rejects_path_replacement(
    tmp_path: Path,
):
    row = {
        "text": "Bu, güvenilir ve yeterince uzun bir Türkçe aday metindir.",
        "source_id": "fineweb2_hq_tr",
        "document_id": "candidate-1",
        "url": "https://ornek.test/1",
        "source_lid_label": "tur_Latn",
        "source_lid_probability": 0.99,
        "lid_label": "tur_Latn",
        "lid_probability": 0.98,
        "lid_margin": 0.75,
        "paragraph_min_probability": 0.98,
        "paragraph_min_margin": 0.75,
        "failed_long_paragraph_fraction": 0.0,
        "dedup_cluster_id": "a" * 64,
        "dedup_keep": True,
        "quality_score": 0.8,
        "wds_bin": None,
        "web-register": "{}",
        "genre": "",
        "pii_replacements": 0,
        "harmful_signal_hits": 0,
        "quality_filter_flags": "[]",
        "formatting_changes": "{}",
        "candidate_rank": 0,
        "candidate_doc_index": 0,
    }
    second = dict(row, document_id="candidate-2", candidate_doc_index=1)
    target = tmp_path / "objects" / "00000" / "candidates.parquet"
    target.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([row, second], schema=backend._INTERNAL_SCHEMA),
        target,
        compression="zstd",
    )
    replacement = tmp_path / "replacement-candidates.parquet"
    replacement.write_bytes(target.read_bytes())
    payload = target.read_bytes()
    record = {
        "path": target.relative_to(tmp_path).as_posix(),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": 2,
    }

    rows = backend._iter_candidate_rows(
        tmp_path, record, label="test candidate input"
    )
    assert next(rows)["document_id"] == "candidate-1"
    os.replace(replacement, target)
    with pytest.raises(TurkishCorpusError, match="path changed during consumption"):
        list(rows)


def test_staged_parquet_consumes_held_descriptor_during_parent_swap(tmp_path: Path):
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"value": "original-1"}, {"value": "original-2"}]),
        source,
    )
    payload = source.read_bytes()
    build = tmp_path / "build"
    build.mkdir()
    staged = corpus.stage_receipt_file(
        {
            "uri": source.as_uri(),
            "checksum": {
                "algorithm": "sha256",
                "value": hashlib.sha256(payload).hexdigest(),
            },
            "size_bytes": len(payload),
        },
        build,
    )
    rows = iter_input_records(staged)
    saved_build = tmp_path / "saved-build"
    malicious_build = tmp_path / "malicious-build"
    build.rename(saved_build)
    build.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"value": "malicious"}]),
        build / staged.name,
    )
    try:
        first = next(rows)
    finally:
        build.rename(malicious_build)
        saved_build.rename(build)
    assert first["value"] == "original-1"
    assert [row["value"] for row in rows] == ["original-2"]
    staged.unlink()


def test_staged_parquet_rejects_in_place_mutation_during_consumption(
    tmp_path: Path,
):
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"value": "original-1"}, {"value": "original-2"}]),
        source,
    )
    payload = source.read_bytes()
    build = tmp_path / "build"
    build.mkdir()
    staged = corpus.stage_receipt_file(
        {
            "uri": source.as_uri(),
            "checksum": {
                "algorithm": "sha256",
                "value": hashlib.sha256(payload).hexdigest(),
            },
            "size_bytes": len(payload),
        },
        build,
    )
    rows = iter_input_records(staged)
    assert next(rows)["value"] == "original-1"
    staged.path.write_bytes(b"mutated in place")
    with pytest.raises(TurkishCorpusError, match="changed during consumption"):
        list(rows)


def test_bucket_signature_stage_consumes_held_descriptor_during_directory_swap(
    tmp_path: Path,
):
    signature = tmp_path / "signatures" / "bucket_003" / "00000.minhash.sig"
    signature.parent.mkdir(parents=True)
    original = b"verified signature bytes"
    signature.write_bytes(original)
    record = {
        "path": signature.relative_to(tmp_path).as_posix(),
        "size_bytes": len(original),
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    objects = [{"rank": 0, "signature_files": [record]}]

    class FakeDataFolder:
        def open_files(self, paths, mode="rb", **kwargs):
            return [self.open(path, mode=mode, **kwargs) for path in paths]

    with backend._verified_signature_data_folder(
        tmp_path,
        objects,
        bucket_rank=3,
        data_folder_class=FakeDataFolder,
    ) as verified:
        paths = verified.list_files(subdirectory="bucket_003")
        assert paths == ["bucket_003/00000.minhash.sig"]

        original_bucket = signature.parent
        saved_bucket = tmp_path / "saved_bucket_003"
        malicious_bucket = tmp_path / "malicious_bucket_003"
        original_bucket.rename(saved_bucket)
        original_bucket.mkdir()
        (original_bucket / signature.name).write_bytes(b"malicious signature data")
        try:
            reader = verified.open_files(paths, mode="rb", block_size=4096)[0]
            with reader:
                assert reader.path == paths[0]
                assert reader.size == len(original)
                assert reader.read() == original
        finally:
            original_bucket.rename(malicious_bucket)
            saved_bucket.rename(original_bucket)


def test_object_receipt_raw_object_is_exactly_bound_to_source_plan(tmp_path: Path):
    candidate_path = tmp_path / "objects" / "00000" / "candidates.parquet"
    candidate_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([], schema=backend._INTERNAL_SCHEMA),
        candidate_path,
        compression="zstd",
    )
    candidate = backend._file_record(candidate_path, root=tmp_path, rows=0)
    signatures = []
    for bucket in range(14):
        path = (
            tmp_path
            / "signatures"
            / f"bucket_{bucket:03d}"
            / "00000.minhash.sig"
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
        signatures.append(backend._file_record(path, root=tmp_path))

    observed_sha256 = hashlib.sha256(b"planned raw object").hexdigest()
    checksums = [{"algorithm": "sha256", "value": observed_sha256}]
    plan = {
        "canonical_sha256": "a" * 64,
        "objects": [
            {
                "rank": 0,
                "source_id": "source-a",
                "uri": "https://example.test/object.parquet",
                "size_bytes": 18,
                "expected_checksums": checksums,
            }
        ],
    }
    calibration = {"canonical_sha256": "b" * 64}
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": backend.OBJECT_RECEIPT_KIND,
            "sample_mode": False,
            "rank": 0,
            "source_id": "source-a",
            "source_uri": "https://example.test/object.parquet",
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "raw_object": {
                "uri": "https://example.test/object.parquet",
                "size_bytes": 18,
                "sha256": observed_sha256,
                "upstream_checksums_verified": checksums,
            },
            "candidate_file": candidate,
            "signature_files": signatures,
            "canonical_sha256": None,
        }
    )
    backend._validate_object_receipt(
        receipt,
        plan=plan,
        calibration=calibration,
        run_root=tmp_path,
        sample_mode=False,
    )

    tampered = copy.deepcopy(receipt)
    tampered["raw_object"]["sha256"] = "f" * 64
    tampered = seal_manifest(tampered)
    with pytest.raises(TurkishCorpusError, match="raw-object SHA-256 drift"):
        backend._validate_object_receipt(
            tampered,
            plan=plan,
            calibration=calibration,
            run_root=tmp_path,
            sample_mode=False,
        )


def _production_chain_fixture(launch_sha256: str) -> tuple[dict, dict]:
    launch = {
        "production_pack_plan_sha256": "1" * 64,
        "resource_approval_sha256": "2" * 64,
        "mixture_quality_approval_sha256": "3" * 64,
        "data_prep_storage_gate_sha256": "4" * 64,
        "sample_cluster_receipt_sha256": "5" * 64,
    }
    chain = {
        "cluster_launch_receipt_sha256": launch_sha256,
        **launch,
    }
    return launch, chain


def test_source_seal_rejects_objects_not_bound_by_launch_cluster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    launch_sha256 = "6" * 64
    launch, _chain = _production_chain_fixture(launch_sha256)
    cluster = {"object_receipt_sha256": ["7" * 64]}
    objects = [
        {
            "canonical_sha256": "8" * 64,
            "source_id": "source-a",
            "source_uri": "https://example.test/object.parquet",
            "raw_object": {"sha256": "9" * 64, "size_bytes": 123},
        }
    ]
    policy = {
        "sources": [
            {
                "id": "source-a",
                "repo_id": "repo",
                "resolved_revision": "revision",
                "license_id": "license",
            }
        ]
    }
    plan = {"canonical_sha256": "a" * 64, "derived_sources": {}}
    calibration = {"canonical_sha256": "b" * 64}
    monkeypatch.setattr(backend, "validate_corpus_policy", lambda _value: None)
    monkeypatch.setattr(
        backend, "validate_source_plan", lambda _plan, _policy: None
    )
    monkeypatch.setattr(
        backend, "validate_backend_calibration", lambda _value, _policy: None
    )
    monkeypatch.setattr(
        backend,
        "_validate_production_cluster_launch",
        lambda *args, **kwargs: (launch, launch_sha256, cluster),
    )
    monkeypatch.setattr(
        backend, "_load_object_receipts", lambda *args, **kwargs: objects
    )

    with pytest.raises(TurkishCorpusError, match="object/cluster binding drift"):
        backend.seal_source_receipt_from_objects(
            policy,
            plan,
            calibration,
            tmp_path,
            tmp_path / "launch.json",
            tmp_path / "source.json",
        )


@pytest.mark.parametrize("mismatch", ["source_objects", "cluster_buckets"])
def test_backend_seal_rejects_cross_receipt_lineage_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
):
    launch_sha256 = "6" * 64
    launch, chain = _production_chain_fixture(launch_sha256)
    object_sha256 = "7" * 64
    bucket_sha256 = "8" * 64
    objects = [
        {
            "canonical_sha256": object_sha256,
            "rank": 0,
            "quality_score_semantics": backend.OBJECT_SOURCE_QUALITY_SEMANTICS,
        }
    ]
    buckets = [{"canonical_sha256": bucket_sha256}]
    cluster = {
        "kind": backend.CLUSTER_RECEIPT_KIND,
        "sample_mode": False,
        "source_plan_sha256": "a" * 64,
        "processing": {},
        "code_identity": {},
        "object_receipt_sha256": [object_sha256],
        "bucket_receipt_sha256": [bucket_sha256],
        "winner_policy": backend.CLUSTER_WINNER_POLICY,
        "quality_score_semantics": backend.CLUSTER_QUALITY_SCORE_SEMANTICS,
        "legacy_quality_score_neutralized_ranks": [],
        "output_files": [],
    }
    source_receipt = {
        "canonical_sha256": "9" * 64,
        "production_chain": chain,
        "object_receipt_sha256": [object_sha256],
    }
    if mismatch == "source_objects":
        source_receipt["object_receipt_sha256"] = ["f" * 64]
        expected_error = "source/object/cluster receipt binding drift"
    else:
        cluster["bucket_receipt_sha256"] = ["f" * 64]
        expected_error = "cluster quality/dedup provenance drift"

    policy = {}
    plan = {"canonical_sha256": "a" * 64}
    calibration = {"canonical_sha256": "b" * 64}
    monkeypatch.setattr(backend, "validate_corpus_policy", lambda _value: None)
    monkeypatch.setattr(
        backend, "validate_source_plan", lambda _plan, _policy: None
    )
    monkeypatch.setattr(
        backend, "validate_source_receipt", lambda _receipt, _policy: None
    )
    monkeypatch.setattr(
        backend, "validate_backend_calibration", lambda _value, _policy: None
    )
    monkeypatch.setattr(
        backend,
        "_validate_production_cluster_launch",
        lambda *args, **kwargs: (launch, launch_sha256, cluster),
    )
    monkeypatch.setattr(
        backend, "production_processing_binding", lambda _policy: {}
    )
    monkeypatch.setattr(
        backend, "validate_production_code_identity", lambda _value: None
    )
    monkeypatch.setattr(
        backend, "_load_object_receipts", lambda *args, **kwargs: objects
    )
    monkeypatch.setattr(
        backend, "_load_bucket_receipts", lambda *args, **kwargs: buckets
    )

    with pytest.raises(TurkishCorpusError, match=expected_error):
        backend.seal_backend_receipt_from_cluster(
            policy,
            plan,
            source_receipt,
            calibration,
            tmp_path,
            tmp_path / "launch.json",
            tmp_path / "backend.json",
        )


def test_source_quality_score_never_uses_lid_confidence():
    assert backend._source_quality_score({"quality_score": 0.31}) == pytest.approx(0.31)
    assert backend._source_quality_score({"fineweb2_hq_score": 0.44}) == pytest.approx(
        0.44
    )
    assert backend._source_quality_score({"lid_probability": 0.999}) == 0.0
    assert backend._source_quality_score({"score": 0.999}) == 0.0


def test_qa_metrics_include_exact_encoding_corruption_counts():
    _text, metrics = corpus._qa_document_metrics(
        {"text": "bozuk\ufffd Ã¼ ve Ã¼ \x81 \ud800"}
    )

    assert metrics["unicode_replacement_characters"] == 1
    assert metrics["mojibake_sequence_hits"] == 2
    assert metrics["c1_control_characters"] == 1
    assert metrics["unicode_surrogate_characters"] == 1


def test_legacy_candidate_quality_is_neutral_until_object_receipt_attests_it():
    row = {"quality_score": 0.999, "lid_probability": 0.999}

    assert backend._attested_source_quality(row, {}) == 0.0
    assert backend._attested_source_quality(
        row,
        {"quality_score_semantics": backend.OBJECT_SOURCE_QUALITY_SEMANTICS},
    ) == pytest.approx(0.999)


def test_attested_candidate_quality_must_be_finite_and_non_negative():
    receipt = {"quality_score_semantics": backend.OBJECT_SOURCE_QUALITY_SEMANTICS}

    with pytest.raises(TurkishCorpusError, match="finite and non-negative"):
        backend._attested_source_quality({"quality_score": float("nan")}, receipt)
    with pytest.raises(TurkishCorpusError, match="finite and non-negative"):
        backend._attested_source_quality({"quality_score": -0.1}, receipt)


def test_processing_binding_seals_executable_no_code_patterns_and_thresholds():
    policy = load_corpus_policy(POLICY)
    binding = backend.production_processing_binding(policy)
    local = binding["project_additions"]["local_policy_audit"]
    no_code = binding["project_additions"]["no_code"]

    assert local["patterns"]["code_line"]["sha256"] == no_code[
        "code_line_pattern_sha256"
    ]
    assert local["scalar_thresholds"]["max_code_line_fraction"] == no_code[
        "max_code_line_fraction"
    ]
    assert local["patterns"]["scalar_assignment_line"]["sha256"] == no_code[
        "scalar_assignment_line_pattern_sha256"
    ]
    assert no_code["minimum_consecutive_scalar_assignment_lines"] == 3
    assert no_code["minimum_compact_scalar_assignments"] == 3
    assert no_code["scalar_assignment_classifier"] == (
        "code_shape_with_prose_equality_exception_v2"
    )
    changed = copy.deepcopy(policy)
    changed["content_policy"]["max_code_line_fraction"] = 0.03
    assert (
        backend.production_processing_binding(changed)["binding_sha256"]
        != binding["binding_sha256"]
    )
