from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest
import pyarrow as pa

import nanochat.turkish_backend as backend
import nanochat.turkish_corpus as corpus
from nanochat.experiment_manifest import (
    canonical_json,
    seal_manifest,
    write_json_atomic,
)
from nanochat.turkish_backend import (
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
)
from nanochat.turkish_corpus import (
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
    return seal_manifest(
        {
            "schema_version": "1.0",
            "kind": RESOURCE_REPORT_KIND,
            "policy_sha256": hashlib.sha256(
                canonical_json(policy).encode("utf-8")
            ).hexdigest(),
            "source_plan_sha256": plan_sha256,
            "calibration_sha256": "c" * 64,
            "billing_contract": dict(RESOURCE_BILLING_CONTRACT),
            "projection": {"safety_factor": 1.5, **_resource_accounting()},
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
    # All fixture records deliberately share one shard so this catches missing
    # JSONL framing between adjacent canonical objects.
    assert len(first["shards"]) == 1
    assert first["shards"][0]["rows"] == len(rows)
    assert [item["sha256"] for item in first["shards"]] == [
        item["sha256"] for item in second["shards"]
    ]
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
    plan = {
        "canonical_sha256": "a" * 64,
        "objects": [
            {
                "rank": 0,
                "source_id": "source",
                "size_bytes": 1_000,
                "uri": "https://example.test/source",
            }
        ],
    }
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
            "telemetry": {"wall_seconds": 11.0, "cpu_seconds": 10.0},
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
    assert report["sample_selection"]["algorithm"] == (
        "uri_ordered_interior_quartiles_per_source_and_hplt_bin_plus_macocu_genres_v4"
    )
    assert report["sample_selection"]["object_order"] == (
        "source_plan_uri_ascending"
    )
    assert report["sample_selection"]["size_based_selection"] is False
    assert report["sample_selection"]["per_source_stream_spread_quantiles"] == [
        0.25,
        0.5,
        0.75,
    ]
    assert report["sample_selection"]["hplt_per_wds_bin_spread_quantiles"] == []
    diagnostic_cpu = report["projection"]["diagnostic_process_cpu"][
        "stage_process_cpu_seconds_before_safety_factor"
    ]
    assert sum(diagnostic_cpu.values()) == 5_882.0
    assert report["billing_contract"] == RESOURCE_BILLING_CONTRACT


def test_resource_report_and_approval_bind_billed_cpu_contract(tmp_path: Path):
    policy = load_corpus_policy(POLICY)
    plan_sha256 = "b" * 64
    report = _sealed_resource_report(policy, plan_sha256)
    assert validate_resource_projection(report) == report["canonical_sha256"]
    report_path = tmp_path / "resource-report.json"
    approval_path = tmp_path / "resource-approval.json"
    write_json_atomic(report_path, report)

    approval = seal_resource_approval(
        report_path,
        approval_path,
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
        plan={"canonical_sha256": plan_sha256},
        policy=policy,
    )

    bad_report = copy.deepcopy(report)
    bad_report["projection"]["billed_cpu_saat_with_safety_factor"] = 0.025
    bad_report = seal_manifest(bad_report)
    with pytest.raises(TurkishCorpusError, match="billed_cpu_saat.*arithmetic drift"):
        validate_resource_projection(bad_report)

    bad_approval = copy.deepcopy(approval)
    bad_approval["approved_projection"]["billing_contract"][
        "billable_cpus_per_job"
    ] = 8
    bad_approval = seal_manifest(bad_approval)
    with pytest.raises(TurkishCorpusError, match="billing contract drift"):
        validate_resource_approval(
            bad_approval,
            plan={"canonical_sha256": plan_sha256},
            policy=policy,
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
