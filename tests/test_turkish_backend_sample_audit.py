from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nanochat.turkish_backend as backend
from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.turkish_corpus import HPLT_WEB_REGISTER_KEYS
from scripts import audit_turkish_backend_sample as sample_audit
from scripts import d32_family_workflow as workflow


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_v2.json"


def _row(
    document_id: str,
    *,
    source_id: str = "hplt3_tr",
    text: str = (
        "Bu, Türkiye hakkında güvenilir ve anlaşılır bir örnek metindir. "
        "İnsanlar günlük yaşamlarını, mahalledeki gelişmeleri, aileleriyle "
        "yaptıkları planları ve hafta sonunda gördükleri yerleri doğal bir "
        "Türkçeyle ayrıntılı biçimde anlatmaktadır."
    ),
    dedup_keep: bool = True,
    flags: tuple[str, ...] = (),
    registers: dict[str, float] | None = None,
) -> dict[str, object]:
    if source_id == "hplt3_tr":
        overrides = registers or {"IN": 0.82, "MT": 0.03}
        registers = dict.fromkeys(HPLT_WEB_REGISTER_KEYS, 0.0)
        registers.update(overrides)
    return {
        "text": text,
        "source_id": source_id,
        "document_id": document_id,
        "url": f"https://ornek.test/{document_id}",
        "source_lid_label": "tur_Latn",
        "source_lid_probability": 0.98,
        "lid_label": "tur_Latn",
        "lid_probability": 0.97,
        "lid_margin": 0.72,
        "paragraph_min_probability": 0.93,
        "paragraph_min_margin": 0.61,
        "failed_long_paragraph_fraction": 0.0,
        "dedup_cluster_id": document_id.encode("utf-8").hex().ljust(64, "0")[:64],
        "dedup_keep": dedup_keep,
        "quality_score": 0.76,
        "wds_bin": 8 if source_id == "hplt3_tr" else None,
        "web-register": canonical_json(registers) if source_id == "hplt3_tr" else "{}",
        "genre": "",
        "pii_replacements": 0,
        "harmful_signal_hits": 0,
        "quality_filter_flags": canonical_json(list(flags)),
        "formatting_changes": "{}",
    }


def _object_receipt(rank: int, source_id: str, rows: int) -> dict[str, object]:
    return seal_manifest(
        {
            "schema_version": "1.0",
            "kind": backend.OBJECT_RECEIPT_KIND,
            "sample_mode": True,
            "rank": rank,
            "source_id": source_id,
            "source_uri": f"https://source.test/{source_id}/{rank}.jsonl.zst",
            "raw_object": {"size_bytes": 10_000 + rank, "sha256": f"{rank + 1:064x}"},
            "candidate_file": {
                "path": f"objects/{rank:05d}/candidates.parquet",
                "size_bytes": 2_000 + rank,
                "sha256": f"{rank + 11:064x}",
                "rows": rows,
            },
            "counts": {
                "documents_seen": rows + 2,
                "candidates": rows,
                "characters_seen": 1_200 + rank,
                "utf8_bytes_seen": 1_600 + rank,
                "candidate_characters": 900 + rank,
                "stage_counts": {
                    "source_lid": {"input": rows + 2, "kept": rows, "removed": 2},
                    "independent_glotlid": {"input": rows, "kept": rows},
                },
            },
            "canonical_sha256": None,
        }
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    paths = {
        "policy": POLICY,
        "plan": tmp_path / "source_plan.json",
        "calibration": tmp_path / "backend_calibration.json",
        "run": tmp_path / "sample_run",
    }
    paths["run"].mkdir()
    plan = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": backend.SOURCE_PLAN_KIND,
            "policy_sha256": backend._policy_sha256(
                backend.load_corpus_policy(paths["policy"])
            ),
            "objects": [
                {"rank": 0, "source_id": "hplt3_tr", "wds_bin": 8},
                {"rank": 1, "source_id": "fineweb2_hq_tr", "wds_bin": None},
            ],
            "canonical_sha256": None,
        }
    )
    calibration = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": backend.CALIBRATION_KIND,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(paths["plan"], plan)
    write_json_atomic(paths["calibration"], calibration)

    rows = [
        _row("accepted-hplt-1"),
        _row(
            "accepted-hplt-2",
            text=(
                "Bu sayfa Türkçe bir açıklamadır. İnsanlar mahalledeki "
                "etkinlikleri konuşuyor, birbirlerine öneriler veriyor ve "
                "gelecek hafta için birlikte sakin bir plan yapıyorlar. "
                "Sohbet boyunca yaşadıkları küçük olayları paylaşarak neden "
                "bu buluşmayı önemsediklerini de açıkça ifade ediyorlar."
            ),
        ),
        _row("quality-rejected", flags=("gopher_quality:too_few_words",)),
        _row("duplicate-rejected", dedup_keep=False),
        _row(
            "selector-rejected",
            flags=("project_mixture_selector:unrouted",),
            registers={"IN": 0.20, "MT": 0.80},
        ),
        _row("accepted-fineweb", source_id="fineweb2_hq_tr"),
    ]
    output_root = paths["run"] / "backend_output"
    output_root.mkdir()
    outputs = []
    for rank, rank_rows in ((0, rows[:5]), (1, rows[5:])):
        output = output_root / f"{rank:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(rank_rows, schema=backend._BACKEND_SCHEMA),
            output,
            compression="zstd",
        )
        outputs.append(
            {
                "path": output.relative_to(paths["run"]).as_posix(),
                "size_bytes": output.stat().st_size,
                "sha256": file_sha256(output),
                "rows": len(rank_rows),
                "source_rank": rank,
            }
        )
    object_receipts = [
        _object_receipt(0, "hplt3_tr", 5),
        _object_receipt(1, "fineweb2_hq_tr", 1),
    ]
    bucket_receipts = [
        seal_manifest(
            {
                "schema_version": "1.0",
                "kind": backend.BUCKET_RECEIPT_KIND,
                "rank": rank,
                "canonical_sha256": None,
            }
        )
        for rank in range(14)
    ]
    object_launch_path = (
        paths["run"] / "packed_sample_launches" / "job123" / "launch_receipt.json"
    )
    object_launch_path.parent.mkdir(parents=True)
    object_launch = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_packed_resource_sample_launch_receipt",
            "sample_mode": True,
            "policy_sha256": plan["policy_sha256"],
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "object_receipts": [
                {
                    "rank": item["rank"],
                    "canonical_sha256": item["canonical_sha256"],
                }
                for item in object_receipts
            ],
            "all_lanes_completed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(object_launch_path, object_launch)
    bucket_launch_path = (
        paths["run"] / "packed_bucket_launches" / "job124" / "launch_receipt.json"
    )
    bucket_launch_path.parent.mkdir(parents=True)
    bucket_launch = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_packed_sample_bucket_launch_receipt",
            "sample_mode": True,
            "policy_sha256": plan["policy_sha256"],
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "object_sample_launch_receipt_sha256": object_launch[
                "canonical_sha256"
            ],
            "backend_bucket_receipts": [
                {
                    "bucket_rank": item["rank"],
                    "canonical_sha256": item["canonical_sha256"],
                }
                for item in bucket_receipts
            ],
            "all_buckets_completed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(bucket_launch_path, bucket_launch)
    sample_launches = {
        "object": {
            "path": object_launch_path.relative_to(paths["run"]).as_posix(),
            "size_bytes": object_launch_path.stat().st_size,
            "sha256": file_sha256(object_launch_path),
            "canonical_sha256": object_launch["canonical_sha256"],
        },
        "bucket": {
            "path": bucket_launch_path.relative_to(paths["run"]).as_posix(),
            "size_bytes": bucket_launch_path.stat().st_size,
            "sha256": file_sha256(bucket_launch_path),
            "canonical_sha256": bucket_launch["canonical_sha256"],
        },
    }
    cluster = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": backend.CLUSTER_RECEIPT_KIND,
            "sample_mode": True,
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "object_receipt_sha256": [
                item["canonical_sha256"] for item in object_receipts
            ],
            "bucket_receipt_sha256": [
                item["canonical_sha256"] for item in bucket_receipts
            ],
            "processing": backend.production_processing_binding(
                backend.load_corpus_policy(paths["policy"])
            ),
            "code_identity": backend.production_code_identity(),
            "sample_launch_receipts": sample_launches,
            "winner_policy": backend.CLUSTER_WINNER_POLICY,
            "quality_score_semantics": backend.CLUSTER_QUALITY_SCORE_SEMANTICS,
            "legacy_quality_score_neutralized_ranks": [0, 1],
            "counts": {
                "output_rows": len(rows),
                "dedup_kept": 5,
                "dedup_removed": 1,
                "quality_kept": 3,
                "quality_removed": 2,
            },
            "output_files": outputs,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(paths["run"] / "cluster_receipt.json", cluster)

    monkeypatch.setattr(sample_audit.backend, "validate_source_plan", lambda *_args: None)
    monkeypatch.setattr(
        sample_audit.backend, "validate_backend_calibration", lambda *_args: None
    )
    monkeypatch.setattr(
        sample_audit.backend,
        "_load_object_receipts",
        lambda *_args, **_kwargs: object_receipts,
    )
    monkeypatch.setattr(
        sample_audit.backend,
        "_load_bucket_receipts",
        lambda *_args, **_kwargs: bucket_receipts,
    )
    monkeypatch.setattr(
        sample_audit.backend, "select_resource_sample_ranks", lambda _plan: [0, 1]
    )
    return paths, rows


def _reseal_example_rows(
    audit_dir: Path,
    report: dict[str, object],
    decision: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    jsonl_path = audit_dir / f"{decision}_examples.jsonl"
    plaintext_path = audit_dir / f"{decision}_examples.txt"
    jsonl_path.write_text(
        "".join(canonical_json(row).rstrip("\n") + "\n" for row in rows),
        encoding="utf-8",
    )
    plaintext_path.write_text(
        backend.render_sample_quality_plaintext(rows), encoding="utf-8"
    )
    record = report["example_sampling"]["files"][decision]  # type: ignore[index]
    record["jsonl"]["size_bytes"] = jsonl_path.stat().st_size
    record["jsonl"]["sha256"] = file_sha256(jsonl_path)
    record["plaintext"]["size_bytes"] = plaintext_path.stat().st_size
    record["plaintext"]["sha256"] = file_sha256(plaintext_path)
    report["canonical_sha256"] = None
    sealed = seal_manifest(report)
    write_json_atomic(audit_dir / "sample_quality_audit_report.json", sealed)
    return sealed


def test_sample_audit_seals_stratified_manual_review_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, rows = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "audit"

    report = sample_audit.build_sample_quality_audit(
        paths["policy"], paths["plan"], paths["calibration"], paths["run"], output
    )

    assert verify_manifest_hash(report) == report["canonical_sha256"]
    assert report["integrity_checks_passed"] is True
    assert report["manual_review_required"] is True
    assert report["automatic_mixture_approval"] is False
    assert report["review_status"] == "pending"
    assert report["cluster_totals"] == {
        "documents": len(rows),
        "accepted_documents": 3,
        "rejected_documents": 3,
        "accepted_utf8_bytes": sum(
            len(str(row["text"]).encode("utf-8"))
            for row in rows
            if row["document_id"] in {"accepted-hplt-1", "accepted-hplt-2", "accepted-fineweb"}
        ),
        "rejected_utf8_bytes": sum(
            len(str(row["text"]).encode("utf-8"))
            for row in rows
            if str(row["document_id"]).endswith("rejected")
        ),
    }
    hplt = next(
        item
        for item in report["strata"]
        if item["source_id"] == "hplt3_tr"
        and item["mixture_id"] == "hplt_general"
        and item["register"] == "IN"
    )
    assert hplt["counts"]["total_documents"] == 4
    assert hplt["counts"]["accepted_documents"] == 2
    assert hplt["counts"]["dedup_removed_documents"] == 1
    assert hplt["dedup_survival_rate"] == pytest.approx(0.75)
    assert hplt["quality_filter_flag_rejections"] == {
        "gopher_quality:too_few_words": 1
    }
    assert hplt["rejection_reasons"] == {
        "dedup_removed": 1,
        "quality_filter": 1,
    }
    numeric = hplt["numeric_distributions"]["all"]
    for metric in (
        "text_characters",
        "source_lid_probability",
        "quality_score",
        "code_line_fraction",
        "cookie_ui_hits",
    ):
        assert numeric[metric]["observations"] == 4

    unrouted = next(item for item in report["strata"] if item["mixture_id"] == "unrouted")
    assert unrouted["register"] == "MT"
    assert unrouted["counts"]["selector_unrouted_documents"] == 1
    assert unrouted["rejection_reasons"] == {"selector_unrouted": 1}
    assert unrouted["quality_filter_flag_rejections"] == {
        "project_mixture_selector:unrouted": 1
    }
    assert report["source_input_and_candidates"][0]["sampled_objects"] == 1
    assert report["sampled_objects"][0]["source_uri_sha256"]

    for decision, expected_rows in (("accepted", 3), ("rejected", 3)):
        record = report["example_sampling"]["files"][decision]
        jsonl_path = output / record["jsonl"]["path"]
        raw = jsonl_path.read_bytes()
        assert raw.endswith(b"\n")
        assert raw.count(b"\n") == expected_rows
        decoded = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        assert len(decoded) == record["rows"] == expected_rows
        assert all(row["decision"] == decision for row in decoded)
        assert file_sha256(jsonl_path) == record["jsonl"]["sha256"]
        plaintext_path = output / record["plaintext"]["path"]
        assert plaintext_path.is_file()
        assert file_sha256(plaintext_path) == record["plaintext"]["sha256"]


def test_sample_audit_is_deterministic_and_rejects_cluster_output_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    first = sample_audit.build_sample_quality_audit(
        paths["policy"],
        paths["plan"],
        paths["calibration"],
        paths["run"],
        tmp_path / "audit-a",
    )
    second = sample_audit.build_sample_quality_audit(
        paths["policy"],
        paths["plan"],
        paths["calibration"],
        paths["run"],
        tmp_path / "audit-b",
    )
    assert first == second

    parquet = paths["run"] / "backend_output" / "00000.parquet"
    with parquet.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(sample_audit.SampleAuditError, match="cluster output drift"):
        sample_audit.build_sample_quality_audit(
            paths["policy"],
            paths["plan"],
            paths["calibration"],
            paths["run"],
            tmp_path / "audit-after-drift",
        )


def test_sample_audit_rejects_cluster_path_replacement_during_parquet_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    target = paths["run"] / "backend_output" / "00000.parquet"
    replacement = tmp_path / "replacement.parquet"
    replacement.write_bytes(target.read_bytes())
    target_inode = target.stat().st_ino
    real_parquet_file = pq.ParquetFile
    swapped = False

    def swap_path(source, *args, **kwargs):
        nonlocal swapped
        if (
            not swapped
            and hasattr(source, "fileno")
            and os.fstat(source.fileno()).st_ino == target_inode
        ):
            swapped = True
            os.replace(replacement, target)
        return real_parquet_file(source, *args, **kwargs)

    monkeypatch.setattr(sample_audit.pq, "ParquetFile", swap_path)
    with pytest.raises(
        sample_audit.SampleAuditError, match="cluster output drift.*path changed"
    ):
        sample_audit.build_sample_quality_audit(
            paths["policy"],
            paths["plan"],
            paths["calibration"],
            paths["run"],
            tmp_path / "audit-path-swap",
        )
    assert swapped is True


def test_sample_audit_rejects_in_place_parquet_mutation_during_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    target = paths["run"] / "backend_output" / "00000.parquet"
    target_inode = target.stat().st_ino
    real_parquet_file = pq.ParquetFile
    mutated = False

    def mutate_in_place(source, *args, **kwargs):
        nonlocal mutated
        parquet = real_parquet_file(source, *args, **kwargs)
        if (
            not mutated
            and hasattr(source, "fileno")
            and os.fstat(source.fileno()).st_ino == target_inode
        ):
            mutated = True
            with target.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(first)
                handle.flush()
                os.fsync(handle.fileno())
        return parquet

    monkeypatch.setattr(sample_audit.pq, "ParquetFile", mutate_in_place)
    with pytest.raises(
        sample_audit.SampleAuditError, match="cluster output drift.*changed"
    ):
        sample_audit.build_sample_quality_audit(
            paths["policy"],
            paths["plan"],
            paths["calibration"],
            paths["run"],
            tmp_path / "audit-in-place-mutation",
        )
    assert mutated is True


def test_sample_audit_infers_legacy_one_file_per_rank_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    cluster_path = paths["run"] / "cluster_receipt.json"
    cluster = backend.load_json_strict(cluster_path)
    for record in cluster["output_files"]:
        del record["source_rank"]
    write_json_atomic(cluster_path, seal_manifest(cluster))

    report = sample_audit.build_sample_quality_audit(
        paths["policy"],
        paths["plan"],
        paths["calibration"],
        paths["run"],
        tmp_path / "audit-legacy-rank-paths",
    )

    assert report["coverage"]["source_ranks_with_accepted_rows"] == [0, 1]


def test_sample_audit_refuses_weaker_policy_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    with pytest.raises(sample_audit.SampleAuditError, match="weakens the frozen QA policy"):
        sample_audit.build_sample_quality_audit(
            paths["policy"],
            paths["plan"],
            paths["calibration"],
            paths["run"],
            tmp_path / "weak-audit",
            examples_per_stratum=1,
        )


def test_actual_audit_bundle_is_verified_before_manual_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    audit_dir = tmp_path / "sample_quality_audit"
    report = sample_audit.build_sample_quality_audit(
        paths["policy"], paths["plan"], paths["calibration"], paths["run"], audit_dir
    )
    approval_path = tmp_path / "mixture_quality_approval.json"
    workflow.command_seal_mixture_quality_approval(
        argparse.Namespace(
            policy=paths["policy"],
            source_plan=paths["plan"],
            calibration=paths["calibration"],
            audit_report=audit_dir / "sample_quality_audit_report.json",
            reviewer="quality-reviewer",
            reviewed_at_utc="2026-08-20T17:00:00Z",
            decision="rejected",
            notes="fixture",
            output=approval_path,
        )
    )
    approval = backend.load_json_strict(approval_path)
    policy = backend.load_corpus_policy(paths["policy"])
    plan = backend.load_json_strict(paths["plan"])
    calibration = backend.load_json_strict(paths["calibration"])

    assert approval["schema_version"] == "3.0"
    assert approval["sample_quality_audit_sha256"] == report["canonical_sha256"]
    verified, digest = backend.validate_sample_quality_audit_bundle(
        audit_dir,
        approval["evidence_bundle"]["report"],
        policy=policy,
        plan=plan,
        calibration=calibration,
    )
    assert verified == report
    assert digest == report["canonical_sha256"]

    accepted = audit_dir / "accepted_examples.jsonl"
    accepted.write_text(accepted.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(backend.TurkishCorpusError, match="content drift"):
        backend.validate_sample_quality_audit_bundle(
            audit_dir,
            approval["evidence_bundle"]["report"],
            policy=policy,
            plan=plan,
            calibration=calibration,
        )


def test_resealed_bundle_cannot_claim_ruby_code_is_an_accepted_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    audit_dir = tmp_path / "sample_quality_audit"
    report = sample_audit.build_sample_quality_audit(
        paths["policy"], paths["plan"], paths["calibration"], paths["run"], audit_dir
    )
    accepted_path = audit_dir / "accepted_examples.jsonl"
    accepted_rows = [
        json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines()
    ]
    accepted_rows[0]["text"] = """Bu açıklama görünüşte Türkçe bir paragraf olarak başlıyor ve örneğin amacını yeterince uzun bir girişle anlatıyor.
10.times do
  puts "Merhaba"
end
Son cümle de metni uzatıyor, ancak ortadaki Ruby kodu nedeniyle bu içerik kabul edilmiş gibi gösterilemez."""
    accepted_rows[0]["text_truncated"] = False
    accepted_path.write_text(
        "".join(canonical_json(row).rstrip("\n") + "\n" for row in accepted_rows),
        encoding="utf-8",
    )
    plaintext_path = audit_dir / "accepted_examples.txt"
    plaintext_path.write_text(
        backend.render_sample_quality_plaintext(accepted_rows), encoding="utf-8"
    )
    accepted_record = report["example_sampling"]["files"]["accepted"]
    accepted_record["jsonl"]["size_bytes"] = accepted_path.stat().st_size
    accepted_record["jsonl"]["sha256"] = file_sha256(accepted_path)
    accepted_record["plaintext"]["size_bytes"] = plaintext_path.stat().st_size
    accepted_record["plaintext"]["sha256"] = file_sha256(plaintext_path)
    report["canonical_sha256"] = None
    report = seal_manifest(report)
    report_path = audit_dir / "sample_quality_audit_report.json"
    write_json_atomic(report_path, report)
    report_record = {
        "path": report_path.name,
        "size_bytes": report_path.stat().st_size,
        "sha256": file_sha256(report_path),
    }

    with pytest.raises(
        backend.TurkishCorpusError,
        match="row content drift|deterministic live cluster rows|full-text evidence",
    ):
        backend.validate_sample_quality_audit_bundle(
            audit_dir,
            report_record,
            policy=backend.load_corpus_policy(paths["policy"]),
            plan=backend.load_json_strict(paths["plan"]),
            calibration=backend.load_json_strict(paths["calibration"]),
        )


def test_resealed_accepted_text_substitution_is_not_a_cluster_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    audit_dir = tmp_path / "sample_quality_audit"
    report = sample_audit.build_sample_quality_audit(
        paths["policy"], paths["plan"], paths["calibration"], paths["run"], audit_dir
    )
    accepted_path = audit_dir / "accepted_examples.jsonl"
    accepted_rows = [
        json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines()
    ]
    replacement = (
        "Bu metin tamamen farklı fakat görünüşte temiz bir Türkçe örnektir. "
        "Mahalle yaşamını, arkadaşlarla yapılan planları ve gündelik konuşmaları "
        "doğal bir dille anlatarak biçimsel denetimlerden geçmeye çalışmaktadır."
    )
    row = accepted_rows[0]
    row["text"] = replacement
    row["text_truncated"] = False
    row["full_text_characters"] = len(replacement)
    row["full_text_utf8_bytes"] = len(replacement.encode("utf-8"))
    row["full_text_sha256"] = hashlib.sha256(replacement.encode("utf-8")).hexdigest()
    row["cluster_row_sha256"] = hashlib.sha256(b"fabricated-cluster-row").hexdigest()
    row["sample_sha256"] = hashlib.sha256(b"fabricated-selection").hexdigest()
    report = _reseal_example_rows(audit_dir, report, "accepted", accepted_rows)
    report_path = audit_dir / "sample_quality_audit_report.json"
    report_record = {
        "path": report_path.name,
        "size_bytes": report_path.stat().st_size,
        "sha256": file_sha256(report_path),
    }

    with pytest.raises(
        backend.TurkishCorpusError, match="deterministic live cluster rows"
    ):
        backend.validate_sample_quality_audit_bundle(
            audit_dir,
            report_record,
            policy=backend.load_corpus_policy(paths["policy"]),
            plan=backend.load_json_strict(paths["plan"]),
            calibration=backend.load_json_strict(paths["calibration"]),
        )


def test_truncated_accepted_example_cannot_hide_ruby_after_visible_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    audit_dir = tmp_path / "sample_quality_audit"
    report = sample_audit.build_sample_quality_audit(
        paths["policy"], paths["plan"], paths["calibration"], paths["run"], audit_dir
    )
    accepted_path = audit_dir / "accepted_examples.jsonl"
    accepted_rows = [
        json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines()
    ]
    row = accepted_rows[0]
    visible = str(row["text"])
    hidden_full = visible + "\n10.times do\n  puts \"Merhaba\"\nend"
    row["text_truncated"] = True
    row["full_text_characters"] = len(hidden_full)
    row["full_text_utf8_bytes"] = len(hidden_full.encode("utf-8"))
    row["full_text_sha256"] = hashlib.sha256(hidden_full.encode("utf-8")).hexdigest()
    row["cluster_row_sha256"] = hashlib.sha256(b"ruby-hidden-row").hexdigest()
    row["sample_sha256"] = hashlib.sha256(b"ruby-hidden-selection").hexdigest()
    report = _reseal_example_rows(audit_dir, report, "accepted", accepted_rows)
    report_path = audit_dir / "sample_quality_audit_report.json"
    report_record = {
        "path": report_path.name,
        "size_bytes": report_path.stat().st_size,
        "sha256": file_sha256(report_path),
    }

    with pytest.raises(
        backend.TurkishCorpusError, match="deterministic live cluster rows"
    ):
        backend.validate_sample_quality_audit_bundle(
            audit_dir,
            report_record,
            policy=backend.load_corpus_policy(paths["policy"]),
            plan=backend.load_json_strict(paths["plan"]),
            calibration=backend.load_json_strict(paths["calibration"]),
        )


def test_one_bad_sample_rank_cannot_be_masked_by_other_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    output = paths["run"] / "backend_output" / "00001.parquet"
    row = pq.read_table(output).to_pylist()[0]
    row["quality_filter_flags"] = canonical_json(
        ["project_local_audit:fixture_rejection"]
    )
    pq.write_table(
        pa.Table.from_pylist([row], schema=backend._BACKEND_SCHEMA),
        output,
        compression="zstd",
    )
    cluster_path = paths["run"] / "cluster_receipt.json"
    cluster = backend.load_json_strict(cluster_path)
    rank_record = next(
        item for item in cluster["output_files"] if item["source_rank"] == 1
    )
    rank_record["size_bytes"] = output.stat().st_size
    rank_record["sha256"] = file_sha256(output)
    cluster["counts"]["quality_kept"] = 2
    cluster["counts"]["quality_removed"] = 3
    write_json_atomic(cluster_path, seal_manifest(cluster))

    audit_dir = tmp_path / "audit-bad-rank"
    report = sample_audit.build_sample_quality_audit(
        paths["policy"], paths["plan"], paths["calibration"], paths["run"], audit_dir
    )
    assert report["coverage"]["source_ranks_without_accepted_rows"] == [1]
    assert report["coverage"]["source_ranks_without_accepted_examples"] == [1]
    with pytest.raises(workflow.FamilyWorkflowError, match="accepted-row|coverage"):
        workflow.command_seal_mixture_quality_approval(
            argparse.Namespace(
                policy=paths["policy"],
                source_plan=paths["plan"],
                calibration=paths["calibration"],
                audit_report=audit_dir / "sample_quality_audit_report.json",
                reviewer="quality-reviewer",
                reviewed_at_utc="2026-08-20T17:00:00Z",
                decision="accepted",
                notes="must fail",
                output=tmp_path / "bad-rank-approval.json",
            )
        )


def test_sample_audit_rechecks_rows_claimed_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths, _rows = _fixture(tmp_path, monkeypatch)
    output = paths["run"] / "backend_output" / "00000.parquet"
    rows = pq.read_table(output).to_pylist()
    rows[0]["text"] = """Bu uzun Türkçe açıklama, bir örneğin neden tekrarlandığını ve okuyucunun beklenen sonucu nasıl değerlendireceğini gündelik bir dille ayrıntılı biçimde anlatmaktadır.
10.times do
  puts "Merhaba"
end
Metnin geri kalanı yeterince uzun olsa da yukarıdaki satırlar açıkça çalıştırılabilir bir Ruby kod parçasıdır ve kabul edilmemelidir."""
    pq.write_table(
        pa.Table.from_pylist(rows, schema=backend._BACKEND_SCHEMA),
        output,
        compression="zstd",
    )
    cluster_path = paths["run"] / "cluster_receipt.json"
    cluster = backend.load_json_strict(cluster_path)
    record = next(item for item in cluster["output_files"] if item["source_rank"] == 0)
    record["size_bytes"] = output.stat().st_size
    record["sha256"] = file_sha256(output)
    write_json_atomic(cluster_path, seal_manifest(cluster))

    with pytest.raises(sample_audit.SampleAuditError, match="disagrees"):
        sample_audit.build_sample_quality_audit(
            paths["policy"],
            paths["plan"],
            paths["calibration"],
            paths["run"],
            tmp_path / "audit-claimed-accepted",
        )
