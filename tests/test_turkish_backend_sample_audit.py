from __future__ import annotations

import json
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
from scripts import audit_turkish_backend_sample as sample_audit


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "pretrain" / "tr_d32_turkish_general_v2.json"


def _row(
    document_id: str,
    *,
    source_id: str = "hplt3_tr",
    text: str = "Bu, Türkiye hakkında güvenilir ve anlaşılır bir örnek metindir.",
    dedup_keep: bool = True,
    flags: tuple[str, ...] = (),
    registers: dict[str, float] | None = None,
) -> dict[str, object]:
    if registers is None:
        registers = {"IN": 0.82, "MT": 0.03}
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
                "Bu sayfa Türkçe bir açıklamadır. "
                "Çerez ayarları kullanıcıya gösterilir.\n"
                "function deneme() { return 1; }"
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
    output = paths["run"] / "backend_output" / "part-00000.parquet"
    output.parent.mkdir()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=backend._BACKEND_SCHEMA),
        output,
        compression="zstd",
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
            "counts": {
                "output_rows": len(rows),
                "dedup_kept": 5,
                "dedup_removed": 1,
                "quality_kept": 3,
                "quality_removed": 2,
            },
            "output_files": [
                {
                    "path": output.relative_to(paths["run"]).as_posix(),
                    "size_bytes": output.stat().st_size,
                    "sha256": file_sha256(output),
                    "rows": len(rows),
                }
            ],
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

    parquet = paths["run"] / "backend_output" / "part-00000.parquet"
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
