from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tarfile
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest

import nanochat.turkish_anchor_preparation as anchor_preparation
from nanochat.experiment_manifest import (
    file_sha256,
    seal_manifest,
    write_json_atomic,
)
from nanochat.turkish_anchor_preparation import (
    AnchorPreparationError,
    MOT_SOURCE_ID,
    MOT_V1_11_CONTRACT,
    MotAssetContract,
    MotContract,
    PARLAMINT_NATIVE_META_HEADER,
    PARLAMINT_SOURCE_ID,
    ParlaMintContract,
    prepare_mot_v1_11,
    prepare_parlamint_tr_v5,
    seal_anchor_acquisition_receipt,
    seal_anchor_count_acceptance,
    validate_anchor_preparation,
)


_TEST_TIMESTAMP = "2026-08-20T00:00:00Z"


def _write_tgz(
    path: Path,
    members: list[tuple[str, bytes | None]],
    *,
    archive_format: int = tarfile.PAX_FORMAT,
    pax_headers: dict[str, dict[str, str]] | None = None,
    symlink: tuple[str, str] | None = None,
    hardlink: tuple[str, str] | None = None,
) -> None:
    with tarfile.open(path, "w:gz", format=archive_format) as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.mtime = 1_700_000_000
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if pax_headers and name in pax_headers:
                info.pax_headers = pax_headers[name]
            if payload is None:
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                info.size = 0
                archive.addfile(info)
            else:
                info.mode = 0o644
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        for link, link_type in ((symlink, tarfile.SYMTYPE), (hardlink, tarfile.LNKTYPE)):
            if link is None:
                continue
            info = tarfile.TarInfo(link[0])
            info.type = link_type
            info.linkname = link[1]
            info.mtime = 1_700_000_000
            archive.addfile(info)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_jsonl_zstd(path: Path) -> list[dict]:
    with pa.input_stream(str(path), compression="zstd") as stream:
        payload = stream.read().decode("utf-8")
    return [json.loads(line) for line in payload.splitlines()]


def _read_artifact(output: Path, artifact: dict) -> list[dict]:
    rows: list[dict] = []
    for shard in artifact["shards"]:
        rows.extend(_read_jsonl_zstd(output / shard["path"]))
    return rows


def _seal_acquisition(
    tmp_path: Path,
    source_id: str,
    archives: list[Path],
    contract: MotContract | ParlaMintContract,
    *,
    name: str,
    decision: str = "accepted",
) -> Path:
    receipt_path = tmp_path / name
    seal_anchor_acquisition_receipt(
        source_id,
        archives,
        receipt_path,
        reviewer="unit-test-reviewer",
        acquired_at_utc=_TEST_TIMESTAMP,
        decision=decision,
        contract=contract,
    )
    return receipt_path


def _mot_record(
    member_name: str,
    *,
    paragraphs: list[str],
    title: str = "Önemli bir başlık",
    modified: str = "2020-01-01T00:00:00Z",
    retrieved: str = "2020-01-02T00:00:00Z",
    site_language: str = "tur",
    predicted_language: str = "tur",
    authors: list[str] | None = None,
    article_url_id: str | None = None,
) -> dict:
    article_id = member_name.removesuffix(".json").rsplit("_", 1)[1]
    return {
        "filename": member_name.removesuffix(".json"),
        "url": f"https://example.invalid/a/{article_url_id or article_id}.html",
        "url_origin": "https://example.invalid/",
        "content_type": "article",
        "site_language": site_language,
        "time_published": "2019-12-31T00:00:00Z",
        "time_modified": modified,
        "time_retrieved": retrieved,
        "title": title,
        "authors": [] if authors is None else authors,
        "paragraphs": paragraphs,
        "n_paragraphs": len(paragraphs),
        "n_chars": sum(len(paragraph) for paragraph in paragraphs),
        "predicted_language": predicted_language,
    }


def _mot_contract(old_path: Path, new_path: Path, **changes: int) -> MotContract:
    contract = MotContract(
        assets=(
            MotAssetContract(
                old_path.name,
                "tur_amerikaninsesi",
                old_path.stat().st_size,
                MOT_V1_11_CONTRACT.assets[0].source_url,
            ),
            MotAssetContract(
                new_path.name,
                "tur_voaturkce",
                new_path.stat().st_size,
                MOT_V1_11_CONTRACT.assets[1].source_url,
            ),
        )
    )
    return replace(contract, **changes)


def _mot_fixture(tmp_path: Path) -> tuple[Path, Path, MotContract]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    old_path = tmp_path / "tur_amerikaninsesi.tgz"
    new_path = tmp_path / "tur_voaturkce.tgz"

    identical_old = _mot_record(
        "haber_100.json",
        paragraphs=["Aynı haber metni."],
        modified="2024-03-01T00:00:00Z",
    )
    identical_new = dict(identical_old)
    identical_new["url"] = "https://example.invalid/new/100.html"
    conflict_old = _mot_record(
        "haber_200.json",
        paragraphs=["Daha yeni ve geçerli kopya."],
        modified="2024-04-01T00:00:00Z",
    )
    conflict_new = _mot_record(
        "haber_200.json",
        paragraphs=["Eski farklı kopya."],
        modified="2024-02-01T00:00:00Z",
    )
    bad_language = _mot_record(
        "haber_300.json", paragraphs=["English text."], predicted_language="eng"
    )
    agency = _mot_record(
        "haber_400.json", paragraphs=["Ajans metni."], authors=["Reuters"]
    )
    unicode_boundary = _mot_record(
        "haber_500.json", paragraphs=["Türkçe metin."], authors=["Çap Haber Merkezi"]
    )

    _write_tgz(
        old_path,
        [
            ("tur_amerikaninsesi/", None),
            ("tur_amerikaninsesi/article/", None),
            ("tur_amerikaninsesi/audio/", None),
            ("tur_amerikaninsesi/audio/ignored.json", b"{}"),
            ("tur_amerikaninsesi/article/haber_100.json", _json_bytes(identical_old)),
            ("tur_amerikaninsesi/article/haber_200.json", _json_bytes(conflict_old)),
            ("tur_amerikaninsesi/article/haber_300.json", _json_bytes(bad_language)),
            ("tur_amerikaninsesi/article/haber_400.json", _json_bytes(agency)),
            ("tur_amerikaninsesi/article/haber_500.json", _json_bytes(unicode_boundary)),
        ],
    )
    _write_tgz(
        new_path,
        [
            ("tur_voaturkce/", None),
            ("tur_voaturkce/article/", None),
            ("tur_voaturkce/article/haber_100.json", _json_bytes(identical_new)),
            ("tur_voaturkce/article/haber_200.json", _json_bytes(conflict_new)),
        ],
    )
    return old_path, new_path, _mot_contract(old_path, new_path)


def _run_mot(tmp_path: Path, output_name: str = "mot-out") -> tuple[Path, dict, MotContract, Path]:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    receipt = _seal_acquisition(
        tmp_path,
        MOT_SOURCE_ID,
        [old_path, new_path],
        contract,
        name=f"{output_name}-acquisition.json",
    )
    output = tmp_path / output_name
    manifest = prepare_mot_v1_11(
        old_path,
        new_path,
        output,
        acquisition_receipt_path=receipt,
        discovery=True,
        shard_target_bytes=300,
        evidence_target_bytes=400,
        contract=contract,
    )
    return output, manifest, contract, receipt


def test_mot_preparation_resolves_overlap_and_declares_downstream_gates(
    tmp_path: Path,
) -> None:
    output, manifest, contract, _receipt = _run_mot(tmp_path)

    rows = _read_artifact(output, manifest["artifacts"]["data"])
    by_id = {row["id"]: row for row in rows}
    assert sorted(by_id) == ["mot:v1.11:100", "mot:v1.11:200", "mot:v1.11:500"]
    assert by_id["mot:v1.11:100"]["provenance"]["selected_site"] == "tur_voaturkce"
    assert by_id["mot:v1.11:100"]["provenance"]["selection_reason"] == (
        "identical_clean_text_prefer_voaturkce"
    )
    assert by_id["mot:v1.11:200"]["provenance"]["selected_site"] == (
        "tur_amerikaninsesi"
    )
    assert by_id["mot:v1.11:200"]["provenance"]["selection_reason"] == (
        "conflicting_clean_text_newest_modified_then_retrieved"
    )
    assert len(by_id["mot:v1.11:100"]["provenance"]["aliases"]) == 2
    assert manifest["counts"]["quarantine_reasons"] == {
        "mot_ap_afp_reuters_provenance": 1,
        "mot_language_not_tur": 1,
    }
    assert manifest["counts"]["resolution"]["conflicting_article_ids"] == 1
    assert manifest["production_acceptance"]["stage"] == "discovery_unaccepted"
    assert manifest["production_acceptance"]["eligible_for_production"] is False
    assert manifest["downstream_admission"]["eligible_for_training"] is False
    assert manifest["downstream_admission"]["integration_status"] == (
        "not_implemented_by_anchor_preparer"
    )
    assert "required_next_stage" not in manifest["downstream_admission"]
    assert manifest["downstream_admission"]["turkish_language_gate"]["required"] is True
    assert manifest["downstream_admission"]["no_code_gate"]["required"] is True
    assert manifest["archive_member_policy"]["pdf_ocr_fallback"] == "forbidden"
    assert manifest["cleaning"]["pdf_ocr_fallback"] is False
    assert validate_anchor_preparation(output, contract=contract) == manifest


def test_mot_discovery_acceptance_and_production_rerun(tmp_path: Path) -> None:
    output, discovery, contract, acquisition = _run_mot(tmp_path, "discovery")
    acceptance = tmp_path / "count-acceptance.json"
    receipt = seal_anchor_count_acceptance(
        output,
        acceptance,
        reviewer="count-reviewer",
        reviewed_at_utc=_TEST_TIMESTAMP,
        decision="accepted",
        contract=contract,
    )
    old_path = tmp_path / "tur_amerikaninsesi.tgz"
    new_path = tmp_path / "tur_voaturkce.tgz"
    production_dir = tmp_path / "production"
    production = prepare_mot_v1_11(
        old_path,
        new_path,
        production_dir,
        acquisition_receipt_path=acquisition,
        count_acceptance_path=acceptance,
        shard_target_bytes=300,
        evidence_target_bytes=400,
        contract=contract,
    )
    assert receipt["approved_projection"]["counts"] == discovery["counts"]
    assert receipt["approved_projection"]["frozen_contract_sha256"] == discovery[
        "frozen_contract_sha256"
    ]
    assert receipt["approved_projection"]["frozen_contract"] == (
        anchor_preparation._frozen_contract_projection(discovery)
    )
    assert production["production_acceptance"]["stage"] == "accepted_production"
    assert production["production_acceptance"]["eligible_for_production"] is True
    assert validate_anchor_preparation(production_dir, contract=contract) == production

    rejected_acceptance = tmp_path / "rejected-count-acceptance.json"
    seal_anchor_count_acceptance(
        output,
        rejected_acceptance,
        reviewer="count-reviewer",
        reviewed_at_utc=_TEST_TIMESTAMP,
        decision="rejected",
        contract=contract,
    )
    with pytest.raises(AnchorPreparationError, match="count receipt contract drift"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "rejected-production",
            acquisition_receipt_path=acquisition,
            count_acceptance_path=rejected_acceptance,
            shard_target_bytes=300,
            evidence_target_bytes=400,
            contract=contract,
        )

    with pytest.raises(AnchorPreparationError, match="count receipt contract drift"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "sharding-drift",
            acquisition_receipt_path=acquisition,
            count_acceptance_path=acceptance,
            shard_target_bytes=10_000,
            evidence_target_bytes=10_000,
            contract=contract,
        )


def test_mot_rerun_is_byte_deterministic_and_never_overwrites(tmp_path: Path) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="acquisition.json"
    )
    outputs = [tmp_path / "first", tmp_path / "second"]
    manifests = []
    for output in outputs:
        manifests.append(
            prepare_mot_v1_11(
                old_path,
                new_path,
                output,
                acquisition_receipt_path=acquisition,
                discovery=True,
                shard_target_bytes=300,
                evidence_target_bytes=400,
                contract=contract,
            )
        )
    assert manifests[0] == manifests[1]
    first_files = {
        path.relative_to(outputs[0]).as_posix(): file_sha256(path)
        for path in outputs[0].rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(outputs[1]).as_posix(): file_sha256(path)
        for path in outputs[1].rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    with pytest.raises(AnchorPreparationError, match="refusing to overwrite"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            outputs[0],
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


def test_acquisition_receipt_binds_exact_archive_bytes_and_decision(tmp_path: Path) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="acquisition.json"
    )
    mutated = bytearray(old_path.read_bytes())
    mutated[len(mutated) // 2] ^= 1
    old_path.write_bytes(mutated)
    with pytest.raises(AnchorPreparationError, match="SHA-256 mismatch"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "mutated-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


def test_parser_reaudits_the_exact_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="acquisition.json"
    )
    original = anchor_preparation._snapshot_verified_archive
    corrupted = False

    def corrupt_after_snapshot(*args, **kwargs):
        nonlocal corrupted
        snapshot = original(*args, **kwargs)
        if not corrupted:
            payload = bytearray(snapshot.path.read_bytes())
            payload[-1] ^= 1
            os.chmod(snapshot.path, 0o600)
            snapshot.path.write_bytes(payload)
            os.chmod(snapshot.path, 0o400)
            corrupted = True
        return snapshot

    monkeypatch.setattr(
        anchor_preparation, "_snapshot_verified_archive", corrupt_after_snapshot
    )
    with pytest.raises(AnchorPreparationError):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "parser-binding-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


def test_rejected_acquisition_receipt_cannot_prepare(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    old_path, new_path, contract = _mot_fixture(fixture)
    acquisition = _seal_acquisition(
        tmp_path,
        MOT_SOURCE_ID,
        [old_path, new_path],
        contract,
        name="rejected-acquisition.json",
        decision="rejected",
    )
    with pytest.raises(AnchorPreparationError, match="accepted acquisition receipt"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "rejected-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


def test_acquisition_contract_freezes_official_asset_urls_and_identity(
    tmp_path: Path,
) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="acquisition.json"
    )
    receipt = json.loads(acquisition.read_text(encoding="utf-8"))
    assert [asset["source_url"] for asset in receipt["assets"]] == [
        MOT_V1_11_CONTRACT.assets[0].source_url,
        MOT_V1_11_CONTRACT.assets[1].source_url,
    ]
    assert [asset.size_bytes for asset in MOT_V1_11_CONTRACT.assets] == [
        219_280_046,
        264_239_626,
    ]
    assert len(receipt["asset_contract_sha256"]) == 64
    official_parlamint = anchor_preparation.PARLAMINT_TR_V5_CONTRACT
    assert official_parlamint.size_bytes == 297_184_431
    assert official_parlamint.md5 == "9b0f2d5588c689e648555957f2668ff1"
    assert official_parlamint.source_url.endswith(
        "ParlaMint-TR.tgz?sequence=28&isAllowed=y"
    )

    rogue_mot = replace(
        contract,
        assets=(
            replace(
                contract.assets[0],
                source_url="https://example.invalid/arbitrary-mot.tgz",
            ),
            contract.assets[1],
        ),
    )
    with pytest.raises(AnchorPreparationError, match="official release identity drift"):
        seal_anchor_acquisition_receipt(
            MOT_SOURCE_ID,
            [old_path, new_path],
            tmp_path / "rogue-mot-receipt.json",
            reviewer="unit-test-reviewer",
            acquired_at_utc=_TEST_TIMESTAMP,
            contract=rogue_mot,
        )
    with pytest.raises(AnchorPreparationError, match="official release identity drift"):
        seal_anchor_acquisition_receipt(
            PARLAMINT_SOURCE_ID,
            [old_path],
            tmp_path / "rogue-parlamint-receipt.json",
            reviewer="unit-test-reviewer",
            acquired_at_utc=_TEST_TIMESTAMP,
            contract=replace(
                official_parlamint,
                source_url="https://example.invalid/arbitrary-parlamint.tgz",
            ),
        )

    receipt["assets"][0]["source_url"] = "https://example.invalid/not-official.tgz"
    write_json_atomic(acquisition, seal_manifest(receipt))
    with pytest.raises(AnchorPreparationError, match="identity drift"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "url-drift-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )

    from scripts.prepare_turkish_anchors import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "seal-parlamint-acquisition",
                "--archive-tgz",
                "ParlaMint-TR.tgz",
                "--source-url",
                "https://example.invalid/override.tgz",
                "--output",
                "receipt.json",
                "--reviewer",
                "reviewer",
                "--acquired-at-utc",
                _TEST_TIMESTAMP,
            ]
        )


def test_receipt_publication_is_atomic_no_replace_under_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    receipt = tmp_path / "raced-receipt.json"
    marker = b"competitor-must-survive"
    original = anchor_preparation._rename_noreplace_at
    raced = False
    monkeypatch.setattr(
        anchor_preparation,
        "_native_rename_noreplace_at",
        lambda *_args, **_kwargs: anchor_preparation.errno.EINVAL,
    )

    def install_competitor_before_publish(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal raced
        if destination_name == receipt.name and not raced:
            raced = True
            descriptor = os.open(
                destination_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=destination_parent_fd,
            )
            try:
                os.write(descriptor, marker)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        original(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
            label=label,
        )

    monkeypatch.setattr(
        anchor_preparation, "_rename_noreplace_at", install_competitor_before_publish
    )
    with pytest.raises(AnchorPreparationError, match="refusing to overwrite"):
        _seal_acquisition(
            tmp_path,
            MOT_SOURCE_ID,
            [old_path, new_path],
            contract,
            name=receipt.name,
        )
    assert raced
    assert receipt.read_bytes() == marker
    assert not list(tmp_path.glob(f".{receipt.name}.*.tmp"))
    assert not list(tmp_path.glob(".anchor-publish-*.lock"))


def test_beegfs_einval_fallback_publishes_receipt_and_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    monkeypatch.setattr(
        anchor_preparation,
        "_native_rename_noreplace_at",
        lambda *_args, **_kwargs: anchor_preparation.errno.EINVAL,
    )
    monkeypatch.setattr(
        anchor_preparation,
        "_plain_directory_rename_is_noreplace",
        lambda _parent_fd: True,
    )
    acquisition = _seal_acquisition(
        tmp_path,
        MOT_SOURCE_ID,
        [old_path, new_path],
        contract,
        name="beegfs-acquisition.json",
    )
    output = tmp_path / "beegfs-output"
    manifest = prepare_mot_v1_11(
        old_path,
        new_path,
        output,
        acquisition_receipt_path=acquisition,
        discovery=True,
        contract=contract,
    )
    assert acquisition.is_file()
    assert validate_anchor_preparation(output, contract=contract) == manifest
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(f".{output.name}.build-*"))
    assert not list(tmp_path.glob(".anchor-publish-*.lock"))


def test_beegfs_file_fallback_rolls_back_inode_linked_after_source_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.tmp"
    replacement = tmp_path / "replacement.tmp"
    parked = tmp_path / "parked-original.tmp"
    destination = tmp_path / "receipt.json"
    source.write_bytes(b"original")
    replacement.write_bytes(b"replacement-must-survive")
    original_link = os.link
    swapped = False

    monkeypatch.setattr(
        anchor_preparation,
        "_native_rename_noreplace_at",
        lambda *_args, **_kwargs: anchor_preparation.errno.EINVAL,
    )

    def swap_source_before_link(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.rename(source, parked)
            os.rename(replacement, source)
        return original_link(*args, **kwargs)

    monkeypatch.setattr(anchor_preparation.os, "link", swap_source_before_link)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(AnchorPreparationError, match="source inode binding drift"):
            anchor_preparation._rename_noreplace_at(
                parent_fd,
                source.name,
                parent_fd,
                destination.name,
                label="receipt",
            )
    finally:
        os.close(parent_fd)
    assert swapped
    assert not destination.exists()
    assert source.read_bytes() == b"replacement-must-survive"
    assert parked.read_bytes() == b"original"


def test_beegfs_file_fallback_does_not_strand_link_when_source_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "receipt.json"
    source.write_bytes(b"owned")
    original_link = os.link

    monkeypatch.setattr(
        anchor_preparation,
        "_native_rename_noreplace_at",
        lambda *_args, **_kwargs: anchor_preparation.errno.EINVAL,
    )

    def remove_source_after_link(*args, **kwargs):
        original_link(*args, **kwargs)
        source.unlink()

    monkeypatch.setattr(anchor_preparation.os, "link", remove_source_after_link)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(FileNotFoundError):
            anchor_preparation._rename_noreplace_at(
                parent_fd,
                source.name,
                parent_fd,
                destination.name,
                label="receipt",
            )
    finally:
        os.close(parent_fd)
    assert not source.exists()
    assert not destination.exists()


def test_directory_fallback_rejects_replacement_capable_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "build"
    source.mkdir()
    (source / "payload").write_bytes(b"owned")
    destination = tmp_path / "published"
    monkeypatch.setattr(
        anchor_preparation,
        "_native_rename_noreplace_at",
        lambda *_args, **_kwargs: anchor_preparation.errno.EINVAL,
    )
    monkeypatch.setattr(
        anchor_preparation,
        "_plain_directory_rename_is_noreplace",
        lambda _parent_fd: False,
    )
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(AnchorPreparationError, match="lacks safe no-replace"):
            anchor_preparation._rename_noreplace_at(
                parent_fd,
                source.name,
                parent_fd,
                destination.name,
                label="output",
            )
    finally:
        os.close(parent_fd)
    assert source.is_dir()
    assert not destination.exists()
    assert not list(tmp_path.glob(".anchor-publish-*.lock"))


def test_directory_noreplace_probe_rejects_local_replacement_semantics_and_cleans(
    tmp_path: Path,
) -> None:
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert anchor_preparation._plain_directory_rename_is_noreplace(parent_fd) is False
    finally:
        os.close(parent_fd)
    assert not list(tmp_path.glob(".anchor-rename-probe-*"))


def test_directory_noreplace_probe_accepts_beegfs_eexist_semantics_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_rename = os.rename

    def emulate_beegfs(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        if str(src).startswith(".anchor-rename-probe-"):
            raise FileExistsError(anchor_preparation.errno.EEXIST, "BeeGFS EEXIST")
        return original_rename(
            src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd
        )

    monkeypatch.setattr(anchor_preparation.os, "rename", emulate_beegfs)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert anchor_preparation._plain_directory_rename_is_noreplace(parent_fd) is True
    finally:
        os.close(parent_fd)
    assert not list(tmp_path.glob(".anchor-rename-probe-*"))


def test_beegfs_directory_fallback_preserves_internal_window_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "build"
    source.mkdir()
    (source / "payload").write_bytes(b"owned")
    destination = tmp_path / "published"
    marker = b"competitor-must-survive"
    original_rename = os.rename
    raced = False

    monkeypatch.setattr(
        anchor_preparation,
        "_native_rename_noreplace_at",
        lambda *_args, **_kwargs: anchor_preparation.errno.EINVAL,
    )
    monkeypatch.setattr(
        anchor_preparation,
        "_plain_directory_rename_is_noreplace",
        lambda _parent_fd: True,
    )

    def emulate_beegfs_race(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        nonlocal raced
        if dst == destination.name and not raced:
            raced = True
            os.mkdir(dst, 0o700, dir_fd=dst_dir_fd)
            competitor_fd = os.open(
                dst, os.O_RDONLY | os.O_DIRECTORY, dir_fd=dst_dir_fd
            )
            try:
                descriptor = os.open(
                    "marker",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=competitor_fd,
                )
                try:
                    os.write(descriptor, marker)
                finally:
                    os.close(descriptor)
            finally:
                os.close(competitor_fd)
            raise FileExistsError(anchor_preparation.errno.EEXIST, "BeeGFS EEXIST")
        return original_rename(
            src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd
        )

    monkeypatch.setattr(anchor_preparation.os, "rename", emulate_beegfs_race)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(AnchorPreparationError, match="refusing to overwrite"):
            anchor_preparation._rename_noreplace_at(
                parent_fd,
                source.name,
                parent_fd,
                destination.name,
                label="output",
            )
    finally:
        os.close(parent_fd)
    assert raced
    assert (destination / "marker").read_bytes() == marker
    assert source.is_dir()
    assert not list(tmp_path.glob(".anchor-publish-*.lock"))


def test_receipt_publication_uses_held_parent_inode_during_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    old_path, new_path, contract = _mot_fixture(fixture)
    safe_parent = tmp_path / "safe-parent"
    evil_parent = tmp_path / "evil-parent"
    parked_parent = tmp_path / "parked-parent"
    safe_parent.mkdir()
    evil_parent.mkdir()
    receipt = safe_parent / "receipt.json"
    original = anchor_preparation._rename_noreplace_at
    swapped = False

    def swap_path_around_publish(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal swapped
        if destination_name == receipt.name and not swapped:
            swapped = True
            os.rename(safe_parent, parked_parent)
            os.rename(evil_parent, safe_parent)
            try:
                original(
                    source_parent_fd,
                    source_name,
                    destination_parent_fd,
                    destination_name,
                    label=label,
                )
            finally:
                os.rename(safe_parent, evil_parent)
                os.rename(parked_parent, safe_parent)
            return
        original(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
            label=label,
        )

    monkeypatch.setattr(anchor_preparation, "_rename_noreplace_at", swap_path_around_publish)
    sealed = _seal_acquisition(
        tmp_path,
        MOT_SOURCE_ID,
        [old_path, new_path],
        contract,
        name="safe-parent/receipt.json",
    )
    assert swapped
    assert sealed == receipt
    assert receipt.is_file()
    assert not (evil_parent / receipt.name).exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_mot_fails_closed_on_tar_links(tmp_path: Path, link_kind: str) -> None:
    old_path = tmp_path / "tur_amerikaninsesi.tgz"
    new_path = tmp_path / "tur_voaturkce.tgz"
    link = ("tur_amerikaninsesi/article/link_100.json", "target")
    kwargs = {link_kind: link}
    _write_tgz(old_path, [("tur_amerikaninsesi/", None)], **kwargs)
    _write_tgz(
        new_path,
        [
            ("tur_voaturkce/", None),
            (
                "tur_voaturkce/article/haber_200.json",
                _json_bytes(_mot_record("haber_200.json", paragraphs=["metin"])),
            ),
        ],
    )
    contract = _mot_contract(old_path, new_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="links.json"
    )
    with pytest.raises(AnchorPreparationError, match="links/devices/special"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "links-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


@pytest.mark.parametrize("header_format", ["pax", "gnu"])
def test_mot_bounds_hidden_tar_control_headers(tmp_path: Path, header_format: str) -> None:
    old_path = tmp_path / "tur_amerikaninsesi.tgz"
    new_path = tmp_path / "tur_voaturkce.tgz"
    if header_format == "pax":
        article_name = "haber_100.json"
        full_name = f"tur_amerikaninsesi/article/{article_name}"
        _write_tgz(
            old_path,
            [(full_name, _json_bytes(_mot_record(article_name, paragraphs=["metin"])))],
            pax_headers={full_name: {"comment": "x" * 512}},
        )
    else:
        article_name = f"{'x' * 140}_100.json"
        full_name = f"tur_amerikaninsesi/article/{article_name}"
        _write_tgz(
            old_path,
            [(full_name, _json_bytes(_mot_record(article_name, paragraphs=["metin"])))],
            archive_format=tarfile.GNU_FORMAT,
        )
    _write_tgz(
        new_path,
        [
            (
                "tur_voaturkce/article/haber_200.json",
                _json_bytes(_mot_record("haber_200.json", paragraphs=["metin"])),
            )
        ],
    )
    contract = _mot_contract(old_path, new_path, max_control_header_bytes=64)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="control.json"
    )
    with pytest.raises(AnchorPreparationError, match="control header exceeds"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "control-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


def test_mot_bounds_total_decompressed_tar_stream(tmp_path: Path) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    contract = replace(contract, max_tar_stream_bytes_per_archive=1_024)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="stream.json"
    )
    with pytest.raises(AnchorPreparationError, match="tar stream exceeds"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "stream-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


def test_mot_rejects_same_site_id_collision_and_url_identity_drift(tmp_path: Path) -> None:
    old_path = tmp_path / "tur_amerikaninsesi.tgz"
    new_path = tmp_path / "tur_voaturkce.tgz"
    _write_tgz(
        old_path,
        [
            (
                "tur_amerikaninsesi/article/haber_700.json",
                _json_bytes(_mot_record("haber_700.json", paragraphs=["ilk"])),
            ),
            (
                "tur_amerikaninsesi/article/baska_700.json",
                _json_bytes(_mot_record("baska_700.json", paragraphs=["ikinci"])),
            ),
        ],
    )
    _write_tgz(
        new_path,
        [
            (
                "tur_voaturkce/article/haber_800.json",
                _json_bytes(_mot_record("haber_800.json", paragraphs=["metin"])),
            )
        ],
    )
    contract = _mot_contract(old_path, new_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="collision.json"
    )
    with pytest.raises(AnchorPreparationError, match="same-site MOT article ID collision"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "collision-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )

    identity_dir = tmp_path / "identity"
    identity_dir.mkdir()
    old_path = identity_dir / "tur_amerikaninsesi.tgz"
    new_path = identity_dir / "tur_voaturkce.tgz"
    _write_tgz(
        old_path,
        [
            (
                "tur_amerikaninsesi/article/haber_900.json",
                _json_bytes(
                    _mot_record("haber_900.json", paragraphs=["metin"], article_url_id="901")
                ),
            )
        ],
    )
    _write_tgz(
        new_path,
        [
            (
                "tur_voaturkce/article/haber_902.json",
                _json_bytes(_mot_record("haber_902.json", paragraphs=["metin"])),
            )
        ],
    )
    contract = _mot_contract(old_path, new_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="identity.json"
    )
    with pytest.raises(AnchorPreparationError, match="URL/article ID identity drift"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "identity-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "tur_amerikaninsesi/article/scanned_source.pdf",
        "tur_amerikaninsesi/article/ocr_output.json",
    ],
)
def test_mot_has_no_pdf_or_ocr_fallback(
    tmp_path: Path, forbidden_name: str
) -> None:
    old_path = tmp_path / "tur_amerikaninsesi.tgz"
    new_path = tmp_path / "tur_voaturkce.tgz"
    _write_tgz(old_path, [(forbidden_name, b"fallback text")])
    _write_tgz(
        new_path,
        [
            (
                "tur_voaturkce/article/haber_200.json",
                _json_bytes(_mot_record("haber_200.json", paragraphs=["metin"])),
            )
        ],
    )
    contract = _mot_contract(old_path, new_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="native-only.json"
    )
    with pytest.raises(AnchorPreparationError, match="PDF/OCR fallback members are forbidden"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            tmp_path / "native-only-output",
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )


def _meta_payload(text_id: str, date: str, rows: list[tuple[str, str]]) -> bytes:
    lines = ["\t".join(PARLAMINT_NATIVE_META_HEADER)]
    indexes = {name: index for index, name in enumerate(PARLAMINT_NATIVE_META_HEADER)}
    for speech_id, language in rows:
        values = ["-" for _ in PARLAMINT_NATIVE_META_HEADER]
        values[indexes["Text_ID"]] = text_id
        values[indexes["ID"]] = speech_id
        values[indexes["Title"]] = "TBMM görüşmesi"
        values[indexes["Date"]] = date
        values[indexes["Lang"]] = language
        values[indexes["Speaker_name"]] = "TESTTE_METNE_GIRMEMELI"
        values[indexes["Speaker_party"]] = "PARTI_METNE_GIRMEMELI"
        lines.append("\t".join(values))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _tei(
    root: str = "TEI",
    language: str = "tr",
    *,
    speeches: int | None = None,
    words: int | None = None,
    includes: list[str] | None = None,
    include_doctype: bool = False,
    nested_depth: int = 0,
) -> bytes:
    extent = ""
    if speeches is not None or words is not None:
        assert speeches is not None and words is not None
        extent = (
            "<teiHeader><fileDesc><extent>"
            f'<measure unit="speeches" quantity="{speeches}"/>'
            f'<measure unit="words" quantity="{words}"/>'
            "</extent></fileDesc></teiHeader>"
        )
    include_xml = "".join(
        f'<xi:include xmlns:xi="http://www.w3.org/2001/XInclude" href="{href}"/>'
        for href in (includes or [])
    )
    nested = "<p>örnek</p>"
    if nested_depth:
        nested = "<div>" * nested_depth + nested + "</div>" * nested_depth
    doctype = '<!DOCTYPE TEI [<!ENTITY forbidden "x">]>' if include_doctype else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"{doctype}"
        f'<{root} xmlns="http://www.tei-c.org/ns/1.0" xml:lang="{language}">'
        f"{extent}{include_xml}<text><body>{nested}</body></text></{root}>"
    ).encode("utf-8")


def _parlamint_fixture(
    tmp_path: Path,
    *,
    first_language: str = "Türkçe",
    first_xml_language: str = "tr",
    include_second_meta: bool = True,
    first_directory_year: str = "2011",
    include_first_xinclude: bool = True,
    first_session_root: str = "TEI",
    aggregate_root: str = "teiCorpus",
    first_date: str = "2011-06-28",
    native_first_date: str | None = None,
    aggregate_doctype: bool = False,
    first_session_depth: int = 0,
    extra_member_name: str | None = None,
) -> tuple[Path, ParlaMintContract]:
    archive_path = tmp_path / "ParlaMint-TR.tgz"
    first_text_id = f"ParlaMint-TR_{first_date}-S1"
    native_first_text_id = (
        first_text_id
        if native_first_date is None
        else f"ParlaMint-TR_{native_first_date}-S1"
    )
    last_text_id = "ParlaMint-TR_2022-11-17-S2"
    first_text = "ilk konuşma [[alkışlar]] devam ediyor"
    last_text = "son konuşma metni"
    raw_words = len(first_text.split()) + len(last_text.split())
    members: list[tuple[str, bytes | None]] = [
        ("README-TR.md", b"ParlaMint-TR test release"),
        ("ParlaMint-TR.TEI/", None),
        (f"ParlaMint-TR.TEI/{first_directory_year}/", None),
        ("ParlaMint-TR.TEI/2022/", None),
        (
            "ParlaMint-TR.TEI/ParlaMint-TR.xml",
            _tei(
                aggregate_root,
                "tr",
                speeches=2,
                words=raw_words,
                includes=(
                    ([f"2011/{first_text_id}.xml"] if include_first_xinclude else [])
                    + [f"2022/{last_text_id}.xml"]
                ),
                include_doctype=aggregate_doctype,
            ),
        ),
        (
            f"ParlaMint-TR.TEI/{first_directory_year}/{first_text_id}.xml",
            _tei(
                root=first_session_root,
                language=first_xml_language,
                nested_depth=first_session_depth,
            ),
        ),
        (f"ParlaMint-TR.TEI/2022/{last_text_id}.xml", _tei()),
        ("ParlaMint-TR.txt/", None),
        (f"ParlaMint-TR.txt/{first_directory_year}/", None),
        ("ParlaMint-TR.txt/2022/", None),
        (
            f"ParlaMint-TR.txt/{first_directory_year}/{native_first_text_id}.txt",
            f"u1\t{first_text}\n".encode("utf-8"),
        ),
        (
            f"ParlaMint-TR.txt/{first_directory_year}/{first_text_id}-meta.tsv",
            _meta_payload(first_text_id, first_date, [("u1", first_language)]),
        ),
        (
            f"ParlaMint-TR.txt/2022/{last_text_id}.txt",
            f"u2\t{last_text}\n".encode("utf-8"),
        ),
    ]
    if include_second_meta:
        members.append(
            (
                f"ParlaMint-TR.txt/2022/{last_text_id}-meta.tsv",
                _meta_payload(last_text_id, "2022-11-17", [("u2", "Türkçe")]),
            )
        )
    if extra_member_name is not None:
        members.append((extra_member_name, b"forbidden fallback"))
    _write_tgz(archive_path, members)
    md5 = hashlib.md5(archive_path.read_bytes(), usedforsecurity=False).hexdigest()
    contract = ParlaMintContract(
        size_bytes=archive_path.stat().st_size,
        md5=md5,
        expected_speeches=2,
        expected_declared_words=raw_words,
        raw_word_count_min=raw_words,
        raw_word_count_max=raw_words,
    )
    return archive_path, contract


def _prepare_parlamint(
    tmp_path: Path,
    archive: Path,
    contract: ParlaMintContract,
    *,
    output_name: str,
) -> tuple[Path, dict]:
    receipt = _seal_acquisition(
        tmp_path,
        PARLAMINT_SOURCE_ID,
        [archive],
        contract,
        name=f"{output_name}-acquisition.json",
    )
    output = tmp_path / output_name
    manifest = prepare_parlamint_tr_v5(
        archive,
        output,
        acquisition_receipt_path=receipt,
        discovery=True,
        shard_target_bytes=250,
        evidence_target_bytes=400,
        contract=contract,
    )
    return output, manifest


def test_parlamint_preparation_strips_only_comments_and_joins_native_meta(
    tmp_path: Path,
) -> None:
    archive, contract = _parlamint_fixture(tmp_path)
    output, manifest = _prepare_parlamint(
        tmp_path, archive, contract, output_name="parlamint-out"
    )
    rows = _read_artifact(output, manifest["artifacts"]["data"])
    assert [row["id"] for row in rows] == [
        "parlamint-tr:v5.0:u1",
        "parlamint-tr:v5.0:u2",
    ]
    assert rows[0]["text"] == "ilk konuşma devam ediyor"
    rendered = json.dumps(rows, ensure_ascii=False)
    assert "TESTTE_METNE_GIRMEMELI" not in rendered
    assert "PARTI_METNE_GIRMEMELI" not in rendered
    assert manifest["counts"]["join"]["raw_whitespace_words"] == 8
    assert manifest["counts"]["join"]["first_date"] == "2011-06-28"
    assert manifest["counts"]["join"]["last_date"] == "2022-11-17"
    assert len(
        set(manifest["counts"]["join"]["cross_format_identity_counts"].values())
    ) == 1
    assert not any(
        manifest["counts"]["join"]["cross_format_identity_mismatches"].values()
    )
    assert manifest["archive_member_policy"]["pdf_ocr_fallback"] == "forbidden"
    assert manifest["cleaning"]["pdf_ocr_fallback"] is False
    assert validate_anchor_preparation(output, contract=contract) == manifest


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"first_language": "Turkish"}, "Lang must be exactly 'Türkçe'"),
        ({"first_xml_language": "en"}, "root xml:lang must be 'tr'"),
        ({"include_second_meta": False}, "identity equality failed"),
        ({"first_directory_year": "2012"}, "directory-year/date drift"),
        ({"include_first_xinclude": False}, "identity equality failed"),
        ({"first_session_root": "teiCorpus"}, "session root must be TEI"),
        ({"aggregate_root": "TEI"}, "aggregate root must be teiCorpus"),
        ({"first_date": "2011-02-30"}, "invalid calendar date"),
        ({"first_date": "2011-06-29"}, "date-span drift"),
        ({"native_first_date": "2011-06-29"}, "identity equality failed"),
        ({"aggregate_doctype": True}, "DTD/entity declarations are forbidden"),
        ({"first_session_depth": 260}, "TEI depth exceeds"),
    ],
)
def test_parlamint_fails_closed_on_language_xml_join_and_year_drift(
    tmp_path: Path, fixture_kwargs: dict, message: str
) -> None:
    archive, contract = _parlamint_fixture(tmp_path, **fixture_kwargs)
    receipt = _seal_acquisition(
        tmp_path, PARLAMINT_SOURCE_ID, [archive], contract, name="acquisition.json"
    )
    output = tmp_path / "failed-output"
    with pytest.raises(AnchorPreparationError, match=message):
        prepare_parlamint_tr_v5(
            archive,
            output,
            acquisition_receipt_path=receipt,
            discovery=True,
            contract=contract,
        )
    assert not output.exists()


def test_parlamint_fails_closed_on_official_tei_extent_drift(tmp_path: Path) -> None:
    archive, contract = _parlamint_fixture(tmp_path)
    drifted_contract = replace(
        contract, expected_declared_words=contract.expected_declared_words + 1
    )
    receipt = _seal_acquisition(
        tmp_path,
        PARLAMINT_SOURCE_ID,
        [archive],
        drifted_contract,
        name="acquisition.json",
    )
    with pytest.raises(AnchorPreparationError, match="aggregate TEI extent drift"):
        prepare_parlamint_tr_v5(
            archive,
            tmp_path / "extent-drift-output",
            acquisition_receipt_path=receipt,
            discovery=True,
            contract=drifted_contract,
        )


@pytest.mark.parametrize(
    "forbidden_name",
    ["ParlaMint-TR.TEI/source.pdf", "ParlaMint-TR.txt/ocr/session.txt"],
)
def test_parlamint_has_no_pdf_or_ocr_fallback(
    tmp_path: Path, forbidden_name: str
) -> None:
    archive, contract = _parlamint_fixture(
        tmp_path, extra_member_name=forbidden_name
    )
    receipt = _seal_acquisition(
        tmp_path, PARLAMINT_SOURCE_ID, [archive], contract, name="acquisition.json"
    )
    with pytest.raises(AnchorPreparationError, match="PDF/OCR fallback members are forbidden"):
        prepare_parlamint_tr_v5(
            archive,
            tmp_path / "fallback-output",
            acquisition_receipt_path=receipt,
            discovery=True,
            contract=contract,
        )


def test_parlamint_unbalanced_comment_is_quarantined_with_evidence(
    tmp_path: Path,
) -> None:
    archive, _contract = _parlamint_fixture(tmp_path)
    text_id = "ParlaMint-TR_2011-06-28-S1"
    last_id = "ParlaMint-TR_2022-11-17-S2"
    first_text = "bozuk [[yorum"
    last_text = "geçerli konuşma"
    raw_words = len(first_text.split()) + len(last_text.split())
    _write_tgz(
        archive,
        [
            ("README-TR.md", b"test"),
            (
                "ParlaMint-TR.TEI/ParlaMint-TR.xml",
                _tei(
                    "teiCorpus",
                    speeches=2,
                    words=raw_words,
                    includes=[f"2011/{text_id}.xml", f"2022/{last_id}.xml"],
                ),
            ),
            (f"ParlaMint-TR.TEI/2011/{text_id}.xml", _tei()),
            (f"ParlaMint-TR.TEI/2022/{last_id}.xml", _tei()),
            (f"ParlaMint-TR.txt/2011/{text_id}.txt", f"u1\t{first_text}\n".encode()),
            (
                f"ParlaMint-TR.txt/2011/{text_id}-meta.tsv",
                _meta_payload(text_id, "2011-06-28", [("u1", "Türkçe")]),
            ),
            (f"ParlaMint-TR.txt/2022/{last_id}.txt", f"u2\t{last_text}\n".encode()),
            (
                f"ParlaMint-TR.txt/2022/{last_id}-meta.tsv",
                _meta_payload(last_id, "2022-11-17", [("u2", "Türkçe")]),
            ),
        ],
    )
    contract = ParlaMintContract(
        size_bytes=archive.stat().st_size,
        md5=hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest(),
        expected_speeches=2,
        expected_declared_words=raw_words,
        raw_word_count_min=raw_words,
        raw_word_count_max=raw_words,
    )
    output, manifest = _prepare_parlamint(
        tmp_path, archive, contract, output_name="quarantine-out"
    )
    rows = _read_artifact(output, manifest["artifacts"]["data"])
    quarantine = _read_artifact(output, manifest["artifacts"]["quarantine"])
    assert [row["id"] for row in rows] == ["parlamint-tr:v5.0:u2"]
    assert quarantine[0]["reason"] == "parlamint_unbalanced_transcriber_comment"
    assert quarantine[0]["evidence"]["kind"] == "missing_closing_delimiter"


def test_validator_semantically_recounts_shards(tmp_path: Path) -> None:
    output, manifest, contract, _receipt = _run_mot(tmp_path)
    tampered = copy.deepcopy(manifest)
    shard = tampered["artifacts"]["data"]["shards"][0]
    shard["rows"] += 1
    shard["row_end_exclusive"] += 1
    tampered["artifacts"]["data"]["totals"]["rows"] += 1
    tampered["clean"]["documents"] += 1
    tampered["counts"]["resolution"]["documents"] += 1
    write_json_atomic(output / "manifest.json", seal_manifest(tampered))
    with pytest.raises(AnchorPreparationError, match="semantic count drift"):
        validate_anchor_preparation(output, contract=contract)


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("source", "license", "unreviewed-license"),
        (
            "archive_member_policy",
            "forbidden_member_types",
            ["symlink", "hardlink"],
        ),
        ("cleaning", "agency_policy", "accept-everything"),
    ],
)
def test_validator_rejects_self_consistently_resealed_frozen_contract_drift(
    tmp_path: Path, section: str, key: str, replacement: object
) -> None:
    output, manifest, contract, _receipt = _run_mot(tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered[section][key] = replacement
    tampered["frozen_contract_sha256"] = anchor_preparation._frozen_contract_sha256(
        tampered
    )
    write_json_atomic(output / "manifest.json", seal_manifest(tampered))
    with pytest.raises(AnchorPreparationError, match="frozen manifest contract drift"):
        validate_anchor_preparation(output, contract=contract)


def test_output_publication_is_atomic_no_replace_under_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="acquisition.json"
    )
    output = tmp_path / "raced-output"
    original = anchor_preparation._rename_noreplace_at
    raced = False
    monkeypatch.setattr(
        anchor_preparation,
        "_native_rename_noreplace_at",
        lambda *_args, **_kwargs: anchor_preparation.errno.EINVAL,
    )
    monkeypatch.setattr(
        anchor_preparation,
        "_plain_directory_rename_is_noreplace",
        lambda _parent_fd: True,
    )

    def install_competing_directory(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
        *,
        label: str,
    ) -> None:
        nonlocal raced
        if destination_name == output.name and not raced:
            raced = True
            os.mkdir(destination_name, 0o700, dir_fd=destination_parent_fd)
        original(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
            label=label,
        )

    monkeypatch.setattr(
        anchor_preparation, "_rename_noreplace_at", install_competing_directory
    )
    with pytest.raises(AnchorPreparationError, match="refusing to overwrite"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            output,
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )
    assert raced
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(f".{output.name}.build-*"))
    assert not list(tmp_path.glob(".anchor-publish-*.lock"))


def test_shard_replacement_after_prepublication_validation_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_path, new_path, contract = _mot_fixture(tmp_path)
    acquisition = _seal_acquisition(
        tmp_path, MOT_SOURCE_ID, [old_path, new_path], contract, name="acquisition.json"
    )
    output = tmp_path / "post-validation-race"
    original = anchor_preparation._validate_anchor_preparation_fd
    validations = 0
    replaced = False

    def replace_after_second_validation(*args, **kwargs):
        nonlocal validations, replaced
        manifest = original(*args, **kwargs)
        validations += 1
        if validations == 2:
            builds = list(tmp_path.glob(f".{output.name}.build-*"))
            assert len(builds) == 1
            shard_path = builds[0] / manifest["artifacts"]["data"]["shards"][0]["path"]
            replacement = shard_path.with_name(f".{shard_path.name}.replacement")
            replacement.write_bytes(shard_path.read_bytes() + b"race")
            os.replace(replacement, shard_path)
            replaced = True
        return manifest

    monkeypatch.setattr(
        anchor_preparation,
        "_validate_anchor_preparation_fd",
        replace_after_second_validation,
    )
    with pytest.raises(AnchorPreparationError, match="size drift"):
        prepare_mot_v1_11(
            old_path,
            new_path,
            output,
            acquisition_receipt_path=acquisition,
            discovery=True,
            contract=contract,
        )
    assert replaced
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.build-*"))
    assert not list(tmp_path.glob(f".{output.name}.failed-*"))


def test_validator_detects_inode_replacement_during_shard_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _manifest, contract, _receipt = _run_mot(tmp_path)
    original = anchor_preparation._stream_validate_shard
    replaced = False

    def replace_validated_shard(*args, **kwargs):
        nonlocal replaced
        result = original(*args, **kwargs)
        if not replaced and kwargs.get("artifact_key") == "data":
            shard_path = output / kwargs["item"]["path"]
            replacement = shard_path.with_name(f".{shard_path.name}.replacement")
            replacement.write_bytes(shard_path.read_bytes())
            os.replace(replacement, shard_path)
            replaced = True
        return result

    monkeypatch.setattr(
        anchor_preparation, "_stream_validate_shard", replace_validated_shard
    )
    with pytest.raises(AnchorPreparationError, match="replaced after validation"):
        validate_anchor_preparation(output, contract=contract)
    assert replaced


def test_validator_rejects_tampered_unrecorded_and_symlink_files(tmp_path: Path) -> None:
    output, manifest, contract, _receipt = _run_mot(tmp_path)
    shard = output / manifest["artifacts"]["data"]["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(AnchorPreparationError, match="size drift"):
        validate_anchor_preparation(output, contract=contract)

    output, _manifest, contract, _receipt = _run_mot(
        tmp_path / "rogue-fixture", "rogue-output"
    )
    (output / "rogue.txt").write_text("unrecorded", encoding="utf-8")
    with pytest.raises(AnchorPreparationError, match="tree is not closed"):
        validate_anchor_preparation(output, contract=contract)

    output, _manifest, contract, _receipt = _run_mot(
        tmp_path / "link-fixture", "link-output"
    )
    os.symlink("manifest.json", output / "rogue-link")
    with pytest.raises(AnchorPreparationError, match="contains a symlink"):
        validate_anchor_preparation(output, contract=contract)

    output, manifest, contract, _receipt = _run_mot(
        tmp_path / "hardlink-fixture", "hardlink-output"
    )
    shard = output / manifest["artifacts"]["data"]["shards"][0]["path"]
    os.link(shard, tmp_path / "external-hardlink.zst")
    with pytest.raises(AnchorPreparationError, match="hard-linked file"):
        validate_anchor_preparation(output, contract=contract)
