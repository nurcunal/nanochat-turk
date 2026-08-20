"""Offline, deterministic preparation for the Turkish high-trust anchors.

This module intentionally does not download anything. It accepts already staged
release archives plus sealed acquisition receipts, and emits canonical JSONL
compressed with Zstandard plus sealed provenance/evidence manifests. Archive members
are streamed; SQLite holds the bounded on-disk state needed for overlap resolution and
exact joins. A discovery run must be manually accepted by exact counts and logical
hashes before a fresh matching rerun can be marked production-eligible.

The production entry points are :func:`prepare_mot_v1_11` and
:func:`prepare_parlamint_tr_v5`.  Their default contracts are frozen to the official
MOT v1.11 and ParlaMint 5.0 releases.  Tests may inject a smaller contract, but the CLI
does not expose that escape hatch.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
import ctypes
import errno
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlparse

import pyarrow as pa

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)


PREPARER_VERSION = "turkish_anchor_preparation_v2"
MANIFEST_KIND = "turkish_high_trust_anchor_preparation"
MANIFEST_SCHEMA_VERSION = "1.0"
ACQUISITION_RECEIPT_KIND = "turkish_anchor_acquisition_receipt"
COUNT_ACCEPTANCE_KIND = "turkish_anchor_count_acceptance"
RECEIPT_SCHEMA_VERSION = "1.0"
MOT_SOURCE_ID = "mot_tr_v1_11"
PARLAMINT_SOURCE_ID = "parlamint_tr_v5_0"

MOT_RELEASE_TAG = "v1.11"
MOT_RELEASE_COMMIT = "9204cad746dfff0921e7a8f64c0cc0917bc75554"
MOT_RELEASE_URL = "https://github.com/bltlab/mot/releases/tag/v1.11"
MOT_REPOSITORY_URL = "https://github.com/bltlab/mot"
MOT_AMERIKANINSESI_URL = (
    "https://github.com/bltlab/mot/releases/download/v1.11/"
    "tur_amerikaninsesi.tgz"
)
MOT_VOATURKCE_URL = (
    "https://github.com/bltlab/mot/releases/download/v1.11/tur_voaturkce.tgz"
)
MOT_LICENSE = "CC BY 4.0"
MOT_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
MOT_CITATION_URL = "https://aclanthology.org/2022.lrec-1.224/"

PARLAMINT_RELEASE = "5.0"
PARLAMINT_RELEASE_TAG = "v5.0"
PARLAMINT_RELEASE_COMMIT = "978f9051793367fc2192eedc80c1e71ac8ec7ea5"
PARLAMINT_HANDLE = "http://hdl.handle.net/11356/2004"
PARLAMINT_REPOSITORY_URL = "https://github.com/clarin-eric/ParlaMint"
PARLAMINT_BITSTREAM_URL = (
    "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2004/"
    "ParlaMint-TR.tgz?sequence=28&isAllowed=y"
)
PARLAMINT_LICENSE = "CC BY 4.0"
PARLAMINT_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
PARLAMINT_OFFICIAL_MD5 = "9b0f2d5588c689e648555957f2668ff1"

DEFAULT_SHARD_TARGET_BYTES = 256 * 1024 * 1024
DEFAULT_EVIDENCE_TARGET_BYTES = 64 * 1024 * 1024
MAX_JSON_MEMBER_BYTES = 64 * 1024 * 1024
MAX_PARLAMINT_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 5_000_000
MAX_MOT_UNCOMPRESSED_BYTES_PER_ARCHIVE = 128 * 1024 * 1024 * 1024
MAX_PARLAMINT_UNCOMPRESSED_BYTES = 64 * 1024 * 1024 * 1024
MAX_MOT_TAR_STREAM_BYTES_PER_ARCHIVE = 136 * 1024 * 1024 * 1024
MAX_PARLAMINT_TAR_STREAM_BYTES = 72 * 1024 * 1024 * 1024
MAX_TAR_CONTROL_MEMBER_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
VALIDATION_CHUNK_BYTES = 1024 * 1024
MAX_TEI_PROLOG_BYTES = 1024 * 1024
MAX_TEI_DEPTH = 256
MAX_TEI_ATTRIBUTES_PER_ELEMENT = 256
MAX_TEI_ATTRIBUTE_BYTES = 1024 * 1024
REPEATED_PARAGRAPH_MIN_DOCUMENTS = 10
REPEATED_PARAGRAPH_SAMPLE_CHARS = 500

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_RFC3339_UTC_RE = re.compile(
    r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])T"
    r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\dZ$"
)
_MOT_ARTICLE_NAME_RE = re.compile(r"^.+_(\d+)\.json$")
_PARLAMINT_SESSION_XML_RE = re.compile(
    r"^ParlaMint-TR\.TEI/(\d{4})/(ParlaMint-TR_(\d{4}-\d{2}-\d{2})-.+)\.xml$"
)
_PARLAMINT_TEXT_RE = re.compile(
    r"^ParlaMint-TR\.txt/(\d{4})/(ParlaMint-TR_(\d{4}-\d{2}-\d{2})-.+)\.txt$"
)
_PARLAMINT_NATIVE_META_RE = re.compile(
    r"^ParlaMint-TR\.txt/(\d{4})/(ParlaMint-TR_(\d{4}-\d{2}-\d{2})-.+)-meta\.tsv$"
)
_PARLAMINT_EN_META_RE = re.compile(
    r"^ParlaMint-TR\.txt/(\d{4})/ParlaMint-TR_(\d{4}-\d{2}-\d{2})-.+-meta-en\.tsv$"
)
_PARLAMINT_SPEECH_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{1,255}$")
_TEI_NS = "http://www.tei-c.org/ns/1.0"
_XI_NS = "http://www.w3.org/2001/XInclude"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_PARLAMINT_XINCLUDE_RE = re.compile(
    r"^(\d{4})/(ParlaMint-TR_(\d{4}-\d{2}-\d{2})-.+)\.xml$"
)
_PARLAMINT_SCHEMA_SUPPORT_FILES = frozenset(
    {
        "ParlaMint-TEI.ana.rnc",
        "ParlaMint-TEI.ana.rng",
        "ParlaMint-TEI.rnc",
        "ParlaMint-TEI.rng",
        "ParlaMint-listOrg.rnc",
        "ParlaMint-listOrg.rng",
        "ParlaMint-listPerson.rnc",
        "ParlaMint-listPerson.rng",
        "ParlaMint-schemaSpecs.odd.xml",
        "ParlaMint-taxonomy.rnc",
        "ParlaMint-taxonomy.rng",
        "ParlaMint-teiCorpus.ana.rnc",
        "ParlaMint-teiCorpus.ana.rng",
        "ParlaMint-teiCorpus.rnc",
        "ParlaMint-teiCorpus.rng",
        "ParlaMint.odd.rnc",
        "ParlaMint.odd.rng",
        "ParlaMint.rnc",
        "ParlaMint.rng",
        "README.md",
        "parla-clarin.rnc",
        "parla-clarin.rng",
    }
)

PARLAMINT_NATIVE_META_HEADER = (
    "Text_ID",
    "ID",
    "Title",
    "Date",
    "Body",
    "Term",
    "Session",
    "Meeting",
    "Sitting",
    "Agenda",
    "Subcorpus",
    "Lang",
    "Speaker_role",
    "Speaker_MP",
    "Speaker_minister",
    "Speaker_party",
    "Speaker_party_name",
    "Party_status",
    "Party_orientation",
    "Speaker_ID",
    "Speaker_name",
    "Speaker_gender",
    "Speaker_birth",
    "Topic",
)

_MOT_REQUIRED_FIELDS = {
    "filename",
    "url",
    "url_origin",
    "content_type",
    "site_language",
    "time_published",
    "time_modified",
    "time_retrieved",
    "title",
    "paragraphs",
    "n_paragraphs",
    "n_chars",
    "predicted_language",
}
_MOT_NON_ARTICLE_KINDS = frozenset({"audio", "photo", "video"})
_AGENCY_NAMES = (
    "AP",
    "ASSOCIATED PRESS",
    "AFP",
    "AGENCE FRANCE-PRESSE",
    "REUTERS",
)


def _downstream_gate_declaration() -> dict[str, Any]:
    return {
        "eligible_for_training": False,
        "integration_status": "not_implemented_by_anchor_preparer",
        "required_before_training": [
            "independent_turkish_language_gate",
            "independent_no_code_gate",
        ],
        "turkish_language_gate": {
            "required": True,
            "status": "not_applied_by_anchor_preparer",
            "must_inspect_emitted_text": True,
        },
        "no_code_gate": {
            "required": True,
            "status": "not_applied_by_anchor_preparer",
            "must_inspect_emitted_text": True,
        },
        "semantic_rewriting": False,
        "automatic_admission": False,
    }


class AnchorPreparationError(ValueError):
    """Raised when a pinned input or deterministic preparation contract fails."""


@dataclass(frozen=True)
class MotAssetContract:
    filename: str
    root: str
    size_bytes: int
    source_url: str


@dataclass(frozen=True)
class MotContract:
    assets: tuple[MotAssetContract, MotAssetContract]
    release_tag: str = MOT_RELEASE_TAG
    release_commit: str = MOT_RELEASE_COMMIT
    max_members: int = MAX_ARCHIVE_MEMBERS
    max_member_bytes: int = MAX_JSON_MEMBER_BYTES
    max_uncompressed_bytes_per_archive: int = MAX_MOT_UNCOMPRESSED_BYTES_PER_ARCHIVE
    max_tar_stream_bytes_per_archive: int = MAX_MOT_TAR_STREAM_BYTES_PER_ARCHIVE
    max_control_header_bytes: int = MAX_TAR_CONTROL_MEMBER_BYTES


MOT_V1_11_CONTRACT = MotContract(
    assets=(
        MotAssetContract(
            filename="tur_amerikaninsesi.tgz",
            root="tur_amerikaninsesi",
            size_bytes=219_280_046,
            source_url=MOT_AMERIKANINSESI_URL,
        ),
        MotAssetContract(
            filename="tur_voaturkce.tgz",
            root="tur_voaturkce",
            size_bytes=264_239_626,
            source_url=MOT_VOATURKCE_URL,
        ),
    )
)


@dataclass(frozen=True)
class ParlaMintContract:
    filename: str = "ParlaMint-TR.tgz"
    size_bytes: int = 297_184_431
    md5: str = PARLAMINT_OFFICIAL_MD5
    source_url: str = PARLAMINT_BITSTREAM_URL
    expected_speeches: int = 681_052
    # The 5.0 TEI header publishes this exact extent; 49.26M is only its
    # human-readable rounded rendering.  Plain-text whitespace tokenization is
    # retained as audit evidence, not equated with the TEI word annotation.
    expected_declared_words: int = 49_255_262
    raw_word_count_min: int | None = None
    raw_word_count_max: int | None = None
    first_date: str = "2011-06-28"
    last_date: str = "2022-11-17"
    max_members: int = MAX_ARCHIVE_MEMBERS
    max_member_bytes: int = MAX_PARLAMINT_MEMBER_BYTES
    max_uncompressed_bytes: int = MAX_PARLAMINT_UNCOMPRESSED_BYTES
    max_tar_stream_bytes: int = MAX_PARLAMINT_TAR_STREAM_BYTES
    max_control_header_bytes: int = MAX_TAR_CONTROL_MEMBER_BYTES


PARLAMINT_TR_V5_CONTRACT = ParlaMintContract()


def _validate_mot_contract_identity(contract: MotContract) -> None:
    """Permit bounded test sizes while freezing every upstream identity field."""

    if not isinstance(contract, MotContract):
        raise AnchorPreparationError("MOT contract type drift")
    expected_identity = [
        (
            asset.filename,
            asset.root,
            asset.source_url,
        )
        for asset in MOT_V1_11_CONTRACT.assets
    ]
    observed_identity = [
        (asset.filename, asset.root, asset.source_url) for asset in contract.assets
    ]
    if (
        observed_identity != expected_identity
        or contract.release_tag != MOT_RELEASE_TAG
        or contract.release_commit != MOT_RELEASE_COMMIT
    ):
        raise AnchorPreparationError("MOT official release identity drift")
    for asset in contract.assets:
        _require_positive_int(asset.size_bytes, f"{asset.filename} size_bytes")
    for name in (
        "max_members",
        "max_member_bytes",
        "max_uncompressed_bytes_per_archive",
        "max_tar_stream_bytes_per_archive",
        "max_control_header_bytes",
    ):
        _require_positive_int(getattr(contract, name), f"MOT {name}")


def _validate_parlamint_contract_identity(contract: ParlaMintContract) -> None:
    """Permit bounded test totals while freezing the official release identity."""

    if not isinstance(contract, ParlaMintContract):
        raise AnchorPreparationError("ParlaMint contract type drift")
    if (
        contract.filename != PARLAMINT_TR_V5_CONTRACT.filename
        or contract.source_url != PARLAMINT_BITSTREAM_URL
        or contract.first_date != PARLAMINT_TR_V5_CONTRACT.first_date
        or contract.last_date != PARLAMINT_TR_V5_CONTRACT.last_date
    ):
        raise AnchorPreparationError("ParlaMint official release identity drift")
    if not _MD5_RE.fullmatch(contract.md5):
        raise AnchorPreparationError("ParlaMint frozen MD5 contract is malformed")
    for name in (
        "size_bytes",
        "expected_speeches",
        "expected_declared_words",
        "max_members",
        "max_member_bytes",
        "max_uncompressed_bytes",
        "max_tar_stream_bytes",
        "max_control_header_bytes",
    ):
        _require_positive_int(getattr(contract, name), f"ParlaMint {name}")
    bounds = (contract.raw_word_count_min, contract.raw_word_count_max)
    for name, value in zip(("raw_word_count_min", "raw_word_count_max"), bounds):
        if value is not None:
            _require_positive_int(value, f"ParlaMint {name}")
    if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
        raise AnchorPreparationError("ParlaMint raw word-count bounds are inverted")


def _require_sha256(value: str, location: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise AnchorPreparationError(
            f"{location} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _require_positive_int(value: int, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AnchorPreparationError(f"{location} must be a positive integer")
    return value


def _is_rfc3339_utc(value: Any) -> bool:
    if not isinstance(value, str) or not _RFC3339_UTC_RE.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _canonical_line(value: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(value)).encode("utf-8")


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_REGULAR_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


def _inode_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _open_directory_path(
    path: str | Path, *, label: str, create: bool = False
) -> tuple[int, Path]:
    """Open an absolute directory component-by-component without following links."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not parts or parts[0] != os.path.sep:
        raise AnchorPreparationError(f"{label} must resolve to an absolute path")
    try:
        descriptor = os.open(os.path.sep, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise AnchorPreparationError("cannot open filesystem root safely") from exc
    try:
        for component in parts[1:]:
            if component in {"", ".", ".."}:
                raise AnchorPreparationError(
                    f"{label} contains an unsafe directory component"
                )
            try:
                next_descriptor = os.open(
                    component, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(
                    component, _DIRECTORY_OPEN_FLAGS, dir_fd=descriptor
                )
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException as exc:
        os.close(descriptor)
        if isinstance(exc, AnchorPreparationError):
            raise
        raise AnchorPreparationError(
            f"cannot traverse {label} without following symlinks: {absolute}"
        ) from exc
    return descriptor, absolute


def _open_parent_directory(
    path: str | Path, *, label: str, create: bool = False
) -> tuple[int, Path, str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.name in {"", ".", ".."}:
        raise AnchorPreparationError(f"{label} requires a concrete basename")
    parent_fd, parent = _open_directory_path(
        absolute.parent, label=f"{label} parent", create=create
    )
    return parent_fd, parent, absolute.name


def _assert_path_binds_directory_fd(
    path: str | Path, descriptor: int, *, label: str
) -> None:
    probe, _absolute = _open_directory_path(path, label=label)
    try:
        if _inode_identity(os.fstat(probe)) != _inode_identity(os.fstat(descriptor)):
            raise AnchorPreparationError(f"{label} path/inode binding drift")
    finally:
        os.close(probe)


def _open_regular_at(
    parent_fd: int, name: str, *, label: str
) -> tuple[int, os.stat_result]:
    if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
        raise AnchorPreparationError(f"unsafe {label} basename")
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise AnchorPreparationError(f"{label} is missing or unreadable: {name}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise AnchorPreparationError(f"{label} must be a regular non-symlink file")
    try:
        descriptor = os.open(name, _REGULAR_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise AnchorPreparationError(f"cannot safely open {label}: {name}") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or _inode_identity(before) != _inode_identity(
        opened
    ):
        os.close(descriptor)
        raise AnchorPreparationError(f"{label} changed while it was opened: {name}")
    return descriptor, opened


def _open_regular_fd(path: Path, *, label: str) -> tuple[int, os.stat_result]:
    parent_fd, _parent, name = _open_parent_directory(path, label=label)
    try:
        return _open_regular_at(parent_fd, name, label=label)
    finally:
        os.close(parent_fd)


def _same_file_snapshot(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _same_file_identity_record(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_regular_file(
    path: Path, *, label: str, expected_size: int | None = None
) -> dict[str, Any]:
    descriptor, opened = _open_regular_fd(path, label=label)
    if expected_size is not None and opened.st_size != expected_size:
        os.close(descriptor)
        raise AnchorPreparationError(
            f"{label} size drift: expected {expected_size}, got {opened.st_size}"
        )
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    observed = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            for chunk in iter(lambda: handle.read(VALIDATION_CHUNK_BYTES), b""):
                observed += len(chunk)
                sha256.update(chunk)
                md5.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise AnchorPreparationError(f"cannot hash {label}: {path}") from exc
    if observed != opened.st_size or not _same_file_snapshot(opened, after):
        raise AnchorPreparationError(f"{label} changed while it was hashed: {path}")
    return {
        "filename": path.name,
        "size_bytes": observed,
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
    }


def _read_bounded_regular_file(path: Path, *, label: str, max_bytes: int) -> bytes:
    descriptor, opened = _open_regular_fd(path, label=label)
    if opened.st_size > max_bytes:
        os.close(descriptor)
        raise AnchorPreparationError(f"{label} exceeds {max_bytes} bytes")
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            payload = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise AnchorPreparationError(f"cannot read {label}: {path}") from exc
    if (
        len(payload) != opened.st_size
        or len(payload) > max_bytes
        or not _same_file_snapshot(opened, after)
    ):
        raise AnchorPreparationError(f"{label} changed while it was read: {path}")
    return payload


def _read_bounded_regular_at(
    parent_fd: int, name: str, *, label: str, max_bytes: int
) -> tuple[bytes, os.stat_result]:
    descriptor, opened = _open_regular_at(parent_fd, name, label=label)
    if opened.st_size > max_bytes:
        os.close(descriptor)
        raise AnchorPreparationError(f"{label} exceeds {max_bytes} bytes")
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            payload = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise AnchorPreparationError(f"cannot read {label}") from exc
    if (
        len(payload) != opened.st_size
        or len(payload) > max_bytes
        or not _same_file_snapshot(opened, after)
    ):
        raise AnchorPreparationError(f"{label} changed while it was read")
    return payload, after


def _native_rename_noreplace_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> int | None:
    """Return zero, errno, or ``None`` when no native primitive is available."""

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if hasattr(libc, "renameat2"):
        operation = libc.renameat2
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = operation(
            source_parent_fd,
            source,
            destination_parent_fd,
            destination,
            1,  # Linux RENAME_NOREPLACE
        )
    elif hasattr(libc, "renameatx_np"):
        operation = libc.renameatx_np
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = operation(
            source_parent_fd,
            source,
            destination_parent_fd,
            destination,
            0x00000004,  # Darwin RENAME_EXCL
        )
    else:
        return None
    return 0 if result == 0 else ctypes.get_errno()


def _require_publication_basename(name: str, *, label: str) -> None:
    if (
        not isinstance(name, str)
        or name in {"", ".", ".."}
        or os.path.sep in name
        or (os.path.altsep is not None and os.path.altsep in name)
    ):
        raise AnchorPreparationError(f"unsafe {label} basename")


def _same_directory(source_parent_fd: int, destination_parent_fd: int) -> bool:
    return _inode_identity(os.fstat(source_parent_fd)) == _inode_identity(
        os.fstat(destination_parent_fd)
    )


def _rollback_owned_name(parent_fd: int, name: str, owned: os.stat_result) -> None:
    try:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _inode_identity(named) == _inode_identity(owned):
            os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _link_unlink_noreplace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
    source: os.stat_result,
    *,
    label: str,
) -> None:
    """Publish a regular file with link(2)'s atomic exclusive destination."""

    linked_destination: os.stat_result | None = None
    try:
        os.link(
            source_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        named_destination = os.stat(
            destination_name, dir_fd=parent_fd, follow_symlinks=False
        )
        # Capture an owned destination before reopening the source name so a
        # concurrent source removal cannot strand the link we just created.
        if _inode_identity(named_destination) == _inode_identity(source):
            linked_destination = named_destination
        named_source = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
        # Also claim a link to the current source inode, covering a source-name
        # swap before link(). A destination replaced by an unrelated inode is
        # never claimed and therefore survives rollback.
        if _inode_identity(named_destination) in {
            _inode_identity(named_source),
            _inode_identity(source),
        }:
            linked_destination = named_destination
        if (
            not stat.S_ISREG(named_source.st_mode)
            or _inode_identity(named_source) != _inode_identity(source)
            or _inode_identity(named_destination) != _inode_identity(source)
        ):
            raise AnchorPreparationError(f"{label} source inode binding drift")
        os.unlink(source_name, dir_fd=parent_fd)
    except FileExistsError as exc:
        raise AnchorPreparationError(f"refusing to overwrite {label}") from exc
    except BaseException:
        if linked_destination is not None:
            _rollback_owned_name(parent_fd, destination_name, linked_destination)
        raise


def _plain_directory_rename_is_noreplace(parent_fd: int) -> bool:
    """Probe whether plain same-directory rename refuses an existing directory."""

    token = secrets.token_hex(16)
    source_name = f".anchor-rename-probe-{token}.source"
    destination_name = f".anchor-rename-probe-{token}.destination"
    source_fd: int | None = None
    destination_fd: int | None = None
    owned: list[os.stat_result] = []
    try:
        os.mkdir(source_name, 0o700, dir_fd=parent_fd)
        owned.append(
            os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
        )
        source_fd = os.open(source_name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        if _inode_identity(owned[-1]) != _inode_identity(os.fstat(source_fd)):
            return False
        marker_fd = os.open(
            ".owned-marker",
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=source_fd,
        )
        os.close(marker_fd)
        os.mkdir(destination_name, 0o700, dir_fd=parent_fd)
        owned.append(
            os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
        )
        destination_fd = os.open(
            destination_name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd
        )
        if _inode_identity(owned[-1]) != _inode_identity(os.fstat(destination_fd)):
            return False
        try:
            os.rename(
                source_name,
                destination_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                return False
            named_source = os.stat(
                source_name, dir_fd=parent_fd, follow_symlinks=False
            )
            named_destination = os.stat(
                destination_name, dir_fd=parent_fd, follow_symlinks=False
            )
            return (
                stat.S_ISDIR(named_source.st_mode)
                and stat.S_ISDIR(named_destination.st_mode)
                and _inode_identity(named_source)
                == _inode_identity(os.fstat(source_fd))
                and _inode_identity(named_destination)
                == _inode_identity(os.fstat(destination_fd))
            )
        return False
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
        owned_identities = {_inode_identity(item) for item in owned}
        for name in (source_name, destination_name):
            try:
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    stat.S_ISDIR(named.st_mode)
                    and _inode_identity(named) in owned_identities
                ):
                    shutil.rmtree(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def _locked_directory_rename_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
    source: os.stat_result,
    *,
    label: str,
) -> None:
    """Use plain rename only after the filesystem proves no-replace semantics."""

    digest = hashlib.sha256(os.fsencode(destination_name)).hexdigest()
    lock_name = f".anchor-publish-{digest}.lock"
    lock_fd: int | None = None
    lock_stat: os.stat_result | None = None
    try:
        try:
            lock_fd = os.open(
                lock_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise AnchorPreparationError(
                f"exclusive publication lock already exists for {label}"
            ) from exc
        lock_stat = os.fstat(lock_fd)
        os.fsync(lock_fd)
        named_lock = os.stat(lock_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(named_lock.st_mode)
            or _inode_identity(named_lock) != _inode_identity(lock_stat)
        ):
            raise AnchorPreparationError(f"{label} publication lock binding drift")
        if not _plain_directory_rename_is_noreplace(parent_fd):
            raise AnchorPreparationError(
                f"filesystem lacks safe no-replace directory rename for {label}"
            )
        try:
            os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AnchorPreparationError(f"refusing to overwrite {label}")
        named_source = os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(named_source.st_mode)
            or stat.S_ISLNK(named_source.st_mode)
            or _inode_identity(named_source) != _inode_identity(source)
        ):
            raise AnchorPreparationError(f"{label} source inode binding drift")
        try:
            os.rename(
                source_name,
                destination_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                raise AnchorPreparationError(f"refusing to overwrite {label}") from exc
            if exc.errno in {errno.EISDIR, errno.ENOTDIR}:
                try:
                    os.stat(
                        destination_name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise AnchorPreparationError(
                        f"refusing to overwrite {label}"
                    ) from exc
            raise
        published = os.stat(
            destination_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _inode_identity(published) != _inode_identity(source):
            raise AnchorPreparationError(f"{label} published inode binding drift")
    finally:
        if lock_stat is not None:
            _rollback_owned_name(parent_fd, lock_name, lock_stat)
        if lock_fd is not None:
            os.close(lock_fd)


def _rename_noreplace_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
    *,
    label: str,
) -> None:
    """Rename without replacement, with BeeGFS-safe same-directory fallbacks."""

    _require_publication_basename(source_name, label="publication source")
    _require_publication_basename(destination_name, label="publication destination")
    native_error = _native_rename_noreplace_at(
        source_parent_fd,
        source_name,
        destination_parent_fd,
        destination_name,
    )
    if native_error == 0:
        return
    if native_error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise AnchorPreparationError(f"refusing to overwrite {label}")
    unsupported = {
        None,
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    if native_error not in unsupported:
        raise AnchorPreparationError(
            f"atomic no-replace publication failed for {label}: "
            f"{os.strerror(native_error)}"
        )
    if not _same_directory(source_parent_fd, destination_parent_fd):
        raise AnchorPreparationError(f"same-directory fallback required for {label}")
    source = os.stat(source_name, dir_fd=source_parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(source.st_mode):
        raise AnchorPreparationError(f"refusing symlink publication for {label}")
    if stat.S_ISREG(source.st_mode):
        _link_unlink_noreplace_at(
            source_parent_fd, source_name, destination_name, source, label=label
        )
        return
    if stat.S_ISDIR(source.st_mode):
        _locked_directory_rename_at(
            source_parent_fd, source_name, destination_name, source, label=label
        )
        return
    raise AnchorPreparationError(f"unsupported publication inode type for {label}")


@dataclass(frozen=True)
class _VerifiedArchiveSnapshot:
    path: Path
    input_record: dict[str, Any]


def _snapshot_verified_archive(
    source: Path,
    snapshot_path: Path,
    *,
    expected_name: str,
    expected_size: int,
    expected_sha256: str,
    expected_md5: str | None = None,
) -> _VerifiedArchiveSnapshot:
    _require_sha256(expected_sha256, f"{expected_name} expected SHA-256")
    if source.name != expected_name:
        raise AnchorPreparationError(
            f"expected local input basename {expected_name!r}, got {source.name!r}"
        )
    descriptor, opened = _open_regular_fd(source, label=expected_name)
    if opened.st_size != expected_size:
        os.close(descriptor)
        raise AnchorPreparationError(
            f"{expected_name} size drift: expected {expected_size}, got {opened.st_size}"
        )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        destination_descriptor = os.open(snapshot_path, destination_flags, 0o600)
    except OSError as exc:
        os.close(descriptor)
        raise AnchorPreparationError(
            f"cannot create private archive snapshot: {snapshot_path}"
        ) from exc
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    observed = 0
    try:
        with (
            os.fdopen(descriptor, "rb", closefd=True) as input_handle,
            os.fdopen(destination_descriptor, "wb", closefd=True) as output_handle,
        ):
            for chunk in iter(
                lambda: input_handle.read(VALIDATION_CHUNK_BYTES), b""
            ):
                observed += len(chunk)
                if observed > expected_size:
                    raise AnchorPreparationError(
                        f"{expected_name} grew while it was snapshotted"
                    )
                sha256.update(chunk)
                md5.update(chunk)
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
            after = os.fstat(input_handle.fileno())
    except BaseException:
        try:
            snapshot_path.unlink()
        except OSError:
            pass
        raise
    if observed != expected_size or not _same_file_snapshot(opened, after):
        snapshot_path.unlink(missing_ok=True)
        raise AnchorPreparationError(f"{expected_name} changed while it was snapshotted")
    observed_sha256 = sha256.hexdigest()
    observed_md5 = md5.hexdigest()
    if observed_sha256 != expected_sha256:
        snapshot_path.unlink(missing_ok=True)
        raise AnchorPreparationError(f"{expected_name} SHA-256 mismatch")
    if expected_md5 is not None:
        if not _MD5_RE.fullmatch(expected_md5):
            snapshot_path.unlink(missing_ok=True)
            raise AnchorPreparationError("frozen MD5 contract is malformed")
        if observed_md5 != expected_md5:
            snapshot_path.unlink(missing_ok=True)
            raise AnchorPreparationError(f"{expected_name} MD5 mismatch")
    os.chmod(snapshot_path, 0o400)
    return _VerifiedArchiveSnapshot(
        path=snapshot_path,
        input_record={
            "filename": expected_name,
            "size_bytes": observed,
            "sha256": observed_sha256,
            "md5": observed_md5,
        },
    )


def _require_manual_attestation(
    *, reviewer: str, timestamp: str, decision: str
) -> tuple[str, str, str]:
    if not isinstance(reviewer, str):
        raise AnchorPreparationError("manual receipt reviewer must be a string")
    reviewer = reviewer.strip()
    if not reviewer or not _is_rfc3339_utc(timestamp):
        raise AnchorPreparationError(
            "manual receipt requires a reviewer and RFC3339 UTC timestamp"
        )
    if decision not in {"accepted", "rejected"}:
        raise AnchorPreparationError("manual receipt decision must be accepted/rejected")
    return reviewer, timestamp, decision


def _write_new_sealed_json(
    path: Path, payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    sealed = seal_manifest(dict(payload))
    verify_manifest_hash(sealed)
    encoded = (
        json.dumps(
            sealed,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise AnchorPreparationError(f"{label} exceeds {MAX_MANIFEST_BYTES} bytes")
    parent_fd, parent_path, destination_name = _open_parent_directory(
        path, label=label, create=True
    )
    temporary_name = f".{destination_name}.{secrets.token_hex(16)}.tmp"
    published = False
    descriptor: int | None = None
    owned_stat: os.stat_result | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        owned_stat = os.fstat(descriptor)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AnchorPreparationError(f"cannot write {label}")
            view = view[written:]
        os.fsync(descriptor)
        owned_stat = os.fstat(descriptor)
        named_temporary = os.stat(
            temporary_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _same_file_identity_record(named_temporary) != _same_file_identity_record(
            owned_stat
        ):
            raise AnchorPreparationError(f"{label} temporary inode binding drift")
        _assert_path_binds_directory_fd(
            parent_path, parent_fd, label=f"{label} parent"
        )
        _rename_noreplace_at(
            parent_fd,
            temporary_name,
            parent_fd,
            destination_name,
            label=label,
        )
        published = True
        os.fsync(parent_fd)
        published_descriptor_stat = os.fstat(descriptor)
        named_published = os.stat(
            destination_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _same_file_identity_record(named_published) != _same_file_identity_record(
            published_descriptor_stat
        ):
            raise AnchorPreparationError(f"{label} published inode binding drift")
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = bytearray()
        while len(observed) <= MAX_MANIFEST_BYTES:
            chunk = os.read(
                descriptor,
                min(VALIDATION_CHUNK_BYTES, MAX_MANIFEST_BYTES + 1 - len(observed)),
            )
            if not chunk:
                break
            observed.extend(chunk)
        after = os.fstat(descriptor)
        if (
            bytes(observed) != encoded
            or not _same_file_snapshot(published_descriptor_stat, after)
        ):
            raise AnchorPreparationError(f"{label} publication byte drift")
        final_named = os.stat(
            destination_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _same_file_identity_record(final_named) != _same_file_identity_record(after):
            raise AnchorPreparationError(f"{label} final inode binding drift")
        _assert_path_binds_directory_fd(
            parent_path, parent_fd, label=f"{label} parent"
        )
        return sealed
    except BaseException:
        cleanup_name = destination_name if published else temporary_name
        if owned_stat is not None:
            try:
                cleanup_stat = os.stat(
                    cleanup_name, dir_fd=parent_fd, follow_symlinks=False
                )
                if _inode_identity(cleanup_stat) == _inode_identity(owned_stat):
                    os.unlink(cleanup_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _acquisition_source_contract(
    source_id: str, contract: MotContract | ParlaMintContract
) -> tuple[str, str, list[tuple[str, int, str | None, str]]]:
    if source_id == MOT_SOURCE_ID and isinstance(contract, MotContract):
        _validate_mot_contract_identity(contract)
        return (
            contract.release_tag,
            contract.release_commit,
            [
                (asset.filename, asset.size_bytes, None, asset.source_url)
                for asset in contract.assets
            ],
        )
    if source_id == PARLAMINT_SOURCE_ID and isinstance(contract, ParlaMintContract):
        _validate_parlamint_contract_identity(contract)
        return (
            PARLAMINT_RELEASE,
            PARLAMINT_RELEASE_COMMIT,
            [
                (
                    contract.filename,
                    contract.size_bytes,
                    contract.md5,
                    contract.source_url,
                )
            ],
        )
    raise AnchorPreparationError("acquisition source/contract mismatch")


def _acquisition_contract_projection(
    source_id: str, contract: MotContract | ParlaMintContract
) -> dict[str, Any]:
    release, revision, expected = _acquisition_source_contract(source_id, contract)
    return {
        "source_id": source_id,
        "release": release,
        "resolved_revision": revision,
        "assets": [
            {
                "filename": filename,
                "size_bytes": size_bytes,
                "official_md5": official_md5,
                "source_url": source_url,
            }
            for filename, size_bytes, official_md5, source_url in expected
        ],
    }


def _acquisition_contract_sha256(
    source_id: str, contract: MotContract | ParlaMintContract
) -> str:
    return hashlib.sha256(
        _canonical_line(_acquisition_contract_projection(source_id, contract))
    ).hexdigest()


def seal_anchor_acquisition_receipt(
    source_id: str,
    archive_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    reviewer: str,
    acquired_at_utc: str,
    decision: str = "accepted",
    notes: str = "",
    contract: MotContract | ParlaMintContract,
) -> dict[str, Any]:
    """Seal a manual acquisition decision for release assets lacking SHA-256s."""

    if not isinstance(notes, str):
        raise AnchorPreparationError("acquisition receipt notes must be a string")
    reviewer, acquired_at_utc, decision = _require_manual_attestation(
        reviewer=reviewer, timestamp=acquired_at_utc, decision=decision
    )
    release, resolved_revision, expected = _acquisition_source_contract(
        source_id, contract
    )
    supplied = {Path(path).name: Path(path) for path in archive_paths}
    if len(supplied) != len(archive_paths):
        raise AnchorPreparationError("acquisition archive basenames must be unique")
    if set(supplied) != {name for name, _size, _md5, _url in expected}:
        raise AnchorPreparationError("acquisition receipt archive set is incomplete or extra")
    assets: list[dict[str, Any]] = []
    for filename, size_bytes, official_md5, source_url in expected:
        parsed_url = urlparse(source_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise AnchorPreparationError(
                f"acquisition source URL must be absolute HTTPS: {filename}"
            )
        observed = _hash_regular_file(
            supplied[filename], label=filename, expected_size=size_bytes
        )
        if official_md5 is not None and observed["md5"] != official_md5:
            raise AnchorPreparationError(f"{filename} official MD5 mismatch")
        assets.append(
            {
                **observed,
                "source_url": source_url,
                "official_md5": official_md5,
                "upstream_sha256_published": False,
            }
        )
    return _write_new_sealed_json(
        Path(output_path),
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": ACQUISITION_RECEIPT_KIND,
            "source_id": source_id,
            "release": release,
            "resolved_revision": resolved_revision,
            "asset_contract_sha256": _acquisition_contract_sha256(
                source_id, contract
            ),
            "assets": assets,
            "attestation": {
                "reviewer": reviewer,
                "acquired_at_utc": acquired_at_utc,
                "decision": decision,
                "notes": notes,
                "statement": (
                    "reviewer_attests_assets_were_acquired_from_recorded_release_urls"
                ),
            },
            "canonical_sha256": None,
        },
        label="acquisition receipt",
    )


def validate_anchor_acquisition_receipt(
    receipt: Mapping[str, Any],
    *,
    source_id: str,
    contract: MotContract | ParlaMintContract,
) -> dict[str, dict[str, Any]]:
    verify_manifest_hash(receipt)
    release, revision, expected = _acquisition_source_contract(source_id, contract)
    attestation = receipt.get("attestation")
    if (
        set(receipt)
        != {
            "schema_version",
            "kind",
            "source_id",
            "release",
            "resolved_revision",
            "asset_contract_sha256",
            "assets",
            "attestation",
            "canonical_sha256",
        }
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("kind") != ACQUISITION_RECEIPT_KIND
        or receipt.get("source_id") != source_id
        or receipt.get("release") != release
        or receipt.get("resolved_revision") != revision
        or receipt.get("asset_contract_sha256")
        != _acquisition_contract_sha256(source_id, contract)
        or not isinstance(attestation, dict)
        or set(attestation)
        != {"reviewer", "acquired_at_utc", "decision", "notes", "statement"}
        or attestation.get("decision") != "accepted"
        or not isinstance(attestation.get("reviewer"), str)
        or not attestation["reviewer"].strip()
        or not _is_rfc3339_utc(attestation.get("acquired_at_utc"))
        or not isinstance(attestation.get("notes"), str)
        or attestation.get("statement")
        != "reviewer_attests_assets_were_acquired_from_recorded_release_urls"
    ):
        raise AnchorPreparationError("accepted acquisition receipt contract drift")
    assets = receipt.get("assets")
    if not isinstance(assets, list):
        raise AnchorPreparationError("acquisition receipt assets must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    for raw in assets:
        if (
            not isinstance(raw, dict)
            or set(raw)
            != {
                "filename",
                "size_bytes",
                "sha256",
                "md5",
                "source_url",
                "official_md5",
                "upstream_sha256_published",
            }
            or not isinstance(raw.get("filename"), str)
        ):
            raise AnchorPreparationError("malformed acquisition receipt asset")
        filename = raw["filename"]
        if filename in by_name:
            raise AnchorPreparationError("duplicate acquisition receipt asset")
        parsed_url = urlparse(str(raw.get("source_url", "")))
        if (
            raw.get("upstream_sha256_published") is not False
            or parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or isinstance(raw.get("size_bytes"), bool)
            or not isinstance(raw.get("size_bytes"), int)
            or raw["size_bytes"] <= 0
            or not _SHA256_RE.fullmatch(str(raw.get("sha256", "")))
            or not _MD5_RE.fullmatch(str(raw.get("md5", "")))
        ):
            raise AnchorPreparationError(f"malformed acquisition asset: {filename}")
        by_name[filename] = raw
    if set(by_name) != {name for name, _size, _md5, _url in expected}:
        raise AnchorPreparationError("acquisition receipt asset inventory drift")
    for filename, size_bytes, official_md5, source_url in expected:
        asset = by_name[filename]
        if (
            asset.get("size_bytes") != size_bytes
            or asset.get("official_md5") != official_md5
            or asset.get("source_url") != source_url
        ):
            raise AnchorPreparationError(f"acquisition receipt identity drift: {filename}")
        if official_md5 is not None and asset.get("md5") != official_md5:
            raise AnchorPreparationError(f"acquisition receipt MD5 drift: {filename}")
    return by_name


def _load_acquisition_receipt(
    path: str | Path,
    *,
    source_id: str,
    contract: MotContract | ParlaMintContract,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source = Path(path)
    payload = _read_bounded_regular_file(
        source, label="acquisition receipt", max_bytes=MAX_MANIFEST_BYTES
    )
    receipt = _strict_json_object(payload, "acquisition receipt")
    assets = validate_anchor_acquisition_receipt(
        receipt, source_id=source_id, contract=contract
    )
    return receipt, assets


def _safe_member_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise AnchorPreparationError(f"unsafe tar member name: {name!r}")
    if name.startswith("/"):
        raise AnchorPreparationError(f"absolute tar member path is forbidden: {name!r}")
    trimmed = name[:-1] if name.endswith("/") else name
    if not trimmed:
        raise AnchorPreparationError("empty tar member path is forbidden")
    path = PurePosixPath(trimmed)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AnchorPreparationError(f"unsafe tar member path: {name!r}")
    if path.as_posix() != trimmed:
        raise AnchorPreparationError(f"non-canonical tar member path: {name!r}")
    return path


def _member_kind(member: tarfile.TarInfo) -> str:
    if member.isdir():
        return "directory"
    if member.isfile():
        return "file"
    raise AnchorPreparationError(
        f"tar links/devices/special members are forbidden: {member.name!r}"
    )


class _CompressedArchiveAuditReader(io.RawIOBase):
    """Hash the exact compressed bytes consumed by the parser descriptor."""

    def __init__(self, handle: BinaryIO, *, expected: Mapping[str, Any]) -> None:
        super().__init__()
        self.handle = handle
        self.expected = expected
        self.observed_size = 0
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5(usedforsecurity=False)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self.handle.read(len(buffer))
        size = len(data)
        if size:
            buffer[:size] = data
            self.observed_size += size
            if self.observed_size > self.expected["size_bytes"]:
                raise AnchorPreparationError("archive snapshot grew while it was parsed")
            self.sha256.update(data)
            self.md5.update(data)
        return size

    def finish(self) -> None:
        while self.read(VALIDATION_CHUNK_BYTES):
            pass
        if (
            self.observed_size != self.expected["size_bytes"]
            or self.sha256.hexdigest() != self.expected["sha256"]
            or self.md5.hexdigest() != self.expected["md5"]
        ):
            raise AnchorPreparationError(
                "archive bytes parsed do not match the verified private snapshot"
            )


class _BoundedTarStreamReader(io.RawIOBase):
    """Bound every decompressed tar byte, including hidden control headers."""

    def __init__(self, handle: BinaryIO, *, max_bytes: int) -> None:
        super().__init__()
        self.handle = handle
        self.max_bytes = _require_positive_int(max_bytes, "max_tar_stream_bytes")
        self.observed_size = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        remaining_with_probe = self.max_bytes - self.observed_size + 1
        if remaining_with_probe <= 0:
            raise AnchorPreparationError("decompressed tar stream exceeds its byte bound")
        view = memoryview(buffer)[: min(len(buffer), remaining_with_probe)]
        data = self.handle.read(len(view))
        size = len(data)
        if size:
            view[:size] = data
            self.observed_size += size
            if self.observed_size > self.max_bytes:
                raise AnchorPreparationError(
                    "decompressed tar stream exceeds its byte bound"
                )
        return size

    def finish(self, buffered_tail: bytes) -> int:
        tail_bytes = len(buffered_tail)
        if any(buffered_tail):
            raise AnchorPreparationError("tar archive contains non-zero trailing payload")
        while True:
            chunk = self.read(VALIDATION_CHUNK_BYTES)
            if not chunk:
                break
            tail_bytes += len(chunk)
            if any(chunk):
                raise AnchorPreparationError("tar archive contains non-zero trailing payload")
        if tail_bytes < tarfile.BLOCKSIZE:
            raise AnchorPreparationError("tar archive lacks the second zero end block")
        return self.observed_size


class _BoundedTarInfo(tarfile.TarInfo):
    """Cap pseudo-members before stdlib tarfile buffers their full payload."""

    def _guard_control(self, archive: tarfile.TarFile, kind: str) -> None:
        maximum = int(getattr(archive, "_anchor_max_control_header_bytes"))
        count = int(getattr(archive, "_anchor_control_headers", 0)) + 1
        max_count = int(getattr(archive, "_anchor_max_control_headers"))
        if self.size < 0 or self.size > maximum:
            raise AnchorPreparationError(
                f"tar {kind} control header exceeds {maximum} bytes"
            )
        if count > max_count:
            raise AnchorPreparationError("tar control-header count exceeds its bound")
        setattr(archive, "_anchor_control_headers", count)

    def _proc_pax(self, archive: tarfile.TarFile) -> tarfile.TarInfo:
        self._guard_control(archive, "PAX")
        return super()._proc_pax(archive)

    def _proc_gnulong(self, archive: tarfile.TarFile) -> tarfile.TarInfo:
        self._guard_control(archive, "GNU long-name")
        return super()._proc_gnulong(archive)

    def _proc_sparse(self, archive: tarfile.TarFile) -> tarfile.TarInfo:
        raise AnchorPreparationError("GNU sparse tar members are forbidden")

    def _proc_gnusparse_00(self, *args: Any, **kwargs: Any) -> None:
        raise AnchorPreparationError("GNU sparse PAX members are forbidden")

    def _proc_gnusparse_01(self, *args: Any, **kwargs: Any) -> None:
        raise AnchorPreparationError("GNU sparse PAX members are forbidden")

    def _proc_gnusparse_10(self, *args: Any, **kwargs: Any) -> None:
        raise AnchorPreparationError("GNU sparse PAX members are forbidden")


class _BoundedTarFile(tarfile.TarFile):
    """Install control-header limits before ``TarFile`` parses its first member."""

    def __init__(
        self,
        *args: Any,
        anchor_max_control_header_bytes: int,
        anchor_max_control_headers: int,
        **kwargs: Any,
    ) -> None:
        self._anchor_max_control_header_bytes = _require_positive_int(
            anchor_max_control_header_bytes, "max_control_header_bytes"
        )
        self._anchor_max_control_headers = _require_positive_int(
            anchor_max_control_headers, "max_control_headers"
        )
        self._anchor_control_headers = 0
        super().__init__(*args, **kwargs)


@contextmanager
def _open_verified_tar(
    snapshot: _VerifiedArchiveSnapshot,
    *,
    max_tar_stream_bytes: int,
    max_control_header_bytes: int,
    max_control_headers: int,
) -> Iterator[tuple[tarfile.TarFile, dict[str, int]]]:
    descriptor, opened = _open_regular_fd(snapshot.path, label="private archive snapshot")
    if opened.st_size != snapshot.input_record["size_bytes"]:
        os.close(descriptor)
        raise AnchorPreparationError("private archive snapshot size drift")
    state: dict[str, int] = {}
    with os.fdopen(descriptor, "rb", buffering=0, closefd=True) as raw_handle:
        audit = _CompressedArchiveAuditReader(
            raw_handle, expected=snapshot.input_record
        )
        decompressor = gzip.GzipFile(fileobj=audit, mode="rb")
        bounded = _BoundedTarStreamReader(
            decompressor, max_bytes=max_tar_stream_bytes
        )
        try:
            archive = _BoundedTarFile.open(
                fileobj=bounded,
                mode="r|",
                bufsize=64 * 1024,
                tarinfo=_BoundedTarInfo,
                anchor_max_control_header_bytes=max_control_header_bytes,
                anchor_max_control_headers=max_control_headers,
            )
        except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
            decompressor.close()
            raise AnchorPreparationError(
                f"cannot open verified archive snapshot {snapshot.path.name}"
            ) from exc
        try:
            yield archive, state
        except AnchorPreparationError:
            archive.close()
            decompressor.close()
            raise
        except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
            archive.close()
            decompressor.close()
            raise AnchorPreparationError(
                f"cannot parse verified archive snapshot {snapshot.path.name}"
            ) from exc
        except BaseException:
            archive.close()
            decompressor.close()
            raise
        else:
            stream = archive.fileobj
            buffered_tail = bytes(getattr(stream, "buf", b""))
            archive.close()
            try:
                state["tar_stream_bytes"] = bounded.finish(buffered_tail)
                state["tar_control_headers"] = int(
                    getattr(archive, "_anchor_control_headers", 0)
                )
                decompressor.close()
                audit.finish()
            except (OSError, EOFError, gzip.BadGzipFile, tarfile.TarError) as exc:
                raise AnchorPreparationError(
                    f"cannot finish verified archive snapshot {snapshot.path.name}"
                ) from exc


def _read_member_bytes(
    archive: tarfile.TarFile, member: tarfile.TarInfo, *, limit: int
) -> bytes:
    if member.size > limit:
        raise AnchorPreparationError(
            f"tar member exceeds {limit} bytes: {member.name!r} ({member.size})"
        )
    handle = archive.extractfile(member)
    if handle is None:
        raise AnchorPreparationError(f"cannot read tar member: {member.name!r}")
    payload = handle.read(limit + 1)
    if len(payload) != member.size or len(payload) > limit:
        raise AnchorPreparationError(f"tar member size/read drift: {member.name!r}")
    return payload


def _hash_member_stream(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> str:
    handle = archive.extractfile(member)
    if handle is None:
        raise AnchorPreparationError(f"cannot read tar member: {member.name!r}")
    digest = hashlib.sha256()
    observed = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        observed += len(chunk)
        digest.update(chunk)
    if observed != member.size:
        raise AnchorPreparationError(f"tar member size/read drift: {member.name!r}")
    return digest.hexdigest()


def _strict_json_object(payload: bytes, location: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AnchorPreparationError(f"{location}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise AnchorPreparationError(f"{location}: non-finite JSON number {token!r}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except AnchorPreparationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AnchorPreparationError(f"{location}: invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise AnchorPreparationError(f"{location}: JSON root must be an object")
    return value


def _normalize_inline(text: str) -> str:
    if not isinstance(text, str):
        raise AnchorPreparationError("text value must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _normalize_paragraphs(paragraphs: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for paragraph in paragraphs:
        value = _normalize_inline(paragraph)
        if value:
            normalized.append(value)
    return normalized


def _is_substantive_title(title: str) -> bool:
    return len(title) >= 4 and sum(character.isalpha() for character in title) >= 2


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class _BuildTarget:
    destination: Path
    parent_path: Path
    parent_fd: int
    destination_name: str
    build_name: str
    build_fd: int
    build_path: Path
    published: bool = False
    closed: bool = False


def _prepare_destination(output_dir: str | Path) -> _BuildTarget:
    parent_fd, parent_path, destination_name = _open_parent_directory(
        output_dir, label="output path", create=True
    )
    build_name: str | None = None
    build_fd: int | None = None
    try:
        try:
            os.stat(destination_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AnchorPreparationError(
                f"refusing to overwrite output path: {Path(output_dir)}"
            )
        for _attempt in range(128):
            candidate_name = f".{destination_name}.build-{secrets.token_hex(12)}"
            try:
                os.mkdir(candidate_name, 0o700, dir_fd=parent_fd)
                build_name = candidate_name
                break
            except FileExistsError:
                continue
        else:
            raise AnchorPreparationError("cannot allocate a private build directory")
        build_fd = os.open(build_name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        named_build = os.stat(
            build_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _inode_identity(named_build) != _inode_identity(os.fstat(build_fd)):
            raise AnchorPreparationError("new private build inode binding drift")
        os.fsync(parent_fd)
        _assert_path_binds_directory_fd(
            parent_path, parent_fd, label="output parent"
        )
        return _BuildTarget(
            destination=parent_path / destination_name,
            parent_path=parent_path,
            parent_fd=parent_fd,
            destination_name=destination_name,
            build_name=build_name,
            build_fd=build_fd,
            # Producers that require pathname APIs write here, while every
            # validation/publication decision is bound to ``build_fd``.
            build_path=parent_path / build_name,
        )
    except BaseException:
        if build_name is not None and build_fd is not None:
            try:
                named_build = os.stat(
                    build_name, dir_fd=parent_fd, follow_symlinks=False
                )
                if _inode_identity(named_build) == _inode_identity(os.fstat(build_fd)):
                    shutil.rmtree(build_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        if build_fd is not None:
            os.close(build_fd)
        os.close(parent_fd)
        raise


def _fsync_tree_fd(directory_fd: int) -> None:
    scan_fd = os.dup(directory_fd)
    try:
        with os.scandir(scan_fd) as entries:
            names = sorted(entry.name for entry in entries)
    finally:
        os.close(scan_fd)
    for name in names:
        entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISREG(entry_stat.st_mode):
            file_fd, opened = _open_regular_at(
                directory_fd, name, label=f"durability file {name}"
            )
            try:
                if _inode_identity(entry_stat) != _inode_identity(opened):
                    raise AnchorPreparationError("durability file identity drift")
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
        elif stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode):
            child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
            try:
                if _inode_identity(entry_stat) != _inode_identity(os.fstat(child_fd)):
                    raise AnchorPreparationError("durability directory identity drift")
                _fsync_tree_fd(child_fd)
            finally:
                os.close(child_fd)
        else:
            raise AnchorPreparationError(
                f"cannot durable-publish special output entry: {name}"
            )
    os.fsync(directory_fd)


def _cleanup_build(target: _BuildTarget) -> None:
    if target.closed:
        return
    try:
        if not target.published:
            try:
                named = os.stat(
                    target.build_name,
                    dir_fd=target.parent_fd,
                    follow_symlinks=False,
                )
                if _inode_identity(named) == _inode_identity(os.fstat(target.build_fd)):
                    shutil.rmtree(target.build_name, dir_fd=target.parent_fd)
                    os.fsync(target.parent_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(target.build_fd)
        os.close(target.parent_fd)
        target.closed = True


def _publish_build(
    target: _BuildTarget, *, contract: MotContract | ParlaMintContract
) -> dict[str, Any]:
    """Durably validate and atomically publish the held build inode."""

    _assert_path_binds_directory_fd(
        target.parent_path, target.parent_fd, label="output parent"
    )
    named = os.stat(
        target.build_name, dir_fd=target.parent_fd, follow_symlinks=False
    )
    if (
        not stat.S_ISDIR(named.st_mode)
        or _inode_identity(named) != _inode_identity(os.fstat(target.build_fd))
    ):
        raise AnchorPreparationError("private build path/inode binding drift")
    _fsync_tree_fd(target.build_fd)
    manifest = _validate_anchor_preparation_fd(
        target.build_fd,
        display_path=target.destination,
        verify_files=True,
        contract=contract,
    )
    _rename_noreplace_at(
        target.parent_fd,
        target.build_name,
        target.parent_fd,
        target.destination_name,
        label=f"output path: {target.destination}",
    )
    target.published = True
    try:
        os.fsync(target.parent_fd)
        published = os.stat(
            target.destination_name,
            dir_fd=target.parent_fd,
            follow_symlinks=False,
        )
        if _inode_identity(published) != _inode_identity(os.fstat(target.build_fd)):
            raise AnchorPreparationError("published output inode binding drift")
        _assert_path_binds_directory_fd(
            target.parent_path, target.parent_fd, label="output parent"
        )
        post_manifest = _validate_anchor_preparation_fd(
            target.build_fd,
            display_path=target.destination,
            verify_files=True,
            contract=contract,
        )
        if post_manifest != manifest:
            raise AnchorPreparationError("published manifest changed across publication")
        os.fsync(target.build_fd)
        os.fsync(target.parent_fd)
        return manifest
    except BaseException:
        recovery_name = f".{target.destination_name}.failed-{secrets.token_hex(12)}"
        try:
            failed_named = os.stat(
                target.destination_name,
                dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
            if _inode_identity(failed_named) == _inode_identity(
                os.fstat(target.build_fd)
            ):
                _rename_noreplace_at(
                    target.parent_fd,
                    target.destination_name,
                    target.parent_fd,
                    recovery_name,
                    label="failed output recovery",
                )
                target.destination_name = recovery_name
                target.build_name = recovery_name
                target.published = False
                os.fsync(target.parent_fd)
        except Exception:
            pass
        raise


class _JsonlZstdShardWriter:
    """Canonical JSONL writer with deterministic uncompressed-byte boundaries."""

    def __init__(self, root: Path, relative_dir: str, target_bytes: int) -> None:
        self.root = root
        self.relative_dir = PurePosixPath(relative_dir).as_posix()
        self.target_bytes = _require_positive_int(target_bytes, "target_bytes")
        self.directory = root / self.relative_dir
        self.directory.mkdir(parents=True, exist_ok=False)
        self._stream: pa.NativeFile | None = None
        self._path: Path | None = None
        self._rows = 0
        self._bytes = 0
        self._row_start = 0
        self._shards: list[dict[str, Any]] = []
        self._logical_sha256 = hashlib.sha256()

    def _open(self) -> None:
        if self._stream is not None:
            return
        self._path = self.directory / f"part-{len(self._shards):05d}.jsonl.zst"
        self._stream = pa.output_stream(str(self._path), compression="zstd")

    def write(self, record: Mapping[str, Any]) -> None:
        encoded = _canonical_line(record)
        would_exceed_target = self._bytes + len(encoded) > self.target_bytes
        if self._stream is not None and self._rows and would_exceed_target:
            self._close_shard()
        self._open()
        assert self._stream is not None
        self._stream.write(encoded)
        self._logical_sha256.update(encoded)
        self._rows += 1
        self._bytes += len(encoded)

    def _close_shard(self) -> None:
        if self._stream is None or self._path is None:
            return
        self._stream.close()
        relative = self._path.relative_to(self.root).as_posix()
        self._shards.append(
            {
                "path": relative,
                "rows": self._rows,
                "row_start": self._row_start,
                "row_end_exclusive": self._row_start + self._rows,
                "uncompressed_bytes": self._bytes,
                "size_bytes": self._path.stat().st_size,
                "sha256": file_sha256(self._path),
            }
        )
        self._row_start += self._rows
        self._stream = None
        self._path = None
        self._rows = 0
        self._bytes = 0

    def finish(self, *, emit_empty: bool = False) -> dict[str, Any]:
        if emit_empty and self._stream is None and not self._shards:
            self._open()
        self._close_shard()
        return {
            "format": "jsonl.zst",
            "compression": "zstd",
            "canonicalization": "canonical_json_sorted_utf8_one_lf_v1",
            "target_uncompressed_bytes": self.target_bytes,
            "pyarrow_version": pa.__version__,
            "logical_jsonl_sha256": self._logical_sha256.hexdigest(),
            "shards": self._shards,
            "totals": {
                "shards": len(self._shards),
                "rows": sum(item["rows"] for item in self._shards),
                "uncompressed_bytes": sum(
                    item["uncompressed_bytes"] for item in self._shards
                ),
                "size_bytes": sum(item["size_bytes"] for item in self._shards),
            },
        }


def _open_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _inventory_record(
    *,
    archive: str,
    member: tarfile.TarInfo,
    kind: str,
    disposition: str,
    sha256: str | None,
) -> dict[str, Any]:
    return {
        "archive": archive,
        "path": member.name[:-1] if member.name.endswith("/") else member.name,
        "member_type": kind,
        "disposition": disposition,
        "size_bytes": member.size,
        "sha256": sha256,
        "mode": member.mode,
        "mtime": member.mtime,
    }


def _write_inventory(
    connection: sqlite3.Connection,
    build: Path,
    *,
    target_bytes: int,
) -> tuple[dict[str, Any], str]:
    writer = _JsonlZstdShardWriter(build, "evidence/input_inventory", target_bytes)
    digest = hashlib.sha256()
    for (payload,) in connection.execute(
        "SELECT payload FROM inventory ORDER BY archive COLLATE BINARY, path COLLATE BINARY"
    ):
        record = json.loads(payload)
        encoded = _canonical_line(record)
        digest.update(encoded)
        writer.write(record)
    return writer.finish(emit_empty=True), digest.hexdigest()


def _insert_inventory(
    connection: sqlite3.Connection, record: Mapping[str, Any]
) -> None:
    try:
        connection.execute(
            "INSERT INTO inventory(archive, path, payload) VALUES (?, ?, ?)",
            (record["archive"], record["path"], canonical_json(dict(record))),
        )
    except sqlite3.IntegrityError as exc:
        raise AnchorPreparationError(
            f"duplicate tar member: {record['archive']}:{record['path']}"
        ) from exc


def _insert_quarantine(
    connection: sqlite3.Connection,
    *,
    anchor: str,
    source: str,
    source_id: str,
    member: str,
    reason: str,
    evidence: Mapping[str, Any],
) -> None:
    record = {
        "anchor": anchor,
        "source": source,
        "source_id": source_id,
        "member": member,
        "reason": reason,
        "evidence": dict(evidence),
    }
    connection.execute(
        "INSERT INTO quarantine(anchor, source, source_id, member, reason, payload) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (anchor, source, source_id, member, reason, canonical_json(record)),
    )


def _write_quarantine(
    connection: sqlite3.Connection,
    build: Path,
    *,
    target_bytes: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    writer = _JsonlZstdShardWriter(build, "evidence/quarantine", target_bytes)
    reasons: Counter[str] = Counter()
    for reason, payload in connection.execute(
        "SELECT reason, payload FROM quarantine "
        "ORDER BY anchor COLLATE BINARY, source COLLATE BINARY, "
        "source_id COLLATE BINARY, member COLLATE BINARY, reason COLLATE BINARY"
    ):
        reasons[reason] += 1
        writer.write(json.loads(payload))
    return writer.finish(emit_empty=True), dict(sorted(reasons.items()))


def _base_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE inventory (
            archive TEXT NOT NULL,
            path TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (archive, path)
        ) WITHOUT ROWID;
        CREATE TABLE quarantine (
            anchor TEXT NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            member TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )


def _mot_source_manifest(contract: MotContract) -> dict[str, Any]:
    return {
        "name": "Multilingual Open Text Turkish news",
        "release": contract.release_tag,
        "resolved_revision": contract.release_commit,
        "release_url": MOT_RELEASE_URL,
        "repository_url": MOT_REPOSITORY_URL,
        "asset_urls": [asset.source_url for asset in contract.assets],
        "license": MOT_LICENSE,
        "license_url": MOT_LICENSE_URL,
        "citation_url": MOT_CITATION_URL,
        "attribution": (
            "Chester Palen-Michel, June Kim, Ryan Partlan, and "
            "Constantine Lignos; Voice of America source material"
        ),
    }


def _mot_archive_policy(contract: MotContract) -> dict[str, Any]:
    return {
        "mode": "streaming_allowlist_fail_closed_v1",
        "permitted_directories": [
            "tur_amerikaninsesi/{article,audio,photo,video}",
            "tur_voaturkce/{article,audio,photo,video}",
        ],
        "processed_files": "tur_<approved-site>/article/*.json",
        "inventoried_ignored_files": (
            "tur_<approved-site>/{audio,photo,video}/*.json"
        ),
        "native_structured_text_only": True,
        "pdf_ocr_fallback": "forbidden",
        "forbidden_member_types": [
            "symlink",
            "hardlink",
            "device",
            "fifo",
            "pdf",
            "ocr_derived_text",
            "unexpected_path",
        ],
        "max_members_per_archive": contract.max_members,
        "max_member_bytes": contract.max_member_bytes,
        "max_uncompressed_bytes_per_archive": (
            contract.max_uncompressed_bytes_per_archive
        ),
        "max_total_tar_stream_bytes_per_archive": (
            contract.max_tar_stream_bytes_per_archive
        ),
        "max_tar_control_header_bytes": contract.max_control_header_bytes,
    }


def _mot_cleaning_manifest() -> dict[str, Any]:
    return {
        "input_mode": "native_MOT_JSON_fields_only",
        "pdf_ocr_fallback": False,
        "text_fields": "nonduplicate_substantive_title_then_paragraphs",
        "unicode": "NFC",
        "whitespace": "unicode_whitespace_collapse_preserve_paragraph_breaks",
        "rewriting": False,
        "agency_policy": "reject_explicit_AP_AFP_Reuters_provenance_v1",
        "overlap_key": "trailing_numeric_VOA_article_id",
        "overlap_selection": (
            "identical_prefer_voaturkce_else_newest_modified_then_retrieved_v1"
        ),
        "boilerplate": "report_repeated_paragraphs_without_deletion",
    }


def _parlamint_source_manifest(contract: ParlaMintContract) -> dict[str, Any]:
    return {
        "name": "ParlaMint-TR",
        "release": PARLAMINT_RELEASE,
        "resolved_revision": PARLAMINT_RELEASE_COMMIT,
        "release_tag": PARLAMINT_RELEASE_TAG,
        "persistent_handle": PARLAMINT_HANDLE,
        "bitstream_url": contract.source_url,
        "repository_url": PARLAMINT_REPOSITORY_URL,
        "license": PARLAMINT_LICENSE,
        "license_url": PARLAMINT_LICENSE_URL,
        "attribution": (
            "ParlaMint 5.0 authors (Tomaž Erjavec et al.) and the "
            "ParlaMint-TR contributor Çağrı Çöltekin"
        ),
    }


def _parlamint_archive_policy(contract: ParlaMintContract) -> dict[str, Any]:
    return {
        "mode": "streaming_allowlist_fail_closed_v1",
        "permitted_roots": [
            "README-TR.md",
            "ParlaMint-TR.TEI/",
            "ParlaMint-TR.txt/",
        ],
        "processed_files": [
            "ParlaMint-TR.txt/<year>/ParlaMint-TR_<date>-*.txt",
            "ParlaMint-TR.txt/<year>/ParlaMint-TR_<date>-*-meta.tsv",
        ],
        "validated_native_identity_files": [
            "ParlaMint-TR.TEI/<year>/ParlaMint-TR_<date>-*.xml",
            "ParlaMint-TR.TEI/ParlaMint-TR.xml XIncludes",
        ],
        "inventoried_ignored_files": [
            "ParlaMint-TR.txt/<year>/ParlaMint-TR_<date>-*-meta-en.tsv",
            "README-TR.md",
            "ParlaMint-TR.TEI/00README.txt",
            "ParlaMint-TR.txt/00README.txt",
            "ParlaMint-TR.TEI/Schema/<frozen release schema support allowlist>",
        ],
        "tei_validation": (
            "bounded_stream_parse_all_XML; exact_session_native_text_native_meta_"
            "aggregate_XInclude_identity; require_session_TEI_and_aggregate_teiCorpus"
        ),
        "native_structured_text_only": True,
        "pdf_ocr_fallback": "forbidden",
        "forbidden_member_types": [
            "symlink",
            "hardlink",
            "device",
            "fifo",
            "pdf",
            "ocr_derived_text",
            "unexpected_path",
        ],
        "max_members": contract.max_members,
        "max_member_bytes": contract.max_member_bytes,
        "max_uncompressed_bytes": contract.max_uncompressed_bytes,
        "max_total_tar_stream_bytes": contract.max_tar_stream_bytes,
        "max_tar_control_header_bytes": contract.max_control_header_bytes,
        "max_tei_prolog_bytes": MAX_TEI_PROLOG_BYTES,
        "max_tei_depth": MAX_TEI_DEPTH,
        "max_tei_attributes_per_element": MAX_TEI_ATTRIBUTES_PER_ELEMENT,
        "max_tei_attribute_bytes_per_element": MAX_TEI_ATTRIBUTE_BYTES,
    }


def _parlamint_expected_totals(contract: ParlaMintContract) -> dict[str, Any]:
    return {
        "speeches_exact": contract.expected_speeches,
        "tei_declared_words_exact": contract.expected_declared_words,
        "tei_words_human_readable_millions": "49.26",
        "raw_whitespace_word_test_min": contract.raw_word_count_min,
        "raw_whitespace_word_test_max": contract.raw_word_count_max,
        "first_date_exact": contract.first_date,
        "last_date_exact": contract.last_date,
    }


def _parlamint_cleaning_manifest() -> dict[str, Any]:
    return {
        "input_mode": "native_ParlaMint_plain_text_plus_native_metadata_only",
        "pdf_ocr_fallback": False,
        "cross_format_identity": (
            "exact_TEI_session_equals_aggregate_XInclude_equals_native_txt_equals_"
            "native_meta"
        ),
        "native_text_metadata_join": "exact_one_to_one_by_speech_ID",
        "required_native_language": "Türkçe",
        "stable_id": "parlamint-tr:v5.0:<speech_ID>",
        "transcriber_comments": "strip_only_balanced_double_square_brackets",
        "unbalanced_comments": "quarantine",
        "unicode": "NFC",
        "whitespace": "unicode_whitespace_collapse",
        "rewriting": False,
        "metadata_injection": False,
        "speaker_party_session_metadata_in_text": False,
        "boilerplate": "report_exact_repeated_speech_candidates_without_deletion",
    }


def _frozen_contract_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    acquisition = manifest.get("acquisition_receipt")
    if not isinstance(acquisition, Mapping):
        raise AnchorPreparationError("manifest lacks frozen acquisition contract")
    return {
        "preparer_version": manifest.get("preparer_version"),
        "source_id": manifest.get("source_id"),
        "asset_contract_sha256": acquisition.get("asset_contract_sha256"),
        "source": manifest.get("source"),
        "archive_member_policy": manifest.get("archive_member_policy"),
        "expected_release_totals": manifest.get("expected_release_totals"),
        "cleaning": manifest.get("cleaning"),
        "downstream_admission": manifest.get("downstream_admission"),
    }


def _frozen_contract_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_line(_frozen_contract_projection(manifest))
    ).hexdigest()


def _acceptance_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    acquisition = manifest.get("acquisition_receipt")
    if not isinstance(artifacts, Mapping) or not isinstance(acquisition, Mapping):
        raise AnchorPreparationError("manifest lacks acceptance projection inputs")
    projected_artifacts: dict[str, Any] = {}
    for key in sorted(artifacts):
        artifact = artifacts[key]
        if not isinstance(artifact, Mapping):
            raise AnchorPreparationError("manifest artifact is malformed")
        projected_artifacts[key] = {
            "format": artifact.get("format"),
            "logical_jsonl_sha256": artifact.get("logical_jsonl_sha256"),
            "totals": artifact.get("totals"),
        }
    return {
        "preparer_version": manifest.get("preparer_version"),
        "source_id": manifest.get("source_id"),
        "frozen_contract_sha256": manifest.get("frozen_contract_sha256"),
        "frozen_contract": _frozen_contract_projection(manifest),
        "acquisition_receipt_sha256": acquisition.get("canonical_sha256"),
        "inputs": manifest.get("inputs"),
        "artifacts": projected_artifacts,
        "counts": manifest.get("counts"),
        "raw": manifest.get("raw"),
        "clean": manifest.get("clean"),
        "downstream_admission": manifest.get("downstream_admission"),
    }


def _projection_sha256(projection: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_line(projection)).hexdigest()


def validate_anchor_count_acceptance(
    receipt: Mapping[str, Any], *, projection: Mapping[str, Any]
) -> None:
    verify_manifest_hash(receipt)
    attestation = receipt.get("attestation")
    if (
        set(receipt)
        != {
            "schema_version",
            "kind",
            "preparer_version",
            "source_id",
            "discovery_manifest_sha256",
            "acquisition_receipt_sha256",
            "projection_sha256",
            "approved_projection",
            "attestation",
            "canonical_sha256",
        }
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("kind") != COUNT_ACCEPTANCE_KIND
        or receipt.get("preparer_version") != PREPARER_VERSION
        or receipt.get("source_id") != projection.get("source_id")
        or receipt.get("acquisition_receipt_sha256")
        != projection.get("acquisition_receipt_sha256")
        or receipt.get("projection_sha256") != _projection_sha256(projection)
        or receipt.get("approved_projection") != projection
        or not _SHA256_RE.fullmatch(
            str(receipt.get("discovery_manifest_sha256", ""))
        )
        or not isinstance(attestation, Mapping)
        or set(attestation)
        != {"reviewer", "reviewed_at_utc", "decision", "notes", "statement"}
        or attestation.get("decision") != "accepted"
        or not isinstance(attestation.get("reviewer"), str)
        or not str(attestation["reviewer"]).strip()
        or not _is_rfc3339_utc(attestation.get("reviewed_at_utc"))
        or not isinstance(attestation.get("notes"), str)
        or attestation.get("statement")
        != "reviewer_accepts_exact_discovery_counts_and_logical_hashes"
    ):
        raise AnchorPreparationError("accepted anchor count receipt contract drift")


def _load_count_acceptance(
    path: str | Path, *, projection: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _read_bounded_regular_file(
        Path(path), label="anchor count acceptance", max_bytes=MAX_MANIFEST_BYTES
    )
    receipt = _strict_json_object(payload, "anchor count acceptance")
    validate_anchor_count_acceptance(receipt, projection=projection)
    return receipt


def _finalize_manifest(
    build: Path,
    manifest: Mapping[str, Any],
    *,
    discovery: bool,
    count_acceptance_path: str | Path | None,
) -> dict[str, Any]:
    candidate = dict(manifest)
    candidate["frozen_contract_sha256"] = _frozen_contract_sha256(candidate)
    projection = _acceptance_projection(candidate)
    if discovery:
        candidate["production_acceptance"] = {
            "stage": "discovery_unaccepted",
            "projection_sha256": _projection_sha256(projection),
            "eligible_for_production": False,
        }
    else:
        if count_acceptance_path is None:
            raise AnchorPreparationError("production requires an anchor count acceptance")
        receipt = _load_count_acceptance(
            count_acceptance_path, projection=projection
        )
        candidate["production_acceptance"] = {
            "stage": "accepted_production",
            "projection_sha256": _projection_sha256(projection),
            "eligible_for_production": True,
            "receipt": receipt,
        }
    sealed = seal_manifest(candidate)
    write_json_atomic(build / "manifest.json", sealed)
    verify_manifest_hash(sealed)
    return sealed


def seal_anchor_count_acceptance(
    discovery_output_dir: str | Path,
    output_path: str | Path,
    *,
    reviewer: str,
    reviewed_at_utc: str,
    decision: str,
    notes: str = "",
    contract: MotContract | ParlaMintContract | None = None,
) -> dict[str, Any]:
    if not isinstance(notes, str):
        raise AnchorPreparationError("count acceptance notes must be a string")
    reviewer, reviewed_at_utc, decision = _require_manual_attestation(
        reviewer=reviewer, timestamp=reviewed_at_utc, decision=decision
    )
    discovery_manifest = validate_anchor_preparation(
        discovery_output_dir, contract=contract
    )
    acceptance = discovery_manifest.get("production_acceptance")
    if not isinstance(acceptance, Mapping) or acceptance.get("stage") != "discovery_unaccepted":
        raise AnchorPreparationError("count acceptance requires a discovery preparation")
    projection = _acceptance_projection(discovery_manifest)
    if acceptance.get("projection_sha256") != _projection_sha256(projection):
        raise AnchorPreparationError("discovery acceptance projection drift")
    acquisition = discovery_manifest.get("acquisition_receipt")
    assert isinstance(acquisition, Mapping)
    return _write_new_sealed_json(
        Path(output_path),
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": COUNT_ACCEPTANCE_KIND,
            "preparer_version": PREPARER_VERSION,
            "source_id": discovery_manifest["source_id"],
            "discovery_manifest_sha256": discovery_manifest["canonical_sha256"],
            "acquisition_receipt_sha256": acquisition["canonical_sha256"],
            "projection_sha256": _projection_sha256(projection),
            "approved_projection": projection,
            "attestation": {
                "reviewer": reviewer,
                "reviewed_at_utc": reviewed_at_utc,
                "decision": decision,
                "notes": notes,
                "statement": "reviewer_accepts_exact_discovery_counts_and_logical_hashes",
            },
            "canonical_sha256": None,
        },
        label="anchor count acceptance",
    )


def _artifact_prefix(key: str) -> PurePosixPath:
    prefixes = {
        "data": "data",
        "input_inventory": "evidence/input_inventory",
        "quarantine": "evidence/quarantine",
        "overlap_audit": "evidence/overlap_audit",
        "repeated_paragraph_candidates": "evidence/repeated_paragraph_candidates",
    }
    try:
        return PurePosixPath(prefixes[key])
    except KeyError as exc:
        raise AnchorPreparationError(f"unknown anchor artifact {key!r}") from exc


def _validate_canonical_record(
    raw_line: bytes,
    *,
    location: str,
    artifact_key: str,
    source_id: str,
    state: dict[str, Any],
) -> None:
    record = _strict_json_object(raw_line, location)
    if _canonical_line(record) != raw_line:
        raise AnchorPreparationError(f"{location}: JSONL record is not canonical")
    if artifact_key == "data":
        if set(record) != {"id", "text", "source_id", "source_revision", "provenance"}:
            raise AnchorPreparationError(f"{location}: data schema drift")
        if (
            not isinstance(record["id"], str)
            or not record["id"]
            or not isinstance(record["text"], str)
            or not record["text"]
            or record["source_id"] != source_id
            or record["source_revision"]
            != (MOT_RELEASE_TAG if source_id == MOT_SOURCE_ID else PARLAMINT_RELEASE)
            or not isinstance(record["provenance"], dict)
        ):
            raise AnchorPreparationError(f"{location}: data value contract drift")
        expected_prefix = (
            "mot:v1.11:" if source_id == MOT_SOURCE_ID else "parlamint-tr:v5.0:"
        )
        if not record["id"].startswith(expected_prefix):
            raise AnchorPreparationError(f"{location}: data ID namespace drift")
        if record["id"] in state["data_ids"]:
            raise AnchorPreparationError(f"{location}: duplicate data ID")
        state["data_ids"].add(record["id"])
        if source_id == MOT_SOURCE_ID:
            aliases = record["provenance"].get("aliases")
            if not isinstance(aliases, list) or not aliases:
                raise AnchorPreparationError(f"{location}: MOT aliases drift")
            state["mot_candidate_copies"] += len(aliases)
    elif artifact_key == "input_inventory":
        required = {
            "archive",
            "path",
            "member_type",
            "disposition",
            "size_bytes",
            "sha256",
            "mode",
            "mtime",
        }
        if set(record) != required:
            raise AnchorPreparationError(f"{location}: inventory schema drift")
        if (
            not isinstance(record["archive"], str)
            or record["archive"] not in state["allowed_inventory_archives"]
            or not isinstance(record["path"], str)
            or not isinstance(record["member_type"], str)
            or record["member_type"] not in {"directory", "file"}
            or not isinstance(record["disposition"], str)
            or not record["disposition"]
            or isinstance(record["size_bytes"], bool)
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] < 0
            or (
                record["sha256"] is not None
                and not _SHA256_RE.fullmatch(str(record["sha256"]))
            )
            or isinstance(record["mode"], bool)
            or not isinstance(record["mode"], int)
            or isinstance(record["mtime"], bool)
            or not isinstance(record["mtime"], int | float)
        ):
            raise AnchorPreparationError(f"{location}: inventory value contract drift")
        _safe_member_name(record["path"])
        identity = (record["archive"], record["path"])
        if identity in state["inventory_ids"]:
            raise AnchorPreparationError(f"{location}: duplicate inventory identity")
        state["inventory_ids"].add(identity)
        state["inventory_archive_counts"][record["archive"]] += 1
        state["inventory_dispositions"][
            (record["archive"], record["disposition"])
        ] += 1
    elif artifact_key == "quarantine":
        if set(record) != {
            "anchor",
            "source",
            "source_id",
            "member",
            "reason",
            "evidence",
        }:
            raise AnchorPreparationError(f"{location}: quarantine schema drift")
        reason = record.get("reason")
        if (
            record.get("anchor") != source_id
            or not isinstance(record.get("source"), str)
            or not record["source"]
            or not isinstance(record.get("source_id"), str)
            or not record["source_id"]
            or not isinstance(record.get("member"), str)
            or not record["member"]
            or not isinstance(reason, str)
            or not reason
            or not isinstance(record.get("evidence"), dict)
        ):
            raise AnchorPreparationError(f"{location}: quarantine reason drift")
        state["quarantine_reasons"][reason] += 1
    elif artifact_key == "overlap_audit":
        if (
            set(record)
            != {"id", "selection_reason", "clean_text_conflict", "aliases"}
            or not isinstance(record.get("id"), str)
            or not record["id"].startswith("mot:v1.11:")
            or not isinstance(record.get("selection_reason"), str)
            or not isinstance(record.get("clean_text_conflict"), bool)
            or not isinstance(record.get("aliases"), list)
            or len(record["aliases"]) < 2
        ):
            raise AnchorPreparationError(f"{location}: MOT overlap schema drift")
        if record["id"] in state["overlap_ids"]:
            raise AnchorPreparationError(f"{location}: duplicate MOT overlap ID")
        state["overlap_ids"].add(record["id"])
        state["mot_conflicting_ids"] += int(record["clean_text_conflict"])
    elif artifact_key == "repeated_paragraph_candidates":
        if (
            set(record)
            != {"paragraph_sha256", "occurrences", "documents", "sample", "action"}
            or not _SHA256_RE.fullmatch(str(record.get("paragraph_sha256", "")))
            or isinstance(record.get("occurrences"), bool)
            or not isinstance(record.get("occurrences"), int)
            or record["occurrences"] <= 0
            or isinstance(record.get("documents"), bool)
            or not isinstance(record.get("documents"), int)
            or record["documents"] < REPEATED_PARAGRAPH_MIN_DOCUMENTS
            or not isinstance(record.get("sample"), str)
            or record.get("action") != "report_only_no_deletion"
        ):
            raise AnchorPreparationError(
                f"{location}: repeated-paragraph evidence schema drift"
            )
        digest = record["paragraph_sha256"]
        if digest in state["repeated_ids"]:
            raise AnchorPreparationError(
                f"{location}: duplicate repeated-paragraph digest"
            )
        state["repeated_ids"].add(digest)


def _open_relative_regular_fd(
    root_fd: int, relative: str, *, label: str
) -> tuple[int, os.stat_result]:
    pure = _safe_member_name(relative)
    directory_fd = os.dup(root_fd)
    try:
        for component in pure.parts[:-1]:
            before = os.stat(
                component, dir_fd=directory_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise AnchorPreparationError(
                    f"{label} parent must be a real directory: {component}"
                )
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
            opened = os.fstat(next_fd)
            if _inode_identity(before) != _inode_identity(opened):
                os.close(next_fd)
                raise AnchorPreparationError(f"{label} parent inode drift")
            os.close(directory_fd)
            directory_fd = next_fd
        return _open_regular_at(directory_fd, pure.name, label=label)
    finally:
        os.close(directory_fd)


def _stream_validate_shard(
    root_fd: int,
    *,
    relative: str,
    item: Mapping[str, Any],
    artifact_key: str,
    source_id: str,
    artifact_digest: Any,
    state: dict[str, Any],
) -> tuple[int, int]:
    descriptor, opened = _open_relative_regular_fd(
        root_fd, relative, label=f"artifact {item.get('path')}"
    )
    expected_size = item.get("size_bytes")
    if opened.st_size != expected_size:
        os.close(descriptor)
        raise AnchorPreparationError(f"artifact size drift: {item.get('path')}")
    compressed_digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as hash_handle:
            for chunk in iter(
                lambda: hash_handle.read(VALIDATION_CHUNK_BYTES), b""
            ):
                compressed_digest.update(chunk)
        if compressed_digest.hexdigest() != item.get("sha256"):
            raise AnchorPreparationError(f"artifact SHA-256 drift: {item.get('path')}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        semantic_descriptor = os.dup(descriptor)
        rows = 0
        uncompressed_bytes = 0
        pending = bytearray()
        try:
            with os.fdopen(semantic_descriptor, "rb", closefd=True) as semantic_handle:
                with pa.input_stream(semantic_handle, compression="zstd") as stream:
                    while True:
                        chunk = stream.read(VALIDATION_CHUNK_BYTES)
                        if not chunk:
                            break
                        uncompressed_bytes += len(chunk)
                        artifact_digest.update(chunk)
                        pending.extend(chunk)
                        while True:
                            newline = pending.find(b"\n")
                            if newline < 0:
                                break
                            raw_line = bytes(pending[: newline + 1])
                            del pending[: newline + 1]
                            rows += 1
                            _validate_canonical_record(
                                raw_line,
                                location=f"{item.get('path')}:{rows}",
                                artifact_key=artifact_key,
                                source_id=source_id,
                                state=state,
                            )
                        if len(pending) > MAX_LINE_BYTES:
                            raise AnchorPreparationError(
                                f"artifact line exceeds {MAX_LINE_BYTES} bytes"
                            )
        except (OSError, pa.ArrowException, UnicodeError) as exc:
            raise AnchorPreparationError(
                f"cannot decode artifact: {item.get('path')}"
            ) from exc
        if pending:
            raise AnchorPreparationError(
                f"artifact lacks final LF: {item.get('path')}"
            )
        after = os.fstat(descriptor)
        if not _same_file_snapshot(opened, after):
            raise AnchorPreparationError(
                f"artifact changed during validation: {item.get('path')}"
            )
        state["validated_files"][relative] = _same_file_identity_record(after)
        return rows, uncompressed_bytes
    finally:
        os.close(descriptor)


def _validate_artifact(
    root_fd: int,
    *,
    key: str,
    artifact: Mapping[str, Any],
    source_id: str,
    expected_paths: set[str],
    verify_files: bool,
    state: dict[str, Any],
) -> dict[str, int]:
    if (
        artifact.get("format") != "jsonl.zst"
        or artifact.get("compression") != "zstd"
        or artifact.get("canonicalization") != "canonical_json_sorted_utf8_one_lf_v1"
        or not isinstance(artifact.get("shards"), list)
        or not isinstance(artifact.get("totals"), dict)
        or not _SHA256_RE.fullmatch(str(artifact.get("logical_jsonl_sha256", "")))
    ):
        raise AnchorPreparationError(f"manifest artifact {key!r} is malformed")
    _require_positive_int(artifact.get("target_uncompressed_bytes"), f"{key}.target")
    prefix = _artifact_prefix(key)
    running_rows = 0
    running_bytes = 0
    running_size = 0
    logical_digest = hashlib.sha256()
    shards = artifact["shards"]
    for index, raw_item in enumerate(shards):
        if not isinstance(raw_item, Mapping):
            raise AnchorPreparationError(f"{key} shard entry is malformed")
        item = raw_item
        relative = item.get("path")
        if not isinstance(relative, str):
            raise AnchorPreparationError("artifact path is missing")
        pure = _safe_member_name(relative)
        if pure.parent != prefix or pure.name != f"part-{index:05d}.jsonl.zst":
            raise AnchorPreparationError(f"artifact shard path/order drift: {relative}")
        if relative == "manifest.json" or relative in expected_paths:
            raise AnchorPreparationError(f"duplicate artifact path: {relative}")
        expected_paths.add(relative)
        rows = item.get("rows")
        row_start = item.get("row_start")
        row_end = item.get("row_end_exclusive")
        uncompressed = item.get("uncompressed_bytes")
        size_bytes = item.get("size_bytes")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (
            rows,
            row_start,
            row_end,
            uncompressed,
            size_bytes,
        )):
            raise AnchorPreparationError(f"artifact shard counters drift: {relative}")
        if row_start != running_rows or row_end != row_start + rows:
            raise AnchorPreparationError(f"artifact shard row continuity drift: {relative}")
        if not _SHA256_RE.fullmatch(str(item.get("sha256", ""))):
            raise AnchorPreparationError(f"artifact shard SHA-256 malformed: {relative}")
        if verify_files:
            observed_rows, observed_bytes = _stream_validate_shard(
                root_fd,
                relative=pure.as_posix(),
                item=item,
                artifact_key=key,
                source_id=source_id,
                artifact_digest=logical_digest,
                state=state,
            )
            if observed_rows != rows or observed_bytes != uncompressed:
                raise AnchorPreparationError(f"artifact semantic count drift: {relative}")
        running_rows += rows
        running_bytes += uncompressed
        running_size += size_bytes
    totals = {
        "shards": len(shards),
        "rows": running_rows,
        "uncompressed_bytes": running_bytes,
        "size_bytes": running_size,
    }
    if artifact["totals"] != totals:
        raise AnchorPreparationError(f"artifact totals drift: {key}")
    if verify_files and logical_digest.hexdigest() != artifact["logical_jsonl_sha256"]:
        raise AnchorPreparationError(f"artifact logical JSONL SHA-256 drift: {key}")
    return totals


def _validate_closed_tree(
    root_fd: int,
    expected_files: set[str],
    *,
    validated_files: Mapping[str, tuple[int, ...]],
) -> None:
    expected_dirs: set[str] = set()
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_dirs.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()

    def walk(directory_fd: int, relative_parent: PurePosixPath) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise AnchorPreparationError("cannot inspect prepared output") from exc
        for name in names:
            relative = relative_parent / name
            relative_text = relative.as_posix()
            try:
                entry_stat = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise AnchorPreparationError(
                    f"prepared output entry changed during traversal: {relative_text}"
                ) from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise AnchorPreparationError(
                    f"prepared output contains a symlink: {relative_text}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
                if _inode_identity(entry_stat) != _inode_identity(os.fstat(child_fd)):
                    os.close(child_fd)
                    raise AnchorPreparationError(
                        f"prepared output directory inode drift: {relative_text}"
                    )
                observed_dirs.add(relative_text)
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(entry_stat.st_mode):
                if entry_stat.st_nlink != 1:
                    raise AnchorPreparationError(
                        f"prepared output contains a hard-linked file: {relative_text}"
                    )
                observed_files.add(relative_text)
                expected_identity = validated_files.get(relative_text)
                if relative_text in expected_files and (
                    expected_identity is None
                    or _same_file_identity_record(entry_stat) != expected_identity
                ):
                    raise AnchorPreparationError(
                        f"prepared output file replaced after validation: {relative_text}"
                    )
            else:
                raise AnchorPreparationError(
                    f"prepared output contains a special file: {relative_text}"
                )

    walk(root_fd, PurePosixPath())
    if observed_files != expected_files or observed_dirs != expected_dirs:
        raise AnchorPreparationError(
            "prepared output tree is not closed: "
            f"files_expected={sorted(expected_files)!r}, files_observed={sorted(observed_files)!r}, "
            f"dirs_expected={sorted(expected_dirs)!r}, dirs_observed={sorted(observed_dirs)!r}"
        )


def _validate_frozen_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    source_id: str,
    contract: MotContract | ParlaMintContract,
) -> None:
    if source_id == MOT_SOURCE_ID:
        if not isinstance(contract, MotContract):
            raise AnchorPreparationError("MOT validator contract type drift")
        if (
            manifest.get("source") != _mot_source_manifest(contract)
            or manifest.get("archive_member_policy")
            != _mot_archive_policy(contract)
            or manifest.get("expected_release_totals") is not None
            or manifest.get("cleaning") != _mot_cleaning_manifest()
        ):
            raise AnchorPreparationError("MOT frozen manifest contract drift")
    else:
        if not isinstance(contract, ParlaMintContract):
            raise AnchorPreparationError("ParlaMint validator contract type drift")
        if (
            manifest.get("source") != _parlamint_source_manifest(contract)
            or manifest.get("archive_member_policy")
            != _parlamint_archive_policy(contract)
            or manifest.get("expected_release_totals")
            != _parlamint_expected_totals(contract)
            or manifest.get("cleaning") != _parlamint_cleaning_manifest()
        ):
            raise AnchorPreparationError(
                "ParlaMint frozen manifest contract drift"
            )
    if (
        manifest.get("downstream_admission") != _downstream_gate_declaration()
        or manifest.get("frozen_contract_sha256")
        != _frozen_contract_sha256(manifest)
    ):
        raise AnchorPreparationError("anchor frozen contract hash drift")


def _validate_anchor_preparation_fd(
    root_fd: int,
    *,
    display_path: str | Path,
    verify_files: bool = True,
    contract: MotContract | ParlaMintContract | None = None,
) -> dict[str, Any]:
    """Validate a preparation entirely relative to one already-held directory."""

    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise AnchorPreparationError("prepared output root must be a real directory")
    manifest_payload, manifest_stat = _read_bounded_regular_at(
        root_fd,
        "manifest.json",
        label="anchor manifest",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _strict_json_object(manifest_payload, "anchor manifest")
    verify_manifest_hash(manifest)
    source_id = manifest.get("source_id")
    if (
        set(manifest)
        != {
            "schema_version",
            "kind",
            "preparer_version",
            "source_id",
            "acquisition_receipt",
            "source",
            "inputs",
            "archive_member_policy",
            "expected_release_totals",
            "cleaning",
            "downstream_admission",
            "artifacts",
            "counts",
            "raw",
            "clean",
            "frozen_contract_sha256",
            "production_acceptance",
            "canonical_sha256",
        }
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or manifest.get("preparer_version") != PREPARER_VERSION
        or source_id not in {MOT_SOURCE_ID, PARLAMINT_SOURCE_ID}
        or manifest.get("downstream_admission") != _downstream_gate_declaration()
    ):
        raise AnchorPreparationError("unexpected anchor manifest kind/version/admission")
    if contract is None:
        contract = (
            MOT_V1_11_CONTRACT
            if source_id == MOT_SOURCE_ID
            else PARLAMINT_TR_V5_CONTRACT
        )
    _validate_frozen_manifest_contract(
        manifest, source_id=source_id, contract=contract
    )
    acquisition = manifest.get("acquisition_receipt")
    if not isinstance(acquisition, Mapping):
        raise AnchorPreparationError("anchor manifest lacks acquisition receipt")
    acquired_assets = validate_anchor_acquisition_receipt(
        acquisition, source_id=source_id, contract=contract
    )
    expected_input_names = [
        asset.filename for asset in contract.assets
    ] if isinstance(contract, MotContract) else [contract.filename]
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) != len(expected_input_names)
        or not all(isinstance(item, Mapping) for item in inputs)
        or [item.get("filename") for item in inputs] != expected_input_names
    ):
        raise AnchorPreparationError("anchor manifest input order/inventory drift")
    for item in inputs:
        if not isinstance(item, Mapping):  # narrowed above; retained for -O safety
            raise AnchorPreparationError("anchor manifest input record drift")
        acquired = acquired_assets[item["filename"]]
        if dict(item) != {
            key: acquired[key] for key in ("filename", "size_bytes", "sha256", "md5")
        }:
            raise AnchorPreparationError("anchor manifest/acquisition input binding drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AnchorPreparationError("anchor manifest artifacts must be an object")
    expected_artifact_keys = (
        {"data", "input_inventory", "quarantine", "overlap_audit", "repeated_paragraph_candidates"}
        if source_id == MOT_SOURCE_ID
        else {"data", "input_inventory", "quarantine", "repeated_paragraph_candidates"}
    )
    if set(artifacts) != expected_artifact_keys:
        raise AnchorPreparationError("anchor artifact inventory drift")
    expected_paths: set[str] = {"manifest.json"}
    state: dict[str, Any] = {
        "data_ids": set(),
        "inventory_ids": set(),
        "quarantine_reasons": Counter(),
        "allowed_inventory_archives": set(expected_input_names),
        "inventory_archive_counts": Counter(),
        "inventory_dispositions": Counter(),
        "overlap_ids": set(),
        "repeated_ids": set(),
        "mot_candidate_copies": 0,
        "mot_conflicting_ids": 0,
        "validated_files": {
            "manifest.json": _same_file_identity_record(manifest_stat)
        },
    }
    totals: dict[str, dict[str, int]] = {}
    for key in sorted(artifacts):
        artifact = artifacts[key]
        if not isinstance(artifact, Mapping):
            raise AnchorPreparationError(f"manifest artifact {key!r} is malformed")
        totals[key] = _validate_artifact(
            root_fd,
            key=key,
            artifact=artifact,
            source_id=source_id,
            expected_paths=expected_paths,
            verify_files=verify_files,
            state=state,
        )
    if verify_files:
        _validate_closed_tree(
            root_fd,
            expected_paths,
            validated_files=state["validated_files"],
        )
        if manifest.get("clean") != {
            "documents": totals["data"]["rows"],
            "logical_jsonl_sha256": artifacts["data"]["logical_jsonl_sha256"],
        }:
            raise AnchorPreparationError("anchor clean summary drift")
        counts = manifest.get("counts")
        if not isinstance(counts, Mapping):
            raise AnchorPreparationError("anchor counts must be an object")
        if counts.get("quarantine_reasons") != dict(
            sorted(state["quarantine_reasons"].items())
        ):
            raise AnchorPreparationError("anchor quarantine reason totals drift")
        raw = manifest.get("raw")
        if (
            not isinstance(raw, Mapping)
            or raw.get("member_inventory_canonical_sha256")
            != artifacts["input_inventory"]["logical_jsonl_sha256"]
        ):
            raise AnchorPreparationError("anchor raw inventory summary drift")
        if source_id == MOT_SOURCE_ID:
            archives = counts.get("archives")
            resolution = counts.get("resolution")
            if not isinstance(archives, Mapping) or not isinstance(
                resolution, Mapping
            ):
                raise AnchorPreparationError("MOT count structure drift")
            if set(archives) != set(expected_input_names):
                raise AnchorPreparationError("MOT archive count inventory drift")
            for archive_name in expected_input_names:
                archive_counts = archives.get(archive_name)
                if not isinstance(archive_counts, Mapping):
                    raise AnchorPreparationError("MOT archive counts must be objects")
                if archive_counts.get("members") != state[
                    "inventory_archive_counts"
                ][archive_name]:
                    raise AnchorPreparationError("MOT member inventory/count drift")
                for (observed_archive, disposition), observed in state[
                    "inventory_dispositions"
                ].items():
                    if (
                        observed_archive == archive_name
                        and archive_counts.get(disposition) != observed
                    ):
                        raise AnchorPreparationError(
                            "MOT disposition inventory/count drift"
                        )
            if (
                resolution.get("documents") != totals["data"]["rows"]
                or resolution.get("overlap_article_ids", 0)
                != totals["overlap_audit"]["rows"]
                or resolution.get("candidate_copies")
                != state["mot_candidate_copies"]
                or resolution.get("conflicting_article_ids", 0)
                != state["mot_conflicting_ids"]
                or resolution.get("repeated_paragraph_candidates", 0)
                != totals["repeated_paragraph_candidates"]["rows"]
            ):
                raise AnchorPreparationError("MOT semantic artifact/count drift")
            if raw.get("archive_sha256") != {
                item["filename"]: item["sha256"] for item in inputs
            }:
                raise AnchorPreparationError("MOT raw archive hash summary drift")
        else:
            archive_counts = counts.get("archive")
            join = counts.get("join")
            if not isinstance(archive_counts, Mapping) or not isinstance(join, Mapping):
                raise AnchorPreparationError("ParlaMint count structure drift")
            if archive_counts.get("members") != state[
                "inventory_archive_counts"
            ][expected_input_names[0]]:
                raise AnchorPreparationError("ParlaMint member inventory/count drift")
            for (archive_name, disposition), observed in state[
                "inventory_dispositions"
            ].items():
                if archive_counts.get(disposition) != observed:
                    raise AnchorPreparationError(
                        "ParlaMint disposition inventory/count drift"
                    )
            if (
                join.get("quarantined_speeches") != totals["quarantine"]["rows"]
                or not isinstance(join.get("text_rows"), int)
                or not isinstance(join.get("quarantined_speeches"), int)
                or join["text_rows"] - join["quarantined_speeches"]
                != totals["data"]["rows"]
            ):
                raise AnchorPreparationError("ParlaMint semantic artifact/count drift")
            identity_counts = join.get("cross_format_identity_counts")
            identity_mismatches = join.get("cross_format_identity_mismatches")
            if (
                not isinstance(identity_counts, Mapping)
                or not identity_counts
                or len(set(identity_counts.values())) != 1
                or not isinstance(identity_mismatches, Mapping)
                or any(identity_mismatches.values())
                or archive_counts.get("aggregate_xincludes")
                != identity_counts.get("tei_includes")
                or archive_counts.get("tei_session_identities")
                != identity_counts.get("tei_sessions")
            ):
                raise AnchorPreparationError(
                    "ParlaMint cross-format identity evidence drift"
                )
            if (
                raw.get("archive_sha256") != inputs[0]["sha256"]
                or raw.get("archive_md5") != inputs[0]["md5"]
                or raw.get("cross_format_identity_counts") != identity_counts
            ):
                raise AnchorPreparationError("ParlaMint raw archive hash summary drift")
    projection = _acceptance_projection(manifest)
    production = manifest.get("production_acceptance")
    if not isinstance(production, Mapping) or production.get("projection_sha256") != _projection_sha256(projection):
        raise AnchorPreparationError("anchor production acceptance projection drift")
    if production.get("stage") == "discovery_unaccepted":
        if production.get("eligible_for_production") is not False or "receipt" in production:
            raise AnchorPreparationError("malformed discovery acceptance state")
    elif production.get("stage") == "accepted_production":
        receipt = production.get("receipt")
        if production.get("eligible_for_production") is not True or not isinstance(receipt, Mapping):
            raise AnchorPreparationError("malformed production acceptance state")
        validate_anchor_count_acceptance(receipt, projection=projection)
    else:
        raise AnchorPreparationError("unknown anchor production acceptance stage")
    return manifest


def validate_anchor_preparation(
    output_dir: str | Path,
    *,
    verify_files: bool = True,
    contract: MotContract | ParlaMintContract | None = None,
) -> dict[str, Any]:
    """Validate a sealed preparation through no-follow directory descriptors."""

    root_fd, absolute = _open_directory_path(
        output_dir, label="prepared output", create=False
    )
    try:
        manifest = _validate_anchor_preparation_fd(
            root_fd,
            display_path=absolute,
            verify_files=verify_files,
            contract=contract,
        )
        _assert_path_binds_directory_fd(
            absolute, root_fd, label="prepared output"
        )
        return manifest
    finally:
        os.close(root_fd)


def _mot_database(connection: sqlite3.Connection) -> None:
    _base_database(connection)
    connection.executescript(
        """
        CREATE TABLE candidates (
            article_id INTEGER NOT NULL,
            site TEXT NOT NULL,
            member TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            time_published TEXT,
            time_modified TEXT,
            time_retrieved TEXT,
            modified_sort INTEGER NOT NULL,
            retrieved_sort INTEGER NOT NULL,
            raw_sha256 TEXT NOT NULL,
            clean_sha256 TEXT NOT NULL,
            text TEXT NOT NULL,
            paragraphs_json TEXT NOT NULL,
            title_included INTEGER NOT NULL CHECK(title_included IN (0, 1)),
            PRIMARY KEY (site, article_id, member)
        ) WITHOUT ROWID;
        CREATE INDEX candidates_article_id ON candidates(article_id);
        CREATE TABLE paragraph_stats (
            paragraph_sha256 TEXT PRIMARY KEY,
            paragraph TEXT NOT NULL,
            occurrences INTEGER NOT NULL,
            documents INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _parse_timestamp(value: Any, location: str) -> tuple[str | None, int]:
    if value is None or value == "":
        return None, -1
    if not isinstance(value, str):
        raise AnchorPreparationError(f"{location} must be null or an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnchorPreparationError(f"{location} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    micros = int(parsed.timestamp() * 1_000_000)
    return value, micros


def _agency_provenance(record: Mapping[str, Any]) -> dict[str, str] | None:
    authors = record.get("authors")
    if authors is not None:
        if not isinstance(authors, list) or not all(
            isinstance(author, str) for author in authors
        ):
            raise AnchorPreparationError("MOT authors must be null or a list of strings")
        for author in authors:
            normalized = _normalize_inline(author).upper()
            for agency in _AGENCY_NAMES:
                # ``\w`` is deliberately Unicode-aware here.  An ASCII-only
                # boundary would misclassify Turkish names such as ``Çap`` as
                # containing the AP news agency marker.
                if re.search(rf"(?<!\w){re.escape(agency)}(?!\w)", normalized):
                    return {"field": "authors", "agency": agency, "value": author[:200]}

    title = record.get("title")
    if isinstance(title, str):
        normalized_title = _normalize_inline(title).upper()
        match = re.match(
            r"^(AP|AFP|REUTERS|ASSOCIATED PRESS|AGENCE FRANCE-PRESSE)"
            r"(?:\s|:|[-–—])",
            normalized_title,
        )
        if match:
            return {"field": "title", "agency": match.group(1), "value": title[:200]}

    paragraphs = record.get("paragraphs")
    if isinstance(paragraphs, list):
        for index, paragraph in enumerate(paragraphs[:3]):
            if not isinstance(paragraph, str):
                continue
            normalized = _normalize_inline(paragraph)
            match = re.match(
                r"^(?:(?:KAYNAK|HABER|FOTOĞRAF)\s*:\s*)?"
                r"(?:\([^)]{0,80}\)\s*)?"
                r"(AP|AFP|REUTERS|ASSOCIATED PRESS|AGENCE FRANCE-PRESSE)"
                r"(?:\s|:|[-–—]|$)",
                normalized.upper(),
            )
            if match:
                return {
                    "field": f"paragraphs[{index}]",
                    "agency": match.group(1),
                    "value": normalized[:200],
                }
    return None


def _path_has_token(path: PurePosixPath, token: str) -> bool:
    return token.casefold() in {
        item
        for item in re.split(r"[_\W]+", path.as_posix().casefold())
        if item
    }


def _mot_member_disposition(
    path: PurePosixPath, *, asset: MotAssetContract, kind: str
) -> str:
    parts = path.parts
    ocr_marker = _path_has_token(path, "ocr")
    if kind != "directory" and (
        path.suffix.casefold() == ".pdf" or ocr_marker
    ):
        raise AnchorPreparationError(
            f"MOT PDF/OCR fallback members are forbidden: {path.as_posix()!r}"
        )
    if kind == "directory":
        if parts == (asset.root,):
            return "structural_directory"
        if len(parts) == 2 and parts[0] == asset.root and parts[1] in (
            {"article"} | _MOT_NON_ARTICLE_KINDS
        ):
            return "structural_directory"
        raise AnchorPreparationError(
            f"unexpected directory in {asset.filename}: {path.as_posix()!r}"
        )
    if (
        len(parts) != 3
        or parts[0] != asset.root
        or parts[1] not in ({"article"} | _MOT_NON_ARTICLE_KINDS)
        or not parts[2].endswith(".json")
    ):
        raise AnchorPreparationError(
            f"unexpected member in {asset.filename}: {path.as_posix()!r}"
        )
    return "article_candidate" if parts[1] == "article" else "ignored_non_article"


def _validate_mot_record(
    record: Mapping[str, Any], *, member_path: PurePosixPath, asset: MotAssetContract
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    missing = _MOT_REQUIRED_FIELDS - set(record)
    if missing:
        raise AnchorPreparationError(
            f"{member_path.as_posix()}: missing MOT fields {sorted(missing)!r}"
        )
    filename = record.get("filename")
    if not isinstance(filename, str) or filename != member_path.stem:
        raise AnchorPreparationError(
            f"{member_path.as_posix()}: filename field/member drift"
        )
    match = _MOT_ARTICLE_NAME_RE.fullmatch(member_path.name)
    if match is None:
        raise AnchorPreparationError(
            f"{member_path.as_posix()}: article filename has no trailing numeric VOA ID"
        )
    article_id_text = match.group(1)
    article_id = int(article_id_text)
    if article_id <= 0:
        raise AnchorPreparationError(f"{member_path.as_posix()}: invalid VOA article ID")
    if article_id_text != str(article_id):
        raise AnchorPreparationError(
            f"{member_path.as_posix()}: VOA article ID must be canonical decimal"
        )
    if record.get("content_type") != "article":
        raise AnchorPreparationError(
            f"{member_path.as_posix()}: article path/content_type drift"
        )
    for field in ("url", "url_origin", "title"):
        if not isinstance(record.get(field), str):
            raise AnchorPreparationError(
                f"{member_path.as_posix()}: {field} must be a string"
            )
    for field in ("url", "url_origin"):
        parsed = urlparse(record[field])
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise AnchorPreparationError(
                f"{member_path.as_posix()}: {field} must be an absolute web URL"
            )
    article_url_path = urlparse(record["url"]).path
    if re.search(
        rf"(?:^|[-/]){re.escape(article_id_text)}(?:\.html?)?/?$",
        article_url_path,
    ) is None:
        raise AnchorPreparationError(
            f"{member_path.as_posix()}: URL/article ID identity drift"
        )
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not all(
        isinstance(paragraph, str) for paragraph in paragraphs
    ):
        raise AnchorPreparationError(
            f"{member_path.as_posix()}: paragraphs must be a list of strings"
        )
    n_paragraphs = record.get("n_paragraphs")
    n_chars = record.get("n_chars")
    if (
        isinstance(n_paragraphs, bool)
        or not isinstance(n_paragraphs, int)
        or n_paragraphs != len(paragraphs)
    ):
        raise AnchorPreparationError(
            f"{member_path.as_posix()}: n_paragraphs drift"
        )
    if (
        isinstance(n_chars, bool)
        or not isinstance(n_chars, int)
        or n_chars != sum(len(paragraph) for paragraph in paragraphs)
    ):
        raise AnchorPreparationError(f"{member_path.as_posix()}: n_chars drift")

    language_evidence: dict[str, Any] | None = None
    if record.get("site_language") != "tur" or record.get("predicted_language") != "tur":
        language_evidence = {
            "site_language": record.get("site_language"),
            "predicted_language": record.get("predicted_language"),
        }
    agency_evidence = _agency_provenance(record)
    return article_id, language_evidence, agency_evidence


def _process_mot_archive(
    connection: sqlite3.Connection,
    *,
    snapshot: _VerifiedArchiveSnapshot,
    asset: MotAssetContract,
    contract: MotContract,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    total_uncompressed = 0
    with _open_verified_tar(
        snapshot,
        max_tar_stream_bytes=contract.max_tar_stream_bytes_per_archive,
        max_control_header_bytes=contract.max_control_header_bytes,
        max_control_headers=contract.max_members,
    ) as (archive, tar_state):
        for member_index, member in enumerate(archive, 1):
            if member_index > contract.max_members:
                raise AnchorPreparationError(
                    f"{asset.filename} exceeds the frozen member-count bound"
                )
            path = _safe_member_name(member.name)
            kind = _member_kind(member)
            if member.size < 0 or member.size > contract.max_member_bytes:
                raise AnchorPreparationError(
                    f"{asset.filename}:{member.name} violates member-size bound"
                )
            total_uncompressed += member.size
            if total_uncompressed > contract.max_uncompressed_bytes_per_archive:
                raise AnchorPreparationError(
                    f"{asset.filename} violates uncompressed archive bound"
                )
            disposition = _mot_member_disposition(path, asset=asset, kind=kind)
            member_sha256: str | None = None
            if kind == "file":
                if disposition == "article_candidate":
                    payload = _read_member_bytes(
                        archive, member, limit=contract.max_member_bytes
                    )
                    member_sha256 = hashlib.sha256(payload).hexdigest()
                    record = _strict_json_object(
                        payload, f"{asset.filename}:{path.as_posix()}"
                    )
                    article_id, language_evidence, agency_evidence = _validate_mot_record(
                        record, member_path=path, asset=asset
                    )
                    stable_source_id = f"mot:v1.11:{article_id}"
                    if language_evidence is not None:
                        disposition = "quarantined_language"
                        _insert_quarantine(
                            connection,
                            anchor=MOT_SOURCE_ID,
                            source=asset.root,
                            source_id=stable_source_id,
                            member=path.as_posix(),
                            reason="mot_language_not_tur",
                            evidence=language_evidence,
                        )
                    elif agency_evidence is not None:
                        disposition = "quarantined_agency_provenance"
                        _insert_quarantine(
                            connection,
                            anchor=MOT_SOURCE_ID,
                            source=asset.root,
                            source_id=stable_source_id,
                            member=path.as_posix(),
                            reason="mot_ap_afp_reuters_provenance",
                            evidence=agency_evidence,
                        )
                    else:
                        normalized_paragraphs = _normalize_paragraphs(record["paragraphs"])
                        title = _normalize_inline(record["title"])
                        title_included = _is_substantive_title(title) and title.casefold() not in {
                            paragraph.casefold() for paragraph in normalized_paragraphs
                        }
                        text_parts = ([title] if title_included else []) + normalized_paragraphs
                        text = "\n\n".join(text_parts)
                        if not text:
                            disposition = "quarantined_empty_text"
                            _insert_quarantine(
                                connection,
                                anchor=MOT_SOURCE_ID,
                                source=asset.root,
                                source_id=stable_source_id,
                                member=path.as_posix(),
                                reason="mot_empty_after_whitespace_normalization",
                                evidence={"title": record["title"][:200]},
                            )
                        else:
                            time_published, _published_sort = _parse_timestamp(
                                record.get("time_published"),
                                f"{path.as_posix()}.time_published",
                            )
                            time_modified, modified_sort = _parse_timestamp(
                                record.get("time_modified"),
                                f"{path.as_posix()}.time_modified",
                            )
                            time_retrieved, retrieved_sort = _parse_timestamp(
                                record.get("time_retrieved"),
                                f"{path.as_posix()}.time_retrieved",
                            )
                            if retrieved_sort < 0:
                                raise AnchorPreparationError(
                                    f"{path.as_posix()}: a valid retrieval timestamp is required"
                                )
                            try:
                                connection.execute(
                                    "INSERT INTO candidates VALUES "
                                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        article_id,
                                        asset.root,
                                        path.as_posix(),
                                        record["url"],
                                        title,
                                        time_published,
                                        time_modified,
                                        time_retrieved,
                                        modified_sort,
                                        retrieved_sort,
                                        member_sha256,
                                        _text_sha256(text),
                                        text,
                                        canonical_json(normalized_paragraphs),
                                        int(title_included),
                                    ),
                                )
                            except sqlite3.IntegrityError as exc:
                                raise AnchorPreparationError(
                                    "same-site MOT article ID collision: "
                                    f"{asset.root}:{article_id}"
                                ) from exc
                            disposition = "accepted_candidate"
                else:
                    member_sha256 = _hash_member_stream(archive, member)
            inventory = _inventory_record(
                archive=asset.filename,
                member=member,
                kind=kind,
                disposition=disposition,
                sha256=member_sha256,
            )
            _insert_inventory(connection, inventory)
            counts[disposition] += 1
            if member_index % 10_000 == 0:
                connection.commit()
    counts.update(tar_state)
    connection.commit()
    counts["members"] = sum(
        value
        for key, value in counts.items()
        if key not in {"members", "tar_stream_bytes", "tar_control_headers"}
    )
    counts["uncompressed_bytes"] = total_uncompressed
    return dict(sorted(counts.items()))


def _mot_alias_record(row: sqlite3.Row, *, selected: bool) -> dict[str, Any]:
    return {
        "site": row["site"],
        "member": row["member"],
        "url": row["url"],
        "time_published": row["time_published"],
        "time_modified": row["time_modified"],
        "time_retrieved": row["time_retrieved"],
        "raw_sha256": row["raw_sha256"],
        "clean_text_sha256": row["clean_sha256"],
        "selected": selected,
    }


def _choose_mot_candidate(rows: Sequence[sqlite3.Row]) -> tuple[sqlite3.Row, str]:
    if not rows:
        raise AnchorPreparationError("cannot resolve an empty MOT candidate group")
    if len(rows) == 1:
        return rows[0], "only_valid_copy"
    clean_hashes = {row["clean_sha256"] for row in rows}
    if len(clean_hashes) == 1:
        chosen = max(
            rows,
            key=lambda row: (
                row["site"] == "tur_voaturkce",
                row["modified_sort"],
                row["retrieved_sort"],
                row["member"],
            ),
        )
        return chosen, "identical_clean_text_prefer_voaturkce"
    chosen = max(
        rows,
        key=lambda row: (
            row["modified_sort"],
            row["retrieved_sort"],
            row["site"] == "tur_voaturkce",
            row["clean_sha256"],
            row["member"],
        ),
    )
    return chosen, "conflicting_clean_text_newest_modified_then_retrieved"


def _update_paragraph_stats(
    connection: sqlite3.Connection, paragraphs_json: str
) -> None:
    paragraphs = json.loads(paragraphs_json)
    seen: set[str] = set()
    for paragraph in paragraphs:
        digest = _text_sha256(paragraph)
        connection.execute(
            "INSERT INTO paragraph_stats(paragraph_sha256, paragraph, occurrences, documents) "
            "VALUES (?, ?, 1, ?) "
            "ON CONFLICT(paragraph_sha256) DO UPDATE SET "
            "occurrences = occurrences + 1, documents = documents + excluded.documents",
            (digest, paragraph, int(digest not in seen)),
        )
        seen.add(digest)


def _write_mot_outputs(
    connection: sqlite3.Connection,
    build: Path,
    *,
    shard_target_bytes: int,
    evidence_target_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, int]]:
    connection.row_factory = sqlite3.Row
    data_writer = _JsonlZstdShardWriter(build, "data", shard_target_bytes)
    overlap_writer = _JsonlZstdShardWriter(
        build, "evidence/overlap_audit", evidence_target_bytes
    )
    counts: Counter[str] = Counter()
    article_ids = connection.execute(
        "SELECT DISTINCT article_id FROM candidates ORDER BY article_id"
    )
    for index, article_row in enumerate(article_ids, 1):
        article_id = article_row[0]
        rows = list(
            connection.execute(
                "SELECT * FROM candidates WHERE article_id = ? "
                "ORDER BY site COLLATE BINARY, member COLLATE BINARY",
                (article_id,),
            )
        )
        chosen, reason = _choose_mot_candidate(rows)
        aliases = [
            _mot_alias_record(
                row,
                selected=(
                    row["site"] == chosen["site"]
                    and row["member"] == chosen["member"]
                ),
            )
            for row in rows
        ]
        data_writer.write(
            {
                "id": f"mot:v1.11:{article_id}",
                "text": chosen["text"],
                "source_id": MOT_SOURCE_ID,
                "source_revision": MOT_RELEASE_TAG,
                "provenance": {
                    "article_id": str(article_id),
                    "selected_site": chosen["site"],
                    "selected_member": chosen["member"],
                    "selection_reason": reason,
                    "url": chosen["url"],
                    "time_published": chosen["time_published"],
                    "time_modified": chosen["time_modified"],
                    "time_retrieved": chosen["time_retrieved"],
                    "raw_sha256": chosen["raw_sha256"],
                    "clean_text_sha256": chosen["clean_sha256"],
                    "title_included": bool(chosen["title_included"]),
                    "aliases": aliases,
                },
            }
        )
        _update_paragraph_stats(connection, chosen["paragraphs_json"])
        counts["documents"] += 1
        counts["candidate_copies"] += len(rows)
        if len(rows) > 1:
            overlap_writer.write(
                {
                    "id": f"mot:v1.11:{article_id}",
                    "selection_reason": reason,
                    "clean_text_conflict": len({row["clean_sha256"] for row in rows}) > 1,
                    "aliases": aliases,
                }
            )
            counts["overlap_article_ids"] += 1
            if len({row["clean_sha256"] for row in rows}) > 1:
                counts["conflicting_article_ids"] += 1
        if index % 10_000 == 0:
            connection.commit()
    connection.commit()

    repeated_writer = _JsonlZstdShardWriter(
        build, "evidence/repeated_paragraph_candidates", evidence_target_bytes
    )
    for row in connection.execute(
        "SELECT paragraph_sha256, paragraph, occurrences, documents "
        "FROM paragraph_stats WHERE documents >= ? "
        "ORDER BY documents DESC, occurrences DESC, paragraph_sha256 COLLATE BINARY",
        (REPEATED_PARAGRAPH_MIN_DOCUMENTS,),
    ):
        repeated_writer.write(
            {
                "paragraph_sha256": row["paragraph_sha256"],
                "occurrences": row["occurrences"],
                "documents": row["documents"],
                "sample": row["paragraph"][:REPEATED_PARAGRAPH_SAMPLE_CHARS],
                "action": "report_only_no_deletion",
            }
        )
        counts["repeated_paragraph_candidates"] += 1
    return (
        data_writer.finish(),
        overlap_writer.finish(emit_empty=True),
        repeated_writer.finish(emit_empty=True),
        dict(sorted(counts.items())),
    )


def prepare_mot_v1_11(
    amerikaninsesi_tgz: str | Path,
    voaturkce_tgz: str | Path,
    output_dir: str | Path,
    *,
    acquisition_receipt_path: str | Path,
    discovery: bool = False,
    count_acceptance_path: str | Path | None = None,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    evidence_target_bytes: int = DEFAULT_EVIDENCE_TARGET_BYTES,
    contract: MotContract = MOT_V1_11_CONTRACT,
) -> dict[str, Any]:
    """Prepare the two pinned Turkish MOT v1.11 site archives offline."""

    if len(contract.assets) != 2 or {asset.root for asset in contract.assets} != {
        "tur_amerikaninsesi",
        "tur_voaturkce",
    }:
        raise AnchorPreparationError("MOT contract must contain exactly the two Turkish sites")
    if discovery == (count_acceptance_path is not None):
        raise AnchorPreparationError(
            "choose exactly one MOT mode: discovery or accepted production"
        )
    supplied = {
        "tur_amerikaninsesi": Path(amerikaninsesi_tgz),
        "tur_voaturkce": Path(voaturkce_tgz),
    }
    acquisition_receipt, acquired_assets = _load_acquisition_receipt(
        acquisition_receipt_path, source_id=MOT_SOURCE_ID, contract=contract
    )

    target = _prepare_destination(output_dir)
    build = target.build_path
    database_path = build / ".mot_work.sqlite3"
    snapshot_dir = build / ".verified_inputs"
    connection: sqlite3.Connection | None = None
    try:
        snapshots: dict[str, _VerifiedArchiveSnapshot] = {}
        verified_inputs: list[dict[str, Any]] = []
        for asset in contract.assets:
            acquired = acquired_assets[asset.filename]
            snapshot = _snapshot_verified_archive(
                supplied[asset.root],
                snapshot_dir / asset.filename,
                expected_name=asset.filename,
                expected_size=asset.size_bytes,
                expected_sha256=acquired["sha256"],
            )
            snapshots[asset.root] = snapshot
            verified_inputs.append(snapshot.input_record)
        connection = _open_sqlite(database_path)
        _mot_database(connection)
        archive_counts: dict[str, dict[str, int]] = {}
        for asset in contract.assets:
            archive_counts[asset.filename] = _process_mot_archive(
                connection,
                snapshot=snapshots[asset.root],
                asset=asset,
                contract=contract,
            )
        candidate_count = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        if candidate_count <= 0:
            raise AnchorPreparationError("MOT preparation produced no valid article candidates")
        data, overlaps, repeated, resolution_counts = _write_mot_outputs(
            connection,
            build,
            shard_target_bytes=shard_target_bytes,
            evidence_target_bytes=evidence_target_bytes,
        )
        input_inventory, inventory_sha256 = _write_inventory(
            connection, build, target_bytes=evidence_target_bytes
        )
        quarantine, quarantine_reasons = _write_quarantine(
            connection, build, target_bytes=evidence_target_bytes
        )
        connection.close()
        database_path.unlink()
        shutil.rmtree(snapshot_dir)
        manifest = _finalize_manifest(
            build,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "kind": MANIFEST_KIND,
                "preparer_version": PREPARER_VERSION,
                "source_id": MOT_SOURCE_ID,
                "acquisition_receipt": acquisition_receipt,
                "source": _mot_source_manifest(contract),
                "inputs": verified_inputs,
                "archive_member_policy": _mot_archive_policy(contract),
                "expected_release_totals": None,
                "cleaning": _mot_cleaning_manifest(),
                "downstream_admission": _downstream_gate_declaration(),
                "artifacts": {
                    "data": data,
                    "input_inventory": input_inventory,
                    "quarantine": quarantine,
                    "overlap_audit": overlaps,
                    "repeated_paragraph_candidates": repeated,
                },
                "counts": {
                    "archives": archive_counts,
                    "resolution": resolution_counts,
                    "quarantine_reasons": quarantine_reasons,
                },
                "raw": {
                    "archive_sha256": {
                        item["filename"]: item["sha256"] for item in verified_inputs
                    },
                    "member_inventory_canonical_sha256": inventory_sha256,
                },
                "clean": {
                    "documents": data["totals"]["rows"],
                    "logical_jsonl_sha256": data["logical_jsonl_sha256"],
                },
                "canonical_sha256": None,
            },
            discovery=discovery,
            count_acceptance_path=count_acceptance_path,
        )
        _validate_anchor_preparation_fd(
            target.build_fd,
            display_path=target.destination,
            contract=contract,
        )
        published_manifest = _publish_build(target, contract=contract)
        if published_manifest != manifest:
            raise AnchorPreparationError("MOT manifest drift at publication")
        _cleanup_build(target)
        return manifest
    except BaseException:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        _cleanup_build(target)
        raise


class _HashingReader:
    def __init__(
        self,
        handle: BinaryIO,
        *,
        expected_size: int,
        forbidden_tokens: Sequence[bytes] = (),
    ) -> None:
        self.handle = handle
        self.expected_size = expected_size
        self.observed_size = 0
        self.digest = hashlib.sha256()
        self.forbidden_tokens = tuple(token.upper() for token in forbidden_tokens)
        self._scan_tail = b""

    def _observe(self, data: bytes) -> bytes:
        self.observed_size += len(data)
        if self.observed_size > self.expected_size:
            raise AnchorPreparationError("streamed member exceeds declared size")
        self.digest.update(data)
        if self.forbidden_tokens and data:
            combined = (self._scan_tail + data).upper()
            if any(token in combined for token in self.forbidden_tokens):
                raise AnchorPreparationError(
                    "TEI DTD/entity declarations are forbidden"
                )
            tail_size = max(len(token) for token in self.forbidden_tokens) - 1
            self._scan_tail = combined[-tail_size:] if tail_size > 0 else b""
        return data

    def read(self, size: int = -1) -> bytes:
        return self._observe(self.handle.read(size))

    def readline(self, size: int = -1) -> bytes:
        return self._observe(self.handle.readline(size))

    def finish(self) -> str:
        if self.observed_size != self.expected_size:
            raise AnchorPreparationError(
                "streamed tar member size drift: "
                f"expected {self.expected_size}, got {self.observed_size}"
            )
        return self.digest.hexdigest()


def _parlamint_database(connection: sqlite3.Connection) -> None:
    _base_database(connection)
    connection.executescript(
        """
        CREATE TABLE speech_text (
            speech_id TEXT PRIMARY KEY,
            member TEXT NOT NULL,
            text_id TEXT NOT NULL,
            raw_text_sha256 TEXT NOT NULL,
            clean_text_sha256 TEXT,
            clean_text TEXT,
            raw_words INTEGER NOT NULL,
            quarantine_reason TEXT
        ) WITHOUT ROWID;
        CREATE TABLE speech_meta (
            speech_id TEXT PRIMARY KEY,
            member TEXT NOT NULL,
            text_id TEXT NOT NULL,
            speech_date TEXT NOT NULL,
            language TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE tei_sessions (
            text_id TEXT PRIMARY KEY,
            member TEXT NOT NULL UNIQUE,
            session_date TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE tei_includes (
            text_id TEXT PRIMARY KEY,
            href TEXT NOT NULL UNIQUE,
            session_date TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE native_text_files (
            text_id TEXT PRIMARY KEY,
            member TEXT NOT NULL UNIQUE,
            session_date TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE native_meta_files (
            text_id TEXT PRIMARY KEY,
            member TEXT NOT NULL UNIQUE,
            session_date TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE paragraph_stats (
            paragraph_sha256 TEXT PRIMARY KEY,
            paragraph TEXT NOT NULL,
            occurrences INTEGER NOT NULL,
            documents INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _validate_parlamint_date(
    value: str, *, contract: ParlaMintContract, location: str
) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise AnchorPreparationError(f"{location}: invalid calendar date") from exc
    canonical = parsed.strftime("%Y-%m-%d")
    if canonical != value or not contract.first_date <= value <= contract.last_date:
        raise AnchorPreparationError(
            f"{location}: date falls outside the frozen ParlaMint span "
            f"{contract.first_date}..{contract.last_date}"
        )
    return value


def _parlamint_member_disposition(
    path: PurePosixPath, *, kind: str, contract: ParlaMintContract
) -> str:
    parts = path.parts
    if kind == "directory":
        if parts in {
            ("ParlaMint-TR.TEI",),
            ("ParlaMint-TR.TEI", "Schema"),
            ("ParlaMint-TR.txt",),
        }:
            return "structural_directory"
        if (
            len(parts) == 2
            and parts[0] in {"ParlaMint-TR.TEI", "ParlaMint-TR.txt"}
            and re.fullmatch(r"\d{4}", parts[1])
        ):
            return "structural_directory"
        raise AnchorPreparationError(
            f"unexpected ParlaMint directory: {path.as_posix()!r}"
        )
    name = path.as_posix()
    if name == "README-TR.md":
        return "readme"
    if path.suffix.casefold() == ".pdf" or _path_has_token(path, "ocr"):
        raise AnchorPreparationError(
            f"ParlaMint PDF/OCR fallback members are forbidden: {name!r}"
        )
    if parts[0] == "ParlaMint-TR.TEI":
        if len(parts) == 2 and parts[1] == "00README.txt":
            return "readme"
        if (
            len(parts) == 3
            and parts[1] == "Schema"
            and parts[2] in _PARLAMINT_SCHEMA_SUPPORT_FILES
        ):
            return "schema_support"
        if (
            len(parts) == 2
            and parts[1].endswith(".xml")
            and parts[1].startswith(("ParlaMint-TR", "ParlaMint-taxonomy"))
        ):
            return "tei_xml"
        session_match = _PARLAMINT_SESSION_XML_RE.fullmatch(name)
        if session_match:
            session_date = _validate_parlamint_date(
                session_match.group(3), contract=contract, location=name
            )
            if session_match.group(1) != session_date[:4]:
                raise AnchorPreparationError(
                    f"ParlaMint TEI directory-year/date drift: {name!r}"
                )
            return "tei_session_xml"
        raise AnchorPreparationError(f"unexpected ParlaMint TEI member: {name!r}")
    if parts[0] == "ParlaMint-TR.txt":
        if len(parts) == 2 and parts[1] == "00README.txt":
            return "readme"
        text_match = _PARLAMINT_TEXT_RE.fullmatch(name)
        if text_match:
            text_date = _validate_parlamint_date(
                text_match.group(3), contract=contract, location=name
            )
            if text_match.group(1) != text_date[:4]:
                raise AnchorPreparationError(
                    f"ParlaMint text directory-year/date drift: {name!r}"
                )
            return "native_speech_text"
        native_meta_match = _PARLAMINT_NATIVE_META_RE.fullmatch(name)
        if native_meta_match:
            meta_date = _validate_parlamint_date(
                native_meta_match.group(3), contract=contract, location=name
            )
            if native_meta_match.group(1) != meta_date[:4]:
                raise AnchorPreparationError(
                    f"ParlaMint metadata directory-year/date drift: {name!r}"
                )
            return "native_speech_metadata"
        english_meta_match = _PARLAMINT_EN_META_RE.fullmatch(name)
        if english_meta_match:
            english_date = _validate_parlamint_date(
                english_meta_match.group(2), contract=contract, location=name
            )
            if english_meta_match.group(1) != english_date[:4]:
                raise AnchorPreparationError(
                    f"ParlaMint English metadata directory-year/date drift: {name!r}"
                )
            return "ignored_english_metadata"
        raise AnchorPreparationError(f"unexpected ParlaMint text member: {name!r}")
    raise AnchorPreparationError(f"unexpected ParlaMint member: {name!r}")


def _validate_tei_stream(
    connection: sqlite3.Connection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    path: PurePosixPath,
    disposition: str,
    contract: ParlaMintContract,
    require_turkish_root: bool,
) -> tuple[str, str, str | None, dict[str, int]]:
    handle = archive.extractfile(member)
    if handle is None:
        raise AnchorPreparationError(f"cannot read TEI member {path.as_posix()!r}")
    reader = _HashingReader(
        handle,
        expected_size=member.size,
        forbidden_tokens=(b"<!DOCTYPE", b"<!ENTITY"),
    )
    root_name: str | None = None
    root_language: str | None = None
    root_element: ET.Element | None = None
    declared_extents: dict[str, int] = {}
    stack: list[ET.Element] = []
    aggregate = path.as_posix() == "ParlaMint-TR.TEI/ParlaMint-TR.xml"
    try:
        for event, element in ET.iterparse(reader, events=("start", "end")):
            if event == "start":
                stack.append(element)
                if len(stack) > MAX_TEI_DEPTH:
                    raise AnchorPreparationError(
                        f"{path.as_posix()}: TEI depth exceeds {MAX_TEI_DEPTH}"
                    )
                if len(element.attrib) > MAX_TEI_ATTRIBUTES_PER_ELEMENT:
                    raise AnchorPreparationError(
                        f"{path.as_posix()}: too many TEI attributes"
                    )
                attribute_bytes = sum(
                    len(str(key).encode("utf-8")) + len(str(value).encode("utf-8"))
                    for key, value in element.attrib.items()
                )
                if attribute_bytes > MAX_TEI_ATTRIBUTE_BYTES:
                    raise AnchorPreparationError(
                        f"{path.as_posix()}: TEI attribute bytes exceed bound"
                    )
            if root_name is None and event == "start":
                if reader.observed_size > MAX_TEI_PROLOG_BYTES:
                    raise AnchorPreparationError(
                        f"{path.as_posix()}: TEI prolog exceeds bound"
                    )
                if element.tag.startswith("{"):
                    namespace, local = element.tag[1:].split("}", 1)
                else:
                    namespace, local = "", element.tag
                if namespace != _TEI_NS:
                    raise AnchorPreparationError(
                        f"{path.as_posix()}: TEI namespace drift"
                    )
                root_name = local
                root_language = element.attrib.get(_XML_LANG)
                root_element = element
                if require_turkish_root and root_language != "tr":
                    raise AnchorPreparationError(
                        f"{path.as_posix()}: root xml:lang must be 'tr'"
                    )
            if event == "end":
                if element.tag == f"{{{_TEI_NS}}}measure":
                    unit = element.attrib.get("unit")
                    quantity = element.attrib.get("quantity")
                    if unit in {"speeches", "words"}:
                        if quantity is None or not re.fullmatch(r"\d+", quantity):
                            raise AnchorPreparationError(
                                f"{path.as_posix()}: invalid declared {unit} extent"
                            )
                        parsed_quantity = int(quantity)
                        previous = declared_extents.get(unit)
                        if previous is not None and previous != parsed_quantity:
                            raise AnchorPreparationError(
                                f"{path.as_posix()}: conflicting declared {unit} extents"
                            )
                        declared_extents[unit] = parsed_quantity
                if element.tag == f"{{{_XI_NS}}}include":
                    if aggregate:
                        href = element.attrib.get("href")
                        if not isinstance(href, str):
                            raise AnchorPreparationError(
                                f"{path.as_posix()}: XInclude lacks href"
                            )
                        include_match = _PARLAMINT_XINCLUDE_RE.fullmatch(href)
                        if include_match is None:
                            raise AnchorPreparationError(
                                f"{path.as_posix()}: non-session aggregate XInclude {href!r}"
                            )
                        include_date = _validate_parlamint_date(
                            include_match.group(3), contract=contract, location=href
                        )
                        if include_match.group(1) != include_date[:4]:
                            raise AnchorPreparationError(
                                f"{path.as_posix()}: XInclude directory-year/date drift"
                            )
                        text_id = include_match.group(2)
                        try:
                            connection.execute(
                                "INSERT INTO tei_includes VALUES (?, ?, ?)",
                                (text_id, href, include_date),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise AnchorPreparationError(
                                "duplicate ParlaMint aggregate XInclude identity: "
                                f"{text_id}"
                            ) from exc
                if not stack or stack[-1] is not element:
                    raise AnchorPreparationError(
                        f"{path.as_posix()}: TEI parser stack drift"
                    )
                parent = stack[-2] if len(stack) >= 2 else None
                element.clear()
                if parent is not None:
                    try:
                        parent.remove(element)
                    except ValueError:
                        pass
                stack.pop()
    except ET.ParseError as exc:
        raise AnchorPreparationError(
            f"{path.as_posix()}: malformed TEI XML"
        ) from exc
    if root_name is None:
        raise AnchorPreparationError(f"{path.as_posix()}: empty TEI XML")
    if stack or root_element is None:
        raise AnchorPreparationError(f"{path.as_posix()}: incomplete TEI parse")
    if disposition == "tei_session_xml":
        if root_name != "TEI" or root_language != "tr":
            raise AnchorPreparationError(
                f"{path.as_posix()}: session root must be TEI with xml:lang='tr'"
            )
        session_match = _PARLAMINT_SESSION_XML_RE.fullmatch(path.as_posix())
        if session_match is None:
            raise AnchorPreparationError("ParlaMint session path identity drift")
        session_date = _validate_parlamint_date(
            session_match.group(3), contract=contract, location=path.as_posix()
        )
        text_id = session_match.group(2)
        try:
            connection.execute(
                "INSERT INTO tei_sessions VALUES (?, ?, ?)",
                (text_id, path.as_posix(), session_date),
            )
        except sqlite3.IntegrityError as exc:
            raise AnchorPreparationError(
                f"duplicate ParlaMint TEI session identity: {text_id}"
            ) from exc
    elif aggregate and (root_name != "teiCorpus" or root_language != "tr"):
        raise AnchorPreparationError(
            "ParlaMint aggregate root must be teiCorpus with xml:lang='tr'"
        )
    return reader.finish(), root_name, root_language, declared_extents


def _strip_balanced_comments(text: str) -> tuple[str | None, dict[str, Any] | None]:
    pieces: list[str] = []
    cursor = 0
    removed = 0
    while cursor < len(text):
        opening = text.find("[[", cursor)
        closing_before = text.find("]]", cursor)
        if closing_before != -1 and (opening == -1 or closing_before < opening):
            return None, {
                "kind": "orphan_closing_delimiter",
                "character_offset": closing_before,
                "excerpt": text[max(0, closing_before - 80) : closing_before + 82],
            }
        if opening == -1:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor:opening])
        closing = text.find("]]", opening + 2)
        if closing == -1:
            return None, {
                "kind": "missing_closing_delimiter",
                "character_offset": opening,
                "excerpt": text[max(0, opening - 80) : opening + 160],
            }
        nested = text.find("[[", opening + 2, closing)
        if nested != -1:
            return None, {
                "kind": "nested_opening_delimiter",
                "character_offset": nested,
                "excerpt": text[max(0, nested - 80) : nested + 160],
            }
        pieces.append(" ")
        removed += 1
        cursor = closing + 2
    cleaned = _normalize_inline("".join(pieces))
    return cleaned, {"balanced_comments_removed": removed}


def _read_binary_lines(reader: _HashingReader, *, location: str) -> Iterator[tuple[int, bytes]]:
    line_number = 0
    while True:
        raw = reader.readline(MAX_LINE_BYTES + 1)
        if not raw:
            break
        line_number += 1
        if len(raw) > MAX_LINE_BYTES:
            raise AnchorPreparationError(
                f"{location}:{line_number}: line exceeds {MAX_LINE_BYTES} bytes"
            )
        yield line_number, raw


def _decode_tsv_line(raw: bytes, *, location: str, line_number: int) -> str:
    if raw.endswith(b"\n"):
        raw = raw[:-1]
        if raw.endswith(b"\r"):
            raw = raw[:-1]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnchorPreparationError(
            f"{location}:{line_number}: invalid UTF-8"
        ) from exc


def _process_parlamint_text_member(
    connection: sqlite3.Connection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    path: PurePosixPath,
) -> tuple[str, int, int]:
    match = _PARLAMINT_TEXT_RE.fullmatch(path.as_posix())
    assert match is not None
    text_id = match.group(2)
    text_date = match.group(3)
    try:
        connection.execute(
            "INSERT INTO native_text_files VALUES (?, ?, ?)",
            (text_id, path.as_posix(), text_date),
        )
    except sqlite3.IntegrityError as exc:
        raise AnchorPreparationError(
            f"duplicate ParlaMint native text file identity: {text_id}"
        ) from exc
    handle = archive.extractfile(member)
    if handle is None:
        raise AnchorPreparationError(f"cannot read {path.as_posix()!r}")
    reader = _HashingReader(handle, expected_size=member.size)
    rows = 0
    raw_words_total = 0
    for line_number, raw_line in _read_binary_lines(
        reader, location=path.as_posix()
    ):
        line = _decode_tsv_line(
            raw_line, location=path.as_posix(), line_number=line_number
        )
        if line.count("\t") != 1:
            raise AnchorPreparationError(
                f"{path.as_posix()}:{line_number}: expected exactly one TAB"
            )
        speech_id, raw_text = line.split("\t")
        if not _PARLAMINT_SPEECH_ID_RE.fullmatch(speech_id):
            raise AnchorPreparationError(
                f"{path.as_posix()}:{line_number}: invalid speech ID"
            )
        raw_words = len(re.findall(r"\S+", raw_text))
        raw_words_total += raw_words
        clean_text, comment_evidence = _strip_balanced_comments(raw_text)
        quarantine_reason: str | None = None
        clean_sha256: str | None = None
        if clean_text is None:
            quarantine_reason = "parlamint_unbalanced_transcriber_comment"
            _insert_quarantine(
                connection,
                anchor=PARLAMINT_SOURCE_ID,
                source="ParlaMint-TR.txt",
                source_id=f"parlamint-tr:v5.0:{speech_id}",
                member=path.as_posix(),
                reason=quarantine_reason,
                evidence={"line_number": line_number, **(comment_evidence or {})},
            )
        elif not clean_text:
            quarantine_reason = "parlamint_empty_after_comment_removal"
            _insert_quarantine(
                connection,
                anchor=PARLAMINT_SOURCE_ID,
                source="ParlaMint-TR.txt",
                source_id=f"parlamint-tr:v5.0:{speech_id}",
                member=path.as_posix(),
                reason=quarantine_reason,
                evidence={"line_number": line_number, **(comment_evidence or {})},
            )
        else:
            clean_sha256 = _text_sha256(clean_text)
        try:
            connection.execute(
                "INSERT INTO speech_text VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    speech_id,
                    path.as_posix(),
                    text_id,
                    _text_sha256(raw_text),
                    clean_sha256,
                    clean_text,
                    raw_words,
                    quarantine_reason,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise AnchorPreparationError(
                f"duplicate ParlaMint speech text ID: {speech_id!r}"
            ) from exc
        rows += 1
    return reader.finish(), rows, raw_words_total


def _process_parlamint_meta_member(
    connection: sqlite3.Connection,
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    path: PurePosixPath,
) -> tuple[str, int]:
    match = _PARLAMINT_NATIVE_META_RE.fullmatch(path.as_posix())
    assert match is not None
    expected_text_id = match.group(2)
    expected_date = match.group(3)
    try:
        connection.execute(
            "INSERT INTO native_meta_files VALUES (?, ?, ?)",
            (expected_text_id, path.as_posix(), expected_date),
        )
    except sqlite3.IntegrityError as exc:
        raise AnchorPreparationError(
            f"duplicate ParlaMint native metadata file identity: {expected_text_id}"
        ) from exc
    handle = archive.extractfile(member)
    if handle is None:
        raise AnchorPreparationError(f"cannot read {path.as_posix()!r}")
    reader = _HashingReader(handle, expected_size=member.size)
    rows = 0
    iterator = _read_binary_lines(reader, location=path.as_posix())
    try:
        header_number, raw_header = next(iterator)
    except StopIteration as exc:
        raise AnchorPreparationError(f"{path.as_posix()}: empty native metadata") from exc
    header = tuple(
        _decode_tsv_line(
            raw_header, location=path.as_posix(), line_number=header_number
        ).split("\t")
    )
    if header != PARLAMINT_NATIVE_META_HEADER:
        raise AnchorPreparationError(f"{path.as_posix()}: native metadata header drift")
    indexes = {name: index for index, name in enumerate(header)}
    for line_number, raw_line in iterator:
        line = _decode_tsv_line(
            raw_line, location=path.as_posix(), line_number=line_number
        )
        fields = line.split("\t")
        if len(fields) != len(PARLAMINT_NATIVE_META_HEADER):
            raise AnchorPreparationError(
                f"{path.as_posix()}:{line_number}: metadata column-count drift"
            )
        speech_id = fields[indexes["ID"]]
        text_id = fields[indexes["Text_ID"]]
        speech_date = fields[indexes["Date"]]
        language = fields[indexes["Lang"]]
        if not _PARLAMINT_SPEECH_ID_RE.fullmatch(speech_id):
            raise AnchorPreparationError(
                f"{path.as_posix()}:{line_number}: invalid metadata speech ID"
            )
        if text_id != expected_text_id or speech_date != expected_date:
            raise AnchorPreparationError(
                f"{path.as_posix()}:{line_number}: native file identity/date drift"
            )
        try:
            datetime.strptime(speech_date, "%Y-%m-%d")
        except ValueError as exc:
            raise AnchorPreparationError(
                f"{path.as_posix()}:{line_number}: invalid speech date"
            ) from exc
        if language != "Türkçe":
            raise AnchorPreparationError(
                f"{path.as_posix()}:{line_number}: Lang must be exactly 'Türkçe'"
            )
        try:
            connection.execute(
                "INSERT INTO speech_meta VALUES (?, ?, ?, ?, ?)",
                (speech_id, path.as_posix(), text_id, speech_date, language),
            )
        except sqlite3.IntegrityError as exc:
            raise AnchorPreparationError(
                f"duplicate ParlaMint native metadata ID: {speech_id!r}"
            ) from exc
        rows += 1
    return reader.finish(), rows


def _process_parlamint_archive(
    connection: sqlite3.Connection,
    *,
    snapshot: _VerifiedArchiveSnapshot,
    contract: ParlaMintContract,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    total_uncompressed = 0
    with _open_verified_tar(
        snapshot,
        max_tar_stream_bytes=contract.max_tar_stream_bytes,
        max_control_header_bytes=contract.max_control_header_bytes,
        max_control_headers=contract.max_members,
    ) as (archive, tar_state):
        for member_index, member in enumerate(archive, 1):
            if member_index > contract.max_members:
                raise AnchorPreparationError("ParlaMint archive exceeds member-count bound")
            path = _safe_member_name(member.name)
            kind = _member_kind(member)
            if member.size < 0 or member.size > contract.max_member_bytes:
                raise AnchorPreparationError(
                    f"ParlaMint member violates size bound: {path.as_posix()!r}"
                )
            total_uncompressed += member.size
            if total_uncompressed > contract.max_uncompressed_bytes:
                raise AnchorPreparationError("ParlaMint archive violates uncompressed bound")
            disposition = _parlamint_member_disposition(
                path, kind=kind, contract=contract
            )
            member_sha256: str | None = None
            if kind == "file":
                if disposition in {"tei_xml", "tei_session_xml"}:
                    require_turkish = disposition == "tei_session_xml" or path.name.startswith(
                        "ParlaMint-TR"
                    )
                    (
                        member_sha256,
                        root_name,
                        root_language,
                        declared_extents,
                    ) = _validate_tei_stream(
                        connection,
                        archive,
                        member,
                        path=path,
                        disposition=disposition,
                        contract=contract,
                        require_turkish_root=require_turkish,
                    )
                    if path.as_posix() == "ParlaMint-TR.TEI/ParlaMint-TR.xml":
                        counts["aggregate_tei_roots"] += 1
                        if root_name != "teiCorpus":
                            raise AnchorPreparationError(
                                "ParlaMint aggregate TEI root must be teiCorpus"
                            )
                        expected_extents = {
                            "speeches": contract.expected_speeches,
                            "words": contract.expected_declared_words,
                        }
                        if declared_extents != expected_extents:
                            raise AnchorPreparationError(
                                "ParlaMint aggregate TEI extent drift: "
                                f"expected={expected_extents!r}, got={declared_extents!r}"
                            )
                        counts["tei_declared_speeches"] = declared_extents["speeches"]
                        counts["tei_declared_words"] = declared_extents["words"]
                    if root_language == "tr":
                        counts["tei_roots_xml_lang_tr"] += 1
                    else:
                        counts["tei_ancillary_roots_other_language"] += 1
                elif disposition == "native_speech_text":
                    member_sha256, rows, raw_words = _process_parlamint_text_member(
                        connection, archive, member, path=path
                    )
                    counts["native_text_rows"] += rows
                    counts["raw_words"] += raw_words
                elif disposition == "native_speech_metadata":
                    member_sha256, rows = _process_parlamint_meta_member(
                        connection, archive, member, path=path
                    )
                    counts["native_metadata_rows"] += rows
                else:
                    member_sha256 = _hash_member_stream(archive, member)
            _insert_inventory(
                connection,
                _inventory_record(
                    archive=contract.filename,
                    member=member,
                    kind=kind,
                    disposition=disposition,
                    sha256=member_sha256,
                ),
            )
            counts[disposition] += 1
            if member_index % 1_000 == 0:
                connection.commit()
    counts.update(tar_state)
    connection.commit()
    if counts["aggregate_tei_roots"] != 1:
        raise AnchorPreparationError(
            "ParlaMint archive must contain exactly one "
            "ParlaMint-TR.TEI/ParlaMint-TR.xml aggregate root"
        )
    counts["aggregate_xincludes"] = connection.execute(
        "SELECT COUNT(*) FROM tei_includes"
    ).fetchone()[0]
    counts["tei_session_identities"] = connection.execute(
        "SELECT COUNT(*) FROM tei_sessions"
    ).fetchone()[0]
    counts["members"] = sum(
        value
        for key, value in counts.items()
        if key
        not in {
            "native_text_rows",
            "native_metadata_rows",
            "raw_words",
            "tei_roots_xml_lang_tr",
            "tei_ancillary_roots_other_language",
            "aggregate_tei_roots",
            "tei_declared_speeches",
            "tei_declared_words",
            "aggregate_xincludes",
            "tei_session_identities",
            "tar_stream_bytes",
            "tar_control_headers",
        }
    )
    counts["uncompressed_bytes"] = total_uncompressed
    return dict(sorted(counts.items()))


def _validate_parlamint_join(
    connection: sqlite3.Connection, contract: ParlaMintContract
) -> dict[str, Any]:
    identity_counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "tei_sessions",
            "tei_includes",
            "native_text_files",
            "native_meta_files",
        )
    }
    identity_mismatches = {
        "session_without_include": connection.execute(
            "SELECT COUNT(*) FROM tei_sessions AS s LEFT JOIN tei_includes AS i "
            "ON i.text_id = s.text_id WHERE i.text_id IS NULL"
        ).fetchone()[0],
        "include_without_session": connection.execute(
            "SELECT COUNT(*) FROM tei_includes AS i LEFT JOIN tei_sessions AS s "
            "ON s.text_id = i.text_id WHERE s.text_id IS NULL"
        ).fetchone()[0],
        "session_without_native_text": connection.execute(
            "SELECT COUNT(*) FROM tei_sessions AS s LEFT JOIN native_text_files AS n "
            "ON n.text_id = s.text_id WHERE n.text_id IS NULL"
        ).fetchone()[0],
        "native_text_without_session": connection.execute(
            "SELECT COUNT(*) FROM native_text_files AS n LEFT JOIN tei_sessions AS s "
            "ON s.text_id = n.text_id WHERE s.text_id IS NULL"
        ).fetchone()[0],
        "session_without_native_meta": connection.execute(
            "SELECT COUNT(*) FROM tei_sessions AS s LEFT JOIN native_meta_files AS n "
            "ON n.text_id = s.text_id WHERE n.text_id IS NULL"
        ).fetchone()[0],
        "native_meta_without_session": connection.execute(
            "SELECT COUNT(*) FROM native_meta_files AS n LEFT JOIN tei_sessions AS s "
            "ON s.text_id = n.text_id WHERE s.text_id IS NULL"
        ).fetchone()[0],
        "identity_date_or_href_mismatch": connection.execute(
            "SELECT COUNT(*) FROM tei_sessions AS s "
            "JOIN tei_includes AS i ON i.text_id = s.text_id "
            "JOIN native_text_files AS t ON t.text_id = s.text_id "
            "JOIN native_meta_files AS m ON m.text_id = s.text_id "
            "WHERE s.session_date != i.session_date "
            "OR s.session_date != t.session_date "
            "OR s.session_date != m.session_date "
            "OR i.href != substr(s.member, length('ParlaMint-TR.TEI/') + 1)"
        ).fetchone()[0],
    }
    if (
        not identity_counts["tei_sessions"]
        or len(set(identity_counts.values())) != 1
        or any(identity_mismatches.values())
    ):
        raise AnchorPreparationError(
            "ParlaMint TEI/XInclude/native file identity equality failed: "
            f"counts={identity_counts!r}, mismatches={identity_mismatches!r}"
        )
    text_rows = connection.execute("SELECT COUNT(*) FROM speech_text").fetchone()[0]
    meta_rows = connection.execute("SELECT COUNT(*) FROM speech_meta").fetchone()[0]
    text_without_meta = connection.execute(
        "SELECT COUNT(*) FROM speech_text AS t LEFT JOIN speech_meta AS m "
        "ON m.speech_id = t.speech_id WHERE m.speech_id IS NULL"
    ).fetchone()[0]
    meta_without_text = connection.execute(
        "SELECT COUNT(*) FROM speech_meta AS m LEFT JOIN speech_text AS t "
        "ON t.speech_id = m.speech_id WHERE t.speech_id IS NULL"
    ).fetchone()[0]
    mismatched_text_id = connection.execute(
        "SELECT COUNT(*) FROM speech_text AS t JOIN speech_meta AS m "
        "ON m.speech_id = t.speech_id WHERE m.text_id != t.text_id"
    ).fetchone()[0]
    if (
        text_rows != contract.expected_speeches
        or meta_rows != contract.expected_speeches
        or text_without_meta
        or meta_without_text
        or mismatched_text_id
    ):
        raise AnchorPreparationError(
            "ParlaMint native text/meta one-to-one contract failed: "
            f"text={text_rows}, meta={meta_rows}, text_without_meta={text_without_meta}, "
            f"meta_without_text={meta_without_text}, mismatched_text_id={mismatched_text_id}"
        )
    raw_words = connection.execute(
        "SELECT COALESCE(SUM(raw_words), 0) FROM speech_text"
    ).fetchone()[0]
    raw_gate = (contract.raw_word_count_min, contract.raw_word_count_max)
    if (raw_gate[0] is None) != (raw_gate[1] is None):
        raise AnchorPreparationError(
            "ParlaMint whitespace-word test bounds must both be set or both be null"
        )
    if raw_gate[0] is not None and not raw_gate[0] <= raw_words <= raw_gate[1]:
        raise AnchorPreparationError(
            "ParlaMint raw whitespace-word count drift: "
            f"expected [{raw_gate[0]}, {raw_gate[1]}], got {raw_words}"
        )
    first_date, last_date = connection.execute(
        "SELECT MIN(speech_date), MAX(speech_date) FROM speech_meta"
    ).fetchone()
    if first_date != contract.first_date or last_date != contract.last_date:
        raise AnchorPreparationError(
            "ParlaMint date-span drift: "
            f"expected {contract.first_date}..{contract.last_date}, "
            f"got {first_date}..{last_date}"
        )
    quarantined = connection.execute(
        "SELECT COUNT(*) FROM speech_text WHERE quarantine_reason IS NOT NULL"
    ).fetchone()[0]
    return {
        "text_rows": text_rows,
        "native_metadata_rows": meta_rows,
        "text_without_metadata": text_without_meta,
        "metadata_without_text": meta_without_text,
        "mismatched_text_id": mismatched_text_id,
        "raw_whitespace_words": raw_words,
        "first_date": first_date,
        "last_date": last_date,
        "quarantined_speeches": quarantined,
        "cross_format_identity_counts": identity_counts,
        "cross_format_identity_mismatches": identity_mismatches,
    }


def _write_parlamint_outputs(
    connection: sqlite3.Connection,
    build: Path,
    *,
    shard_target_bytes: int,
    evidence_target_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_writer = _JsonlZstdShardWriter(build, "data", shard_target_bytes)
    rows = connection.execute(
        "SELECT speech_id, member, raw_text_sha256, clean_text_sha256, clean_text "
        "FROM speech_text WHERE quarantine_reason IS NULL "
        "ORDER BY speech_id COLLATE BINARY"
    )
    for index, row in enumerate(rows, 1):
        speech_id, member, raw_sha, clean_sha, text = row
        data_writer.write(
            {
                "id": f"parlamint-tr:v5.0:{speech_id}",
                "text": text,
                "source_id": PARLAMINT_SOURCE_ID,
                "source_revision": PARLAMINT_RELEASE,
                "provenance": {
                    "source_member": member,
                    "raw_text_sha256": raw_sha,
                    "clean_text_sha256": clean_sha,
                },
            }
        )
        digest = _text_sha256(text)
        connection.execute(
            "INSERT INTO paragraph_stats VALUES (?, ?, 1, 1) "
            "ON CONFLICT(paragraph_sha256) DO UPDATE SET "
            "occurrences = occurrences + 1, documents = documents + 1",
            (digest, text),
        )
        if index % 10_000 == 0:
            connection.commit()
    connection.commit()
    repeated_writer = _JsonlZstdShardWriter(
        build, "evidence/repeated_paragraph_candidates", evidence_target_bytes
    )
    for digest, paragraph, occurrences, documents in connection.execute(
        "SELECT paragraph_sha256, paragraph, occurrences, documents "
        "FROM paragraph_stats WHERE documents >= ? "
        "ORDER BY documents DESC, occurrences DESC, paragraph_sha256 COLLATE BINARY",
        (REPEATED_PARAGRAPH_MIN_DOCUMENTS,),
    ):
        repeated_writer.write(
            {
                "paragraph_sha256": digest,
                "occurrences": occurrences,
                "documents": documents,
                "sample": paragraph[:REPEATED_PARAGRAPH_SAMPLE_CHARS],
                "action": "report_only_no_deletion",
            }
        )
    return data_writer.finish(), repeated_writer.finish(emit_empty=True)


def prepare_parlamint_tr_v5(
    archive_tgz: str | Path,
    output_dir: str | Path,
    *,
    acquisition_receipt_path: str | Path,
    discovery: bool = False,
    count_acceptance_path: str | Path | None = None,
    shard_target_bytes: int = DEFAULT_SHARD_TARGET_BYTES,
    evidence_target_bytes: int = DEFAULT_EVIDENCE_TARGET_BYTES,
    contract: ParlaMintContract = PARLAMINT_TR_V5_CONTRACT,
) -> dict[str, Any]:
    """Prepare the pinned native Turkish ParlaMint 5.0 release offline."""

    if discovery == (count_acceptance_path is not None):
        raise AnchorPreparationError(
            "choose exactly one ParlaMint mode: discovery or accepted production"
        )
    archive_path = Path(archive_tgz)
    acquisition_receipt, acquired_assets = _load_acquisition_receipt(
        acquisition_receipt_path, source_id=PARLAMINT_SOURCE_ID, contract=contract
    )
    target = _prepare_destination(output_dir)
    build = target.build_path
    database_path = build / ".parlamint_work.sqlite3"
    snapshot_dir = build / ".verified_inputs"
    connection: sqlite3.Connection | None = None
    try:
        acquired = acquired_assets[contract.filename]
        snapshot = _snapshot_verified_archive(
            archive_path,
            snapshot_dir / contract.filename,
            expected_name=contract.filename,
            expected_size=contract.size_bytes,
            expected_sha256=acquired["sha256"],
            expected_md5=contract.md5,
        )
        verified_input = snapshot.input_record
        connection = _open_sqlite(database_path)
        _parlamint_database(connection)
        archive_counts = _process_parlamint_archive(
            connection, snapshot=snapshot, contract=contract
        )
        join_evidence = _validate_parlamint_join(connection, contract)
        data, repeated = _write_parlamint_outputs(
            connection,
            build,
            shard_target_bytes=shard_target_bytes,
            evidence_target_bytes=evidence_target_bytes,
        )
        if data["totals"]["rows"] <= 0:
            raise AnchorPreparationError("ParlaMint preparation produced no clean speeches")
        input_inventory, inventory_sha256 = _write_inventory(
            connection, build, target_bytes=evidence_target_bytes
        )
        quarantine, quarantine_reasons = _write_quarantine(
            connection, build, target_bytes=evidence_target_bytes
        )
        connection.close()
        database_path.unlink()
        shutil.rmtree(snapshot_dir)
        manifest = _finalize_manifest(
            build,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "kind": MANIFEST_KIND,
                "preparer_version": PREPARER_VERSION,
                "source_id": PARLAMINT_SOURCE_ID,
                "acquisition_receipt": acquisition_receipt,
                "source": _parlamint_source_manifest(contract),
                "inputs": [verified_input],
                "archive_member_policy": _parlamint_archive_policy(contract),
                "expected_release_totals": _parlamint_expected_totals(contract),
                "cleaning": _parlamint_cleaning_manifest(),
                "downstream_admission": _downstream_gate_declaration(),
                "artifacts": {
                    "data": data,
                    "input_inventory": input_inventory,
                    "quarantine": quarantine,
                    "repeated_paragraph_candidates": repeated,
                },
                "counts": {
                    "archive": archive_counts,
                    "join": join_evidence,
                    "quarantine_reasons": quarantine_reasons,
                },
                "raw": {
                    "archive_sha256": verified_input["sha256"],
                    "archive_md5": verified_input["md5"],
                    "member_inventory_canonical_sha256": inventory_sha256,
                    "speech_text_rows": join_evidence["text_rows"],
                    "raw_whitespace_words": join_evidence["raw_whitespace_words"],
                    "cross_format_identity_counts": join_evidence[
                        "cross_format_identity_counts"
                    ],
                },
                "clean": {
                    "documents": data["totals"]["rows"],
                    "logical_jsonl_sha256": data["logical_jsonl_sha256"],
                },
                "canonical_sha256": None,
            },
            discovery=discovery,
            count_acceptance_path=count_acceptance_path,
        )
        _validate_anchor_preparation_fd(
            target.build_fd,
            display_path=target.destination,
            contract=contract,
        )
        published_manifest = _publish_build(target, contract=contract)
        if published_manifest != manifest:
            raise AnchorPreparationError("ParlaMint manifest drift at publication")
        _cleanup_build(target)
        return manifest
    except BaseException:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        _cleanup_build(target)
        raise


__all__ = [
    "ACQUISITION_RECEIPT_KIND",
    "AnchorPreparationError",
    "COUNT_ACCEPTANCE_KIND",
    "DEFAULT_EVIDENCE_TARGET_BYTES",
    "DEFAULT_SHARD_TARGET_BYTES",
    "MOT_V1_11_CONTRACT",
    "MOT_SOURCE_ID",
    "MotAssetContract",
    "MotContract",
    "PARLAMINT_TR_V5_CONTRACT",
    "PARLAMINT_NATIVE_META_HEADER",
    "PARLAMINT_SOURCE_ID",
    "ParlaMintContract",
    "prepare_mot_v1_11",
    "prepare_parlamint_tr_v5",
    "seal_anchor_acquisition_receipt",
    "seal_anchor_count_acceptance",
    "validate_anchor_acquisition_receipt",
    "validate_anchor_count_acceptance",
    "validate_anchor_preparation",
]
