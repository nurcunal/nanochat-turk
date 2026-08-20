"""Runnable, receipt-driven CPU backend for the Turkish d32 corpus.

The production path is intentionally separate from Nanochat's GPU training
environment.  Every expensive stage is rank-addressable and idempotent, and a
full run is refused until a bounded sample has produced a resource estimate
that a human explicitly accepted.

Data flow::

    immutable source plan -> per-object GlotLID/quality/PII + MinHash sigs
      -> DataTrove LSH buckets -> priority-aware cluster merge
      -> sealed source/backend receipts

Imports of FastText, Hugging Face Hub and DataTrove are deferred so the
contracts and pure helpers remain testable in the regular Nanochat environment.
"""

from __future__ import annotations

import gzip
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import re
import resource
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.turkish_corpus import (
    BACKEND_RECEIPT_KIND,
    FINEWEB2_STRICT_SOURCE_ID,
    MACOCU_CONVERSATION_GENRES,
    MACOCU_EXPECTED_ROWS,
    MACOCU_GENERAL_GENRES,
    MACOCU_GENRES,
    MACOCU_HANDLE,
    MACOCU_MD5,
    MACOCU_SCHEMA,
    MACOCU_SIZE_BYTES,
    MACOCU_SOURCE_ID,
    MACOCU_SOURCE_URL,
    MOT_SOURCE_ID,
    PARLAMINT_SOURCE_ID,
    SOURCE_RECEIPT_KIND,
    TurkishCorpusError,
    V3_FINEWEB2_INVENTORY_SEMANTICS,
    V3_FINEWEB2_INVENTORY_SHA256,
    V3_FINEWEB2_OBJECT_COUNT,
    V3_FINEWEB2_TOTAL_BYTES,
    V3_FINEWEB2_UPSTREAM_COMMIT,
    VerifiedStagedArtifact,
    _qa_document_metrics,
    audit_policy_binding,
    audit_document,
    canonical_text_hash,
    dominant_register,
    infer_wds_bin,
    iter_input_records,
    load_corpus_policy,
    normalize_document,
    register_scores,
    select_mixture_bucket,
    source_lid_result,
    source_object_inventory,
    strict_macocu_genre,
    strict_hplt_register_scores,
    validate_backend_receipt,
    validate_corpus_policy,
    validate_source_receipt,
)


SOURCE_PLAN_KIND = "turkish_source_object_plan"
CALIBRATION_KIND = "turkish_production_backend_calibration"
OBJECT_RECEIPT_KIND = "turkish_backend_object_result"
BUCKET_RECEIPT_KIND = "turkish_datatrove_bucket_result"
CLUSTER_RECEIPT_KIND = "turkish_priority_cluster_result"
RESOURCE_REPORT_KIND = "turkish_backend_resource_projection"
RESOURCE_APPROVAL_KIND = "turkish_backend_resource_approval"
PRODUCTION_CLUSTER_LAUNCH_KIND = "turkish_packed_production_cluster_launch_receipt"
MIXTURE_QUALITY_APPROVAL_KIND = (
    "turkish_bounded_backend_sample_quality_approval"
)
MACOCU_PREPARATION_KIND = "turkish_macocu_genre_preparation"

OBJECT_SOURCE_QUALITY_SEMANTICS = (
    "maximum_finite_source_quality_field_excluding_lid_confidence_v1"
)
CLUSTER_WINNER_POLICY = (
    "minimum_source_priority_then_negative_attested_source_quality_else_zero_"
    "then_stable_id_v2"
)
CLUSTER_QUALITY_SCORE_SEMANTICS = (
    "source_quality_only_when_object_receipt_attests_else_zero_v2"
)

RESOURCE_BILLING_CONTRACT = {
    "scheduler_partition": "cpu2dq",
    "billable_cpus_per_job": 128,
    "accounting_basis": "projected_stage_wall_seconds_times_billable_cpus_per_job",
    "process_cpu_seconds_role": "efficiency_diagnostic_only",
}
RESOURCE_PEAK_DISK_MODEL = (
    "raw_largest+candidates+signatures+dups+backend_output_v2"
)

DATATROVE_VERSION = "0.10.0"
DATATROVE_REVISION = "a649de79c14a550dc90f48a15c025f2dd3fd3b57"
HPLT_MAP_SHA256 = "3619a23b7aa1a261ec1100117296801aa43feacd1e517db796830f3814a95367"
HPLT_MD5_LIST_URL = "https://data.hplt-project.org/three/sorted/tur_Latn.md5"
HPLT_MD5_LIST_SHA256 = "1c52b5e0ac204149ece78ef76e5e00ec129182b6a90864106a6a7c9731d9d675"
FINEWEB2_CONFIG_REVISION = "d0defb24f193bb9a5a11b8b14524a03c4858e1b6"
FINEWEB2_CONFIG_SHA256 = "f0ccd5fef17c5978f0c8863809dc6a3ec9bededa772f6d25bfa0a4f7f20d67c1"
FINEWEB2_CONFIG_PATH = Path(__file__).parents[1] / "configs/pretrain/fineweb2_tur_Latn.yml"

BACKEND_COLUMNS = (
    "text",
    "source_id",
    "document_id",
    "url",
    "source_lid_label",
    "source_lid_probability",
    "lid_label",
    "lid_probability",
    "lid_margin",
    "paragraph_min_probability",
    "paragraph_min_margin",
    "failed_long_paragraph_fraction",
    "dedup_cluster_id",
    "dedup_keep",
    "quality_score",
    "wds_bin",
    "web-register",
    "genre",
    "pii_replacements",
    "harmful_signal_hits",
    "quality_filter_flags",
    "formatting_changes",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_QUALITY_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?90[ .()/-]*|0)?(?:5\d{2})[ .()/-]*\d{3}[ .()/-]*\d{2}[ .()/-]*\d{2}(?!\d)"
)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}\b"
)
_IP_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_HARMFUL_SIGNAL_RE = re.compile(
    r"\b(?:öldür(?:mek|me|dü|ül)|işkence|tecavüz|çocuk\s+istismarı|nefret\s+suçu|"
    r"ırkçı(?:lık)?|intihar|kendine\s+zarar|bomba\s+yapımı|silahlı\s+saldırı)\b",
    re.IGNORECASE,
)

# Manual-review evidence is intentionally small.  These caps make every
# hash/parse operation an immutable in-memory snapshot and fail closed if a
# malformed receipt attempts to turn evidence validation into an unbounded
# read.  Cluster Parquets use a separate stable-descriptor streaming path.
_MAX_RECEIPT_EVIDENCE_BYTES = 32 * 1024 * 1024
_MAX_AUDIT_REPORT_BYTES = 64 * 1024 * 1024
_MAX_EXAMPLE_EVIDENCE_BYTES = 128 * 1024 * 1024
_MAX_CLUSTER_PARQUET_BYTES = 64 * 1024**3
_MAX_CLUSTER_PARQUET_TOTAL_BYTES = 128 * 1024**3


def _policy_sha256(policy: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TurkishCorpusError(f"{label} must be a JSON object")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TurkishCorpusError(f"{label} must be a positive integer")
    return value


def _nested_get(record: Mapping[str, Any], field: str | None, default: Any = "") -> Any:
    if not field:
        return default
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _descriptor_sha256(handle: Any) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _safe_run_artifact_path(run_root: Path, raw: Any, label: str) -> Path:
    relative_raw = str(raw or "")
    relative = Path(relative_raw)
    if (
        not relative_raw
        or urllib.parse.urlparse(relative_raw).scheme
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise TurkishCorpusError(f"{label} path is unsafe")
    root = run_root.resolve()
    unresolved = root / relative
    resolved = unresolved.resolve()
    if root not in resolved.parents:
        raise TurkishCorpusError(f"{label} path escapes the run directory")
    current = unresolved
    while current != root:
        if current.is_symlink():
            raise TurkishCorpusError(f"{label} path is symlinked")
        current = current.parent
    return resolved


def _artifact_content_contract(
    record: Mapping[str, Any],
    label: str,
    *,
    max_bytes: int | None = None,
) -> tuple[int, str]:
    size = record.get("size_bytes")
    digest = str(record.get("sha256") or "")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or (max_bytes is not None and size > max_bytes)
        or not _SHA256_RE.fullmatch(digest)
    ):
        raise TurkishCorpusError(f"{label} content contract is invalid")
    return size, digest


def _assert_descriptor_path_binding(
    path: Path, descriptor: int, label: str
) -> None:
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise TurkishCorpusError(f"{label} path changed during consumption") from exc
    descriptor_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino)
        != (descriptor_stat.st_dev, descriptor_stat.st_ino)
    ):
        raise TurkishCorpusError(f"{label} path changed during consumption")


@contextmanager
def _open_verified_regular_artifact(
    path: Path,
    *,
    label: str,
    expected_size: int,
    expected_sha256: str,
) -> Iterator[Any]:
    """Verify and consume one path through a single stable descriptor.

    The same ``O_NOFOLLOW`` descriptor supplies the pre-consumption digest,
    the consumer's bytes, and the post-consumption digest.  The final inode
    binding additionally rejects rename/path-substitution races.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TurkishCorpusError(f"{label} is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise TurkishCorpusError(f"{label} size drift")
        _assert_descriptor_path_binding(path, descriptor, label)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            if _descriptor_sha256(handle) != expected_sha256:
                raise TurkishCorpusError(f"{label} hash drift")
            _assert_descriptor_path_binding(path, descriptor, label)
            if _stat_fingerprint(os.fstat(descriptor)) != _stat_fingerprint(before):
                raise TurkishCorpusError(f"{label} changed during verification")
            handle.seek(0)
            try:
                yield handle
            finally:
                if _descriptor_sha256(handle) != expected_sha256:
                    raise TurkishCorpusError(f"{label} changed during consumption")
                _assert_descriptor_path_binding(path, descriptor, label)
                after = os.fstat(descriptor)
                if _stat_fingerprint(after) != _stat_fingerprint(before):
                    raise TurkishCorpusError(f"{label} changed during consumption")
    finally:
        os.close(descriptor)


@contextmanager
def _open_verified_run_artifact(
    run_root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
    max_bytes: int | None = None,
) -> Iterator[tuple[Path, Any]]:
    size, digest = _artifact_content_contract(record, label, max_bytes=max_bytes)
    path = _safe_run_artifact_path(run_root, record.get("path"), label)
    with _open_verified_regular_artifact(
        path,
        label=label,
        expected_size=size,
        expected_sha256=digest,
    ) as handle:
        yield path, handle


def _descriptor_file_record(
    path: Path, *, root: Path | None = None, rows: int | None = None
) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TurkishCorpusError(f"cannot record unsafe artifact: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TurkishCorpusError(f"artifact is not a regular file: {path}")
        _assert_descriptor_path_binding(path, descriptor, str(path))
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = _descriptor_sha256(handle)
            _assert_descriptor_path_binding(path, descriptor, str(path))
        after = os.fstat(descriptor)
        if _stat_fingerprint(after) != _stat_fingerprint(before):
            raise TurkishCorpusError(f"artifact changed while recorded: {path}")
        _assert_descriptor_path_binding(path, descriptor, str(path))
    finally:
        os.close(descriptor)
    relative = path.relative_to(root).as_posix() if root is not None else path.name
    result: dict[str, Any] = {
        "path": relative,
        "size_bytes": before.st_size,
        "sha256": digest,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _parquet_file_record(
    path: Path,
    *,
    root: Path,
    expected_rows: int,
    expected_columns: set[str] | None = None,
) -> dict[str, Any]:
    """Record a newly written Parquet from the descriptor used for metadata."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TurkishCorpusError(f"cannot record unsafe Parquet: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TurkishCorpusError(f"Parquet is not a regular file: {path}")
        _assert_descriptor_path_binding(path, descriptor, str(path))
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            digest = _descriptor_sha256(handle)
            _assert_descriptor_path_binding(path, descriptor, str(path))
            handle.seek(0)
            parquet = pq.ParquetFile(handle)
            if parquet.metadata.num_rows != expected_rows:
                raise TurkishCorpusError(f"Parquet row-count drift: {path}")
            if (
                expected_columns is not None
                and set(parquet.schema_arrow.names) != expected_columns
            ):
                raise TurkishCorpusError(f"Parquet schema drift: {path}")
            if _descriptor_sha256(handle) != digest:
                raise TurkishCorpusError(f"Parquet changed while recorded: {path}")
            _assert_descriptor_path_binding(path, descriptor, str(path))
        after = os.fstat(descriptor)
        if _stat_fingerprint(after) != _stat_fingerprint(before):
            raise TurkishCorpusError(f"Parquet changed while recorded: {path}")
        _assert_descriptor_path_binding(path, descriptor, str(path))
    finally:
        os.close(descriptor)
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": before.st_size,
        "sha256": digest,
        "rows": expected_rows,
    }


def _file_record(path: Path, *, root: Path | None = None, rows: int | None = None) -> dict[str, Any]:
    return _descriptor_file_record(path, root=root, rows=rows)


def _file_sha256_md5(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()  # noqa: S324 - official upstream integrity checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def _verify_local_macocu_upstream(path: str | Path) -> dict[str, Any]:
    """Seal a stable local official gzip identity before preparation starts."""

    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise TurkishCorpusError(
            f"local MaCoCu upstream is missing, symlinked, or unsafe: {source}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TurkishCorpusError("local MaCoCu upstream must be a regular file")
        if before.st_size != MACOCU_SIZE_BYTES:
            raise TurkishCorpusError(
                "local MaCoCu upstream size drift: "
                f"expected {MACOCU_SIZE_BYTES}, got {before.st_size}"
            )
        _assert_descriptor_path_binding(source, descriptor, "local MaCoCu upstream")
        sha256 = hashlib.sha256()
        md5 = hashlib.md5()  # noqa: S324 - official upstream integrity checksum
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
                md5.update(chunk)
        after = os.fstat(descriptor)
        _assert_descriptor_path_binding(source, descriptor, "local MaCoCu upstream")
        if _stat_fingerprint(after) != _stat_fingerprint(before):
            raise TurkishCorpusError(
                "local MaCoCu upstream changed during checksum verification"
            )
        observed_md5 = md5.hexdigest()
        if observed_md5 != MACOCU_MD5:
            raise TurkishCorpusError("local MaCoCu upstream MD5 drift")
        return {
            "path": source.resolve(strict=True),
            "size_bytes": before.st_size,
            "md5": observed_md5,
            "sha256": sha256.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _elapsed(start_wall: float, start_cpu: float) -> dict[str, float]:
    return {
        "wall_seconds": max(0.0, time.monotonic() - start_wall),
        "cpu_seconds": max(0.0, time.process_time() - start_cpu),
    }


def _peak_rss_bytes() -> int:
    """Return process peak RSS using the platform-specific ru_maxrss unit."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _http_bytes(uri: str, *, request_get: Any = requests.get) -> bytes:
    response = request_get(uri, timeout=120)
    response.raise_for_status()
    return bytes(response.content)


def _head_size(uri: str, *, request_head: Any = requests.head) -> int:
    response = request_head(uri, allow_redirects=True, timeout=120)
    response.raise_for_status()
    raw = response.headers.get("Content-Length")
    try:
        size = int(raw)
    except (TypeError, ValueError) as exc:
        raise TurkishCorpusError(f"immutable object has no usable Content-Length: {uri}") from exc
    return _require_positive_int(size, f"Content-Length for {uri}")


def _parse_hplt_objects(
    source: Mapping[str, Any],
    *,
    request_get: Any = requests.get,
    request_head: Any = requests.head,
) -> list[dict[str, Any]]:
    configured_bins = source.get("selected_wds_bins", [8, 9, 10])
    if (
        not isinstance(configured_bins, list)
        or not configured_bins
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in configured_bins
        )
        or len(configured_bins) != len(set(configured_bins))
        or not set(configured_bins) <= {8, 9, 10}
    ):
        raise TurkishCorpusError("HPLT selected_wds_bins contract is malformed")
    selected_bins = set(configured_bins)
    map_bytes = _http_bytes(str(source["source_url"]), request_get=request_get)
    if hashlib.sha256(map_bytes).hexdigest() != HPLT_MAP_SHA256:
        raise TurkishCorpusError("HPLT tur_Latn map hash drift")
    md5_bytes = _http_bytes(HPLT_MD5_LIST_URL, request_get=request_get)
    if hashlib.sha256(md5_bytes).hexdigest() != HPLT_MD5_LIST_SHA256:
        raise TurkishCorpusError("HPLT tur_Latn checksum-list hash drift")
    md5_by_name: dict[str, str] = {}
    for raw in md5_bytes.decode("utf-8").splitlines():
        fields = raw.split()
        if len(fields) != 2 or not _MD5_RE.fullmatch(fields[0]):
            raise TurkishCorpusError("malformed HPLT MD5 checksum list")
        md5_by_name[Path(fields[1]).name] = fields[0]
    objects: list[dict[str, Any]] = []
    for raw in map_bytes.decode("utf-8").splitlines():
        uri = raw.strip()
        if not uri:
            continue
        name = Path(urllib.parse.urlparse(uri).path).name
        match = re.fullmatch(r"(\d+)_\d+\.jsonl\.zst", name)
        if match is None:
            raise TurkishCorpusError(f"unexpected HPLT map object: {uri}")
        wds_bin = int(match.group(1))
        if wds_bin not in selected_bins:
            continue
        expected_md5 = md5_by_name.get(name)
        if expected_md5 is None:
            raise TurkishCorpusError(f"HPLT map object has no checksum: {name}")
        objects.append(
            {
                "source_id": source["id"],
                "uri": uri,
                "size_bytes": _head_size(uri, request_head=request_head),
                "expected_checksums": [{"algorithm": "md5", "value": expected_md5}],
                "adapter": source["adapter"],
                "wds_bin": wds_bin,
            }
        )
    if not objects:
        raise TurkishCorpusError(
            f"HPLT plan resolved no selected WDS objects: {sorted(selected_bins)}"
        )
    return sorted(objects, key=lambda item: item["uri"])


def _fineweb2_inventory_projection(
    objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the receipt-bound immutable raw inventory for strict FineWeb v3."""

    return [
        {
            "uri": str(item["uri"]),
            "size_bytes": int(item["size_bytes"]),
            "expected_checksums": list(item["expected_checksums"]),
        }
        for item in sorted(objects, key=lambda item: str(item["uri"]))
    ]


def _strict_fineweb_derivation(
    source: Mapping[str, Any],
    policy: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal raw acquisition separately from strict candidate admission."""

    if source.get("id") != FINEWEB2_STRICT_SOURCE_ID:
        raise TurkishCorpusError("strict FineWeb derivation received another source")
    configured = _require_mapping(source.get("derivation"), "strict FineWeb derivation")
    projection = _fineweb2_inventory_projection(objects)
    inventory_sha256 = hashlib.sha256(
        canonical_json(projection).encode("utf-8")
    ).hexdigest()
    total_bytes = sum(item["size_bytes"] for item in projection)
    processing_sha256 = production_processing_binding(policy)["binding_sha256"]
    audit_sha256 = audit_policy_binding(policy["content_policy"])["binding_sha256"]
    expected = {
        "expected_object_count": V3_FINEWEB2_OBJECT_COUNT,
        "expected_total_bytes": V3_FINEWEB2_TOTAL_BYTES,
        "expected_inventory_sha256": V3_FINEWEB2_INVENTORY_SHA256,
        "inventory_hash_semantics": V3_FINEWEB2_INVENTORY_SEMANTICS,
        "processing_binding_sha256": processing_sha256,
        "audit_policy_binding_sha256": audit_sha256,
    }
    if any(configured.get(key) != value for key, value in expected.items()):
        raise TurkishCorpusError("strict FineWeb configured derivation hash/size drift")
    if (
        len(projection) != V3_FINEWEB2_OBJECT_COUNT
        or total_bytes != V3_FINEWEB2_TOTAL_BYTES
        or inventory_sha256 != V3_FINEWEB2_INVENTORY_SHA256
    ):
        raise TurkishCorpusError(
            "strict FineWeb must resolve the complete frozen 30-object inventory"
        )
    if (
        source.get("resolved_revision") != V3_FINEWEB2_UPSTREAM_COMMIT
        or configured.get("raw_fallback_allowed") is not False
    ):
        raise TurkishCorpusError("strict FineWeb upstream/fallback contract drift")
    return {
        "contract": dict(configured),
        "resolved_inventory": {
            "object_count": len(projection),
            "total_bytes": total_bytes,
            "sha256": inventory_sha256,
            "hash_semantics": V3_FINEWEB2_INVENTORY_SEMANTICS,
        },
        "admission": {
            "candidate_source_id": FINEWEB2_STRICT_SOURCE_ID,
            "raw_source_id": "fineweb2_tr",
            "only_passing_rows_enter_candidates": True,
            "direct_raw_fallback": False,
            "processing_binding_sha256": processing_sha256,
            "audit_policy_binding_sha256": audit_sha256,
        },
    }


def _hub_prefix(source: Mapping[str, Any]) -> str:
    parsed = urllib.parse.urlparse(str(source["source_url"]))
    marker = f"/tree/{source['resolved_revision']}/"
    if marker not in parsed.path:
        raise TurkishCorpusError(f"{source['id']}: Hub URL is not bound to configured commit")
    return urllib.parse.unquote(parsed.path.split(marker, 1)[1]).strip("/")


def _parse_hub_objects(source: Mapping[str, Any], *, api: Any | None = None) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_url
    except ImportError as exc:  # pragma: no cover - production environment only
        raise TurkishCorpusError("huggingface-hub is required to resolve source objects") from exc
    hub = api or HfApi()
    prefix = _hub_prefix(source)
    entries = hub.list_repo_tree(
        repo_id=source["repo_id"],
        path_in_repo=prefix,
        recursive=True,
        expand=True,
        revision=source["resolved_revision"],
        repo_type="dataset",
    )
    expected_suffix = ".parquet" if source["adapter"]["format"] == "parquet" else ""
    objects: list[dict[str, Any]] = []
    for entry in entries:
        path = getattr(entry, "path", None)
        if not isinstance(path, str) or (expected_suffix and not path.endswith(expected_suffix)):
            continue
        size = getattr(entry, "size", None)
        lfs = getattr(entry, "lfs", None)
        if isinstance(lfs, Mapping):
            sha256 = lfs.get("sha256")
        else:
            sha256 = getattr(lfs, "sha256", None)
        if not _SHA256_RE.fullmatch(str(sha256 or "")):
            raise TurkishCorpusError(
                f"{source['id']}: {path} has no immutable LFS SHA-256; resolver fails closed"
            )
        objects.append(
            {
                "source_id": source["id"],
                "uri": hf_hub_url(
                    source["repo_id"],
                    path,
                    repo_type="dataset",
                    revision=source["resolved_revision"],
                ),
                "size_bytes": _require_positive_int(size, f"Hub size for {path}"),
                "expected_checksums": [{"algorithm": "sha256", "value": sha256}],
                "adapter": source["adapter"],
                "hub_path": path,
                "wds_bin": None,
            }
        )
    if not objects:
        raise TurkishCorpusError(
            f"{source['id']}: immutable Hub tree resolved no {expected_suffix or 'data'} objects"
        )
    return sorted(objects, key=lambda item: item["uri"])


def _macocu_source(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [source for source in policy["sources"] if source["id"] == MACOCU_SOURCE_ID]
    if len(matches) != 1:
        raise TurkishCorpusError("policy must contain exactly one MaCoCu-Genre source")
    return matches[0]


def _macocu_source_contract_sha256(policy: Mapping[str, Any]) -> str:
    contract = {
        "source": dict(_macocu_source(policy)),
        "schema": list(MACOCU_SCHEMA),
        "genre_labels": sorted(MACOCU_GENRES),
    }
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def validate_macocu_preparation_manifest(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    root: str | Path,
    *,
    verify_files: bool = True,
) -> None:
    """Validate the official gzip -> deterministic zstd-shard preparation."""

    verify_manifest_hash(manifest)
    if manifest.get("schema_version") != "1.0" or manifest.get("kind") != MACOCU_PREPARATION_KIND:
        raise TurkishCorpusError("unexpected MaCoCu preparation manifest")
    if manifest.get("source_contract_sha256") != _macocu_source_contract_sha256(policy):
        raise TurkishCorpusError("MaCoCu preparation is bound to another source contract")
    upstream = _require_mapping(manifest.get("upstream"), "macocu.upstream")
    expected_upstream = {
        "source_id": MACOCU_SOURCE_ID,
        "persistent_handle": MACOCU_HANDLE,
        "uri": MACOCU_SOURCE_URL,
        "size_bytes": MACOCU_SIZE_BYTES,
        "md5": MACOCU_MD5,
        "rows": MACOCU_EXPECTED_ROWS,
    }
    for key, expected in expected_upstream.items():
        if upstream.get(key) != expected:
            raise TurkishCorpusError(f"MaCoCu upstream {key} drift")
    if not _SHA256_RE.fullmatch(str(upstream.get("sha256", ""))):
        raise TurkishCorpusError("MaCoCu upstream SHA-256 is missing")
    if manifest.get("schema") != list(MACOCU_SCHEMA):
        raise TurkishCorpusError("MaCoCu prepared schema drift")
    if manifest.get("genre_labels") != sorted(MACOCU_GENRES):
        raise TurkishCorpusError("MaCoCu genre vocabulary drift")
    preparation = _require_mapping(manifest.get("preparation"), "macocu.preparation")
    if preparation.get("canonicalization") != "canonical_json_sorted_utf8_one_lf_v1":
        raise TurkishCorpusError("MaCoCu canonicalization drift")
    if preparation.get("compression") != "zstd" or preparation.get("format") != "jsonl.zst":
        raise TurkishCorpusError("MaCoCu prepared format drift")
    _require_positive_int(
        preparation.get("target_uncompressed_bytes"),
        "MaCoCu target_uncompressed_bytes",
    )

    base = Path(root).resolve()
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise TurkishCorpusError("MaCoCu preparation has no shards")
    expected_row_start = 0
    genre_totals: Counter[str] = Counter()
    for index, raw in enumerate(shards):
        shard = _require_mapping(raw, f"macocu.shards[{index}]")
        expected_path = f"shards/part-{index:05d}.jsonl.zst"
        if shard.get("path") != expected_path:
            raise TurkishCorpusError("MaCoCu shard order/path drift")
        rows = _require_positive_int(shard.get("rows"), "MaCoCu shard rows")
        size = _require_positive_int(shard.get("size_bytes"), "MaCoCu shard size")
        _require_positive_int(
            shard.get("uncompressed_bytes"), "MaCoCu shard uncompressed bytes"
        )
        if shard.get("row_start") != expected_row_start or shard.get("row_end_exclusive") != expected_row_start + rows:
            raise TurkishCorpusError("MaCoCu shard row ranges are not contiguous")
        expected_row_start += rows
        sha256 = str(shard.get("sha256", ""))
        if not _SHA256_RE.fullmatch(sha256):
            raise TurkishCorpusError("MaCoCu shard SHA-256 is missing")
        counts = _require_mapping(shard.get("genre_counts"), "MaCoCu shard genre_counts")
        if set(counts) - MACOCU_GENRES or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in counts.values()
        ) or sum(counts.values()) != rows:
            raise TurkishCorpusError("MaCoCu shard genre counts drift")
        genre_totals.update(counts)
        if verify_files:
            path = (base / expected_path).resolve()
            if base not in path.parents or path.is_symlink() or not path.is_file():
                raise TurkishCorpusError("MaCoCu shard path is unsafe or missing")
            if path.stat().st_size != size or file_sha256(path) != sha256:
                raise TurkishCorpusError("MaCoCu prepared shard bytes drift")

    totals = _require_mapping(manifest.get("totals"), "macocu.totals")
    expected_totals = {
        "shards": len(shards),
        "rows": sum(item["rows"] for item in shards),
        "size_bytes": sum(item["size_bytes"] for item in shards),
        "uncompressed_bytes": sum(item["uncompressed_bytes"] for item in shards),
        "genre_counts": dict(sorted(genre_totals.items())),
    }
    if dict(totals) != expected_totals or expected_totals["rows"] != MACOCU_EXPECTED_ROWS:
        raise TurkishCorpusError("MaCoCu preparation totals drift")
    if verify_files:
        upstream_path = (base / str(upstream.get("path", ""))).resolve()
        if base not in upstream_path.parents or upstream_path.is_symlink() or not upstream_path.is_file():
            raise TurkishCorpusError("MaCoCu upstream gzip is unsafe or missing")
        if upstream_path.stat().st_size != MACOCU_SIZE_BYTES:
            raise TurkishCorpusError("MaCoCu retained gzip size drift")
        observed_sha256, observed_md5 = _file_sha256_md5(upstream_path)
        if observed_sha256 != upstream["sha256"] or observed_md5 != MACOCU_MD5:
            raise TurkishCorpusError("MaCoCu retained gzip checksum drift")


def prepare_macocu_genre(
    policy: Mapping[str, Any],
    output_dir: str | Path,
    *,
    target_uncompressed_bytes: int = 512 * 1024 * 1024,
    request_get: Any = requests.get,
    upstream_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify MaCoCu once and atomically create deterministic shards.

    ``upstream_path`` reuses an existing official gzip.  It is accepted only
    after stable-descriptor regular-file, size, MD5, and SHA-256 verification;
    the staged copy must then reproduce both checksums before any gzip row is
    parsed.  Omitting it preserves the pinned HTTPS acquisition path.
    """

    validate_corpus_policy(policy)
    _macocu_source(policy)
    target_uncompressed_bytes = _require_positive_int(
        target_uncompressed_bytes, "target_uncompressed_bytes"
    )
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = load_json_strict(manifest_path)
        validate_macocu_preparation_manifest(manifest, policy, destination)
        if manifest["preparation"]["target_uncompressed_bytes"] != target_uncompressed_bytes:
            raise TurkishCorpusError("existing MaCoCu preparation uses another shard target")
        return manifest
    if destination.exists():
        if any(destination.iterdir()):
            raise TurkishCorpusError(
                f"incomplete MaCoCu preparation requires audit: {destination}"
            )
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    build = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.build-", dir=destination.parent)
    )
    try:
        upstream_dir = build / "upstream"
        shards_dir = build / "shards"
        upstream_dir.mkdir()
        shards_dir.mkdir()
        staged_upstream_path = upstream_dir / "MaCoCu-Genre.tr.jsonl.gz"
        if upstream_path is None:
            source_object = {
                "uri": MACOCU_SOURCE_URL,
                "size_bytes": MACOCU_SIZE_BYTES,
                "expected_checksums": [
                    {"algorithm": "md5", "value": MACOCU_MD5}
                ],
            }
        else:
            local_identity = _verify_local_macocu_upstream(upstream_path)
            source_object = {
                "uri": local_identity["path"].as_uri(),
                "size_bytes": local_identity["size_bytes"],
                "expected_checksums": [
                    {"algorithm": "md5", "value": local_identity["md5"]},
                    {"algorithm": "sha256", "value": local_identity["sha256"]},
                ],
            }
        staged, staged_artifact = _stage_source_object(
            source_object,
            staged_upstream_path,
            request_get=request_get,
        )

        shards: list[dict[str, Any]] = []
        total_genres: Counter[str] = Counter()
        total_rows = 0
        total_uncompressed = 0
        stream: pa.NativeFile | None = None
        shard_path: Path | None = None
        shard_rows = 0
        shard_bytes = 0
        shard_genres: Counter[str] = Counter()
        shard_row_start = 0

        def close_shard() -> None:
            nonlocal stream, shard_path, shard_rows, shard_bytes, shard_genres, shard_row_start
            if stream is None or shard_path is None:
                return
            stream.close()
            shards.append(
                {
                    "path": shard_path.relative_to(build).as_posix(),
                    "size_bytes": shard_path.stat().st_size,
                    "sha256": file_sha256(shard_path),
                    "rows": shard_rows,
                    "row_start": shard_row_start,
                    "row_end_exclusive": shard_row_start + shard_rows,
                    "uncompressed_bytes": shard_bytes,
                    "genre_counts": dict(sorted(shard_genres.items())),
                }
            )
            shard_row_start += shard_rows
            stream = None
            shard_path = None
            shard_rows = 0
            shard_bytes = 0
            shard_genres = Counter()

        staged_reader = staged_artifact.open()
        with gzip.GzipFile(fileobj=staged_reader, mode="rb") as source_stream:
            for line_number, raw in enumerate(source_stream, 1):
                if not raw.strip():
                    raise TurkishCorpusError(f"MaCoCu row {line_number} is blank")
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise TurkishCorpusError(f"MaCoCu row {line_number} is invalid UTF-8 JSON") from exc
                if not isinstance(record, dict) or set(record) != set(MACOCU_SCHEMA):
                    raise TurkishCorpusError(f"MaCoCu row {line_number} schema drift")
                if not all(isinstance(record[field], str) for field in MACOCU_SCHEMA if field != "title"):
                    raise TurkishCorpusError(f"MaCoCu row {line_number} contains a non-string field")
                if record["title"] is not None and not isinstance(record["title"], str):
                    raise TurkishCorpusError(f"MaCoCu row {line_number} title type drift")
                genre = strict_macocu_genre(record)
                # canonical_json already emits exactly one terminal LF. Keep
                # that framing byte-for-byte so adjacent rows remain JSONL
                # without introducing blank records between documents.
                encoded = canonical_json(record).encode("utf-8")
                if stream is not None and shard_rows and shard_bytes + len(encoded) > target_uncompressed_bytes:
                    close_shard()
                if stream is None:
                    shard_path = shards_dir / f"part-{len(shards):05d}.jsonl.zst"
                    stream = pa.output_stream(str(shard_path), compression="zstd")
                stream.write(encoded)
                shard_rows += 1
                shard_bytes += len(encoded)
                shard_genres[genre] += 1
                total_genres[genre] += 1
                total_rows += 1
                total_uncompressed += len(encoded)
        staged_reader.close()
        staged_artifact.close()
        close_shard()
        if total_rows != MACOCU_EXPECTED_ROWS:
            raise TurkishCorpusError(
                f"MaCoCu row-count drift: expected {MACOCU_EXPECTED_ROWS}, got {total_rows}"
            )
        manifest = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": MACOCU_PREPARATION_KIND,
                "source_contract_sha256": _macocu_source_contract_sha256(policy),
                "upstream": {
                    "source_id": MACOCU_SOURCE_ID,
                    "persistent_handle": MACOCU_HANDLE,
                    "uri": MACOCU_SOURCE_URL,
                    "path": staged_upstream_path.relative_to(build).as_posix(),
                    "size_bytes": staged["size_bytes"],
                    "md5": MACOCU_MD5,
                    "sha256": staged["sha256"],
                    "rows": MACOCU_EXPECTED_ROWS,
                },
                "schema": list(MACOCU_SCHEMA),
                "genre_labels": sorted(MACOCU_GENRES),
                "preparation": {
                    "format": "jsonl.zst",
                    "compression": "zstd",
                    "canonicalization": "canonical_json_sorted_utf8_one_lf_v1",
                    "target_uncompressed_bytes": target_uncompressed_bytes,
                    "pyarrow_version": pa.__version__,
                },
                "shards": shards,
                "totals": {
                    "shards": len(shards),
                    "rows": total_rows,
                    "size_bytes": sum(item["size_bytes"] for item in shards),
                    "uncompressed_bytes": total_uncompressed,
                    "genre_counts": dict(sorted(total_genres.items())),
                },
                "canonical_sha256": None,
            }
        )
        write_json_atomic(build / "manifest.json", manifest)
        # The download path already streamed both official MD5 and observed
        # SHA-256, and every shard hash was computed after close. Avoid a
        # second full read of tens of GB before the atomic directory publish.
        validate_macocu_preparation_manifest(
            manifest, policy, build, verify_files=False
        )
        os.replace(build, destination)
        return manifest
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise


def _parse_macocu_objects(
    source: Mapping[str, Any],
    policy: Mapping[str, Any],
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = load_json_strict(path)
    root = path.parent
    validate_macocu_preparation_manifest(manifest, policy, root)
    objects = []
    for shard in manifest["shards"]:
        shard_path = root / shard["path"]
        objects.append(
            {
                "source_id": source["id"],
                "uri": shard_path.resolve().as_uri(),
                "size_bytes": shard["size_bytes"],
                "expected_checksums": [
                    {"algorithm": "sha256", "value": shard["sha256"]}
                ],
                "adapter": source["adapter"],
                "wds_bin": None,
                "genre_counts": shard["genre_counts"],
                "preparation_manifest_sha256": manifest["canonical_sha256"],
            }
        )
    provenance = {
        "manifest_uri": path.resolve().as_uri(),
        "manifest_sha256": manifest["canonical_sha256"],
        "upstream": manifest["upstream"],
        "prepared_totals": manifest["totals"],
    }
    return objects, provenance


def _parse_anchor_objects(
    source: Mapping[str, Any], manifest_path: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve a fully accepted native-text anchor through its sealed manifest."""

    from nanochat.turkish_anchor_preparation import validate_anchor_preparation

    supplied = Path(manifest_path).expanduser()
    if supplied.is_symlink():
        raise TurkishCorpusError("anchor preparation path must not be a symlink")
    if supplied.is_dir():
        root = supplied
    else:
        if supplied.name != "manifest.json" or not supplied.is_file():
            raise TurkishCorpusError(
                "anchor preparation must be an existing directory or manifest.json"
            )
        root = supplied.parent
    try:
        manifest = validate_anchor_preparation(root, verify_files=True)
    except (OSError, ValueError) as exc:
        raise TurkishCorpusError(
            f"invalid sealed anchor preparation for {source['id']}"
        ) from exc
    manifest_file = root / "manifest.json"
    if (
        manifest.get("source_id") != source["id"]
        or manifest.get("canonical_sha256") is None
        or manifest.get("production_acceptance", {}).get("stage")
        != "accepted_production"
        or manifest.get("production_acceptance", {}).get(
            "eligible_for_production"
        )
        is not True
    ):
        raise TurkishCorpusError(
            f"{source['id']} requires an accepted-production preparation manifest"
        )
    configured = _require_mapping(
        source.get("prepared_source"), f"{source['id']}.prepared_source"
    )
    if (
        configured.get("manifest_kind") != manifest.get("kind")
        or configured.get("required_source_id") != manifest.get("source_id")
        or configured.get("required_preparer_version")
        != manifest.get("preparer_version")
        or configured.get("required_production_acceptance_stage")
        != manifest["production_acceptance"]["stage"]
        or configured.get("downstream_turkish_no_code_audit_required") is not True
    ):
        raise TurkishCorpusError(f"{source['id']} prepared-source contract drift")
    data = _require_mapping(
        _require_mapping(manifest.get("artifacts"), "anchor artifacts").get("data"),
        "anchor data artifact",
    )
    objects: list[dict[str, Any]] = []
    for shard in data.get("shards", []):
        shard_path = root / str(shard.get("path", ""))
        objects.append(
            {
                "source_id": source["id"],
                "uri": shard_path.resolve().as_uri(),
                "size_bytes": shard.get("size_bytes"),
                "expected_checksums": [
                    {"algorithm": "sha256", "value": shard.get("sha256")}
                ],
                "adapter": source["adapter"],
                "wds_bin": None,
                "preparation_manifest_sha256": manifest["canonical_sha256"],
            }
        )
    if not objects:
        raise TurkishCorpusError(f"{source['id']} preparation contains no data shards")
    acceptance = _require_mapping(
        manifest["production_acceptance"].get("receipt"),
        "anchor production acceptance receipt",
    )
    provenance = {
        "manifest_uri": manifest_file.resolve().as_uri(),
        "manifest_sha256": manifest["canonical_sha256"],
        "source_id": manifest["source_id"],
        "preparer_version": manifest["preparer_version"],
        "production_acceptance": {
            "stage": "accepted_production",
            "receipt_sha256": acceptance.get("canonical_sha256"),
        },
        "acquisition_receipt_sha256": manifest["acquisition_receipt"].get(
            "canonical_sha256"
        ),
        "clean": manifest["clean"],
        "data_artifact": {
            "logical_jsonl_sha256": data.get("logical_jsonl_sha256"),
            "totals": data.get("totals"),
        },
        "downstream_admission": {
            "preparer_automatically_admits_training": False,
            "backend_turkish_no_code_audit_required": True,
        },
    }
    return sorted(objects, key=lambda item: item["uri"]), provenance


def validate_source_plan(plan: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    verify_manifest_hash(plan)
    if plan.get("schema_version") != "1.0" or plan.get("kind") != SOURCE_PLAN_KIND:
        raise TurkishCorpusError("unexpected source-plan kind/version")
    if plan.get("policy_sha256") != _policy_sha256(policy):
        raise TurkishCorpusError("source plan is bound to another policy")
    objects = plan.get("objects")
    if not isinstance(objects, list) or not objects:
        raise TurkishCorpusError("source plan has no objects")
    expected_sources = {item["id"] for item in policy["sources"]}
    derived_sources = _require_mapping(
        plan.get("derived_sources", {}), "derived_sources"
    )
    expected_derived = {
        source_id
        for source_id in (
            MACOCU_SOURCE_ID,
            FINEWEB2_STRICT_SOURCE_ID,
            MOT_SOURCE_ID,
            PARLAMINT_SOURCE_ID,
        )
        if source_id in expected_sources
    }
    if set(derived_sources) != expected_derived:
        raise TurkishCorpusError("source-plan derived-source inventory drift")
    macocu_manifest_sha256 = None
    macocu_manifest_objects: dict[str, Mapping[str, Any]] = {}
    if MACOCU_SOURCE_ID in derived_sources:
        macocu_derived = _require_mapping(
            derived_sources[MACOCU_SOURCE_ID], "derived_sources.macocu_genre_tr"
        )
        macocu_manifest_sha256 = str(macocu_derived.get("manifest_sha256", ""))
        if not _SHA256_RE.fullmatch(macocu_manifest_sha256):
            raise TurkishCorpusError("MaCoCu preparation manifest hash is missing")
        manifest_uri = str(macocu_derived.get("manifest_uri", ""))
        if urllib.parse.urlparse(manifest_uri).scheme != "file":
            raise TurkishCorpusError("MaCoCu preparation manifest must be a local sealed file")
        manifest_path = Path(
            urllib.parse.unquote(urllib.parse.urlparse(manifest_uri).path)
        )
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise TurkishCorpusError("MaCoCu preparation manifest is unsafe or missing")
        prepared_manifest = load_json_strict(manifest_path)
        verify_manifest_hash(prepared_manifest)
        if prepared_manifest.get("canonical_sha256") != macocu_manifest_sha256:
            raise TurkishCorpusError("MaCoCu preparation manifest bytes/hash drift")
        for shard in prepared_manifest.get("shards", []):
            shard_uri = (manifest_path.parent / str(shard.get("path", ""))).resolve().as_uri()
            macocu_manifest_objects[shard_uri] = shard
        if len(macocu_manifest_objects) != len(prepared_manifest.get("shards", [])):
            raise TurkishCorpusError("MaCoCu preparation shard inventory is duplicated")
        upstream = _require_mapping(macocu_derived.get("upstream"), "MaCoCu upstream")
        if upstream.get("uri") != MACOCU_SOURCE_URL or upstream.get("md5") != MACOCU_MD5:
            raise TurkishCorpusError("MaCoCu plan upstream binding drift")
    strict_derivation = None
    if FINEWEB2_STRICT_SOURCE_ID in derived_sources:
        strict_source = next(
            source
            for source in policy["sources"]
            if source["id"] == FINEWEB2_STRICT_SOURCE_ID
        )
        strict_derivation = _require_mapping(
            derived_sources[FINEWEB2_STRICT_SOURCE_ID],
            "derived_sources.fineweb2_strict_tr_v3",
        )
        if strict_derivation.get("contract") != strict_source.get("derivation"):
            raise TurkishCorpusError("strict FineWeb plan derivation contract drift")
        admission = _require_mapping(
            strict_derivation.get("admission"), "strict FineWeb admission"
        )
        expected_processing = production_processing_binding(policy)["binding_sha256"]
        expected_audit = audit_policy_binding(policy["content_policy"])[
            "binding_sha256"
        ]
        if admission != {
            "candidate_source_id": FINEWEB2_STRICT_SOURCE_ID,
            "raw_source_id": "fineweb2_tr",
            "only_passing_rows_enter_candidates": True,
            "direct_raw_fallback": False,
            "processing_binding_sha256": expected_processing,
            "audit_policy_binding_sha256": expected_audit,
        }:
            raise TurkishCorpusError("strict FineWeb admission contract drift")
    anchor_manifest_objects: dict[str, dict[str, Mapping[str, Any]]] = {}
    for anchor_id in (MOT_SOURCE_ID, PARLAMINT_SOURCE_ID):
        if anchor_id not in derived_sources:
            continue
        derived = _require_mapping(
            derived_sources[anchor_id], f"derived_sources.{anchor_id}"
        )
        manifest_uri = str(derived.get("manifest_uri", ""))
        parsed_manifest = urllib.parse.urlparse(manifest_uri)
        if parsed_manifest.scheme != "file":
            raise TurkishCorpusError("anchor preparation manifest must be local")
        manifest_path = Path(urllib.parse.unquote(parsed_manifest.path))
        source = next(item for item in policy["sources"] if item["id"] == anchor_id)
        resolved, expected_provenance = _parse_anchor_objects(source, manifest_path)
        if derived != expected_provenance:
            raise TurkishCorpusError(f"{anchor_id} derived provenance drift")
        anchor_manifest_objects[anchor_id] = {
            item["uri"]: item for item in resolved
        }
    hplt_source = next(
        (source for source in policy["sources"] if source["id"] == "hplt3_tr"),
        None,
    )
    expected_hplt_bins = (
        list(hplt_source.get("selected_wds_bins", [8, 9, 10]))
        if hplt_source is not None
        else []
    )
    if plan.get("hplt_control") != {
        "map_sha256": HPLT_MAP_SHA256,
        "md5_list_url": HPLT_MD5_LIST_URL,
        "md5_list_sha256": HPLT_MD5_LIST_SHA256,
        "selected_wds_bins": expected_hplt_bins,
    }:
        raise TurkishCorpusError("source-plan HPLT control/bin contract drift")
    seen_sources: set[str] = set()
    seen_uris: set[str] = set()
    for rank, raw in enumerate(objects):
        item = _require_mapping(raw, f"objects[{rank}]")
        if item.get("rank") != rank:
            raise TurkishCorpusError("source-plan ranks must be contiguous and ordered")
        source_id = item.get("source_id")
        if source_id not in expected_sources:
            raise TurkishCorpusError("source plan contains an unknown source")
        seen_sources.add(str(source_id))
        uri = str(item.get("uri", ""))
        scheme = urllib.parse.urlparse(uri).scheme
        allowed_scheme = (
            "file"
            if source_id in {MACOCU_SOURCE_ID, MOT_SOURCE_ID, PARLAMINT_SOURCE_ID}
            else "https"
        )
        if uri in seen_uris or scheme != allowed_scheme:
            raise TurkishCorpusError("source plan has duplicate/unsupported object URI")
        seen_uris.add(uri)
        _require_positive_int(item.get("size_bytes"), "source object size")
        checksums = item.get("expected_checksums")
        if not isinstance(checksums, list) or not checksums:
            raise TurkishCorpusError("source object lacks expected checksums")
        for checksum in checksums:
            algorithm = checksum.get("algorithm")
            value = str(checksum.get("value", ""))
            if not (
                (algorithm == "sha256" and _SHA256_RE.fullmatch(value))
                or (algorithm == "md5" and _MD5_RE.fullmatch(value))
            ):
                raise TurkishCorpusError("source object has an unsupported checksum")
        if source_id == MACOCU_SOURCE_ID:
            if item.get("preparation_manifest_sha256") != macocu_manifest_sha256:
                raise TurkishCorpusError("MaCoCu object/preparation binding drift")
            genre_counts = _require_mapping(item.get("genre_counts"), "MaCoCu genre counts")
            if set(genre_counts) - MACOCU_GENRES or not genre_counts:
                raise TurkishCorpusError("MaCoCu object genre-count contract drift")
            prepared_shard = macocu_manifest_objects.get(uri)
            if prepared_shard is None or (
                item["size_bytes"] != prepared_shard.get("size_bytes")
                or item["expected_checksums"]
                != [{"algorithm": "sha256", "value": prepared_shard.get("sha256")}]
                or dict(genre_counts) != prepared_shard.get("genre_counts")
            ):
                raise TurkishCorpusError("MaCoCu source-plan shard inventory drift")
            local_path = Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path))
            if local_path.is_symlink() or not local_path.is_file():
                raise TurkishCorpusError("MaCoCu source-plan object is unsafe or missing")
        if source_id in {MOT_SOURCE_ID, PARLAMINT_SOURCE_ID}:
            expected_anchor = anchor_manifest_objects.get(str(source_id), {}).get(uri)
            if (
                expected_anchor is None
                or item.get("preparation_manifest_sha256")
                != derived_sources[str(source_id)].get("manifest_sha256")
                or item.get("size_bytes") != expected_anchor.get("size_bytes")
                or item.get("expected_checksums")
                != expected_anchor.get("expected_checksums")
            ):
                raise TurkishCorpusError("anchor source-plan shard inventory drift")
    if seen_sources != expected_sources:
        raise TurkishCorpusError("source plan does not cover every configured source")
    if macocu_manifest_objects and {
        uri for uri in seen_uris if uri in macocu_manifest_objects
    } != set(macocu_manifest_objects):
        raise TurkishCorpusError("source plan does not cover every prepared MaCoCu shard")
    for anchor_id, expected_objects_by_uri in anchor_manifest_objects.items():
        observed_anchor_uris = {
            str(item["uri"])
            for item in objects
            if item["source_id"] == anchor_id
        }
        if observed_anchor_uris != set(expected_objects_by_uri):
            raise TurkishCorpusError("source plan does not cover every anchor shard")
    if strict_derivation is not None:
        strict_objects = [
            item for item in objects if item["source_id"] == FINEWEB2_STRICT_SOURCE_ID
        ]
        projection = _fineweb2_inventory_projection(strict_objects)
        observed = {
            "object_count": len(projection),
            "total_bytes": sum(item["size_bytes"] for item in projection),
            "sha256": hashlib.sha256(
                canonical_json(projection).encode("utf-8")
            ).hexdigest(),
            "hash_semantics": V3_FINEWEB2_INVENTORY_SEMANTICS,
        }
        if observed != strict_derivation.get("resolved_inventory") or observed != {
            "object_count": V3_FINEWEB2_OBJECT_COUNT,
            "total_bytes": V3_FINEWEB2_TOTAL_BYTES,
            "sha256": V3_FINEWEB2_INVENTORY_SHA256,
            "hash_semantics": V3_FINEWEB2_INVENTORY_SEMANTICS,
        }:
            raise TurkishCorpusError("strict FineWeb source-plan inventory drift")
    totals = {
        "objects": len(objects),
        "size_bytes": sum(item["size_bytes"] for item in objects),
        "by_source": dict(sorted(Counter(item["source_id"] for item in objects).items())),
    }
    if plan.get("totals") != totals:
        raise TurkishCorpusError("source-plan totals drift")


def resolve_source_plan(
    policy: Mapping[str, Any],
    output_path: str | Path,
    *,
    request_get: Any = requests.get,
    request_head: Any = requests.head,
    hub_api: Any | None = None,
    macocu_manifest_path: str | Path | None = None,
    prepared_source_manifests: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Resolve every configured source to immutable object identities."""

    validate_corpus_policy(policy)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite source plan: {destination}")
    objects: list[dict[str, Any]] = []
    derived_sources: dict[str, Any] = {}
    prepared_manifests = dict(prepared_source_manifests or {})
    allowed_prepared_ids = {MOT_SOURCE_ID, PARLAMINT_SOURCE_ID}
    if set(prepared_manifests) - allowed_prepared_ids:
        raise TurkishCorpusError("unknown prepared-source manifest id")
    for source in policy["sources"]:
        if source["id"] == "hplt3_tr":
            resolved = _parse_hplt_objects(
                source, request_get=request_get, request_head=request_head
            )
        elif source["id"] == MACOCU_SOURCE_ID:
            if macocu_manifest_path is None:
                raise TurkishCorpusError(
                    "MaCoCu policy requires --macocu-manifest from the CPU preparation job"
                )
            resolved, provenance = _parse_macocu_objects(
                source, policy, macocu_manifest_path
            )
            derived_sources[MACOCU_SOURCE_ID] = provenance
        elif source["id"] == FINEWEB2_STRICT_SOURCE_ID:
            resolved = _parse_hub_objects(source, api=hub_api)
            derived_sources[FINEWEB2_STRICT_SOURCE_ID] = _strict_fineweb_derivation(
                source, policy, resolved
            )
        elif source["id"] in allowed_prepared_ids:
            manifest_path = prepared_manifests.get(source["id"])
            if manifest_path is None:
                raise TurkishCorpusError(
                    f"{source['id']} requires an accepted prepared-source manifest"
                )
            resolved, provenance = _parse_anchor_objects(source, manifest_path)
            derived_sources[source["id"]] = provenance
        else:
            resolved = _parse_hub_objects(source, api=hub_api)
        objects.extend(resolved)
    expected_prepared_ids = {
        source["id"]
        for source in policy["sources"]
        if source["id"] in allowed_prepared_ids
    }
    if set(prepared_manifests) != expected_prepared_ids:
        raise TurkishCorpusError("prepared-source manifest inventory drift")
    objects.sort(key=lambda item: (item["source_id"], item["uri"]))
    for rank, item in enumerate(objects):
        item["rank"] = rank
    plan = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": SOURCE_PLAN_KIND,
            "policy_sha256": _policy_sha256(policy),
            "hplt_control": {
                "map_sha256": HPLT_MAP_SHA256,
                "md5_list_url": HPLT_MD5_LIST_URL,
                "md5_list_sha256": HPLT_MD5_LIST_SHA256,
                "selected_wds_bins": next(
                    (
                        list(source.get("selected_wds_bins", [8, 9, 10]))
                        for source in policy["sources"]
                        if source["id"] == "hplt3_tr"
                    ),
                    [],
                ),
            },
            "derived_sources": derived_sources,
            "objects": objects,
            "totals": {
                "objects": len(objects),
                "size_bytes": sum(item["size_bytes"] for item in objects),
                "by_source": dict(
                    sorted(Counter(item["source_id"] for item in objects).items())
                ),
            },
            "canonical_sha256": None,
        }
    )
    validate_source_plan(plan, policy)
    write_json_atomic(destination, plan)
    return plan


def select_resource_sample_ranks(plan: Mapping[str, Any]) -> list[int]:
    """Select bounded whole-object samples for every source and quality stratum.

    Ordinary source inventories use fixed 25/50/75 percent URI-order points.
    HPLT shards are much larger and sample mode still verifies and scans whole
    immutable objects. Use the smallest complete shard in each WDS bin so the
    bounded audit cannot exceed the 48-hour cpu2dq limit merely because a bin's
    interior shards are 20+ GB. The per-byte fixed overhead makes this choice
    conservative for resource projection, while each selected HPLT shard still
    supplies a large, complete stratum sample for quality review.
    """

    def spread(items: list[Mapping[str, Any]]) -> set[int]:
        ordered = sorted(items, key=lambda item: str(item["uri"]))
        if not ordered:
            return set()
        last = len(ordered) - 1
        return {
            int(ordered[(last * numerator) // 4]["rank"])
            for numerator in (1, 2, 3)
        }

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in plan["objects"]:
        by_source[str(item["source_id"])].append(item)
    selected: set[int] = set()
    for source_id, items in by_source.items():
        if source_id != "hplt3_tr":
            selected.update(spread(items))
    hplt_objects = by_source.get("hplt3_tr", [])
    hplt_bins = {item.get("wds_bin") for item in hplt_objects}
    if any(not isinstance(value, int) or isinstance(value, bool) for value in hplt_bins):
        raise TurkishCorpusError("HPLT resource-sample objects require integer WDS bins")
    for wds_bin in sorted(hplt_bins):
        in_bin = [item for item in hplt_objects if item.get("wds_bin") == wds_bin]
        selected.add(
            int(
                min(
                    in_bin,
                    key=lambda item: (int(item["size_bytes"]), str(item["uri"])),
                )["rank"]
            )
        )
    macocu_objects = by_source.get(MACOCU_SOURCE_ID, [])
    for genre in sorted(MACOCU_CONVERSATION_GENRES | MACOCU_GENERAL_GENRES):
        candidates = [
            item
            for item in macocu_objects
            if int(item.get("genre_counts", {}).get(genre, 0)) > 0
        ]
        if macocu_objects and not candidates:
            raise TurkishCorpusError(
                f"MaCoCu resource sample cannot cover selected genre {genre!r}"
            )
        if candidates:
            max_count = max(int(item["genre_counts"][genre]) for item in candidates)
            meaningful = [
                item
                for item in candidates
                if int(item["genre_counts"][genre]) * 2 >= max_count
            ]
            selected.add(
                sorted(meaningful, key=lambda item: item["uri"])[
                    len(meaningful) // 2
                ]["rank"]
            )
    return sorted(selected)


def _verify_datatrove_runtime() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("datatrove")
        distribution = importlib.metadata.distribution("datatrove")
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise TurkishCorpusError("pinned DataTrove environment is not installed") from exc
    if version != DATATROVE_VERSION:
        raise TurkishCorpusError(
            f"DataTrove version drift: expected {DATATROVE_VERSION}, got {version}"
        )
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise TurkishCorpusError("DataTrove install lacks direct_url.json commit provenance")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise TurkishCorpusError("DataTrove direct_url.json is malformed") from exc
    revision = _nested_get(direct_url, "vcs_info.commit_id")
    if revision != DATATROVE_REVISION:
        raise TurkishCorpusError(
            f"DataTrove commit drift: expected {DATATROVE_REVISION}, got {revision!r}"
        )
    return {
        "implementation": "huggingface_datatrove_minhash",
        "version": version,
        "git_revision": revision,
        "direct_url_sha256": hashlib.sha256(direct_url_text.encode("utf-8")).hexdigest(),
    }


def _verify_environment_files() -> dict[str, Any]:
    root = Path(__file__).parents[1] / "environments/turkish-data"
    manifest = load_json_strict(root / "environment.json")
    for key in ("project_file", "lock_file"):
        entry = _require_mapping(manifest.get(key), f"environment.{key}")
        path = root / str(entry.get("path"))
        if file_sha256(path) != entry.get("sha256"):
            raise TurkishCorpusError(f"isolated preprocessing {key} hash drift")
    return {
        "environment_id": manifest["environment_id"],
        "manifest_sha256": file_sha256(root / "environment.json"),
        "project_sha256": manifest["project_file"]["sha256"],
        "lock_sha256": manifest["lock_file"]["sha256"],
    }


def _minhash_config() -> Any:
    _verify_datatrove_runtime()
    try:
        from datatrove.pipeline.dedup.minhash import MinhashConfig
        from datatrove.utils.hashing import HashConfig
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError("pinned DataTrove MinHash modules are unavailable") from exc
    return MinhashConfig(
        n_grams=5,
        num_buckets=14,
        hashes_per_bucket=8,
        hash_config=HashConfig(precision=64),
    )


def _load_glotlid_model(model_path: Path, policy: Mapping[str, Any]) -> Any:
    expected = policy["language_policy"]["independent_audit"]
    if not model_path.is_file() or model_path.is_symlink():
        raise TurkishCorpusError(f"GlotLID model is missing/unsafe: {model_path}")
    if model_path.stat().st_size != expected["artifact_size_bytes"]:
        raise TurkishCorpusError("GlotLID model size drift")
    if file_sha256(model_path) != expected["artifact_sha256"]:
        raise TurkishCorpusError("GlotLID model SHA-256 drift")
    try:
        import fasttext
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError(
            "fasttext-numpy2-wheel is required for GlotLID"
        ) from exc
    return fasttext.load_model(str(model_path))


def fetch_glotlid_model(
    policy: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Fetch the pinned GlotLID artifact into a dedicated local directory."""

    validate_corpus_policy(policy)
    expected = policy["language_policy"]["independent_audit"]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / expected["artifact"]
    if target.exists():
        if (
            target.stat().st_size == expected["artifact_size_bytes"]
            and file_sha256(target) == expected["artifact_sha256"]
        ):
            return _file_record(target)
        raise TurkishCorpusError("existing GlotLID artifact does not match the pin")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError("huggingface-hub is required to fetch GlotLID") from exc
    downloaded = Path(
        hf_hub_download(
            repo_id=expected["repo_id"],
            filename=expected["artifact"],
            revision=expected["hub_revision"],
            local_dir=destination,
        )
    )
    if downloaded.resolve() != target.resolve():
        shutil.copyfile(downloaded, target)
    if (
        target.stat().st_size != expected["artifact_size_bytes"]
        or file_sha256(target) != expected["artifact_sha256"]
    ):
        raise TurkishCorpusError("downloaded GlotLID artifact failed pinned verification")
    return _file_record(target)


def _predict_lid(model: Any, text: str) -> tuple[str, float, float]:
    prepared = " ".join(text.split())
    if not prepared:
        return "", 0.0, 0.0
    labels, probabilities = model.predict(prepared, k=2)
    clean_labels = [str(label).removeprefix("__label__") for label in labels]
    probs = [float(value) for value in probabilities]
    if not clean_labels or not probs:
        return "", 0.0, 0.0
    second = probs[1] if len(probs) > 1 else 0.0
    return clean_labels[0], probs[0], probs[0] - second


def _calibrate_lid(
    model: Any, fixture_path: Path, policy: Mapping[str, Any]
) -> dict[str, Any]:
    expected_languages = policy["language_policy"]["independent_audit"][
        "calibration_languages"
    ]
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    turkish_false_positive = 0
    rows = 0
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise TurkishCorpusError(
                    f"{fixture_path}:{line_number}: malformed calibration JSON"
                ) from exc
            expected = str(record.get("expected_label", ""))
            text = str(record.get("text", ""))
            if expected not in expected_languages or not text:
                raise TurkishCorpusError("GlotLID calibration fixture schema/languages drift")
            label, _probability, _margin = _predict_lid(model, text)
            totals[expected] += 1
            correct[expected] += int(label == expected)
            if expected != "tur_Latn" and label == "tur_Latn":
                turkish_false_positive += 1
            rows += 1
    if list(totals) != list(expected_languages) or any(totals[item] != 8 for item in expected_languages):
        raise TurkishCorpusError("GlotLID calibration must contain eight ordered probes per language")
    per_language = {
        label: {
            "probes": totals[label],
            "correct_top_label": correct[label],
            "accuracy": correct[label] / totals[label],
        }
        for label in expected_languages
    }
    overall_accuracy = sum(correct.values()) / rows
    foreign_count = rows - totals["tur_Latn"]
    fp_rate = turkish_false_positive / foreign_count
    passed = (
        per_language["tur_Latn"]["accuracy"] >= 0.875
        and overall_accuracy >= 0.75
        and fp_rate <= 0.125
    )
    return {
        "languages": list(expected_languages),
        "fixture_sha256": file_sha256(fixture_path),
        "acceptance": {
            "minimum_turkish_top_label_accuracy": 0.875,
            "minimum_overall_top_label_accuracy": 0.75,
            "maximum_foreign_to_turkish_false_positive_rate": 0.125,
        },
        "metrics": {
            "per_language": per_language,
            "overall_top_label_accuracy": overall_accuracy,
            "foreign_to_turkish_false_positives": turkish_false_positive,
            "foreign_to_turkish_false_positive_rate": fp_rate,
        },
        "passed": passed,
    }


def _calibrate_signature_tokenizer() -> dict[str, Any]:
    try:
        from datatrove.utils.typeshelper import Languages
        from datatrove.utils.word_tokenizers import load_word_tokenizer
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError("DataTrove word tokenizer is unavailable") from exc
    language = Languages.turkish__latn
    if language != "tur_Latn":
        raise TurkishCorpusError("DataTrove Turkish/Latin language constant drift")
    tokenizer = load_word_tokenizer(language)
    probes = [
        "İstanbul'da arkadaşlarımla uzun uzun konuştuk.",
        "Bugün hava güzel; yarın yağmur yağabilir.",
        "Çocukların oynadığı park mahallemize çok yakın.",
    ]
    outputs = [tokenizer.word_tokenize(text) for text in probes]
    passed = all(len(tokens) >= 4 and "" not in tokens for tokens in outputs)
    payload = {
        "language": language,
        "tokenizer_class": type(tokenizer).__name__,
        "probes": probes,
        "outputs": outputs,
    }
    return {
        "language": language,
        "passed": passed,
        "probe_sha256": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
        "probe_count": len(probes),
    }


def _calibrate_lsh(*, trials: int = 1024) -> dict[str, Any]:
    """Empirically verify the actual DataTrove 14x8 candidate curve."""

    if trials < 256:
        raise TurkishCorpusError("LSH calibration requires at least 256 trials")
    try:
        import numpy as np
        from datatrove.pipeline.dedup import MinhashDedupSignature
        from datatrove.utils.hashing import create_hash_func
        from datatrove.utils.typeshelper import Languages
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError("DataTrove/NumPy are required for LSH calibration") from exc
    config = _minhash_config()
    signature = MinhashDedupSignature(
        output_folder=str(Path(tempfile.gettempdir()) / "unused_datatrove_calibration"),
        config=config,
        language=Languages.turkish__latn,
    )
    shingle_hash = create_hash_func(config.hash_config)
    similarities = (0.50, 0.65, 0.70, 0.82, 0.90)
    # DataTrove hashes every production shingle before applying its affine
    # MinHash permutations. Raw sequential integers create pathological band
    # correlations, so exercise the pinned shingle hash here as well.
    observations: list[dict[str, Any]] = []
    universe = 200
    for similarity_index, similarity in enumerate(similarities):
        # For two equal-size sets, intersection n solves J=n/(2m-n).
        intersection = max(5, round((2 * universe * similarity) / (1 + similarity)))
        actual = intersection / (2 * universe - intersection)
        matches = 0
        for trial in range(trials):
            values = np.fromiter(
                (
                    shingle_hash(
                        f"nanochat-lsh-calibration-v2:{similarity_index}:{trial}:{item}"
                    )
                    for item in range(2 * universe - intersection)
                ),
                dtype=np.uint64,
                count=2 * universe - intersection,
            )
            if np.unique(values).size != values.size:
                raise TurkishCorpusError("synthetic MinHash input collision")
            left = values[:universe].reshape((-1, 1))
            right = np.concatenate(
                (values[:intersection], values[universe:])
            ).reshape((-1, 1))
            left_sig = signature.get_signature(left)
            right_sig = signature.get_signature(right)
            matches += int(any(a == b for a, b in zip(left_sig, right_sig)))
        observed = matches / trials
        theoretical = 1 - (1 - actual**8) ** 14
        observations.append(
            {
                "target_similarity": similarity,
                "actual_jaccard": actual,
                "trials": trials,
                "candidate_matches": matches,
                "observed_probability": observed,
                "theoretical_probability": theoretical,
                "absolute_error": abs(observed - theoretical),
            }
        )
    passed = all(item["absolute_error"] <= 0.06 for item in observations)
    body = {
        "implementation": "actual_datatrove_get_signature_random_hashes_v2",
        "synthetic_input": "unique_pinned_xxhash64_shingle_ids_v1",
        "precision_bits": 64,
        "num_buckets": 14,
        "hashes_per_bucket": 8,
        "ngram_words": 5,
        "match_rule": "any_equal_8_hash_bucket_signature",
        "probability_formula": "1-(1-s^8)^14",
        "observations": observations,
        "maximum_absolute_error": 0.06,
        "passed": passed,
    }
    return body | {
        "receipt_sha256": hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    }


def validate_backend_calibration(
    calibration: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    verify_manifest_hash(calibration)
    if calibration.get("schema_version") != "1.0" or calibration.get("kind") != CALIBRATION_KIND:
        raise TurkishCorpusError("unexpected backend calibration kind/version")
    if calibration.get("policy_sha256") != _policy_sha256(policy):
        raise TurkishCorpusError("backend calibration is bound to another policy")
    if calibration.get("model_sha256") != policy["language_policy"]["independent_audit"][
        "artifact_sha256"
    ]:
        raise TurkishCorpusError("backend calibration model hash drift")
    lid = _require_mapping(calibration.get("lid_calibration"), "lid_calibration")
    if lid.get("passed") is not True:
        raise TurkishCorpusError("GlotLID calibration failed")
    lsh = _require_mapping(calibration.get("synthetic_similarity_calibration"), "LSH calibration")
    if lsh.get("passed") is not True or not _SHA256_RE.fullmatch(
        str(lsh.get("receipt_sha256", ""))
    ):
        raise TurkishCorpusError("DataTrove LSH calibration failed")
    probe = _require_mapping(calibration.get("signature_tokenizer_probe"), "signature tokenizer probe")
    if probe.get("language") != "tur_Latn" or probe.get("passed") is not True:
        raise TurkishCorpusError("Turkish DataTrove tokenizer probe failed")


def run_backend_calibration(
    policy: Mapping[str, Any],
    model_path: str | Path,
    fixture_path: str | Path,
    output_path: str | Path,
    *,
    lsh_trials: int = 1024,
) -> dict[str, Any]:
    validate_corpus_policy(policy)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite backend calibration: {destination}")
    model = _load_glotlid_model(Path(model_path), policy)
    lid = _calibrate_lid(model, Path(fixture_path), policy)
    lsh = _calibrate_lsh(trials=lsh_trials)
    tokenizer_probe = _calibrate_signature_tokenizer()
    calibration = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": CALIBRATION_KIND,
            "policy_sha256": _policy_sha256(policy),
            "model_sha256": policy["language_policy"]["independent_audit"][
                "artifact_sha256"
            ],
            "environment": _verify_environment_files(),
            "datatrove_runtime": _verify_datatrove_runtime(),
            "lid_calibration": lid,
            "synthetic_similarity_calibration": lsh,
            "signature_tokenizer_probe": tokenizer_probe,
            "passed": bool(lid["passed"] and lsh["passed"] and tokenizer_probe["passed"]),
            "canonical_sha256": None,
        }
    )
    if calibration["passed"] is not True:
        raise TurkishCorpusError("backend calibration did not pass; receipt is not sealed")
    write_json_atomic(destination, calibration)
    return calibration


def production_processing_binding(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact, receipt-bound quality/formatting implementation.

    ``official_fineweb2_control`` is an exact transcription of the pinned
    Turkish YAML and the pipeline's global adaptations. ``project_additions``
    are deliberately named separately so they cannot be confused with the
    published FineWeb-2 recipe.
    """

    configured = _require_mapping(
        policy["content_policy"].get("production_processing"),
        "content_policy.production_processing",
    )
    control = _require_mapping(
        configured.get("fineweb2_turkish_control"),
        "production_processing.fineweb2_turkish_control",
    )
    if (
        control.get("revision") != FINEWEB2_CONFIG_REVISION
        or control.get("sha256") != FINEWEB2_CONFIG_SHA256
        or file_sha256(FINEWEB2_CONFIG_PATH) != FINEWEB2_CONFIG_SHA256
    ):
        raise TurkishCorpusError("pinned FineWeb-2 Turkish filter control drift")
    official = {
        "repository": "https://github.com/huggingface/fineweb-2",
        "revision": FINEWEB2_CONFIG_REVISION,
        "config_path": "configs/tur_Latn.yml",
        "config_sha256": FINEWEB2_CONFIG_SHA256,
        "language_score_reference": 0.875,
        "sequence_after_minhash": [
            "GopherRepetitionFilter",
            "FineWebQualityFilter",
            "GopherQualityFilter",
            "FTFYFormatter",
            "PIIFormatter",
            "SymbolLinesFormatter",
        ],
        "gopher_repetition": {
            "language": "tur_Latn",
            "dup_line_frac": 0.272,
            "dup_para_frac": 0,
            "dup_line_char_frac": 0,
            "dup_para_char_frac": 0,
            "top_n_grams": [[2, 0.214], [3, 0.168], [4, 0.147]],
            "dup_n_grams": [
                [5, 0.154],
                [6, 0.144],
                [7, 0.134],
                [8, 0.124],
                [9, 0.113],
                [10, 0.103],
            ],
        },
        "fineweb_quality": {
            "language": "tur_Latn",
            "line_punct_thr": 0.091,
            "short_line_thr": 999,
            "char_duplicates_ratio": 0.1,
            "new_line_ratio": 0.222,
        },
        "gopher_quality": {
            "language": "tur_Latn",
            "min_doc_words": 50,
            "min_avg_word_length": 3,
            "max_avg_word_length": 21,
            "max_non_alpha_words_ratio": 0.773,
            "min_stop_words": 2,
            "stop_words": [
                "ve",
                "bir",
                "olarak",
                "bu",
                "ile",
                "için",
                "olan",
                "da",
                "de",
                "tarafından",
                "yılında",
                "sonra",
                "en",
                "daha",
                "ilk",
                "the",
            ],
        },
        "formatters": {
            "ftfy": "DataTrove-0.10.0 defaults",
            "pii": {
                "remove_emails": True,
                "remove_ips": True,
                "only_remove_public_ips": True,
                "email_replacement": "<email>",
                "ip_replacement": "<ip>",
            },
            "symbol_lines": {"symbols_to_remove": ["|"], "replace_char": "\n"},
        },
    }
    local_audit = audit_policy_binding(policy["content_policy"])
    additions = {
        "independent_glotlid": {
            "required_top_label": "tur_Latn",
            "document_min_probability": policy["language_policy"]["independent_audit"][
                "document_min_probability"
            ],
            "document_min_margin": policy["language_policy"]["independent_audit"][
                "document_min_margin"
            ],
            "paragraph_gate": True,
            "note": "independent document+margin+paragraph gate; its 0.8 scalar is not misrepresented as the official 0.875 threshold",
        },
        "conversation_preserving_quality_adaptation": {
            "gopher_min_doc_words": int(policy["content_policy"]["min_words"]),
            "official_default_control": 50,
            "reason": "retain complete short conversational/general prose while the separate 24-word and Turkish gates remain active",
            "requires_stratified_QA": True,
        },
        "turkish_phone_redaction": {
            "pattern_sha256": hashlib.sha256(_PHONE_RE.pattern.encode("utf-8")).hexdigest(),
            "replacement": "<telefon>",
        },
        "harmful_signal_measurement": {
            "pattern_sha256": hashlib.sha256(
                _HARMFUL_SIGNAL_RE.pattern.encode("utf-8")
            ).hexdigest(),
            "action": "measure_and_manual_qa_not_automatic_semantic_rewrite",
        },
        "local_policy_audit": local_audit,
        "no_code": {
            "enforced": True,
            "code_line_pattern_sha256": local_audit["patterns"]["code_line"][
                "sha256"
            ],
            "programming_term_pattern_sha256": local_audit["patterns"][
                "programming_term"
            ]["sha256"],
            "scalar_assignment_line_pattern_sha256": local_audit["patterns"][
                "scalar_assignment_line"
            ]["sha256"],
            "minimum_consecutive_scalar_assignment_lines": local_audit[
                "structural_thresholds"
            ]["minimum_consecutive_scalar_assignment_lines"],
            "minimum_compact_scalar_assignments": local_audit[
                "structural_thresholds"
            ]["minimum_compact_scalar_assignments"],
            "scalar_assignment_classifier": local_audit[
                "structural_thresholds"
            ]["scalar_assignment_classifier"],
            "max_code_line_fraction": policy["content_policy"][
                "max_code_line_fraction"
            ],
            "max_code_punctuation_fraction": policy["content_policy"][
                "max_code_punctuation_fraction"
            ],
            "max_programming_term_hits": policy["content_policy"][
                "max_programming_term_hits"
            ],
        },
    }
    body = {
        "implementation": configured["implementation"],
        "official_fineweb2_control": official,
        "project_additions": additions,
    }
    return body | {
        "binding_sha256": hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    }


def production_code_identity() -> dict[str, Any]:
    """Return the exact executable-file identity used by the cluster stage."""

    root = Path(__file__).resolve().parents[1]
    critical = (
        "nanochat/turkish_backend.py",
        "nanochat/turkish_corpus.py",
    )
    files = [
        {
            "path": relative,
            "size_bytes": (root / relative).stat().st_size,
            "sha256": file_sha256(root / relative),
        }
        for relative in critical
    ]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TurkishCorpusError("cannot resolve production code commit") from exc
    if not _SHA1_RE.fullmatch(commit):
        raise TurkishCorpusError("production code commit is not a full SHA-1")
    body = {
        "identity_kind": "git_commit_plus_critical_file_sha256_v1",
        "git_commit": commit,
        "critical_files": files,
    }
    return body | {
        "binding_sha256": hashlib.sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest()
    }


def validate_production_code_identity(identity: Any) -> dict[str, Any]:
    """Validate a recorded commit while matching the current critical bytes."""

    if not isinstance(identity, Mapping):
        raise TurkishCorpusError("production code identity is missing")
    current = production_code_identity()
    body = {key: value for key, value in identity.items() if key != "binding_sha256"}
    if (
        identity.get("identity_kind")
        != "git_commit_plus_critical_file_sha256_v1"
        or not _SHA1_RE.fullmatch(str(identity.get("git_commit", "")))
        or identity.get("critical_files") != current["critical_files"]
        or identity.get("binding_sha256")
        != hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    ):
        raise TurkishCorpusError("production code identity drift")
    return dict(identity)


def _sample_launch_bindings(
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    buckets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and bind the one packed object and bucket sample launch."""

    def one(pattern: str, label: str) -> Path:
        paths = sorted(run_root.glob(pattern))
        if len(paths) != 1 or paths[0].is_symlink() or not paths[0].is_file():
            raise TurkishCorpusError(f"expected exactly one safe {label} receipt")
        return paths[0]

    def load(path: Path, label: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
        raw = _read_bounded_regular_file_snapshot(
            path, label=label, max_bytes=_MAX_RECEIPT_EVIDENCE_BYTES
        )
        value = _load_json_snapshot(raw, label)
        if not isinstance(value, dict):
            raise TurkishCorpusError(f"{label} must contain a JSON object")
        digest = verify_manifest_hash(value)
        return value, digest, {
            "path": path.relative_to(run_root).as_posix(),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_sha256": digest,
        }

    object_path = one(
        "packed_sample_launches/job*/launch_receipt.json", "object-sample launch"
    )
    object_launch, object_sha, object_record = load(
        object_path, "object-sample launch receipt"
    )
    expected_ranks = [int(item["rank"]) for item in objects]
    object_records = object_launch.get("object_receipts")
    if (
        object_launch.get("schema_version") != "1.0"
        or object_launch.get("kind")
        != "turkish_packed_resource_sample_launch_receipt"
        or object_launch.get("sample_mode") is not True
        or object_launch.get("all_lanes_completed") is not True
        or object_launch.get("policy_sha256") != plan["policy_sha256"]
        or object_launch.get("source_plan_sha256") != plan["canonical_sha256"]
        or object_launch.get("calibration_sha256")
        != calibration["canonical_sha256"]
        or not isinstance(object_records, list)
        or [item.get("rank") for item in object_records] != expected_ranks
        or [item.get("canonical_sha256") for item in object_records]
        != [item["canonical_sha256"] for item in objects]
    ):
        raise TurkishCorpusError("packed object-sample launch binding drift")

    bucket_path = one(
        "packed_bucket_launches/job*/launch_receipt.json", "bucket-sample launch"
    )
    bucket_launch, bucket_sha, bucket_record = load(
        bucket_path, "bucket-sample launch receipt"
    )
    bucket_records = bucket_launch.get("backend_bucket_receipts")
    if (
        bucket_launch.get("schema_version") != "1.0"
        or bucket_launch.get("kind") != "turkish_packed_sample_bucket_launch_receipt"
        or bucket_launch.get("sample_mode") is not True
        or bucket_launch.get("all_buckets_completed") is not True
        or bucket_launch.get("policy_sha256") != plan["policy_sha256"]
        or bucket_launch.get("source_plan_sha256") != plan["canonical_sha256"]
        or bucket_launch.get("calibration_sha256")
        != calibration["canonical_sha256"]
        or bucket_launch.get("object_sample_launch_receipt_sha256") != object_sha
        or not isinstance(bucket_records, list)
        or [item.get("bucket_rank") for item in bucket_records] != list(range(14))
        or [item.get("canonical_sha256") for item in bucket_records]
        != [item["canonical_sha256"] for item in buckets]
    ):
        raise TurkishCorpusError("packed bucket-sample launch binding drift")

    return {
        "object": object_record,
        "bucket": bucket_record,
    }


class _ProductionProcessors:
    def __init__(self, policy: Mapping[str, Any]) -> None:
        try:
            from datatrove.pipeline.filters import (
                FineWebQualityFilter,
                GopherQualityFilter,
                GopherRepetitionFilter,
            )
            from datatrove.pipeline.formatters import (
                FTFYFormatter,
                PIIFormatter,
                SymbolLinesFormatter,
            )
        except ImportError as exc:  # pragma: no cover
            raise TurkishCorpusError("DataTrove processing extras are unavailable") from exc
        self.binding = production_processing_binding(policy)
        official = self.binding["official_fineweb2_control"]
        rep = official["gopher_repetition"]
        fw = official["fineweb_quality"]
        gopher = official["gopher_quality"]
        effective_min_words = self.binding["project_additions"][
            "conversation_preserving_quality_adaptation"
        ]["gopher_min_doc_words"]
        self.filter_stages: list[tuple[str, Any]] = [
            (
                "gopher_repetition",
                GopherRepetitionFilter(
                    language="tur_Latn",
                    dup_line_frac=rep["dup_line_frac"],
                    dup_para_frac=rep["dup_para_frac"],
                    dup_line_char_frac=rep["dup_line_char_frac"],
                    dup_para_char_frac=rep["dup_para_char_frac"],
                    top_n_grams=tuple(tuple(item) for item in rep["top_n_grams"]),
                    dup_n_grams=tuple(tuple(item) for item in rep["dup_n_grams"]),
                ),
            ),
            (
                "fineweb_quality",
                FineWebQualityFilter(
                    language="tur_Latn",
                    line_punct_thr=fw["line_punct_thr"],
                    short_line_thr=fw["short_line_thr"],
                    char_duplicates_ratio=fw["char_duplicates_ratio"],
                    new_line_ratio=fw["new_line_ratio"],
                ),
            ),
            (
                "gopher_quality",
                GopherQualityFilter(
                    language="tur_Latn",
                    min_doc_words=effective_min_words,
                    min_avg_word_length=gopher["min_avg_word_length"],
                    max_avg_word_length=gopher["max_avg_word_length"],
                    max_non_alpha_words_ratio=gopher["max_non_alpha_words_ratio"],
                    min_stop_words=gopher["min_stop_words"],
                    stop_words=gopher["stop_words"],
                ),
            ),
        ]
        self.ftfy = FTFYFormatter()
        self.pii = PIIFormatter(
            email_replacement="<email>",
            ip_replacement="<ip>",
            only_remove_public_ips=True,
        )
        self.symbols = SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n")

    @staticmethod
    def _filter_result(raw: Any) -> tuple[bool, str]:
        if isinstance(raw, tuple):
            return bool(raw[0]), str(raw[1])
        return bool(raw), "unspecified"

    def evaluate_filters(self, text: str) -> tuple[list[str], list[dict[str, Any]]]:
        from datatrove.data import Document

        flags: list[str] = []
        trace: list[dict[str, Any]] = []
        doc = Document(text=text, id="quality-probe")
        active = True
        for name, stage in self.filter_stages:
            if not active:
                trace.append({"stage": name, "evaluated": False, "passed": None})
                continue
            passed, reason = self._filter_result(stage.filter(doc))
            trace.append(
                {"stage": name, "evaluated": True, "passed": passed, "reason": reason}
            )
            if not passed:
                flags.append(f"{name}:{reason}")
                active = False
        return flags, trace

    def format_safely(self, text: str) -> tuple[str, dict[str, int]]:
        counts: dict[str, int] = {
            "email_matches": len(_EMAIL_RE.findall(text)),
            "ip_like_matches": len(_IP_RE.findall(text)),
            "turkish_phone_matches": len(_PHONE_RE.findall(text)),
            "harmful_signal_hits": len(_HARMFUL_SIGNAL_RE.findall(text)),
        }
        ftfy_text = self.ftfy.format(text)
        pii_text = self.pii.format(ftfy_text)
        phone_text, phone_replacements = _PHONE_RE.subn("<telefon>", pii_text)
        final = self.symbols.format(phone_text)
        counts.update(
            {
                "ftfy_changed": int(ftfy_text != text),
                "pii_changed": int(pii_text != ftfy_text),
                "phone_replacements": phone_replacements,
                "symbol_lines_changed": int(final != phone_text),
                "characters_before": len(text),
                "characters_after": len(final),
            }
        )
        return final, counts


class _DeterministicRemovalSamples:
    """Keep the lowest hash examples per stage without order dependence."""

    def __init__(self, capacity: int = 8, max_chars: int = 1200) -> None:
        self.capacity = capacity
        self.max_chars = max_chars
        self._items: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    def add(self, stage: str, identity: str, text: str, metadata: Mapping[str, Any]) -> None:
        key = hashlib.sha256(f"{stage}\x00{identity}".encode("utf-8")).hexdigest()
        payload = {
            "stage": stage,
            "identity_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "text": text[: self.max_chars],
            "text_truncated": len(text) > self.max_chars,
            "metadata": dict(metadata),
        }
        values = self._items[stage]
        values.append((key, payload))
        values.sort(key=lambda item: item[0])
        del values[self.capacity :]

    def rows(self) -> list[dict[str, Any]]:
        return [
            payload
            for stage in sorted(self._items)
            for _key, payload in self._items[stage]
        ]


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_record(path) | {"rows": count}


_INTERNAL_SCHEMA = pa.schema(
    [
        pa.field("text", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("source_lid_label", pa.string(), nullable=False),
        pa.field("source_lid_probability", pa.float64(), nullable=False),
        pa.field("lid_label", pa.string(), nullable=False),
        pa.field("lid_probability", pa.float64(), nullable=False),
        pa.field("lid_margin", pa.float64(), nullable=False),
        pa.field("paragraph_min_probability", pa.float64(), nullable=False),
        pa.field("paragraph_min_margin", pa.float64(), nullable=False),
        pa.field("failed_long_paragraph_fraction", pa.float64(), nullable=False),
        pa.field("dedup_cluster_id", pa.string(), nullable=False),
        pa.field("dedup_keep", pa.bool_(), nullable=False),
        pa.field("quality_score", pa.float64(), nullable=False),
        pa.field("wds_bin", pa.int16(), nullable=True),
        pa.field("web-register", pa.string(), nullable=False),
        pa.field("genre", pa.string(), nullable=False),
        pa.field("pii_replacements", pa.int32(), nullable=False),
        pa.field("harmful_signal_hits", pa.int32(), nullable=False),
        pa.field("quality_filter_flags", pa.string(), nullable=False),
        pa.field("formatting_changes", pa.string(), nullable=False),
        pa.field("candidate_rank", pa.int32(), nullable=False),
        pa.field("candidate_doc_index", pa.int64(), nullable=False),
    ]
)
_BACKEND_SCHEMA = pa.schema([_INTERNAL_SCHEMA.field(name) for name in BACKEND_COLUMNS])


class _ParquetBatchWriter:
    def __init__(self, path: Path, schema: pa.Schema, *, batch_rows: int = 4096) -> None:
        self.path = path
        self.schema = schema
        self.batch_rows = batch_rows
        self.rows: list[Mapping[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=True,
            )
        self.writer.write_table(table, row_group_size=self.batch_rows)
        self.count += len(self.rows)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            # Preserve deterministic rank identity even when a source object has
            # no Turkish candidates. Empty files are internal only and are not
            # included in the final backend receipt.
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.path, compression="zstd")
        else:
            self.writer.close()
            self.writer = None


def _source_lid(record: Mapping[str, Any], adapter: Mapping[str, Any]) -> tuple[str, float, bool]:
    return source_lid_result(record, adapter, strict_schema=True)


def _document_lid_metrics(
    model: Any, text: str, policy: Mapping[str, Any]
) -> dict[str, Any]:
    gate = policy["language_policy"]["independent_audit"]
    label, probability, margin = _predict_lid(model, text)
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if len(paragraph.strip()) >= int(gate["paragraph_min_chars"])
    ]
    paragraph_predictions = [_predict_lid(model, paragraph) for paragraph in paragraphs]
    if paragraph_predictions:
        paragraph_min_probability = min(item[1] for item in paragraph_predictions)
        paragraph_min_margin = min(item[2] for item in paragraph_predictions)
        failed = sum(
            item[0] != gate["required_top_label"]
            or item[1] < float(gate["paragraph_min_probability"])
            or item[2] < float(gate["paragraph_min_margin"])
            for item in paragraph_predictions
        )
        failed_fraction = failed / len(paragraph_predictions)
    else:
        paragraph_min_probability = probability
        paragraph_min_margin = margin
        failed_fraction = 0.0
    passed = (
        label == gate["required_top_label"]
        and probability >= float(gate["document_min_probability"])
        and margin >= float(gate["document_min_margin"])
        and paragraph_min_probability >= float(gate["paragraph_min_probability"])
        and paragraph_min_margin >= float(gate["paragraph_min_margin"])
        and failed_fraction <= float(gate["max_failed_long_paragraph_fraction"])
    )
    return {
        "lid_label": label,
        "lid_probability": probability,
        "lid_margin": margin,
        "paragraph_min_probability": paragraph_min_probability,
        "paragraph_min_margin": paragraph_min_margin,
        "failed_long_paragraph_fraction": failed_fraction,
        "long_paragraphs": len(paragraph_predictions),
        "passed": passed,
    }


def _source_quality_score(record: Mapping[str, Any]) -> float:
    """Return source-provided quality without conflating it with Turkish LID.

    GlotLID confidence answers a language-identification question.  It is not a
    semantic-quality score and therefore must not decide same-priority MinHash
    winners.  ``quality_score`` remains the on-disk compatibility field, but
    its value is now strictly the best finite source-quality signal (or zero
    when the source exposes none).
    """

    candidates: list[float] = [0.0]
    for field in (
        "quality_score",
        "educational_score",
        "fineweb2_hq_score",
    ):
        try:
            value = float(_nested_get(record, field, 0.0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            candidates.append(value)
    return max(candidates)


def _attested_source_quality(
    row: Mapping[str, Any], object_receipt: Mapping[str, Any]
) -> float:
    """Return source quality only when the producing receipt attests its semantics.

    Object candidates produced before the source/LID separation used the same
    compatibility column but sometimes populated it with GlotLID confidence.
    Missing object-receipt semantics therefore fail closed to a neutral zero;
    they never receive a deduplication advantage and are rewritten to zero in
    the merged output.
    """

    if object_receipt.get("quality_score_semantics") != OBJECT_SOURCE_QUALITY_SEMANTICS:
        return 0.0
    try:
        score = float(row["quality_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TurkishCorpusError("attested candidate quality_score is invalid") from exc
    if not math.isfinite(score) or score < 0.0:
        raise TurkishCorpusError("attested candidate quality_score must be finite and non-negative")
    return score


def _redact_sample_text(text: str) -> str:
    text = _EMAIL_RE.sub("<email>", text)
    text = _IP_RE.sub("<ip>", text)
    return _PHONE_RE.sub("<telefon>", text)


def _stage_source_object(
    item: Mapping[str, Any], destination: Path, *, request_get: Any = requests.get
) -> tuple[dict[str, Any], VerifiedStagedArtifact]:
    uri = str(item["uri"])
    parsed = urllib.parse.urlparse(uri)
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()  # noqa: S324 - verifies an upstream checksum, never security
    size = 0
    if parsed.scheme == "file":
        source = Path(urllib.parse.unquote(parsed.path))
        if source.is_symlink() or not source.is_file():
            raise TurkishCorpusError(f"local source object is unsafe or missing: {source}")
        stream: Any = source.open("rb")
        response = None
    elif parsed.scheme == "https":
        response = request_get(uri, stream=True, timeout=600)
        response.raise_for_status()
        stream = response.iter_content(chunk_size=8 * 1024 * 1024)
    else:
        raise TurkishCorpusError(f"unsupported source object scheme: {uri}")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(destination, flags, 0o600)
    try:
        try:
            with os.fdopen(os.dup(descriptor), "wb") as output:
                iterable = (
                    iter(lambda: stream.read(8 * 1024 * 1024), b"")
                    if response is None
                    else stream
                )
                for chunk in iterable:
                    if not chunk:
                        continue
                    sha256.update(chunk)
                    md5.update(chunk)
                    output.write(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
        finally:
            if response is None:
                stream.close()
            else:
                response.close()
        if size != item["size_bytes"]:
            raise TurkishCorpusError(
                f"source size drift for {uri}: expected {item['size_bytes']}, got {size}"
            )
        observed = {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}
        for expected in item["expected_checksums"]:
            if observed[expected["algorithm"]] != expected["value"]:
                raise TurkishCorpusError(
                    f"source checksum drift for {uri} ({expected['algorithm']})"
                )
        record = {
            "uri": uri,
            "size_bytes": size,
            "sha256": observed["sha256"],
            "upstream_checksums_verified": list(item["expected_checksums"]),
        }
        artifact = VerifiedStagedArtifact(
            destination,
            descriptor,
            expected_size=size,
            expected_sha256=observed["sha256"],
        )
        return record, artifact
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        destination.unlink(missing_ok=True)
        raise


def _signature_files(root: Path, rank: int) -> list[Path]:
    return [root / f"bucket_{bucket:03d}" / f"{rank:05d}.minhash.sig" for bucket in range(14)]


def _iter_candidate_documents(
    run_root: Path, record: Mapping[str, Any], *, label: str
) -> Iterator[Any]:
    try:
        from datatrove.data import Document
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError("DataTrove Document type is unavailable") from exc
    with _open_verified_run_artifact(
        run_root,
        record,
        label=label,
        max_bytes=_MAX_CLUSTER_PARQUET_BYTES,
    ) as (_path, handle):
        parquet = pq.ParquetFile(handle)
        rows_expected = record.get("rows")
        if (
            isinstance(rows_expected, bool)
            or not isinstance(rows_expected, int)
            or rows_expected < 0
            or parquet.metadata.num_rows != rows_expected
            or not {"text", "document_id"} <= set(parquet.schema_arrow.names)
        ):
            raise TurkishCorpusError(f"{label} Parquet metadata drift")
        rows_seen = 0
        for batch in parquet.iter_batches(
            batch_size=2048, columns=["text", "document_id"]
        ):
            for row in batch.to_pylist():
                rows_seen += 1
                if not isinstance(row.get("text"), str) or not isinstance(
                    row.get("document_id"), str
                ):
                    raise TurkishCorpusError(f"{label} row schema drift")
                yield Document(text=row["text"], id=row["document_id"])
        if rows_seen != rows_expected:
            raise TurkishCorpusError(f"{label} Parquet row scan drift")


def _validate_object_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_root: Path,
    sample_mode: bool,
) -> None:
    verify_manifest_hash(receipt)
    if receipt.get("kind") != OBJECT_RECEIPT_KIND or receipt.get("schema_version") != "1.0":
        raise TurkishCorpusError("unexpected object receipt")
    if receipt.get("source_plan_sha256") != plan["canonical_sha256"]:
        raise TurkishCorpusError("object receipt source-plan binding drift")
    if receipt.get("calibration_sha256") != calibration["canonical_sha256"]:
        raise TurkishCorpusError("object receipt calibration binding drift")
    if receipt.get("sample_mode") is not sample_mode:
        raise TurkishCorpusError("object receipt sample/full mode drift")
    if (
        "quality_score_semantics" in receipt
        and receipt.get("quality_score_semantics") != OBJECT_SOURCE_QUALITY_SEMANTICS
    ):
        raise TurkishCorpusError("object receipt quality-score semantics drift")
    rank = receipt.get("rank")
    if not isinstance(rank, int) or rank < 0 or rank >= len(plan["objects"]):
        raise TurkishCorpusError("object receipt rank is invalid")
    expected = plan["objects"][rank]
    if receipt.get("source_id") != expected["source_id"] or receipt.get("source_uri") != expected["uri"]:
        raise TurkishCorpusError("object receipt source identity drift")
    raw_object = _require_mapping(receipt.get("raw_object"), "raw_object")
    if set(raw_object) != {
        "uri",
        "size_bytes",
        "sha256",
        "upstream_checksums_verified",
    }:
        raise TurkishCorpusError("object receipt raw-object shape drift")
    if (
        raw_object.get("uri") != expected["uri"]
        or raw_object.get("size_bytes") != expected["size_bytes"]
        or not _SHA256_RE.fullmatch(str(raw_object.get("sha256", "")))
        or raw_object.get("upstream_checksums_verified")
        != expected["expected_checksums"]
    ):
        raise TurkishCorpusError("object receipt raw-object provenance drift")
    expected_sha256 = next(
        (
            item["value"]
            for item in expected["expected_checksums"]
            if item["algorithm"] == "sha256"
        ),
        None,
    )
    if expected_sha256 is not None and raw_object["sha256"] != expected_sha256:
        raise TurkishCorpusError("object receipt raw-object SHA-256 drift")
    output = _require_mapping(receipt.get("candidate_file"), "candidate_file")
    rows_expected = output.get("rows")
    if (
        isinstance(rows_expected, bool)
        or not isinstance(rows_expected, int)
        or rows_expected < 0
    ):
        raise TurkishCorpusError("object candidate row contract drift")
    with _open_verified_run_artifact(
        run_root,
        output,
        label="object candidate file",
        max_bytes=_MAX_CLUSTER_PARQUET_BYTES,
    ) as (_path, handle):
        parquet = pq.ParquetFile(handle)
        if (
            parquet.metadata.num_rows != rows_expected
            or set(parquet.schema_arrow.names) != set(_INTERNAL_SCHEMA.names)
        ):
            raise TurkishCorpusError("object candidate file drift")
    signatures = receipt.get("signature_files")
    if not isinstance(signatures, list) or len(signatures) != 14:
        raise TurkishCorpusError("object receipt must contain fourteen signatures")
    expected_paths = {
        path.relative_to(run_root).as_posix()
        for path in _signature_files(run_root / "signatures", rank)
    }
    observed_paths = {
        str(record.get("path") or "")
        for record in signatures
        if isinstance(record, Mapping)
    }
    if observed_paths != expected_paths:
        raise TurkishCorpusError("object MinHash signature path drift")
    for index, raw_record in enumerate(signatures):
        record = _require_mapping(raw_record, "signature file")
        with _open_verified_run_artifact(
            run_root,
            record,
            label=f"object MinHash signature {index}",
        ):
            pass


def process_source_object(
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    model_path: str | Path,
    run_dir: str | Path,
    *,
    rank: int,
    sample_mode: bool,
    resource_approval_path: str | Path | None = None,
    scratch_dir: str | Path | None = None,
    request_get: Any = requests.get,
) -> dict[str, Any]:
    """Acquire, verify and GlotLID-score one immutable source object."""

    validate_corpus_policy(policy)
    validate_source_plan(plan, policy)
    validate_backend_calibration(calibration, policy)
    if rank < 0 or rank >= len(plan["objects"]):
        raise TurkishCorpusError(f"object rank {rank} is outside the source plan")
    if sample_mode and rank not in select_resource_sample_ranks(plan):
        raise TurkishCorpusError("sample mode only permits the deterministic per-source sample ranks")
    if not sample_mode:
        if resource_approval_path is None:
            raise TurkishCorpusError("full object processing requires resource approval")
        validate_resource_approval(
            load_json_strict(resource_approval_path),
            plan=plan,
            policy=policy,
            calibration=calibration,
            approval_path=resource_approval_path,
        )
    run_root = Path(run_dir)
    object_dir = run_root / "objects" / f"{rank:05d}"
    receipt_path = object_dir / "object_receipt.json"
    if receipt_path.exists():
        receipt = load_json_strict(receipt_path)
        _validate_object_receipt(
            receipt,
            plan=plan,
            calibration=calibration,
            run_root=run_root,
            sample_mode=sample_mode,
        )
        return receipt
    if object_dir.exists() and any(object_dir.iterdir()):
        raise TurkishCorpusError(f"incomplete object directory requires audit: {object_dir}")
    object_dir.mkdir(parents=True, exist_ok=True)
    signatures_root = run_root / "signatures"
    signatures_root.mkdir(parents=True, exist_ok=True)
    item = plan["objects"][rank]
    source_policy = next(source for source in policy["sources"] if source["id"] == item["source_id"])
    adapter = source_policy["adapter"]
    suffixes = "".join(Path(urllib.parse.urlparse(item["uri"]).path).suffixes)
    scratch_parent = Path(scratch_dir) if scratch_dir is not None else run_root / ".scratch"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    samples = _DeterministicRemovalSamples()
    counts: Counter[str] = Counter()
    stage_counts: dict[str, Counter[str]] = {
        "mixture_selector": Counter(),
        "source_lid": Counter(),
        "independent_glotlid": Counter(),
    }
    start_all_wall, start_all_cpu = time.monotonic(), time.process_time()
    with tempfile.TemporaryDirectory(prefix=f"turkish-{rank:05d}-", dir=scratch_parent) as temporary:
        staged = Path(temporary) / f"source{suffixes}"
        download_wall, download_cpu = time.monotonic(), time.process_time()
        source_file, staged_artifact = _stage_source_object(
            item, staged, request_get=request_get
        )
        download_telemetry = _elapsed(download_wall, download_cpu)
        model = _load_glotlid_model(Path(model_path), policy)
        candidate_path = object_dir / "candidates.parquet"
        writer = _ParquetBatchWriter(candidate_path, _INTERNAL_SCHEMA)
        score_wall, score_cpu = time.monotonic(), time.process_time()
        characters_seen = 0
        bytes_seen = 0
        candidate_chars = 0
        for row_index, record in enumerate(iter_input_records(staged_artifact)):
            counts["documents_seen"] += 1
            raw_text = _nested_get(record, adapter.get("text_field"), "")
            if not isinstance(raw_text, str):
                raw_text = "" if raw_text is None else str(raw_text)
            text = normalize_document(raw_text)
            characters_seen += len(text)
            bytes_seen += len(text.encode("utf-8"))
            document_id_value = _nested_get(record, adapter.get("id_field"), "")
            document_id = str(document_id_value).strip() if document_id_value is not None else ""
            if not document_id:
                document_id = hashlib.sha256(
                    f"{item['source_id']}\x00{item['uri']}\x00{row_index}\x00{canonical_text_hash(text)}".encode(
                        "utf-8"
                    )
                ).hexdigest()
            url = str(_nested_get(record, adapter.get("url_field"), "") or "")
            genre = ""
            scores: Mapping[str, Any] = {}
            wds = item.get("wds_bin")
            if wds is None:
                wds = infer_wds_bin(record)
            if item["source_id"] == MACOCU_SOURCE_ID:
                genre = strict_macocu_genre(
                    record, field=str(adapter.get("genre_field", "genre"))
                )
                routing_record: Mapping[str, Any] = {"genre": genre}
            elif item["source_id"] == "hplt3_tr":
                scores = strict_hplt_register_scores(
                    record,
                    field=str(adapter.get("register_field", "web-register")),
                )
                routing_record = {"wds_bin": wds, "web-register": scores}
            else:
                routing_record = {}
            if item["source_id"] in {MACOCU_SOURCE_ID, "hplt3_tr"}:
                stage_counts["mixture_selector"]["input"] += 1
                if select_mixture_bucket(
                    item["source_id"], routing_record, policy
                ) is None:
                    stage_counts["mixture_selector"]["removed"] += 1
                    samples.add(
                        "mixture_selector",
                        document_id,
                        _redact_sample_text(text),
                        {
                            "source_id": item["source_id"],
                            "genre": genre,
                            "wds_bin": wds,
                        },
                    )
                    continue
                stage_counts["mixture_selector"]["kept"] += 1
            source_label, source_probability, source_passed = _source_lid(record, adapter)
            stage_counts["source_lid"]["input"] += 1
            if not source_passed:
                stage_counts["source_lid"]["removed"] += 1
                samples.add(
                    "source_lid",
                    document_id,
                    _redact_sample_text(text),
                    {"source_id": item["source_id"], "label": source_label, "probability": source_probability},
                )
                continue
            stage_counts["source_lid"]["kept"] += 1
            stage_counts["independent_glotlid"]["input"] += 1
            lid = _document_lid_metrics(model, text, policy)
            if not lid["passed"]:
                stage_counts["independent_glotlid"]["removed"] += 1
                samples.add(
                    "independent_glotlid",
                    document_id,
                    _redact_sample_text(text),
                    {
                        "source_id": item["source_id"],
                        "label": lid["lid_label"],
                        "probability": lid["lid_probability"],
                        "margin": lid["lid_margin"],
                    },
                )
                continue
            stage_counts["independent_glotlid"]["kept"] += 1
            candidate_index = writer.count + len(writer.rows)
            if item["source_id"] != "hplt3_tr":
                scores = register_scores(record)
            web_register = canonical_json(scores) if scores else "{}"
            score = _source_quality_score(record)
            writer.add(
                {
                    "text": text,
                    "source_id": item["source_id"],
                    "document_id": document_id,
                    "url": url,
                    "source_lid_label": source_label,
                    "source_lid_probability": source_probability,
                    "lid_label": lid["lid_label"],
                    "lid_probability": lid["lid_probability"],
                    "lid_margin": lid["lid_margin"],
                    "paragraph_min_probability": lid["paragraph_min_probability"],
                    "paragraph_min_margin": lid["paragraph_min_margin"],
                    "failed_long_paragraph_fraction": lid[
                        "failed_long_paragraph_fraction"
                    ],
                    "dedup_cluster_id": canonical_text_hash(text),
                    "dedup_keep": True,
                    "quality_score": score,
                    "wds_bin": wds,
                    "web-register": web_register,
                    "genre": genre,
                    "pii_replacements": 0,
                    "harmful_signal_hits": 0,
                    "quality_filter_flags": "[]",
                    "formatting_changes": "{}",
                    "candidate_rank": rank,
                    "candidate_doc_index": candidate_index,
                }
            )
            candidate_chars += len(text)
            counts["candidates"] += 1
        staged_artifact.close()
        writer.close()
        candidate_record = _parquet_file_record(
            candidate_path,
            root=run_root,
            expected_rows=writer.count,
            expected_columns=set(_INTERNAL_SCHEMA.names),
        )
        score_telemetry = _elapsed(score_wall, score_cpu)
        removal_sample_record = _write_jsonl_atomic(
            object_dir / "removal_samples.jsonl", samples.rows()
        )
        signature_wall, signature_cpu = time.monotonic(), time.process_time()
        try:
            from datatrove.pipeline.dedup import MinhashDedupSignature
            from datatrove.utils.typeshelper import Languages
        except ImportError as exc:  # pragma: no cover
            raise TurkishCorpusError("pinned DataTrove MinHash is unavailable") from exc
        config = _minhash_config()
        signature_stage = MinhashDedupSignature(
            output_folder=str(signatures_root),
            config=config,
            language=Languages.turkish__latn,
            skip_existing_sigs=False,
        )
        candidate_documents = _iter_candidate_documents(
            run_root,
            candidate_record,
            label=f"object {rank} candidate signature input",
        )
        try:
            signature_stage.run(
                candidate_documents,
                rank=rank,
                world_size=len(plan["objects"]),
            )
        finally:
            candidate_documents.close()
        signature_telemetry = _elapsed(signature_wall, signature_cpu)
    signature_records = [
        _file_record(path, root=run_root) for path in _signature_files(signatures_root, rank)
    ]
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": OBJECT_RECEIPT_KIND,
            "sample_mode": sample_mode,
            "rank": rank,
            "source_id": item["source_id"],
            "source_uri": item["uri"],
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "quality_score_semantics": OBJECT_SOURCE_QUALITY_SEMANTICS,
            "raw_object": source_file,
            "candidate_file": candidate_record,
            "signature_files": signature_records,
            "counts": {
                "documents_seen": counts["documents_seen"],
                "candidates": counts["candidates"],
                "characters_seen": characters_seen,
                "utf8_bytes_seen": bytes_seen,
                "candidate_characters": candidate_chars,
                "stage_counts": {
                    name: dict(sorted(values.items())) for name, values in stage_counts.items()
                },
            },
            "removal_samples": removal_sample_record,
            "telemetry": {
                "download": download_telemetry
                | {"input_bytes": source_file["size_bytes"], "output_bytes": source_file["size_bytes"]},
                "score_lid": score_telemetry
                | {
                    "documents": counts["documents_seen"],
                    "characters": characters_seen,
                    "utf8_bytes": bytes_seen,
                    "input_bytes": source_file["size_bytes"],
                    "output_bytes": candidate_record["size_bytes"],
                },
                "minhash_signature": signature_telemetry
                | {
                    "documents": counts["candidates"],
                    "characters": candidate_chars,
                    "input_bytes": candidate_record["size_bytes"],
                    "output_bytes": sum(item["size_bytes"] for item in signature_records),
                },
                "total": _elapsed(start_all_wall, start_all_cpu),
            },
            "intermediate_contains_unredacted_source_text": True,
            "cleanup_authorized_after_verified_cluster_merge": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(receipt_path, receipt)
    _validate_object_receipt(
        receipt,
        plan=plan,
        calibration=calibration,
        run_root=run_root,
        sample_mode=sample_mode,
    )
    return receipt


def _expected_object_ranks(plan: Mapping[str, Any], sample_mode: bool) -> list[int]:
    return select_resource_sample_ranks(plan) if sample_mode else list(range(len(plan["objects"])))


def _load_object_receipts(
    run_root: Path,
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    sample_mode: bool,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for rank in _expected_object_ranks(plan, sample_mode):
        path = run_root / "objects" / f"{rank:05d}" / "object_receipt.json"
        if not path.is_file():
            raise TurkishCorpusError(f"object rank {rank} is incomplete")
        receipt = load_json_strict(path)
        _validate_object_receipt(
            receipt,
            plan=plan,
            calibration=calibration,
            run_root=run_root,
            sample_mode=sample_mode,
        )
        receipts.append(receipt)
    return receipts


def _validate_bucket_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_root: Path,
    sample_mode: bool,
    objects: Sequence[Mapping[str, Any]],
) -> None:
    verify_manifest_hash(receipt)
    if receipt.get("kind") != BUCKET_RECEIPT_KIND or receipt.get("schema_version") != "1.0":
        raise TurkishCorpusError("unexpected bucket receipt")
    if (
        receipt.get("source_plan_sha256") != plan["canonical_sha256"]
        or receipt.get("calibration_sha256") != calibration["canonical_sha256"]
        or receipt.get("sample_mode") is not sample_mode
        or receipt.get("object_receipt_sha256")
        != [item["canonical_sha256"] for item in objects]
    ):
        raise TurkishCorpusError("bucket receipt binding drift")
    rank = receipt.get("rank")
    if not isinstance(rank, int) or not 0 <= rank < 14:
        raise TurkishCorpusError("bucket rank must be in [0,14)")
    output = _require_mapping(receipt.get("output"), "bucket output")
    if output.get("path") != f"bucket_matches/{rank:05d}_00.dups":
        raise TurkishCorpusError("DataTrove bucket output path drift")
    size, _digest = _artifact_content_contract(output, "DataTrove bucket output")
    if output.get("duplicate_edges") != size // 16 or size % 16:
        raise TurkishCorpusError("DataTrove .dups structural size drift")
    with _open_verified_run_artifact(
        run_root, output, label="DataTrove bucket output"
    ):
        pass


class _DescriptorBinaryReader:
    """Small seekable binary reader backed only by an already-open inode."""

    def __init__(self, descriptor: int, path: str) -> None:
        self._descriptor = os.dup(descriptor)
        self._position = 0
        self.path = path
        self.size = os.fstat(self._descriptor).st_size
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed descriptor reader")
        remaining = self.size - self._position
        wanted = remaining if size is None or size < 0 else min(size, remaining)
        if wanted <= 0:
            return b""
        data = os.pread(self._descriptor, wanted, self._position)
        self._position += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed descriptor reader")
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("invalid whence")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def tell(self) -> int:
        return self._position

    def close(self) -> None:
        if not self.closed:
            os.close(self._descriptor)
            self.closed = True

    def __enter__(self) -> _DescriptorBinaryReader:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


@contextmanager
def _verified_signature_data_folder(
    run_root: Path,
    objects: Sequence[Mapping[str, Any]],
    *,
    bucket_rank: int,
    data_folder_class: type,
) -> Iterator[Any]:
    """Give DataTrove a sealed inventory whose readers duplicate held FDs.

    DataTrove's stock local folder reopens pathnames.  This adapter is still a
    ``DataFolder`` instance (as required by ``get_datafolder``), but its
    ``list_files`` comes only from the receipt and ``open`` returns a pread
    reader over ``os.dup`` of the descriptor that was hashed.  No signature
    pathname is consulted at the actual consumption point.
    """

    with ExitStack() as stack:
        handles: dict[str, Any] = {}
        expected_prefix = f"signatures/bucket_{bucket_rank:03d}/"
        for object_receipt in objects:
            expected_path = (
                f"{expected_prefix}{int(object_receipt['rank']):05d}.minhash.sig"
            )
            logical_path = expected_path.removeprefix("signatures/")
            matches = [
                item
                for item in object_receipt["signature_files"]
                if isinstance(item, Mapping)
                and str(item.get("path") or "") == expected_path
            ]
            if len(matches) != 1:
                raise TurkishCorpusError(
                    "object receipt does not identify exactly one bucket signature"
                )
            record = _require_mapping(matches[0], "bucket signature")
            label = (
                f"object {int(object_receipt['rank'])} MinHash bucket "
                f"{bucket_rank} signature"
            )
            source_path, handle = stack.enter_context(
                _open_verified_run_artifact(run_root, record, label=label)
            )
            if source_path.name != Path(expected_path).name:
                raise TurkishCorpusError("verified signature filename drift")
            handles[logical_path] = handle

        class _DescriptorSignatureDataFolder(data_folder_class):
            def __init__(self) -> None:
                # DataTrove only uses list_files/open/open_files for this input.
                # Deliberately do not initialize a path-backed filesystem.
                self.path = f"descriptor://bucket_{bucket_rank:03d}"
                self.auto_mkdir = False

            def list_files(
                self,
                subdirectory: str = "",
                recursive: bool = True,
                glob_pattern: str | None = None,
                include_directories: bool = False,
            ) -> list[str]:
                if (
                    subdirectory != f"bucket_{bucket_rank:03d}"
                    or recursive is not True
                    or glob_pattern is not None
                    or include_directories is not False
                ):
                    raise TurkishCorpusError(
                        "DataTrove requested an unsealed signature inventory"
                    )
                return sorted(handles)

            def open(
                self,
                path: str,
                mode: str = "rb",
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                if args or mode != "rb" or set(kwargs) - {"block_size"}:
                    raise TurkishCorpusError(
                        "DataTrove requested unsupported signature access"
                    )
                handle = handles.get(str(path))
                if handle is None or handle.closed:
                    raise TurkishCorpusError(
                        "DataTrove requested an unattested signature path"
                    )
                return _DescriptorBinaryReader(handle.fileno(), str(path))

        yield _DescriptorSignatureDataFolder()


def run_datatrove_bucket(
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_dir: str | Path,
    *,
    rank: int,
    sample_mode: bool,
    resource_approval_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one of exactly fourteen DataTrove LSH bucket workers."""

    validate_corpus_policy(policy)
    validate_source_plan(plan, policy)
    validate_backend_calibration(calibration, policy)
    if not 0 <= rank < 14:
        raise TurkishCorpusError("DataTrove bucket rank must be in [0,14)")
    if not sample_mode:
        if resource_approval_path is None:
            raise TurkishCorpusError("full bucket processing requires resource approval")
        validate_resource_approval(
            load_json_strict(resource_approval_path),
            plan=plan,
            policy=policy,
            calibration=calibration,
            approval_path=resource_approval_path,
        )
    run_root = Path(run_dir)
    objects = _load_object_receipts(
        run_root, plan, calibration, sample_mode=sample_mode
    )
    receipt_path = run_root / "bucket_receipts" / f"{rank:05d}.json"
    if receipt_path.exists():
        receipt = load_json_strict(receipt_path)
        _validate_bucket_receipt(
            receipt,
            plan=plan,
            calibration=calibration,
            run_root=run_root,
            sample_mode=sample_mode,
            objects=objects,
        )
        return receipt
    output_root = run_root / "bucket_matches"
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from datatrove.io import DataFolder
        from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError("pinned DataTrove MinHash buckets are unavailable") from exc
    config = _minhash_config()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    with _verified_signature_data_folder(
        run_root,
        objects,
        bucket_rank=rank,
        data_folder_class=DataFolder,
    ) as verified_signatures:
        stage = MinhashDedupBuckets(
            input_folder=verified_signatures,
            output_folder=str(output_root),
            config=config,
            only_dedup_in_index=False,
            lines_to_buffer=256,
        )
        stage.run(None, rank=rank, world_size=14)
    telemetry = _elapsed(start_wall, start_cpu)
    output_path = output_root / f"{rank:05d}_00.dups"
    output = _file_record(output_path, root=run_root)
    output["duplicate_edges"] = output["size_bytes"] // 16
    input_bytes = sum(
        next(
            item["size_bytes"]
            for item in object_receipt["signature_files"]
            if item["path"].startswith(f"signatures/bucket_{rank:03d}/")
        )
        for object_receipt in objects
    )
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": BUCKET_RECEIPT_KIND,
            "sample_mode": sample_mode,
            "rank": rank,
            "world_size": 14,
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "object_receipt_sha256": [item["canonical_sha256"] for item in objects],
            "input_signature_bytes": input_bytes,
            "output": output,
            "telemetry": telemetry
            | {
                "input_bytes": input_bytes,
                "output_bytes": output["size_bytes"],
                "duplicate_edges": output["duplicate_edges"],
            },
            "canonical_sha256": None,
        }
    )
    write_json_atomic(receipt_path, receipt)
    _validate_bucket_receipt(
        receipt,
        plan=plan,
        calibration=calibration,
        run_root=run_root,
        sample_mode=sample_mode,
        objects=objects,
    )
    return receipt


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}
        self.size: dict[tuple[int, int], int] = {}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent.get(node)
        if parent is None:
            self.parent[node] = node
            self.size[node] = 1
            return node
        trail: list[tuple[int, int]] = []
        while parent != self.parent[parent]:
            trail.append(parent)
            parent = self.parent[parent]
        root = parent
        for item in trail:
            self.parent[item] = root
        self.parent[node] = root
        return root

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        size_left = self.size[root_left]
        size_right = self.size[root_right]
        if (size_left, root_left) < (size_right, root_right):
            root_left, root_right = root_right, root_left
            size_left, size_right = size_right, size_left
        self.parent[root_right] = root_left
        self.size[root_left] = size_left + size_right
        del self.size[root_right]


def _iter_candidate_rows(
    run_root: Path, record: Mapping[str, Any], *, label: str
) -> Iterator[dict[str, Any]]:
    with _open_verified_run_artifact(
        run_root,
        record,
        label=label,
        max_bytes=_MAX_CLUSTER_PARQUET_BYTES,
    ) as (_path, handle):
        parquet = pq.ParquetFile(handle)
        rows_expected = record.get("rows")
        if (
            isinstance(rows_expected, bool)
            or not isinstance(rows_expected, int)
            or rows_expected < 0
            or parquet.metadata.num_rows != rows_expected
            or set(parquet.schema_arrow.names) != set(_INTERNAL_SCHEMA.names)
        ):
            raise TurkishCorpusError(f"{label} Parquet metadata drift")
        rows_seen = 0
        for batch in parquet.iter_batches(batch_size=2048):
            rows = batch.to_pylist()
            rows_seen += len(rows)
            yield from rows
        if rows_seen != rows_expected:
            raise TurkishCorpusError(f"{label} Parquet row scan drift")


def _iter_duplicate_edges(
    run_root: Path, record: Mapping[str, Any], *, label: str
) -> Iterator[tuple[int, int, int, int]]:
    size, _digest = _artifact_content_contract(record, label)
    if size % 16 or record.get("duplicate_edges") != size // 16:
        raise TurkishCorpusError("DataTrove .dups structural size drift")
    seen = 0
    with _open_verified_run_artifact(run_root, record, label=label) as (_path, handle):
        while chunk := handle.read(16):
            if len(chunk) != 16:
                raise TurkishCorpusError("truncated DataTrove duplicate edge")
            seen += 1
            yield struct.unpack("<4I", chunk)
    if seen != record["duplicate_edges"]:
        raise TurkishCorpusError("DataTrove duplicate-edge count drift")


def _load_bucket_receipts(
    run_root: Path,
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    sample_mode: bool,
    objects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for rank in range(14):
        path = run_root / "bucket_receipts" / f"{rank:05d}.json"
        if not path.is_file():
            raise TurkishCorpusError(f"DataTrove bucket {rank} is incomplete")
        receipt = load_json_strict(path)
        _validate_bucket_receipt(
            receipt,
            plan=plan,
            calibration=calibration,
            run_root=run_root,
            sample_mode=sample_mode,
            objects=objects,
        )
        receipts.append(receipt)
    return receipts


def _cluster_id(document_id: str) -> str:
    return hashlib.sha256(
        f"datatrove-priority-cluster-v1\x00{document_id}".encode("utf-8")
    ).hexdigest()


def _parse_flags(value: str) -> list[str]:
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return ["malformed_quality_flags"]
    return [str(item) for item in result] if isinstance(result, list) else ["malformed_quality_flags"]


def run_priority_cluster_merge(
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_dir: str | Path,
    *,
    sample_mode: bool,
    resource_approval_path: str | Path | None = None,
) -> dict[str, Any]:
    """Cluster DataTrove edges and apply the frozen priority winner policy."""

    validate_corpus_policy(policy)
    validate_source_plan(plan, policy)
    validate_backend_calibration(calibration, policy)
    expected_processing = production_processing_binding(policy)
    expected_code_identity = production_code_identity()
    if not sample_mode:
        if resource_approval_path is None:
            raise TurkishCorpusError("full priority clustering requires resource approval")
        validate_resource_approval(
            load_json_strict(resource_approval_path),
            plan=plan,
            policy=policy,
            calibration=calibration,
            approval_path=resource_approval_path,
        )
    run_root = Path(run_dir)
    objects = _load_object_receipts(
        run_root, plan, calibration, sample_mode=sample_mode
    )
    buckets = _load_bucket_receipts(
        run_root, plan, calibration, sample_mode=sample_mode, objects=objects
    )
    object_receipt_hashes = [item["canonical_sha256"] for item in objects]
    bucket_receipt_hashes = [item["canonical_sha256"] for item in buckets]
    legacy_quality_score_neutralized_ranks = sorted(
        int(item["rank"])
        for item in objects
        if item.get("quality_score_semantics") != OBJECT_SOURCE_QUALITY_SEMANTICS
    )
    sample_launches = (
        _sample_launch_bindings(
            run_root,
            plan=plan,
            calibration=calibration,
            objects=objects,
            buckets=buckets,
        )
        if sample_mode
        else None
    )
    receipt_path = run_root / "cluster_receipt.json"
    if receipt_path.exists():
        receipt = load_json_strict(receipt_path)
        verify_manifest_hash(receipt)
        validate_production_code_identity(receipt.get("code_identity"))
        if (
            receipt.get("kind") != CLUSTER_RECEIPT_KIND
            or receipt.get("source_plan_sha256") != plan["canonical_sha256"]
            or receipt.get("calibration_sha256") != calibration["canonical_sha256"]
            or receipt.get("sample_mode") is not sample_mode
            or receipt.get("processing") != expected_processing
            or receipt.get("sample_launch_receipts") != sample_launches
            or receipt.get("object_receipt_sha256") != object_receipt_hashes
            or receipt.get("bucket_receipt_sha256") != bucket_receipt_hashes
            or receipt.get("winner_policy") != CLUSTER_WINNER_POLICY
            or receipt.get("quality_score_semantics")
            != CLUSTER_QUALITY_SCORE_SEMANTICS
            or receipt.get("legacy_quality_score_neutralized_ranks")
            != legacy_quality_score_neutralized_ranks
        ):
            raise TurkishCorpusError("cluster receipt binding drift")
        for index, raw_item in enumerate(receipt["output_files"]):
            item = _require_mapping(raw_item, "cluster output")
            rows = item.get("rows")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise TurkishCorpusError("cluster output row contract drift")
            with _open_verified_run_artifact(
                run_root,
                item,
                label=f"cluster output {index}",
                max_bytes=_MAX_CLUSTER_PARQUET_BYTES,
            ) as (_path, handle):
                parquet = pq.ParquetFile(handle)
                if (
                    parquet.metadata.num_rows != rows
                    or set(parquet.schema_arrow.names) != set(_BACKEND_SCHEMA.names)
                ):
                    raise TurkishCorpusError("cluster output drift")
        return receipt
    output_root = run_root / "backend_output"
    if output_root.exists() and any(output_root.iterdir()):
        raise TurkishCorpusError("incomplete backend output requires audit before retry")
    output_root.mkdir(parents=True, exist_ok=True)
    start_wall, start_cpu = time.monotonic(), time.process_time()
    union = _UnionFind()
    edge_count = 0
    for bucket in buckets:
        for f1, d1, f2, d2 in _iter_duplicate_edges(
            run_root,
            _require_mapping(bucket.get("output"), "bucket output"),
            label=f"DataTrove bucket {int(bucket['rank'])} duplicate edges",
        ):
            if f1 == (1 << 32) - 1 or f2 == (1 << 32) - 1:
                raise TurkishCorpusError("unexpected external-index sentinel in global dedup")
            union.union((f1, d1), (f2, d2))
            edge_count += 1
    priority = {
        source_id: index
        for index, source_id in enumerate(policy["deduplication"]["source_priority"])
    }
    # Singleton candidates are always their own winner, so only candidates
    # participating in a duplicate edge need resident winner state.
    winners: dict[tuple[int, int], tuple[tuple[Any, ...], tuple[int, int], str]] = {}
    seen_edge_nodes: set[tuple[int, int]] = set()
    for object_receipt in objects:
        candidate = _require_mapping(
            object_receipt.get("candidate_file"), "object candidate file"
        )
        for row in _iter_candidate_rows(
            run_root,
            candidate,
            label=f"object {int(object_receipt['rank'])} winner candidate input",
        ):
            node = (int(row["candidate_rank"]), int(row["candidate_doc_index"]))
            if node[0] != object_receipt["rank"]:
                raise TurkishCorpusError("candidate rank/file identity drift")
            if node not in union.parent:
                continue
            root = union.find(node)
            source_quality = _attested_source_quality(row, object_receipt)
            key = (
                priority[row["source_id"]],
                -source_quality,
                str(row["document_id"]),
                node,
            )
            current = winners.get(root)
            if current is None or key < current[0]:
                winners[root] = (key, node, str(row["document_id"]))
            seen_edge_nodes.add(node)
    missing_edge_nodes = set(union.parent) - seen_edge_nodes
    if missing_edge_nodes:
        raise TurkishCorpusError(
            f"DataTrove duplicate edges reference {len(missing_edge_nodes)} absent candidate rows"
        )
    processors = _ProductionProcessors(policy)
    if processors.binding != expected_processing:
        raise TurkishCorpusError("production processor construction binding drift")
    samples = _DeterministicRemovalSamples(
        capacity=int(policy["quality_assurance"]["examples_per_stratum_and_decision"]),
        max_chars=int(policy["quality_assurance"]["max_example_characters"]),
    )
    counters: Counter[str] = Counter()
    filter_counts: dict[str, Counter[str]] = {
        "gopher_repetition": Counter(),
        "fineweb_quality": Counter(),
        "gopher_quality": Counter(),
        "project_local_audit": Counter(),
        "project_mixture_selector": Counter(),
    }
    formatting_counts: Counter[str] = Counter()
    output_files: list[dict[str, Any]] = []
    total_output_rows = 0
    for object_receipt in objects:
        candidate = _require_mapping(
            object_receipt.get("candidate_file"), "object candidate file"
        )
        output_path = output_root / f"{object_receipt['rank']:05d}.parquet"
        writer = _ParquetBatchWriter(output_path, _BACKEND_SCHEMA)
        for row in _iter_candidate_rows(
            run_root,
            candidate,
            label=f"object {int(object_receipt['rank'])} cluster candidate input",
        ):
            row["quality_score"] = _attested_source_quality(row, object_receipt)
            node = (int(row.pop("candidate_rank")), int(row.pop("candidate_doc_index")))
            if node in union.parent:
                winner = winners[union.find(node)]
                dedup_keep = node == winner[1]
                winner_document_id = winner[2]
            else:
                dedup_keep = True
                winner_document_id = str(row["document_id"])
            row["dedup_cluster_id"] = _cluster_id(winner_document_id)
            row["dedup_keep"] = dedup_keep
            flags: list[str] = []
            if not dedup_keep:
                counters["dedup_removed"] += 1
                samples.add(
                    "datatrove_minhash_duplicate",
                    row["document_id"],
                    _redact_sample_text(row["text"]),
                    {"source_id": row["source_id"], "cluster_id": row["dedup_cluster_id"]},
                )
            else:
                counters["dedup_kept"] += 1
                official_flags, trace = processors.evaluate_filters(row["text"])
                flags.extend(official_flags)
                for step in trace:
                    name = step["stage"]
                    if step["evaluated"]:
                        filter_counts[name]["input"] += 1
                        if step["passed"]:
                            filter_counts[name]["kept"] += 1
                        else:
                            filter_counts[name]["removed"] += 1
                            samples.add(
                                f"official_{name}:{step['reason']}",
                                row["document_id"],
                                _redact_sample_text(row["text"]),
                                {"source_id": row["source_id"]},
                            )
                if not flags:
                    filter_counts["project_local_audit"]["input"] += 1
            formatted, formatting = processors.format_safely(row["text"])
            for key, value in formatting.items():
                formatting_counts[key] += int(value)
            row["text"] = normalize_document(formatted)
            row["pii_replacements"] = int(
                formatting["email_matches"]
                + formatting["ip_like_matches"]
                + formatting["phone_replacements"]
            )
            row["harmful_signal_hits"] = int(formatting["harmful_signal_hits"])
            row["formatting_changes"] = canonical_json(formatting)
            if dedup_keep and not flags:
                audit = audit_document(
                    row["text"],
                    url=row["url"],
                    source_lid_ok=True,
                    content_policy=policy["content_policy"],
                )
                if audit.accepted:
                    filter_counts["project_local_audit"]["kept"] += 1
                    row["text"] = audit.normalized_text
                    filter_counts["project_mixture_selector"]["input"] += 1
                    routed = select_mixture_bucket(row["source_id"], row, policy)
                    if routed is None:
                        filter_counts["project_mixture_selector"]["removed"] += 1
                        flags.append("project_mixture_selector:unrouted")
                        samples.add(
                            "project_mixture_selector:unrouted",
                            row["document_id"],
                            row["text"],
                            {"source_id": row["source_id"]},
                        )
                    else:
                        filter_counts["project_mixture_selector"]["kept"] += 1
                else:
                    filter_counts["project_local_audit"]["removed"] += 1
                    flags.append(f"project_local_audit:{audit.reason}")
                    samples.add(
                        f"project_local_audit:{audit.reason}",
                        row["document_id"],
                        row["text"],
                        {"source_id": row["source_id"]},
                    )
            row["quality_filter_flags"] = canonical_json(flags)
            if dedup_keep and not flags:
                counters["quality_kept"] += 1
            elif dedup_keep:
                counters["quality_removed"] += 1
            counters["output_rows"] += 1
            writer.add({name: row[name] for name in BACKEND_COLUMNS})
        writer.close()
        rows = writer.count
        if rows:
            output_files.append(
                _parquet_file_record(
                    output_path,
                    root=run_root,
                    expected_rows=rows,
                    expected_columns=set(_BACKEND_SCHEMA.names),
                )
                | {"source_rank": int(object_receipt["rank"])}
            )
            total_output_rows += rows
        else:
            output_path.unlink()
    removal_samples = _write_jsonl_atomic(
        run_root / "cluster_removal_samples.jsonl", samples.rows()
    )
    telemetry = _elapsed(start_wall, start_cpu) | {
        "input_documents": sum(item["candidate_file"]["rows"] for item in objects),
        "duplicate_edges": edge_count,
        "output_documents": total_output_rows,
        "input_bytes": sum(item["candidate_file"]["size_bytes"] for item in objects)
        + sum(item["output"]["size_bytes"] for item in buckets),
        "output_bytes": sum(item["size_bytes"] for item in output_files),
        "peak_rss_bytes": _peak_rss_bytes(),
        "edge_participating_documents": len(seen_edge_nodes),
        "winner_components": len(winners),
    }
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": CLUSTER_RECEIPT_KIND,
            "sample_mode": sample_mode,
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "object_receipt_sha256": object_receipt_hashes,
            "bucket_receipt_sha256": bucket_receipt_hashes,
            "winner_policy": CLUSTER_WINNER_POLICY,
            "quality_score_semantics": CLUSTER_QUALITY_SCORE_SEMANTICS,
            "legacy_quality_score_neutralized_ranks": (
                legacy_quality_score_neutralized_ranks
            ),
            "processing": processors.binding,
            "code_identity": expected_code_identity,
            "sample_launch_receipts": sample_launches,
            "duplicate_edges": edge_count,
            "counts": dict(sorted(counters.items())),
            "filter_stage_counts": {
                name: dict(sorted(values.items())) for name, values in filter_counts.items()
            },
            "formatting_and_safety_incidence": dict(sorted(formatting_counts.items())),
            "removal_samples": removal_samples,
            "output_files": output_files,
            "telemetry": telemetry,
            "intermediate_cleanup_authorized_after_backend_receipt": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(receipt_path, receipt)
    return receipt


def _validate_production_cluster_launch(
    path: str | Path,
    *,
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    launch_raw = _read_bounded_regular_file_snapshot(
        Path(path),
        label="production cluster launch receipt",
        max_bytes=_MAX_RECEIPT_EVIDENCE_BYTES,
    )
    launch = _load_json_snapshot(launch_raw, "production cluster launch receipt")
    digest = verify_manifest_hash(launch)
    cluster_raw = _read_bounded_regular_file_snapshot(
        run_root / "cluster_receipt.json",
        label="production cluster receipt",
        max_bytes=_MAX_RECEIPT_EVIDENCE_BYTES,
    )
    cluster = _load_json_snapshot(cluster_raw, "production cluster receipt")
    cluster_sha = verify_manifest_hash(cluster)
    if (
        launch.get("schema_version") != "2.0"
        or launch.get("kind") != PRODUCTION_CLUSTER_LAUNCH_KIND
        or launch.get("cluster_completed") is not True
        or launch.get("policy_sha256") != _policy_sha256(policy)
        or launch.get("source_plan_sha256") != plan["canonical_sha256"]
        or launch.get("calibration_sha256") != calibration["canonical_sha256"]
        or launch.get("cluster_receipt_sha256") != cluster_sha
        or cluster.get("schema_version") != "1.0"
        or cluster.get("kind") != CLUSTER_RECEIPT_KIND
        or cluster.get("sample_mode") is not False
        or cluster.get("source_plan_sha256") != plan["canonical_sha256"]
        or cluster.get("calibration_sha256") != calibration["canonical_sha256"]
    ):
        raise TurkishCorpusError("production cluster launch receipt binding drift")
    for field in (
        "recipe_sha256",
        "production_pack_plan_sha256",
        "resource_approval_sha256",
        "mixture_quality_approval_sha256",
        "data_prep_storage_gate_sha256",
        "sample_cluster_receipt_sha256",
        "cluster_input_receipt_sha256",
    ):
        if not _SHA256_RE.fullmatch(str(launch.get(field, ""))):
            raise TurkishCorpusError("production cluster launch provenance drift")
    allocation = launch.get("allocation")
    if (
        not isinstance(allocation, Mapping)
        or allocation.get("partition") != "cpu2dq"
        or allocation.get("nodes") != 1
        or allocation.get("tasks") != 1
        or allocation.get("cpus_per_task") != 16
        or allocation.get("billable_cpus") != 128
        or allocation.get("memory_bytes") != 192 * 1024**3
        or allocation.get("maximum_wall_seconds") != 172_800
        or not str(allocation.get("slurm_job_id", "")).isdigit()
        or not allocation.get("node_list")
    ):
        raise TurkishCorpusError("production cluster launch allocation drift")
    return launch, digest, cluster


def seal_source_receipt_from_objects(
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_dir: str | Path,
    cluster_launch_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal source SHA-256 identities after every bounded acquisition task."""

    validate_corpus_policy(policy)
    validate_source_plan(plan, policy)
    validate_backend_calibration(calibration, policy)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite source receipt: {destination}")
    run_root = Path(run_dir)
    cluster_launch, cluster_launch_sha, cluster = _validate_production_cluster_launch(
        cluster_launch_receipt_path,
        policy=policy,
        plan=plan,
        calibration=calibration,
        run_root=run_root,
    )
    objects = _load_object_receipts(
        run_root, plan, calibration, sample_mode=False
    )
    object_receipt_hashes = [item["canonical_sha256"] for item in objects]
    if cluster.get("object_receipt_sha256") != object_receipt_hashes:
        raise TurkishCorpusError("source receipt object/cluster binding drift")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for receipt in objects:
        by_source[receipt["source_id"]].append(
            {
                "uri": receipt["source_uri"],
                "checksum": {
                    "algorithm": "sha256",
                    "value": receipt["raw_object"]["sha256"],
                },
                "size_bytes": receipt["raw_object"]["size_bytes"],
            }
        )
    sources = []
    for source in policy["sources"]:
        files = sorted(by_source[source["id"]], key=lambda item: item["uri"])
        if not files:
            raise TurkishCorpusError(f"source {source['id']} has no verified objects")
        sources.append(
            {
                "id": source["id"],
                "repo_id": source["repo_id"],
                "resolved_revision": source["resolved_revision"],
                "license_id": source["license_id"],
                "files": files,
            }
        )
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": SOURCE_RECEIPT_KIND,
            "policy_sha256": _policy_sha256(policy),
            "source_plan_sha256": plan["canonical_sha256"],
            "derived_sources": plan.get("derived_sources", {}),
            "object_receipt_sha256": object_receipt_hashes,
            "production_chain": {
                "cluster_launch_receipt_sha256": cluster_launch_sha,
                "production_pack_plan_sha256": cluster_launch[
                    "production_pack_plan_sha256"
                ],
                "resource_approval_sha256": cluster_launch[
                    "resource_approval_sha256"
                ],
                "mixture_quality_approval_sha256": cluster_launch[
                    "mixture_quality_approval_sha256"
                ],
                "data_prep_storage_gate_sha256": cluster_launch[
                    "data_prep_storage_gate_sha256"
                ],
                "sample_cluster_receipt_sha256": cluster_launch[
                    "sample_cluster_receipt_sha256"
                ],
            },
            "sources": sources,
            "canonical_sha256": None,
        }
    )
    validate_source_receipt(receipt, policy)
    write_json_atomic(destination, receipt)
    return receipt


def seal_backend_receipt_from_cluster(
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_dir: str | Path,
    cluster_launch_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    validate_corpus_policy(policy)
    validate_source_plan(plan, policy)
    validate_source_receipt(source_receipt, policy)
    validate_backend_calibration(calibration, policy)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite backend receipt: {destination}")
    run_root = Path(run_dir)
    cluster_launch, cluster_launch_sha, cluster = _validate_production_cluster_launch(
        cluster_launch_receipt_path,
        policy=policy,
        plan=plan,
        calibration=calibration,
        run_root=run_root,
    )
    expected_chain = {
        "cluster_launch_receipt_sha256": cluster_launch_sha,
        "production_pack_plan_sha256": cluster_launch[
            "production_pack_plan_sha256"
        ],
        "resource_approval_sha256": cluster_launch["resource_approval_sha256"],
        "mixture_quality_approval_sha256": cluster_launch[
            "mixture_quality_approval_sha256"
        ],
        "data_prep_storage_gate_sha256": cluster_launch[
            "data_prep_storage_gate_sha256"
        ],
        "sample_cluster_receipt_sha256": cluster_launch[
            "sample_cluster_receipt_sha256"
        ],
    }
    if source_receipt.get("production_chain") != expected_chain:
        raise TurkishCorpusError("source receipt production-chain binding drift")
    if cluster.get("kind") != CLUSTER_RECEIPT_KIND or cluster.get("sample_mode") is not False:
        raise TurkishCorpusError("production cluster receipt is missing/invalid")
    if cluster.get("source_plan_sha256") != plan["canonical_sha256"]:
        raise TurkishCorpusError("cluster/source-plan binding drift")
    if (
        cluster.get("processing") != production_processing_binding(policy)
    ):
        raise TurkishCorpusError("cluster frozen processing/code binding drift")
    validate_production_code_identity(cluster.get("code_identity"))
    objects = _load_object_receipts(
        run_root, plan, calibration, sample_mode=False
    )
    object_receipt_hashes = [item["canonical_sha256"] for item in objects]
    if (
        cluster.get("object_receipt_sha256") != object_receipt_hashes
        or source_receipt.get("object_receipt_sha256") != object_receipt_hashes
    ):
        raise TurkishCorpusError("source/object/cluster receipt binding drift")
    buckets = _load_bucket_receipts(
        run_root, plan, calibration, sample_mode=False, objects=objects
    )
    neutralized_ranks = sorted(
        int(item["rank"])
        for item in objects
        if item.get("quality_score_semantics") != OBJECT_SOURCE_QUALITY_SEMANTICS
    )
    if (
        cluster.get("bucket_receipt_sha256")
        != [item["canonical_sha256"] for item in buckets]
        or cluster.get("winner_policy") != CLUSTER_WINNER_POLICY
        or cluster.get("quality_score_semantics")
        != CLUSTER_QUALITY_SCORE_SEMANTICS
        or cluster.get("legacy_quality_score_neutralized_ranks")
        != neutralized_ranks
    ):
        raise TurkishCorpusError("cluster quality/dedup provenance drift")
    file_records: list[dict[str, Any]] = []
    for index, raw_item in enumerate(cluster["output_files"]):
        item = _require_mapping(raw_item, "cluster output")
        source_rank = item.get("source_rank")
        if isinstance(source_rank, bool) or not isinstance(source_rank, int):
            raise TurkishCorpusError("cluster backend file lacks exact source rank")
        rows = item.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise TurkishCorpusError("cluster backend file row contract drift")
        with _open_verified_run_artifact(
            run_root,
            item,
            label=f"production cluster output {index}",
            max_bytes=_MAX_CLUSTER_PARQUET_BYTES,
        ) as (path, handle):
            parquet = pq.ParquetFile(handle)
            if (
                parquet.metadata.num_rows != rows
                or set(parquet.schema_arrow.names) != set(_BACKEND_SCHEMA.names)
            ):
                raise TurkishCorpusError(
                    "cluster output file drift before backend seal"
                )
            file_records.append(
                {
                    "uri": path.as_uri(),
                    "checksum": {
                        "algorithm": "sha256",
                        "value": item["sha256"],
                    },
                    "size_bytes": item["size_bytes"],
                    "rows": rows,
                    "source_rank": source_rank,
                }
            )
    if not file_records:
        raise TurkishCorpusError("production cluster emitted no backend files")
    inventory = source_object_inventory(source_receipt)
    expected_lid = policy["language_policy"]["independent_audit"]
    expected_dedup = policy["deduplication"]["production_backend"]
    lid = {
        key: expected_lid[key]
        for key in (
            "implementation",
            "repo_id",
            "hub_revision",
            "artifact",
            "artifact_sha256",
            "required_top_label",
            "document_min_probability",
            "document_min_margin",
            "paragraph_min_probability",
            "paragraph_min_margin",
            "max_failed_long_paragraph_fraction",
        )
    }
    dedup = {
        key: expected_dedup[key]
        for key in (
            "implementation",
            "version",
            "git_revision",
            "precision_bits",
            "num_buckets",
            "hashes_per_bucket",
            "ngram_words",
            "signature_language",
            "match_rule",
            "candidate_probability_formula",
            "similarity_at_50_percent_candidate_probability",
            "candidate_probability_at_similarity_0_82",
            "synthetic_similarity_calibration_required",
            "cluster_representative",
        )
    }
    dedup.update(
        {
            "global_cross_source": True,
            "winner_priority": list(policy["deduplication"]["source_priority"]),
            "synthetic_similarity_calibration": calibration[
                "synthetic_similarity_calibration"
            ],
            "signature_tokenizer_probe": calibration["signature_tokenizer_probe"],
        }
    )
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": BACKEND_RECEIPT_KIND,
            "policy_sha256": _policy_sha256(policy),
            "source_receipt_sha256": source_receipt["canonical_sha256"],
            "source_plan_sha256": plan["canonical_sha256"],
            "source_inventory": inventory,
            "source_inventory_sha256": hashlib.sha256(
                canonical_json(inventory).encode("utf-8")
            ).hexdigest(),
            "source_totals": {
                "objects": len(inventory),
                "size_bytes": sum(item["size_bytes"] for item in inventory),
            },
            "lid": lid,
            "lid_calibration": calibration["lid_calibration"],
            "dedup": dedup,
            "processing": cluster["processing"],
            "quality_filter_stage_counts": cluster["filter_stage_counts"],
            "formatting_and_safety_incidence": cluster[
                "formatting_and_safety_incidence"
            ],
            "removal_samples": cluster["removal_samples"],
            "cluster_receipt_sha256": cluster["canonical_sha256"],
            "production_chain": expected_chain,
            "columns": list(BACKEND_COLUMNS),
            "files": file_records,
            "output_totals": {
                "files": len(file_records),
                "rows": sum(item["rows"] for item in file_records),
                "size_bytes": sum(item["size_bytes"] for item in file_records),
            },
            "streaming_import_cleanup": {
                "backend_files_are_run_owned": True,
                "automatic_deletion_during_pool_import": False,
                "explicit_verified_cleanup_required": True,
                "bounded_extra_copy": (
                    "backend_output_plus_open_pool_fragments; conservative resource "
                    "projection includes candidates/signatures/edges/backend output"
                ),
            },
            "canonical_sha256": None,
        }
    )
    validate_backend_receipt(receipt, policy, source_receipt)
    write_json_atomic(destination, receipt)
    return receipt


def _sum_telemetry(receipts: Sequence[Mapping[str, Any]], stage: str) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for receipt in receipts:
        metrics = _require_mapping(receipt["telemetry"].get(stage), f"telemetry.{stage}")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += float(value)
    return dict(sorted(totals.items()))


_RESOURCE_STAGE_NAMES = (
    "download",
    "score_lid",
    "minhash_signature",
    "minhash_buckets",
    "priority_cluster_quality_format",
)


def _resource_projection_accounting(
    stage_wall_seconds: Mapping[str, Any],
    stage_process_cpu_seconds: Mapping[str, Any],
    *,
    safety_factor: float,
    billable_cpus_per_job: int,
) -> dict[str, Any]:
    """Convert projected one-node job wall time to UHeM's billed CPU-saat."""

    if (
        isinstance(billable_cpus_per_job, bool)
        or not isinstance(billable_cpus_per_job, int)
        or billable_cpus_per_job <= 0
    ):
        raise TurkishCorpusError("billable_cpus_per_job must be a positive integer")
    if set(stage_wall_seconds) != set(_RESOURCE_STAGE_NAMES) or set(
        stage_process_cpu_seconds
    ) != set(_RESOURCE_STAGE_NAMES):
        raise TurkishCorpusError("resource projection stage accounting is incomplete")
    if (
        isinstance(safety_factor, bool)
        or not isinstance(safety_factor, (int, float))
        or not math.isfinite(float(safety_factor))
        or not 1.0 <= float(safety_factor) <= 3.0
    ):
        raise TurkishCorpusError("invalid resource-projection safety factor")

    def normalize(values: Mapping[str, Any], label: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for stage in _RESOURCE_STAGE_NAMES:
            value = values[stage]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise TurkishCorpusError(f"{label}.{stage} must be finite and non-negative")
            result[stage] = float(value)
        return result

    wall = normalize(stage_wall_seconds, "stage_wall_seconds")
    process_cpu = normalize(stage_process_cpu_seconds, "stage_process_cpu_seconds")
    total_wall = sum(wall.values())
    if total_wall <= 0:
        raise TurkishCorpusError("projected stage wall time must be positive")
    total_process_cpu = sum(process_cpu.values())
    billed_capacity_seconds = total_wall * billable_cpus_per_job
    wall_with_safety = total_wall * float(safety_factor)
    billed_cpu_seconds_with_safety = wall_with_safety * billable_cpus_per_job
    return {
        "stage_wall_seconds_before_safety_factor": wall,
        "wall_seconds_with_safety_factor": wall_with_safety,
        "stage_billed_cpu_saat_before_safety_factor": {
            stage: seconds * billable_cpus_per_job / 3600.0
            for stage, seconds in wall.items()
        },
        "billed_cpu_seconds_with_safety_factor": billed_cpu_seconds_with_safety,
        "billed_cpu_saat_with_safety_factor": billed_cpu_seconds_with_safety / 3600.0,
        "diagnostic_process_cpu": {
            "stage_process_cpu_seconds_before_safety_factor": process_cpu,
            "process_cpu_seconds_with_safety_factor": total_process_cpu
            * float(safety_factor),
            "process_cpu_efficiency_against_billable_capacity": (
                total_process_cpu / billed_capacity_seconds
            ),
        },
    }


def validate_resource_projection(report: Mapping[str, Any]) -> str:
    """Validate the sealed wall-time billing arithmetic used for manual approval."""

    report_hash = verify_manifest_hash(report)
    if report.get("schema_version") != "2.0" or report.get("kind") != RESOURCE_REPORT_KIND:
        raise TurkishCorpusError("unexpected resource projection kind/version")
    for key in (
        "policy_sha256",
        "source_plan_sha256",
        "calibration_sha256",
        "sample_cluster_receipt_sha256",
    ):
        if not _SHA256_RE.fullmatch(str(report.get(key, ""))):
            raise TurkishCorpusError(f"resource projection {key} is missing")
    contract = _require_mapping(report.get("billing_contract"), "billing_contract")
    if dict(contract) != RESOURCE_BILLING_CONTRACT:
        raise TurkishCorpusError("resource projection billing contract drift")
    projection = _require_mapping(report.get("projection"), "projection")
    diagnostics = _require_mapping(
        projection.get("diagnostic_process_cpu"), "projection.diagnostic_process_cpu"
    )
    expected = _resource_projection_accounting(
        _require_mapping(
            projection.get("stage_wall_seconds_before_safety_factor"),
            "projection.stage_wall_seconds_before_safety_factor",
        ),
        _require_mapping(
            diagnostics.get("stage_process_cpu_seconds_before_safety_factor"),
            "projection.diagnostic_process_cpu.stage_process_cpu_seconds_before_safety_factor",
        ),
        safety_factor=projection.get("safety_factor"),
        billable_cpus_per_job=contract["billable_cpus_per_job"],
    )

    def require_close(actual: Any, wanted: float, label: str) -> None:
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(float(actual), wanted, rel_tol=1e-12, abs_tol=1e-9)
        ):
            raise TurkishCorpusError(f"resource projection {label} arithmetic drift")

    component_names = {
        "raw_largest_object_bytes",
        "candidate_bytes",
        "signature_bytes",
        "duplicate_edge_bytes",
        "backend_output_bytes",
    }
    components = _require_mapping(
        projection.get("peak_disk_components_before_safety_factor"),
        "projection.peak_disk_components_before_safety_factor",
    )
    if set(components) != component_names:
        raise TurkishCorpusError("resource projection peak-disk components drift")
    numeric_components: dict[str, float] = {}
    for name in sorted(component_names):
        raw_value = components.get(name)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
            or float(raw_value) < 0
        ):
            raise TurkishCorpusError(
                f"resource projection peak-disk component {name} drift"
            )
        numeric_components[name] = float(raw_value)
        require_close(projection.get(name), numeric_components[name], name)
    peak_before_safety = sum(numeric_components.values())
    require_close(
        projection.get("peak_disk_bytes_before_safety_factor"),
        peak_before_safety,
        "peak disk before safety factor",
    )
    require_close(
        projection.get("peak_disk_bytes_with_safety_factor"),
        peak_before_safety * float(projection["safety_factor"]),
        "peak disk with safety factor",
    )
    if projection.get("peak_disk_model") != RESOURCE_PEAK_DISK_MODEL:
        raise TurkishCorpusError("resource projection peak-disk model drift")

    for key in (
        "wall_seconds_with_safety_factor",
        "billed_cpu_seconds_with_safety_factor",
        "billed_cpu_saat_with_safety_factor",
    ):
        require_close(projection.get(key), expected[key], key)
    billed_by_stage = _require_mapping(
        projection.get("stage_billed_cpu_saat_before_safety_factor"),
        "projection.stage_billed_cpu_saat_before_safety_factor",
    )
    if set(billed_by_stage) != set(_RESOURCE_STAGE_NAMES):
        raise TurkishCorpusError("resource projection billed stage accounting is incomplete")
    for stage, wanted in expected["stage_billed_cpu_saat_before_safety_factor"].items():
        require_close(billed_by_stage.get(stage), wanted, f"stage billed CPU-saat ({stage})")
    expected_diagnostics = expected["diagnostic_process_cpu"]
    for key in (
        "process_cpu_seconds_with_safety_factor",
        "process_cpu_efficiency_against_billable_capacity",
    ):
        require_close(diagnostics.get(key), expected_diagnostics[key], key)
    cluster_scaling = _require_mapping(
        projection.get("cluster_scaling"), "projection.cluster_scaling"
    )
    limits = _require_mapping(report.get("limits"), "limits")
    sample_candidates = cluster_scaling.get("sample_candidate_documents")
    projected_candidates = projection.get("candidate_documents")
    sample_peak_rss = cluster_scaling.get("sample_peak_rss_bytes")
    if (
        isinstance(sample_candidates, bool)
        or not isinstance(sample_candidates, (int, float))
        or float(sample_candidates) <= 0
        or isinstance(projected_candidates, bool)
        or not isinstance(projected_candidates, (int, float))
        or float(projected_candidates) < 0
        or isinstance(sample_peak_rss, bool)
        or not isinstance(sample_peak_rss, int)
        or sample_peak_rss <= 0
    ):
        raise TurkishCorpusError("resource projection cluster scaling is invalid")
    candidate_scale = float(projected_candidates) / float(sample_candidates)
    require_close(
        cluster_scaling.get("projected_candidate_scale"),
        candidate_scale,
        "cluster projected candidate scale",
    )
    sample_edge_documents = cluster_scaling.get(
        "sample_edge_participating_documents"
    )
    if (
        isinstance(sample_edge_documents, bool)
        or not isinstance(sample_edge_documents, int)
        or sample_edge_documents < 0
    ):
        raise TurkishCorpusError("resource projection sample edge count is invalid")
    require_close(
        cluster_scaling.get("projected_edge_participating_documents"),
        sample_edge_documents * candidate_scale,
        "cluster projected edge-participating documents",
    )
    projected_peak_rss = float(sample_peak_rss) * max(1.0, candidate_scale)
    require_close(
        cluster_scaling.get("projected_peak_rss_bytes"),
        projected_peak_rss,
        "cluster projected peak RSS",
    )
    projected_peak_rss_with_safety = projected_peak_rss * float(
        projection["safety_factor"]
    )
    require_close(
        cluster_scaling.get("projected_peak_rss_bytes_with_safety_factor"),
        projected_peak_rss_with_safety,
        "cluster projected peak RSS with safety factor",
    )
    projected_cluster_wall_with_safety = float(
        projection["stage_wall_seconds_before_safety_factor"][
            "priority_cluster_quality_format"
        ]
    ) * float(projection["safety_factor"])
    require_close(
        cluster_scaling.get("projected_wall_seconds_with_safety_factor"),
        projected_cluster_wall_with_safety,
        "cluster projected wall with safety factor",
    )
    if (
        limits.get("cluster_memory_limit_bytes") != 192 * 1024**3
        or limits.get("cluster_wall_limit_seconds") != 172_800
    ):
        raise TurkishCorpusError("resource projection cluster limits drift")
    expected_automated_gate = (
        float(projection.get("peak_disk_bytes_with_safety_factor", math.inf))
        <= float(limits.get("effective_peak_limit_bytes", -1))
        and sample_peak_rss < limits["cluster_memory_limit_bytes"]
        and projected_peak_rss_with_safety < limits["cluster_memory_limit_bytes"]
        and projected_cluster_wall_with_safety <= limits["cluster_wall_limit_seconds"]
    )
    if report.get("manual_approval_required") is not True:
        raise TurkishCorpusError("resource projection must require manual approval")
    if report.get("automated_gate_passed") is not expected_automated_gate:
        raise TurkishCorpusError("resource projection automated gate result drift")
    return report_hash


def build_resource_projection(
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    sample_run_dir: str | Path,
    output_path: str | Path,
    *,
    quota_headroom_bytes: int,
    billable_cpus_per_job: int,
    safety_factor: float = 1.5,
) -> dict[str, Any]:
    """Extrapolate full CPU/storage needs from the deterministic source sample."""

    validate_corpus_policy(policy)
    validate_source_plan(plan, policy)
    validate_backend_calibration(calibration, policy)
    if quota_headroom_bytes <= 0 or not 1.0 <= safety_factor <= 3.0:
        raise TurkishCorpusError("invalid resource-projection headroom/safety factor")
    if billable_cpus_per_job != RESOURCE_BILLING_CONTRACT["billable_cpus_per_job"]:
        raise TurkishCorpusError(
            "cpu2dq resource projection requires exactly 128 billable CPUs per job"
        )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite resource projection: {destination}")
    root = Path(sample_run_dir)
    objects = _load_object_receipts(root, plan, calibration, sample_mode=True)
    buckets = _load_bucket_receipts(
        root, plan, calibration, sample_mode=True, objects=objects
    )
    cluster = load_json_strict(root / "cluster_receipt.json")
    verify_manifest_hash(cluster)
    if cluster.get("sample_mode") is not True:
        raise TurkishCorpusError("resource projection requires the sampled cluster run")
    sample_sources = {item["source_id"] for item in objects}
    configured_sources = {source["id"] for source in policy["sources"]}
    if sample_sources != configured_sources:
        raise TurkishCorpusError("resource sample does not cover every configured source")
    full_bytes_by_source: Counter[str] = Counter()
    for item in plan["objects"]:
        full_bytes_by_source[item["source_id"]] += item["size_bytes"]
    sample_bytes_by_source: Counter[str] = Counter()
    projected_candidates = 0.0
    projected_candidate_bytes = 0.0
    projected_download_wall = 0.0
    projected_score_wall = 0.0
    projected_signature_wall = 0.0
    projected_download_process_cpu = 0.0
    projected_score_process_cpu = 0.0
    projected_signature_process_cpu = 0.0
    source_projections: dict[str, Any] = {}
    for receipt in objects:
        source_id = receipt["source_id"]
        sample_bytes = float(receipt["raw_object"]["size_bytes"])
        sample_bytes_by_source[source_id] += int(sample_bytes)
    for source_id in sorted(configured_sources):
        source_receipts = [item for item in objects if item["source_id"] == source_id]
        sample_bytes = sum(item["raw_object"]["size_bytes"] for item in source_receipts)
        full_bytes = full_bytes_by_source[source_id]
        scale = full_bytes / sample_bytes
        candidate_rows = sum(item["candidate_file"]["rows"] for item in source_receipts)
        candidate_bytes = sum(item["candidate_file"]["size_bytes"] for item in source_receipts)
        download_wall = sum(
            item["telemetry"]["download"]["wall_seconds"] for item in source_receipts
        )
        score_wall = sum(
            item["telemetry"]["score_lid"]["wall_seconds"] for item in source_receipts
        )
        signature_wall = sum(
            item["telemetry"]["minhash_signature"]["wall_seconds"]
            for item in source_receipts
        )
        download_process_cpu = sum(
            item["telemetry"]["download"]["cpu_seconds"] for item in source_receipts
        )
        score_process_cpu = sum(
            item["telemetry"]["score_lid"]["cpu_seconds"] for item in source_receipts
        )
        signature_process_cpu = sum(
            item["telemetry"]["minhash_signature"]["cpu_seconds"]
            for item in source_receipts
        )
        projected_candidates += candidate_rows * scale
        projected_candidate_bytes += candidate_bytes * scale
        projected_download_wall += download_wall * scale
        projected_score_wall += score_wall * scale
        projected_signature_wall += signature_wall * scale
        projected_download_process_cpu += download_process_cpu * scale
        projected_score_process_cpu += score_process_cpu * scale
        projected_signature_process_cpu += signature_process_cpu * scale
        source_projections[source_id] = {
            "sample_objects": len(source_receipts),
            "sample_input_bytes": sample_bytes,
            "full_input_bytes": full_bytes,
            "scale": scale,
            "sample_candidate_documents": candidate_rows,
            "projected_candidate_documents": candidate_rows * scale,
            "projected_candidate_bytes": candidate_bytes * scale,
            "projected_download_wall_seconds": download_wall * scale,
            "projected_score_lid_wall_seconds": score_wall * scale,
            "projected_minhash_signature_wall_seconds": signature_wall * scale,
            "diagnostic_projected_process_cpu_seconds": {
                "download": download_process_cpu * scale,
                "score_lid": score_process_cpu * scale,
                "minhash_signature": signature_process_cpu * scale,
            },
        }
    sample_signature_bytes = sum(item["input_signature_bytes"] for item in buckets)
    sample_bucket_wall = sum(item["telemetry"]["wall_seconds"] for item in buckets)
    sample_bucket_process_cpu = sum(
        item["telemetry"]["cpu_seconds"] for item in buckets
    )
    projected_signature_bytes = projected_candidates * 14 * (8 * 8 + 4)
    projected_bucket_wall = (
        sample_bucket_wall * projected_signature_bytes / sample_signature_bytes
        if sample_signature_bytes
        else 0.0
    )
    projected_bucket_process_cpu = (
        sample_bucket_process_cpu * projected_signature_bytes / sample_signature_bytes
        if sample_signature_bytes
        else 0.0
    )
    sample_candidates = sum(item["candidate_file"]["rows"] for item in objects)
    sample_edges = sum(item["output"]["duplicate_edges"] for item in buckets)
    projected_edges = (
        sample_edges * projected_candidates / sample_candidates if sample_candidates else 0.0
    )
    projected_dups_bytes = projected_edges * 16
    cluster_wall = float(cluster["telemetry"]["wall_seconds"])
    cluster_process_cpu = float(cluster["telemetry"]["cpu_seconds"])
    projected_cluster_wall = (
        cluster_wall * projected_candidates / sample_candidates if sample_candidates else 0.0
    )
    projected_cluster_process_cpu = (
        cluster_process_cpu * projected_candidates / sample_candidates
        if sample_candidates
        else 0.0
    )
    cluster_sample_peak_rss = int(cluster["telemetry"].get("peak_rss_bytes", 0))
    if cluster_sample_peak_rss <= 0:
        raise TurkishCorpusError("sample cluster telemetry is missing peak_rss_bytes")
    cluster_sample_edge_documents = int(
        cluster["telemetry"].get("edge_participating_documents", 0)
    )
    candidate_scale = (
        projected_candidates / sample_candidates if sample_candidates else 0.0
    )
    projected_cluster_edge_documents = cluster_sample_edge_documents * candidate_scale
    projected_cluster_peak_rss = cluster_sample_peak_rss * max(1.0, candidate_scale)
    sample_backend_bytes = sum(item["size_bytes"] for item in cluster["output_files"])
    projected_backend_bytes = (
        sample_backend_bytes * projected_candidates / sample_candidates
        if sample_candidates
        else 0.0
    )
    raw_largest = max(item["size_bytes"] for item in plan["objects"])
    peak_disk_components = {
        "raw_largest_object_bytes": raw_largest,
        "candidate_bytes": projected_candidate_bytes,
        "signature_bytes": projected_signature_bytes,
        "duplicate_edge_bytes": projected_dups_bytes,
        "backend_output_bytes": projected_backend_bytes,
    }
    projected_peak_before_safety = sum(peak_disk_components.values())
    projected_peak = projected_peak_before_safety * safety_factor
    stage_wall = {
        "download": projected_download_wall,
        "score_lid": projected_score_wall,
        "minhash_signature": projected_signature_wall,
        "minhash_buckets": projected_bucket_wall,
        "priority_cluster_quality_format": projected_cluster_wall,
    }
    diagnostic_stage_process_cpu = {
        "download": projected_download_process_cpu,
        "score_lid": projected_score_process_cpu,
        "minhash_signature": projected_signature_process_cpu,
        "minhash_buckets": projected_bucket_process_cpu,
        "priority_cluster_quality_format": projected_cluster_process_cpu,
    }
    accounting = _resource_projection_accounting(
        stage_wall,
        diagnostic_stage_process_cpu,
        safety_factor=safety_factor,
        billable_cpus_per_job=billable_cpus_per_job,
    )
    limit = min(int(policy["materialization"]["max_peak_disk_bytes"]), quota_headroom_bytes)
    cluster_memory_limit = 192 * 1024**3
    projected_cluster_peak_rss_with_safety = projected_cluster_peak_rss * safety_factor
    projected_cluster_wall_with_safety = projected_cluster_wall * safety_factor
    automated_pass = (
        projected_peak <= limit
        and cluster_sample_peak_rss < cluster_memory_limit
        and projected_cluster_peak_rss_with_safety < cluster_memory_limit
        and projected_cluster_wall_with_safety <= 172_800
    )
    report = seal_manifest(
        {
            "schema_version": "2.0",
            "kind": RESOURCE_REPORT_KIND,
            "policy_sha256": _policy_sha256(policy),
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "sample_object_receipt_sha256": [item["canonical_sha256"] for item in objects],
            "sample_bucket_receipt_sha256": [item["canonical_sha256"] for item in buckets],
            "sample_cluster_receipt_sha256": cluster["canonical_sha256"],
            "billing_contract": dict(RESOURCE_BILLING_CONTRACT),
            "sample_selection": {
                "algorithm": (
                    "non_hplt_uri_quartiles_plus_hplt_smallest_complete_shard_"
                    "per_wds_bin_plus_macocu_genres_v5"
                ),
                "ranks": select_resource_sample_ranks(plan),
                "covers_every_source": True,
                "object_order": "source_plan_uri_ascending",
                "size_based_selection": True,
                "size_based_selection_scope": "hplt3_tr_only",
                "non_hplt_per_source_stream_spread_quantiles": [0.25, 0.5, 0.75],
                "covers_hplt_wds_bins": [8, 9, 10],
                "hplt_per_wds_bin_selection": (
                    "minimum_size_complete_shard_then_uri_v1"
                ),
                "hplt_selected_objects": [
                    {
                        "rank": item["rank"],
                        "wds_bin": item["wds_bin"],
                        "size_bytes": item["size_bytes"],
                        "uri": item["uri"],
                    }
                    for item in plan["objects"]
                    if item["source_id"] == "hplt3_tr"
                    and item["rank"] in select_resource_sample_ranks(plan)
                ],
                "covers_macocu_selected_genres": (
                    sorted(MACOCU_CONVERSATION_GENRES | MACOCU_GENERAL_GENRES)
                    if any(item["source_id"] == MACOCU_SOURCE_ID for item in plan["objects"])
                    else []
                ),
                "macocu_stream_spread_quantiles": (
                    [0.25, 0.5, 0.75]
                    if any(item["source_id"] == MACOCU_SOURCE_ID for item in plan["objects"])
                    else []
                ),
            },
            "sample_stage_telemetry": {
                "download": _sum_telemetry(objects, "download"),
                "score_lid": _sum_telemetry(objects, "score_lid"),
                "minhash_signature": _sum_telemetry(objects, "minhash_signature"),
                "minhash_buckets": {
                    key: sum(float(item["telemetry"].get(key, 0)) for item in buckets)
                    for key in ("wall_seconds", "cpu_seconds", "input_bytes", "output_bytes", "duplicate_edges")
                },
                "priority_cluster_quality_format": cluster["telemetry"],
            },
            "source_projections": source_projections,
            "projection": {
                "safety_factor": safety_factor,
                "candidate_documents": projected_candidates,
                "candidate_bytes": projected_candidate_bytes,
                "raw_largest_object_bytes": raw_largest,
                "signature_bytes": projected_signature_bytes,
                "duplicate_edges": projected_edges,
                "duplicate_edge_bytes": projected_dups_bytes,
                "backend_output_bytes": projected_backend_bytes,
                **accounting,
                "peak_disk_components_before_safety_factor": (
                    peak_disk_components
                ),
                "peak_disk_bytes_before_safety_factor": (
                    projected_peak_before_safety
                ),
                "peak_disk_bytes_with_safety_factor": projected_peak,
                "peak_disk_model": RESOURCE_PEAK_DISK_MODEL,
                "cluster_scaling": {
                    "sample_candidate_documents": sample_candidates,
                    "projected_candidate_scale": candidate_scale,
                    "sample_edge_participating_documents": cluster_sample_edge_documents,
                    "projected_edge_participating_documents": projected_cluster_edge_documents,
                    "sample_peak_rss_bytes": cluster_sample_peak_rss,
                    "projected_peak_rss_bytes": projected_cluster_peak_rss,
                    "projected_peak_rss_bytes_with_safety_factor": (
                        projected_cluster_peak_rss_with_safety
                    ),
                    "projected_wall_seconds_with_safety_factor": (
                        projected_cluster_wall_with_safety
                    ),
                    "rss_projection_model": (
                        "sample_peak_rss_times_max_one_and_candidate_scale"
                    ),
                },
            },
            "limits": {
                "policy_max_peak_disk_bytes": policy["materialization"]["max_peak_disk_bytes"],
                "reported_quota_headroom_bytes": quota_headroom_bytes,
                "effective_peak_limit_bytes": limit,
                "cluster_memory_limit_bytes": cluster_memory_limit,
                "cluster_wall_limit_seconds": 172_800,
            },
            "automated_gate_passed": automated_pass,
            "manual_approval_required": True,
            "canonical_sha256": None,
        }
    )
    validate_resource_projection(report)
    write_json_atomic(destination, report)
    return report


def seal_resource_approval(
    report_path: str | Path,
    mixture_quality_approval_path: str | Path,
    output_path: str | Path,
    *,
    policy_path: str | Path,
    source_plan_path: str | Path,
    calibration_path: str | Path,
    reviewer: str,
    reviewed_at_utc: str,
    decision: str,
    notes: str = "",
) -> dict[str, Any]:
    report_source = Path(report_path).expanduser().resolve()
    report_raw = _read_bounded_regular_file_snapshot(
        report_source,
        label="resource report",
        max_bytes=_MAX_AUDIT_REPORT_BYTES,
    )
    report = _load_json_snapshot(report_raw, "resource report")
    report_hash = validate_resource_projection(report)
    policy = load_corpus_policy(policy_path)
    plan = load_json_strict(source_plan_path)
    calibration = load_json_strict(calibration_path)
    validate_source_plan(plan, policy)
    validate_backend_calibration(calibration, policy)
    if (
        report.get("policy_sha256") != _policy_sha256(policy)
        or report.get("source_plan_sha256") != plan["canonical_sha256"]
        or report.get("calibration_sha256") != calibration["canonical_sha256"]
    ):
        raise TurkishCorpusError("resource report provenance binding drift")
    mixture_source = Path(mixture_quality_approval_path).expanduser().resolve()
    mixture_raw = _read_bounded_regular_file_snapshot(
        mixture_source,
        label="mixture-quality approval",
        max_bytes=_MAX_RECEIPT_EVIDENCE_BYTES,
    )
    mixture_quality_approval = _load_json_snapshot(
        mixture_raw, "mixture-quality approval"
    )
    mixture_quality_approval_hash = validate_mixture_quality_approval(
        mixture_quality_approval,
        policy=policy,
        plan=plan,
        calibration=calibration,
        approval_path=mixture_quality_approval_path,
    )
    if (
        mixture_quality_approval.get("sample_cluster_receipt_sha256")
        != report.get("sample_cluster_receipt_sha256")
    ):
        raise TurkishCorpusError(
            "resource report/quality approval sample-cluster lineage drift"
        )
    if report.get("automated_gate_passed") is not True and decision == "accepted":
        raise TurkishCorpusError("cannot accept a resource projection that exceeds the hard peak limit")
    if not reviewer.strip() or not _RFC3339_UTC_RE.fullmatch(reviewed_at_utc):
        raise TurkishCorpusError("resource approval requires reviewer and RFC3339 UTC timestamp")
    if decision not in {"accepted", "rejected"}:
        raise TurkishCorpusError("resource approval decision must be accepted/rejected")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite resource approval: {destination}")
    evidence_root = destination.expanduser().resolve().parent

    def evidence_record(
        path_value: str | Path, label: str, snapshot: bytes
    ) -> dict[str, Any]:
        path = Path(path_value).expanduser().resolve()
        if evidence_root != path.parent and evidence_root not in path.parents:
            raise TurkishCorpusError(
                f"{label} must remain inside the resource-approval directory tree"
            )
        return {
            "path": path.relative_to(evidence_root).as_posix(),
            "size_bytes": len(snapshot),
            "sha256": hashlib.sha256(snapshot).hexdigest(),
        }

    approval = seal_manifest(
        {
            "schema_version": "4.0",
            "kind": RESOURCE_APPROVAL_KIND,
            "resource_report_sha256": report_hash,
            "policy_sha256": report["policy_sha256"],
            "source_plan_sha256": report["source_plan_sha256"],
            "calibration_sha256": report["calibration_sha256"],
            "mixture_quality_approval_sha256": mixture_quality_approval_hash,
            "sample_cluster_receipt_sha256": report[
                "sample_cluster_receipt_sha256"
            ],
            "evidence_bundle": {
                "schema_version": "1.0",
                "resource_report": evidence_record(
                    report_path, "resource report evidence", report_raw
                ),
                "mixture_quality_approval": evidence_record(
                    mixture_quality_approval_path,
                    "mixture-quality approval evidence",
                    mixture_raw,
                ),
            },
            "approved_projection": {
                "billing_contract": report["billing_contract"],
                "billed_cpu_saat_with_safety_factor": report["projection"][
                    "billed_cpu_saat_with_safety_factor"
                ],
            },
            "reviewer": reviewer.strip(),
            "reviewed_at_utc": reviewed_at_utc,
            "decision": decision,
            "notes": notes,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, approval)
    return approval


def validate_resource_approval(
    approval: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    calibration: Mapping[str, Any],
    approval_path: str | Path,
) -> None:
    verify_manifest_hash(approval)
    if (
        approval.get("schema_version") != "4.0"
        or approval.get("kind") != RESOURCE_APPROVAL_KIND
        or approval.get("decision") != "accepted"
        or not isinstance(approval.get("reviewer"), str)
        or not approval["reviewer"].strip()
        or not _RFC3339_UTC_RE.fullmatch(str(approval.get("reviewed_at_utc", "")))
    ):
        raise TurkishCorpusError("full backend requires an accepted resource approval")
    if (
        approval.get("source_plan_sha256") != plan["canonical_sha256"]
        or approval.get("policy_sha256") != _policy_sha256(policy)
        or approval.get("calibration_sha256")
        != calibration["canonical_sha256"]
    ):
        raise TurkishCorpusError("resource approval binding drift")
    sample_cluster_sha = str(approval.get("sample_cluster_receipt_sha256") or "")
    if not _SHA256_RE.fullmatch(sample_cluster_sha):
        raise TurkishCorpusError("resource approval sample-cluster binding drift")
    approval_source = Path(approval_path).expanduser()
    if approval_source.is_symlink() or not approval_source.is_file():
        raise TurkishCorpusError("resource approval path is unsafe or missing")
    evidence_root = approval_source.resolve().parent
    bundle = _require_mapping(
        approval.get("evidence_bundle"), "resource approval evidence_bundle"
    )
    if bundle.get("schema_version") != "1.0":
        raise TurkishCorpusError("resource approval evidence bundle version drift")
    _report_path, report_raw = _quality_evidence_snapshot(
        evidence_root,
        bundle.get("resource_report"),
        "resource report",
        max_bytes=_MAX_AUDIT_REPORT_BYTES,
    )
    report = _load_json_snapshot(report_raw, "resource report")
    report_sha = validate_resource_projection(report)
    if (
        report_sha != approval.get("resource_report_sha256")
        or report.get("automated_gate_passed") is not True
        or report.get("policy_sha256") != _policy_sha256(policy)
        or report.get("source_plan_sha256") != plan["canonical_sha256"]
        or report.get("calibration_sha256") != calibration["canonical_sha256"]
        or report.get("sample_cluster_receipt_sha256") != sample_cluster_sha
    ):
        raise TurkishCorpusError("resource approval report evidence drift")
    mixture_path, mixture_raw = _quality_evidence_snapshot(
        evidence_root,
        bundle.get("mixture_quality_approval"),
        "mixture-quality approval",
        max_bytes=_MAX_RECEIPT_EVIDENCE_BYTES,
    )
    mixture = _load_json_snapshot(mixture_raw, "mixture-quality approval")
    mixture_sha = validate_mixture_quality_approval(
        mixture,
        policy=policy,
        plan=plan,
        calibration=calibration,
        approval_path=mixture_path,
    )
    if (
        mixture_sha != approval.get("mixture_quality_approval_sha256")
        or mixture.get("sample_cluster_receipt_sha256") != sample_cluster_sha
    ):
        raise TurkishCorpusError("resource approval mixture evidence drift")
    projection = _require_mapping(
        approval.get("approved_projection"), "resource approval approved_projection"
    )
    contract = _require_mapping(
        projection.get("billing_contract"),
        "resource approval approved_projection.billing_contract",
    )
    if dict(contract) != RESOURCE_BILLING_CONTRACT:
        raise TurkishCorpusError("resource approval billing contract drift")
    billed_cpu_saat = projection.get("billed_cpu_saat_with_safety_factor")
    if (
        isinstance(billed_cpu_saat, bool)
        or not isinstance(billed_cpu_saat, (int, float))
        or not math.isfinite(float(billed_cpu_saat))
        or float(billed_cpu_saat) <= 0
    ):
        raise TurkishCorpusError("resource approval lacks billed CPU-saat")
    if projection != {
        "billing_contract": report["billing_contract"],
        "billed_cpu_saat_with_safety_factor": report["projection"][
            "billed_cpu_saat_with_safety_factor"
        ],
    }:
        raise TurkishCorpusError("resource approval projection/report drift")


def _read_bounded_regular_file_snapshot(
    path: Path, *, label: str, max_bytes: int
) -> bytes:
    """Read one regular file once and return the immutable bytes inspected.

    ``O_NOFOLLOW`` plus a single descriptor prevents a path replacement from
    changing the bytes between hashing and parsing.  The pre/post ``fstat``
    check also rejects in-place mutation during the bounded read.
    """

    if max_bytes <= 0:
        raise TurkishCorpusError(f"{label} evidence cap is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TurkishCorpusError(f"{label} evidence file is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TurkishCorpusError(f"{label} evidence is not a regular file")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise TurkishCorpusError(
                f"{label} evidence exceeds the {max_bytes}-byte safety cap"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise TurkishCorpusError(f"{label} evidence was truncated while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise TurkishCorpusError(f"{label} evidence grew while read")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise TurkishCorpusError(f"{label} evidence changed while read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_json_snapshot(raw: bytes, label: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TurkishCorpusError(f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise TurkishCorpusError(f"{label} has non-finite JSON number {token!r}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except TurkishCorpusError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TurkishCorpusError(f"{label} is not strict UTF-8 JSON") from exc


def sample_quality_example_payload(
    stratum: tuple[str, str, str, str, str],
    decision: str,
    row: Mapping[str, Any],
    *,
    rejection_reason: str | None,
    quality_flags: Sequence[str],
    metrics: Mapping[str, Any],
    max_characters: int,
) -> dict[str, Any]:
    """Build the content-bound representation used for manual QA sampling."""

    text = str(row["text"])
    cluster_row = {
        "source_rank": int(stratum[0]),
        **{column: row.get(column) for column in BACKEND_COLUMNS},
    }
    cluster_row_sha256 = hashlib.sha256(
        canonical_json(cluster_row).encode("utf-8")
    ).hexdigest()
    selection_identity = hashlib.sha256(
        "\0".join(
            (
                *stratum,
                decision,
                str(row["document_id"]),
                cluster_row_sha256,
            )
        ).encode("utf-8")
    ).hexdigest()
    return {
        "sample_sha256": selection_identity,
        "cluster_row_sha256": cluster_row_sha256,
        "full_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "full_text_characters": len(text),
        "full_text_utf8_bytes": len(text.encode("utf-8")),
        "source_rank": int(stratum[0]),
        "source_id": stratum[1],
        "mixture_id": stratum[2],
        "wds_bin": stratum[3],
        "register": stratum[4],
        "decision": decision,
        "rejection_reason": rejection_reason,
        "quality_filter_flags": list(quality_flags),
        "document_id": str(row["document_id"]),
        "dedup_cluster_id": str(row["dedup_cluster_id"]),
        "url": str(row.get("url") or ""),
        "metrics": dict(sorted(metrics.items())),
        "text": text[:max_characters],
        "text_truncated": len(text) > max_characters,
    }


def sample_quality_example_heap_item(
    payload: Mapping[str, Any],
) -> tuple[int, str, str]:
    """Return the stable heap item for smallest content-bound SHA selection."""

    payload_json = canonical_json(payload).removesuffix("\n")
    identity = str(payload["sample_sha256"])
    return (-int(identity[:16], 16), identity, payload_json)


def _absolute_file_uri_path(raw: Any, label: str) -> Path:
    parsed = urllib.parse.urlparse(str(raw or ""))
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        raise TurkishCorpusError(f"{label} must be a local file URI")
    path = Path(urllib.parse.unquote(parsed.path))
    if not path.is_absolute() or path != path.resolve() or path.is_symlink():
        raise TurkishCorpusError(f"{label} path is unsafe")
    return path


def _live_sample_evidence(
    input_artifacts: Mapping[str, Any],
    *,
    copied_cluster: Mapping[str, Any],
    copied_cluster_sha: str,
    cluster_outputs: Sequence[Mapping[str, Any]],
) -> Path:
    """Reopen the exact live sample lineage that owns the audited Parquets."""

    root_record = _require_mapping(
        input_artifacts.get("live_sample_run"), "sample audit live_sample_run"
    )
    root = _absolute_file_uri_path(root_record.get("uri"), "live sample run")
    if not root.is_dir():
        raise TurkishCorpusError("live sample run directory is missing")
    root_stat = root.stat()
    expected_total = sum(int(item.get("size_bytes", -1)) for item in cluster_outputs)
    if (
        isinstance(root_record.get("filesystem_device"), bool)
        or root_record.get("filesystem_device") != root_stat.st_dev
        or root_record.get("cluster_output_bytes") != expected_total
        or root_record.get("maximum_validation_bytes")
        != _MAX_CLUSTER_PARQUET_TOTAL_BYTES
        or expected_total < 0
        or expected_total > _MAX_CLUSTER_PARQUET_TOTAL_BYTES
    ):
        raise TurkishCorpusError("live sample run bounded-I/O contract drift")

    live_record = _require_mapping(
        input_artifacts.get("live_cluster_receipt"),
        "sample audit live_cluster_receipt",
    )
    live_path = _absolute_file_uri_path(
        live_record.get("uri"), "live cluster receipt"
    )
    if live_path != root / "cluster_receipt.json":
        raise TurkishCorpusError("live cluster receipt does not belong to sample run")
    raw = _read_bounded_regular_file_snapshot(
        live_path,
        label="live cluster receipt",
        max_bytes=_MAX_RECEIPT_EVIDENCE_BYTES,
    )
    if (
        live_record.get("size_bytes") != len(raw)
        or live_record.get("sha256") != hashlib.sha256(raw).hexdigest()
        or live_record.get("canonical_sha256") != copied_cluster_sha
    ):
        raise TurkishCorpusError("live cluster receipt content drift")
    live_cluster = _load_json_snapshot(raw, "live cluster receipt")
    if live_cluster != copied_cluster or verify_manifest_hash(live_cluster) != copied_cluster_sha:
        raise TurkishCorpusError("copied/live cluster receipt mismatch")
    return root


def _strict_quality_flags(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        raise TurkishCorpusError("cluster quality_filter_flags must be JSON text")
    value = _load_json_snapshot(raw.encode("utf-8"), "cluster quality_filter_flags")
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TurkishCorpusError("cluster quality_filter_flags schema drift")
    return value


class _QualityNumericSketch:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.seen = 0
        self.heap: list[tuple[int, float]] = []

    def add(self, identity: str, raw_value: Any) -> None:
        if isinstance(raw_value, bool):
            value = float(int(raw_value))
        else:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return
        if not math.isfinite(value):
            return
        self.seen += 1
        rank = int.from_bytes(
            hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big"
        )
        item = (-rank, value)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def summary(self, quantiles: Sequence[float]) -> dict[str, Any]:
        values = sorted(value for _rank, value in self.heap)
        result: dict[str, Any] = {
            "observations": self.seen,
            "deterministic_sample_size": len(values),
        }
        if values:
            result.update(
                {
                    "minimum": values[0],
                    "maximum": values[-1],
                    "mean_of_sample": sum(values) / len(values),
                }
            )
            for quantile in quantiles:
                index = round(float(quantile) * (len(values) - 1))
                result[f"q{float(quantile):.4f}"] = values[index]
        return result


def _recompute_cluster_quality_evidence(
    run_root: Path,
    output_records: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
    examples_per_stratum: int,
    max_example_characters: int,
    quantile_sample_size: int,
    quantiles: Sequence[float],
) -> dict[str, Any]:
    """Recompute exact counts and deterministic examples from live Parquets.

    Each Parquet is opened once by descriptor.  Hashing, parsing, and the
    post-parse hash all use that descriptor, so replacing its path cannot
    switch the audited bytes after verification.
    """

    heaps: dict[tuple[str, str, str, str, str, str], list[tuple[int, str, str]]] = (
        defaultdict(list)
    )
    counts: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    flags: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    reasons: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
    numeric: dict[
        tuple[str, str, str, str, str],
        dict[str, dict[str, _QualityNumericSketch]],
    ] = defaultdict(lambda: defaultdict(dict))
    total_bytes = 0
    total_rows = 0
    seen_paths: set[str] = set()
    seen_ranks: set[int] = set()
    required_columns = set(BACKEND_COLUMNS)

    for record in output_records:
        relative_raw = str(record.get("path") or "")
        relative = Path(relative_raw)
        rank = record.get("source_rank")
        size = record.get("size_bytes")
        digest = str(record.get("sha256") or "")
        rows_expected = record.get("rows")
        if (
            not relative_raw
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_raw in seen_paths
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank in seen_ranks
            or rank < 0
            or rank >= len(plan["objects"])
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > _MAX_CLUSTER_PARQUET_BYTES
            or not _SHA256_RE.fullmatch(digest)
            or isinstance(rows_expected, bool)
            or not isinstance(rows_expected, int)
            or rows_expected <= 0
        ):
            raise TurkishCorpusError("cluster output bounded evidence record drift")
        seen_paths.add(relative_raw)
        seen_ranks.add(rank)
        total_bytes += size
        if total_bytes > _MAX_CLUSTER_PARQUET_TOTAL_BYTES:
            raise TurkishCorpusError("cluster output evidence exceeds aggregate I/O cap")
        unresolved = run_root / relative
        path = unresolved.resolve()
        if run_root not in path.parents or unresolved.is_symlink():
            raise TurkishCorpusError("cluster output evidence path is unsafe")
        current = unresolved.parent
        while current != run_root:
            if current.is_symlink():
                raise TurkishCorpusError("cluster output evidence path is symlinked")
            current = current.parent
        flags_open = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path, flags_open)
        except OSError as exc:
            raise TurkishCorpusError("cluster output evidence is missing or unsafe") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                raise TurkishCorpusError("cluster output evidence size drift")
            _assert_descriptor_path_binding(
                path, descriptor, "cluster output evidence"
            )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                if _descriptor_sha256(handle) != digest:
                    raise TurkishCorpusError("cluster output evidence hash drift")
                _assert_descriptor_path_binding(
                    path, descriptor, "cluster output evidence"
                )
                handle.seek(0)
                parquet = pq.ParquetFile(handle)
                if not required_columns <= set(parquet.schema_arrow.names):
                    raise TurkishCorpusError("cluster output evidence schema drift")
                if parquet.metadata.num_rows != rows_expected:
                    raise TurkishCorpusError("cluster output evidence row-count drift")
                file_rows = 0
                plan_object = plan["objects"][rank]
                for batch in parquet.iter_batches(batch_size=2_048):
                    for row in batch.to_pylist():
                        file_rows += 1
                        total_rows += 1
                        if (
                            not isinstance(row.get("dedup_keep"), bool)
                            or not isinstance(row.get("text"), str)
                            or not isinstance(row.get("document_id"), str)
                            or not row["document_id"]
                            or row.get("source_id") != plan_object["source_id"]
                        ):
                            raise TurkishCorpusError("cluster output row schema drift")
                        quality_flags = _strict_quality_flags(
                            row.get("quality_filter_flags")
                        )
                        source_id = str(row["source_id"])
                        try:
                            routed = select_mixture_bucket(source_id, row, policy)
                            register = dominant_register(row)
                        except (KeyError, TypeError, ValueError) as exc:
                            raise TurkishCorpusError(
                                "cluster output routing schema drift"
                            ) from exc
                        mixture_id = routed[0] if routed is not None else "unrouted"
                        wds_bin = (
                            str(row["wds_bin"])
                            if row.get("wds_bin") is not None
                            else "not_applicable"
                        )
                        stratum = (
                            str(rank),
                            source_id,
                            mixture_id,
                            wds_bin,
                            register,
                        )
                        text = row["text"]
                        utf8_bytes = len(text.encode("utf-8"))
                        row_counts = counts[stratum]
                        row_counts["total_documents"] += 1
                        row_counts["total_utf8_bytes"] += utf8_bytes
                        if routed is None:
                            row_counts["selector_unrouted_documents"] += 1
                        else:
                            row_counts["selector_routed_documents"] += 1
                        if row["dedup_keep"]:
                            row_counts["dedup_survived_documents"] += 1
                        else:
                            row_counts["dedup_removed_documents"] += 1
                        if row["dedup_keep"] and quality_flags:
                            row_counts["quality_rejected_documents"] += 1
                        elif row["dedup_keep"]:
                            row_counts["quality_passed_documents"] += 1
                        for flag in quality_flags:
                            flags[stratum][flag] += 1

                        accepted = bool(
                            row["dedup_keep"]
                            and not quality_flags
                            and routed is not None
                        )
                        if accepted:
                            independent = audit_document(
                                text,
                                url=str(row.get("url") or ""),
                                source_lid_ok=True,
                                content_policy=policy["content_policy"],
                            )
                            if (
                                not independent.accepted
                                or independent.normalized_text != text
                            ):
                                raise TurkishCorpusError(
                                    "accepted live cluster row fails full content audit"
                                )
                            decision = "accepted"
                            rejection_reason = None
                            row_counts["accepted_documents"] += 1
                            row_counts["accepted_utf8_bytes"] += utf8_bytes
                        else:
                            decision = "rejected"
                            if row["dedup_keep"] is not True:
                                rejection_reason = "dedup_removed"
                            elif routed is None:
                                rejection_reason = "selector_unrouted"
                            else:
                                rejection_reason = "quality_filter"
                            row_counts["rejected_documents"] += 1
                            row_counts["rejected_utf8_bytes"] += utf8_bytes
                            reasons[stratum][rejection_reason] += 1

                        _normalized, qa_metrics = _qa_document_metrics(row)
                        qa_metrics = dict(qa_metrics)
                        qa_metrics.update(
                            {
                                "source_lid_probability": row.get(
                                    "source_lid_probability"
                                ),
                                "text_characters": len(text),
                                "text_utf8_bytes": utf8_bytes,
                                "text_words": len(_QUALITY_WORD_RE.findall(text)),
                            }
                        )
                        numeric_metrics = {
                            key: value
                            for key, value in qa_metrics.items()
                            if isinstance(value, (int, float))
                            and not isinstance(value, bool)
                        }
                        for population in ("all", decision):
                            for metric_name, value in numeric_metrics.items():
                                sketch = numeric[stratum][population].setdefault(
                                    metric_name,
                                    _QualityNumericSketch(quantile_sample_size),
                                )
                                sketch.add(
                                    f"{row['document_id']}\0{population}\0{metric_name}",
                                    value,
                                )
                        payload = sample_quality_example_payload(
                            stratum,
                            decision,
                            row,
                            rejection_reason=rejection_reason,
                            quality_flags=quality_flags,
                            metrics=qa_metrics,
                            max_characters=max_example_characters,
                        )
                        item = sample_quality_example_heap_item(payload)
                        heap = heaps[(*stratum, decision)]
                        if len(heap) < examples_per_stratum:
                            heapq.heappush(heap, item)
                        elif item > heap[0]:
                            heapq.heapreplace(heap, item)
                if file_rows != rows_expected:
                    raise TurkishCorpusError("cluster output evidence row scan drift")
                if _descriptor_sha256(handle) != digest:
                    raise TurkishCorpusError("cluster output changed during validation")
                _assert_descriptor_path_binding(
                    path, descriptor, "cluster output evidence"
                )
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise TurkishCorpusError("cluster output changed during validation")
        finally:
            os.close(descriptor)

    examples: dict[str, list[dict[str, Any]]] = {}
    for decision in ("accepted", "rejected"):
        selected = [
            json.loads(payload_json)
            for key, heap in heaps.items()
            if key[-1] == decision
            for _rank_value, _identity, payload_json in heap
        ]
        examples[decision] = sorted(
            selected,
            key=lambda row: (
                row["source_rank"],
                row["source_id"],
                row["mixture_id"],
                row["wds_bin"],
                row["register"],
                row["sample_sha256"],
            ),
        )
    return {
        "examples": examples,
        "counts": counts,
        "flags": flags,
        "reasons": reasons,
        "numeric_distributions": {
            stratum: {
                population: {
                    metric_name: sketch.summary(quantiles)
                    for metric_name, sketch in sorted(metric_sketches.items())
                }
                for population, metric_sketches in sorted(populations.items())
            }
            for stratum, populations in numeric.items()
        },
        "total_rows": total_rows,
    }


def _quality_evidence_snapshot(
    root: Path,
    record: Any,
    label: str,
    *,
    max_bytes: int,
) -> tuple[Path, bytes]:
    if not isinstance(record, Mapping):
        raise TurkishCorpusError(f"{label} evidence record is missing")
    raw = str(record.get("path") or "")
    relative = Path(raw)
    if (
        not raw
        or urllib.parse.urlparse(raw).scheme
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise TurkishCorpusError(f"{label} evidence path is unsafe")
    resolved_root = root.resolve()
    unresolved = resolved_root / relative
    path = unresolved.resolve()
    if resolved_root not in path.parents:
        raise TurkishCorpusError(f"{label} evidence file is missing")
    current = unresolved
    while current != resolved_root:
        if current.is_symlink():
            raise TurkishCorpusError(f"{label} evidence path is symlinked")
        current = current.parent
    size = record.get("size_bytes")
    digest = str(record.get("sha256", ""))
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or size > max_bytes
        or not _SHA256_RE.fullmatch(digest)
    ):
        raise TurkishCorpusError(f"{label} evidence content drift")
    raw_bytes = _read_bounded_regular_file_snapshot(
        path, label=label, max_bytes=max_bytes
    )
    if len(raw_bytes) != size or hashlib.sha256(raw_bytes).hexdigest() != digest:
        raise TurkishCorpusError(f"{label} evidence content drift")
    return path, raw_bytes


def _read_quality_example_rows(
    raw: bytes, *, decision: str, expected_rows: int
) -> list[dict[str, Any]]:
    if raw and not raw.endswith(b"\n"):
        raise TurkishCorpusError(f"{decision} example JSONL lacks terminal newline")
    rows: list[dict[str, Any]] = []
    required = {
        "sample_sha256",
        "cluster_row_sha256",
        "full_text_sha256",
        "full_text_characters",
        "full_text_utf8_bytes",
        "source_rank",
        "source_id",
        "mixture_id",
        "wds_bin",
        "register",
        "decision",
        "rejection_reason",
        "quality_filter_flags",
        "document_id",
        "dedup_cluster_id",
        "url",
        "metrics",
        "text",
        "text_truncated",
    }
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            row = _load_json_snapshot(line.encode("utf-8"), f"{decision} example")
        except TurkishCorpusError as exc:
            raise TurkishCorpusError(
                f"{decision} example JSONL line {line_number} is malformed"
            ) from exc
        if not isinstance(row, Mapping) or set(row) != required:
            raise TurkishCorpusError(f"{decision} example row structure drift")
        if (
            row.get("decision") != decision
            or isinstance(row.get("source_rank"), bool)
            or not isinstance(row.get("source_rank"), int)
            or row["source_rank"] < 0
            or not isinstance(row.get("source_id"), str)
            or not row["source_id"]
            or not isinstance(row.get("mixture_id"), str)
            or not row["mixture_id"]
            or not isinstance(row.get("wds_bin"), str)
            or not row["wds_bin"]
            or not isinstance(row.get("register"), str)
            or not row["register"]
            or not isinstance(row.get("document_id"), str)
            or not row["document_id"]
            or not isinstance(row.get("url"), str)
            or not isinstance(row.get("text"), str)
            or not row["text"]
            or not isinstance(row.get("text_truncated"), bool)
            or not isinstance(row.get("metrics"), Mapping)
            or not isinstance(row.get("quality_filter_flags"), list)
            or any(
                not isinstance(flag, str) or not flag
                for flag in row["quality_filter_flags"]
            )
            or not _SHA256_RE.fullmatch(str(row.get("sample_sha256", "")))
            or not _SHA256_RE.fullmatch(str(row.get("cluster_row_sha256", "")))
            or not _SHA256_RE.fullmatch(str(row.get("full_text_sha256", "")))
            or not _SHA256_RE.fullmatch(str(row.get("dedup_cluster_id", "")))
            or isinstance(row.get("full_text_characters"), bool)
            or not isinstance(row.get("full_text_characters"), int)
            or row["full_text_characters"] < len(row["text"])
            or isinstance(row.get("full_text_utf8_bytes"), bool)
            or not isinstance(row.get("full_text_utf8_bytes"), int)
            or row["full_text_utf8_bytes"] < len(row["text"].encode("utf-8"))
        ):
            raise TurkishCorpusError(f"{decision} example row content drift")
        if row["text_truncated"] is not (
            row["full_text_characters"] > len(row["text"])
        ):
            raise TurkishCorpusError(f"{decision} example truncation evidence drift")
        if not row["text_truncated"] and (
            row["full_text_sha256"]
            != hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            or row["full_text_characters"] != len(row["text"])
            or row["full_text_utf8_bytes"] != len(row["text"].encode("utf-8"))
        ):
            raise TurkishCorpusError(f"{decision} example full-text evidence drift")
        if decision == "accepted" and (
            row.get("rejection_reason") is not None
            or row["quality_filter_flags"] != []
        ):
            raise TurkishCorpusError("accepted example carries rejection evidence")
        if decision == "rejected":
            reason = row.get("rejection_reason")
            if reason not in {"dedup_removed", "selector_unrouted", "quality_filter"}:
                raise TurkishCorpusError("rejected example lacks a valid rejection reason")
            if reason != "dedup_removed" and not row["quality_filter_flags"]:
                raise TurkishCorpusError("rejected example lacks quality/selector evidence")
        rows.append(dict(row))
    if len(rows) != expected_rows:
        raise TurkishCorpusError(f"{decision} example row-count drift")
    return rows


def validate_sample_quality_audit_bundle(
    bundle_root: str | Path,
    report_record: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    require_complete_accepted_coverage: bool = True,
    report_snapshot: bytes | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate the actual bounded report, examples, and launch provenance."""

    validate_corpus_policy(policy)
    validate_source_plan(plan, policy)
    validate_backend_calibration(calibration, policy)
    root = Path(bundle_root)
    if root.is_symlink() or not root.is_dir():
        raise TurkishCorpusError("sample-quality evidence root is unsafe or missing")
    if report_snapshot is None:
        _report_path, report_raw = _quality_evidence_snapshot(
            root,
            report_record,
            "sample audit report",
            max_bytes=_MAX_AUDIT_REPORT_BYTES,
        )
    else:
        report_raw = bytes(report_snapshot)
        relative = Path(str(report_record.get("path") or ""))
        resolved_root = root.resolve()
        resolved_report = (resolved_root / relative).resolve()
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or resolved_root not in resolved_report.parents
            or len(report_raw) > _MAX_AUDIT_REPORT_BYTES
            or report_record.get("size_bytes") != len(report_raw)
            or report_record.get("sha256")
            != hashlib.sha256(report_raw).hexdigest()
        ):
            raise TurkishCorpusError("sample audit report snapshot/record drift")
    report = _load_json_snapshot(report_raw, "sample audit report")
    report_sha = verify_manifest_hash(report)
    policy_sha = _policy_sha256(policy)
    expected_processing = production_processing_binding(policy)
    if (
        report.get("schema_version") != "2.0"
        or report.get("kind") != "turkish_bounded_backend_sample_quality_audit"
        or report.get("integrity_checks_passed") is not True
        or report.get("manual_review_required") is not True
        or report.get("automatic_mixture_approval") is not False
        or report.get("review_status") != "pending"
        or report.get("policy_sha256") != policy_sha
        or report.get("source_plan_sha256") != plan["canonical_sha256"]
        or report.get("calibration_sha256") != calibration["canonical_sha256"]
        or report.get("processing") != expected_processing
    ):
        raise TurkishCorpusError("sample-quality audit report binding drift")

    qa_policy = _require_mapping(
        policy.get("quality_assurance"), "quality_assurance"
    )
    sample_contract = _require_mapping(
        report.get("sample_contract"), "sample audit sample_contract"
    )
    if (
        sample_contract.get("sample_mode") is not True
        or sample_contract.get("expected_object_ranks")
        != select_resource_sample_ranks(plan)
        or sample_contract.get("quantiles") != qa_policy["quantiles"]
        or sample_contract.get("quantile_sampling")
        != "smallest_sha256_per_document_population_metric"
        or isinstance(sample_contract.get("quantile_sample_size"), bool)
        or not isinstance(sample_contract.get("quantile_sample_size"), int)
        or sample_contract["quantile_sample_size"]
        < int(qa_policy["quantile_sample_size"])
    ):
        raise TurkishCorpusError("sample-quality audit sampling contract drift")

    input_artifacts = _require_mapping(
        report.get("input_artifacts"), "sample audit input_artifacts"
    )
    provenance: dict[str, tuple[dict[str, Any], str]] = {}
    for name in (
        "cluster_receipt",
        "object_launch_receipt",
        "bucket_launch_receipt",
    ):
        record = _require_mapping(input_artifacts.get(name), f"sample audit {name}")
        _path, receipt_raw = _quality_evidence_snapshot(
            root,
            record,
            f"sample audit {name}",
            max_bytes=_MAX_RECEIPT_EVIDENCE_BYTES,
        )
        receipt = _load_json_snapshot(receipt_raw, f"sample audit {name}")
        digest = verify_manifest_hash(receipt)
        if record.get("canonical_sha256") != digest:
            raise TurkishCorpusError(f"sample audit {name} canonical hash drift")
        provenance[name] = (receipt, digest)
    cluster, cluster_sha = provenance["cluster_receipt"]
    object_launch, object_launch_sha = provenance["object_launch_receipt"]
    bucket_launch, bucket_launch_sha = provenance["bucket_launch_receipt"]
    expected_ranks = select_resource_sample_ranks(plan)
    expected_objects = report.get("sampled_objects")
    cluster_code_identity = validate_production_code_identity(
        cluster.get("code_identity")
    )
    if (
        cluster.get("schema_version") != "1.0"
        or cluster.get("kind") != CLUSTER_RECEIPT_KIND
        or cluster.get("sample_mode") is not True
        or cluster.get("source_plan_sha256") != plan["canonical_sha256"]
        or cluster.get("calibration_sha256") != calibration["canonical_sha256"]
        or cluster.get("processing") != expected_processing
        or cluster.get("winner_policy") != CLUSTER_WINNER_POLICY
        or cluster.get("quality_score_semantics")
        != CLUSTER_QUALITY_SCORE_SEMANTICS
        or report.get("cluster_receipt_sha256") != cluster_sha
        or report.get("sample_cluster_receipt_sha256") != cluster_sha
        or report.get("code_identity") != cluster_code_identity
        or not isinstance(expected_objects, list)
        or [item.get("rank") for item in expected_objects] != expected_ranks
    ):
        raise TurkishCorpusError("sample-quality cluster/rank provenance drift")
    neutralized_ranks: list[int] = []
    for item in expected_objects:
        rank = int(item["rank"])
        planned = plan["objects"][rank]
        quality_semantics = item.get("quality_score_semantics")
        if (
            item.get("source_id") != planned["source_id"]
            or item.get("wds_bin") != planned.get("wds_bin")
            or (
                quality_semantics is not None
                and quality_semantics != OBJECT_SOURCE_QUALITY_SEMANTICS
            )
        ):
            raise TurkishCorpusError("sample-quality sampled-object identity drift")
        if quality_semantics != OBJECT_SOURCE_QUALITY_SEMANTICS:
            neutralized_ranks.append(rank)
    if cluster.get("legacy_quality_score_neutralized_ranks") != neutralized_ranks:
        raise TurkishCorpusError("sample-quality legacy quality neutralization drift")
    object_hashes = cluster.get("object_receipt_sha256")
    bucket_hashes = cluster.get("bucket_receipt_sha256")
    if [item.get("receipt_sha256") for item in expected_objects] != object_hashes:
        raise TurkishCorpusError("sample audit object receipt inventory drift")
    cluster_outputs = cluster.get("output_files")
    report_outputs = input_artifacts.get("cluster_output_files")
    if not isinstance(cluster_outputs, list) or not isinstance(report_outputs, list):
        raise TurkishCorpusError("sample audit cluster output inventory is missing")
    normalized_outputs: list[dict[str, Any]] = []
    for raw in cluster_outputs:
        item = _require_mapping(raw, "cluster output file")
        source_rank = item.get("source_rank")
        if source_rank is None:
            stem = Path(str(item.get("path") or "")).stem
            if not stem.isdigit():
                raise TurkishCorpusError("legacy cluster output rank is ambiguous")
            source_rank = int(stem)
        normalized_outputs.append(
            {
                "path": item.get("path"),
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
                "rows": item.get("rows"),
                "source_rank": source_rank,
            }
        )
    if report_outputs != normalized_outputs:
        raise TurkishCorpusError("sample audit cluster output/rank inventory drift")
    live_run_root = _live_sample_evidence(
        input_artifacts,
        copied_cluster=cluster,
        copied_cluster_sha=cluster_sha,
        cluster_outputs=normalized_outputs,
    )
    object_records = object_launch.get("object_receipts")
    bucket_records = bucket_launch.get("backend_bucket_receipts")
    if (
        object_launch.get("kind")
        != "turkish_packed_resource_sample_launch_receipt"
        or object_launch.get("all_lanes_completed") is not True
        or object_launch.get("source_plan_sha256") != plan["canonical_sha256"]
        or object_launch.get("calibration_sha256")
        != calibration["canonical_sha256"]
        or not isinstance(object_records, list)
        or [item.get("rank") for item in object_records] != expected_ranks
        or [item.get("canonical_sha256") for item in object_records] != object_hashes
        or bucket_launch.get("kind")
        != "turkish_packed_sample_bucket_launch_receipt"
        or bucket_launch.get("all_buckets_completed") is not True
        or bucket_launch.get("source_plan_sha256") != plan["canonical_sha256"]
        or bucket_launch.get("calibration_sha256")
        != calibration["canonical_sha256"]
        or bucket_launch.get("object_sample_launch_receipt_sha256")
        != object_launch_sha
        or not isinstance(bucket_records, list)
        or [item.get("bucket_rank") for item in bucket_records] != list(range(14))
        or [item.get("canonical_sha256") for item in bucket_records] != bucket_hashes
    ):
        raise TurkishCorpusError("sample-quality packed launch provenance drift")
    launch_bindings = _require_mapping(
        cluster.get("sample_launch_receipts"), "cluster sample launch receipts"
    )
    if (
        _require_mapping(launch_bindings.get("object"), "cluster object launch").get(
            "canonical_sha256"
        )
        != object_launch_sha
        or _require_mapping(
            launch_bindings.get("bucket"), "cluster bucket launch"
        ).get("canonical_sha256")
        != bucket_launch_sha
    ):
        raise TurkishCorpusError("cluster sample-launch hash binding drift")

    example_sampling = _require_mapping(
        report.get("example_sampling"), "sample audit example_sampling"
    )
    requested = example_sampling.get("examples_per_stratum_and_decision")
    max_example_characters = example_sampling.get("max_example_characters")
    if (
        example_sampling.get("method")
        != (
            "smallest_content_bound_sha256_per_rank_source_mixture_"
            "wds_register_decision_v2"
        )
        or isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested < int(qa_policy["examples_per_stratum_and_decision"])
        or isinstance(max_example_characters, bool)
        or not isinstance(max_example_characters, int)
        or max_example_characters < int(qa_policy["max_example_characters"])
    ):
        raise TurkishCorpusError("sample audit example request count drift")
    files = _require_mapping(example_sampling.get("files"), "sample audit example files")
    example_rows: dict[str, list[dict[str, Any]]] = {}
    for decision in ("accepted", "rejected"):
        record = _require_mapping(files.get(decision), f"{decision} examples")
        rows = record.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise TurkishCorpusError(f"{decision} example count drift")
        _jsonl_path, jsonl_raw = _quality_evidence_snapshot(
            root,
            record.get("jsonl"),
            f"{decision} JSONL",
            max_bytes=_MAX_EXAMPLE_EVIDENCE_BYTES,
        )
        _plaintext_path, plaintext_raw = _quality_evidence_snapshot(
            root,
            record.get("plaintext"),
            f"{decision} plaintext",
            max_bytes=_MAX_EXAMPLE_EVIDENCE_BYTES,
        )
        parsed = _read_quality_example_rows(
            jsonl_raw, decision=decision, expected_rows=rows
        )
        try:
            plaintext_text = plaintext_raw.decode("utf-8")
        except UnicodeError as exc:
            raise TurkishCorpusError(
                f"{decision} plaintext is not UTF-8"
            ) from exc
        if plaintext_text != render_sample_quality_plaintext(parsed):
            raise TurkishCorpusError(f"{decision} plaintext/example content drift")
        example_rows[decision] = parsed
    live_evidence = _recompute_cluster_quality_evidence(
        live_run_root,
        normalized_outputs,
        plan=plan,
        policy=policy,
        examples_per_stratum=requested,
        max_example_characters=max_example_characters,
        quantile_sample_size=sample_contract["quantile_sample_size"],
        quantiles=sample_contract["quantiles"],
    )
    if example_rows != live_evidence["examples"]:
        raise TurkishCorpusError(
            "sample audit examples differ from deterministic live cluster rows"
        )
    live_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for live_key, live_counts in live_evidence["counts"].items():
        live_source_counts[live_key[1]].update(live_counts)
    expected_source_counts = [
        {
            "source_id": source_id,
            **dict(sorted(live_source_counts[source_id].items())),
        }
        for source_id in sorted(live_source_counts)
    ]
    if report.get("cluster_source_counts") != expected_source_counts:
        raise TurkishCorpusError("sample audit live source-count drift")

    strata = report.get("strata")
    if not isinstance(strata, list) or not strata:
        raise TurkishCorpusError("sample audit strata are missing")
    strata_keys: set[tuple[Any, ...]] = set()
    ranks_with_rows: set[int] = set()
    ranks_with_accepted_rows: set[int] = set()
    hplt_bins_with_accepted_rows: set[int] = set()
    mixtures_with_accepted_rows: set[str] = set()
    mixtures_with_rejected_rows: set[str] = set()
    recomputed_insufficiencies: list[dict[str, Any]] = []
    aggregate_counts: Counter[str] = Counter()
    sampled_counts = Counter(
        (
            row["source_rank"],
            row["source_id"],
            row["mixture_id"],
            row["wds_bin"],
            row["register"],
            row["decision"],
        )
        for rows in example_rows.values()
        for row in rows
    )
    for raw in strata:
        item = _require_mapping(raw, "sample audit stratum")
        rank = item.get("source_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in expected_ranks:
            raise TurkishCorpusError("sample audit stratum rank drift")
        planned = plan["objects"][rank]
        if item.get("source_id") != planned["source_id"]:
            raise TurkishCorpusError("sample audit stratum source/rank drift")
        expected_wds = (
            str(planned["wds_bin"])
            if planned.get("wds_bin") is not None
            else "not_applicable"
        )
        if item.get("wds_bin") != expected_wds:
            raise TurkishCorpusError("sample audit stratum WDS/rank drift")
        key = (
            rank,
            item.get("source_id"),
            item.get("mixture_id"),
            item.get("wds_bin"),
            item.get("register"),
        )
        if key in strata_keys:
            raise TurkishCorpusError("sample audit contains a duplicate stratum")
        strata_keys.add(key)
        live_key = (
            str(rank),
            str(item.get("source_id")),
            str(item.get("mixture_id")),
            str(item.get("wds_bin")),
            str(item.get("register")),
        )
        live_counts = live_evidence["counts"].get(live_key)
        if live_counts is None:
            raise TurkishCorpusError("sample audit stratum is absent from live cluster")
        counts = _require_mapping(item.get("counts"), "sample audit stratum counts")
        if dict(counts) != dict(sorted(live_counts.items())):
            raise TurkishCorpusError("sample audit stratum/live count drift")
        count_fields = (
            "total_documents",
            "total_utf8_bytes",
            "accepted_documents",
            "accepted_utf8_bytes",
            "rejected_documents",
            "rejected_utf8_bytes",
            "selector_routed_documents",
            "selector_unrouted_documents",
            "dedup_survived_documents",
            "dedup_removed_documents",
            "quality_passed_documents",
            "quality_rejected_documents",
        )
        exact_counts: dict[str, int] = {}
        for name in count_fields:
            value = counts.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TurkishCorpusError("sample audit stratum count type drift")
            exact_counts[name] = value
        total = exact_counts["total_documents"]
        if total <= 0:
            raise TurkishCorpusError("sample audit stratum total drift")
        if (
            exact_counts["accepted_documents"]
            + exact_counts["rejected_documents"]
            != total
            or exact_counts["accepted_utf8_bytes"]
            + exact_counts["rejected_utf8_bytes"]
            != exact_counts["total_utf8_bytes"]
            or exact_counts["selector_routed_documents"]
            + exact_counts["selector_unrouted_documents"]
            != total
            or exact_counts["dedup_survived_documents"]
            + exact_counts["dedup_removed_documents"]
            != total
            or exact_counts["quality_passed_documents"]
            + exact_counts["quality_rejected_documents"]
            != exact_counts["dedup_survived_documents"]
            or exact_counts["accepted_documents"]
            != exact_counts["quality_passed_documents"]
        ):
            raise TurkishCorpusError("sample audit stratum accounting drift")
        if item.get("dedup_survival_rate") != (
            exact_counts["dedup_survived_documents"] / total
        ) or item.get("accepted_document_rate") != (
            exact_counts["accepted_documents"] / total
        ):
            raise TurkishCorpusError("sample audit stratum rate drift")
        aggregate_counts.update(exact_counts)
        reasons = _require_mapping(
            item.get("rejection_reasons"), "sample audit rejection reasons"
        )
        if dict(reasons) != dict(sorted(live_evidence["reasons"][live_key].items())):
            raise TurkishCorpusError("sample audit live rejection-reason drift")
        quality_rejections = _require_mapping(
            item.get("quality_filter_flag_rejections"),
            "sample audit quality-filter flag rejections",
        )
        if dict(quality_rejections) != dict(
            sorted(live_evidence["flags"][live_key].items())
        ):
            raise TurkishCorpusError("sample audit live quality-flag drift")
        if item.get("numeric_distributions") != live_evidence[
            "numeric_distributions"
        ][live_key]:
            raise TurkishCorpusError("sample audit live numeric-distribution drift")
        if any(
            reason not in {"dedup_removed", "selector_unrouted", "quality_filter"}
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for reason, value in reasons.items()
        ) or sum(reasons.values()) != exact_counts["rejected_documents"]:
            raise TurkishCorpusError("sample audit rejection-reason accounting drift")
        ranks_with_rows.add(rank)
        accepted = exact_counts["accepted_documents"]
        if accepted > 0:
            ranks_with_accepted_rows.add(rank)
            if item.get("mixture_id") != "unrouted":
                mixtures_with_accepted_rows.add(str(item["mixture_id"]))
            if planned["source_id"] == "hplt3_tr":
                hplt_bins_with_accepted_rows.add(int(planned["wds_bin"]))
        if exact_counts["rejected_documents"] > 0 and item.get(
            "mixture_id"
        ) != "unrouted":
            mixtures_with_rejected_rows.add(str(item["mixture_id"]))
        for decision in ("accepted", "rejected"):
            available = exact_counts[f"{decision}_documents"]
            sampled = sampled_counts[(*key, decision)]
            if sampled != min(available, requested):
                raise TurkishCorpusError("sample audit example sampling count drift")
            if sampled < requested:
                recomputed_insufficiencies.append(
                    {
                        "source_rank": rank,
                        "source_id": item["source_id"],
                        "mixture_id": item["mixture_id"],
                        "wds_bin": item["wds_bin"],
                        "register": item["register"],
                        "decision": decision,
                        "available_rows": available,
                        "sampled_examples": sampled,
                        "requested_examples": requested,
                        "shortfall": requested - sampled,
                    }
                )
    for rows in example_rows.values():
        for row in rows:
            example_key = (
                row["source_rank"],
                row["source_id"],
                row["mixture_id"],
                row["wds_bin"],
                row["register"],
            )
            if example_key not in strata_keys:
                raise TurkishCorpusError("sample audit example has no matching stratum")
    live_strata_keys = {
        (int(key[0]), key[1], key[2], key[3], key[4])
        for key in live_evidence["counts"]
    }
    if strata_keys != live_strata_keys:
        raise TurkishCorpusError("sample audit/live cluster stratum inventory drift")
    if live_evidence["total_rows"] != sum(
        int(item["rows"]) for item in normalized_outputs
    ):
        raise TurkishCorpusError("sample audit/live cluster row-total drift")
    exact_cluster_totals = {
        "documents": aggregate_counts["total_documents"],
        "accepted_documents": aggregate_counts["accepted_documents"],
        "rejected_documents": aggregate_counts["rejected_documents"],
        "accepted_utf8_bytes": aggregate_counts["accepted_utf8_bytes"],
        "rejected_utf8_bytes": aggregate_counts["rejected_utf8_bytes"],
    }
    if report.get("cluster_totals") != exact_cluster_totals:
        raise TurkishCorpusError("sample audit cluster totals drift")
    receipt_counts = _require_mapping(cluster.get("counts"), "cluster counts")
    exact_audited_counts = {
        "output_rows": aggregate_counts["total_documents"],
        "dedup_kept": aggregate_counts["dedup_survived_documents"],
        "dedup_removed": aggregate_counts["dedup_removed_documents"],
        "quality_kept": aggregate_counts["quality_passed_documents"],
        "quality_removed": aggregate_counts["quality_rejected_documents"],
    }
    for field, value in exact_audited_counts.items():
        if receipt_counts.get(field) != value:
            raise TurkishCorpusError("sample audit cluster receipt count drift")
    processing_summary = _require_mapping(
        report.get("cluster_receipt_processing_summary"),
        "sample audit cluster processing summary",
    )
    if (
        processing_summary.get("receipt_counts")
        != dict(sorted(receipt_counts.items()))
        or processing_summary.get("audited_counts") != exact_audited_counts
        or processing_summary.get("filter_stage_counts")
        != cluster.get("filter_stage_counts")
        or processing_summary.get("formatting_and_safety_incidence")
        != cluster.get("formatting_and_safety_incidence")
    ):
        raise TurkishCorpusError("sample audit cluster processing summary drift")
    if example_sampling.get("insufficiencies") != recomputed_insufficiencies:
        raise TurkishCorpusError("sample audit insufficiency inventory drift")

    accepted_examples = example_rows["accepted"]
    ranks_with_accepted_examples = {row["source_rank"] for row in accepted_examples}
    expected_hplt_bins = {
        int(plan["objects"][rank]["wds_bin"])
        for rank in expected_ranks
        if plan["objects"][rank]["source_id"] == "hplt3_tr"
    }
    hplt_bins_with_accepted_examples = {
        int(row["wds_bin"])
        for row in accepted_examples
        if row["source_id"] == "hplt3_tr"
    }
    coverage = _require_mapping(report.get("coverage"), "sample audit coverage")
    expected_mixtures = sorted(str(item["id"]) for item in policy["mixture"])
    exact_coverage = {
        "expected_mixtures": expected_mixtures,
        "mixtures_with_accepted_rows": sorted(mixtures_with_accepted_rows),
        "mixtures_with_rejected_rows": sorted(mixtures_with_rejected_rows),
        "mixtures_without_accepted_rows": sorted(
            set(expected_mixtures) - mixtures_with_accepted_rows
        ),
        "mixtures_without_rejected_rows": sorted(
            set(expected_mixtures) - mixtures_with_rejected_rows
        ),
        "expected_source_ranks": expected_ranks,
        "source_ranks_with_accepted_rows": sorted(ranks_with_accepted_rows),
        "source_ranks_with_accepted_examples": sorted(ranks_with_accepted_examples),
        "source_ranks_without_accepted_rows": sorted(
            set(expected_ranks) - ranks_with_accepted_rows
        ),
        "source_ranks_without_accepted_examples": sorted(
            set(expected_ranks) - ranks_with_accepted_examples
        ),
        "expected_hplt_wds_bins": sorted(expected_hplt_bins),
        "hplt_wds_bins_with_accepted_rows": sorted(hplt_bins_with_accepted_rows),
        "hplt_wds_bins_with_accepted_examples": sorted(
            hplt_bins_with_accepted_examples
        ),
        "hplt_wds_bins_without_accepted_rows": sorted(
            expected_hplt_bins - hplt_bins_with_accepted_rows
        ),
        "hplt_wds_bins_without_accepted_examples": sorted(
            expected_hplt_bins - hplt_bins_with_accepted_examples
        ),
    }
    for field, expected in exact_coverage.items():
        if coverage.get(field) != expected:
            raise TurkishCorpusError(f"sample audit coverage {field} drift")
    if ranks_with_rows != set(expected_ranks):
        raise TurkishCorpusError("sample audit does not contain every sampled rank")
    if require_complete_accepted_coverage and (
        ranks_with_accepted_rows != set(expected_ranks)
        or ranks_with_accepted_examples != set(expected_ranks)
        or hplt_bins_with_accepted_rows != expected_hplt_bins
        or hplt_bins_with_accepted_examples != expected_hplt_bins
    ):
        raise TurkishCorpusError(
            "sample audit lacks accepted row/example coverage for every rank/WDS bin"
        )
    return report, report_sha


def validate_mixture_quality_approval(
    approval: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    approval_path: str | Path,
) -> str:
    """Validate a human decision and its actual bounded evidence bundle."""

    digest = verify_manifest_hash(approval)
    if (
        approval.get("schema_version") != "3.0"
        or approval.get("kind") != MIXTURE_QUALITY_APPROVAL_KIND
        or approval.get("decision") != "accepted"
        or approval.get("automatic_decision") is not False
        or approval.get("coverage_complete") is not True
        or approval.get("review_confirmation")
        != "bounded_strata_and_accepted_rejected_examples_reviewed"
        or not isinstance(approval.get("reviewer"), str)
        or not approval["reviewer"].strip()
        or not _RFC3339_UTC_RE.fullmatch(str(approval.get("reviewed_at_utc", "")))
    ):
        raise TurkishCorpusError(
            "full backend requires an accepted manual mixture-quality approval"
        )
    bindings = {
        "policy_sha256": _policy_sha256(policy),
        "source_plan_sha256": plan["canonical_sha256"],
        "calibration_sha256": calibration["canonical_sha256"],
    }
    for field, expected in bindings.items():
        if approval.get(field) != expected:
            raise TurkishCorpusError("mixture-quality approval binding drift")
    sample_cluster_sha = str(approval.get("sample_cluster_receipt_sha256") or "")
    if not _SHA256_RE.fullmatch(sample_cluster_sha):
        raise TurkishCorpusError("mixture-quality sample-cluster binding drift")
    bundle = _require_mapping(
        approval.get("evidence_bundle"), "mixture-quality evidence_bundle"
    )
    if bundle.get("schema_version") != "1.0":
        raise TurkishCorpusError("mixture-quality evidence bundle version drift")
    root_raw = str(bundle.get("root") or "")
    root_relative = Path(root_raw)
    approval_source = Path(approval_path).expanduser()
    if approval_source.is_symlink() or not approval_source.is_file():
        raise TurkishCorpusError("mixture-quality approval path is unsafe or missing")
    approval_file = approval_source.resolve()
    if (
        not root_raw
        or root_relative.is_absolute()
        or ".." in root_relative.parts
    ):
        raise TurkishCorpusError("mixture-quality evidence root is unsafe")
    root = (approval_file.parent / root_relative).resolve()
    if root != approval_file.parent and approval_file.parent not in root.parents:
        raise TurkishCorpusError("mixture-quality evidence root escapes approval tree")
    report, report_sha = validate_sample_quality_audit_bundle(
        root,
        _require_mapping(bundle.get("report"), "mixture-quality audit report"),
        policy=policy,
        plan=plan,
        calibration=calibration,
    )
    coverage = _require_mapping(report.get("coverage"), "sample audit coverage")
    if (
        approval.get("sample_quality_audit_sha256") != report_sha
        or approval.get("cluster_receipt_sha256")
        != report["cluster_receipt_sha256"]
        or sample_cluster_sha != report["sample_cluster_receipt_sha256"]
        or approval.get("cluster_receipt_sha256") != sample_cluster_sha
        or approval.get("reviewed_example_files")
        != report["example_sampling"]["files"]
        or coverage.get("mixtures_without_accepted_rows") != []
    ):
        raise TurkishCorpusError("mixture-quality approval evidence drift")
    return digest


def render_sample_quality_plaintext(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the exact human-readable companion to sample-audit JSONL."""

    chunks: list[str] = []
    for index, row in enumerate(rows, 1):
        chunks.extend(
            [
                (
                    f"[{index}] {row['decision']} / rank={row['source_rank']} / "
                    f"{row['source_id']} / {row['mixture_id']} / "
                    f"WDS={row['wds_bin']} / register={row['register']}\n"
                ),
                f"sample_sha256: {row['sample_sha256']}\n",
                f"reason: {row['rejection_reason']}\n",
                "quality_filter_flags: "
                + json.dumps(row["quality_filter_flags"], ensure_ascii=False)
                + "\n",
                f"url: {row['url']}\n",
                str(row["text"]) + "\n\n",
            ]
        )
    return "".join(chunks)


__all__ = [
    "BACKEND_COLUMNS",
    "MACOCU_PREPARATION_KIND",
    "MIXTURE_QUALITY_APPROVAL_KIND",
    "build_resource_projection",
    "fetch_glotlid_model",
    "process_source_object",
    "prepare_macocu_genre",
    "production_processing_binding",
    "production_code_identity",
    "validate_production_code_identity",
    "render_sample_quality_plaintext",
    "resolve_source_plan",
    "run_backend_calibration",
    "run_datatrove_bucket",
    "run_priority_cluster_merge",
    "seal_backend_receipt_from_cluster",
    "seal_resource_approval",
    "seal_source_receipt_from_objects",
    "select_resource_sample_ranks",
    "validate_backend_calibration",
    "validate_resource_approval",
    "validate_mixture_quality_approval",
    "validate_sample_quality_audit_bundle",
    "validate_resource_projection",
    "validate_source_plan",
    "validate_macocu_preparation_manifest",
]
