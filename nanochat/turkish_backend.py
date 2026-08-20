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
import importlib.metadata
import json
import math
import os
import re
import shutil
import struct
import tempfile
import time
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
    SOURCE_RECEIPT_KIND,
    TurkishCorpusError,
    audit_document,
    canonical_text_hash,
    infer_wds_bin,
    iter_input_records,
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
MACOCU_PREPARATION_KIND = "turkish_macocu_genre_preparation"

RESOURCE_BILLING_CONTRACT = {
    "scheduler_partition": "cpu2dq",
    "billable_cpus_per_job": 128,
    "accounting_basis": "projected_stage_wall_seconds_times_billable_cpus_per_job",
    "process_cpu_seconds_role": "efficiency_diagnostic_only",
}

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


def _file_record(path: Path, *, root: Path | None = None, rows: int | None = None) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix() if root is not None else path.name
    result: dict[str, Any] = {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _file_sha256_md5(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()  # noqa: S324 - official upstream integrity checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def _uri_file_record(path: Path, *, rows: int) -> dict[str, Any]:
    return {
        "uri": path.resolve().as_uri(),
        "checksum": {"algorithm": "sha256", "value": file_sha256(path)},
        "size_bytes": path.stat().st_size,
        "rows": rows,
    }


def _elapsed(start_wall: float, start_cpu: float) -> dict[str, float]:
    return {
        "wall_seconds": max(0.0, time.monotonic() - start_wall),
        "cpu_seconds": max(0.0, time.process_time() - start_cpu),
    }


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
        if wds_bin not in {8, 9, 10}:
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
        raise TurkishCorpusError("HPLT plan resolved no WDS 8-10 objects")
    return sorted(objects, key=lambda item: item["uri"])


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
) -> dict[str, Any]:
    """Download/verify MaCoCu once and atomically create deterministic shards."""

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
        upstream_path = upstream_dir / "MaCoCu-Genre.tr.jsonl.gz"
        staged = _stage_source_object(
            {
                "uri": MACOCU_SOURCE_URL,
                "size_bytes": MACOCU_SIZE_BYTES,
                "expected_checksums": [{"algorithm": "md5", "value": MACOCU_MD5}],
            },
            upstream_path,
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

        with gzip.open(upstream_path, "rb") as source_stream:
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
                    "path": upstream_path.relative_to(build).as_posix(),
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
    expected_derived = {MACOCU_SOURCE_ID} if MACOCU_SOURCE_ID in expected_sources else set()
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
        allowed_scheme = "file" if source_id == MACOCU_SOURCE_ID else "https"
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
    if seen_sources != expected_sources:
        raise TurkishCorpusError("source plan does not cover every configured source")
    if macocu_manifest_objects and {
        uri for uri in seen_uris if uri in macocu_manifest_objects
    } != set(macocu_manifest_objects):
        raise TurkishCorpusError("source plan does not cover every prepared MaCoCu shard")
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
) -> dict[str, Any]:
    """Resolve every configured source to immutable object identities."""

    validate_corpus_policy(policy)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite source plan: {destination}")
    objects: list[dict[str, Any]] = []
    derived_sources: dict[str, Any] = {}
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
        else:
            resolved = _parse_hub_objects(source, api=hub_api)
        objects.extend(resolved)
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
                "selected_wds_bins": [8, 9, 10],
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
        "local_policy_audit": "nanochat.turkish_corpus.audit_document after safe formatting",
        "no_code": True,
    }
    body = {
        "implementation": configured["implementation"],
        "official_fineweb2_control": official,
        "project_additions": additions,
    }
    return body | {
        "binding_sha256": hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
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


def _quality_score(record: Mapping[str, Any], lid_probability: float) -> float:
    candidates: list[float] = [lid_probability]
    for field in (
        "quality_score",
        "score",
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


def _redact_sample_text(text: str) -> str:
    text = _EMAIL_RE.sub("<email>", text)
    text = _IP_RE.sub("<ip>", text)
    return _PHONE_RE.sub("<telefon>", text)


def _stage_source_object(
    item: Mapping[str, Any], destination: Path, *, request_get: Any = requests.get
) -> dict[str, Any]:
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
    try:
        with destination.open("wb") as output:
            iterable = iter(lambda: stream.read(8 * 1024 * 1024), b"") if response is None else stream
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
    return {
        "uri": uri,
        "size_bytes": size,
        "sha256": observed["sha256"],
        "upstream_checksums_verified": list(item["expected_checksums"]),
    }


def _signature_files(root: Path, rank: int) -> list[Path]:
    return [root / f"bucket_{bucket:03d}" / f"{rank:05d}.minhash.sig" for bucket in range(14)]


def _iter_candidate_documents(path: Path) -> Iterator[Any]:
    try:
        from datatrove.data import Document
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError("DataTrove Document type is unavailable") from exc
    parquet = pq.ParquetFile(path)
    index = 0
    for batch in parquet.iter_batches(batch_size=2048, columns=["text", "document_id"]):
        for row in batch.to_pylist():
            yield Document(text=row["text"], id=row["document_id"])
            index += 1


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
    rank = receipt.get("rank")
    if not isinstance(rank, int) or rank < 0 or rank >= len(plan["objects"]):
        raise TurkishCorpusError("object receipt rank is invalid")
    expected = plan["objects"][rank]
    if receipt.get("source_id") != expected["source_id"] or receipt.get("source_uri") != expected["uri"]:
        raise TurkishCorpusError("object receipt source identity drift")
    output = _require_mapping(receipt.get("candidate_file"), "candidate_file")
    path = run_root / output["path"]
    if (
        path.is_symlink()
        or path.stat().st_size != output["size_bytes"]
        or file_sha256(path) != output["sha256"]
        or pq.ParquetFile(path).metadata.num_rows != output["rows"]
    ):
        raise TurkishCorpusError("object candidate file drift")
    signatures = receipt.get("signature_files")
    if not isinstance(signatures, list) or len(signatures) != 14:
        raise TurkishCorpusError("object receipt must contain fourteen signatures")
    for record in signatures:
        sig_path = run_root / record["path"]
        if sig_path.stat().st_size != record["size_bytes"] or file_sha256(sig_path) != record["sha256"]:
            raise TurkishCorpusError("object MinHash signature drift")


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
            load_json_strict(resource_approval_path), plan=plan, policy=policy
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
        source_file = _stage_source_object(item, staged, request_get=request_get)
        download_telemetry = _elapsed(download_wall, download_cpu)
        model = _load_glotlid_model(Path(model_path), policy)
        candidate_path = object_dir / "candidates.parquet"
        writer = _ParquetBatchWriter(candidate_path, _INTERNAL_SCHEMA)
        score_wall, score_cpu = time.monotonic(), time.process_time()
        characters_seen = 0
        bytes_seen = 0
        candidate_chars = 0
        for row_index, record in enumerate(iter_input_records(staged)):
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
            score = _quality_score(record, lid["lid_probability"])
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
        writer.close()
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
        signature_stage.run(
            _iter_candidate_documents(candidate_path),
            rank=rank,
            world_size=len(plan["objects"]),
        )
        signature_telemetry = _elapsed(signature_wall, signature_cpu)
    signature_records = [
        _file_record(path, root=run_root) for path in _signature_files(signatures_root, rank)
    ]
    candidate_record = _file_record(
        candidate_path, root=run_root, rows=pq.ParquetFile(candidate_path).metadata.num_rows
    )
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
) -> None:
    verify_manifest_hash(receipt)
    if receipt.get("kind") != BUCKET_RECEIPT_KIND or receipt.get("schema_version") != "1.0":
        raise TurkishCorpusError("unexpected bucket receipt")
    if (
        receipt.get("source_plan_sha256") != plan["canonical_sha256"]
        or receipt.get("calibration_sha256") != calibration["canonical_sha256"]
        or receipt.get("sample_mode") is not sample_mode
    ):
        raise TurkishCorpusError("bucket receipt binding drift")
    rank = receipt.get("rank")
    if not isinstance(rank, int) or not 0 <= rank < 14:
        raise TurkishCorpusError("bucket rank must be in [0,14)")
    output = _require_mapping(receipt.get("output"), "bucket output")
    path = run_root / output["path"]
    if path.stat().st_size != output["size_bytes"] or file_sha256(path) != output["sha256"]:
        raise TurkishCorpusError("DataTrove bucket output drift")
    if output.get("duplicate_edges") != output["size_bytes"] // 16 or output["size_bytes"] % 16:
        raise TurkishCorpusError("DataTrove .dups structural size drift")


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
            load_json_strict(resource_approval_path), plan=plan, policy=policy
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
        )
        return receipt
    output_root = run_root / "bucket_matches"
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets
    except ImportError as exc:  # pragma: no cover
        raise TurkishCorpusError("pinned DataTrove MinHash buckets are unavailable") from exc
    config = _minhash_config()
    stage = MinhashDedupBuckets(
        input_folder=str(run_root / "signatures"),
        output_folder=str(output_root),
        config=config,
        only_dedup_in_index=False,
        lines_to_buffer=256,
    )
    start_wall, start_cpu = time.monotonic(), time.process_time()
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


def _iter_candidate_rows(path: Path) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048):
        yield from batch.to_pylist()


def _load_bucket_receipts(
    run_root: Path,
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    sample_mode: bool,
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
    if not sample_mode:
        if resource_approval_path is None:
            raise TurkishCorpusError("full priority clustering requires resource approval")
        validate_resource_approval(
            load_json_strict(resource_approval_path), plan=plan, policy=policy
        )
    run_root = Path(run_dir)
    objects = _load_object_receipts(
        run_root, plan, calibration, sample_mode=sample_mode
    )
    buckets = _load_bucket_receipts(
        run_root, plan, calibration, sample_mode=sample_mode
    )
    receipt_path = run_root / "cluster_receipt.json"
    if receipt_path.exists():
        receipt = load_json_strict(receipt_path)
        verify_manifest_hash(receipt)
        if (
            receipt.get("kind") != CLUSTER_RECEIPT_KIND
            or receipt.get("source_plan_sha256") != plan["canonical_sha256"]
            or receipt.get("sample_mode") is not sample_mode
        ):
            raise TurkishCorpusError("cluster receipt binding drift")
        for item in receipt["output_files"]:
            path = run_root / item["path"]
            if path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
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
        path = run_root / bucket["output"]["path"]
        with path.open("rb") as handle:
            while chunk := handle.read(16):
                if len(chunk) != 16:
                    raise TurkishCorpusError("truncated DataTrove duplicate edge")
                f1, d1, f2, d2 = struct.unpack("<4I", chunk)
                if f1 == (1 << 32) - 1 or f2 == (1 << 32) - 1:
                    raise TurkishCorpusError("unexpected external-index sentinel in global dedup")
                union.union((f1, d1), (f2, d2))
                edge_count += 1
    priority = {
        source_id: index
        for index, source_id in enumerate(policy["deduplication"]["source_priority"])
    }
    winners: dict[tuple[int, int], tuple[tuple[Any, ...], tuple[int, int], str]] = {}
    seen_nodes: set[tuple[int, int]] = set()
    for object_receipt in objects:
        path = run_root / object_receipt["candidate_file"]["path"]
        for row in _iter_candidate_rows(path):
            node = (int(row["candidate_rank"]), int(row["candidate_doc_index"]))
            if node[0] != object_receipt["rank"]:
                raise TurkishCorpusError("candidate rank/file identity drift")
            root = union.find(node) if node in union.parent else node
            key = (
                priority[row["source_id"]],
                -float(row["quality_score"]),
                str(row["document_id"]),
                node,
            )
            current = winners.get(root)
            if current is None or key < current[0]:
                winners[root] = (key, node, str(row["document_id"]))
            seen_nodes.add(node)
    missing_edge_nodes = set(union.parent) - seen_nodes
    if missing_edge_nodes:
        raise TurkishCorpusError(
            f"DataTrove duplicate edges reference {len(missing_edge_nodes)} absent candidate rows"
        )
    processors = _ProductionProcessors(policy)
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
        input_path = run_root / object_receipt["candidate_file"]["path"]
        output_path = output_root / f"{object_receipt['rank']:05d}.parquet"
        writer = _ParquetBatchWriter(output_path, _BACKEND_SCHEMA)
        for row in _iter_candidate_rows(input_path):
            node = (int(row.pop("candidate_rank")), int(row.pop("candidate_doc_index")))
            root = union.find(node) if node in union.parent else node
            winner = winners[root]
            dedup_keep = node == winner[1]
            row["dedup_cluster_id"] = _cluster_id(winner[2])
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
        rows = pq.ParquetFile(output_path).metadata.num_rows
        if rows:
            output_files.append(_file_record(output_path, root=run_root, rows=rows))
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
    }
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": CLUSTER_RECEIPT_KIND,
            "sample_mode": sample_mode,
            "source_plan_sha256": plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "object_receipt_sha256": [item["canonical_sha256"] for item in objects],
            "bucket_receipt_sha256": [item["canonical_sha256"] for item in buckets],
            "winner_policy": "minimum_source_priority_then_negative_quality_then_stable_id",
            "processing": processors.binding,
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


def seal_source_receipt_from_objects(
    policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_dir: str | Path,
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
    objects = _load_object_receipts(
        run_root, plan, calibration, sample_mode=False
    )
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
            "object_receipt_sha256": [item["canonical_sha256"] for item in objects],
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
    cluster = load_json_strict(run_root / "cluster_receipt.json")
    verify_manifest_hash(cluster)
    if cluster.get("kind") != CLUSTER_RECEIPT_KIND or cluster.get("sample_mode") is not False:
        raise TurkishCorpusError("production cluster receipt is missing/invalid")
    if cluster.get("source_plan_sha256") != plan["canonical_sha256"]:
        raise TurkishCorpusError("cluster/source-plan binding drift")
    file_records: list[dict[str, Any]] = []
    for item in cluster["output_files"]:
        path = run_root / item["path"]
        if (
            path.stat().st_size != item["size_bytes"]
            or file_sha256(path) != item["sha256"]
            or pq.ParquetFile(path).metadata.num_rows != item["rows"]
        ):
            raise TurkishCorpusError("cluster output file drift before backend seal")
        file_records.append(_uri_file_record(path, rows=item["rows"]))
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
    if report.get("schema_version") != "1.0" or report.get("kind") != RESOURCE_REPORT_KIND:
        raise TurkishCorpusError("unexpected resource projection kind/version")
    for key in ("policy_sha256", "source_plan_sha256", "calibration_sha256"):
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
    if report.get("manual_approval_required") is not True:
        raise TurkishCorpusError("resource projection must require manual approval")
    if not isinstance(report.get("automated_gate_passed"), bool):
        raise TurkishCorpusError("resource projection automated gate result is missing")
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
    buckets = _load_bucket_receipts(root, plan, calibration, sample_mode=True)
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
    sample_backend_bytes = sum(item["size_bytes"] for item in cluster["output_files"])
    projected_backend_bytes = (
        sample_backend_bytes * projected_candidates / sample_candidates
        if sample_candidates
        else 0.0
    )
    raw_largest = max(item["size_bytes"] for item in plan["objects"])
    projected_peak = (
        raw_largest
        + projected_candidate_bytes
        + projected_signature_bytes
        + projected_dups_bytes
        + projected_backend_bytes
    ) * safety_factor
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
    automated_pass = projected_peak <= limit
    report = seal_manifest(
        {
            "schema_version": "1.0",
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
                "signature_bytes": projected_signature_bytes,
                "duplicate_edges": projected_edges,
                "duplicate_edge_bytes": projected_dups_bytes,
                "backend_output_bytes": projected_backend_bytes,
                **accounting,
                "peak_disk_bytes_with_safety_factor": projected_peak,
                "peak_disk_model": "raw_largest+candidates+signatures+dups+backend_output; conservative until streaming cleanup is proven",
            },
            "limits": {
                "policy_max_peak_disk_bytes": policy["materialization"]["max_peak_disk_bytes"],
                "reported_quota_headroom_bytes": quota_headroom_bytes,
                "effective_peak_limit_bytes": limit,
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
    output_path: str | Path,
    *,
    reviewer: str,
    reviewed_at_utc: str,
    decision: str,
    notes: str = "",
) -> dict[str, Any]:
    report = load_json_strict(report_path)
    report_hash = validate_resource_projection(report)
    if report.get("automated_gate_passed") is not True and decision == "accepted":
        raise TurkishCorpusError("cannot accept a resource projection that exceeds the hard peak limit")
    if not reviewer.strip() or not _RFC3339_UTC_RE.fullmatch(reviewed_at_utc):
        raise TurkishCorpusError("resource approval requires reviewer and RFC3339 UTC timestamp")
    if decision not in {"accepted", "rejected"}:
        raise TurkishCorpusError("resource approval decision must be accepted/rejected")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite resource approval: {destination}")
    approval = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": RESOURCE_APPROVAL_KIND,
            "resource_report_sha256": report_hash,
            "policy_sha256": report["policy_sha256"],
            "source_plan_sha256": report["source_plan_sha256"],
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
    approval: Mapping[str, Any], *, plan: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    verify_manifest_hash(approval)
    if (
        approval.get("schema_version") != "1.0"
        or approval.get("kind") != RESOURCE_APPROVAL_KIND
        or approval.get("decision") != "accepted"
    ):
        raise TurkishCorpusError("full backend requires an accepted resource approval")
    if (
        approval.get("source_plan_sha256") != plan["canonical_sha256"]
        or approval.get("policy_sha256") != _policy_sha256(policy)
        or not _SHA256_RE.fullmatch(str(approval.get("resource_report_sha256", "")))
    ):
        raise TurkishCorpusError("resource approval binding drift")
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


__all__ = [
    "BACKEND_COLUMNS",
    "MACOCU_PREPARATION_KIND",
    "build_resource_projection",
    "fetch_glotlid_model",
    "process_source_object",
    "prepare_macocu_genre",
    "production_processing_binding",
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
    "validate_resource_projection",
    "validate_source_plan",
    "validate_macocu_preparation_manifest",
]
