"""Deterministic, dependency-free experiment manifest primitives.

The project has two provenance profiles:

``strict``
    New work. Dataset revisions must be immutable and all required provenance
    fields must be present.

``legacy-partial``
    Historical evidence. A mutable requested revision and an unresolved
    revision are allowed, but the manifest must say that it is partial. File
    paths, sizes, and hashes remain subject to the same safety checks.

JSON Schema files document the wire format.  This module intentionally uses
only the standard library so local provenance checks do not depend on a
particular training or release environment.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


SCHEMA_VERSION = "1.0"
STRICT_PROFILE = "strict"
LEGACY_PROFILE = "legacy-partial"
SELF_HASH_FIELD = "canonical_sha256"
MUTABLE_REVISIONS = frozenset({"main", "master", "latest", "head", "null"})

_SHA256_HEX = frozenset("0123456789abcdef")
_DATASET_ROOT_FIELDS = frozenset(
    {
        "$schema",
        "schema_version",
        "manifest_type",
        "profile",
        "dataset",
        "text_column",
        "ordered_files",
        "validation_file",
        "created_by",
        SELF_HASH_FIELD,
        "metadata",
    }
)
_ARTIFACT_ROOT_FIELDS = frozenset(
    {
        "$schema",
        "schema_version",
        "manifest_type",
        "profile",
        "artifact_origin",
        "packaging_provenance",
        "data_provenance",
        "segmenter_provenance",
        "metric_environment",
        "publication_target",
        "files",
        SELF_HASH_FIELD,
        "metadata",
    }
)
_DATASET_FIELDS = frozenset(
    {"repo_id", "path", "requested_revision", "resolved_revision", "repo_type"}
)
_DATASET_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_ARTIFACT_FILE_FIELDS = frozenset({"path", "role", "size_bytes", "sha256"})
_CREATOR_FIELDS = frozenset(
    {
        "git_commit",
        "environment_lock_sha256",
        "tool",
        "tool_version",
        "created_at_utc",
    }
)


class ManifestValidationError(ValueError):
    """Raised when a manifest is malformed, unsafe, or internally inconsistent."""


def _error(location: str, message: str) -> ManifestValidationError:
    return ManifestValidationError(f"{location}: {message}")


def _validate_json_value(value: Any, location: str = "$") -> None:
    """Reject values that JSON would coerce or serialize non-deterministically."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(location, "non-finite numbers are not valid canonical JSON")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error(location, "JSON object keys must be strings")
            _validate_json_value(item, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    raise _error(location, f"unsupported JSON value type {type(value).__name__!r}")


def canonical_json_bytes(
    value: Any, *, self_hash_field: str | None = None
) -> bytes:
    """Return canonical UTF-8 JSON bytes.

    Object keys are sorted, insignificant whitespace is omitted, Unicode is
    retained as UTF-8, NaN/infinities are rejected, and one final newline is
    emitted. Array order is intentionally preserved because dataset and
    artifact inventories are ordered contracts. When ``self_hash_field`` is
    provided for a root object, that field is forced to null.
    """

    canonical_value = value
    if self_hash_field is not None:
        if not isinstance(self_hash_field, str) or not self_hash_field:
            raise _error("self_hash_field", "must be null or a non-empty string")
        if isinstance(value, Mapping):
            canonical_value = copy.deepcopy(dict(value))
            canonical_value[self_hash_field] = None
    _validate_json_value(canonical_value)
    try:
        rendered = json.dumps(
            canonical_value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (rendered + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _error("$", f"cannot encode canonical JSON: {exc}") from exc


def canonical_json(value: Any, *, self_hash_field: str | None = None) -> str:
    """Return canonical JSON as text; see :func:`canonical_json_bytes`."""

    return canonical_json_bytes(value, self_hash_field=self_hash_field).decode("utf-8")


def canonical_manifest_bytes(
    manifest: Mapping[str, Any], *, self_hash_field: str = SELF_HASH_FIELD
) -> bytes:
    """Canonicalize a manifest with its root self-hash forced to JSON null.

    The field is inserted when absent, which makes the pre-seal and post-seal
    representations hash identically. Nested fields with the same name are not
    modified.
    """

    if not isinstance(manifest, Mapping):
        raise _error("$", "manifest must be a JSON object")
    return canonical_json_bytes(manifest, self_hash_field=self_hash_field)


def manifest_sha256(
    manifest: Mapping[str, Any], *, self_hash_field: str = SELF_HASH_FIELD
) -> str:
    """Hash a manifest deterministically without hashing the hash itself."""

    return hashlib.sha256(
        canonical_manifest_bytes(manifest, self_hash_field=self_hash_field)
    ).hexdigest()


# A readable alias for callers whose manifests name the field canonical_sha256.
canonical_sha256 = manifest_sha256


def seal_manifest(
    manifest: Mapping[str, Any], *, self_hash_field: str = SELF_HASH_FIELD
) -> dict[str, Any]:
    """Return a deep-copied manifest containing its deterministic self-hash."""

    if not isinstance(manifest, Mapping):
        raise _error("$", "manifest must be a JSON object")
    sealed = copy.deepcopy(dict(manifest))
    sealed[self_hash_field] = manifest_sha256(
        sealed, self_hash_field=self_hash_field
    )
    return sealed


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def verify_manifest_hash(
    manifest: Mapping[str, Any],
    *,
    self_hash_field: str = SELF_HASH_FIELD,
    required: bool = True,
) -> str:
    """Validate a stored manifest hash and return the recomputed digest.

    A missing or null hash is allowed only when ``required`` is false. This
    supports contracts that keep the digest in a sidecar.
    """

    if not isinstance(manifest, Mapping):
        raise _error("$", "manifest must be a JSON object")
    stored = manifest.get(self_hash_field)
    computed = manifest_sha256(manifest, self_hash_field=self_hash_field)
    if stored is None:
        if required:
            raise _error(self_hash_field, "a stored SHA-256 is required")
        return computed
    if not _is_sha256(stored):
        raise _error(self_hash_field, "must be 64 lowercase hexadecimal characters")
    if not hmac.compare_digest(stored, computed):
        raise _error(self_hash_field, "does not match the canonical manifest hash")
    return computed


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of one regular file using bounded memory."""

    source = Path(path)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not source.is_file():
        raise ManifestValidationError(f"not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_strict(path: str | Path) -> Any:
    """Load UTF-8 JSON while rejecting duplicate keys and non-finite numbers."""

    source = Path(path)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise _error("$", f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_non_finite(token: str) -> None:
        raise _error("$", f"non-finite JSON number {token!r} is forbidden")

    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_non_finite,
            )
    except ManifestValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot load JSON manifest {source}: {exc}") from exc


def validate_relative_path(value: Any, *, location: str = "path") -> str:
    """Validate and return a normalized, relative POSIX manifest path."""

    if not isinstance(value, str) or not value:
        raise _error(location, "must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise _error(location, "must not contain control characters")
    if "\\" in value:
        raise _error(location, "must use POSIX '/' separators")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    parts = value.split("/")
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise _error(location, "absolute paths are forbidden")
    if any(part in {"", ".", ".."} for part in parts):
        raise _error(location, "must be normalized and contain no '.' or '..' segments")
    if posix_path.as_posix() != value:
        raise _error(location, "must be a normalized POSIX path")
    return value


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(location, "must be a JSON object")
    return value


def _require_fields(value: Mapping[str, Any], fields: Iterable[str], location: str) -> None:
    missing = sorted(field for field in fields if field not in value)
    if missing:
        raise _error(location, f"missing required field(s): {', '.join(missing)}")


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: frozenset[str], location: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise _error(location, f"unknown field(s): {', '.join(unknown)}")


def _validate_nonempty_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(location, "must be a non-empty string")
    if value != value.strip():
        raise _error(location, "must not have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise _error(location, "must not contain control characters")
    return value


def _normalize_profile(
    value: Any, location: str = "profile", *, allow_legacy_alias: bool = False
) -> str:
    if value == "legacy" and allow_legacy_alias:
        return LEGACY_PROFILE
    if value not in {STRICT_PROFILE, LEGACY_PROFILE}:
        raise _error(location, f"must be {STRICT_PROFILE!r} or {LEGACY_PROFILE!r}")
    return value


def _effective_profile(manifest: Mapping[str, Any], requested: str | None) -> str:
    declared = _normalize_profile(manifest["profile"]) if "profile" in manifest else None
    selected = (
        _normalize_profile(requested, allow_legacy_alias=True)
        if requested is not None
        else None
    )
    if declared is not None and selected is not None and declared != selected:
        raise _error("profile", "declared profile disagrees with validation profile")
    effective = selected or declared or STRICT_PROFILE
    if effective == LEGACY_PROFILE and declared != LEGACY_PROFILE:
        raise _error("profile", "legacy manifests must explicitly declare 'legacy-partial'")
    return effective


def validate_immutable_revision(value: Any, *, location: str = "revision") -> str:
    """Reject null, empty, and well-known mutable revision names."""

    revision = _validate_nonempty_text(value, location)
    if revision.casefold() in MUTABLE_REVISIONS:
        raise _error(location, f"mutable revision {revision!r} is forbidden in strict mode")
    return revision


def validate_hub_commit(value: Any, *, location: str = "revision") -> str:
    """Validate a full lowercase 40-character Hugging Face commit SHA."""

    revision = validate_immutable_revision(value, location=location)
    if len(revision) != 40 or any(character not in _SHA256_HEX for character in revision):
        raise _error(location, "must be a full lowercase 40-character Hub commit SHA")
    return revision


def validate_git_commit(value: Any, *, location: str = "git_commit") -> str:
    """Validate a full lowercase SHA-1 or SHA-256 Git commit identity."""

    commit = _validate_nonempty_text(value, location)
    if len(commit) not in {40, 64} or any(
        character not in _SHA256_HEX for character in commit
    ):
        raise _error(location, "must be a full lowercase Git commit SHA")
    return commit


def validate_file_record(
    record: Any,
    *,
    location: str = "file",
    require_role: bool = False,
    allow_role: bool = True,
) -> None:
    """Validate one content-addressed file record.

    Dataset records contain path, size, and digest. Artifact records additionally
    require a semantic role. Generic callers may accept the role as optional.
    """

    value = _require_mapping(record, location)
    required = set(_DATASET_FILE_FIELDS)
    if require_role:
        required.add("role")
    allowed = _ARTIFACT_FILE_FIELDS if allow_role else _DATASET_FILE_FIELDS
    _require_fields(value, required, location)
    _reject_unknown_fields(value, allowed, location)
    validate_relative_path(value["path"], location=f"{location}.path")
    if "role" in value:
        role = _validate_nonempty_text(value["role"], f"{location}.role")
        if role[0] not in "abcdefghijklmnopqrstuvwxyz" or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in role
        ):
            raise _error(
                f"{location}.role",
                "must be a lowercase identifier using letters, digits, '_' or '-'",
            )
    size = value["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise _error(f"{location}.size_bytes", "must be a non-negative integer")
    if not _is_sha256(value["sha256"]):
        raise _error(
            f"{location}.sha256", "must be 64 lowercase hexadecimal characters"
        )


def validate_ordered_file_records(
    records: Any,
    *,
    location: str = "ordered_files",
    require_role: bool = False,
    allow_role: bool = True,
) -> tuple[str, ...]:
    """Validate an ordered, non-empty list of unique file records."""

    if not isinstance(records, list) or not records:
        raise _error(location, "must be a non-empty JSON array")
    paths: list[str] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        item_location = f"{location}[{index}]"
        validate_file_record(
            record,
            location=item_location,
            require_role=require_role,
            allow_role=allow_role,
        )
        path = record["path"]
        if path in seen:
            raise _error(item_location, f"duplicate path {path!r}")
        paths.append(path)
        seen.add(path)
    return tuple(paths)


def _validate_created_by(
    value: Any,
    *,
    profile: str,
    location: str = "created_by",
) -> None:
    creator = _require_mapping(value, location)
    _require_fields(creator, {"git_commit"}, location)
    _reject_unknown_fields(creator, _CREATOR_FIELDS, location)
    commit = creator["git_commit"]
    if profile == STRICT_PROFILE:
        validate_git_commit(commit, location=f"{location}.git_commit")
    elif commit is not None and commit != "":
        _validate_nonempty_text(commit, f"{location}.git_commit")
    for field in ("environment_lock_sha256",):
        if field in creator and creator[field] is not None and not _is_sha256(creator[field]):
            raise _error(
                f"{location}.{field}", "must be 64 lowercase hexadecimal characters"
            )
    for field in ("tool", "tool_version", "created_at_utc"):
        if field in creator:
            _validate_nonempty_text(creator[field], f"{location}.{field}")


def _validate_packaging_provenance(
    value: Any, *, profile: str, location: str = "packaging_provenance"
) -> None:
    """Validate the common immutable portion of extensible packaging metadata."""

    provenance = _require_mapping(value, location)
    _require_fields(provenance, {"git_commit"}, location)
    commit = provenance["git_commit"]
    if profile == STRICT_PROFILE:
        validate_git_commit(commit, location=f"{location}.git_commit")
    elif commit is not None and commit != "":
        _validate_nonempty_text(commit, f"{location}.git_commit")
    if "environment_lock_sha256" in provenance:
        lock_hash = provenance["environment_lock_sha256"]
        if lock_hash is not None and not _is_sha256(lock_hash):
            raise _error(
                f"{location}.environment_lock_sha256",
                "must be 64 lowercase hexadecimal characters",
            )
    if "source_date_epoch" in provenance:
        epoch = provenance["source_date_epoch"]
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise _error(
                f"{location}.source_date_epoch", "must be a non-negative integer"
            )


def _validate_optional_hash(manifest: Mapping[str, Any]) -> None:
    if SELF_HASH_FIELD not in manifest or manifest[SELF_HASH_FIELD] is None:
        return
    verify_manifest_hash(manifest)


def validate_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    profile: str | None = None,
    require_acquisition_fields: bool = False,
) -> None:
    """Validate a dataset manifest under the strict or legacy profile.

    The strict profile always applies the complete on-disk contract: explicit
    validation-file identity and a full creator Git commit. The
    ``require_acquisition_fields`` argument is retained for call-site
    compatibility; it cannot weaken strict validation.
    """

    value = _require_mapping(manifest, "$")
    _validate_json_value(value)
    _require_fields(
        value,
        {"schema_version", "dataset", "text_column", "ordered_files", "created_by"},
        "$",
    )
    _reject_unknown_fields(value, _DATASET_ROOT_FIELDS, "$")
    if value["schema_version"] != SCHEMA_VERSION:
        raise _error("schema_version", f"must equal {SCHEMA_VERSION!r}")
    if value.get("manifest_type", "dataset") != "dataset":
        raise _error("manifest_type", "must equal 'dataset'")
    if "$schema" in value:
        _validate_nonempty_text(value["$schema"], "$schema")
    effective_profile = _effective_profile(value, profile)

    dataset = _require_mapping(value["dataset"], "dataset")
    _require_fields(
        dataset,
        {"repo_id", "path", "requested_revision", "resolved_revision"},
        "dataset",
    )
    _reject_unknown_fields(dataset, _DATASET_FIELDS, "dataset")
    _validate_nonempty_text(dataset["repo_id"], "dataset.repo_id")
    validate_relative_path(dataset["path"], location="dataset.path")
    if "repo_type" in dataset and dataset["repo_type"] != "dataset":
        raise _error("dataset.repo_type", "must equal 'dataset'")

    if effective_profile == STRICT_PROFILE:
        requested_revision = validate_hub_commit(
            dataset["requested_revision"], location="dataset.requested_revision"
        )
        resolved_revision = validate_hub_commit(
            dataset["resolved_revision"], location="dataset.resolved_revision"
        )
        if requested_revision != resolved_revision:
            raise _error(
                "dataset.resolved_revision",
                "must equal requested_revision in a strict acquisition",
            )
    else:
        _validate_nonempty_text(
            dataset["requested_revision"], "dataset.requested_revision"
        )
        resolved = dataset["resolved_revision"]
        if resolved is not None:
            _validate_nonempty_text(resolved, "dataset.resolved_revision")

    _validate_nonempty_text(value["text_column"], "text_column")
    ordered_paths = validate_ordered_file_records(
        value["ordered_files"], allow_role=False
    )
    if effective_profile == STRICT_PROFILE and "validation_file" not in value:
        raise _error("$", "missing required field(s): validation_file")
    if "validation_file" in value:
        validation_file = validate_relative_path(
            value["validation_file"], location="validation_file"
        )
        if validation_file not in ordered_paths:
            raise _error(
                "validation_file", "must identify one path in ordered_files"
            )
    _validate_created_by(
        value["created_by"],
        profile=effective_profile,
    )
    if "metadata" in value:
        _require_mapping(value["metadata"], "metadata")
    _validate_optional_hash(value)


def validate_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    profile: str | None = None,
    inventory_root: str | Path | None = None,
    require_exact_inventory: bool = False,
    ignored_paths: Iterable[str] = (),
) -> None:
    """Validate an artifact manifest and optionally verify its local files."""

    value = _require_mapping(manifest, "$")
    _validate_json_value(value)
    _require_fields(
        value,
        {
            "schema_version",
            "artifact_origin",
            "packaging_provenance",
            "data_provenance",
            "segmenter_provenance",
            "metric_environment",
            "publication_target",
            "files",
        },
        "$",
    )
    _reject_unknown_fields(value, _ARTIFACT_ROOT_FIELDS, "$")
    if value["schema_version"] != SCHEMA_VERSION:
        raise _error("schema_version", f"must equal {SCHEMA_VERSION!r}")
    if value.get("manifest_type", "artifact") != "artifact":
        raise _error("manifest_type", "must equal 'artifact'")
    if "$schema" in value:
        _validate_nonempty_text(value["$schema"], "$schema")
    effective_profile = _effective_profile(value, profile)

    _require_mapping(value["artifact_origin"], "artifact_origin")
    _validate_packaging_provenance(
        value["packaging_provenance"], profile=effective_profile
    )
    _require_mapping(value["data_provenance"], "data_provenance")
    segmenter = value["segmenter_provenance"]
    if segmenter is not None:
        _require_mapping(segmenter, "segmenter_provenance")
    _require_mapping(value["metric_environment"], "metric_environment")
    _require_mapping(value["publication_target"], "publication_target")
    validate_ordered_file_records(
        value["files"], location="files", require_role=True
    )
    if "metadata" in value:
        _require_mapping(value["metadata"], "metadata")
    _validate_optional_hash(value)

    if inventory_root is not None:
        verify_file_inventory(
            inventory_root,
            value["files"],
            require_exact=require_exact_inventory,
            ignored_paths=ignored_paths,
            require_role=True,
            location="files",
        )
    elif require_exact_inventory:
        raise _error("inventory_root", "is required when require_exact_inventory is true")


def _ensure_no_symlink_components(root: Path, relative_path: str) -> None:
    current = root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise _error(relative_path, "symlinks are forbidden in a verified inventory")


def verify_file_inventory(
    root: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    require_exact: bool = False,
    ignored_paths: Iterable[str] = (),
    require_role: bool = False,
    location: str = "ordered_files",
) -> None:
    """Verify recorded size/hash values against a local artifact directory.

    When ``require_exact`` is true, every regular file below ``root`` must be
    recorded unless its relative POSIX path is listed in ``ignored_paths``.
    Symlinks are rejected both for safety and to keep inventory meaning stable.
    """

    root_path = Path(root)
    if root_path.is_symlink():
        raise _error("root", "a verified inventory root must not be a symlink")
    if not root_path.is_dir():
        raise _error("root", f"not a directory: {root_path}")
    paths = validate_ordered_file_records(
        records, location=location, require_role=require_role
    )
    root_resolved = root_path.resolve()

    ignored: set[str] = set()
    for index, path in enumerate(ignored_paths):
        ignored.add(validate_relative_path(path, location=f"ignored_paths[{index}]"))

    for index, (relative_path, record) in enumerate(zip(paths, records)):
        _ensure_no_symlink_components(root_path, relative_path)
        candidate = root_path.joinpath(*PurePosixPath(relative_path).parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root_resolved)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise _error(
                f"{location}[{index}].path",
                f"does not resolve to a file contained by the inventory root: {relative_path!r}",
            ) from exc
        if not resolved.is_file():
            raise _error(f"{location}[{index}].path", "is not a regular file")
        size = resolved.stat().st_size
        if size != record["size_bytes"]:
            raise _error(
                f"{location}[{index}].size_bytes",
                f"recorded {record['size_bytes']}, found {size}",
            )
        digest = file_sha256(resolved)
        if not hmac.compare_digest(digest, record["sha256"]):
            raise _error(
                f"{location}[{index}].sha256",
                f"recorded {record['sha256']}, found {digest}",
            )

    if require_exact:
        discovered: set[str] = set()
        for candidate in root_path.rglob("*"):
            relative_path = candidate.relative_to(root_path).as_posix()
            if candidate.is_symlink():
                raise _error(relative_path, "symlinks are forbidden in a verified inventory")
            if candidate.is_file() and relative_path not in ignored:
                discovered.add(relative_path)
        recorded = set(paths)
        if discovered != recorded:
            missing = sorted(discovered - recorded)
            absent = sorted(recorded - discovered)
            details: list[str] = []
            if missing:
                details.append(f"unrecorded files: {', '.join(missing)}")
            if absent:
                details.append(f"recorded files not found: {', '.join(absent)}")
            raise _error(location, "; ".join(details))


def validate_artifact_inventory(
    manifest: Mapping[str, Any],
    root: str | Path,
    *,
    profile: str | None = None,
    require_exact: bool = False,
    ignored_paths: Iterable[str] = (),
) -> None:
    """Validate an artifact manifest and its referenced local inventory."""

    validate_artifact_manifest(
        manifest,
        profile=profile,
        inventory_root=root,
        require_exact_inventory=require_exact,
        ignored_paths=ignored_paths,
    )


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Atomically replace ``path`` with deterministic, human-readable JSON."""

    _validate_json_value(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "LEGACY_PROFILE",
    "MUTABLE_REVISIONS",
    "ManifestValidationError",
    "SCHEMA_VERSION",
    "SELF_HASH_FIELD",
    "STRICT_PROFILE",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_manifest_bytes",
    "canonical_sha256",
    "file_sha256",
    "load_json_strict",
    "manifest_sha256",
    "seal_manifest",
    "validate_artifact_inventory",
    "validate_artifact_manifest",
    "validate_dataset_manifest",
    "validate_file_record",
    "validate_git_commit",
    "validate_immutable_revision",
    "validate_hub_commit",
    "validate_ordered_file_records",
    "validate_relative_path",
    "verify_file_inventory",
    "verify_manifest_hash",
    "write_json_atomic",
]
