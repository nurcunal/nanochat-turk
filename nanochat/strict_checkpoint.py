"""Transactional, integrity-checked checkpoints for strict WSD training.

This module is additive: the pinned upstream checkpoint manager remains the
legacy path, while production runs use only the transaction API defined here.
A checkpoint becomes visible only when rank 0 atomically publishes
``completion.json`` after every rank-local payload is durable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from nanochat.common import setup_default_logging
from nanochat.experiment_manifest import (
    canonical_json_bytes,
    seal_manifest,
    verify_manifest_hash,
)
from nanochat.gpt import GPT, GPTConfig
from nanochat.strict_tokenizer import (
    TokenizerPackageError,
    load_tokenizer_from_directory,
    verify_tokenizer_package,
)

setup_default_logging()
logger = logging.getLogger(__name__)


def _patch_missing_config_keys(model_config_kwargs: dict[str, Any]) -> None:
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"


def _patch_missing_keys(model_data: dict[str, Any], model_config: GPTConfig) -> None:
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(model_config.n_layer)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(model_config.n_layer)


# Strict, transactional checkpoints for integrity-critical runs

STRICT_CHECKPOINT_SCHEMA_VERSION = "1.0"
STRICT_CHECKPOINT_KIND = "nanochat_strict_checkpoint_completion"
STRICT_CHECKPOINT_PREFIX = "strict_"
STRICT_COMPLETION_FILE = "completion.json"
STRICT_RANK_ROLES = ("optimizer", "loader", "rng")
STRICT_HASH_IDENTITY_FIELDS = (
    "study_manifest_sha256",
    "run_sha256",
    "tokenizer_artifact_sha256",
    "exposure_plan_sha256",
)
_STRICT_DIRECTORY_RE = re.compile(r"^strict_(\d{6,})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CheckpointTransactionError(ValueError):
    """Base error for an unsafe or corrupt strict checkpoint transaction."""


class IncompleteCheckpointError(CheckpointTransactionError):
    """Raised when strict resume is requested before publication completed."""


class CheckpointIntegrityError(CheckpointTransactionError):
    """Raised when a completed strict checkpoint fails integrity validation."""


@dataclass(frozen=True)
class StrictCheckpointPayload:
    """Fully verified state returned by :func:`load_strict_checkpoint`."""

    step: int
    updates_completed: int
    model_data: Any
    optimizer_data: Any
    loader_state: Any
    rng_state: Any
    meta_data: dict[str, Any]
    manifest: dict[str, Any]


def capture_rank_rng_state(device: str | torch.device) -> dict[str, Any]:
    """Capture the Python, CPU, and rank-local CUDA stochastic streams."""

    resolved = torch.device(device)
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(resolved) if resolved.type == "cuda" else None
        ),
    }


def restore_rank_rng_state(
    state: Mapping[str, Any], device: str | torch.device
) -> None:
    """Restore a state returned by :func:`capture_rank_rng_state` exactly."""

    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "torch_cpu",
        "torch_cuda",
    }:
        raise CheckpointTransactionError("checkpoint rank RNG state is invalid")
    cpu_state = state["torch_cpu"]
    cuda_state = state["torch_cuda"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise CheckpointTransactionError("checkpoint CPU RNG state is invalid")
    try:
        random.setstate(state["python"])
        torch.set_rng_state(cpu_state.detach().cpu())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CheckpointTransactionError(
            "checkpoint rank RNG state is not restorable"
        ) from exc
    resolved = torch.device(device)
    if cuda_state is not None:
        if resolved.type != "cuda" or not isinstance(cuda_state, torch.Tensor):
            raise CheckpointTransactionError(
                "checkpoint CUDA RNG cannot load on this device"
            )
        try:
            torch.cuda.set_rng_state(cuda_state.detach().cpu(), resolved)
        except RuntimeError as exc:
            raise CheckpointTransactionError(
                "checkpoint CUDA RNG state is not restorable"
            ) from exc


def _require_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointTransactionError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CheckpointTransactionError(f"{name} must be a positive integer")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointTransactionError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalise_curve_log_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointTransactionError("identity.curve_log must be an object")
    state = dict(value)
    required = {
        "event_count",
        "last_event_sha256",
        "last_updates_completed",
        "file_sha256",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise CheckpointTransactionError(
            "identity.curve_log is missing: " + ", ".join(missing)
        )
    event_count = _require_nonnegative_integer(
        state["event_count"], "identity.curve_log.event_count"
    )
    last_updates = _require_nonnegative_integer(
        state["last_updates_completed"],
        "identity.curve_log.last_updates_completed",
    )
    last_hash = state["last_event_sha256"]
    if event_count == 0:
        if last_hash is not None:
            raise CheckpointTransactionError(
                "an empty curve log must have last_event_sha256=null"
            )
    else:
        _require_sha256(last_hash, "identity.curve_log.last_event_sha256")
    _require_sha256(state["file_sha256"], "identity.curve_log.file_sha256")
    if "recovered_truncated_bytes" in state:
        _require_nonnegative_integer(
            state["recovered_truncated_bytes"],
            "identity.curve_log.recovered_truncated_bytes",
        )
    # This also rejects non-finite numbers and non-JSON audit extensions.
    canonical_json_bytes(state)
    state["event_count"] = event_count
    state["last_updates_completed"] = last_updates
    return state


def _normalise_strict_identity(identity: Any) -> dict[str, Any]:
    """Validate identity material and add hashes for structured audit fields."""

    if not isinstance(identity, Mapping):
        raise CheckpointTransactionError("checkpoint identity must be an object")
    result = dict(identity)
    required = {
        "study_id",
        "run_id",
        *STRICT_HASH_IDENTITY_FIELDS,
        "optimizer_audit",
        "curve_log",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise CheckpointTransactionError(
            "checkpoint identity is missing: " + ", ".join(missing)
        )
    for name in ("study_id", "run_id"):
        if not isinstance(result[name], str) or not result[name]:
            raise CheckpointTransactionError(f"identity.{name} must be non-empty")
    for name in STRICT_HASH_IDENTITY_FIELDS:
        _require_sha256(result[name], f"identity.{name}")

    audit = result["optimizer_audit"]
    if not isinstance(audit, Mapping) or not audit:
        raise CheckpointTransactionError("identity.optimizer_audit must be an object")
    audit = dict(audit)
    canonical_json_bytes(audit)
    audit_hash = _json_sha256(audit)
    supplied_audit_hash = result.pop("optimizer_audit_sha256", None)
    if supplied_audit_hash is not None and not hmac.compare_digest(
        _require_sha256(
            supplied_audit_hash, "identity.optimizer_audit_sha256"
        ),
        audit_hash,
    ):
        raise CheckpointTransactionError("optimizer audit self-hash mismatch")
    result["optimizer_audit"] = audit
    result["optimizer_audit_sha256"] = audit_hash

    curve_log = _normalise_curve_log_state(result["curve_log"])
    curve_hash = _json_sha256(curve_log)
    supplied_curve_hash = result.pop("curve_log_state_sha256", None)
    if supplied_curve_hash is not None and not hmac.compare_digest(
        _require_sha256(
            supplied_curve_hash, "identity.curve_log_state_sha256"
        ),
        curve_hash,
    ):
        raise CheckpointTransactionError("curve-log state self-hash mismatch")
    result["curve_log"] = curve_log
    result["curve_log_state_sha256"] = curve_hash
    canonical_json_bytes(result)
    return result


def build_strict_checkpoint_identity(
    *,
    study_id: str,
    run_id: str,
    study_manifest_sha256: str,
    run_sha256: str,
    tokenizer_artifact_sha256: str,
    exposure_plan_sha256: str,
    optimizer_audit: Mapping[str, Any],
    curve_log_state: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable identity object consumed by strict save and load.

    ``curve_log_state`` includes both the hash-chain cursor and the SHA-256 of
    the complete JSONL file at the checkpoint boundary.  ``run_sha256`` is the
    canonical expanded-run hash, not a hash of the human-readable run ID.
    """

    identity = dict(extra or {})
    reserved = {
        "study_id",
        "run_id",
        *STRICT_HASH_IDENTITY_FIELDS,
        "optimizer_audit",
        "optimizer_audit_sha256",
        "curve_log",
        "curve_log_state_sha256",
    }
    collision = sorted(reserved & identity.keys())
    if collision:
        raise CheckpointTransactionError(
            "extra identity fields collide with reserved fields: "
            + ", ".join(collision)
        )
    identity.update(
        {
            "study_id": study_id,
            "run_id": run_id,
            "study_manifest_sha256": study_manifest_sha256,
            "run_sha256": run_sha256,
            "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
            "exposure_plan_sha256": exposure_plan_sha256,
            "optimizer_audit": dict(optimizer_audit),
            "curve_log": dict(curve_log_state),
        }
    )
    return _normalise_strict_identity(identity)


def strict_checkpoint_dir(checkpoint_dir: str | os.PathLike[str], step: int) -> Path:
    """Return the transaction directory for ``step`` after validating it."""

    _require_nonnegative_integer(step, "step")
    return Path(checkpoint_dir) / f"{STRICT_CHECKPOINT_PREFIX}{step:06d}"


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, role: str, rank: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }
    if rank is not None:
        record["rank"] = rank
    return record


def _rank_file_name(rank: int, role: str) -> str:
    if role not in STRICT_RANK_ROLES:
        raise CheckpointTransactionError(f"unsupported rank-state role: {role}")
    return f"rank_{rank:05d}_{role}.pt"


def _assert_not_published(step_dir: Path) -> None:
    if (step_dir / STRICT_COMPLETION_FILE).exists():
        raise CheckpointTransactionError(
            f"strict checkpoint is already complete and immutable: {step_dir}"
        )


def save_strict_rank_state(
    checkpoint_dir: str | os.PathLike[str],
    step: int,
    *,
    rank: int,
    expected_world_size: int,
    optimizer_data: Any,
    loader_state: Any,
    rng_state: Any,
) -> dict[str, Any]:
    """Atomically publish one rank's optimizer, loader, and RNG payloads.

    The returned, self-hashed record is the value ranks exchange (for example
    with ``all_gather_object``) before rank 0 finalizes the transaction.
    """

    world_size = _require_positive_integer(
        expected_world_size, "expected_world_size"
    )
    _require_nonnegative_integer(rank, "rank")
    if rank >= world_size:
        raise CheckpointTransactionError("rank must be smaller than world size")
    step_dir = strict_checkpoint_dir(checkpoint_dir, step)
    step_dir.mkdir(parents=True, exist_ok=True)
    _assert_not_published(step_dir)
    values = {
        "optimizer": optimizer_data,
        "loader": loader_state,
        "rng": rng_state,
    }
    files: list[dict[str, Any]] = []
    for role in STRICT_RANK_ROLES:
        destination = step_dir / _rank_file_name(rank, role)
        _atomic_torch_save(destination, values[role])
        files.append(_file_record(destination, role=role, rank=rank))
    record = seal_manifest(
        {
            "rank": rank,
            "expected_world_size": world_size,
            "files": files,
            "record_sha256": None,
        },
        self_hash_field="record_sha256",
    )
    return record


def _safe_payload_path(step_dir: Path, value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CheckpointIntegrityError(f"{location}.path must be non-empty")
    posix = PurePosixPath(value)
    if (
        posix.is_absolute()
        or len(posix.parts) != 1
        or value in {".", ".."}
        or "\\" in value
    ):
        raise CheckpointIntegrityError(f"{location}.path is unsafe")
    return step_dir / value


def _validate_file_record(
    record: Any,
    *,
    step_dir: Path,
    location: str,
    expected_role: str,
    expected_name: str,
    expected_rank: int | None,
    verify_bytes: bool,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise CheckpointIntegrityError(f"{location} must be an object")
    materialized = dict(record)
    expected_keys = {"role", "path", "size_bytes", "sha256"}
    if expected_rank is not None:
        expected_keys.add("rank")
    if set(materialized) != expected_keys:
        raise CheckpointIntegrityError(f"{location} has unexpected fields")
    if materialized["role"] != expected_role:
        raise CheckpointIntegrityError(f"{location} role mismatch")
    if materialized["path"] != expected_name:
        raise CheckpointIntegrityError(f"{location} path mismatch")
    path = _safe_payload_path(step_dir, materialized["path"], location)
    size = _require_nonnegative_integer(materialized["size_bytes"], f"{location}.size_bytes")
    digest = _require_sha256(materialized["sha256"], f"{location}.sha256")
    if expected_rank is not None:
        if materialized["rank"] != expected_rank:
            raise CheckpointIntegrityError(f"{location} rank mismatch")
    if verify_bytes:
        try:
            actual_size = path.stat().st_size
        except FileNotFoundError as exc:
            raise CheckpointIntegrityError(f"recorded file is missing: {path}") from exc
        if actual_size != size:
            raise CheckpointIntegrityError(f"recorded file size mismatch: {path}")
        actual_digest = _file_sha256(path)
        if not hmac.compare_digest(actual_digest, digest):
            raise CheckpointIntegrityError(f"recorded file hash mismatch: {path}")
    return materialized


def _validate_rank_record(
    value: Any,
    *,
    step_dir: Path,
    expected_rank: int,
    expected_world_size: int,
    verify_bytes: bool,
) -> dict[str, Any]:
    location = f"ranks[{expected_rank}]"
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError(f"{location} must be an object")
    record = dict(value)
    if set(record) != {
        "rank",
        "expected_world_size",
        "files",
        "record_sha256",
    }:
        raise CheckpointIntegrityError(f"{location} has unexpected fields")
    try:
        verify_manifest_hash(record, self_hash_field="record_sha256")
    except ValueError as exc:
        raise CheckpointIntegrityError(f"{location} self-hash mismatch") from exc
    if record["rank"] != expected_rank:
        raise CheckpointIntegrityError(f"{location} rank mismatch")
    if record["expected_world_size"] != expected_world_size:
        raise CheckpointIntegrityError(f"{location} world-size mismatch")
    files = record["files"]
    if not isinstance(files, list) or len(files) != len(STRICT_RANK_ROLES):
        raise CheckpointIntegrityError(f"{location}.files must contain every rank role")
    validated_files = []
    for index, role in enumerate(STRICT_RANK_ROLES):
        validated_files.append(
            _validate_file_record(
                files[index],
                step_dir=step_dir,
                location=f"{location}.files[{index}]",
                expected_role=role,
                expected_name=_rank_file_name(expected_rank, role),
                expected_rank=expected_rank,
                verify_bytes=verify_bytes,
            )
        )
    return {**record, "files": validated_files}


def _materialise_rank_records(
    rank_records: Any,
    *,
    step_dir: Path,
    expected_world_size: int,
    verify_bytes: bool,
) -> list[dict[str, Any]]:
    if (
        not isinstance(rank_records, Sequence)
        or isinstance(rank_records, (str, bytes, bytearray))
        or len(rank_records) != expected_world_size
    ):
        raise CheckpointIntegrityError(
            "rank records must contain exactly expected_world_size entries"
        )
    by_rank: dict[int, Any] = {}
    for value in rank_records:
        if not isinstance(value, Mapping):
            raise CheckpointIntegrityError("rank record must be an object")
        rank = value.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise CheckpointIntegrityError("rank record has an invalid rank")
        if rank in by_rank:
            raise CheckpointIntegrityError(f"duplicate rank record: {rank}")
        by_rank[rank] = value
    if set(by_rank) != set(range(expected_world_size)):
        raise CheckpointIntegrityError(
            "rank records do not exactly cover the expected ranks"
        )
    return [
        _validate_rank_record(
            by_rank[rank],
            step_dir=step_dir,
            expected_rank=rank,
            expected_world_size=expected_world_size,
            verify_bytes=verify_bytes,
        )
        for rank in range(expected_world_size)
    ]


def _strict_payload_inventory(step_dir: Path) -> set[str]:
    try:
        children = list(step_dir.iterdir())
    except FileNotFoundError:
        return set()
    if any(not child.is_file() for child in children):
        raise CheckpointIntegrityError("strict checkpoint contains a nested directory")
    return {child.name for child in children}


def finalize_strict_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    step: int,
    model_data: Any,
    meta_data: Mapping[str, Any],
    *,
    rank_records: Sequence[Mapping[str, Any]],
    expected_world_size: int,
    identity: Mapping[str, Any],
    updates_completed: int,
    rank: int = 0,
    barrier: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Have rank 0 publish model, metadata, then the completion manifest.

    The caller supplies records gathered from every rank.  An injected barrier
    may additionally be supplied for local orchestration/tests; it is invoked
    before any rank record is accepted or any rank-0 payload is written.
    """

    if rank != 0:
        raise CheckpointTransactionError("only rank 0 may finalize a checkpoint")
    world_size = _require_positive_integer(
        expected_world_size, "expected_world_size"
    )
    updates = _require_nonnegative_integer(updates_completed, "updates_completed")
    normal_identity = _normalise_strict_identity(identity)
    if normal_identity["curve_log"]["last_updates_completed"] != updates:
        raise CheckpointTransactionError(
            "curve-log updates_completed does not match the checkpoint boundary"
        )
    if not isinstance(meta_data, Mapping):
        raise CheckpointTransactionError("meta_data must be a JSON object")
    meta = dict(meta_data)
    canonical_json_bytes(meta)
    step_dir = strict_checkpoint_dir(checkpoint_dir, step)
    step_dir.mkdir(parents=True, exist_ok=True)
    _assert_not_published(step_dir)
    if barrier is not None:
        barrier()

    ranks = _materialise_rank_records(
        rank_records,
        step_dir=step_dir,
        expected_world_size=world_size,
        verify_bytes=True,
    )
    model_path = step_dir / "model.pt"
    meta_path = step_dir / "meta.json"
    _atomic_torch_save(model_path, model_data)
    _atomic_write_bytes(meta_path, canonical_json_bytes(meta))
    model_record = _file_record(model_path, role="model")
    meta_record = _file_record(meta_path, role="meta")
    files = [model_record, meta_record]
    for record in ranks:
        files.extend(record["files"])

    expected_payload_names = {record["path"] for record in files}
    actual_payload_names = _strict_payload_inventory(step_dir)
    if actual_payload_names != expected_payload_names:
        unexpected = sorted(actual_payload_names - expected_payload_names)
        missing = sorted(expected_payload_names - actual_payload_names)
        raise CheckpointIntegrityError(
            "strict checkpoint payload inventory mismatch; "
            f"unexpected={unexpected}, missing={missing}"
        )
    # Re-verify rank payloads immediately before publication.  A completion
    # manifest must never bless bytes different from the gathered records.
    _materialise_rank_records(
        ranks,
        step_dir=step_dir,
        expected_world_size=world_size,
        verify_bytes=True,
    )
    manifest = seal_manifest(
        {
            "schema_version": STRICT_CHECKPOINT_SCHEMA_VERSION,
            "kind": STRICT_CHECKPOINT_KIND,
            "step": step,
            "updates_completed": updates,
            "expected_world_size": world_size,
            "identity": normal_identity,
            "ranks": ranks,
            "files": files,
            "canonical_sha256": None,
        }
    )
    completion_path = step_dir / STRICT_COMPLETION_FILE
    _assert_not_published(step_dir)
    _atomic_write_bytes(completion_path, canonical_json_bytes(manifest))
    logger.info("Published strict checkpoint transaction: %s", completion_path)
    return manifest


def save_strict_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    step: int,
    model_data: Any,
    optimizer_data: Any,
    meta_data: Mapping[str, Any],
    *,
    loader_state: Any,
    rng_state: Any,
    rank: int,
    expected_world_size: int,
    identity: Mapping[str, Any],
    updates_completed: int,
    gather_rank_records: Callable[[dict[str, Any]], Sequence[Mapping[str, Any]]]
    | None = None,
    barrier: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for a full transaction.

    All ranks call this function. ``gather_rank_records`` is an injected
    collective that returns records for every rank (an ``all_gather_object``
    wrapper in DDP).  The optional barrier is called by every rank both before
    rank 0 starts publication and after ``completion.json`` is durable.  The
    post-publication barrier prevents nonzero ranks from entering the next
    optimizer collective while rank 0 is still serializing the model.
    Nonzero ranks return their local rank record; rank 0 returns the completed
    manifest.  For a one-rank run, the gather callback is optional.
    """

    local_record = save_strict_rank_state(
        checkpoint_dir,
        step,
        rank=rank,
        expected_world_size=expected_world_size,
        optimizer_data=optimizer_data,
        loader_state=loader_state,
        rng_state=rng_state,
    )
    if gather_rank_records is None:
        if expected_world_size != 1:
            raise CheckpointTransactionError(
                "multi-rank strict save requires gather_rank_records"
            )
        records: Sequence[Mapping[str, Any]] = [local_record]
    else:
        records = gather_rank_records(local_record)
    if barrier is not None:
        barrier()
    result: dict[str, Any]
    if rank == 0:
        result = finalize_strict_checkpoint(
            checkpoint_dir,
            step,
            model_data,
            meta_data,
            rank_records=records,
            expected_world_size=expected_world_size,
            identity=identity,
            updates_completed=updates_completed,
            rank=0,
        )
    else:
        result = local_record
    if barrier is not None:
        barrier()
    return result


def _strict_json_object(payload: bytes, *, location: str) -> dict[str, Any]:
    def reject_duplicate_pairs(pairs):
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CheckpointIntegrityError(
                    f"{location}: duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value):
        raise CheckpointIntegrityError(
            f"{location}: non-finite JSON number {value!r}"
        )

    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"{location}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"{location}: expected a JSON object")
    return value


def _read_completion_manifest(step_dir: Path) -> dict[str, Any]:
    path = step_dir / STRICT_COMPLETION_FILE
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise IncompleteCheckpointError(
            f"strict checkpoint has no completion manifest: {step_dir}"
        ) from exc
    return _strict_json_object(payload, location=str(path))


def _validate_completion_structure(
    manifest: Any,
    *,
    step_dir: Path,
    expected_step: int,
    verify_bytes: bool,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise CheckpointIntegrityError("completion manifest must be an object")
    value = dict(manifest)
    if set(value) != {
        "schema_version",
        "kind",
        "step",
        "updates_completed",
        "expected_world_size",
        "identity",
        "ranks",
        "files",
        "canonical_sha256",
    }:
        raise CheckpointIntegrityError("completion manifest has unexpected fields")
    try:
        verify_manifest_hash(value)
    except ValueError as exc:
        raise CheckpointIntegrityError("completion manifest self-hash mismatch") from exc
    if value["schema_version"] != STRICT_CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointIntegrityError("completion schema version mismatch")
    if value["kind"] != STRICT_CHECKPOINT_KIND:
        raise CheckpointIntegrityError("completion kind mismatch")
    if value["step"] != expected_step:
        raise CheckpointIntegrityError("checkpoint step mismatch")
    _require_nonnegative_integer(value["updates_completed"], "updates_completed")
    world_size = _require_positive_integer(
        value["expected_world_size"], "expected_world_size"
    )
    identity = _normalise_strict_identity(value["identity"])
    if identity != value["identity"]:
        raise CheckpointIntegrityError("checkpoint identity is not canonical")
    if identity["curve_log"]["last_updates_completed"] != value["updates_completed"]:
        raise CheckpointIntegrityError(
            "curve-log checkpoint boundary does not match updates_completed"
        )
    ranks = _materialise_rank_records(
        value["ranks"],
        step_dir=step_dir,
        expected_world_size=world_size,
        verify_bytes=verify_bytes,
    )
    files = value["files"]
    expected_file_count = 2 + world_size * len(STRICT_RANK_ROLES)
    if not isinstance(files, list) or len(files) != expected_file_count:
        raise CheckpointIntegrityError("completion file inventory has the wrong size")
    validated_files = [
        _validate_file_record(
            files[0],
            step_dir=step_dir,
            location="files[0]",
            expected_role="model",
            expected_name="model.pt",
            expected_rank=None,
            verify_bytes=verify_bytes,
        ),
        _validate_file_record(
            files[1],
            step_dir=step_dir,
            location="files[1]",
            expected_role="meta",
            expected_name="meta.json",
            expected_rank=None,
            verify_bytes=verify_bytes,
        ),
    ]
    offset = 2
    for rank_record in ranks:
        for role_index, role in enumerate(STRICT_RANK_ROLES):
            record = _validate_file_record(
                files[offset],
                step_dir=step_dir,
                location=f"files[{offset}]",
                expected_role=role,
                expected_name=_rank_file_name(rank_record["rank"], role),
                expected_rank=rank_record["rank"],
                verify_bytes=verify_bytes,
            )
            if record != rank_record["files"][role_index]:
                raise CheckpointIntegrityError(
                    "flattened file inventory disagrees with rank record"
                )
            validated_files.append(record)
            offset += 1
    if verify_bytes:
        expected_names = {STRICT_COMPLETION_FILE}
        expected_names.update(record["path"] for record in validated_files)
        actual_names = _strict_payload_inventory(step_dir)
        if actual_names != expected_names:
            raise CheckpointIntegrityError(
                "completed strict checkpoint contains unrecorded or missing files"
            )
    return {
        **value,
        "identity": identity,
        "ranks": ranks,
        "files": validated_files,
    }


def verify_strict_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    step: int,
    *,
    expected_world_size: int,
    rank: int,
    expected_identity: Mapping[str, Any],
    expected_updates_completed: int | None = None,
) -> dict[str, Any]:
    """Verify a complete transaction without deserializing any pickle data."""

    manifest = inspect_strict_checkpoint(checkpoint_dir, step)
    return _verify_strict_manifest_expectations(
        manifest,
        expected_world_size=expected_world_size,
        rank=rank,
        expected_identity=expected_identity,
        expected_updates_completed=expected_updates_completed,
    )


def _verify_strict_manifest_expectations(
    manifest: Mapping[str, Any],
    *,
    expected_world_size: int,
    rank: int,
    expected_identity: Mapping[str, Any],
    expected_updates_completed: int | None,
) -> dict[str, Any]:
    """Apply caller-owned resume expectations to an inspected manifest."""

    manifest = dict(manifest)
    world_size = _require_positive_integer(
        expected_world_size, "expected_world_size"
    )
    if manifest["expected_world_size"] != world_size:
        raise CheckpointIntegrityError("resume world size does not match checkpoint")
    _require_nonnegative_integer(rank, "rank")
    if rank >= world_size:
        raise CheckpointIntegrityError("resume rank is outside the expected world")
    identity = _normalise_strict_identity(expected_identity)
    if manifest["identity"] != identity:
        raise CheckpointIntegrityError("resume identity does not match checkpoint")
    if expected_updates_completed is not None:
        updates = _require_nonnegative_integer(
            expected_updates_completed, "expected_updates_completed"
        )
        if manifest["updates_completed"] != updates:
            raise CheckpointIntegrityError(
                "resume update boundary does not match checkpoint"
            )
    return manifest


def inspect_strict_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    step: int,
) -> dict[str, Any]:
    """Return a fully integrity-checked manifest without deserializing state.

    This deliberately does not accept or assert an external identity.  Resume
    orchestration uses it to read the checkpointed curve-log cursor, reconcile
    a durable JSONL tail, and independently rebuild the expected static
    identity before calling :func:`load_strict_checkpoint`.  Completion
    self-hash, schema, rank coverage, exact inventory, and every payload
    size/SHA-256 are nevertheless verified here; ``torch.load`` is never used.
    """

    step_dir = strict_checkpoint_dir(checkpoint_dir, step)
    return _validate_completion_structure(
        _read_completion_manifest(step_dir),
        step_dir=step_dir,
        expected_step=step,
        verify_bytes=True,
    )


def verify_strict_checkpoint_for_evaluation(
    checkpoint_dir: str | os.PathLike[str],
    step: int,
    *,
    expected_study_id: str,
    expected_run_id: str,
    expected_study_manifest_sha256: str,
    expected_run_sha256: str,
    expected_tokenizer_artifact_sha256: str,
    expected_exposure_plan_sha256: str,
    expected_updates_completed: int,
) -> dict[str, Any]:
    """Verify a strict checkpoint against an evaluator-owned identity.

    Evaluation must not infer its comparison identity from the checkpoint it is
    about to score.  This entry point therefore accepts the independently
    supplied study, run ID, expanded-run hash, artifact, exposure, and
    update-boundary values one by one.  It first performs
    :func:`inspect_strict_checkpoint`, which checks the
    completion self-hash, exact inventory, rank coverage, and every payload
    byte.  No pickle is deserialized by this function.
    """

    manifest = inspect_strict_checkpoint(checkpoint_dir, step)
    if not isinstance(expected_study_id, str) or not expected_study_id:
        raise CheckpointTransactionError("expected_study_id must be non-empty")
    if not isinstance(expected_run_id, str) or not expected_run_id:
        raise CheckpointTransactionError("expected_run_id must be non-empty")
    expected_hashes = {
        "study_manifest_sha256": _require_sha256(
            expected_study_manifest_sha256,
            "expected_study_manifest_sha256",
        ),
        "run_sha256": _require_sha256(
            expected_run_sha256,
            "expected_run_sha256",
        ),
        "tokenizer_artifact_sha256": _require_sha256(
            expected_tokenizer_artifact_sha256,
            "expected_tokenizer_artifact_sha256",
        ),
        "exposure_plan_sha256": _require_sha256(
            expected_exposure_plan_sha256,
            "expected_exposure_plan_sha256",
        ),
    }
    expected_updates = _require_nonnegative_integer(
        expected_updates_completed,
        "expected_updates_completed",
    )
    identity = manifest["identity"]
    if identity["study_id"] != expected_study_id:
        raise CheckpointIntegrityError(
            "evaluation study_id does not match checkpoint"
        )
    if identity["run_id"] != expected_run_id:
        raise CheckpointIntegrityError("evaluation run_id does not match checkpoint")
    for field, expected in expected_hashes.items():
        if not hmac.compare_digest(identity[field], expected):
            raise CheckpointIntegrityError(
                f"evaluation {field} does not match checkpoint"
            )
    if manifest["updates_completed"] != expected_updates:
        raise CheckpointIntegrityError(
            "evaluation update boundary does not match checkpoint"
        )
    return manifest


def _torch_load_model_weights_after_verification(path: Path, device: Any) -> Any:
    """Deserialize a model state dict, using the restricted loader when present."""

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location=device)


def build_strict_eval_model(
    checkpoint_dir: str | os.PathLike[str],
    step: int,
    device: Any,
    *,
    expected_study_id: str,
    expected_run_id: str,
    expected_study_manifest_sha256: str,
    expected_run_sha256: str,
    expected_tokenizer_artifact_sha256: str,
    expected_tokenizer_manifest_path: str | os.PathLike[str],
    expected_exposure_plan_sha256: str,
    expected_updates_completed: int,
):
    """Build an evaluation-only GPT/tokenizer from a strict transaction.

    All transaction bytes and evaluator-owned identity fields are checked
    before ``model.pt`` is deserialized.  ``model.pt`` is the only strict
    checkpoint pickle this path ever opens; optimizer, loader, and RNG rank
    payloads are integrity-checked as bytes but never deserialized.
    """

    device = torch.device(device)
    manifest = verify_strict_checkpoint_for_evaluation(
        checkpoint_dir,
        step,
        expected_study_id=expected_study_id,
        expected_run_id=expected_run_id,
        expected_study_manifest_sha256=expected_study_manifest_sha256,
        expected_run_sha256=expected_run_sha256,
        expected_tokenizer_artifact_sha256=expected_tokenizer_artifact_sha256,
        expected_exposure_plan_sha256=expected_exposure_plan_sha256,
        expected_updates_completed=expected_updates_completed,
    )
    step_dir = strict_checkpoint_dir(checkpoint_dir, step)
    meta_path = step_dir / "meta.json"
    meta_data = _strict_json_object(meta_path.read_bytes(), location=str(meta_path))

    # These cross-checks also precede torch.load.  Metadata is part of the
    # completion inventory, but it must still describe the selected boundary.
    if meta_data.get("step") != step:
        raise CheckpointIntegrityError("strict evaluation metadata step mismatch")
    if meta_data.get("updates_completed") != manifest["updates_completed"]:
        raise CheckpointIntegrityError(
            "strict evaluation metadata update boundary mismatch"
        )
    model_config_value = meta_data.get("model_config")
    if not isinstance(model_config_value, Mapping):
        raise CheckpointIntegrityError(
            "strict evaluation metadata has no model_config object"
        )
    model_config_kwargs = dict(model_config_value)
    canonical_json_bytes(model_config_kwargs)
    model_vocab_size = model_config_kwargs.get("vocab_size")
    if (
        isinstance(model_vocab_size, bool)
        or not isinstance(model_vocab_size, int)
        or model_vocab_size <= 0
    ):
        raise CheckpointIntegrityError(
            "strict evaluation model_config has no positive vocab_size"
        )
    tokenizer_name = meta_data.get("tokenizer_name")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise CheckpointIntegrityError(
            "strict evaluation metadata has no tokenizer_name"
        )
    protocol = manifest["identity"].get("protocol")
    if isinstance(protocol, Mapping) and "model_config" in protocol:
        if protocol["model_config"] != model_config_kwargs:
            raise CheckpointIntegrityError(
                "strict identity and metadata model configs differ"
            )
    tokenizer_config = meta_data.get("tokenizer_config")
    if not isinstance(tokenizer_config, Mapping):
        raise CheckpointIntegrityError(
            "strict evaluation metadata has no tokenizer_config object"
        )
    try:
        verified_tokenizer = verify_tokenizer_package(
            expected_tokenizer_manifest_path,
            expected_sha256=expected_tokenizer_artifact_sha256,
            expected_name=tokenizer_name,
            expected_vocab_size=model_vocab_size,
        )
    except (TokenizerPackageError, TypeError, ValueError) as exc:
        raise CheckpointIntegrityError(
            f"strict tokenizer package verification failed: {exc}"
        ) from exc
    if dict(tokenizer_config) != verified_tokenizer.config:
        raise CheckpointIntegrityError(
            "strict metadata tokenizer config differs from verified package"
        )

    model_data = _torch_load_model_weights_after_verification(
        step_dir / "model.pt", device
    )
    if not isinstance(model_data, Mapping):
        raise CheckpointIntegrityError("strict model payload is not a state dict")
    if getattr(device, "type", None) in {"cpu", "mps"}:
        model_data = {
            key: value.float()
            if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16
            else value
            for key, value in model_data.items()
        }
    model_data = {
        key.removeprefix("_orig_mod."): value for key, value in model_data.items()
    }
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # The verified manifest's root is the runtime source. Strict evaluation
    # never falls back to an environment-selected tokenizer directory.
    tokenizer = load_tokenizer_from_directory(verified_tokenizer.root)
    if tokenizer.get_vocab_size() != model_config_kwargs["vocab_size"]:
        raise CheckpointIntegrityError(
            "strict checkpoint tokenizer vocabulary does not match model config"
        )
    return model, tokenizer, meta_data


def _torch_load_after_verification(path: Path, device: Any) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.load(path, map_location=device)


def load_strict_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    step: int,
    device: Any,
    *,
    rank: int,
    expected_world_size: int,
    expected_identity: Mapping[str, Any],
    expected_updates_completed: int | None = None,
    verified_manifest: Mapping[str, Any] | None = None,
    load_model: bool = True,
) -> StrictCheckpointPayload:
    """Verify all files and identities, then deserialize this rank's state.

    No call to ``torch.load`` occurs until the completion self-hash, full file
    inventory, every size/SHA-256, world size, rank, and expected identity have
    all been checked. ``verified_manifest`` is the already byte-verified rank-0
    manifest broadcast to peer ranks; peers still validate its canonical
    structure and all caller-owned expectations.  ``load_model=False`` avoids
    opening the shared model payload on ranks that receive parameters by
    broadcast.
    """

    step_dir = strict_checkpoint_dir(checkpoint_dir, step)
    if verified_manifest is None:
        manifest = verify_strict_checkpoint(
            checkpoint_dir,
            step,
            expected_world_size=expected_world_size,
            rank=rank,
            expected_identity=expected_identity,
            expected_updates_completed=expected_updates_completed,
        )
    else:
        # Rank 0 may inspect and hash a shared-filesystem checkpoint once, then
        # broadcast that verified manifest.  Revalidate its canonical structure
        # locally without rereading every multi-gigabyte payload.
        manifest = _validate_completion_structure(
            dict(verified_manifest),
            step_dir=step_dir,
            expected_step=step,
            verify_bytes=False,
        )
        manifest = _verify_strict_manifest_expectations(
            manifest,
            expected_world_size=expected_world_size,
            rank=rank,
            expected_identity=expected_identity,
            expected_updates_completed=expected_updates_completed,
        )
    model_data = (
        _torch_load_model_weights_after_verification(
            step_dir / "model.pt", device
        )
        if load_model
        else None
    )
    optimizer_data = _torch_load_after_verification(
        step_dir / _rank_file_name(rank, "optimizer"), device
    )
    loader_state = _torch_load_after_verification(
        step_dir / _rank_file_name(rank, "loader"), device
    )
    rng_state = _torch_load_after_verification(
        step_dir / _rank_file_name(rank, "rng"), device
    )
    meta_path = step_dir / "meta.json"
    meta_data = _strict_json_object(meta_path.read_bytes(), location=str(meta_path))
    return StrictCheckpointPayload(
        step=step,
        updates_completed=manifest["updates_completed"],
        model_data=model_data,
        optimizer_data=optimizer_data,
        loader_state=loader_state,
        rng_state=rng_state,
        meta_data=meta_data,
        manifest=manifest,
    )


def find_latest_complete_strict_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    *,
    expected_world_size: int,
    rank: int,
    expected_identity: Mapping[str, Any],
) -> int:
    """Return the latest verified strict step, ignoring torn transactions.

    A directory without ``completion.json`` is an expected interrupted save and
    is skipped.  Once a completion manifest exists, corruption is an error and
    is never silently converted into rollback to an older checkpoint.
    """

    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"No strict checkpoints found in {root}")
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _STRICT_DIRECTORY_RE.fullmatch(child.name)
        if match is not None:
            candidates.append((int(match.group(1)), child))
    for step, step_dir in sorted(candidates, reverse=True):
        if not (step_dir / STRICT_COMPLETION_FILE).is_file():
            continue
        verify_strict_checkpoint(
            root,
            step,
            expected_world_size=expected_world_size,
            rank=rank,
            expected_identity=expected_identity,
        )
        return step
    raise FileNotFoundError(f"No complete strict checkpoints found in {root}")



__all__ = [
    "CheckpointIntegrityError",
    "CheckpointTransactionError",
    "IncompleteCheckpointError",
    "STRICT_CHECKPOINT_KIND",
    "STRICT_COMPLETION_FILE",
    "StrictCheckpointPayload",
    "build_strict_checkpoint_identity",
    "build_strict_eval_model",
    "capture_rank_rng_state",
    "finalize_strict_checkpoint",
    "find_latest_complete_strict_checkpoint",
    "inspect_strict_checkpoint",
    "load_strict_checkpoint",
    "restore_rank_rng_state",
    "save_strict_checkpoint",
    "save_strict_rank_state",
    "strict_checkpoint_dir",
    "verify_strict_checkpoint",
    "verify_strict_checkpoint_for_evaluation",
]
