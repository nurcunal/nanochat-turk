"""Durable canonical JSONL curves with hash chaining and safe resume."""

from __future__ import annotations

import json
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nanochat.experiment_manifest import canonical_json_bytes, seal_manifest, verify_manifest_hash


class TrainingLogError(ValueError):
    """Raised when a curve log is corrupt, ambiguous, or identity-mismatched."""


def _strict_json(line: str, *, location: str) -> dict[str, Any]:
    def duplicate_pairs(pairs):
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TrainingLogError(f"{location}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise TrainingLogError(f"{location}: non-finite JSON number {value!r}")

    try:
        value = json.loads(
            line,
            object_pairs_hook=duplicate_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise TrainingLogError(f"{location}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingLogError(f"{location}: event must be a JSON object")
    return value


@dataclass(frozen=True)
class TrainingLogState:
    event_count: int
    last_event_sha256: str | None
    last_updates_completed: int
    recovered_truncated_bytes: int = 0


def read_training_log(
    path: str | Path,
    *,
    expected_study_id: str | None = None,
    expected_run_id: str | None = None,
    recover_incomplete_tail: bool = False,
) -> tuple[list[dict[str, Any]], TrainingLogState]:
    """Read and verify a full hash chain; optionally remove only a torn tail."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        return [], TrainingLogState(0, None, 0)
    recovered = 0
    if payload and not payload.endswith(b"\n"):
        if not recover_incomplete_tail:
            raise TrainingLogError("training log has an incomplete final line")
        cutoff = payload.rfind(b"\n") + 1
        recovered = len(payload) - cutoff
        payload = payload[:cutoff]
        with source.open("r+b") as handle:
            handle.truncate(cutoff)
            handle.flush()
            os.fsync(handle.fileno())

    events: list[dict[str, Any]] = []
    previous: str | None = None
    last_updates = 0
    for index, raw_line in enumerate(payload.splitlines(), start=1):
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TrainingLogError(f"line {index}: invalid UTF-8") from exc
        event = _strict_json(line, location=f"line {index}")
        try:
            verify_manifest_hash(event, self_hash_field="event_sha256")
        except ValueError as exc:
            raise TrainingLogError(f"line {index}: event hash mismatch") from exc
        if event.get("schema_version") != "1.0" or event.get("kind") != "training_curve_event":
            raise TrainingLogError(f"line {index}: unexpected event schema/kind")
        if event.get("event_index") != index - 1:
            raise TrainingLogError(f"line {index}: event index is not contiguous")
        if event.get("previous_event_sha256") != previous:
            raise TrainingLogError(f"line {index}: event hash chain is broken")
        study_id = event.get("study_id")
        run_id = event.get("run_id")
        if not isinstance(study_id, str) or not study_id:
            raise TrainingLogError(f"line {index}: study_id is invalid")
        if not isinstance(run_id, str) or not run_id:
            raise TrainingLogError(f"line {index}: run_id is invalid")
        if expected_study_id is not None and study_id != expected_study_id:
            raise TrainingLogError("training log study identity mismatch")
        if expected_run_id is not None and run_id != expected_run_id:
            raise TrainingLogError("training log run identity mismatch")
        updates = event.get("updates_completed")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise TrainingLogError(f"line {index}: updates_completed is invalid")
        if updates < last_updates:
            raise TrainingLogError(f"line {index}: updates_completed moved backwards")
        if event.get("event_type") == "train_update" and events and updates <= last_updates:
            raise TrainingLogError(f"line {index}: train update is not strictly increasing")
        metrics = event.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TrainingLogError(f"line {index}: metrics must be an object")
        for key, value in metrics.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(
                value, (int, float)
            ) or not math.isfinite(float(value)):
                raise TrainingLogError(f"line {index}: metric {key!r} is not finite numeric")
        previous = event["event_sha256"]
        last_updates = updates
        events.append(event)
    return events, TrainingLogState(len(events), previous, last_updates, recovered)


class CanonicalTrainingLog:
    """Append-only fsynced curve writer with collision and resume checks."""

    def __init__(
        self,
        path: str | Path,
        *,
        study_id: str,
        run_id: str,
        resume: bool = False,
    ) -> None:
        if not study_id or not run_id:
            raise TrainingLogError("study_id and run_id must be non-empty")
        self.path = Path(path)
        self.study_id = study_id
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            _events, self.state = read_training_log(
                self.path,
                expected_study_id=study_id,
                expected_run_id=run_id,
                recover_incomplete_tail=True,
            )
        else:
            try:
                with self.path.open("x", encoding="utf-8"):
                    pass
            except FileExistsError as exc:
                raise TrainingLogError(
                    f"refusing to overwrite existing curve log: {self.path}"
                ) from exc
            self.state = TrainingLogState(0, None, 0)

    def append(
        self,
        *,
        event_type: str,
        updates_completed: int,
        metrics: Mapping[str, int | float],
        identities: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(event_type, str) or not event_type:
            raise TrainingLogError("event_type must be non-empty")
        if (
            isinstance(updates_completed, bool)
            or not isinstance(updates_completed, int)
            or updates_completed < self.state.last_updates_completed
        ):
            raise TrainingLogError("updates_completed is invalid or moved backwards")
        if event_type == "train_update" and updates_completed <= self.state.last_updates_completed:
            raise TrainingLogError("train_update must advance updates_completed")
        clean_metrics: dict[str, int | float] = {}
        for key, value in metrics.items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TrainingLogError(f"metric {key!r} must be finite numeric")
            clean_metrics[key] = value
        event = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": "training_curve_event",
                "event_index": self.state.event_count,
                "event_type": event_type,
                "study_id": self.study_id,
                "run_id": self.run_id,
                "updates_completed": updates_completed,
                "previous_event_sha256": self.state.last_event_sha256,
                "identities": dict(identities or {}),
                "metrics": clean_metrics,
                "event_sha256": None,
            },
            self_hash_field="event_sha256",
        )
        encoded = canonical_json_bytes(event)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            with os.fdopen(fd, "ab", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        self.state = TrainingLogState(
            event_count=self.state.event_count + 1,
            last_event_sha256=event["event_sha256"],
            last_updates_completed=updates_completed,
            recovered_truncated_bytes=self.state.recovered_truncated_bytes,
        )
        return event


def checkpoint_curve_log_state(
    path: str | Path, state: TrainingLogState
) -> dict[str, Any]:
    """Return the exact fsynced curve cursor bound into a checkpoint."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise TrainingLogError(f"cannot read curve log for checkpoint: {exc}") from exc
    events, verified = read_training_log(source)
    del events
    if (
        verified.event_count != state.event_count
        or verified.last_event_sha256 != state.last_event_sha256
        or verified.last_updates_completed != state.last_updates_completed
    ):
        raise TrainingLogError("in-memory curve cursor differs from the fsynced log")
    return {
        "event_count": state.event_count,
        "last_event_sha256": state.last_event_sha256,
        "last_updates_completed": state.last_updates_completed,
        "recovered_truncated_bytes": state.recovered_truncated_bytes,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def reconcile_training_log_to_checkpoint(
    path: str | Path,
    checkpoint_state: Mapping[str, Any],
    *,
    expected_study_id: str,
    expected_run_id: str,
) -> TrainingLogState:
    """Verify a checkpointed log prefix and discard only later tail bytes.

    The checkpoint stores the exact event count, terminal event hash, update
    counter, and full-prefix file SHA-256. Those values define the only safe
    truncation boundary; malformed or valid-but-uncheckpointed tail events are
    both removed. No byte inside the checkpointed prefix is repaired.
    """

    required = {
        "event_count",
        "last_event_sha256",
        "last_updates_completed",
        "file_sha256",
    }
    if not isinstance(checkpoint_state, Mapping) or not required.issubset(
        checkpoint_state
    ):
        raise TrainingLogError("checkpoint curve-log state is incomplete")
    event_count = checkpoint_state["event_count"]
    updates = checkpoint_state["last_updates_completed"]
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 0
        or isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates < 0
    ):
        raise TrainingLogError("checkpoint curve-log counters are invalid")
    file_hash = checkpoint_state["file_sha256"]
    if (
        not isinstance(file_hash, str)
        or len(file_hash) != 64
        or any(character not in "0123456789abcdef" for character in file_hash)
    ):
        raise TrainingLogError("checkpoint curve-log file hash is invalid")
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise TrainingLogError(f"cannot read curve log during resume: {exc}") from exc
    cutoff = 0
    for _ in range(event_count):
        newline = payload.find(b"\n", cutoff)
        if newline < 0:
            raise TrainingLogError("curve log is shorter than the checkpointed prefix")
        cutoff = newline + 1
    prefix = payload[:cutoff]
    if hashlib.sha256(prefix).hexdigest() != file_hash:
        raise TrainingLogError("curve log checkpointed prefix SHA-256 mismatch")
    removed = len(payload) - cutoff
    if removed:
        with source.open("r+b") as handle:
            handle.truncate(cutoff)
            handle.flush()
            os.fsync(handle.fileno())
    _events, verified = read_training_log(
        source,
        expected_study_id=expected_study_id,
        expected_run_id=expected_run_id,
    )
    if verified.event_count != event_count:
        raise TrainingLogError("checkpointed curve event count mismatch")
    if verified.last_event_sha256 != checkpoint_state["last_event_sha256"]:
        raise TrainingLogError("checkpointed curve terminal event hash mismatch")
    if verified.last_updates_completed != updates:
        raise TrainingLogError("checkpointed curve update counter mismatch")
    return TrainingLogState(
        verified.event_count,
        verified.last_event_sha256,
        verified.last_updates_completed,
        removed,
    )


__all__ = [
    "CanonicalTrainingLog",
    "TrainingLogError",
    "TrainingLogState",
    "checkpoint_curve_log_state",
    "read_training_log",
    "reconcile_training_log_to_checkpoint",
]
