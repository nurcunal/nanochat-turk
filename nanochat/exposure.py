"""Deterministic training-text exposure and exact-resume contracts.

This module implements the immutable contracts used by the strict WP7 runtime
loaders and trainer. It supplies:

* immutable fixed-document, fixed-byte, and whole-document validation plans;
* strict equal-token versus sequential equal-text training-plan validation;
* serializable per-rank loader state with explicit resume bindings; and
* atomic, non-overwriting JSON publication.

The concrete loaders live in :mod:`nanochat.dataloader`; transactional
checkpoint publication lives in :mod:`nanochat.checkpoint_manager`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanochat.experiment_manifest import (
    canonical_json_bytes,
    seal_manifest,
    validate_dataset_manifest,
    validate_relative_path,
    verify_manifest_hash,
)


EXPOSURE_SCHEMA_VERSION = "1.0"
EXPOSURE_MANIFEST_TYPE = "text_exposure"
TRAINING_PLAN_TYPE = "training_exposure_plan"
LOADER_STATE_TYPE = "resumable_loader_state"
IMPLEMENTATION_SCOPE = "strict_runtime_integrated_v1"

_HEX = frozenset("0123456789abcdef")
_EXPOSURE_MODES = frozenset({"documents", "raw_bytes", "validation"})
_TRAINING_ESTIMANDS = frozenset({"equal_token", "equal_text"})


class ExposureError(ValueError):
    """Raised when an exposure or resume contract is unsafe or inconsistent."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _require_sha256(value: Any, name: str) -> str:
    if not _is_sha256(value):
        raise ExposureError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ExposureError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result == 0:
        raise ExposureError(f"{name} must be a positive integer")
    return result


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExposureError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str] | frozenset[str], name: str
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise ExposureError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ExposureError(f"{name} has unknown fields: {', '.join(unknown)}")


def _require_nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ExposureError(f"{name} must be a non-empty trimmed string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ExposureError(f"{name} must not contain control characters")
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    """One raw document at a stable position in an ordered Parquet inventory."""

    source_path: str
    row_group_index: int
    row_index: int
    text: str

    def __post_init__(self) -> None:
        try:
            validate_relative_path(self.source_path, location="source_path")
        except ValueError as exc:
            raise ExposureError(str(exc)) from exc
        _require_nonnegative_int(self.row_group_index, "row_group_index")
        _require_nonnegative_int(self.row_index, "row_index")
        if not isinstance(self.text, str):
            raise ExposureError("document text must be a string")
        try:
            self.text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ExposureError("document text is not valid Unicode") from exc

    def document_id(self, dataset_manifest_sha256: str) -> str:
        """Return the content-addressed-manifest-bound locator identity."""

        dataset_hash = _require_sha256(
            dataset_manifest_sha256, "dataset_manifest_sha256"
        )
        identity = {
            "source_dataset_manifest_sha256": dataset_hash,
            "source_path": self.source_path,
            "row_group_index": self.row_group_index,
            "row_index": self.row_index,
        }
        return f"doc_{_sha256_json(identity)}"


def _largest_utf8_prefix(payload: bytes, maximum_bytes: int) -> bytes:
    """Return the longest prefix no larger than the limit at a UTF-8 boundary."""

    if maximum_bytes >= len(payload):
        return payload
    end = maximum_bytes
    while end > 0:
        try:
            payload[:end].decode("utf-8", errors="strict")
            return payload[:end]
        except UnicodeDecodeError:
            end -= 1
    return b""


def _selected_document_record(
    document: SourceDocument,
    *,
    ordinal: int,
    dataset_manifest_sha256: str,
    included_payload: bytes,
) -> dict[str, Any]:
    payload = document.text.encode("utf-8", errors="strict")
    complete = len(included_payload) == len(payload)
    return {
        "document_ordinal": ordinal,
        "document_id": document.document_id(dataset_manifest_sha256),
        "source_path": document.source_path,
        "row_group_index": document.row_group_index,
        "row_index": document.row_index,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "included_bytes": len(included_payload),
        "included_sha256": hashlib.sha256(included_payload).hexdigest(),
        "complete": complete,
    }


def _finalize_exposure_manifest(
    *,
    mode: str,
    split: str,
    source_dataset_manifest_sha256: str,
    study_sha256: str | None,
    text_column: str,
    target_unit: str,
    target_value: int,
    selection_rule: str,
    records: list[dict[str, Any]],
    included_payload_digest: Any,
) -> dict[str, Any]:
    realized_bytes = sum(record["included_bytes"] for record in records)
    total_source_bytes = sum(record["payload_bytes"] for record in records)
    complete_documents = sum(bool(record["complete"]) for record in records)
    source_digest = hashlib.sha256()
    # A digest over per-document hashes is independently checkable without
    # embedding raw corpus text.  The actual included byte-stream digest below
    # is recomputed when the source is verified.
    for record in records:
        source_digest.update(bytes.fromhex(record["payload_sha256"]))
    manifest = {
        "schema_version": EXPOSURE_SCHEMA_VERSION,
        "manifest_type": EXPOSURE_MANIFEST_TYPE,
        "profile": "strict",
        "mode": mode,
        "source_split": split,
        "source_dataset_manifest_sha256": source_dataset_manifest_sha256,
        "study_sha256": study_sha256,
        "text_column": text_column,
        "selection": {
            "target_unit": target_unit,
            "target_value": target_value,
            "selection_rule": selection_rule,
            "realized_documents": len(records),
            "complete_documents": complete_documents,
            "partial_documents": len(records) - complete_documents,
            "realized_payload_bytes": realized_bytes,
            "target_shortfall_bytes": max(0, target_value - realized_bytes)
            if target_unit == "raw_bytes"
            else 0,
            "total_source_payload_bytes_loaded": total_source_bytes,
        },
        "documents": records,
        "ordered_document_ids_sha256": _sha256_json(
            [record["document_id"] for record in records]
        ),
        "ordered_document_records_sha256": _sha256_json(records),
        "source_document_hashes_sha256": source_digest.hexdigest(),
        "included_payload_sha256": included_payload_digest.hexdigest(),
        "test_data_accessed": False,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "canonical_sha256": None,
    }
    sealed = seal_manifest(manifest)
    validate_exposure_manifest(sealed)
    return sealed


def build_exposure_manifest(
    documents: Iterable[SourceDocument],
    *,
    mode: str,
    target_value: int,
    source_dataset_manifest_sha256: str,
    text_column: str = "text",
    study_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic fixed-text or fixed-validation manifest.

    ``documents`` and ``raw_bytes`` are training-only modes.  ``validation``
    consumes validation-only documents and selects the smallest whole-document
    prefix totaling at least ``target_value`` UTF-8 payload bytes.
    """

    if mode not in _EXPOSURE_MODES:
        raise ExposureError(f"mode must be one of {sorted(_EXPOSURE_MODES)}")
    target = _require_positive_int(target_value, "target_value")
    dataset_hash = _require_sha256(
        source_dataset_manifest_sha256, "source_dataset_manifest_sha256"
    )
    if study_sha256 is not None:
        _require_sha256(study_sha256, "study_sha256")
    column = _require_nonempty_text(text_column, "text_column")
    split = "validation" if mode == "validation" else "train"
    target_unit = "documents" if mode == "documents" else "raw_bytes"
    rule = {
        "documents": "first-n-complete-ordered-documents",
        "raw_bytes": "longest-ordered-utf8-prefix-at-or-below-target",
        "validation": "smallest-whole-document-prefix-at-or-above-target",
    }[mode]

    selected: list[dict[str, Any]] = []
    included_digest = hashlib.sha256()
    seen_locators: set[tuple[str, int, int]] = set()
    realized_bytes = 0
    raw_boundary_reached = False

    for document in documents:
        if not isinstance(document, SourceDocument):
            raise ExposureError("documents must contain SourceDocument instances")
        locator = (
            document.source_path,
            document.row_group_index,
            document.row_index,
        )
        if locator in seen_locators:
            raise ExposureError(f"duplicate source document locator: {locator!r}")
        seen_locators.add(locator)
        payload = document.text.encode("utf-8", errors="strict")

        if mode == "documents":
            included = payload
        elif mode == "validation":
            included = payload
        else:
            remaining = target - realized_bytes
            if remaining <= 0:
                break
            included = _largest_utf8_prefix(payload, remaining)
            # A zero-byte terminal prefix does not identify any part of the
            # document and is omitted.  The preceding whole-document boundary
            # remains the deterministic corpus-prefix boundary.
            if not included and payload:
                if not selected:
                    raise ExposureError(
                        "raw-byte target ends before the first positive UTF-8 boundary"
                    )
                raw_boundary_reached = True
                break

        record = _selected_document_record(
            document,
            ordinal=len(selected),
            dataset_manifest_sha256=dataset_hash,
            included_payload=included,
        )
        selected.append(record)
        included_digest.update(included)
        realized_bytes += len(included)

        if mode == "documents" and len(selected) == target:
            break
        if mode == "validation" and realized_bytes >= target:
            break
        if mode == "raw_bytes" and (
            realized_bytes >= target or len(included) < len(payload)
        ):
            raw_boundary_reached = True
            break

    if mode == "documents" and len(selected) != target:
        raise ExposureError(
            f"source contains only {len(selected)} documents; {target} required"
        )
    if mode == "validation" and realized_bytes < target:
        raise ExposureError(
            f"validation source contains only {realized_bytes} UTF-8 bytes; "
            f"at least {target} required"
        )
    if mode == "raw_bytes":
        if not selected or realized_bytes <= 0:
            raise ExposureError("raw-byte selection must contain a positive prefix")
        if realized_bytes < target and not raw_boundary_reached:
            raise ExposureError(
                f"source contains only {realized_bytes} UTF-8 bytes; {target} required"
            )
        shortfall = target - realized_bytes
        if shortfall < 0 or shortfall > 3:
            raise ExposureError(
                "source ended before the raw-byte target or UTF-8 shortfall exceeds 3 bytes"
            )

    return _finalize_exposure_manifest(
        mode=mode,
        split=split,
        source_dataset_manifest_sha256=dataset_hash,
        study_sha256=study_sha256,
        text_column=column,
        target_unit=target_unit,
        target_value=target,
        selection_rule=rule,
        records=selected,
        included_payload_digest=included_digest,
    )


_EXPOSURE_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_type",
        "profile",
        "mode",
        "source_split",
        "source_dataset_manifest_sha256",
        "study_sha256",
        "text_column",
        "selection",
        "documents",
        "ordered_document_ids_sha256",
        "ordered_document_records_sha256",
        "source_document_hashes_sha256",
        "included_payload_sha256",
        "test_data_accessed",
        "implementation_scope",
        "canonical_sha256",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "target_unit",
        "target_value",
        "selection_rule",
        "realized_documents",
        "complete_documents",
        "partial_documents",
        "realized_payload_bytes",
        "target_shortfall_bytes",
        "total_source_payload_bytes_loaded",
    }
)
_DOCUMENT_FIELDS = frozenset(
    {
        "document_ordinal",
        "document_id",
        "source_path",
        "row_group_index",
        "row_index",
        "payload_bytes",
        "payload_sha256",
        "included_bytes",
        "included_sha256",
        "complete",
    }
)


def _validate_against_dataset_manifest(
    exposure_manifest: Mapping[str, Any],
    source_dataset_manifest: Mapping[str, Any],
) -> None:
    try:
        validate_dataset_manifest(source_dataset_manifest, profile="strict")
        dataset_hash = verify_manifest_hash(source_dataset_manifest)
    except ValueError as exc:
        raise ExposureError(f"invalid strict source dataset manifest: {exc}") from exc
    if dataset_hash != exposure_manifest["source_dataset_manifest_sha256"]:
        raise ExposureError("exposure manifest is bound to a different dataset manifest")
    if exposure_manifest["text_column"] != source_dataset_manifest["text_column"]:
        raise ExposureError("exposure text_column differs from the dataset manifest")

    ordered_files = source_dataset_manifest["ordered_files"]
    file_positions = {record["path"]: index for index, record in enumerate(ordered_files)}
    validation_path = source_dataset_manifest["validation_file"]
    split = exposure_manifest["source_split"]
    previous: tuple[int, int, int] | None = None
    for index, record in enumerate(exposure_manifest["documents"]):
        source_path = record["source_path"]
        if source_path not in file_positions:
            raise ExposureError(f"documents[{index}] references an unknown source file")
        if split == "train" and source_path == validation_path:
            raise ExposureError("training exposure contains the held-out validation file")
        if split == "validation" and source_path != validation_path:
            raise ExposureError("fixed validation exposure contains a training file")
        locator = (
            file_positions[source_path],
            record["row_group_index"],
            record["row_index"],
        )
        if previous is not None and locator <= previous:
            raise ExposureError("document locators are not in strict dataset order")
        previous = locator


def validate_exposure_manifest(
    manifest: Mapping[str, Any],
    *,
    source_dataset_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate an immutable exposure manifest and optional source binding."""

    value = _require_mapping(manifest, "manifest")
    _require_exact_keys(value, _EXPOSURE_ROOT_FIELDS, "manifest")
    if value["schema_version"] != EXPOSURE_SCHEMA_VERSION:
        raise ExposureError(f"schema_version must equal {EXPOSURE_SCHEMA_VERSION!r}")
    if value["manifest_type"] != EXPOSURE_MANIFEST_TYPE:
        raise ExposureError(f"manifest_type must equal {EXPOSURE_MANIFEST_TYPE!r}")
    if value["profile"] != "strict":
        raise ExposureError("profile must equal 'strict'")
    try:
        verify_manifest_hash(value)
    except ValueError as exc:
        raise ExposureError(str(exc)) from exc

    mode = value["mode"]
    if mode not in _EXPOSURE_MODES:
        raise ExposureError("manifest mode is unsupported")
    expected_split = "validation" if mode == "validation" else "train"
    if value["source_split"] != expected_split:
        raise ExposureError(f"{mode} mode must use the {expected_split!r} split")
    _require_sha256(
        value["source_dataset_manifest_sha256"],
        "source_dataset_manifest_sha256",
    )
    if value["study_sha256"] is not None:
        _require_sha256(value["study_sha256"], "study_sha256")
    _require_nonempty_text(value["text_column"], "text_column")
    if value["test_data_accessed"] is not False:
        raise ExposureError("test_data_accessed must be false")
    if value["implementation_scope"] != IMPLEMENTATION_SCOPE:
        raise ExposureError("implementation_scope must identify the strict runtime layer")

    selection = _require_mapping(value["selection"], "selection")
    _require_exact_keys(selection, _SELECTION_FIELDS, "selection")
    target = _require_positive_int(selection["target_value"], "selection.target_value")
    for field_name in (
        "realized_documents",
        "complete_documents",
        "partial_documents",
        "realized_payload_bytes",
        "target_shortfall_bytes",
        "total_source_payload_bytes_loaded",
    ):
        _require_nonnegative_int(selection[field_name], f"selection.{field_name}")

    documents = value["documents"]
    if not isinstance(documents, list) or not documents:
        raise ExposureError("documents must be a non-empty JSON array")
    ids: list[str] = []
    locators: set[tuple[str, int, int]] = set()
    realized_bytes = 0
    source_bytes = 0
    complete_count = 0
    source_digest = hashlib.sha256()
    for index, raw_record in enumerate(documents):
        record = _require_mapping(raw_record, f"documents[{index}]")
        _require_exact_keys(record, _DOCUMENT_FIELDS, f"documents[{index}]")
        if record["document_ordinal"] != index:
            raise ExposureError("document ordinals must be contiguous and ordered")
        document_id = _require_nonempty_text(
            record["document_id"], f"documents[{index}].document_id"
        )
        if not document_id.startswith("doc_") or not _is_sha256(document_id[4:]):
            raise ExposureError(f"documents[{index}].document_id is invalid")
        expected_document_id = "doc_" + _sha256_json(
            {
                "source_dataset_manifest_sha256": value[
                    "source_dataset_manifest_sha256"
                ],
                "source_path": record["source_path"],
                "row_group_index": record["row_group_index"],
                "row_index": record["row_index"],
            }
        )
        if document_id != expected_document_id:
            raise ExposureError(
                f"documents[{index}].document_id does not match its source locator"
            )
        ids.append(document_id)
        try:
            validate_relative_path(
                record["source_path"], location=f"documents[{index}].source_path"
            )
        except ValueError as exc:
            raise ExposureError(str(exc)) from exc
        row_group = _require_nonnegative_int(
            record["row_group_index"], f"documents[{index}].row_group_index"
        )
        row = _require_nonnegative_int(
            record["row_index"], f"documents[{index}].row_index"
        )
        locator = (record["source_path"], row_group, row)
        if locator in locators:
            raise ExposureError("duplicate document locator")
        locators.add(locator)
        payload_bytes = _require_nonnegative_int(
            record["payload_bytes"], f"documents[{index}].payload_bytes"
        )
        included_bytes = _require_nonnegative_int(
            record["included_bytes"], f"documents[{index}].included_bytes"
        )
        if included_bytes > payload_bytes:
            raise ExposureError("included_bytes cannot exceed payload_bytes")
        _require_sha256(record["payload_sha256"], f"documents[{index}].payload_sha256")
        _require_sha256(
            record["included_sha256"], f"documents[{index}].included_sha256"
        )
        if not isinstance(record["complete"], bool):
            raise ExposureError(f"documents[{index}].complete must be boolean")
        if record["complete"] != (included_bytes == payload_bytes):
            raise ExposureError("complete must exactly describe included payload size")
        if record["complete"]:
            complete_count += 1
            if record["included_sha256"] != record["payload_sha256"]:
                raise ExposureError("complete documents must have identical payload hashes")
        elif index != len(documents) - 1:
            raise ExposureError("only the final selected document may be partial")
        source_digest.update(bytes.fromhex(record["payload_sha256"]))
        realized_bytes += included_bytes
        source_bytes += payload_bytes

    if len(ids) != len(set(ids)):
        raise ExposureError("document IDs must be unique")
    if selection["realized_documents"] != len(documents):
        raise ExposureError("realized_documents does not match documents")
    if selection["complete_documents"] != complete_count:
        raise ExposureError("complete_documents does not match documents")
    if selection["partial_documents"] != len(documents) - complete_count:
        raise ExposureError("partial_documents does not match documents")
    if selection["realized_payload_bytes"] != realized_bytes:
        raise ExposureError("realized_payload_bytes does not match documents")
    if selection["total_source_payload_bytes_loaded"] != source_bytes:
        raise ExposureError("total_source_payload_bytes_loaded does not match documents")
    if value["ordered_document_ids_sha256"] != _sha256_json(ids):
        raise ExposureError("ordered_document_ids_sha256 does not match documents")
    if value["ordered_document_records_sha256"] != _sha256_json(documents):
        raise ExposureError("ordered_document_records_sha256 does not match documents")
    if value["source_document_hashes_sha256"] != source_digest.hexdigest():
        raise ExposureError("source_document_hashes_sha256 does not match documents")
    _require_sha256(value["included_payload_sha256"], "included_payload_sha256")

    expected_unit = "documents" if mode == "documents" else "raw_bytes"
    if selection["target_unit"] != expected_unit:
        raise ExposureError(f"{mode} mode requires target_unit={expected_unit!r}")
    expected_rule = {
        "documents": "first-n-complete-ordered-documents",
        "raw_bytes": "longest-ordered-utf8-prefix-at-or-below-target",
        "validation": "smallest-whole-document-prefix-at-or-above-target",
    }[mode]
    if selection["selection_rule"] != expected_rule:
        raise ExposureError("selection_rule does not match mode")

    if mode == "documents":
        if len(documents) != target or complete_count != len(documents):
            raise ExposureError("fixed-document selection must contain exactly N whole documents")
        if selection["target_shortfall_bytes"] != 0:
            raise ExposureError("fixed-document selection cannot record a byte shortfall")
    elif mode == "raw_bytes":
        shortfall = target - realized_bytes
        if realized_bytes <= 0 or shortfall < 0 or shortfall > 3:
            raise ExposureError("raw-byte realization must end within 3 bytes below target")
        if selection["target_shortfall_bytes"] != shortfall:
            raise ExposureError("raw-byte target shortfall is inconsistent")
        if len(documents) - complete_count > 1:
            raise ExposureError("raw-byte selection has multiple partial documents")
    else:
        if complete_count != len(documents) or realized_bytes < target:
            raise ExposureError("validation selection must use whole documents and reach target")
        prefix_without_last = realized_bytes - documents[-1]["payload_bytes"]
        if prefix_without_last >= target:
            raise ExposureError("validation selection is not the smallest whole-document prefix")
        if selection["target_shortfall_bytes"] != 0:
            raise ExposureError("validation selection cannot record a byte shortfall")

    if source_dataset_manifest is not None:
        _validate_against_dataset_manifest(value, source_dataset_manifest)


def verify_exposure_source(
    manifest: Mapping[str, Any],
    documents: Iterable[SourceDocument],
    *,
    source_dataset_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Independently rebuild a plan from source documents and compare all bytes."""

    validate_exposure_manifest(
        manifest, source_dataset_manifest=source_dataset_manifest
    )
    rebuilt = build_exposure_manifest(
        documents,
        mode=manifest["mode"],
        target_value=manifest["selection"]["target_value"],
        source_dataset_manifest_sha256=manifest[
            "source_dataset_manifest_sha256"
        ],
        text_column=manifest["text_column"],
        study_sha256=manifest["study_sha256"],
    )
    if rebuilt != dict(manifest):
        raise ExposureError("exposure source does not reproduce the sealed manifest")


_TRAINING_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_type",
        "estimand",
        "source_split",
        "source_dataset_manifest_sha256",
        "study_sha256",
        "tokenizer_sha256",
        "seed",
        "world_size",
        "data_order",
        "schedule_basis",
        "horizon",
        "exposure_manifest_sha256",
        "derived",
        "test_data_accessed",
        "implementation_scope",
        "canonical_sha256",
    }
)


def build_training_exposure_plan(
    *,
    estimand: str,
    source_dataset_manifest_sha256: str,
    study_sha256: str,
    tokenizer_sha256: str,
    seed: int,
    world_size: int,
    target_token_positions: int | None = None,
    exposure_manifest: Mapping[str, Any] | None = None,
    derived_token_positions: int | None = None,
    derived_optimizer_steps: int | None = None,
) -> dict[str, Any]:
    """Build one tokenizer-specific equal-token or equal-text planning record."""

    if estimand not in _TRAINING_ESTIMANDS:
        raise ExposureError(f"estimand must be one of {sorted(_TRAINING_ESTIMANDS)}")
    dataset_hash = _require_sha256(
        source_dataset_manifest_sha256, "source_dataset_manifest_sha256"
    )
    study_hash = _require_sha256(study_sha256, "study_sha256")
    tokenizer_hash = _require_sha256(tokenizer_sha256, "tokenizer_sha256")
    run_seed = _require_nonnegative_int(seed, "seed")
    run_world_size = _require_positive_int(world_size, "world_size")

    if estimand == "equal_token":
        if exposure_manifest is not None:
            raise ExposureError("equal-token plans must not use a fixed-text manifest")
        horizon_value = _require_positive_int(
            target_token_positions, "target_token_positions"
        )
        if derived_token_positions not in {None, horizon_value}:
            raise ExposureError(
                "equal-token derived_token_positions must equal the token horizon"
            )
        if derived_optimizer_steps is not None:
            _require_positive_int(derived_optimizer_steps, "derived_optimizer_steps")
        unit = "token_positions"
        data_order = "bestfit"
        manifest_hash: str | None = None
        derived_positions = horizon_value
    else:
        if target_token_positions is not None:
            raise ExposureError("equal-text plans do not accept a token-position horizon")
        if exposure_manifest is None:
            raise ExposureError("equal-text plans require a sealed exposure manifest")
        validate_exposure_manifest(exposure_manifest)
        if exposure_manifest["source_split"] != "train":
            raise ExposureError("equal-text training cannot use the validation manifest")
        if exposure_manifest["source_dataset_manifest_sha256"] != dataset_hash:
            raise ExposureError("equal-text plan and exposure manifest use different datasets")
        if (
            exposure_manifest["study_sha256"] is not None
            and exposure_manifest["study_sha256"] != study_hash
        ):
            raise ExposureError("equal-text plan and exposure manifest use different studies")
        if exposure_manifest["mode"] not in {"documents", "raw_bytes"}:
            raise ExposureError("equal-text training requires documents or raw_bytes mode")
        unit = (
            "documents"
            if exposure_manifest["mode"] == "documents"
            else "raw_bytes"
        )
        horizon_value = (
            exposure_manifest["selection"]["realized_documents"]
            if unit == "documents"
            else exposure_manifest["selection"]["realized_payload_bytes"]
        )
        data_order = "sequential"
        manifest_hash = exposure_manifest["canonical_sha256"]
        if derived_token_positions is not None:
            _require_positive_int(derived_token_positions, "derived_token_positions")
        if derived_optimizer_steps is not None:
            _require_positive_int(derived_optimizer_steps, "derived_optimizer_steps")
        derived_positions = derived_token_positions

    plan = {
        "schema_version": EXPOSURE_SCHEMA_VERSION,
        "manifest_type": TRAINING_PLAN_TYPE,
        "estimand": estimand,
        "source_split": "train",
        "source_dataset_manifest_sha256": dataset_hash,
        "study_sha256": study_hash,
        "tokenizer_sha256": tokenizer_hash,
        "seed": run_seed,
        "world_size": run_world_size,
        "data_order": data_order,
        "schedule_basis": "normalized_exposure_fraction",
        "horizon": {"unit": unit, "value": horizon_value},
        "exposure_manifest_sha256": manifest_hash,
        "derived": {
            "token_positions": derived_positions,
            "optimizer_steps": derived_optimizer_steps,
        },
        "test_data_accessed": False,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "canonical_sha256": None,
    }
    sealed = seal_manifest(plan)
    validate_training_exposure_plan(
        sealed, exposure_manifest=exposure_manifest
    )
    return sealed


def validate_training_exposure_plan(
    plan: Mapping[str, Any],
    *,
    exposure_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate one tokenizer-specific training exposure plan."""

    value = _require_mapping(plan, "plan")
    _require_exact_keys(value, _TRAINING_PLAN_FIELDS, "plan")
    if value["schema_version"] != EXPOSURE_SCHEMA_VERSION:
        raise ExposureError("training plan schema_version is unsupported")
    if value["manifest_type"] != TRAINING_PLAN_TYPE:
        raise ExposureError(f"manifest_type must equal {TRAINING_PLAN_TYPE!r}")
    try:
        verify_manifest_hash(value)
    except ValueError as exc:
        raise ExposureError(str(exc)) from exc
    estimand = value["estimand"]
    if estimand not in _TRAINING_ESTIMANDS:
        raise ExposureError("training plan estimand is unsupported")
    if value["source_split"] != "train" or value["test_data_accessed"] is not False:
        raise ExposureError("training plans must be train-only and test-blind")
    for field_name in (
        "source_dataset_manifest_sha256",
        "study_sha256",
        "tokenizer_sha256",
    ):
        _require_sha256(value[field_name], field_name)
    _require_nonnegative_int(value["seed"], "seed")
    _require_positive_int(value["world_size"], "world_size")
    if value["schedule_basis"] != "normalized_exposure_fraction":
        raise ExposureError("schedule_basis must equal normalized_exposure_fraction")
    if value["implementation_scope"] != IMPLEMENTATION_SCOPE:
        raise ExposureError("training plan must identify the strict runtime layer")

    horizon = _require_mapping(value["horizon"], "horizon")
    _require_exact_keys(horizon, {"unit", "value"}, "horizon")
    horizon_value = _require_positive_int(horizon["value"], "horizon.value")
    derived = _require_mapping(value["derived"], "derived")
    _require_exact_keys(derived, {"token_positions", "optimizer_steps"}, "derived")
    for field_name in ("token_positions", "optimizer_steps"):
        if derived[field_name] is not None:
            _require_positive_int(derived[field_name], f"derived.{field_name}")

    if estimand == "equal_token":
        if horizon["unit"] != "token_positions" or value["data_order"] != "bestfit":
            raise ExposureError("equal-token plans require token_positions and bestfit")
        if value["exposure_manifest_sha256"] is not None:
            raise ExposureError("equal-token plans cannot bind a fixed-text manifest")
        if derived["token_positions"] != horizon_value:
            raise ExposureError("equal-token derived token count must equal the horizon")
        if exposure_manifest is not None:
            raise ExposureError("equal-token validation does not accept an exposure manifest")
        return

    if horizon["unit"] not in {"documents", "raw_bytes"}:
        raise ExposureError("equal-text horizon unit must be documents or raw_bytes")
    if value["data_order"] != "sequential":
        raise ExposureError("equal-text plans require sequential document order")
    _require_sha256(
        value["exposure_manifest_sha256"], "exposure_manifest_sha256"
    )
    if exposure_manifest is not None:
        validate_exposure_manifest(exposure_manifest)
        if exposure_manifest["canonical_sha256"] != value["exposure_manifest_sha256"]:
            raise ExposureError("training plan binds a different exposure manifest")
        if exposure_manifest["source_dataset_manifest_sha256"] != value[
            "source_dataset_manifest_sha256"
        ]:
            raise ExposureError("training plan and exposure manifest datasets differ")
        if (
            exposure_manifest["study_sha256"] is not None
            and exposure_manifest["study_sha256"] != value["study_sha256"]
        ):
            raise ExposureError("training plan and exposure manifest studies differ")
        expected_unit = (
            "documents"
            if exposure_manifest["mode"] == "documents"
            else "raw_bytes"
        )
        expected_value = (
            exposure_manifest["selection"]["realized_documents"]
            if expected_unit == "documents"
            else exposure_manifest["selection"]["realized_payload_bytes"]
        )
        if exposure_manifest["source_split"] != "train" or exposure_manifest[
            "mode"
        ] not in {"documents", "raw_bytes"}:
            raise ExposureError("equal-text plan must bind a training exposure manifest")
        if horizon != {"unit": expected_unit, "value": expected_value}:
            raise ExposureError("equal-text horizon does not match its exposure manifest")


def validate_training_plan_pair(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> None:
    """Fail unless two plans implement one valid paired estimand."""

    validate_training_exposure_plan(left)
    validate_training_exposure_plan(right)
    if left["estimand"] != right["estimand"]:
        raise ExposureError("paired plans must use the same estimand")
    if left["tokenizer_sha256"] == right["tokenizer_sha256"]:
        raise ExposureError("paired plans must bind two distinct tokenizers")
    common_fields = (
        "schema_version",
        "manifest_type",
        "estimand",
        "source_split",
        "source_dataset_manifest_sha256",
        "study_sha256",
        "seed",
        "world_size",
        "data_order",
        "schedule_basis",
        "horizon",
        "exposure_manifest_sha256",
        "test_data_accessed",
        "implementation_scope",
    )
    for field_name in common_fields:
        if left[field_name] != right[field_name]:
            raise ExposureError(f"paired plans differ at {field_name}")
    if left["estimand"] == "equal_token" and left["derived"] != right["derived"]:
        raise ExposureError("equal-token paired plans must have identical derived horizons")
    # Tokenizer-derived counts and absolute step counts may differ only for the
    # sequential equal-text estimand.  No other pair fields are allowlisted.


def _validate_base64(value: Any, name: str) -> str:
    text = _require_nonempty_text(value, name)
    try:
        decoded = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ExposureError(f"{name} must be canonical base64") from exc
    if not decoded or base64.b64encode(decoded).decode("ascii") != text:
        raise ExposureError(f"{name} must be non-empty canonical base64")
    return text


@dataclass(frozen=True)
class LoaderPosition:
    """Exact next-source position for one dataloader rank."""

    file_index: int
    row_group_index: int
    row_index: int
    document_ordinal: int
    epoch: int

    def __post_init__(self) -> None:
        for name in (
            "file_index",
            "row_group_index",
            "row_index",
            "document_ordinal",
            "epoch",
        ):
            _require_nonnegative_int(getattr(self, name), f"position.{name}")

    def to_dict(self) -> dict[str, int]:
        return {
            "file_index": self.file_index,
            "row_group_index": self.row_group_index,
            "row_index": self.row_index,
            "document_ordinal": self.document_ordinal,
            "epoch": self.epoch,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LoaderPosition":
        mapping = _require_mapping(value, "position")
        fields = {
            "file_index",
            "row_group_index",
            "row_index",
            "document_ordinal",
            "epoch",
        }
        _require_exact_keys(mapping, fields, "position")
        return cls(**{name: mapping[name] for name in fields})


@dataclass(frozen=True)
class BufferCursor:
    """Regenerable buffered-document identity and packing offsets."""

    document_id: str
    token_offset: int
    payload_byte_offset: int

    def __post_init__(self) -> None:
        _require_nonempty_text(self.document_id, "buffer.document_id")
        _require_nonnegative_int(self.token_offset, "buffer.token_offset")
        _require_nonnegative_int(
            self.payload_byte_offset, "buffer.payload_byte_offset"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "token_offset": self.token_offset,
            "payload_byte_offset": self.payload_byte_offset,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BufferCursor":
        mapping = _require_mapping(value, "buffer cursor")
        fields = {"document_id", "token_offset", "payload_byte_offset"}
        _require_exact_keys(mapping, fields, "buffer cursor")
        return cls(**{name: mapping[name] for name in fields})


@dataclass(frozen=True)
class PrefetchedBatch:
    """Identity needed to regenerate and verify one already-prefetched batch."""

    document_ids: tuple[str, ...]
    token_count: int
    token_ids_sha256: str

    def __post_init__(self) -> None:
        if not self.document_ids:
            raise ExposureError("prefetched batch must contain document IDs")
        for index, document_id in enumerate(self.document_ids):
            _require_nonempty_text(document_id, f"prefetch.document_ids[{index}]")
        if len(self.document_ids) != len(set(self.document_ids)):
            raise ExposureError("prefetched document IDs must be unique")
        _require_positive_int(self.token_count, "prefetch.token_count")
        _require_sha256(self.token_ids_sha256, "prefetch.token_ids_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_ids": list(self.document_ids),
            "token_count": self.token_count,
            "token_ids_sha256": self.token_ids_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PrefetchedBatch":
        mapping = _require_mapping(value, "prefetched_batch")
        fields = {"document_ids", "token_count", "token_ids_sha256"}
        _require_exact_keys(mapping, fields, "prefetched_batch")
        ids = mapping["document_ids"]
        if not isinstance(ids, list):
            raise ExposureError("prefetched_batch.document_ids must be an array")
        return cls(
            document_ids=tuple(ids),
            token_count=mapping["token_count"],
            token_ids_sha256=mapping["token_ids_sha256"],
        )


@dataclass(frozen=True)
class RNGSnapshot:
    """Serialized Python/CPU and optional per-rank CUDA RNG state."""

    python_state_b64: str
    cpu_state_b64: str
    cuda_state_b64: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_base64(self.python_state_b64, "rng.python_state_b64")
        _validate_base64(self.cpu_state_b64, "rng.cpu_state_b64")
        if len(self.cuda_state_b64) > 1:
            raise ExposureError("per-rank loader state may contain at most one CUDA RNG state")
        for index, state in enumerate(self.cuda_state_b64):
            _validate_base64(state, f"rng.cuda_state_b64[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_state_b64": self.python_state_b64,
            "cpu_state_b64": self.cpu_state_b64,
            "cuda_state_b64": list(self.cuda_state_b64),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RNGSnapshot":
        mapping = _require_mapping(value, "rng")
        fields = {"python_state_b64", "cpu_state_b64", "cuda_state_b64"}
        _require_exact_keys(mapping, fields, "rng")
        cuda = mapping["cuda_state_b64"]
        if not isinstance(cuda, list):
            raise ExposureError("rng.cuda_state_b64 must be an array")
        return cls(
            python_state_b64=mapping["python_state_b64"],
            cpu_state_b64=mapping["cpu_state_b64"],
            cuda_state_b64=tuple(cuda),
        )


@dataclass(frozen=True)
class ExposureTotals:
    """Monotone per-rank counters required for exposure accounting."""

    token_positions: int = 0
    valid_target_tokens: int = 0
    payload_bytes: int = 0
    source_bytes_loaded: int = 0
    documents_loaded: int = 0
    documents_started: int = 0
    documents_completed: int = 0
    documents_cropped: int = 0
    discarded_tokens: int = 0
    discarded_bytes: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_nonnegative_int(getattr(self, name), f"totals.{name}")
        if self.valid_target_tokens > self.token_positions:
            raise ExposureError("valid target tokens cannot exceed token positions")
        if self.payload_bytes > self.source_bytes_loaded:
            raise ExposureError("represented payload bytes cannot exceed loaded source bytes")
        if self.documents_started > self.documents_loaded:
            raise ExposureError("documents started cannot exceed documents loaded")
        if self.documents_completed > self.documents_started:
            raise ExposureError("documents completed cannot exceed documents started")
        if self.documents_cropped > self.documents_completed:
            raise ExposureError("cropped documents cannot exceed completed documents")
        if self.discarded_bytes > self.source_bytes_loaded:
            raise ExposureError("discarded bytes cannot exceed loaded source bytes")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExposureTotals":
        mapping = _require_mapping(value, "totals")
        fields = set(cls.__dataclass_fields__)
        _require_exact_keys(mapping, fields, "totals")
        return cls(**{name: mapping[name] for name in fields})


@dataclass(frozen=True)
class ResumableLoaderState:
    """Per-rank loader state consumed by strict loaders and checkpoints."""

    loader_kind: str
    source_dataset_manifest_sha256: str
    study_sha256: str
    tokenizer_sha256: str
    exposure_manifest_sha256: str | None
    rank: int
    world_size: int
    next_batch_index: int
    position: LoaderPosition
    buffer: tuple[BufferCursor, ...]
    current_document: BufferCursor | None
    prefetched_batch: PrefetchedBatch | None
    totals: ExposureTotals
    rng: RNGSnapshot
    resume_lineage: tuple[str, ...] = ()
    canonical_sha256: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.loader_kind not in {"bestfit", "sequential"}:
            raise ExposureError("loader_kind must be 'bestfit' or 'sequential'")
        _require_sha256(
            self.source_dataset_manifest_sha256,
            "source_dataset_manifest_sha256",
        )
        _require_sha256(self.study_sha256, "study_sha256")
        _require_sha256(self.tokenizer_sha256, "tokenizer_sha256")
        if self.loader_kind == "sequential":
            _require_sha256(
                self.exposure_manifest_sha256, "exposure_manifest_sha256"
            )
        elif self.exposure_manifest_sha256 is not None:
            raise ExposureError("bestfit loader state cannot bind a fixed-text manifest")
        rank = _require_nonnegative_int(self.rank, "rank")
        world_size = _require_positive_int(self.world_size, "world_size")
        if rank >= world_size:
            raise ExposureError("rank must be smaller than world_size")
        _require_nonnegative_int(self.next_batch_index, "next_batch_index")
        if not isinstance(self.position, LoaderPosition):
            raise ExposureError("position must be a LoaderPosition")
        if not isinstance(self.totals, ExposureTotals):
            raise ExposureError("totals must be ExposureTotals")
        if not isinstance(self.rng, RNGSnapshot):
            raise ExposureError("rng must be RNGSnapshot")

        buffer_ids: list[str] = []
        for item in self.buffer:
            if not isinstance(item, BufferCursor):
                raise ExposureError("buffer must contain BufferCursor values")
            buffer_ids.append(item.document_id)
        if len(buffer_ids) != len(set(buffer_ids)):
            raise ExposureError("buffer document IDs must be unique")
        occupied = set(buffer_ids)
        if self.current_document is not None:
            if not isinstance(self.current_document, BufferCursor):
                raise ExposureError("current_document must be a BufferCursor or null")
            if self.current_document.document_id in occupied:
                raise ExposureError("current document must not also appear in the buffer")
            occupied.add(self.current_document.document_id)
        if self.prefetched_batch is not None:
            if not isinstance(self.prefetched_batch, PrefetchedBatch):
                raise ExposureError("prefetched_batch must be PrefetchedBatch or null")
            overlap = occupied.intersection(self.prefetched_batch.document_ids)
            if overlap:
                raise ExposureError("prefetched documents overlap active buffer/current state")

        for index, parent_hash in enumerate(self.resume_lineage):
            _require_sha256(parent_hash, f"resume_lineage[{index}]")
        if len(self.resume_lineage) != len(set(self.resume_lineage)):
            raise ExposureError("resume lineage must not contain cycles or duplicates")
        if self.canonical_sha256 is not None:
            _require_sha256(self.canonical_sha256, "canonical_sha256")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": EXPOSURE_SCHEMA_VERSION,
            "manifest_type": LOADER_STATE_TYPE,
            "implementation_scope": IMPLEMENTATION_SCOPE,
            "loader_kind": self.loader_kind,
            "source_dataset_manifest_sha256": self.source_dataset_manifest_sha256,
            "study_sha256": self.study_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "exposure_manifest_sha256": self.exposure_manifest_sha256,
            "rank": self.rank,
            "world_size": self.world_size,
            "next_batch_index": self.next_batch_index,
            "position": self.position.to_dict(),
            "buffer": [item.to_dict() for item in self.buffer],
            "current_document": None
            if self.current_document is None
            else self.current_document.to_dict(),
            "prefetched_batch": None
            if self.prefetched_batch is None
            else self.prefetched_batch.to_dict(),
            "totals": self.totals.to_dict(),
            "rng": self.rng.to_dict(),
            "resume_lineage": list(self.resume_lineage),
            "test_data_accessed": False,
            "canonical_sha256": None,
        }
        sealed = seal_manifest(payload)
        if self.canonical_sha256 is not None and sealed["canonical_sha256"] != self.canonical_sha256:
            raise ExposureError("loader state fields do not match canonical_sha256")
        return sealed

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResumableLoaderState":
        mapping = _require_mapping(value, "loader state")
        fields = {
            "schema_version",
            "manifest_type",
            "implementation_scope",
            "loader_kind",
            "source_dataset_manifest_sha256",
            "study_sha256",
            "tokenizer_sha256",
            "exposure_manifest_sha256",
            "rank",
            "world_size",
            "next_batch_index",
            "position",
            "buffer",
            "current_document",
            "prefetched_batch",
            "totals",
            "rng",
            "resume_lineage",
            "test_data_accessed",
            "canonical_sha256",
        }
        _require_exact_keys(mapping, fields, "loader state")
        if mapping["schema_version"] != EXPOSURE_SCHEMA_VERSION:
            raise ExposureError("loader state schema_version is unsupported")
        if mapping["manifest_type"] != LOADER_STATE_TYPE:
            raise ExposureError(f"manifest_type must equal {LOADER_STATE_TYPE!r}")
        if mapping["implementation_scope"] != IMPLEMENTATION_SCOPE:
            raise ExposureError("loader state must identify the strict runtime layer")
        if mapping["test_data_accessed"] is not False:
            raise ExposureError("loader states must remain test-blind")
        try:
            verify_manifest_hash(mapping)
        except ValueError as exc:
            raise ExposureError(str(exc)) from exc
        raw_buffer = mapping["buffer"]
        if not isinstance(raw_buffer, list):
            raise ExposureError("buffer must be a JSON array")
        raw_lineage = mapping["resume_lineage"]
        if not isinstance(raw_lineage, list):
            raise ExposureError("resume_lineage must be a JSON array")
        state = cls(
            loader_kind=mapping["loader_kind"],
            source_dataset_manifest_sha256=mapping[
                "source_dataset_manifest_sha256"
            ],
            study_sha256=mapping["study_sha256"],
            tokenizer_sha256=mapping["tokenizer_sha256"],
            exposure_manifest_sha256=mapping["exposure_manifest_sha256"],
            rank=mapping["rank"],
            world_size=mapping["world_size"],
            next_batch_index=mapping["next_batch_index"],
            position=LoaderPosition.from_mapping(mapping["position"]),
            buffer=tuple(BufferCursor.from_mapping(item) for item in raw_buffer),
            current_document=None
            if mapping["current_document"] is None
            else BufferCursor.from_mapping(mapping["current_document"]),
            prefetched_batch=None
            if mapping["prefetched_batch"] is None
            else PrefetchedBatch.from_mapping(mapping["prefetched_batch"]),
            totals=ExposureTotals.from_mapping(mapping["totals"]),
            rng=RNGSnapshot.from_mapping(mapping["rng"]),
            resume_lineage=tuple(raw_lineage),
            canonical_sha256=mapping["canonical_sha256"],
        )
        # Also checks that the typed representation did not coerce any field.
        if state.to_dict() != dict(mapping):
            raise ExposureError("loader state is not in canonical typed form")
        return state

    def assert_resume_compatible(
        self,
        *,
        loader_kind: str,
        source_dataset_manifest_sha256: str,
        study_sha256: str,
        tokenizer_sha256: str,
        exposure_manifest_sha256: str | None,
        rank: int,
        world_size: int,
    ) -> None:
        """Reject resume under any changed immutable loader identity."""

        expected = {
            "loader_kind": loader_kind,
            "source_dataset_manifest_sha256": source_dataset_manifest_sha256,
            "study_sha256": study_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "exposure_manifest_sha256": exposure_manifest_sha256,
            "rank": rank,
            "world_size": world_size,
        }
        for name, wanted in expected.items():
            if getattr(self, name) != wanted:
                raise ExposureError(
                    f"resume binding mismatch for {name}: "
                    f"state={getattr(self, name)!r}, requested={wanted!r}"
                )


def write_json_new_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Publish deterministic JSON atomically and refuse every overwrite race."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Validate canonical JSON support before creating any filesystem artifact.
    canonical_json_bytes(value)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard link publishes the complete inode and, unlike replace,
            # fails atomically when another process already created the path.
            os.link(temporary_name, destination)
        except FileExistsError as exc:
            raise ExposureError(f"refusing to overwrite existing file: {destination}") from exc
        Path(temporary_name).unlink()
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
    "BufferCursor",
    "EXPOSURE_MANIFEST_TYPE",
    "EXPOSURE_SCHEMA_VERSION",
    "ExposureError",
    "ExposureTotals",
    "IMPLEMENTATION_SCOPE",
    "LoaderPosition",
    "PrefetchedBatch",
    "RNGSnapshot",
    "ResumableLoaderState",
    "SourceDocument",
    "build_exposure_manifest",
    "build_training_exposure_plan",
    "validate_exposure_manifest",
    "validate_training_exposure_plan",
    "validate_training_plan_pair",
    "verify_exposure_source",
    "write_json_new_atomic",
]
