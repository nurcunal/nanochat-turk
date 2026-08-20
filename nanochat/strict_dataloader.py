"""Exact-resume Parquet loaders for the strict Turkish WSD lane.

The pinned upstream dataloader remains untouched.  The training loader below
preserves its BOS best-fit *cropping* semantics and row-group rank sharding;
the additive behavior is a byte-bound source inventory and an exact,
serializable cursor/buffer.  The validation loader is a separate finite,
whole-document contract.
"""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
import torch

from nanochat.common import get_dist_info
from nanochat.experiment_manifest import (
    canonical_json_bytes,
    load_json_strict,
    seal_manifest,
    validate_dataset_manifest,
    verify_file_inventory,
    verify_manifest_hash,
)
from nanochat.exposure import (
    BufferCursor,
    ExposureError,
    ExposureTotals,
    LoaderPosition,
    RNGSnapshot,
    ResumableLoaderState,
    validate_exposure_manifest,
    validate_training_exposure_plan,
)

MANIFEST_FILE = "fineweb2_manifest.json"
PINNED_UPSTREAM_REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
UPSTREAM_BESTFIT_STATE_TYPE = "strict_upstream_bos_bestfit_loader_state"


def verify_strict_dataset(
    data_dir: str | Path, *, verify_bytes: bool = True
) -> dict[str, Any]:
    """Validate the sealed dataset and optionally hash every recorded file.

    Production orchestration performs the byte pass on rank 0 once and
    broadcasts this canonical contract.  Loader instances on every rank then
    revalidate the manifest identity without repeatedly hashing the entire
    corpus at each validation interval.
    """

    root = Path(data_dir).resolve()
    manifest = load_json_strict(root / MANIFEST_FILE)
    if not isinstance(manifest, dict):
        raise ExposureError("strict dataset manifest must be a JSON object")
    try:
        validate_dataset_manifest(manifest, profile="strict")
        manifest_hash = verify_manifest_hash(manifest)
        if verify_bytes:
            verify_file_inventory(root, manifest["ordered_files"])
    except (OSError, ValueError) as exc:
        raise ExposureError(f"invalid strict dataset manifest: {exc}") from exc
    ordered_relative = [record["path"] for record in manifest["ordered_files"]]
    validation_path = manifest["validation_file"]
    if ordered_relative != sorted(ordered_relative):
        raise ExposureError(
            "strict dataset inventory must equal pinned upstream filename order"
        )
    if ordered_relative[-1] != validation_path:
        raise ExposureError(
            "strict validation file must be the final pinned-upstream parquet"
        )
    train_relative = [path for path in ordered_relative if path != validation_path]
    if not train_relative:
        raise ExposureError("strict training split has no Parquet files")
    return {
        "root": str(root),
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "ordered_relative": ordered_relative,
        "train_relative": train_relative,
    }


def _verified_dataset_contract(
    data_dir: str | Path,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return verify_strict_dataset(data_dir, verify_bytes=True)
    required = {
        "root",
        "manifest",
        "manifest_sha256",
        "ordered_relative",
        "train_relative",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ExposureError("verified dataset contract has invalid fields")
    contract = dict(value)
    if contract["root"] != str(Path(data_dir).resolve()):
        raise ExposureError("verified dataset contract root mismatch")
    manifest = contract["manifest"]
    if not isinstance(manifest, dict):
        raise ExposureError("verified dataset manifest must be an object")
    try:
        validate_dataset_manifest(manifest, profile="strict")
        digest = verify_manifest_hash(manifest)
    except ValueError as exc:
        raise ExposureError(f"invalid verified dataset contract: {exc}") from exc
    if digest != contract["manifest_sha256"]:
        raise ExposureError("verified dataset contract hash mismatch")
    ordered = [record["path"] for record in manifest["ordered_files"]]
    train = [path for path in ordered if path != manifest["validation_file"]]
    if ordered != sorted(ordered) or ordered[-1] != manifest["validation_file"]:
        raise ExposureError("verified dataset does not preserve pinned upstream split order")
    if contract["ordered_relative"] != ordered or contract["train_relative"] != train:
        raise ExposureError("verified dataset contract inventory mismatch")
    return contract

# Strict paper-study loaders


@dataclass(frozen=True)
class _EncodedDocument:
    document_id: str
    source_path: str
    file_index: int
    row_group_index: int
    row_index: int
    epoch: int
    payload_bytes: int
    token_ids: tuple[int, ...]
    token_byte_lengths: tuple[int, ...]
    token_offset: int = 0
    payload_byte_offset: int = 0


def _tensor_state_b64(value: torch.Tensor) -> str:
    return base64.b64encode(value.detach().cpu().numpy().tobytes()).decode("ascii")


def capture_loader_rng_snapshot(device: str | torch.device = "cpu") -> RNGSnapshot:
    """Capture rank-local RNG state in the loader-state wire format."""

    python_state = base64.b64encode(
        pickle.dumps(random.getstate(), protocol=5)
    ).decode("ascii")
    cpu_state = _tensor_state_b64(torch.get_rng_state())
    resolved = torch.device(device)
    cuda_states: tuple[str, ...] = ()
    if resolved.type == "cuda" and torch.cuda.is_available():
        cuda_states = (_tensor_state_b64(torch.cuda.get_rng_state(resolved)),)
    return RNGSnapshot(python_state, cpu_state, cuda_states)


def _uint8_state_from_b64(value: str) -> torch.Tensor:
    payload = base64.b64decode(value, validate=True)
    # ``frombuffer`` is read-only for immutable bytes on some PyTorch builds.
    return torch.tensor(list(payload), dtype=torch.uint8)


def restore_loader_rng_snapshot(
    snapshot: RNGSnapshot, device: str | torch.device = "cpu"
) -> None:
    """Restore the Python, CPU, and optional rank-local CUDA RNG states."""

    try:
        random.setstate(pickle.loads(base64.b64decode(snapshot.python_state_b64)))
    except Exception as exc:
        raise ExposureError("loader Python RNG snapshot is not restorable") from exc
    torch.set_rng_state(_uint8_state_from_b64(snapshot.cpu_state_b64))
    resolved = torch.device(device)
    if snapshot.cuda_state_b64:
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise ExposureError("CUDA loader RNG state cannot be restored on this device")
        torch.cuda.set_rng_state(
            _uint8_state_from_b64(snapshot.cuda_state_b64[0]), resolved
        )


def _token_byte_table(token_bytes: torch.Tensor | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(token_bytes, torch.Tensor):
        if token_bytes.ndim != 1:
            raise ExposureError("token_bytes must be one-dimensional")
        values = token_bytes.detach().cpu().tolist()
    else:
        values = list(token_bytes)
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExposureError(f"token_bytes[{index}] must be a non-negative integer")
        result.append(value)
    if not result:
        raise ExposureError("token_bytes must not be empty")
    return tuple(result)


def _encode_document(
    tokenizer,
    token_byte_lengths: tuple[int, ...],
    *,
    text: str,
    document_id: str,
    source_path: str,
    file_index: int,
    row_group_index: int,
    row_index: int,
    epoch: int,
) -> _EncodedDocument:
    try:
        raw_ids = tokenizer.encode(text)
    except Exception as exc:
        raise ExposureError(f"tokenization failed for {document_id}: {exc}") from exc
    if not isinstance(raw_ids, (list, tuple)):
        raise ExposureError("tokenizer.encode(text) must return one token-ID sequence")
    token_ids: list[int] = []
    lengths: list[int] = []
    for offset, token_id in enumerate(raw_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ExposureError(f"{document_id}: token {offset} is not an integer")
        if token_id < 0 or token_id >= len(token_byte_lengths):
            raise ExposureError(f"{document_id}: token ID {token_id} is outside token_bytes")
        length = token_byte_lengths[token_id]
        if length <= 0:
            raise ExposureError(
                f"{document_id}: payload encoded to zero-byte/special token {token_id}"
            )
        token_ids.append(token_id)
        lengths.append(length)
    payload = text.encode("utf-8", errors="strict")
    if sum(lengths) != len(payload):
        raise ExposureError(
            f"{document_id}: token byte accounting differs from UTF-8 payload"
        )
    try:
        decoded = tokenizer.decode(token_ids)
    except Exception as exc:
        raise ExposureError(f"decode failed for {document_id}: {exc}") from exc
    if decoded != text:
        raise ExposureError(f"{document_id}: tokenizer round trip is not lossless")
    return _EncodedDocument(
        document_id=document_id,
        source_path=source_path,
        file_index=file_index,
        row_group_index=row_group_index,
        row_index=row_index,
        epoch=epoch,
        payload_bytes=len(payload),
        token_ids=tuple(token_ids),
        token_byte_lengths=tuple(lengths),
    )


def _encoded_document_from_upstream_tokens(
    token_ids_with_bos: list[int] | tuple[int, ...],
    token_byte_lengths: tuple[int, ...],
    *,
    bos_token: int,
    text: str,
    document_id: str,
    source_path: str,
    file_index: int,
    row_group_index: int,
    row_index: int,
    epoch: int,
) -> _EncodedDocument:
    """Validate one result from upstream's batched ``encode(..., prepend=BOS)``."""

    if not isinstance(token_ids_with_bos, (list, tuple)) or not token_ids_with_bos:
        raise ExposureError("batched tokenizer returned an invalid document token list")
    if token_ids_with_bos[0] != bos_token:
        raise ExposureError("batched tokenizer did not prepend the requested BOS token")
    token_ids: list[int] = []
    lengths: list[int] = []
    for offset, token_id in enumerate(token_ids_with_bos[1:]):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ExposureError(f"{document_id}: token {offset} is not an integer")
        if token_id < 0 or token_id >= len(token_byte_lengths):
            raise ExposureError(f"{document_id}: token ID {token_id} is outside token_bytes")
        length = token_byte_lengths[token_id]
        if length <= 0:
            raise ExposureError(
                f"{document_id}: payload encoded to zero-byte/special token {token_id}"
            )
        token_ids.append(token_id)
        lengths.append(length)
    payload = text.encode("utf-8", errors="strict")
    if sum(lengths) != len(payload):
        raise ExposureError(
            f"{document_id}: token byte accounting differs from UTF-8 payload"
        )
    return _EncodedDocument(
        document_id=document_id,
        source_path=source_path,
        file_index=file_index,
        row_group_index=row_group_index,
        row_index=row_index,
        epoch=epoch,
        payload_bytes=len(payload),
        token_ids=tuple(token_ids),
        token_byte_lengths=tuple(lengths),
    )


class _ParquetLocatorReader:
    def __init__(self, data_dir: str | Path, ordered_paths: list[str], text_column: str):
        self.root = Path(data_dir).resolve()
        self.ordered_paths = tuple(ordered_paths)
        self.file_index = {path: index for index, path in enumerate(self.ordered_paths)}
        self.text_column = text_column
        self._cached_key: tuple[str, int] | None = None
        self._cached_values: list[Any] | None = None

    def _absolute(self, source_path: str) -> Path:
        path = (self.root / source_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ExposureError(f"source path escapes dataset directory: {source_path}") from exc
        if source_path not in self.file_index:
            raise ExposureError(f"source path is not in strict dataset inventory: {source_path}")
        return path

    def read(self, source_path: str, row_group_index: int, row_index: int) -> str:
        key = (source_path, row_group_index)
        if key != self._cached_key:
            path = self._absolute(source_path)
            parquet_file = pq.ParquetFile(path)
            if self.text_column not in parquet_file.schema_arrow.names:
                raise ExposureError(
                    f"missing text column {self.text_column!r} in {source_path}"
                )
            if row_group_index < 0 or row_group_index >= parquet_file.num_row_groups:
                raise ExposureError(f"row group is outside {source_path}")
            self._cached_values = parquet_file.read_row_group(
                row_group_index, columns=[self.text_column]
            ).column(self.text_column).to_pylist()
            self._cached_key = key
        assert self._cached_values is not None
        if row_index < 0 or row_index >= len(self._cached_values):
            raise ExposureError(f"row index is outside {source_path} row group")
        value = self._cached_values[row_index]
        if not isinstance(value, str):
            raise ExposureError("strict dataset text payload is not a string")
        return value


def _load_strict_dataset(
    data_dir: str | Path,
    dataset_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str, list[str], list[str]]:
    root = Path(data_dir).resolve()
    contract = _verified_dataset_contract(data_dir, dataset_contract)
    manifest = contract["manifest"]
    manifest_hash = contract["manifest_sha256"]
    ordered_relative = contract["ordered_relative"]
    train_relative = contract["train_relative"]
    train_absolute = [str((root / path).resolve()) for path in train_relative]
    return manifest, manifest_hash, ordered_relative, train_absolute


def _coerce_resume_state(
    value: ResumableLoaderState | Mapping[str, Any] | None,
) -> ResumableLoaderState | None:
    if value is None:
        return None
    if isinstance(value, ResumableLoaderState):
        # Round-trip through the sealed representation even for typed callers.
        return ResumableLoaderState.from_mapping(value.to_dict())
    return ResumableLoaderState.from_mapping(value)




@dataclass(frozen=True)
class ValidationRowSet:
    """World-size-independent complete-document validation rows."""

    inputs: torch.Tensor
    targets: torch.Tensor
    target_byte_lengths: torch.Tensor
    exposure_manifest_sha256: str
    source_dataset_manifest_sha256: str
    study_sha256: str
    tokenizer_sha256: str
    sequence_length: int
    documents: int
    target_tokens: int
    payload_bytes: int
    layout_sha256: str

    def __post_init__(self) -> None:
        shape = self.inputs.shape
        if (
            self.inputs.dtype != torch.long
            or self.targets.dtype != torch.long
            or self.target_byte_lengths.dtype != torch.long
            or self.targets.shape != shape
            or self.target_byte_lengths.shape != shape
            or len(shape) != 2
            or shape[1] != self.sequence_length
        ):
            raise ExposureError("validation row tensors have an invalid layout")
        if int((self.targets >= 0).sum().item()) != self.target_tokens:
            raise ExposureError("validation target-token accounting mismatch")
        if int(self.target_byte_lengths.sum().item()) != self.payload_bytes:
            raise ExposureError("validation payload-byte accounting mismatch")


def _validation_layout_sha256(
    inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"shape": list(inputs.shape), "dtype": "int64"}))
    for tensor in (inputs, targets, lengths):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_validation_rows(
    tokenizer,
    *,
    exposure_manifest: Mapping[str, Any],
    data_dir: str | Path,
    token_bytes: torch.Tensor | list[int] | tuple[int, ...],
    study_sha256: str,
    tokenizer_sha256: str,
    sequence_length: int,
    dataset_contract: Mapping[str, Any] | None = None,
) -> ValidationRowSet:
    """Pack whole documents into fixed rows before any distributed sharding.

    Every exposed document must fit as ``BOS + payload`` in a row.  Documents
    are never cropped or split.  BOS targets between packed documents and all
    padding targets are masked, while every payload token is scored exactly
    once.  This gives each target the same row/context at ws1, ws8, or ws16.
    """

    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise ExposureError("sequence_length must be a positive integer")
    token_byte_lengths = _token_byte_table(token_bytes)
    bos_token = tokenizer.get_bos_token_id()
    dataset, dataset_hash, ordered_relative, _ = _load_strict_dataset(
        data_dir, dataset_contract
    )
    try:
        validate_exposure_manifest(exposure_manifest, source_dataset_manifest=dataset)
    except ValueError as exc:
        raise ExposureError(f"invalid validation exposure manifest: {exc}") from exc
    if exposure_manifest.get("mode") != "validation":
        raise ExposureError("validation rows require a validation exposure manifest")
    if exposure_manifest.get("study_sha256") not in {None, study_sha256}:
        raise ExposureError("validation exposure is bound to another study")
    file_positions = {path: index for index, path in enumerate(ordered_relative)}
    reader = _ParquetLocatorReader(
        data_dir, ordered_relative, exposure_manifest["text_column"]
    )
    documents: list[_EncodedDocument] = []
    for record in exposure_manifest["documents"]:
        full_text = reader.read(
            record["source_path"], record["row_group_index"], record["row_index"]
        )
        payload = full_text.encode("utf-8", errors="strict")
        if (
            len(payload) != record["payload_bytes"]
            or hashlib.sha256(payload).hexdigest() != record["payload_sha256"]
        ):
            raise ExposureError(f"validation source payload drift for {record['document_id']}")
        included = payload[: record["included_bytes"]]
        try:
            text = included.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ExposureError("validation prefix is not on a UTF-8 boundary") from exc
        if hashlib.sha256(included).hexdigest() != record["included_sha256"]:
            raise ExposureError(f"validation included payload drift for {record['document_id']}")
        encoded = _encode_document(
            tokenizer,
            token_byte_lengths,
            text=text,
            document_id=record["document_id"],
            source_path=record["source_path"],
            file_index=file_positions[record["source_path"]],
            row_group_index=record["row_group_index"],
            row_index=record["row_index"],
            epoch=0,
        )
        if len(encoded.token_ids) > sequence_length:
            raise ExposureError(
                "validation document exceeds sealed whole_document_no_crop limit"
            )
        documents.append(encoded)

    row_inputs: list[list[int]] = []
    row_targets: list[list[int]] = []
    row_lengths: list[list[int]] = []
    packed_tokens: list[int] = []
    packed_lengths: list[int] = []

    def flush_row() -> None:
        nonlocal packed_tokens, packed_lengths
        if len(packed_tokens) >= 2:
            sources = packed_tokens[:-1]
            targets = packed_tokens[1:]
            lengths = packed_lengths[1:]
            for index, target in enumerate(targets):
                if target == bos_token:
                    targets[index] = -1
                    lengths[index] = 0
            padding = sequence_length - len(targets)
            row_inputs.append([*sources, *([bos_token] * padding)])
            row_targets.append([*targets, *([-1] * padding)])
            row_lengths.append([*lengths, *([0] * padding)])
        packed_tokens = []
        packed_lengths = []

    for document in documents:
        if not document.token_ids:
            continue
        segment_tokens = [bos_token, *document.token_ids]
        segment_lengths = [0, *document.token_byte_lengths]
        if packed_tokens and len(packed_tokens) + len(segment_tokens) > sequence_length + 1:
            flush_row()
        packed_tokens.extend(segment_tokens)
        packed_lengths.extend(segment_lengths)
    flush_row()
    if not row_inputs:
        raise ExposureError("validation exposure produced no target rows")
    inputs = torch.tensor(row_inputs, dtype=torch.long)
    targets = torch.tensor(row_targets, dtype=torch.long)
    lengths = torch.tensor(row_lengths, dtype=torch.long)
    target_tokens = int((targets >= 0).sum().item())
    payload_bytes = int(lengths.sum().item())
    expected_bytes = exposure_manifest["selection"]["realized_payload_bytes"]
    if payload_bytes != expected_bytes:
        raise ExposureError("validation rows do not cover every exposed payload byte")
    return ValidationRowSet(
        inputs=inputs,
        targets=targets,
        target_byte_lengths=lengths,
        exposure_manifest_sha256=exposure_manifest["canonical_sha256"],
        source_dataset_manifest_sha256=dataset_hash,
        study_sha256=study_sha256,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        documents=exposure_manifest["selection"]["realized_documents"],
        target_tokens=target_tokens,
        payload_bytes=payload_bytes,
        layout_sha256=_validation_layout_sha256(inputs, targets, lengths),
    )


class StatefulSequentialDocumentLoader:
    """Finite complete-row sharding over one topology-invariant layout."""

    def __init__(
        self,
        tokenizer,
        B: int,
        T: int,
        *,
        exposure_manifest: Mapping[str, Any],
        data_dir: str | Path,
        token_bytes: torch.Tensor | list[int] | tuple[int, ...],
        study_sha256: str,
        tokenizer_sha256: str,
        device: str | torch.device = "cpu",
        rank: int | None = None,
        world_size: int | None = None,
        dataset_contract: Mapping[str, Any] | None = None,
        prepared_rows: ValidationRowSet | None = None,
        resume_state: Mapping[str, Any] | None = None,
        restore_rng: bool = True,
    ) -> None:
        del restore_rng
        if resume_state is not None:
            raise ExposureError("validation is rebuilt from immutable rows, not resumed")
        if min(B, T) <= 0:
            raise ExposureError("validation B and T must be positive")
        _ddp, detected_rank, _local_rank, detected_world = get_dist_info()
        self.rank = detected_rank if rank is None else rank
        self.world_size = detected_world if world_size is None else world_size
        if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
            raise ExposureError("invalid validation rank/world_size")
        self.B, self.T = B, T
        self.device = torch.device(device)
        self.bos_token = tokenizer.get_bos_token_id()
        self.rows = prepared_rows or build_validation_rows(
            tokenizer,
            exposure_manifest=exposure_manifest,
            data_dir=data_dir,
            token_bytes=token_bytes,
            study_sha256=study_sha256,
            tokenizer_sha256=tokenizer_sha256,
            sequence_length=T,
            dataset_contract=dataset_contract,
        )
        expected = {
            "exposure_manifest_sha256": exposure_manifest["canonical_sha256"],
            "study_sha256": study_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "sequence_length": T,
        }
        for field, wanted in expected.items():
            if getattr(self.rows, field) != wanted:
                raise ExposureError(f"prepared validation rows mismatch at {field}")
        self.next_batch_index = 0
        self.total_batches = (
            len(self.rows.inputs) + B * self.world_size - 1
        ) // (B * self.world_size)
        self.valid_target_tokens = 0
        self.payload_bytes = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch_index >= self.total_batches:
            raise StopIteration
        global_start = self.next_batch_index * self.B * self.world_size
        rank_start = global_start + self.rank * self.B
        inputs = torch.full((self.B, self.T), self.bos_token, dtype=torch.long)
        targets = torch.full((self.B, self.T), -1, dtype=torch.long)
        lengths = torch.zeros((self.B, self.T), dtype=torch.long)
        available = max(0, min(self.B, len(self.rows.inputs) - rank_start))
        if available:
            inputs[:available].copy_(
                self.rows.inputs[rank_start : rank_start + available]
            )
            targets[:available].copy_(
                self.rows.targets[rank_start : rank_start + available]
            )
            lengths[:available].copy_(
                self.rows.target_byte_lengths[rank_start : rank_start + available]
            )
        self.valid_target_tokens += int((targets >= 0).sum().item())
        self.payload_bytes += int(lengths.sum().item())
        self.next_batch_index += 1
        state = {
            "layout_sha256": self.rows.layout_sha256,
            "next_batch_index": self.next_batch_index,
            "total_batches": self.total_batches,
            "valid_target_tokens": self.valid_target_tokens,
            "payload_bytes": self.payload_bytes,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        return inputs.to(self.device), targets.to(self.device), state


def _bestfit_document_id(
    *, source_path: str, row_group_index: int, row_index: int, epoch: int, text: str
) -> str:
    locator = canonical_json_bytes(
        {
            "source_path": source_path,
            "row_group_index": row_group_index,
            "row_index": row_index,
            "epoch": epoch,
        }
    )
    encoded = base64.urlsafe_b64encode(locator).decode("ascii")
    return f"bestfit1.{encoded}.{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _parse_bestfit_document_id(document_id: str) -> tuple[dict[str, Any], str]:
    try:
        prefix, encoded, payload_hash = document_id.split(".", 2)
        if prefix != "bestfit1" or len(payload_hash) != 64:
            raise ValueError
        locator = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    except Exception as exc:
        raise ExposureError("invalid bestfit buffered document identity") from exc
    if set(locator) != {"source_path", "row_group_index", "row_index", "epoch"}:
        raise ExposureError("bestfit buffered document identity has invalid locator fields")
    return locator, payload_hash


def _source_order_sha256(
    train_relative_paths: list[str],
    *,
    world_size: int,
    tokenizer_batch_size: int,
    tokenizer_threads: int,
) -> str:
    """Identify the exact pinned-upstream source traversal, not a claimed shuffle."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "ordering_authority": "sealed_dataset_manifest_materialization_order",
                "rank_sharding": "row_group_start_rank_step_world_size",
                "train_relative_paths": train_relative_paths,
                "world_size": world_size,
                "tokenizer_batch_size": tokenizer_batch_size,
                "tokenizer_threads": tokenizer_threads,
                "upstream_revision": PINNED_UPSTREAM_REVISION,
            }
        )
    ).hexdigest()


def _validate_bestfit_exposure_plan(
    plan: Mapping[str, Any],
    *,
    dataset_sha256: str,
    study_sha256: str,
    tokenizer_sha256: str,
    world_size: int,
) -> dict[str, Any]:
    try:
        validate_training_exposure_plan(plan)
    except ValueError as exc:
        raise ExposureError(f"invalid training exposure plan: {exc}") from exc
    value = dict(plan)
    expected = {
        "estimand": "equal_token",
        "source_dataset_manifest_sha256": dataset_sha256,
        "study_sha256": study_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "world_size": world_size,
        "data_order": "bestfit",
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ExposureError(f"training exposure plan mismatch at {field}")
    return value


def _assert_plan_prefix_compatible(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    consumed_token_positions: int,
) -> None:
    """Allow a cooldown fork to rebind the same consumed data prefix."""

    for field in (
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
        "test_data_accessed",
        "implementation_scope",
    ):
        if parent.get(field) != child.get(field):
            raise ExposureError(f"parent exposure plan is incompatible at {field}")
    if child["horizon"]["value"] < consumed_token_positions:
        raise ExposureError("child exposure horizon precedes the consumed parent prefix")


def _coerce_bestfit_state(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExposureError("bestfit resume state must be an object")
    fields = {
        "schema_version",
        "manifest_type",
        "implementation_scope",
        "loader_kind",
        "upstream_revision",
        "source_dataset_manifest_sha256",
        "source_order_sha256",
        "study_sha256",
        "tokenizer_sha256",
        "exposure_plan_sha256",
        "exposure_plan_lineage",
        "rank",
        "world_size",
        "batch_sequences",
        "sequence_length",
        "buffer_size",
        "tokenizer_batch_size",
        "tokenizer_threads",
        "next_batch_index",
        "position",
        "buffer",
        "totals",
        "resume_lineage",
        "test_data_accessed",
        "canonical_sha256",
    }
    if set(value) != fields:
        raise ExposureError("bestfit resume state fields are not canonical")
    try:
        verify_manifest_hash(value)
    except ValueError as exc:
        raise ExposureError(f"invalid bestfit resume state hash: {exc}") from exc
    if (
        value["schema_version"] != "1.0"
        or value["manifest_type"] != UPSTREAM_BESTFIT_STATE_TYPE
        or value["implementation_scope"] != "strict_runtime_layer"
        or value["loader_kind"] != "upstream_bos_bestfit_crop_v1"
        or value["upstream_revision"] != PINNED_UPSTREAM_REVISION
        or value["test_data_accessed"] is not False
    ):
        raise ExposureError("bestfit resume state runtime identity mismatch")
    position = LoaderPosition.from_mapping(value["position"])
    buffer = value["buffer"]
    if not isinstance(buffer, list):
        raise ExposureError("bestfit resume buffer must be an array")
    parsed_buffer = tuple(BufferCursor.from_mapping(item) for item in buffer)
    if any(item.token_offset or item.payload_byte_offset for item in parsed_buffer):
        raise ExposureError("upstream crop semantics never retain document tails")
    totals = ExposureTotals.from_mapping(value["totals"])
    for name in ("exposure_plan_lineage", "resume_lineage"):
        lineage = value[name]
        if not isinstance(lineage, list) or len(lineage) != len(set(lineage)):
            raise ExposureError(f"{name} must be a unique hash array")
        if any(not isinstance(item, str) or len(item) != 64 for item in lineage):
            raise ExposureError(f"{name} contains an invalid SHA-256")
    parsed = dict(value)
    parsed["position"] = position
    parsed["buffer"] = parsed_buffer
    parsed["totals"] = totals
    return parsed


class _UpstreamRowGroupCursor:
    """Exact form of upstream ``_document_batches`` row-group sharding."""

    def __init__(
        self,
        reader: _ParquetLocatorReader,
        train_relative_paths: list[str],
        *,
        rank: int,
        world_size: int,
        position: LoaderPosition | None = None,
    ) -> None:
        self.reader = reader
        self.paths = tuple(train_relative_paths)
        if not self.paths:
            raise ExposureError("strict training split has no files")
        self.rank, self.world_size = rank, world_size
        self._metadata: dict[str, tuple[int, tuple[int, ...]]] = {}
        if position is None:
            self.file_index = 0
            self.row_group_index = rank
            self.row_index = 0
            self.document_ordinal = 0
            self.epoch = 1
        else:
            if position.file_index >= len(reader.ordered_paths):
                raise ExposureError("bestfit resume file index is outside the dataset")
            resume_path = reader.ordered_paths[position.file_index]
            if resume_path not in self.paths:
                raise ExposureError("bestfit resume points at held-out validation")
            if position.row_group_index % world_size != rank:
                raise ExposureError("bestfit resume row group belongs to another rank")
            self.file_index = self.paths.index(resume_path)
            self.row_group_index = position.row_group_index
            self.row_index = position.row_index
            self.document_ordinal = position.document_ordinal
            self.epoch = position.epoch
        self._normalize()

    def _shape(self, path: str) -> tuple[int, tuple[int, ...]]:
        if path not in self._metadata:
            parquet_file = pq.ParquetFile(self.reader._absolute(path))
            rows = tuple(
                parquet_file.metadata.row_group(index).num_rows
                for index in range(parquet_file.num_row_groups)
            )
            self._metadata[path] = (parquet_file.num_row_groups, rows)
        return self._metadata[path]

    def _normalize(self) -> None:
        while True:
            if self.file_index >= len(self.paths):
                self.file_index = 0
                self.row_group_index = self.rank
                self.row_index = 0
                self.document_ordinal = 0
                self.epoch += 1
            path = self.paths[self.file_index]
            num_groups, rows = self._shape(path)
            if self.row_group_index >= num_groups:
                self.file_index += 1
                self.row_group_index = self.rank
                self.row_index = 0
                continue
            if self.row_index >= rows[self.row_group_index]:
                self.row_group_index += self.world_size
                self.row_index = 0
                continue
            return

    def position(self) -> LoaderPosition:
        return LoaderPosition(
            self.reader.file_index[self.paths[self.file_index]],
            self.row_group_index,
            self.row_index,
            self.document_ordinal,
            self.epoch,
        )

    def next_tokenizer_batch(
        self, batch_size: int
    ) -> list[tuple[str, int, int, int, str]]:
        path = self.paths[self.file_index]
        row_group = self.row_group_index
        _num_groups, rows = self._shape(path)
        end = min(self.row_index + batch_size, rows[row_group])
        result = []
        while self.row_index < end:
            row = self.row_index
            result.append(
                (path, row_group, row, self.epoch, self.reader.read(path, row_group, row))
            )
            self.row_index += 1
            self.document_ordinal += 1
        self._normalize()
        return result


class StatefulBestFitLoader:
    """Pinned-upstream BOS best-fit cropping with exact rank-local resume."""

    def __init__(
        self,
        tokenizer,
        B: int,
        T: int,
        *,
        data_dir: str | Path,
        token_bytes: torch.Tensor | list[int] | tuple[int, ...],
        study_sha256: str,
        tokenizer_sha256: str,
        exposure_plan: Mapping[str, Any],
        parent_exposure_plan: Mapping[str, Any] | None = None,
        device: str | torch.device = "cpu",
        rank: int | None = None,
        world_size: int | None = None,
        buffer_size: int = 1000,
        tokenizer_batch_size: int = 128,
        tokenizer_threads: int = 4,
        dataset_contract: Mapping[str, Any] | None = None,
        resume_state: Mapping[str, Any] | None = None,
        restore_rng: bool = True,
    ) -> None:
        del restore_rng  # The upstream best-fit loader is deterministic and RNG-free.
        if min(B, T, buffer_size, tokenizer_batch_size, tokenizer_threads) <= 0:
            raise ExposureError("bestfit dimensions must be positive")
        _ddp, detected_rank, _local, detected_world = get_dist_info()
        self.rank = detected_rank if rank is None else rank
        self.world_size = detected_world if world_size is None else world_size
        if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
            raise ExposureError("invalid rank/world_size")
        self.B, self.T = B, T
        self.buffer_size = buffer_size
        self.tokenizer_batch_size = tokenizer_batch_size
        self.tokenizer_threads = tokenizer_threads
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.get_bos_token_id()
        self.token_byte_lengths = _token_byte_table(token_bytes)
        dataset, dataset_hash, ordered_relative, train_absolute = _load_strict_dataset(
            data_dir, dataset_contract
        )
        self.dataset_hash = dataset_hash
        self.study_sha256 = study_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self.exposure_plan = _validate_bestfit_exposure_plan(
            exposure_plan,
            dataset_sha256=dataset_hash,
            study_sha256=study_sha256,
            tokenizer_sha256=tokenizer_sha256,
            world_size=self.world_size,
        )
        self.exposure_plan_sha256 = self.exposure_plan["canonical_sha256"]
        root = Path(data_dir).resolve()
        train_relative = [
            Path(path).resolve().relative_to(root).as_posix()
            for path in train_absolute
        ]
        self.source_order_sha256 = _source_order_sha256(
            train_relative,
            world_size=self.world_size,
            tokenizer_batch_size=self.tokenizer_batch_size,
            tokenizer_threads=self.tokenizer_threads,
        )
        self.reader = _ParquetLocatorReader(
            data_dir, ordered_relative, dataset["text_column"]
        )
        restored = _coerce_bestfit_state(resume_state)
        position = None if restored is None else restored["position"]
        self.cursor = _UpstreamRowGroupCursor(
            self.reader,
            train_relative,
            rank=self.rank,
            world_size=self.world_size,
            position=position,
        )
        self.buffer: list[_EncodedDocument] = []
        self.next_batch_index = 0
        self.totals = ExposureTotals()
        self.resume_lineage: tuple[str, ...] = ()
        self.exposure_plan_lineage: tuple[str, ...] = ()
        # Preserve pinned upstream's persistent staging layout and single HtoD
        # transfer.  Byte lengths stay in a parallel CPU-only accounting row.
        self._use_cuda = self.device.type == "cuda"
        self._row_buffer = torch.empty((B, T + 1), dtype=torch.long)
        self._row_length_buffer = torch.empty((B, T + 1), dtype=torch.long)
        self._cpu_buffer = torch.empty(
            2 * B * T, dtype=torch.long, pin_memory=self._use_cuda
        )
        self._cpu_inputs = self._cpu_buffer[: B * T].view(B, T)
        self._cpu_targets = self._cpu_buffer[B * T :].view(B, T)
        self._device_buffer = torch.empty(
            2 * B * T, dtype=torch.long, device=self.device
        )
        self._device_inputs = self._device_buffer[: B * T].view(B, T)
        self._device_targets = self._device_buffer[B * T :].view(B, T)
        if restored is not None:
            expected = {
                "source_dataset_manifest_sha256": dataset_hash,
                "source_order_sha256": self.source_order_sha256,
                "study_sha256": study_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "rank": self.rank,
                "world_size": self.world_size,
                "batch_sequences": B,
                "sequence_length": T,
                "buffer_size": buffer_size,
                "tokenizer_batch_size": tokenizer_batch_size,
                "tokenizer_threads": tokenizer_threads,
            }
            for field, wanted in expected.items():
                if restored[field] != wanted:
                    raise ExposureError(f"bestfit resume mismatch at {field}")
            old_plan_hash = restored["exposure_plan_sha256"]
            consumed = restored["next_batch_index"] * B * T * self.world_size
            if old_plan_hash != self.exposure_plan_sha256:
                if parent_exposure_plan is None:
                    raise ExposureError("resume state is bound to another exposure plan")
                parent = _validate_bestfit_exposure_plan(
                    parent_exposure_plan,
                    dataset_sha256=dataset_hash,
                    study_sha256=study_sha256,
                    tokenizer_sha256=tokenizer_sha256,
                    world_size=self.world_size,
                )
                if parent["canonical_sha256"] != old_plan_hash:
                    raise ExposureError("parent exposure plan does not match resume state")
                _assert_plan_prefix_compatible(
                    parent,
                    self.exposure_plan,
                    consumed_token_positions=consumed,
                )
                self.exposure_plan_lineage = (
                    *restored["exposure_plan_lineage"],
                    old_plan_hash,
                )
            else:
                self.exposure_plan_lineage = tuple(restored["exposure_plan_lineage"])
            self.next_batch_index = restored["next_batch_index"]
            self.totals = restored["totals"]
            self.resume_lineage = (
                *restored["resume_lineage"],
                restored["canonical_sha256"],
            )
            for cursor in restored["buffer"]:
                self.buffer.append(self._reload_buffered_cursor(cursor))
        if self.exposure_plan["horizon"]["value"] < (
            self.next_batch_index * B * T * self.world_size
        ):
            raise ExposureError("exposure plan horizon precedes loader state")

    def __iter__(self):
        return self

    def _reload_buffered_cursor(self, cursor: BufferCursor) -> _EncodedDocument:
        if cursor.token_offset or cursor.payload_byte_offset:
            raise ExposureError("upstream cropping cannot resume a retained tail")
        locator, payload_hash = _parse_bestfit_document_id(cursor.document_id)
        text = self.reader.read(
            locator["source_path"], locator["row_group_index"], locator["row_index"]
        )
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != payload_hash:
            raise ExposureError("bestfit buffered source payload drift")
        return _encode_document(
            self.tokenizer,
            self.token_byte_lengths,
            text=text,
            document_id=cursor.document_id,
            source_path=locator["source_path"],
            file_index=self.reader.file_index[locator["source_path"]],
            row_group_index=locator["row_group_index"],
            row_index=locator["row_index"],
            epoch=locator["epoch"],
        )

    def _refill_buffer(self) -> None:
        records = self.cursor.next_tokenizer_batch(self.tokenizer_batch_size)
        texts = [record[4] for record in records]
        try:
            token_lists = self.tokenizer.encode(
                texts,
                prepend=self.bos_token,
                num_threads=self.tokenizer_threads,
            )
        except Exception as exc:
            raise ExposureError(f"upstream batched tokenization failed: {exc}") from exc
        if not isinstance(token_lists, (list, tuple)) or len(token_lists) != len(records):
            raise ExposureError("batched tokenizer returned the wrong document count")
        values = self.totals.to_dict()
        for (path, row_group, row, epoch, text), token_ids in zip(
            records, token_lists, strict=True
        ):
            document_id = _bestfit_document_id(
                source_path=path,
                row_group_index=row_group,
                row_index=row,
                epoch=epoch,
                text=text,
            )
            encoded = _encoded_document_from_upstream_tokens(
                token_ids,
                self.token_byte_lengths,
                bos_token=self.bos_token,
                text=text,
                document_id=document_id,
                source_path=path,
                file_index=self.reader.file_index[path],
                row_group_index=row_group,
                row_index=row,
                epoch=epoch,
            )
            self.buffer.append(encoded)
            values["source_bytes_loaded"] += encoded.payload_bytes
            values["documents_loaded"] += 1
        self.totals = ExposureTotals.from_mapping(values)

    def state(self) -> dict[str, Any]:
        return seal_manifest(
            {
                "schema_version": "1.0",
                "manifest_type": UPSTREAM_BESTFIT_STATE_TYPE,
                "implementation_scope": "strict_runtime_layer",
                "loader_kind": "upstream_bos_bestfit_crop_v1",
                "upstream_revision": PINNED_UPSTREAM_REVISION,
                "source_dataset_manifest_sha256": self.dataset_hash,
                "source_order_sha256": self.source_order_sha256,
                "study_sha256": self.study_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "exposure_plan_sha256": self.exposure_plan_sha256,
                "exposure_plan_lineage": list(self.exposure_plan_lineage),
                "rank": self.rank,
                "world_size": self.world_size,
                "batch_sequences": self.B,
                "sequence_length": self.T,
                "buffer_size": self.buffer_size,
                "tokenizer_batch_size": self.tokenizer_batch_size,
                "tokenizer_threads": self.tokenizer_threads,
                "next_batch_index": self.next_batch_index,
                "position": self.cursor.position().to_dict(),
                "buffer": [
                    BufferCursor(item.document_id, 0, 0).to_dict()
                    for item in self.buffer
                ],
                "totals": self.totals.to_dict(),
                "resume_lineage": list(self.resume_lineage),
                "test_data_accessed": False,
                "canonical_sha256": None,
            }
        )

    def __next__(self):
        row_capacity = self.T + 1
        row_tokens = self._row_buffer
        row_lengths = self._row_length_buffer
        row_lengths.zero_()
        for row_index in range(self.B):
            position = 0
            while position < row_capacity:
                # Pinned upstream refills a full tokenizer chunk and may
                # overshoot ``buffer_size``; this is semantically significant.
                while len(self.buffer) < self.buffer_size:
                    self._refill_buffer()
                remaining = row_capacity - position
                best_index = -1
                best_length = 0
                for index, document in enumerate(self.buffer):
                    document_length = len(document.token_ids) + 1
                    if document_length <= remaining and document_length > best_length:
                        best_index = index
                        best_length = document_length
                if best_index >= 0:
                    selected = best_index
                else:
                    selected = min(
                        range(len(self.buffer)),
                        key=lambda index: len(self.buffer[index].token_ids) + 1,
                    )
                document = self.buffer.pop(selected)
                tokens = (self.bos_token, *document.token_ids)
                lengths = (0, *document.token_byte_lengths)
                used = min(remaining, len(tokens))
                row_tokens[row_index, position : position + used] = torch.tensor(
                    tokens[:used], dtype=torch.long
                )
                row_lengths[row_index, position : position + used] = torch.tensor(
                    lengths[:used], dtype=torch.long
                )
                position += used
                consumed_payload_tokens = max(0, used - 1)
                values = self.totals.to_dict()
                values["documents_started"] += 1
                values["documents_completed"] += 1
                if used < len(tokens):
                    values["documents_cropped"] += 1
                    values["discarded_tokens"] += (
                        len(document.token_ids) - consumed_payload_tokens
                    )
                    values["discarded_bytes"] += sum(
                        document.token_byte_lengths[consumed_payload_tokens:]
                    )
                self.totals = ExposureTotals.from_mapping(values)
        target_lengths = row_lengths[:, 1:]
        values = self.totals.to_dict()
        values["token_positions"] += self.B * self.T
        values["valid_target_tokens"] += self.B * self.T
        values["payload_bytes"] += int(target_lengths.sum().item())
        self.totals = ExposureTotals.from_mapping(values)
        self.next_batch_index += 1
        state = self.state()
        self._cpu_inputs.copy_(row_tokens[:, :-1])
        self._cpu_targets.copy_(row_tokens[:, 1:])
        self._device_buffer.copy_(self._cpu_buffer, non_blocking=self._use_cuda)
        return self._device_inputs, self._device_targets, state


def finite_sequential_document_loader(*args, **kwargs):
    """Convenience constructor used for fixed-text validation evaluation."""

    return StatefulSequentialDocumentLoader(*args, **kwargs)


def measure_validation_coverage(
    tokenizer,
    *,
    exposure_manifest: Mapping[str, Any],
    data_dir: str | Path,
    token_bytes: torch.Tensor | list[int] | tuple[int, ...],
    study_sha256: str,
    tokenizer_sha256: str,
    dataset_contract: Mapping[str, Any],
    sequence_length: int,
) -> dict[str, Any]:
    """Tokenize the frozen validation set once and return exact coverage.

    The result is tokenizer-specific identity material.  Training binds it into
    every checkpoint and later requires each distributed evaluation to produce
    the same target-token and UTF-8-byte totals.
    """

    rows = build_validation_rows(
        tokenizer,
        exposure_manifest=exposure_manifest,
        data_dir=data_dir,
        token_bytes=(
            token_bytes.detach().cpu()
            if isinstance(token_bytes, torch.Tensor)
            else token_bytes
        ),
        study_sha256=study_sha256,
        tokenizer_sha256=tokenizer_sha256,
        sequence_length=sequence_length,
        dataset_contract=dataset_contract,
    )
    expected_bytes = exposure_manifest["selection"]["realized_payload_bytes"]
    expected_documents = exposure_manifest["selection"]["realized_documents"]
    if rows.payload_bytes != expected_bytes:
        raise ExposureError("validation byte accounting differs from its manifest")
    if rows.documents != expected_documents:
        raise ExposureError("validation document accounting differs from its manifest")
    return {
        "target_tokens": rows.target_tokens,
        "payload_bytes": expected_bytes,
        "documents": expected_documents,
        "logical_rows": len(rows.inputs),
        "padded_token_positions_world1": len(rows.inputs) * sequence_length,
        "row_layout_sha256": rows.layout_sha256,
    }



__all__ = [
    "StatefulBestFitLoader",
    "StatefulSequentialDocumentLoader",
    "ValidationRowSet",
    "build_validation_rows",
    "capture_loader_rng_snapshot",
    "finite_sequential_document_loader",
    "measure_validation_coverage",
    "restore_loader_rng_snapshot",
    "verify_strict_dataset",
]
