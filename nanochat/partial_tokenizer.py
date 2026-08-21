"""Fail-closed verification for the frozen partial-18 Turkish tokenizer.

This module is intentionally separate from :mod:`nanochat.strict_tokenizer`.
The partial tokenizer is useful for the explicitly salvaged corpus lane, but
it did not pass global near-deduplication, sample clustering, or manual corpus
QA and therefore must never satisfy the production tokenizer contract.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
import torch

from nanochat.experiment_manifest import (
    ManifestValidationError,
    canonical_json,
    file_sha256,
    load_json_strict,
    validate_file_record,
    verify_file_inventory,
    verify_manifest_hash,
)
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN


PARTIAL_PACKAGE_KIND = "turkish_raw_bpe_partial_tokenizer_package"
PARTIAL_RECEIPT_KIND = "turkish_raw_bpe_partial_training_receipt"
PINNED_PACKAGE_SHA256 = (
    "909bfa20516c79b7349d3e35aacd655ef584aac431c055966366cf6e1545d871"
)
PINNED_RECEIPT_SHA256 = (
    "5014766c50fee069fde94806a9eb82de9c42e78d8d55632c4e37f2ebed94c445"
)
PINNED_INPUT_INVENTORY_SHA256 = (
    "0d67d69cc47128c1f966d3c67da3c85a158bc1e453dc12cd540e46ad8dd4cfa6"
)
PINNED_NAME = "tr_general_raw_bpe_32k_partial18_v1"
VOCAB_SIZE = 32768
EXPECTED_RANKS = (
    26,
    52,
    78,
    112,
    119,
    126,
    135,
    154,
    158,
    165,
    169,
    171,
    179,
    184,
    188,
    207,
    209,
    210,
)
MISSING_OBJECT_RECEIPT_RANKS = (112, 119, 126)
EXPECTED_FILES = (
    "tokenizer.pkl",
    "tokenizer.tiktoken",
    "token_bytes.pt",
    "tokenizer_config.json",
    "partial_training_receipt.json",
)
PAYLOAD_FILES = EXPECTED_FILES[:-1]
INCOMPLETE_GATES = (
    "global_near_dedup_completed",
    "sample_cluster_completed",
    "manual_corpus_qa_completed",
)
CANONICAL_EXPORT_FORMAT = (
    "tiktoken_bpe_base64_token_space_decimal_rank_newline"
)
REQUESTED_TRAINING_CHARACTERS = 2_000_000_000
MAX_CHARACTERS_PER_DOCUMENT = 10_000

_PACKAGE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "name",
        "vocab_size",
        "training_receipt_sha256",
        "input_inventory_sha256",
        "production_eligible",
        "files",
        "canonical_sha256",
    }
)
_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_PROBES = (
    "İstanbul'da bugün hava nasıl? Türkçe ğüşiöç karakterleri.",
    "Merhaba! Nasılsın? Ben iyiyim; teşekkür ederim. 😊",
    "TÜRKİYE Türkiye türkiye; IĞDIR Iğdır ığdır.",
    "sorumluluklarımızdakilerdenmişsinizcesine",
    "Ankara—İzmir 2026: %42,5 / 1.234,56 TL",
    "Kırgızca değil; Türkçe bir Unicode sınamasıdır: â, î, û.",
)


class PartialTokenizerPackageError(ValueError):
    """Raised when the partial tokenizer package is unsafe or inconsistent."""


@dataclass(frozen=True)
class VerifiedPartialTokenizerPackage:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    receipt: dict[str, Any]
    config: dict[str, Any]
    canonical_sha256: str
    receipt_sha256: str


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PartialTokenizerPackageError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PartialTokenizerPackageError(f"{name} must be a positive integer")
    return value


def _validate_file_records(records: Any) -> list[Mapping[str, Any]]:
    if not isinstance(records, list):
        raise PartialTokenizerPackageError("partial tokenizer files must be an array")
    if tuple(
        record.get("path") if isinstance(record, Mapping) else None
        for record in records
    ) != EXPECTED_FILES:
        raise PartialTokenizerPackageError(
            "partial tokenizer runtime inventory differs from the frozen package"
        )
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or frozenset(record) != _FILE_FIELDS:
            raise PartialTokenizerPackageError(
                f"files[{index}] must contain only path, size_bytes, and sha256"
            )
        try:
            validate_file_record(
                record,
                location=f"files[{index}]",
                allow_role=False,
            )
        except ManifestValidationError as exc:
            raise PartialTokenizerPackageError(
                f"invalid partial tokenizer file record: {exc}"
            ) from exc
    return records


def _validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PartialTokenizerPackageError("tokenizer_config.json must be an object")
    config = dict(value)
    if (
        config.get("schema_version") != "1.0"
        or config.get("name") != PINNED_NAME
        or config.get("implementation") != "raw_byte_bpe"
        or config.get("vocab_size") != VOCAB_SIZE
        or tuple(config.get("special_tokens", ())) != tuple(SPECIAL_TOKENS)
        or config.get("split_pattern") != SPLIT_PATTERN
        or config.get("requires_runtime_segmentation") is not False
        or config.get("decode_strip") != ""
        or config.get("production_eligible") is not False
    ):
        raise PartialTokenizerPackageError("partial tokenizer runtime config drifted")
    realized = _positive_integer(
        config.get("realized_training_characters"),
        "realized_training_characters",
    )
    if (
        config.get("max_chars") != REQUESTED_TRAINING_CHARACTERS
        or config.get("doc_cap") != MAX_CHARACTERS_PER_DOCUMENT
        or not REQUESTED_TRAINING_CHARACTERS
        < realized
        <= REQUESTED_TRAINING_CHARACTERS + MAX_CHARACTERS_PER_DOCUMENT
    ):
        raise PartialTokenizerPackageError(
            "partial tokenizer 2B-character training contract drifted"
        )
    return config


def _validate_input_inventory(receipt: Mapping[str, Any]) -> str:
    inventory = receipt.get("input_inventory")
    if not isinstance(inventory, list) or len(inventory) != len(EXPECTED_RANKS):
        raise PartialTokenizerPackageError("partial tokenizer input inventory drifted")
    ranks: list[int] = []
    missing_receipts: list[int] = []
    for index, item in enumerate(inventory):
        if not isinstance(item, Mapping):
            raise PartialTokenizerPackageError(
                f"input_inventory[{index}] must be an object"
            )
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise PartialTokenizerPackageError(
                f"input_inventory[{index}].rank must be an integer"
            )
        ranks.append(rank)
        for field in ("rows", "row_groups", "size_bytes"):
            _positive_integer(item.get(field), f"input_inventory[{index}].{field}")
        _sha256(item.get("sha256"), f"input_inventory[{index}].sha256")
        if not isinstance(item.get("source_id"), str) or not item["source_id"]:
            raise PartialTokenizerPackageError(
                f"input_inventory[{index}].source_id must be non-empty"
            )
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise PartialTokenizerPackageError(
                f"input_inventory[{index}].path must be non-empty"
            )
        present = item.get("object_receipt_present")
        receipt_sha = item.get("object_receipt_sha256")
        if present is True:
            _sha256(
                receipt_sha,
                f"input_inventory[{index}].object_receipt_sha256",
            )
        elif present is False and receipt_sha is None:
            missing_receipts.append(rank)
        else:
            raise PartialTokenizerPackageError(
                f"input_inventory[{index}] object-receipt accounting drifted"
            )
    if tuple(ranks) != EXPECTED_RANKS:
        raise PartialTokenizerPackageError("partial tokenizer rank inventory drifted")
    if tuple(missing_receipts) != MISSING_OBJECT_RECEIPT_RANKS:
        raise PartialTokenizerPackageError(
            "partial tokenizer missing object-receipt ranks drifted"
        )
    digest = hashlib.sha256(canonical_json(inventory).encode("utf-8")).hexdigest()
    if receipt.get("input_inventory_sha256") != digest:
        raise PartialTokenizerPackageError("partial tokenizer input inventory hash drifted")
    return digest


def _validate_distribution(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    distribution = receipt.get("sample_distribution")
    if distribution != config.get("sample_distribution") or not isinstance(
        distribution, list
    ):
        raise PartialTokenizerPackageError(
            "tokenizer config/receipt sample distribution differs"
        )
    seen: set[str] = set()
    target_total = 0.0
    realized_total = 0.0
    character_total = 0
    document_total = 0
    for index, item in enumerate(distribution):
        if not isinstance(item, Mapping):
            raise PartialTokenizerPackageError(
                f"sample_distribution[{index}] must be an object"
            )
        mixture_id = item.get("mixture_id")
        if not isinstance(mixture_id, str) or not mixture_id or mixture_id in seen:
            raise PartialTokenizerPackageError(
                "sample distribution mixture IDs must be unique strings"
            )
        seen.add(mixture_id)
        characters = _positive_integer(
            item.get("characters"), f"sample_distribution[{index}].characters"
        )
        documents = _positive_integer(
            item.get("documents"), f"sample_distribution[{index}].documents"
        )
        target = item.get("target_share")
        realized = item.get("realized_share")
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or isinstance(realized, bool)
            or not isinstance(realized, (int, float))
            or not 0 < float(target) <= 1
            or not 0 < float(realized) <= 1
            or abs(float(realized) - float(target)) > 0.0001
        ):
            raise PartialTokenizerPackageError(
                "partial tokenizer sample mixture share drifted"
            )
        character_total += characters
        document_total += documents
        target_total += float(target)
        realized_total += float(realized)
    if (
        not distribution
        or abs(target_total - 1.0) > 1e-12
        or abs(realized_total - 1.0) > 1e-12
        or character_total != receipt.get("training_characters")
        or document_total != receipt.get("training_documents")
    ):
        raise PartialTokenizerPackageError(
            "partial tokenizer aggregate sample distribution drifted"
        )


def _validate_receipt(
    receipt: Any,
    *,
    config: Mapping[str, Any],
    expected_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(receipt, dict):
        raise PartialTokenizerPackageError("partial training receipt must be an object")
    try:
        digest = verify_manifest_hash(receipt)
    except ManifestValidationError as exc:
        raise PartialTokenizerPackageError(
            f"invalid partial tokenizer training receipt: {exc}"
        ) from exc
    if not hmac.compare_digest(
        digest, _sha256(expected_sha256, "expected_training_receipt_sha256")
    ):
        raise PartialTokenizerPackageError("partial training receipt SHA-256 mismatch")
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("kind") != PARTIAL_RECEIPT_KIND
        or receipt.get("name") != PINNED_NAME
        or receipt.get("vocab_size") != VOCAB_SIZE
        or receipt.get("algorithm") != "raw_byte_bpe"
        or receipt.get("production_eligible") is not False
        or tuple(receipt.get("expected_ranks", ())) != EXPECTED_RANKS
    ):
        raise PartialTokenizerPackageError("partial tokenizer training receipt drifted")
    for gate in INCOMPLETE_GATES:
        if receipt.get(gate) is not False:
            raise PartialTokenizerPackageError(
                f"partial tokenizer must declare {gate}=false"
            )
    if (
        receipt.get("requested_max_characters") != REQUESTED_TRAINING_CHARACTERS
        or receipt.get("max_chars_per_document") != MAX_CHARACTERS_PER_DOCUMENT
        or receipt.get("training_characters")
        != config.get("realized_training_characters")
        or receipt.get("terminal_overshoot_characters")
        != receipt["training_characters"] - REQUESTED_TRAINING_CHARACTERS
        or not 0
        < receipt["terminal_overshoot_characters"]
        <= MAX_CHARACTERS_PER_DOCUMENT
    ):
        raise PartialTokenizerPackageError(
            "partial tokenizer receipt character accounting drifted"
        )
    _positive_integer(receipt.get("training_documents"), "training_documents")
    inventory_sha = _validate_input_inventory(receipt)
    if (
        receipt.get("object_receipts_present")
        != len(EXPECTED_RANKS) - len(MISSING_OBJECT_RECEIPT_RANKS)
        or tuple(receipt.get("object_receipts_missing", ()))
        != MISSING_OBJECT_RECEIPT_RANKS
    ):
        raise PartialTokenizerPackageError(
            "partial tokenizer object-receipt summary drifted"
        )
    _validate_distribution(config, receipt)
    return dict(receipt), digest, inventory_sha


def _parse_canonical_export(path: Path, *, lexical_size: int) -> dict[bytes, int]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise PartialTokenizerPackageError(
            "canonical tokenizer export must use final-newline LF records"
        )
    lines = raw[:-1].split(b"\n")
    if len(lines) != lexical_size:
        raise PartialTokenizerPackageError("canonical tokenizer lexical size drifted")
    reconstructed: dict[bytes, int] = {}
    for expected_rank, line in enumerate(lines):
        parts = line.split(b" ")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise PartialTokenizerPackageError(
                "canonical tokenizer export line shape drifted"
            )
        try:
            token_bytes = base64.b64decode(parts[0], validate=True)
            rank = int(parts[1].decode("ascii"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise PartialTokenizerPackageError(
                "canonical tokenizer export parse failure"
            ) from exc
        if (
            not token_bytes
            or base64.b64encode(token_bytes) != parts[0]
            or parts[1] != str(expected_rank).encode("ascii")
            or rank != expected_rank
            or token_bytes in reconstructed
        ):
            raise PartialTokenizerPackageError(
                "canonical tokenizer ranks are not dense and canonical"
            )
        reconstructed[token_bytes] = rank
    return reconstructed


def _load_token_bytes(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        value = torch.load(path, map_location="cpu")
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (VOCAB_SIZE,)
        or value.dtype != torch.int32
        or value.device.type != "cpu"
    ):
        raise PartialTokenizerPackageError(
            "token-byte table must be a 32768-element CPU torch.int32 tensor"
        )
    return value


def _validate_canonical_runtime(
    root: Path,
    *,
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    lexical = VOCAB_SIZE - len(SPECIAL_TOKENS)
    metadata = config.get("canonical_export")
    validation = receipt.get("validation")
    if not isinstance(metadata, Mapping) or not isinstance(validation, Mapping):
        raise PartialTokenizerPackageError(
            "partial tokenizer canonical validation metadata is missing"
        )
    export_path = root / "tokenizer.tiktoken"
    if (
        metadata.get("path") != export_path.name
        or metadata.get("format") != CANONICAL_EXPORT_FORMAT
        or metadata.get("sha256") != file_sha256(export_path)
        or metadata.get("lexical_ranks") != lexical
        or metadata.get("dense_rank_id_identity_verified") is not True
        or metadata.get("special_token_id_order_verified") is not True
        or metadata.get("token_byte_lengths_reconstructed") != VOCAB_SIZE
        or not isinstance(metadata.get("probe_id_sequences_verified"), int)
        or isinstance(metadata.get("probe_id_sequences_verified"), bool)
        or metadata["probe_id_sequences_verified"] <= 0
        or validation.get("canonical_export") != metadata
        or validation.get("exact_vocab_size") != VOCAB_SIZE
        or validation.get("lexical_vocab_size") != lexical
        or validation.get("all_256_bytes_representable") is not True
        or not isinstance(validation.get("unicode_roundtrip_probes"), int)
        or isinstance(validation.get("unicode_roundtrip_probes"), bool)
        or validation["unicode_roundtrip_probes"] <= 0
        or validation.get("split_pattern_sha256")
        != hashlib.sha256(SPLIT_PATTERN.encode("utf-8")).hexdigest()
    ):
        raise PartialTokenizerPackageError(
            "partial tokenizer validation/canonical export metadata drifted"
        )

    reconstructed = _parse_canonical_export(export_path, lexical_size=lexical)
    missing_bytes = [value for value in range(256) if bytes([value]) not in reconstructed]
    if missing_bytes:
        raise PartialTokenizerPackageError(
            f"partial tokenizer byte alphabet is incomplete: {missing_bytes}"
        )
    special_tokens = {
        token: lexical + index for index, token in enumerate(SPECIAL_TOKENS)
    }
    if validation.get("special_token_ids") != special_tokens:
        raise PartialTokenizerPackageError(
            "partial tokenizer declared special-token IDs are not dense and ordered"
        )
    tokenizer = RustBPETokenizer.from_directory(str(root))
    if (
        tokenizer.get_vocab_size() != VOCAB_SIZE
        or tokenizer.enc._pat_str != SPLIT_PATTERN
        or dict(tokenizer.enc._mergeable_ranks) != reconstructed
        or dict(tokenizer.enc._special_tokens) != special_tokens
        or set(tokenizer.enc._mergeable_ranks.values()) != set(range(lexical))
    ):
        raise PartialTokenizerPackageError(
            "tokenizer.pkl differs from canonical ranks/split/special-ID policy"
        )
    rebuilt = tiktoken.Encoding(
        name=PINNED_NAME,
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=reconstructed,
        special_tokens=special_tokens,
    )
    if rebuilt.n_vocab != VOCAB_SIZE:
        raise PartialTokenizerPackageError("rebuilt tokenizer vocabulary drifted")
    token_bytes = _load_token_bytes(root / "token_bytes.pt")
    expected_lengths = torch.tensor(
        [
            len(rebuilt.decode_single_token_bytes(token_id))
            for token_id in range(lexical)
        ]
        + [0] * len(SPECIAL_TOKENS),
        dtype=torch.int32,
        device="cpu",
    )
    if not torch.equal(token_bytes, expected_lengths):
        raise PartialTokenizerPackageError(
            "token-byte table content differs from the canonical tokenizer"
        )
    for token, token_id in special_tokens.items():
        if (
            rebuilt.encode_single_token(token) != token_id
            or tokenizer.encode_special(token) != token_id
            or int(token_bytes[token_id].item()) != 0
        ):
            raise PartialTokenizerPackageError(
                "partial tokenizer special-token IDs are not dense and ordered"
            )
    for probe in _PROBES:
        rebuilt_ids = rebuilt.encode_ordinary(probe)
        tokenizer_ids = tokenizer.encode(probe)
        if (
            rebuilt_ids != tokenizer_ids
            or rebuilt.decode(rebuilt_ids) != probe
            or tokenizer.decode(tokenizer_ids) != probe
        ):
            raise PartialTokenizerPackageError(
                "partial tokenizer Turkish/Unicode round-trip drifted"
            )


def verify_partial_tokenizer_package(
    manifest_path: str | Path,
    *,
    expected_sha256: str = PINNED_PACKAGE_SHA256,
    expected_training_receipt_sha256: str = PINNED_RECEIPT_SHA256,
) -> VerifiedPartialTokenizerPackage:
    """Verify the frozen partial-18 tokenizer and all of its runtime bytes.

    The defaults pin the exact UHeM package.  Explicit hash overrides exist so
    the same semantic contract can be tested against temporary fixtures; they
    do not make the partial package production-eligible.
    """

    path = Path(manifest_path)
    if path.name != "package_manifest.json" or path.is_symlink():
        raise PartialTokenizerPackageError(
            "partial tokenizer manifest must be a regular package_manifest.json"
        )
    try:
        manifest = load_json_strict(path)
        if not isinstance(manifest, dict):
            raise PartialTokenizerPackageError(
                "partial tokenizer package manifest must be an object"
            )
        digest = verify_manifest_hash(manifest)
    except (ManifestValidationError, OSError, ValueError) as exc:
        raise PartialTokenizerPackageError(
            f"invalid partial tokenizer package manifest: {exc}"
        ) from exc
    if not hmac.compare_digest(
        digest, _sha256(expected_sha256, "expected_sha256")
    ):
        raise PartialTokenizerPackageError("partial tokenizer package SHA-256 mismatch")
    if frozenset(manifest) != _PACKAGE_FIELDS:
        raise PartialTokenizerPackageError("partial tokenizer package fields drifted")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("kind") != PARTIAL_PACKAGE_KIND
        or manifest.get("name") != PINNED_NAME
        or manifest.get("vocab_size") != VOCAB_SIZE
        or manifest.get("production_eligible") is not False
    ):
        raise PartialTokenizerPackageError("unsupported partial tokenizer package")
    records = _validate_file_records(manifest.get("files"))
    try:
        verify_file_inventory(
            path.parent,
            records,
            require_exact=True,
            ignored_paths=(path.name,),
            require_role=False,
            location="files",
        )
        config = _validate_config(
            load_json_strict(path.parent / "tokenizer_config.json")
        )
        raw_receipt = load_json_strict(
            path.parent / "partial_training_receipt.json"
        )
    except (ManifestValidationError, OSError, ValueError) as exc:
        raise PartialTokenizerPackageError(
            f"invalid partial tokenizer runtime inventory: {exc}"
        ) from exc
    receipt, receipt_sha, inventory_sha = _validate_receipt(
        raw_receipt,
        config=config,
        expected_sha256=expected_training_receipt_sha256,
    )
    if (
        manifest.get("training_receipt_sha256") != receipt_sha
        or manifest.get("input_inventory_sha256") != inventory_sha
        or records[:-1] != receipt.get("payload")
    ):
        raise PartialTokenizerPackageError(
            "partial tokenizer package/receipt/payload binding mismatch"
        )
    if (
        expected_sha256 == PINNED_PACKAGE_SHA256
        and expected_training_receipt_sha256 == PINNED_RECEIPT_SHA256
        and inventory_sha != PINNED_INPUT_INVENTORY_SHA256
    ):
        raise PartialTokenizerPackageError("pinned input inventory SHA-256 mismatch")
    try:
        _validate_canonical_runtime(path.parent, config=config, receipt=receipt)
    except Exception as exc:
        if isinstance(exc, PartialTokenizerPackageError):
            raise
        raise PartialTokenizerPackageError(
            f"invalid partial tokenizer runtime semantics: {exc}"
        ) from exc
    return VerifiedPartialTokenizerPackage(
        root=path.parent,
        manifest_path=path,
        manifest=manifest,
        receipt=receipt,
        config=config,
        canonical_sha256=digest,
        receipt_sha256=receipt_sha,
    )


def load_verified_partial_tokenizer(
    package: VerifiedPartialTokenizerPackage,
) -> RustBPETokenizer:
    """Load a tokenizer only after :func:`verify_partial_tokenizer_package`."""

    tokenizer = RustBPETokenizer.from_directory(str(package.root))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise PartialTokenizerPackageError("loaded partial tokenizer vocabulary drifted")
    return tokenizer


def load_verified_partial_token_bytes(
    package: VerifiedPartialTokenizerPackage,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load the verified token-byte table onto ``device``."""

    value = _load_token_bytes(package.root / "token_bytes.pt")
    return value.to(device=device)


__all__ = [
    "EXPECTED_FILES",
    "EXPECTED_RANKS",
    "INCOMPLETE_GATES",
    "PARTIAL_PACKAGE_KIND",
    "PARTIAL_RECEIPT_KIND",
    "PINNED_NAME",
    "PINNED_PACKAGE_SHA256",
    "PINNED_RECEIPT_SHA256",
    "PartialTokenizerPackageError",
    "VerifiedPartialTokenizerPackage",
    "load_verified_partial_token_bytes",
    "load_verified_partial_tokenizer",
    "verify_partial_tokenizer_package",
]
