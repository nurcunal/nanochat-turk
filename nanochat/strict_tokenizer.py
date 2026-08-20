"""Verified production tokenizer loading for the Turkish WSD training lane.

This module intentionally supports only the frozen raw-byte BPE package used
by the d32 family.  Keeping the loader narrow avoids coupling production
training to the repository's experimental tokenizer implementations.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from nanochat.experiment_manifest import (
    ManifestValidationError,
    load_json_strict,
    verify_file_inventory,
    verify_manifest_hash,
)
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN


PACKAGE_KIND = "turkish_raw_bpe_tokenizer_package"
EXPECTED_FILES = {
    "tokenizer.pkl": "tokenizer",
    "tokenizer_config.json": "runtime_config",
    "token_bytes.pt": "token_byte_lengths",
    "training_receipt.json": "training_receipt",
}
PINNED_UPSTREAM_REVISION = "92d63d4e"
PINNED_UPSTREAM_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
PINNED_ITERATOR_SOURCE_SHA256 = (
    "206d8c89554ceeb4de7afe22e53786806d567e1c4f5493352b2170c0ac174a29"
)
REQUESTED_TRAINING_CHARACTERS = 2_000_000_000
MAX_CHARACTERS_PER_DOCUMENT = 10_000
PINNED_STOP_RULE = (
    "yield_full_capped_document_then_stop_when_cumulative_characters_"
    "strictly_exceed_threshold"
)
PINNED_ITERATOR_PARITY = {
    "passed": True,
    "upstream_commit": PINNED_UPSTREAM_COMMIT,
    "upstream_iterator_source_sha256": PINNED_ITERATOR_SOURCE_SHA256,
    "fixture_documents": ["abcdef", "xy", "1234567", "sonraki"],
    "yielded_documents": ["abcde", "xy", "12345"],
    "requested_max_characters": 10,
    "realized_characters": 12,
    "terminal_overshoot_characters": 2,
}


class TokenizerPackageError(ValueError):
    """Raised when production tokenizer bytes or metadata are not trustworthy."""


@dataclass(frozen=True)
class VerifiedTokenizerPackage:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    canonical_sha256: str


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TokenizerPackageError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_file_records(manifest: Mapping[str, Any]) -> None:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise TokenizerPackageError("tokenizer package files must be an array")
    by_path: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TokenizerPackageError(f"files[{index}] must be an object")
        path = record.get("path")
        if not isinstance(path, str) or not path or path in by_path:
            raise TokenizerPackageError("tokenizer package paths must be unique strings")
        by_path[path] = record
    if set(by_path) != set(EXPECTED_FILES):
        raise TokenizerPackageError(
            "tokenizer runtime inventory differs from the production package kind"
        )
    for path, role in EXPECTED_FILES.items():
        if by_path[path].get("role") != role:
            raise TokenizerPackageError(f"tokenizer package role mismatch for {path}")


def _validate_config(
    value: Any,
    *,
    expected_name: str | None,
    expected_vocab_size: int | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TokenizerPackageError("tokenizer_config.json must be an object")
    config = dict(value)
    name = config.get("name")
    vocab_size = config.get("vocab_size")
    if not isinstance(name, str) or not name:
        raise TokenizerPackageError("tokenizer config name must be non-empty")
    if expected_name is not None and name != expected_name:
        raise TokenizerPackageError("tokenizer config name differs from the recipe")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise TokenizerPackageError("tokenizer config vocab_size must be positive")
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise TokenizerPackageError("tokenizer vocabulary differs from the recipe")
    if config.get("implementation") != "bpe":
        raise TokenizerPackageError("production tokenizer implementation must be bpe")
    if config.get("algorithm") != "raw_byte_bpe":
        raise TokenizerPackageError("production tokenizer algorithm must be raw_byte_bpe")
    if tuple(config.get("special_tokens", ())) != tuple(SPECIAL_TOKENS):
        raise TokenizerPackageError("production tokenizer special tokens drifted")
    if config.get("split_pattern") != SPLIT_PATTERN:
        raise TokenizerPackageError("production tokenizer split pattern drifted")
    if config.get("requires_runtime_segmentation") is not False:
        raise TokenizerPackageError("production tokenizer cannot require segmentation")
    return config


def _validate_training_contract(
    config: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    requested = receipt.get("requested_max_characters")
    overshoot = receipt.get("terminal_overshoot_characters")
    realized = receipt.get("training_characters")
    documents = receipt.get("training_documents")
    iterator_stats = config.get("iterator_stats")
    if (
        requested != REQUESTED_TRAINING_CHARACTERS
        or isinstance(overshoot, bool)
        or not isinstance(overshoot, int)
        or not 0 < overshoot <= MAX_CHARACTERS_PER_DOCUMENT
        or realized != requested + overshoot
        or receipt.get("sample_characters") != realized
        or receipt.get("iterator_characters") != realized
        or receipt.get("max_chars_per_document") != MAX_CHARACTERS_PER_DOCUMENT
        or receipt.get("stop_rule") != PINNED_STOP_RULE
        or receipt.get("nanochat_upstream_revision") != PINNED_UPSTREAM_REVISION
    ):
        raise TokenizerPackageError(
            "tokenizer receipt does not implement the pinned 2B-character stop rule"
        )
    if (
        config.get("max_chars") != REQUESTED_TRAINING_CHARACTERS
        or config.get("realized_training_characters") != realized
        or config.get("terminal_overshoot_characters") != overshoot
        or config.get("doc_cap") != MAX_CHARACTERS_PER_DOCUMENT
        or config.get("stop_rule") != PINNED_STOP_RULE
        or config.get("nanochat_upstream_revision") != PINNED_UPSTREAM_REVISION
    ):
        raise TokenizerPackageError("tokenizer runtime config training threshold drifted")
    if (
        isinstance(documents, bool)
        or not isinstance(documents, int)
        or documents <= 0
        or receipt.get("iterator_documents") != documents
        or not isinstance(iterator_stats, Mapping)
        or iterator_stats.get("documents") != documents
        or iterator_stats.get("characters") != realized
    ):
        raise TokenizerPackageError("tokenizer trainer-visible accounting drifted")
    if (
        config.get("pinned_iterator_parity") != PINNED_ITERATOR_PARITY
        or receipt.get("pinned_iterator_parity") != PINNED_ITERATOR_PARITY
    ):
        raise TokenizerPackageError("tokenizer pinned iterator parity drifted")


def verify_tokenizer_package(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_name: str | None = None,
    expected_vocab_size: int | None = None,
) -> VerifiedTokenizerPackage:
    """Verify the exact raw-BPE package and all runtime bytes."""

    path = Path(manifest_path)
    if path.name != "package_manifest.json" or path.is_symlink():
        raise TokenizerPackageError(
            "tokenizer manifest must be a regular package_manifest.json"
        )
    try:
        manifest = load_json_strict(path)
        if not isinstance(manifest, dict):
            raise TokenizerPackageError("tokenizer package manifest must be an object")
        digest = verify_manifest_hash(manifest)
    except (ManifestValidationError, OSError, ValueError) as exc:
        raise TokenizerPackageError(f"invalid tokenizer package manifest: {exc}") from exc
    if expected_sha256 is not None and not hmac.compare_digest(
        digest, _sha256(expected_sha256, "expected_sha256")
    ):
        raise TokenizerPackageError("tokenizer package SHA-256 mismatch")
    if manifest.get("schema_version") != "1.0" or manifest.get("kind") != PACKAGE_KIND:
        raise TokenizerPackageError("unsupported production tokenizer package")
    _validate_file_records(manifest)
    try:
        verify_file_inventory(
            path.parent,
            manifest["files"],
            require_exact=True,
            ignored_paths=(path.name,),
            require_role=True,
            location="files",
        )
        config = _validate_config(
            load_json_strict(path.parent / "tokenizer_config.json"),
            expected_name=expected_name,
            expected_vocab_size=expected_vocab_size,
        )
        receipt = load_json_strict(path.parent / "training_receipt.json")
        if not isinstance(receipt, dict):
            raise TokenizerPackageError("training receipt must be an object")
        receipt_hash = verify_manifest_hash(receipt)
    except (ManifestValidationError, OSError, ValueError) as exc:
        raise TokenizerPackageError(f"invalid tokenizer runtime inventory: {exc}") from exc
    if manifest.get("training_receipt_sha256") != receipt_hash:
        raise TokenizerPackageError("tokenizer package/receipt binding mismatch")
    if receipt.get("kind") != "turkish_raw_bpe_training_receipt":
        raise TokenizerPackageError("unexpected tokenizer training receipt kind")
    if receipt.get("name") != config["name"] or receipt.get("vocab_size") != config["vocab_size"]:
        raise TokenizerPackageError("tokenizer receipt and runtime config differ")
    _validate_training_contract(config, receipt)
    validation = receipt.get("validation")
    if not isinstance(validation, Mapping):
        raise TokenizerPackageError("tokenizer validation receipt is missing")
    if validation.get("exact_vocab_size") != config["vocab_size"]:
        raise TokenizerPackageError("tokenizer exact-vocabulary check is missing")
    if validation.get("all_256_bytes_representable") is not True:
        raise TokenizerPackageError("tokenizer byte-alphabet check is missing")
    probes = validation.get("unicode_roundtrip_probes")
    if isinstance(probes, bool) or not isinstance(probes, int) or probes <= 0:
        raise TokenizerPackageError("tokenizer Unicode round-trip check is missing")
    return VerifiedTokenizerPackage(path.parent, path, manifest, config, digest)


def load_tokenizer_from_directory(tokenizer_dir: str | Path) -> RustBPETokenizer:
    return RustBPETokenizer.from_directory(str(tokenizer_dir))


def load_verified_tokenizer(
    package: VerifiedTokenizerPackage,
) -> RustBPETokenizer:
    tokenizer = load_tokenizer_from_directory(package.root)
    if tokenizer.get_vocab_size() != package.config["vocab_size"]:
        raise TokenizerPackageError("loaded tokenizer vocabulary differs from package")
    return tokenizer


def load_verified_token_bytes(
    package: VerifiedTokenizerPackage,
    *,
    device: str | torch.device,
) -> torch.Tensor:
    path = package.root / "token_bytes.pt"
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        value = torch.load(path, map_location=device)
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TokenizerPackageError("token_bytes.pt must contain a 1D tensor")
    if value.numel() != package.config["vocab_size"]:
        raise TokenizerPackageError("token-byte table length differs from vocabulary")
    if bool((value < 0).any().item()):
        raise TokenizerPackageError("token-byte table contains negative lengths")
    return value


__all__ = [
    "TokenizerPackageError",
    "VerifiedTokenizerPackage",
    "load_tokenizer_from_directory",
    "load_verified_token_bytes",
    "load_verified_tokenizer",
    "verify_tokenizer_package",
]
