"""Reproducibility checks and identities for custom SFT artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nanochat.tokenizer import get_tokenizer_config, get_tokenizer_dir, get_tokenizer_name


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    byte_count = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            byte_count += len(line)
            if line.strip():
                rows += 1
    return {
        "path": str(path),
        "rows": rows,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: actual={actual!r}, manifest={expected!r}")


def build_custom_sft_data_identity(
    train_path: str | Path,
    validation_path: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Hash custom JSONL inputs and validate them against their preparation manifest."""

    train = Path(train_path).expanduser().resolve()
    validation = Path(validation_path).expanduser().resolve()
    for label, path in (("training", train), ("validation", validation)):
        if not path.is_file():
            raise FileNotFoundError(f"custom SFT {label} file not found: {path}")
    if train == validation:
        raise ValueError("custom SFT training and validation files must be different")

    explicit_manifest = bool(manifest_path)
    if explicit_manifest:
        manifest = Path(manifest_path).expanduser().resolve()
    elif train.parent == validation.parent:
        manifest = train.parent / "manifest.json"
    else:
        manifest = None

    identity: dict[str, Any] = {
        "format": "nanochat-conversation-jsonl",
        "train": _jsonl_identity(train),
        "validation": _jsonl_identity(validation),
        "manifest": None,
    }
    if manifest is None or not manifest.is_file():
        if explicit_manifest:
            raise FileNotFoundError(f"custom SFT data manifest not found: {manifest}")
        return identity

    try:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid custom SFT data manifest {manifest}: {exc}") from exc
    if not isinstance(manifest_data, dict):
        raise ValueError(f"custom SFT data manifest must contain a JSON object: {manifest}")

    outputs = manifest_data.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"custom SFT data manifest has no outputs object: {manifest}")
    for identity_key, manifest_key in (("train", "train"), ("validation", "validation")):
        declared = outputs.get(manifest_key)
        if not isinstance(declared, dict):
            raise ValueError(f"custom SFT manifest has no outputs.{manifest_key} object")
        actual = identity[identity_key]
        declared_path = (manifest.parent / declared.get("path", "")).resolve()
        _require_equal(f"outputs.{manifest_key}.path", Path(actual["path"]), declared_path)
        for field in ("rows", "bytes", "sha256"):
            _require_equal(f"outputs.{manifest_key}.{field}", actual[field], declared.get(field))

    identity["manifest"] = {
        "path": str(manifest),
        "bytes": manifest.stat().st_size,
        "sha256": sha256_file(manifest),
        "format_version": manifest_data.get("format_version"),
    }
    for field in ("source", "normalization", "split", "tokenizer", "max_seq_len"):
        if field in manifest_data:
            identity[field] = manifest_data[field]
    return identity


def build_tokenizer_identity(tokenizer_name: str | None = None) -> dict[str, Any]:
    """Record the exact tokenizer files selected by the nanochat runtime."""

    resolved_name = get_tokenizer_name(tokenizer_name)
    directory = Path(get_tokenizer_dir(resolved_name)).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"tokenizer directory not found: {directory}")

    files: dict[str, Any] = {}
    for filename in ("tokenizer.pkl", "tokenizer.json", "tokenizer_config.json", "token_bytes.pt"):
        path = directory / filename
        if path.is_file():
            files[filename] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    if not ({"tokenizer.pkl", "tokenizer.json"} & files.keys()):
        raise FileNotFoundError(f"tokenizer model file not found under {directory}")
    for required in ("tokenizer_config.json", "token_bytes.pt"):
        if required not in files:
            raise FileNotFoundError(f"required tokenizer artifact not found: {directory / required}")

    return {
        "name": resolved_name,
        "directory": str(directory),
        "config": get_tokenizer_config(resolved_name),
        "files": files,
    }


def verify_manifest_tokenizer(
    data_identity: dict[str, Any], tokenizer_identity: dict[str, Any]
) -> None:
    """Reject SFT data prepared with tokenizer artifacts different from runtime."""

    declared = data_identity.get("tokenizer")
    if not declared:
        return
    runtime_files = tokenizer_identity["files"]
    tokenizer_model = runtime_files.get("tokenizer.pkl") or runtime_files.get("tokenizer.json")
    if declared.get("tokenizer_sha256"):
        _require_equal(
            "tokenizer model sha256",
            tokenizer_model["sha256"],
            declared["tokenizer_sha256"],
        )
    if declared.get("config_sha256"):
        _require_equal(
            "tokenizer config sha256",
            runtime_files["tokenizer_config.json"]["sha256"],
            declared["config_sha256"],
        )
    if declared.get("token_bytes_sha256"):
        _require_equal(
            "tokenizer token_bytes sha256",
            runtime_files["token_bytes.pt"]["sha256"],
            declared["token_bytes_sha256"],
        )
