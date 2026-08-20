"""Held-out, manually approved quality gate for the Turkish raw-BPE package."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.tokenizer import RustBPETokenizer
from nanochat.strict_tokenizer import verify_tokenizer_package
from nanochat.turkish_corpus import TOKENIZER_NAME, VOCAB_SIZE, TurkishCorpusError


QUALITY_REPORT_KIND = "turkish_tokenizer_heldout_quality"
QUALITY_APPROVAL_KIND = "turkish_tokenizer_quality_approval"
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    result: dict[str, float | int] = {"observations": len(ordered)}
    if not ordered:
        return result
    for quantile in _QUANTILES:
        result[f"q{quantile:.4f}"] = ordered[round(quantile * (len(ordered) - 1))]
    result["mean"] = sum(ordered) / len(ordered)
    return result


def _validation_texts(sample_dir: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    dataset = load_json_strict(sample_dir / "fineweb2_manifest.json")
    dataset_hash = verify_manifest_hash(dataset)
    validation_path = dataset.get("validation_file")
    if not isinstance(validation_path, str):
        raise TurkishCorpusError("tokenizer quality requires a fixed validation file")
    records = {item["path"]: item for item in dataset["ordered_files"]}
    if validation_path not in records:
        raise TurkishCorpusError("tokenizer validation file is absent from strict inventory")
    record = records[validation_path]
    path = sample_dir / validation_path
    if path.stat().st_size != record["size_bytes"] or file_sha256(path) != record["sha256"]:
        raise TurkishCorpusError("tokenizer held-out validation file drift")
    parquet = pq.ParquetFile(path)
    texts: list[str] = []
    for row_group_index in range(parquet.num_row_groups):
        values = parquet.read_row_group(
            row_group_index, columns=[dataset["text_column"]]
        ).column(dataset["text_column"])
        texts.extend(str(value) for value in values.to_pylist() if value is not None)
    if not texts:
        raise TurkishCorpusError("tokenizer held-out validation sample is empty")
    identity = {
        "dataset_manifest_sha256": dataset_hash,
        "path": validation_path,
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
        "rows": len(texts),
        "text_sequence_sha256": hashlib.sha256(
            canonical_json(texts).encode("utf-8")
        ).hexdigest(),
    }
    return texts, dataset, identity


def _metrics(tokenizer: RustBPETokenizer, texts: Sequence[str]) -> dict[str, Any]:
    document_bytes_per_token: list[float] = []
    document_chars_per_token: list[float] = []
    document_tokens_per_word: list[float] = []
    word_fertility: list[float] = []
    totals: Counter[str] = Counter()
    for text in texts:
        tokens = tokenizer.encode(text)
        words = _WORD_RE.findall(text)
        token_count = len(tokens)
        if token_count <= 0:
            raise TurkishCorpusError("held-out non-empty text encoded to zero tokens")
        byte_count = len(text.encode("utf-8"))
        char_count = len(text)
        totals["documents"] += 1
        totals["bytes"] += byte_count
        totals["characters"] += char_count
        totals["tokens"] += token_count
        totals["words"] += len(words)
        document_bytes_per_token.append(byte_count / token_count)
        document_chars_per_token.append(char_count / token_count)
        if words:
            document_tokens_per_word.append(token_count / len(words))
            for word in words:
                word_fertility.append(float(len(tokenizer.encode(word))))
    if totals["words"] <= 0:
        raise TurkishCorpusError("held-out tokenizer validation has no words")
    return {
        "totals": dict(sorted(totals.items())),
        "aggregate": {
            "bytes_per_token": totals["bytes"] / totals["tokens"],
            "characters_per_token": totals["characters"] / totals["tokens"],
            "tokens_per_word": totals["tokens"] / totals["words"],
        },
        "distributions": {
            "document_bytes_per_token": _quantiles(document_bytes_per_token),
            "document_characters_per_token": _quantiles(document_chars_per_token),
            "document_tokens_per_word": _quantiles(document_tokens_per_word),
            "word_fertility_tokens": _quantiles(word_fertility),
        },
    }


def _baseline_identity(path: Path) -> dict[str, Any]:
    package_path = path / "package_manifest.json"
    if package_path.is_file():
        try:
            verified = verify_tokenizer_package(package_path)
        except ValueError:
            verified = None
        if verified is not None:
            return {
                "identity_kind": "verified_package",
                "sha256": verified.canonical_sha256,
                "name": verified.config["name"],
                "vocab_size": verified.config["vocab_size"],
            }
    payloads = []
    for name in ("tokenizer.pkl", "tokenizer_config.json", "token_bytes.pt"):
        candidate = path / name
        if candidate.is_file() and not candidate.is_symlink():
            payloads.append(
                {
                    "path": name,
                    "size_bytes": candidate.stat().st_size,
                    "sha256": file_sha256(candidate),
                }
            )
    if not any(item["path"] == "tokenizer.pkl" for item in payloads):
        raise TurkishCorpusError("baseline tokenizer has no tokenizer.pkl")
    return {
        "identity_kind": "legacy_payload_inventory",
        "sha256": hashlib.sha256(canonical_json(payloads).encode("utf-8")).hexdigest(),
        "files": payloads,
    }


def build_tokenizer_quality_report(
    tokenizer_dir: str | Path,
    sample_dir: str | Path,
    output_dir: str | Path,
    *,
    baseline_tokenizer_dir: str | Path | None = None,
) -> dict[str, Any]:
    tokenizer_root = Path(tokenizer_dir)
    sample_root = Path(sample_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty tokenizer quality directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    package = verify_tokenizer_package(
        tokenizer_root / "package_manifest.json",
        expected_name=TOKENIZER_NAME,
        expected_vocab_size=VOCAB_SIZE,
    )
    training_receipt = load_json_strict(tokenizer_root / "training_receipt.json")
    training_hash = verify_manifest_hash(training_receipt)
    production_chain = training_receipt.get("production_chain")
    if (
        package.manifest.get("production_chain") != production_chain
        or package.manifest.get("policy_sha256")
        != training_receipt.get("policy_sha256")
        or package.manifest.get("parent_corpus_manifest_sha256")
        != training_receipt.get("parent_corpus_manifest_sha256")
        or package.manifest.get("qa_approval_sha256")
        != training_receipt.get("qa_approval_sha256")
    ):
        raise TurkishCorpusError("tokenizer package/training lineage drift")
    texts, _dataset, validation_identity = _validation_texts(sample_root)
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_root))
    current_metrics = _metrics(tokenizer, texts)
    baseline: dict[str, Any]
    if baseline_tokenizer_dir is None:
        baseline = {"available": False, "reason": "not_supplied"}
    else:
        baseline_root = Path(baseline_tokenizer_dir)
        baseline_tokenizer = RustBPETokenizer.from_directory(str(baseline_root))
        baseline_metrics = _metrics(baseline_tokenizer, texts)
        deltas = {
            key: current_metrics["aggregate"][key] - baseline_metrics["aggregate"][key]
            for key in current_metrics["aggregate"]
        }
        baseline = {
            "available": True,
            "identity": _baseline_identity(baseline_root),
            "metrics": baseline_metrics,
            "new_minus_baseline_aggregate": deltas,
            "baseline_win_does_not_auto_reject": True,
        }
    report = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": QUALITY_REPORT_KIND,
            "tokenizer_package_sha256": package.canonical_sha256,
            "training_receipt_sha256": training_hash,
            "policy_sha256": training_receipt["policy_sha256"],
            "production_chain": production_chain,
            "parent_corpus_manifest_sha256": training_receipt[
                "parent_corpus_manifest_sha256"
            ],
            "qa_approval_sha256": training_receipt["qa_approval_sha256"],
            "heldout_validation": validation_identity,
            "metrics": current_metrics,
            "structural_validation": training_receipt["validation"],
            "baseline_comparison": baseline,
            "manual_acceptance_required": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination / "quality_report.json", report)
    return report


def seal_tokenizer_quality_approval(
    quality_dir: str | Path,
    *,
    reviewer: str,
    reviewed_at_utc: str,
    decision: str,
    notes: str = "",
) -> dict[str, Any]:
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise TurkishCorpusError("tokenizer-quality reviewer must be non-empty")
    if not _RFC3339_UTC_RE.fullmatch(reviewed_at_utc):
        raise TurkishCorpusError("reviewed_at_utc must be YYYY-MM-DDTHH:MM:SSZ")
    if decision not in {"accepted", "rejected"}:
        raise TurkishCorpusError("tokenizer-quality decision must be accepted or rejected")
    if not isinstance(notes, str):
        raise TurkishCorpusError("tokenizer-quality notes must be a string")
    root = Path(quality_dir)
    report = load_json_strict(root / "quality_report.json")
    report_hash = verify_manifest_hash(report)
    path = root / "quality_approval.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    approval = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": QUALITY_APPROVAL_KIND,
            "quality_report_sha256": report_hash,
            "tokenizer_package_sha256": report["tokenizer_package_sha256"],
            "heldout_validation_sha256": report["heldout_validation"]["sha256"],
            "policy_sha256": report["policy_sha256"],
            "production_chain": report["production_chain"],
            "parent_corpus_manifest_sha256": report[
                "parent_corpus_manifest_sha256"
            ],
            "qa_approval_sha256": report["qa_approval_sha256"],
            "reviewer": reviewer,
            "reviewed_at_utc": reviewed_at_utc,
            "decision": decision,
            "notes": notes,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(path, approval)
    return approval


def validate_tokenizer_quality_gate(
    quality_dir: str | Path,
    *,
    expected_package_sha256: str,
    expected_production_chain: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(quality_dir)
    report = load_json_strict(root / "quality_report.json")
    report_hash = verify_manifest_hash(report)
    approval = load_json_strict(root / "quality_approval.json")
    verify_manifest_hash(approval)
    if (
        report.get("kind") != QUALITY_REPORT_KIND
        or report.get("tokenizer_package_sha256") != expected_package_sha256
        or report.get("manual_acceptance_required") is not True
        or (
            expected_production_chain is not None
            and report.get("production_chain") != expected_production_chain
        )
    ):
        raise TurkishCorpusError("tokenizer held-out quality report is absent or stale")
    if (
        approval.get("kind") != QUALITY_APPROVAL_KIND
        or approval.get("quality_report_sha256") != report_hash
        or approval.get("tokenizer_package_sha256") != expected_package_sha256
        or approval.get("decision") != "accepted"
        or approval.get("policy_sha256") != report.get("policy_sha256")
        or approval.get("production_chain") != report.get("production_chain")
        or approval.get("parent_corpus_manifest_sha256")
        != report.get("parent_corpus_manifest_sha256")
        or approval.get("qa_approval_sha256") != report.get("qa_approval_sha256")
    ):
        raise TurkishCorpusError("tokenizer quality approval is absent, rejected, or stale")
    if not _RFC3339_UTC_RE.fullmatch(str(approval.get("reviewed_at_utc", ""))):
        raise TurkishCorpusError("tokenizer quality approval timestamp is invalid")
    return report, approval


__all__ = [
    "build_tokenizer_quality_report",
    "seal_tokenizer_quality_approval",
    "validate_tokenizer_quality_gate",
]
