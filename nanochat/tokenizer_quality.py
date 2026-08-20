"""Fail-closed held-out quality gate for the Turkish raw-BPE package."""

from __future__ import annotations

import hashlib
import heapq
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    validate_dataset_manifest,
    verify_file_inventory,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.strict_tokenizer import verify_tokenizer_package
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN
from nanochat.turkish_corpus import (
    TOKENIZER_NAME,
    TOKENIZER_QUALITY_GATE_V1,
    VOCAB_SIZE,
    TurkishCorpusError,
)


QUALITY_REPORT_KIND = "turkish_tokenizer_heldout_quality"
QUALITY_APPROVAL_KIND = "turkish_tokenizer_quality_approval"
QUALITY_GATE_KIND = "turkish_tokenizer_automatic_quality_gate_v1"
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
_ISOLATED_WORDS_PER_STRATUM = 4_096
_METRIC_BATCH_SIZE = 256
_FIXED_TURKISH_PROBES = (
    (
        "ascii_apostrophe",
        "Ankara'da arkadaşlarımla buluşacağız; İstanbul'dan akşam döneceğim.",
    ),
    (
        "curly_apostrophe",
        "Türkiye’nin başkenti Ankara’dır; İzmir’in havası bugün çok güzel.",
    ),
    (
        "turkish_casing",
        "TÜRKİYE Türkiye türkiye; İSTANBUL İstanbul istanbul; IĞDIR Iğdır ığdır.",
    ),
    (
        "long_suffix_chain",
        "sorumluluklarımızdakilerdenmişsinizcesine konuşmayacakmışsınız",
    ),
)
_FIXED_TURKISH_PROBE_SUITE_SHA256 = hashlib.sha256(
    canonical_json([list(item) for item in _FIXED_TURKISH_PROBES]).encode("utf-8")
).hexdigest()


def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    result: dict[str, float | int] = {"observations": len(ordered)}
    if not ordered:
        return result
    for quantile in _QUANTILES:
        result[f"q{quantile:.4f}"] = ordered[round(quantile * (len(ordered) - 1))]
    result["mean"] = sum(ordered) / len(ordered)
    return result


def _stratum(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["mixture_id"]),
        str(row["source_id"]),
        str(row.get("register_bucket") or "not_applicable"),
    )


def _validation_rows(
    sample_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample = load_json_strict(sample_dir / "tokenizer_sample_manifest.json")
    verify_manifest_hash(sample)
    dataset = load_json_strict(sample_dir / "fineweb2_manifest.json")
    validate_dataset_manifest(dataset, profile="strict")
    dataset_hash = verify_manifest_hash(dataset)
    if sample.get("nanochat_dataset_manifest_sha256") != dataset_hash:
        raise TurkishCorpusError("tokenizer holdout dataset is not bound to training sample")
    validation_path = dataset.get("validation_file")
    if not isinstance(validation_path, str):
        raise TurkishCorpusError("tokenizer quality requires a fixed validation file")
    records = {item["path"]: item for item in dataset["ordered_files"]}
    if validation_path not in records:
        raise TurkishCorpusError("tokenizer validation file is absent from strict inventory")
    record = records[validation_path]
    relative = Path(validation_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise TurkishCorpusError("tokenizer validation path escapes sample directory")
    root_resolved = sample_dir.resolve(strict=True)
    path = sample_dir / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TurkishCorpusError("tokenizer validation path cannot be resolved") from exc
    if not resolved.is_relative_to(root_resolved):
        raise TurkishCorpusError("tokenizer validation path escapes sample directory")
    verify_file_inventory(
        sample_dir,
        dataset["ordered_files"],
        require_exact=True,
        ignored_paths=("fineweb2_manifest.json", "tokenizer_sample_manifest.json"),
    )
    if (
        path.is_symlink()
        or path.stat().st_size != record["size_bytes"]
        or file_sha256(path) != record["sha256"]
    ):
        raise TurkishCorpusError("tokenizer held-out validation file drift")
    parquet = pq.ParquetFile(path)
    required = {"text", "source_id", "mixture_id", "document_id", "register_bucket"}
    if not required <= set(parquet.schema_arrow.names):
        raise TurkishCorpusError("tokenizer held-out validation schema drift")
    rows: list[dict[str, Any]] = []
    for row_group_index in range(parquet.num_row_groups):
        rows.extend(parquet.read_row_group(row_group_index).to_pylist())
    rows = [row for row in rows if row.get("text")]
    if not rows:
        raise TurkishCorpusError("tokenizer held-out validation sample is empty")
    holdout = sample.get("quality_holdout")
    dataset_holdout = dataset.get("metadata", {}).get("quality_holdout")
    if not isinstance(holdout, Mapping) or dict(holdout) != dataset_holdout:
        raise TurkishCorpusError("tokenizer held-out selection lineage drift")
    observed_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    payload_bytes = 0
    for row in rows:
        byte_count = len(str(row["text"]).encode("utf-8"))
        payload_bytes += byte_count
        observed_counts[_stratum(row)]["documents"] += 1
        observed_counts[_stratum(row)]["utf8_bytes"] += byte_count
    declared = {
        (
            str(item["mixture_id"]),
            str(item["source_id"]),
            str(item["register_bucket"]),
        ): item
        for item in holdout.get("strata", [])
    }
    if (
        holdout.get("split") != "val"
        or holdout.get("complete_available_stratum_coverage") is not True
        or holdout.get("available_strata") != holdout.get("selected_strata")
        or holdout.get("selected_documents") != len(rows)
        or holdout.get("selected_utf8_bytes") != payload_bytes
        or set(declared) != set(observed_counts)
    ):
        raise TurkishCorpusError("tokenizer held-out coverage receipt drift")
    per_stratum_target = holdout.get("target_documents_per_available_stratum")
    if (
        isinstance(per_stratum_target, bool)
        or not isinstance(per_stratum_target, int)
        or per_stratum_target <= 0
    ):
        raise TurkishCorpusError("tokenizer held-out stratum floor is missing")
    expected_insufficient: list[dict[str, Any]] = []
    for key, counts in observed_counts.items():
        item = declared[key]
        eligible_documents = item.get("eligible_documents")
        coverage_floor = item.get("coverage_floor_documents")
        if (
            item.get("selected_documents") != counts["documents"]
            or item.get("selected_utf8_bytes") != counts["utf8_bytes"]
            or item.get("target_documents") != per_stratum_target
            or isinstance(eligible_documents, bool)
            or not isinstance(eligible_documents, int)
            or eligible_documents < counts["documents"]
            or coverage_floor != min(eligible_documents, per_stratum_target)
            or counts["documents"] < coverage_floor
        ):
            raise TurkishCorpusError("tokenizer held-out stratum counters drift")
        if eligible_documents < per_stratum_target:
            expected_insufficient.append(
                {
                    "mixture_id": key[0],
                    "source_id": key[1],
                    "register_bucket": key[2],
                    "eligible_documents": eligible_documents,
                    "target_documents": per_stratum_target,
                }
            )
    if holdout.get("strata_below_target_due_to_availability") != sorted(
        expected_insufficient,
        key=lambda item: (
            item["mixture_id"], item["source_id"], item["register_bucket"]
        ),
    ):
        raise TurkishCorpusError("tokenizer held-out insufficiency receipt drift")
    if (
        len(rows) < int(holdout["minimum_documents"])
        and payload_bytes < int(holdout["minimum_utf8_bytes"])
    ):
        raise TurkishCorpusError("tokenizer held-out validation size floor failed")
    identity = {
        "dataset_manifest_sha256": dataset_hash,
        "sample_manifest_sha256": sample["canonical_sha256"],
        "path": validation_path,
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
        "rows": len(rows),
        "utf8_bytes": payload_bytes,
        "text_sequence_sha256": hashlib.sha256(
            canonical_json([str(row["text"]) for row in rows]).encode("utf-8")
        ).hexdigest(),
        "selection": dict(holdout),
    }
    return rows, dataset, sample, identity


def _push_smallest_word(
    heap: list[tuple[int, str, str]], *, identity: str, word: str
) -> None:
    rank = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest(), "big")
    item = (-rank, identity, word)
    if len(heap) < _ISOLATED_WORDS_PER_STRATUM:
        heapq.heappush(heap, item)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _empty_accumulator() -> dict[str, Any]:
    return {
        "totals": Counter(),
        "used_tokens": Counter(),
        "bytes_per_token": [],
        "characters_per_token": [],
        "contextual_fertility": [],
        "roundtrip_failure_hashes": [],
    }


def _token_byte_use(
    tokenizer: RustBPETokenizer, token_counts: Mapping[int, int]
) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    unique: Counter[str] = Counter()
    total = sum(token_counts.values())
    for token_id, occurrences in token_counts.items():
        value = tokenizer.decode_single_token_bytes(int(token_id))
        flags = {
            "single_byte": len(value) == 1,
            "non_ascii": any(byte >= 128 for byte in value),
        }
        try:
            value.decode("utf-8", errors="strict")
            flags["invalid_utf8"] = False
        except UnicodeDecodeError:
            flags["invalid_utf8"] = True
        for name, enabled in flags.items():
            if enabled:
                categories[name] += int(occurrences)
                unique[name] += 1
    return {
        "total_token_uses": total,
        "single_byte_token_uses": categories["single_byte"],
        "single_byte_token_use_fraction": categories["single_byte"] / total,
        "non_ascii_token_uses": categories["non_ascii"],
        "non_ascii_token_use_fraction": categories["non_ascii"] / total,
        "invalid_utf8_token_uses": categories["invalid_utf8"],
        "invalid_utf8_token_use_fraction": categories["invalid_utf8"] / total,
        "unique_single_byte_tokens_used": unique["single_byte"],
        "unique_non_ascii_tokens_used": unique["non_ascii"],
        "unique_invalid_utf8_tokens_used": unique["invalid_utf8"],
    }


def _finish_accumulator(
    tokenizer: RustBPETokenizer,
    accumulator: Mapping[str, Any],
    *,
    isolated_token_count: int,
    isolated_word_count: int,
    isolated_values: Sequence[float],
) -> dict[str, Any]:
    totals = accumulator["totals"]
    if totals["tokens"] <= 0 or totals["words"] <= 0 or isolated_word_count <= 0:
        raise TurkishCorpusError("tokenizer metric stratum has insufficient text/words")
    used = accumulator["used_tokens"]
    lexical_size = tokenizer.get_vocab_size() - len(tokenizer.get_special_tokens())
    lexical_ids = {int(token_id) for token_id in used if int(token_id) < lexical_size}
    return {
        "totals": dict(sorted(totals.items())),
        "efficiency": {
            "bytes_per_token": totals["bytes"] / totals["tokens"],
            "characters_per_token": totals["characters"] / totals["tokens"],
        },
        "fertility": {
            "contextual_tokens_per_word": totals["tokens"] / totals["words"],
            "isolated_tokens_per_word": isolated_token_count / isolated_word_count,
            "isolated_word_sample_size": isolated_word_count,
            "isolated_sampling": "smallest_sha256_per_combined_stratum_v1",
        },
        "vocabulary_utilization": {
            "lexical_vocab_size": lexical_size,
            "unique_lexical_tokens_used": len(lexical_ids),
            "fraction": len(lexical_ids) / lexical_size,
        },
        "token_byte_use": _token_byte_use(tokenizer, used),
        "roundtrip": {
            "documents_checked": totals["documents"],
            "failures": len(accumulator["roundtrip_failure_hashes"]),
            "failure_text_sha256": accumulator["roundtrip_failure_hashes"][:16],
        },
        "distributions": {
            "document_bytes_per_token": _quantiles(accumulator["bytes_per_token"]),
            "document_characters_per_token": _quantiles(
                accumulator["characters_per_token"]
            ),
            "contextual_tokens_per_word": _quantiles(
                accumulator["contextual_fertility"]
            ),
            "isolated_word_tokens": _quantiles(isolated_values),
        },
    }


def _fixed_probe_metrics(tokenizer: RustBPETokenizer) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for probe_id, text in _FIXED_TURKISH_PROBES:
        tokens = tokenizer.encode(text)
        words = _WORD_RE.findall(text)
        isolated = [len(tokenizer.encode(word)) for word in words]
        roundtrip = tokenizer.decode(tokens) == text
        cases.append(
            {
                "id": probe_id,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "tokens": len(tokens),
                "bytes_per_token": len(text.encode("utf-8")) / len(tokens),
                "contextual_tokens_per_word": len(tokens) / len(words),
                "isolated_tokens_per_word": sum(isolated) / len(isolated),
                "roundtrip": roundtrip,
            }
        )
    return {
        "suite": "turkish_apostrophe_casing_long_suffix_v1",
        "suite_sha256": _FIXED_TURKISH_PROBE_SUITE_SHA256,
        "passed": all(item["roundtrip"] for item in cases),
        "cases": cases,
    }


def _metrics(
    tokenizer: RustBPETokenizer, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    overall = _empty_accumulator()
    by_stratum: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        _empty_accumulator
    )
    isolated_heaps: dict[tuple[str, str, str], list[tuple[int, str, str]]] = defaultdict(
        list
    )
    for offset in range(0, len(rows), _METRIC_BATCH_SIZE):
        batch = rows[offset : offset + _METRIC_BATCH_SIZE]
        texts = [str(row["text"]) for row in batch]
        encoded = tokenizer.encode(
            texts, num_threads=min(16, max(1, os.cpu_count() or 1))
        )
        if not isinstance(encoded, list) or len(encoded) != len(batch):
            raise TurkishCorpusError("tokenizer quality batched encoding shape drift")
        for row, text, token_ids in zip(batch, texts, encoded, strict=True):
            if not token_ids:
                raise TurkishCorpusError("held-out non-empty text encoded to zero tokens")
            words = _WORD_RE.findall(text)
            key = _stratum(row)
            byte_count = len(text.encode("utf-8"))
            token_count = len(token_ids)
            decoded = tokenizer.decode(token_ids)
            for accumulator in (overall, by_stratum[key]):
                accumulator["totals"]["documents"] += 1
                accumulator["totals"]["bytes"] += byte_count
                accumulator["totals"]["characters"] += len(text)
                accumulator["totals"]["tokens"] += token_count
                accumulator["totals"]["words"] += len(words)
                accumulator["used_tokens"].update(int(value) for value in token_ids)
                accumulator["bytes_per_token"].append(byte_count / token_count)
                accumulator["characters_per_token"].append(len(text) / token_count)
                if words:
                    accumulator["contextual_fertility"].append(
                        token_count / len(words)
                    )
                if decoded != text:
                    accumulator["roundtrip_failure_hashes"].append(
                        hashlib.sha256(text.encode("utf-8")).hexdigest()
                    )
            if not words:
                continue
            document_id = str(row["document_id"])
            for word_index, word in enumerate(words):
                _push_smallest_word(
                    isolated_heaps[key],
                    identity=f"{key}\0{document_id}\0{word_index}\0{word}",
                    word=word,
                )
    isolated_by_stratum: dict[tuple[str, str, str], tuple[int, int, list[float]]] = {}
    for key, heap in isolated_heaps.items():
        words = [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
        encoded = tokenizer.encode(
            words, num_threads=min(16, max(1, os.cpu_count() or 1))
        )
        counts = [len(tokens) for tokens in encoded]
        isolated_by_stratum[key] = (sum(counts), len(counts), [float(v) for v in counts])
    stratum_records = []
    overall_isolated_tokens = 0
    overall_isolated_words = 0
    overall_isolated_values: list[float] = []
    for key in sorted(by_stratum):
        isolated_tokens, isolated_words, isolated_values = isolated_by_stratum[key]
        overall_isolated_tokens += isolated_tokens
        overall_isolated_words += isolated_words
        overall_isolated_values.extend(isolated_values)
        stratum_records.append(
            {
                "mixture_id": key[0],
                "source_id": key[1],
                "register_bucket": key[2],
                "metrics": _finish_accumulator(
                    tokenizer,
                    by_stratum[key],
                    isolated_token_count=isolated_tokens,
                    isolated_word_count=isolated_words,
                    isolated_values=isolated_values,
                ),
            }
        )
    return {
        "overall": _finish_accumulator(
            tokenizer,
            overall,
            isolated_token_count=overall_isolated_tokens,
            isolated_word_count=overall_isolated_words,
            isolated_values=overall_isolated_values,
        ),
        "strata": stratum_records,
        "fixed_turkish_probes": _fixed_probe_metrics(tokenizer),
    }


def validate_pinned_baseline_tokenizer(
    baseline_dir: str | Path,
    contract: Mapping[str, Any],
) -> tuple[RustBPETokenizer, dict[str, Any]]:
    """Verify the exact legacy comparison tokenizer before expensive training."""

    root = Path(baseline_dir)
    if not root.is_dir() or root.is_symlink():
        raise TurkishCorpusError("baseline tokenizer root must be a real directory")
    package_path = root / "package_manifest.json"
    if package_path.exists():
        # Never reinterpret a damaged package as a legacy directory.
        try:
            verify_tokenizer_package(package_path)
        except (OSError, ValueError) as exc:
            raise TurkishCorpusError(
                "baseline package manifest exists but strict verification failed"
            ) from exc
    expected_files = contract.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise TurkishCorpusError("baseline policy has no pinned payload inventory")
    actual_files: list[dict[str, Any]] = []
    for item in expected_files:
        if not isinstance(item, Mapping):
            raise TurkishCorpusError("baseline policy payload record is malformed")
        name = item.get("path")
        if name not in {"tokenizer.pkl", "tokenizer_config.json", "token_bytes.pt"}:
            raise TurkishCorpusError("baseline policy contains an unexpected payload")
        path = root / str(name)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("size_bytes")
            or file_sha256(path) != item.get("sha256")
        ):
            raise TurkishCorpusError(f"baseline tokenizer payload drift: {name}")
        actual_files.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if sorted(item["path"] for item in actual_files) != [
        "token_bytes.pt",
        "tokenizer.pkl",
        "tokenizer_config.json",
    ]:
        raise TurkishCorpusError("baseline tokenizer payload inventory is incomplete")
    config = load_json_strict(root / "tokenizer_config.json")
    if (
        not isinstance(config, Mapping)
        or config.get("name") != contract.get("name")
        or config.get("vocab_size") != contract.get("vocab_size")
        or contract.get("split_pattern")
        != "nanochat_gpt4_style_numbers_1_or_2_v1"
        or contract.get("special_token_policy") != "nanochat_9_v1"
        or contract.get("token_byte_table_semantics")
        != "legacy_decoded_utf8_replacement_v1"
    ):
        raise TurkishCorpusError("baseline tokenizer config/policy identity drift")
    try:
        tokenizer = RustBPETokenizer.from_directory(str(root))
    except Exception as exc:
        raise TurkishCorpusError("baseline tokenizer.pkl cannot be loaded") from exc
    lexical = int(contract["vocab_size"]) - len(SPECIAL_TOKENS)
    expected_specials = {
        token: lexical + index for index, token in enumerate(SPECIAL_TOKENS)
    }
    if (
        tokenizer.get_vocab_size() != contract["vocab_size"]
        or tokenizer.enc._pat_str != SPLIT_PATTERN
        or dict(tokenizer.enc._special_tokens) != expected_specials
        or set(tokenizer.enc._mergeable_ranks.values()) != set(range(lexical))
        or any(bytes([value]) not in tokenizer.enc._mergeable_ranks for value in range(256))
    ):
        raise TurkishCorpusError("baseline tokenizer runtime is not comparable raw BPE")
    try:
        token_bytes = torch.load(
            root / "token_bytes.pt", map_location="cpu", weights_only=True
        )
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        token_bytes = torch.load(root / "token_bytes.pt", map_location="cpu")
    expected_lengths = torch.tensor(
        [
            len(tokenizer.decode([token_id]).encode("utf-8"))
            if token_id < lexical
            else 0
            for token_id in range(int(contract["vocab_size"]))
        ],
        dtype=torch.int32,
    )
    raw_lengths = torch.tensor(
        [
            len(tokenizer.enc.decode_single_token_bytes(token_id))
            if token_id < lexical
            else 0
            for token_id in range(int(contract["vocab_size"]))
        ],
        dtype=torch.int32,
    )
    raw_mismatch_ids = (expected_lengths != raw_lengths).nonzero().flatten().tolist()
    if (
        not isinstance(token_bytes, torch.Tensor)
        or token_bytes.dtype != torch.int32
        or token_bytes.device.type != "cpu"
        or not torch.equal(token_bytes, expected_lengths)
    ):
        raise TurkishCorpusError("baseline token-byte table is incompatible")
    if len(raw_mismatch_ids) != contract.get("raw_byte_length_mismatch_count"):
        raise TurkishCorpusError("baseline historical/raw token-byte mismatch drift")
    actual_files.sort(key=lambda item: item["path"])
    inventory_sha256 = hashlib.sha256(
        canonical_json(actual_files).encode("utf-8")
    ).hexdigest()
    if inventory_sha256 != contract.get("payload_inventory_sha256"):
        raise TurkishCorpusError("baseline tokenizer payload-inventory hash drift")
    identity = {
        "identity_kind": "pinned_legacy_payload_inventory",
        "name": contract["name"],
        "vocab_size": contract["vocab_size"],
        "files": actual_files,
        "sha256": inventory_sha256,
        "split_pattern_sha256": hashlib.sha256(SPLIT_PATTERN.encode("utf-8")).hexdigest(),
        "special_token_map": expected_specials,
        "byte_alphabet_complete": True,
        "token_byte_table_semantics": "legacy_decoded_utf8_replacement_v1",
        "raw_byte_length_semantics_compatible": not raw_mismatch_ids,
        "raw_byte_length_mismatch_count": len(raw_mismatch_ids),
        "raw_byte_length_mismatch_examples": [
            {
                "token_id": int(token_id),
                "legacy_decoded_utf8_bytes": int(expected_lengths[token_id].item()),
                "raw_token_bytes": int(raw_lengths[token_id].item()),
            }
            for token_id in raw_mismatch_ids[:16]
        ],
    }
    return tokenizer, identity


def evaluate_tokenizer_quality_gate(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every frozen automatic threshold; missing evidence fails closed."""

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    current_overall = current["overall"]
    baseline_overall = baseline["overall"]
    check(
        "current_roundtrip_zero",
        current_overall["roundtrip"]["failures"]
        <= int(policy["max_roundtrip_failures"]),
        observed=current_overall["roundtrip"]["failures"],
        maximum=policy["max_roundtrip_failures"],
    )
    check(
        "baseline_roundtrip_zero",
        baseline_overall["roundtrip"]["failures"] == 0,
        observed=baseline_overall["roundtrip"]["failures"],
        maximum=0,
    )
    current_probe_payload = current["fixed_turkish_probes"]
    baseline_probe_payload = baseline["fixed_turkish_probes"]
    probe_suite_evidence = (
        current_probe_payload.get("passed") is True
        and baseline_probe_payload.get("passed") is True
        and current_probe_payload.get("suite")
        == "turkish_apostrophe_casing_long_suffix_v1"
        and baseline_probe_payload.get("suite")
        == "turkish_apostrophe_casing_long_suffix_v1"
        and current_probe_payload.get("suite_sha256")
        == _FIXED_TURKISH_PROBE_SUITE_SHA256
        and baseline_probe_payload.get("suite_sha256")
        == _FIXED_TURKISH_PROBE_SUITE_SHA256
        and policy.get("fixed_probe_suite")
        == "turkish_apostrophe_casing_long_suffix_v1"
        and policy.get("fixed_probe_suite_sha256")
        == _FIXED_TURKISH_PROBE_SUITE_SHA256
    )
    check(
        "fixed_turkish_probes",
        probe_suite_evidence,
        suite=current_probe_payload.get("suite"),
        suite_sha256=current_probe_payload.get("suite_sha256"),
    )
    expected_probes = {
        probe_id: {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        for probe_id, text in _FIXED_TURKISH_PROBES
    }
    expected_probe_ids = set(expected_probes)

    def exact_probe_cases(payload: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        cases = payload.get("cases")
        if not isinstance(cases, list) or len(cases) != len(expected_probes):
            return False, {}
        by_id: dict[str, Any] = {}
        valid = True
        for item in cases:
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                valid = False
                continue
            probe_id = str(item["id"])
            if probe_id in by_id:
                valid = False
            by_id[probe_id] = item
        if set(by_id) != expected_probe_ids:
            valid = False
        for probe_id, expected in expected_probes.items():
            item = by_id.get(probe_id)
            if not isinstance(item, Mapping):
                continue
            if (
                item.get("text_sha256") != expected["text_sha256"]
                or item.get("text") != expected["text"]
                or item.get("roundtrip") is not True
            ):
                valid = False
            for metric in (
                "bytes_per_token",
                "contextual_tokens_per_word",
                "isolated_tokens_per_word",
            ):
                value = item.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    valid = False
        return valid, by_id

    current_probe_evidence, current_probes = exact_probe_cases(current_probe_payload)
    baseline_probe_evidence, baseline_probes = exact_probe_cases(
        baseline_probe_payload
    )
    check(
        "fixed_turkish_probe_case_identity",
        current_probe_evidence and baseline_probe_evidence,
        current_ids=sorted(current_probes),
        baseline_ids=sorted(baseline_probes),
        expected_ids=sorted(expected_probe_ids),
    )
    if probe_suite_evidence and current_probe_evidence and baseline_probe_evidence:
        for probe_id in sorted(expected_probe_ids):
            observed_probe = current_probes[probe_id]
            baseline_probe = baseline_probes[probe_id]
            efficiency_minimum = float(baseline_probe["bytes_per_token"]) * (
                1 - float(policy["max_probe_efficiency_regression_fraction"])
            )
            check(
                f"probe_efficiency:{probe_id}",
                float(observed_probe["bytes_per_token"]) >= efficiency_minimum,
                observed=observed_probe["bytes_per_token"],
                baseline=baseline_probe["bytes_per_token"],
                minimum=efficiency_minimum,
            )
            for metric in (
                "contextual_tokens_per_word",
                "isolated_tokens_per_word",
            ):
                maximum = float(baseline_probe[metric]) * (
                    1 + float(policy["max_probe_fertility_regression_fraction"])
                )
                check(
                    f"probe_fertility_{metric}:{probe_id}",
                    float(observed_probe[metric]) <= maximum,
                    observed=observed_probe[metric],
                    baseline=baseline_probe[metric],
                    maximum=maximum,
                )
    utilization = current_overall["vocabulary_utilization"]["fraction"]
    check(
        "vocabulary_utilization",
        utilization >= float(policy["min_vocab_utilization_fraction"]),
        observed=utilization,
        minimum=policy["min_vocab_utilization_fraction"],
    )
    byte_use = current_overall["token_byte_use"]
    check(
        "single_byte_token_use",
        byte_use["single_byte_token_use_fraction"]
        <= float(policy["max_single_byte_token_use_fraction"]),
        observed=byte_use["single_byte_token_use_fraction"],
        maximum=policy["max_single_byte_token_use_fraction"],
    )
    check(
        "non_ascii_token_use",
        byte_use["non_ascii_token_use_fraction"]
        >= float(policy["min_non_ascii_token_use_fraction"]),
        observed=byte_use["non_ascii_token_use_fraction"],
        minimum=policy["min_non_ascii_token_use_fraction"],
    )
    check(
        "invalid_utf8_token_use",
        byte_use["invalid_utf8_token_use_fraction"]
        <= float(policy["max_invalid_utf8_token_use_fraction"]),
        observed=byte_use["invalid_utf8_token_use_fraction"],
        maximum=policy["max_invalid_utf8_token_use_fraction"],
    )
    for metric in ("bytes_per_token", "characters_per_token"):
        observed = current_overall["efficiency"][metric]
        reference = baseline_overall["efficiency"][metric]
        minimum = reference * (
            1 - float(policy["max_overall_efficiency_regression_fraction"])
        )
        check(
            f"overall_efficiency_{metric}",
            observed >= minimum,
            observed=observed,
            baseline=reference,
            minimum=minimum,
        )
    for metric in ("contextual_tokens_per_word", "isolated_tokens_per_word"):
        observed = current_overall["fertility"][metric]
        reference = baseline_overall["fertility"][metric]
        maximum = reference * (
            1 + float(policy["max_overall_fertility_regression_fraction"])
        )
        check(
            f"overall_fertility_{metric}",
            observed <= maximum,
            observed=observed,
            baseline=reference,
            maximum=maximum,
        )
    current_strata = {
        (item["mixture_id"], item["source_id"], item["register_bucket"]): item[
            "metrics"
        ]
        for item in current["strata"]
    }
    baseline_strata = {
        (item["mixture_id"], item["source_id"], item["register_bucket"]): item[
            "metrics"
        ]
        for item in baseline["strata"]
    }
    check(
        "baseline_stratum_coverage",
        set(current_strata) == set(baseline_strata),
        current_strata=len(current_strata),
        baseline_strata=len(baseline_strata),
    )
    for key in sorted(set(current_strata) | set(baseline_strata)):
        label = "/".join(key)
        if key not in current_strata or key not in baseline_strata:
            check(f"stratum_present:{label}", False)
            continue
        current_metrics = current_strata[key]
        baseline_metrics = baseline_strata[key]
        check(
            f"stratum_roundtrip:{label}",
            current_metrics["roundtrip"]["failures"] == 0,
            observed=current_metrics["roundtrip"]["failures"],
        )
        for metric in ("bytes_per_token", "characters_per_token"):
            observed = current_metrics["efficiency"][metric]
            reference = baseline_metrics["efficiency"][metric]
            minimum = reference * (
                1 - float(policy["max_per_stratum_efficiency_regression_fraction"])
            )
            check(
                f"stratum_efficiency_{metric}:{label}",
                observed >= minimum,
                observed=observed,
                baseline=reference,
                minimum=minimum,
            )
        for metric in ("contextual_tokens_per_word", "isolated_tokens_per_word"):
            observed = current_metrics["fertility"][metric]
            reference = baseline_metrics["fertility"][metric]
            maximum = reference * (
                1 + float(policy["max_per_stratum_fertility_regression_fraction"])
            )
            check(
                f"stratum_fertility_{metric}:{label}",
                observed <= maximum,
                observed=observed,
                baseline=reference,
                maximum=maximum,
            )
    failures = [item["name"] for item in checks if item["passed"] is not True]
    return {
        "kind": QUALITY_GATE_KIND,
        "passed": not failures,
        "policy": dict(policy),
        "checks": checks,
        "failures": failures,
    }


def build_tokenizer_quality_report(
    tokenizer_dir: str | Path,
    sample_dir: str | Path,
    output_dir: str | Path,
    *,
    baseline_tokenizer_dir: str | Path,
) -> dict[str, Any]:
    tokenizer_root = Path(tokenizer_dir)
    sample_root = Path(sample_dir)
    baseline_root = Path(baseline_tokenizer_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty tokenizer quality directory: {destination}")
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
    rows, dataset, sample, validation_identity = _validation_rows(sample_root)
    if (
        training_receipt.get("sample_manifest_sha256") != sample["canonical_sha256"]
        or training_receipt.get("dataset_manifest_sha256")
        != dataset["canonical_sha256"]
        or training_receipt.get("quality_holdout") != sample.get("quality_holdout")
        or training_receipt.get("quality_gate_policy")
        != sample.get("quality_gate_policy")
        or training_receipt.get("baseline_tokenizer")
        != sample.get("baseline_tokenizer")
    ):
        raise TurkishCorpusError("tokenizer training/sample quality lineage drift")
    policy = training_receipt.get("quality_gate_policy")
    if not isinstance(policy, Mapping) or policy.get("baseline_required") is not True:
        raise TurkishCorpusError("tokenizer training receipt lacks mandatory baseline policy")
    baseline_contract = training_receipt.get("baseline_tokenizer")
    if not isinstance(baseline_contract, Mapping):
        raise TurkishCorpusError("tokenizer training receipt lacks pinned baseline inventory")
    baseline_tokenizer, baseline_identity = validate_pinned_baseline_tokenizer(
        baseline_root, baseline_contract
    )
    if training_receipt.get("baseline_identity") != baseline_identity:
        raise TurkishCorpusError("tokenizer training receipt baseline identity drift")
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_root))
    if (
        dict(tokenizer.enc._mergeable_ranks)
        == dict(baseline_tokenizer.enc._mergeable_ranks)
        and dict(tokenizer.enc._special_tokens)
        == dict(baseline_tokenizer.enc._special_tokens)
        and tokenizer.enc._pat_str == baseline_tokenizer.enc._pat_str
    ):
        raise TurkishCorpusError("current and baseline tokenizer identities are equal")
    current_metrics = _metrics(tokenizer, rows)
    baseline_metrics = _metrics(baseline_tokenizer, rows)
    automatic_gate = evaluate_tokenizer_quality_gate(
        current_metrics, baseline_metrics, policy
    )
    baseline = {
        "available": True,
        "required": True,
        "identity": baseline_identity,
        "metrics": baseline_metrics,
        "comparison_gate_passed": automatic_gate["passed"],
    }
    destination.mkdir(parents=True, exist_ok=True)
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
            "quality_gate_policy": dict(policy),
            "automatic_gate": automatic_gate,
            "automated_gate_passed": automatic_gate["passed"],
            "manual_acceptance_required": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination / "quality_report.json", report)
    return report


def _recompute_report_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the complete gate instead of trusting embedded pass flags."""

    baseline = report.get("baseline_comparison")
    current = report.get("metrics")
    policy = report.get("quality_gate_policy")
    if (
        not isinstance(baseline, Mapping)
        or not isinstance(baseline.get("metrics"), Mapping)
        or not isinstance(current, Mapping)
        or not isinstance(policy, Mapping)
    ):
        raise TurkishCorpusError("tokenizer quality report lacks recomputable gate evidence")
    if dict(policy) != TOKENIZER_QUALITY_GATE_V1:
        raise TurkishCorpusError("tokenizer quality report gate policy is not frozen v1")
    try:
        recomputed = evaluate_tokenizer_quality_gate(
            current, baseline["metrics"], policy
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise TurkishCorpusError("tokenizer quality gate evidence is malformed") from exc
    if (
        report.get("automatic_gate") != recomputed
        or report.get("automated_gate_passed") is not recomputed["passed"]
        or baseline.get("comparison_gate_passed") is not recomputed["passed"]
    ):
        raise TurkishCorpusError("tokenizer automatic quality gate does not recompute")
    return recomputed


def seal_tokenizer_quality_approval(
    quality_dir: str | Path,
    *,
    tokenizer_dir: str | Path,
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
    report_package_sha256 = report.get("tokenizer_package_sha256")
    if not isinstance(report_package_sha256, str):
        raise TurkishCorpusError("tokenizer quality report package binding is missing")
    package = verify_tokenizer_package(
        Path(tokenizer_dir) / "package_manifest.json",
        expected_sha256=report_package_sha256,
        expected_name=TOKENIZER_NAME,
        expected_vocab_size=VOCAB_SIZE,
    )
    expected_training_receipt_sha256 = package.manifest.get(
        "training_receipt_sha256"
    )
    if report.get("training_receipt_sha256") != expected_training_receipt_sha256:
        raise TurkishCorpusError(
            "tokenizer quality report/training receipt binding drift"
        )
    baseline = report.get("baseline_comparison")
    automatic = _recompute_report_gate(report)
    if decision == "accepted" and (
        not isinstance(baseline, Mapping)
        or baseline.get("available") is not True
        or baseline.get("required") is not True
        or baseline.get("comparison_gate_passed") is not True
        or not isinstance(automatic, Mapping)
        or automatic.get("passed") is not True
        or report.get("automated_gate_passed") is not True
    ):
        raise TurkishCorpusError(
            "cannot accept tokenizer with a missing baseline or failed automatic quality gate"
        )
    path = root / "quality_approval.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    baseline_identity = baseline.get("identity") if isinstance(baseline, Mapping) else None
    baseline_identity_sha256 = (
        baseline_identity.get("sha256")
        if isinstance(baseline_identity, Mapping)
        else None
    )
    approval = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": QUALITY_APPROVAL_KIND,
            "quality_report_sha256": report_hash,
            "tokenizer_package_sha256": report["tokenizer_package_sha256"],
            "training_receipt_sha256": expected_training_receipt_sha256,
            "heldout_validation_sha256": report["heldout_validation"]["sha256"],
            "policy_sha256": report["policy_sha256"],
            "production_chain": report["production_chain"],
            "parent_corpus_manifest_sha256": report[
                "parent_corpus_manifest_sha256"
            ],
            "qa_approval_sha256": report["qa_approval_sha256"],
            "automatic_gate_passed": report["automated_gate_passed"],
            "automatic_gate_sha256": hashlib.sha256(
                canonical_json(automatic).encode("utf-8")
            ).hexdigest(),
            "baseline_identity_sha256": baseline_identity_sha256,
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
    expected_training_receipt_sha256: str,
    expected_production_chain: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(quality_dir)
    report = load_json_strict(root / "quality_report.json")
    report_hash = verify_manifest_hash(report)
    approval = load_json_strict(root / "quality_approval.json")
    verify_manifest_hash(approval)
    baseline = report.get("baseline_comparison")
    automatic = _recompute_report_gate(report)
    automatic_sha256 = hashlib.sha256(
        canonical_json(automatic).encode("utf-8")
    ).hexdigest()
    if (
        report.get("kind") != QUALITY_REPORT_KIND
        or report.get("tokenizer_package_sha256") != expected_package_sha256
        or report.get("training_receipt_sha256")
        != expected_training_receipt_sha256
        or report.get("manual_acceptance_required") is not True
        or report.get("automated_gate_passed") is not True
        or not isinstance(baseline, Mapping)
        or baseline.get("available") is not True
        or baseline.get("required") is not True
        or baseline.get("comparison_gate_passed") is not True
        or not isinstance(automatic, Mapping)
        or automatic.get("passed") is not True
        or automatic.get("failures") != []
        or (
            expected_production_chain is not None
            and report.get("production_chain") != expected_production_chain
        )
    ):
        raise TurkishCorpusError("tokenizer held-out quality report is absent, failed, or stale")
    if (
        approval.get("kind") != QUALITY_APPROVAL_KIND
        or approval.get("quality_report_sha256") != report_hash
        or approval.get("tokenizer_package_sha256") != expected_package_sha256
        or approval.get("training_receipt_sha256")
        != expected_training_receipt_sha256
        or approval.get("decision") != "accepted"
        or approval.get("automatic_gate_passed") is not True
        or approval.get("automatic_gate_sha256") != automatic_sha256
        or approval.get("baseline_identity_sha256") != baseline["identity"]["sha256"]
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
    "evaluate_tokenizer_quality_gate",
    "seal_tokenizer_quality_approval",
    "validate_pinned_baseline_tokenizer",
    "validate_tokenizer_quality_gate",
]
