"""Deterministic, fail-closed preparation for the Turkish d32 corpus.

This module deliberately separates *acquisition* from *selection*.  Acquisition
receipts pin every upstream object; selection then streams one verified object at
a time, applies the Turkish/no-code policy, removes exact and near duplicates,
and assigns the surviving duplicate cluster to one immutable split.

The implementation is intentionally usable without Hugging Face ``datasets``.
Production jobs can therefore stage and discard one remote shard at a time rather
than keeping every raw corpus beside the final compressed parquet files.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)


CORPUS_NAME = "tr_general_clean_v1"
TOKENIZER_NAME = "tr_general_raw_bpe_32k_v1"
VOCAB_SIZE = 32_768
D32_GLOBAL_BATCH_TOKENS = 2_097_152
D32_EVAL_MAX_PAYLOAD_TOKENS = 2_048
D32_EVAL_ROW_CAPACITY = D32_EVAL_MAX_PAYLOAD_TOKENS + 1
D32_EXPOSURE_MATRIX_V1 = (
    ("trunk_ws8_seed42", "trunk", 8, 42, 28_800, D32_GLOBAL_BATCH_TOKENS),
    ("trunk_ws16_seed42", "trunk", 16, 42, 28_800, D32_GLOBAL_BATCH_TOKENS),
    ("s12_ws8_seed42", "s12", 8, 42, 9_600, D32_GLOBAL_BATCH_TOKENS),
    ("s12_ws16_seed42", "s12", 16, 42, 9_600, D32_GLOBAL_BATCH_TOKENS),
    ("s20_ws8_seed42", "s20", 8, 42, 16_000, D32_GLOBAL_BATCH_TOKENS),
    ("s20_ws16_seed42", "s20", 16, 42, 16_000, D32_GLOBAL_BATCH_TOKENS),
    ("s40_ws8_seed42", "s40", 8, 42, 32_000, D32_GLOBAL_BATCH_TOKENS),
    ("s40_ws16_seed42", "s40", 16, 42, 32_000, D32_GLOBAL_BATCH_TOKENS),
    ("smoke_ws8_seed42", "smoke", 8, 42, 100, D32_GLOBAL_BATCH_TOKENS),
    ("smoke_ws16_seed42", "smoke", 16, 42, 100, D32_GLOBAL_BATCH_TOKENS),
    ("signal_smoke_ws4_seed42", "signal_smoke", 4, 42, 6, D32_GLOBAL_BATCH_TOKENS),
    ("proxy_d12_seed42_ws1", "wd_proxy", 1, 42, 4_200, 524_288),
    ("proxy_d12_seed314159_ws1", "wd_proxy", 1, 314_159, 4_200, 524_288),
    ("proxy_d20_seed42_ws1", "wd_proxy", 1, 42, 4_980, 1_048_576),
)
SOURCE_RECEIPT_KIND = "turkish_pretrain_source_receipt"
BACKEND_RECEIPT_KIND = "turkish_production_backend_output"
CORPUS_MANIFEST_KIND = "turkish_pretrain_corpus"
POOL_OWNERSHIP_KIND = "turkish_run_owned_filtered_pool"
POOL_OWNERSHIP_FILE = "run_owned_pool.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_RFC3339_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SPACE_RE = re.compile(r"[\t\v\f\r ]+")
_BLANK_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_CODE_LINE_RE = re.compile(
    r"^\s*(?:```|~~~|#!\s*/|#include\b|"
    r"(?:def|class|import|from|function|const|let|var|package|namespace|using)\s+|"
    r"(?:public|private|protected)\s+(?:static\s+)?|SELECT\s+.+\s+FROM\s+|"
    r"(?:INSERT\s+INTO|UPDATE\s+\S+\s+SET|DELETE\s+FROM)\b|"
    r"(?:npm|pip|cargo)\s+(?:install|add)\b|</?[a-z][^>]*>|"
    r"(?:print|sorted|len|range|console\.log|system\.out\.println)\s*\(|"
    r"[^\W\d]\w*\s*=\s*(?:[\[{'\"(]|[-+]?\d|[^\W\d]\w*\s*\())",
    re.IGNORECASE,
)
_PROGRAMMING_RE = re.compile(
    r"\b(?:github|gitlab|stackoverflow|javascript|typescript|python|java|golang|"
    r"rustlang|dockerfile|kubernetes|react\.js|node\.js|api\s+endpoint|source\s+code|"
    r"kaynak\s+kod|programlama\s+dili|yazılım\s+geliştirici)\b",
    re.IGNORECASE,
)
_CODE_HOSTS = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "stackoverflow.com",
    "stackexchange.com",
    "npmjs.com",
    "pypi.org",
    "crates.io",
)
_FASTTEXT_PROBABILITY_EPSILON = 1e-4

# Common function words are deliberately more useful than Turkish-specific
# letters: clean Turkish may be ASCII-only, while Azerbaijani also uses several
# Turkish letters.  The audit combines lexical, suffix and script evidence.
_TR_FUNCTION_WORDS = frozenset(
    "acaba ama ancak artık aslında aynı ayrıca bana bazen bazı belki ben beni benim "
    "beri bile bir bize bizim burada bu bunun çünkü daha da de değil diye dolayısıyla "
    "en fakat gibi göre hakkında hangi hem herkes hiç için ile ise işte kadar karşı "
    "kendi ki kim mi nasıl ne neden nerede o olan olarak oldu onun önce rağmen sana "
    "sen siz sonra şöyle şu taraf tüm ve veya ya yalnız yani yapmak yerine yine yok "
    "zaman çok şimdi bugün dün yarın burada orada böyle gerçekten neden".split()
)
_TR_SUFFIX_RE = re.compile(
    r"(?:lar|ler|dır|dir|dur|dür|tır|tir|tur|tür|dan|den|tan|ten|nin|nın|nun|nün|"
    r"ımız|imiz|umuz|ümüz|acak|ecek|mış|miş|muş|müş|yor|ken|daki|deki|daki|liği|lığı)$",
    re.IGNORECASE,
)
_FOREIGN_FUNCTION_WORDS = frozenset(
    "the and that this with from have are was were for not you your und der die das "
    "ein une des les est pour que con los las una del para como não uma não что это".split()
)
_BOILERPLATE_RE = re.compile(
    r"(?:çerez(?:leri)?\s+kabul|gizlilik\s+politikası|tüm\s+hakları\s+saklıdır|"
    r"javascript(?:i|’i|'i)?\s+etkinleştir|reklamı\s+geç|üyelik\s+sözleşmesi)",
    re.IGNORECASE,
)


class TurkishCorpusError(ValueError):
    """Raised when a corpus contract or document fails closed."""


@dataclass(frozen=True)
class AuditDecision:
    accepted: bool
    reason: str
    normalized_text: str
    metrics: Mapping[str, float | int | str]


@dataclass(frozen=True)
class DedupDecision:
    accepted: bool
    cluster_id: str
    duplicate_kind: str | None


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TurkishCorpusError(f"{label} must be a JSON object")
    return value


def _require_nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TurkishCorpusError(f"{label} must be a trimmed non-empty string")
    return value


def load_corpus_policy(path: str | Path) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, dict):
        raise TurkishCorpusError("corpus policy root must be a JSON object")
    validate_corpus_policy(value)
    return value


def validate_corpus_policy(value: Mapping[str, Any]) -> None:
    """Validate the complete Turkish-only corpus policy.

    Unknown top-level fields are refused so a misspelled safety gate cannot be
    silently ignored.
    """

    policy = _require_mapping(value, "policy")
    required = {
        "schema_version",
        "name",
        "language_policy",
        "content_policy",
        "deduplication",
        "splits",
        "sources",
        "mixture",
        "tokenizer_training",
        "quality_assurance",
        "materialization",
    }
    if set(policy) != required:
        raise TurkishCorpusError(f"policy fields must be exactly {sorted(required)}")
    if policy["schema_version"] != "2.0":
        raise TurkishCorpusError("schema_version must be '2.0'")
    if policy["name"] != CORPUS_NAME:
        raise TurkishCorpusError(f"policy name must be {CORPUS_NAME!r}")

    language = _require_mapping(policy["language_policy"], "language_policy")
    if language.get("allowed") != ["tur_Latn"]:
        raise TurkishCorpusError("language_policy.allowed must be exactly ['tur_Latn']")
    if language.get("require_source_lid") is not True:
        raise TurkishCorpusError("source LID must be required")
    if language.get("require_independent_text_audit") is not True:
        raise TurkishCorpusError("independent Turkish text audit must be required")
    independent = _require_mapping(language.get("independent_audit"), "independent_audit")
    if independent.get("implementation") != "GlotLID-v3-fastText":
        raise TurkishCorpusError("production independent LID must be GlotLID v3")
    if not _SHA1_RE.fullmatch(str(independent.get("hub_revision", ""))):
        raise TurkishCorpusError("GlotLID Hub revision must be a full commit")
    if not _SHA256_RE.fullmatch(str(independent.get("artifact_sha256", ""))):
        raise TurkishCorpusError("GlotLID model SHA-256 must be pinned")
    if independent.get("required_top_label") != "tur_Latn":
        raise TurkishCorpusError("GlotLID required label must be tur_Latn")
    if independent.get("calibration_required_before_seal") is not True:
        raise TurkishCorpusError("GlotLID confusion-set calibration is mandatory")

    content = _require_mapping(policy["content_policy"], "content_policy")
    if content.get("allow_code") is not False:
        raise TurkishCorpusError("content_policy.allow_code must be false")
    for key in ("min_chars", "min_words"):
        if not isinstance(content.get(key), int) or content[key] <= 0:
            raise TurkishCorpusError(f"content_policy.{key} must be positive")
    processing = _require_mapping(
        content.get("production_processing"), "content_policy.production_processing"
    )
    if (
        processing.get("implementation")
        != "datatrove_0_10_0_plus_conservative_turkish_pii_v1"
        or processing.get("language") != "tur_Latn"
    ):
        raise TurkishCorpusError("production processing implementation/language drift")
    control = _require_mapping(
        processing.get("fineweb2_turkish_control"),
        "production_processing.fineweb2_turkish_control",
    )
    if (
        control.get("revision") != "d0defb24f193bb9a5a11b8b14524a03c4858e1b6"
        or control.get("sha256")
        != "f0ccd5fef17c5978f0c8863809dc6a3ec9bededa772f6d25bfa0a4f7f20d67c1"
    ):
        raise TurkishCorpusError("FineWeb-2 Turkish quality-control pin drift")

    dedup = _require_mapping(policy["deduplication"], "deduplication")
    if dedup.get("exact") != "sha256_nfc_whitespace_v1":
        raise TurkishCorpusError("exact dedup policy is not frozen")
    if dedup.get("near") != "minhash_word5_v1":
        raise TurkishCorpusError("near dedup policy is not frozen")
    production_dedup = _require_mapping(
        dedup.get("production_backend"), "deduplication.production_backend"
    )
    if production_dedup.get("implementation") != "huggingface_datatrove_minhash":
        raise TurkishCorpusError("production dedup must use the pinned DataTrove backend")
    if not _SHA1_RE.fullmatch(str(production_dedup.get("git_revision", ""))):
        raise TurkishCorpusError("DataTrove production commit must be pinned")
    reference_dedup = _require_mapping(
        dedup.get("reference_backend"), "deduplication.reference_backend"
    )
    if reference_dedup.get("production_allowed") is not False:
        raise TurkishCorpusError("reference Python dedup must be forbidden in production")
    threshold = reference_dedup.get("similarity_threshold")
    if not isinstance(threshold, (int, float)) or not 0.5 < float(threshold) < 1:
        raise TurkishCorpusError("reference dedup similarity_threshold must be in (0.5, 1)")
    if production_dedup.get("num_buckets") != 14 or production_dedup.get(
        "hashes_per_bucket"
    ) != 8:
        raise TurkishCorpusError("production DataTrove MinHash must be exactly 14x8")
    expected_production_semantics = {
        "signature_language": "tur_Latn",
        "match_rule": "any_equal_8_hash_bucket_signature",
        "candidate_probability_formula": "1-(1-s^8)^14",
        "synthetic_similarity_calibration_required": True,
    }
    for key, expected in expected_production_semantics.items():
        if production_dedup.get(key) != expected:
            raise TurkishCorpusError(f"production MinHash semantics drift at {key}")
    if "jaccard_threshold" in production_dedup:
        raise TurkishCorpusError("DataTrove 14x8 LSH must not claim a hard Jaccard threshold")
    if reference_dedup.get("num_hashes") != 64 or reference_dedup.get("bands") != 16:
        raise TurkishCorpusError("reference MinHash must be exactly 64 hashes/16 bands")

    splits = _require_mapping(policy["splits"], "splits")
    if splits.get("unit") != "dedup_cluster":
        raise TurkishCorpusError("splits.unit must be dedup_cluster")
    fractions = _require_mapping(splits.get("fractions"), "splits.fractions")
    if set(fractions) != {"train", "val", "test"}:
        raise TurkishCorpusError("split fractions must define train/val/test")
    if not math.isclose(sum(float(v) for v in fractions.values()), 1.0, abs_tol=1e-12):
        raise TurkishCorpusError("split fractions must sum to 1")
    if any(float(v) <= 0 for v in fractions.values()):
        raise TurkishCorpusError("every split fraction must be positive")
    _require_nonempty(splits.get("seed"), "splits.seed")

    sources = policy["sources"]
    if not isinstance(sources, list) or not sources:
        raise TurkishCorpusError("sources must be a non-empty array")
    source_ids: set[str] = set()
    for index, raw in enumerate(sources):
        source = _require_mapping(raw, f"sources[{index}]")
        source_id = _require_nonempty(source.get("id"), f"sources[{index}].id")
        if source_id in source_ids:
            raise TurkishCorpusError(f"duplicate source id {source_id!r}")
        source_ids.add(source_id)
        _require_nonempty(source.get("repo_id"), f"sources[{index}].repo_id")
        revision = _require_nonempty(source.get("resolved_revision"), f"sources[{index}].resolved_revision")
        revision_kind = source.get("revision_kind")
        if revision_kind == "hub_commit" and not _SHA1_RE.fullmatch(revision):
            raise TurkishCorpusError(f"{source_id}: Hub revision must be a full commit")
        if revision_kind == "sha256" and not _SHA256_RE.fullmatch(revision):
            raise TurkishCorpusError(f"{source_id}: revision must be SHA-256")
        if revision_kind not in {"hub_commit", "sha256"}:
            raise TurkishCorpusError(f"{source_id}: unsupported revision_kind")
        _require_nonempty(source.get("license_id"), f"{source_id}.license_id")
        _require_nonempty(source.get("license_url"), f"{source_id}.license_url")
        adapter = _require_mapping(source.get("adapter"), f"{source_id}.adapter")
        if adapter.get("text_field") != "text":
            raise TurkishCorpusError(f"{source_id}: adapter.text_field must be text")
    priority = dedup.get("source_priority")
    if not isinstance(priority, list) or len(priority) != len(set(priority)):
        raise TurkishCorpusError("deduplication.source_priority must be a unique array")
    if set(priority) != source_ids:
        raise TurkishCorpusError("deduplication.source_priority must cover every source")

    mixture = policy["mixture"]
    if not isinstance(mixture, list) or not mixture:
        raise TurkishCorpusError("mixture must be a non-empty array")
    weights = 0.0
    bucket_ids: set[str] = set()
    hplt_seen = False
    for index, raw in enumerate(mixture):
        bucket = _require_mapping(raw, f"mixture[{index}]")
        bucket_id = _require_nonempty(bucket.get("id"), f"mixture[{index}].id")
        if bucket_id in bucket_ids:
            raise TurkishCorpusError(f"duplicate mixture id {bucket_id!r}")
        bucket_ids.add(bucket_id)
        if bucket.get("source_id") not in source_ids:
            raise TurkishCorpusError(f"{bucket_id}: unknown source_id")
        weight = bucket.get("weight")
        if not isinstance(weight, (int, float)) or not 0 < float(weight) <= 1:
            raise TurkishCorpusError(f"{bucket_id}: invalid weight")
        weights += float(weight)
        if bucket.get("max_passes") != 1:
            raise TurkishCorpusError(f"{bucket_id}: only one unique-data pass is allowed")
        cap = bucket.get("source_cap")
        if not isinstance(cap, (int, float)) or not float(weight) <= float(cap) <= 1:
            raise TurkishCorpusError(f"{bucket_id}: source_cap must be in [weight, 1]")
        fallback = bucket.get("fallback")
        if not isinstance(fallback, list) or len(fallback) != len(set(fallback)):
            raise TurkishCorpusError(f"{bucket_id}: fallback must be a unique array")
        selector = _require_mapping(bucket.get("selector"), f"{bucket_id}.selector")
        if bucket.get("source_id") == "hplt3_tr":
            hplt_seen = True
            source = next(
                item for item in policy["sources"] if item["id"] == "hplt3_tr"
            )
            adapter = _require_mapping(source.get("adapter"), "HPLT adapter")
            if (
                adapter.get("language_field") != "lang"
                or adapter.get("language_probability_field") != "prob"
                or adapter.get("register_field") != "web-register"
            ):
                raise TurkishCorpusError(
                    "HPLT adapter must bind parallel lang/prob and literal web-register"
                )
            bins = selector.get("wds_bins")
            if not isinstance(bins, list) or not bins or min(bins) < 8 or max(bins) > 10:
                raise TurkishCorpusError("HPLT3 is candidate-only and must use WDS bins 8-10")
            registers = selector.get("register_any")
            if not isinstance(registers, list) or not registers:
                raise TurkishCorpusError("HPLT3 selection requires explicit registers")
            if any(item in {"MT", "LY"} for item in registers):
                raise TurkishCorpusError("machine-translated/lyrical HPLT registers are forbidden")
    if not hplt_seen:
        raise TurkishCorpusError("at least one audited HPLT3 candidate bucket is required")
    if not math.isclose(weights, 1.0, abs_tol=1e-12):
        raise TurkishCorpusError(f"mixture weights must sum to 1, got {weights}")
    for bucket in mixture:
        unknown = set(bucket["fallback"]) - bucket_ids
        if unknown or bucket["id"] in bucket["fallback"]:
            raise TurkishCorpusError(
                f"{bucket['id']}: fallback contains self/unknown buckets {sorted(unknown)}"
            )

    tokenizer = _require_mapping(policy["tokenizer_training"], "tokenizer_training")
    if tokenizer.get("name") != TOKENIZER_NAME or tokenizer.get("vocab_size") != VOCAB_SIZE:
        raise TurkishCorpusError("tokenizer identity must be tr_general_raw_bpe_32k_v1/32768")
    if tokenizer.get("algorithm") != "raw_byte_bpe":
        raise TurkishCorpusError("tokenizer algorithm must be raw_byte_bpe")
    if tokenizer.get("sample_scope") != "post_filter_train_only":
        raise TurkishCorpusError("tokenizer sample must be post-filter train-only")
    if tokenizer.get("max_chars") != 2_000_000_000:
        raise TurkishCorpusError("v1 tokenizer must retain Nanochat's 2B-character default")
    if tokenizer.get("max_chars_per_document") != 10_000:
        raise TurkishCorpusError("v1 tokenizer must retain Nanochat's 10k document cap")
    if tokenizer.get("stop_rule") != (
        "yield_full_capped_document_then_stop_when_cumulative_characters_strictly_exceed_threshold"
    ):
        raise TurkishCorpusError("tokenizer sample must retain pinned upstream stop semantics")

    qa = _require_mapping(policy["quality_assurance"], "quality_assurance")
    if qa.get("schema_version") != "1.0":
        raise TurkishCorpusError("quality assurance schema must be 1.0")
    for key in (
        "examples_per_stratum_and_decision",
        "max_example_characters",
        "quantile_sample_size",
    ):
        if not isinstance(qa.get(key), int) or qa[key] <= 0:
            raise TurkishCorpusError(f"quality_assurance.{key} must be positive")
    quantiles = qa.get("quantiles")
    if (
        not isinstance(quantiles, list)
        or not quantiles
        or quantiles != sorted(set(quantiles))
        or any(not isinstance(value, (int, float)) or not 0 <= float(value) <= 1 for value in quantiles)
    ):
        raise TurkishCorpusError("quality assurance quantiles must be unique sorted [0,1]")
    if qa.get("require_accepted_example_per_mixture") is not True:
        raise TurkishCorpusError("every mixture requires accepted QA examples")
    if qa.get("require_manual_approval_before_tokenizer_or_final") is not True:
        raise TurkishCorpusError("manual QA approval gate must be enabled")

    materialization = _require_mapping(policy["materialization"], "materialization")
    if materialization.get("stream_one_input_shard") is not True:
        raise TurkishCorpusError("materialization must stream one input shard")
    if materialization.get("compression") != "zstd":
        raise TurkishCorpusError("final parquet compression must be zstd")
    for key in (
        "rows_per_fragment",
        "rows_per_output_file",
        "shuffle_buckets",
        "max_buffered_rows",
        "target_family_tokens",
        "max_peak_disk_bytes",
    ):
        if not isinstance(materialization.get(key), int) or materialization[key] <= 0:
            raise TurkishCorpusError(f"materialization.{key} must be positive")
    if materialization["rows_per_output_file"] < materialization["rows_per_fragment"]:
        raise TurkishCorpusError("rows_per_output_file must cover one row group")
    if materialization.get("target_family_tokens_semantics") != (
        "minimum_encoded_source_floor_not_a_model-position-capacity-claim"
    ):
        raise TurkishCorpusError("raw source target must not claim model-position capacity")
    capacity = _require_mapping(
        materialization.get("packing_capacity_gate"), "packing_capacity_gate"
    )
    expected_capacity = {
        "implementation": "nanochat_upstream_bos_bestfit_crop_capacity_v2",
        "world_sizes": [8, 16],
        "device_batch_sequences": 4,
        "max_seq_len": 2048,
        "tokenizer_batch_size": 128,
        "buffer_size": 1000,
        "global_batch_tokens": D32_GLOBAL_BATCH_TOKENS,
        "required_optimizer_steps": 32_000,
        "safety_margin_fraction": 0.02,
        "mix_absolute_tolerance": 0.03,
        "require_all_world_sizes": True,
        "require_upstream_fixture_parity": True,
    }
    if dict(capacity) != expected_capacity:
        raise TurkishCorpusError("packing capacity gate differs from the frozen d32 contract")


def validate_source_receipt(receipt: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    verify_manifest_hash(receipt)
    if receipt.get("schema_version") != "1.0" or receipt.get("kind") != SOURCE_RECEIPT_KIND:
        raise TurkishCorpusError("unexpected source receipt kind/version")
    if receipt.get("policy_sha256") != hashlib.sha256(
        canonical_json(policy).encode("utf-8")
    ).hexdigest():
        raise TurkishCorpusError("source receipt is bound to a different corpus policy")
    policy_sources = {source["id"]: source for source in policy["sources"]}
    seen: set[str] = set()
    sources = receipt.get("sources")
    if not isinstance(sources, list) or not sources:
        raise TurkishCorpusError("source receipt has no sources")
    for index, raw in enumerate(sources):
        source = _require_mapping(raw, f"receipt.sources[{index}]")
        source_id = source.get("id")
        if source_id not in policy_sources or source_id in seen:
            raise TurkishCorpusError("receipt source is unknown or duplicated")
        seen.add(source_id)
        expected = policy_sources[source_id]
        for key in ("repo_id", "resolved_revision", "license_id"):
            if source.get(key) != expected[key]:
                raise TurkishCorpusError(f"{source_id}: receipt {key} drift")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise TurkishCorpusError(f"{source_id}: receipt must list input files")
        for file_index, file_raw in enumerate(files):
            item = _require_mapping(file_raw, f"{source_id}.files[{file_index}]")
            uri = _require_nonempty(item.get("uri"), "receipt file uri")
            if urllib.parse.urlparse(uri).scheme not in {"file", "https"}:
                raise TurkishCorpusError("receipt file URI must use file:// or https://")
            checksum = _require_mapping(item.get("checksum"), "receipt checksum")
            if checksum.get("algorithm") != "sha256" or not _SHA256_RE.fullmatch(
                str(checksum.get("value", ""))
            ):
                raise TurkishCorpusError("every materialization input requires SHA-256")
            if (
                isinstance(item.get("size_bytes"), bool)
                or not isinstance(item.get("size_bytes"), int)
                or item["size_bytes"] <= 0
            ):
                raise TurkishCorpusError("every materialization input requires size_bytes")
    if seen != set(policy_sources):
        raise TurkishCorpusError("receipt does not cover every configured source")


def source_object_inventory(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical source-object inventory bound into backend output."""

    inventory: list[dict[str, Any]] = []
    for source in receipt["sources"]:
        for item in source["files"]:
            inventory.append(
                {
                    "source_id": source["id"],
                    "uri": item["uri"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["checksum"]["value"],
                }
            )
    return sorted(inventory, key=lambda item: (item["source_id"], item["uri"]))


def validate_backend_receipt(
    receipt: Mapping[str, Any],
    policy: Mapping[str, Any],
    source_receipt: Mapping[str, Any] | None = None,
) -> None:
    """Validate the production GlotLID + DataTrove output contract."""

    verify_manifest_hash(receipt)
    if receipt.get("schema_version") != "1.0" or receipt.get("kind") != BACKEND_RECEIPT_KIND:
        raise TurkishCorpusError("unexpected production backend receipt")
    policy_hash = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
    if receipt.get("policy_sha256") != policy_hash:
        raise TurkishCorpusError("production backend receipt is bound to another policy")
    if source_receipt is None:
        raise TurkishCorpusError(
            "production backend verification requires the actual sealed source receipt"
        )
    validate_source_receipt(source_receipt, policy)
    if receipt.get("source_receipt_sha256") != source_receipt["canonical_sha256"]:
        raise TurkishCorpusError("production backend/source receipt hash binding differs")
    expected_inventory = source_object_inventory(source_receipt)
    if receipt.get("source_inventory") != expected_inventory:
        raise TurkishCorpusError("production backend source-object inventory differs")
    inventory_hash = hashlib.sha256(
        canonical_json(expected_inventory).encode("utf-8")
    ).hexdigest()
    if receipt.get("source_inventory_sha256") != inventory_hash:
        raise TurkishCorpusError("production backend source inventory hash differs")
    expected_source_totals = {
        "objects": len(expected_inventory),
        "size_bytes": sum(item["size_bytes"] for item in expected_inventory),
    }
    if receipt.get("source_totals") != expected_source_totals:
        raise TurkishCorpusError("production backend source inventory totals differ")

    expected_lid = policy["language_policy"]["independent_audit"]
    lid = _require_mapping(receipt.get("lid"), "backend_receipt.lid")
    exact_lid_bindings = {
        "implementation": expected_lid["implementation"],
        "repo_id": expected_lid["repo_id"],
        "hub_revision": expected_lid["hub_revision"],
        "artifact": expected_lid["artifact"],
        "artifact_sha256": expected_lid["artifact_sha256"],
        "required_top_label": expected_lid["required_top_label"],
        "document_min_probability": expected_lid["document_min_probability"],
        "document_min_margin": expected_lid["document_min_margin"],
        "paragraph_min_probability": expected_lid["paragraph_min_probability"],
        "paragraph_min_margin": expected_lid["paragraph_min_margin"],
        "max_failed_long_paragraph_fraction": expected_lid[
            "max_failed_long_paragraph_fraction"
        ],
    }
    for key, expected in exact_lid_bindings.items():
        if lid.get(key) != expected:
            raise TurkishCorpusError(f"production LID binding drift at {key}")

    calibration = _require_mapping(receipt.get("lid_calibration"), "lid_calibration")
    if calibration.get("languages") != expected_lid["calibration_languages"]:
        raise TurkishCorpusError("LID calibration language set/order drift")
    if not _SHA256_RE.fullmatch(str(calibration.get("fixture_sha256", ""))):
        raise TurkishCorpusError("LID calibration fixture SHA-256 is required")
    if calibration.get("passed") is not True:
        raise TurkishCorpusError("LID calibration did not pass")
    if not isinstance(calibration.get("metrics"), Mapping) or not calibration["metrics"]:
        raise TurkishCorpusError("LID calibration metrics are required")

    expected_dedup = policy["deduplication"]["production_backend"]
    dedup = _require_mapping(receipt.get("dedup"), "backend_receipt.dedup")
    for key in (
        "implementation",
        "version",
        "git_revision",
        "precision_bits",
        "num_buckets",
        "hashes_per_bucket",
        "ngram_words",
        "signature_language",
        "match_rule",
        "candidate_probability_formula",
        "similarity_at_50_percent_candidate_probability",
        "candidate_probability_at_similarity_0_82",
        "synthetic_similarity_calibration_required",
        "cluster_representative",
    ):
        if dedup.get(key) != expected_dedup[key]:
            raise TurkishCorpusError(f"production dedup binding drift at {key}")
    if dedup.get("global_cross_source") is not True:
        raise TurkishCorpusError("production dedup must be global and cross-source")
    if dedup.get("winner_priority") != policy["deduplication"]["source_priority"]:
        raise TurkishCorpusError("production dedup winner priority drift")
    calibration = _require_mapping(
        dedup.get("synthetic_similarity_calibration"),
        "backend_receipt.dedup.synthetic_similarity_calibration",
    )
    if calibration.get("passed") is not True or not _SHA256_RE.fullmatch(
        str(calibration.get("receipt_sha256", ""))
    ):
        raise TurkishCorpusError("production MinHash synthetic calibration is missing")
    tokenizer_probe = _require_mapping(
        dedup.get("signature_tokenizer_probe"),
        "backend_receipt.dedup.signature_tokenizer_probe",
    )
    if (
        tokenizer_probe.get("language") != "tur_Latn"
        or tokenizer_probe.get("passed") is not True
        or not _SHA256_RE.fullmatch(str(tokenizer_probe.get("probe_sha256", "")))
    ):
        raise TurkishCorpusError("DataTrove signature tokenizer is not proven Turkish")

    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise TurkishCorpusError("production backend receipt has no output files")
    required_columns = {
        "text",
        "source_id",
        "document_id",
        "url",
        "source_lid_label",
        "source_lid_probability",
        "lid_label",
        "lid_probability",
        "lid_margin",
        "paragraph_min_probability",
        "paragraph_min_margin",
        "failed_long_paragraph_fraction",
        "dedup_cluster_id",
        "dedup_keep",
        "quality_score",
        "wds_bin",
        "web-register",
        "pii_replacements",
        "harmful_signal_hits",
        "quality_filter_flags",
        "formatting_changes",
    }
    if set(receipt.get("columns", [])) != required_columns:
        raise TurkishCorpusError("production backend output column contract drift")
    for item in files:
        item = _require_mapping(item, "backend file")
        if urllib.parse.urlparse(str(item.get("uri", ""))).scheme not in {"file", "https"}:
            raise TurkishCorpusError("backend file URI must use file:// or https://")
        checksum = _require_mapping(item.get("checksum"), "backend checksum")
        if checksum.get("algorithm") != "sha256" or not _SHA256_RE.fullmatch(
            str(checksum.get("value", ""))
        ):
            raise TurkishCorpusError("backend files require SHA-256")
        for key in ("size_bytes", "rows"):
            if (
                isinstance(item.get(key), bool)
                or not isinstance(item.get(key), int)
                or item[key] <= 0
            ):
                raise TurkishCorpusError(f"backend files require positive {key}")
    expected_output_totals = {
        "files": len(files),
        "rows": sum(item["rows"] for item in files),
        "size_bytes": sum(item["size_bytes"] for item in files),
    }
    if receipt.get("output_totals") != expected_output_totals:
        raise TurkishCorpusError("production backend output totals differ from file inventory")
    processing = _require_mapping(receipt.get("processing"), "backend_receipt.processing")
    if not _SHA256_RE.fullmatch(str(processing.get("binding_sha256", ""))):
        raise TurkishCorpusError("production processing binding hash is missing")
    official = _require_mapping(
        processing.get("official_fineweb2_control"),
        "backend_receipt.processing.official_fineweb2_control",
    )
    if (
        official.get("revision") != "d0defb24f193bb9a5a11b8b14524a03c4858e1b6"
        or official.get("config_sha256")
        != "f0ccd5fef17c5978f0c8863809dc6a3ec9bededa772f6d25bfa0a4f7f20d67c1"
    ):
        raise TurkishCorpusError("backend FineWeb-2 Turkish control binding drift")
    if not isinstance(receipt.get("quality_filter_stage_counts"), Mapping):
        raise TurkishCorpusError("backend quality stage counters are required")
    if not isinstance(receipt.get("formatting_and_safety_incidence"), Mapping):
        raise TurkishCorpusError("backend formatting/PII/harm incidence is required")
    cleanup = _require_mapping(
        receipt.get("streaming_import_cleanup"), "streaming_import_cleanup"
    )
    if cleanup.get("backend_files_are_run_owned") is not True:
        raise TurkishCorpusError("backend files lack run-owned cleanup authorization")


class _DeterministicNumericSample:
    """Fixed-memory, hash-selected numeric sample for reproducible quantiles."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.heap: list[tuple[int, float]] = []
        self.seen = 0

    def add(self, identity: str, value: Any) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(numeric):
            return
        self.seen += 1
        rank = int.from_bytes(
            hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big"
        )
        item = (-rank, numeric)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def summary(self, quantiles: Sequence[float]) -> dict[str, Any]:
        values = sorted(value for _rank, value in self.heap)
        result: dict[str, float | int] = {
            "observations": self.seen,
            "deterministic_sample_size": len(values),
        }
        if not values:
            return result
        result["minimum"] = values[0]
        result["maximum"] = values[-1]
        result["mean_of_sample"] = sum(values) / len(values)
        for quantile in quantiles:
            position = round(float(quantile) * (len(values) - 1))
            result[f"q{float(quantile):.4f}"] = values[position]
        return result


class _DeterministicExampleSample:
    """Keep the lexically smallest content hashes for every audit stratum."""

    def __init__(self, capacity: int, max_characters: int) -> None:
        self.capacity = capacity
        self.max_characters = max_characters
        self.heaps: dict[tuple[str, str, str], list[tuple[int, str, str]]] = defaultdict(list)

    def add(
        self,
        *,
        source_id: str,
        mixture_id: str,
        decision: str,
        reason: str,
        document_id: str,
        url: str,
        text: str,
        metrics: Mapping[str, Any],
    ) -> None:
        identity = hashlib.sha256(
            f"{source_id}\0{mixture_id}\0{decision}\0{reason}\0{document_id}".encode(
                "utf-8"
            )
        ).hexdigest()
        rank = int(identity[:16], 16)
        payload = {
            "sample_sha256": identity,
            "source_id": source_id,
            "mixture_id": mixture_id,
            "decision": decision,
            "reason": reason,
            "document_id": document_id,
            "url": url,
            "text": text[: self.max_characters],
            "text_truncated": len(text) > self.max_characters,
            "metrics": dict(sorted(metrics.items())),
        }
        key = (source_id, mixture_id, decision)
        item = (-rank, identity, canonical_json(payload))
        heap = self.heaps[key]
        if len(heap) < self.capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def rows(self) -> list[dict[str, Any]]:
        rows = [
            json.loads(payload)
            for heap in self.heaps.values()
            for _rank, _identity, payload in heap
        ]
        return sorted(
            rows,
            key=lambda row: (
                row["source_id"],
                row["mixture_id"],
                row["decision"],
                row["sample_sha256"],
            ),
        )


def _qa_document_metrics(record: Mapping[str, Any]) -> tuple[str, dict[str, float | int | str]]:
    text = normalize_document(str(record.get("text") or ""))
    words = [word.casefold() for word in _WORD_RE.findall(text)]
    lines = [line for line in text.split("\n") if line.strip()]
    code_lines = sum(bool(_CODE_LINE_RE.search(line)) for line in lines)
    metrics: dict[str, float | int | str] = {
        "characters": len(text),
        "words": len(words),
        "code_line_fraction": code_lines / max(1, len(lines)),
        "code_punctuation_fraction": sum(text.count(char) for char in "{};`")
        / max(1, len(text)),
        "programming_term_hits": len(_PROGRAMMING_RE.findall(text)),
        "boilerplate_hits": len(_BOILERPLATE_RE.findall(text)),
        "duplicate_line_fraction": (
            1.0 - len(set(lines)) / len(lines) if len(lines) >= 4 else 0.0
        ),
        "quality_filter_flags": str(record.get("quality_filter_flags") or "[]"),
        "formatting_changes": str(record.get("formatting_changes") or "{}"),
    }
    for key in (
        "lid_probability",
        "lid_margin",
        "paragraph_min_probability",
        "paragraph_min_margin",
        "failed_long_paragraph_fraction",
        "quality_score",
        "pii_replacements",
        "harmful_signal_hits",
    ):
        try:
            metrics[key] = float(record.get(key))
        except (TypeError, ValueError):
            pass
    wds_bin = infer_wds_bin(record)
    if wds_bin is not None:
        metrics["wds_bin"] = wds_bin
    metrics["register_bucket"] = dominant_register(record)
    return text, metrics


class ProductionQAAuditor:
    """Bounded, deterministic stratified audit of backend candidates."""

    _NUMERIC_METRICS = (
        "characters",
        "words",
        "lid_probability",
        "lid_margin",
        "paragraph_min_probability",
        "paragraph_min_margin",
        "failed_long_paragraph_fraction",
        "quality_score",
        "code_line_fraction",
        "code_punctuation_fraction",
        "programming_term_hits",
        "boilerplate_hits",
        "duplicate_line_fraction",
        "pii_replacements",
        "harmful_signal_hits",
    )

    def __init__(self, policy: Mapping[str, Any]) -> None:
        qa = policy["quality_assurance"]
        self.policy = policy
        self.quantiles = tuple(float(value) for value in qa["quantiles"])
        self.sample_capacity = int(qa["quantile_sample_size"])
        self.examples = _DeterministicExampleSample(
            int(qa["examples_per_stratum_and_decision"]),
            int(qa["max_example_characters"]),
        )
        self.counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.numeric: dict[
            tuple[str, str], dict[str, _DeterministicNumericSample]
        ] = defaultdict(dict)
        self.registers: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.wds_bins: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    def observe(
        self,
        record: Mapping[str, Any],
        *,
        source_id: str,
        mixture_id: str,
        accepted: bool,
        reason: str,
    ) -> None:
        text, metrics = _qa_document_metrics(record)
        document_id = str(record.get("document_id") or canonical_text_hash(text))
        stratum = (source_id, mixture_id)
        decision = "accepted" if accepted else "rejected"
        self.counts[stratum]["seen"] += 1
        self.counts[stratum][decision] += 1
        self.counts[stratum][f"reason:{reason}"] += 1
        if reason == "backend_duplicate":
            self.counts[stratum]["duplicates"] += 1
        register = str(metrics.get("register_bucket", "not_applicable"))
        self.registers[stratum][register] += 1
        if "wds_bin" in metrics:
            self.wds_bins[stratum][str(metrics["wds_bin"])] += 1
        for metric in self._NUMERIC_METRICS:
            if metric not in metrics:
                continue
            sketch = self.numeric[stratum].setdefault(
                metric, _DeterministicNumericSample(self.sample_capacity)
            )
            sketch.add(f"{document_id}\0{metric}", metrics[metric])
        self.examples.add(
            source_id=source_id,
            mixture_id=mixture_id,
            decision=decision,
            reason=reason,
            document_id=document_id,
            url=str(record.get("url") or ""),
            text=text,
            metrics=metrics,
        )

    def write(
        self,
        destination: Path,
        *,
        backend_receipt: Mapping[str, Any],
        policy_sha256: str,
    ) -> dict[str, Any]:
        qa_dir = destination / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)
        rows = self.examples.rows()
        jsonl_path = qa_dir / "qa_examples.jsonl"
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row))
                handle.write("\n")
        text_path = qa_dir / "qa_examples.txt"
        with text_path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows, 1):
                handle.write(
                    f"[{index}] {row['source_id']} / {row['mixture_id']} / "
                    f"{row['decision']} / {row['reason']}\n"
                )
                handle.write(f"sample_sha256: {row['sample_sha256']}\n")
                handle.write(f"url: {row['url']}\n")
                handle.write(row["text"])
                handle.write("\n\n")

        strata: dict[str, Any] = {}
        for stratum in sorted(self.counts):
            source_id, mixture_id = stratum
            key = f"{source_id}/{mixture_id}"
            seen = self.counts[stratum]["seen"]
            strata[key] = {
                "source_id": source_id,
                "mixture_id": mixture_id,
                "counts": dict(sorted(self.counts[stratum].items())),
                "duplicate_rate": self.counts[stratum]["duplicates"] / max(1, seen),
                "register_distribution": dict(sorted(self.registers[stratum].items())),
                "wds_bin_distribution": dict(sorted(self.wds_bins[stratum].items())),
                "numeric_distributions": {
                    metric: sketch.summary(self.quantiles)
                    for metric, sketch in sorted(self.numeric[stratum].items())
                },
            }
        required_examples = self.examples.capacity
        missing = [
            bucket["id"]
            for bucket in self.policy["mixture"]
            if self.counts[(bucket["source_id"], bucket["id"])]["accepted"]
            < required_examples
        ]
        sparse_rejected = [
            bucket["id"]
            for bucket in self.policy["mixture"]
            if self.counts[(bucket["source_id"], bucket["id"])]["rejected"]
            < required_examples
        ]
        report = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": "turkish_pretrain_stratified_qa_report",
                "policy_sha256": policy_sha256,
                "backend_receipt_sha256": backend_receipt["canonical_sha256"],
                "automated_gate_passed": not missing,
                "missing_accepted_mixture_samples": missing,
                "mixtures_with_fewer_than_requested_rejected_examples": sparse_rejected,
                "sampling": {
                    "method": "smallest_sha256_per_stratum_decision",
                    "examples_per_stratum_and_decision": self.examples.capacity,
                    "max_example_characters": self.examples.max_characters,
                    "quantile_method": "nearest_over_smallest_sha256_sample",
                    "quantile_sample_size": self.sample_capacity,
                    "quantiles": list(self.quantiles),
                },
                "thresholds": {
                    "language_policy": self.policy["language_policy"],
                    "content_policy": self.policy["content_policy"],
                    "deduplication": self.policy["deduplication"],
                },
                "lid_calibration": backend_receipt["lid_calibration"],
                "production_processing": backend_receipt["processing"],
                "backend_quality_filter_stage_counts": backend_receipt[
                    "quality_filter_stage_counts"
                ],
                "backend_formatting_and_safety_incidence": backend_receipt[
                    "formatting_and_safety_incidence"
                ],
                "strata": strata,
                "examples": {
                    "rows": len(rows),
                    "jsonl": {
                        "path": "qa_examples.jsonl",
                        "size_bytes": jsonl_path.stat().st_size,
                        "sha256": file_sha256(jsonl_path),
                    },
                    "plaintext": {
                        "path": "qa_examples.txt",
                        "size_bytes": text_path.stat().st_size,
                        "sha256": file_sha256(text_path),
                    },
                },
                "manual_review_required": True,
                "canonical_sha256": None,
            }
        )
        write_json_atomic(qa_dir / "qa_report.json", report)
        return report


def validate_qa_gate(pool_dir: str | Path, pool_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Require the sealed automated report and its explicit manual approval."""

    root = Path(pool_dir)
    qa_record = _require_mapping(pool_manifest.get("quality_assurance"), "quality_assurance")
    report = load_json_strict(root / str(qa_record.get("report_path", "")))
    report_hash = verify_manifest_hash(report)
    if report_hash != qa_record.get("report_sha256") or report.get(
        "automated_gate_passed"
    ) is not True:
        raise TurkishCorpusError("stratified QA automated gate is absent or failed")
    for key in ("jsonl", "plaintext"):
        record = report["examples"][key]
        path = root / "qa" / record["path"]
        if path.stat().st_size != record["size_bytes"] or file_sha256(path) != record["sha256"]:
            raise TurkishCorpusError(f"stratified QA {key} examples drift")
    approval = load_json_strict(root / "qa" / "qa_approval.json")
    verify_manifest_hash(approval)
    if (
        approval.get("kind") != "turkish_pretrain_qa_approval"
        or approval.get("qa_report_sha256") != report_hash
        or approval.get("pool_manifest_sha256") != pool_manifest["canonical_sha256"]
        or approval.get("decision") != "accepted"
    ):
        raise TurkishCorpusError("manual QA approval is absent, rejected, or stale")
    if not isinstance(approval.get("reviewer"), str) or not approval["reviewer"].strip():
        raise TurkishCorpusError("manual QA approval reviewer is missing")
    if not _RFC3339_UTC_RE.fullmatch(str(approval.get("reviewed_at_utc", ""))):
        raise TurkishCorpusError("manual QA approval timestamp is invalid")
    expected_reviewed = {
        key: report["examples"][key]["sha256"] for key in ("jsonl", "plaintext")
    }
    if approval.get("reviewed_files") != expected_reviewed:
        raise TurkishCorpusError("manual QA approval reviewed hashes differ")
    return approval


def seal_qa_approval(
    pool_dir: str | Path,
    *,
    reviewer: str,
    reviewed_at_utc: str,
    decision: str,
    notes: str = "",
) -> dict[str, Any]:
    """Seal an operator's decision after reviewing both deterministic QA files."""

    if decision not in {"accepted", "rejected"}:
        raise TurkishCorpusError("QA decision must be accepted or rejected")
    reviewer = _require_nonempty(reviewer, "reviewer")
    if not _RFC3339_UTC_RE.fullmatch(reviewed_at_utc):
        raise TurkishCorpusError("reviewed_at_utc must be YYYY-MM-DDTHH:MM:SSZ")
    if not isinstance(notes, str):
        raise TurkishCorpusError("QA notes must be a string")
    root = Path(pool_dir)
    manifest = load_json_strict(root / "corpus_manifest.json")
    pool_hash = verify_manifest_hash(manifest)
    report = load_json_strict(root / manifest["quality_assurance"]["report_path"])
    report_hash = verify_manifest_hash(report)
    if report.get("automated_gate_passed") is not True:
        raise TurkishCorpusError("cannot approve a failed automated QA report")
    path = root / "qa" / "qa_approval.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    approval = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_pretrain_qa_approval",
            "pool_manifest_sha256": pool_hash,
            "qa_report_sha256": report_hash,
            "reviewer": reviewer,
            "reviewed_at_utc": reviewed_at_utc,
            "decision": decision,
            "notes": notes,
            "reviewed_files": {
                key: report["examples"][key]["sha256"]
                for key in ("jsonl", "plaintext")
            },
            "canonical_sha256": None,
        }
    )
    write_json_atomic(path, approval)
    return approval


def write_pool_ownership_manifest(
    pool_dir: str | Path,
    pool_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Mark only the verified fragment inventory as disposable build output."""

    root = Path(pool_dir)
    records = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in pool_manifest["files"]
    ]
    ownership = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": POOL_OWNERSHIP_KIND,
            "pool_manifest_sha256": pool_manifest["canonical_sha256"],
            "generated_fragment_files": records,
            "cleanup_authorized_after_verified_promotion": True,
            "canonical_sha256": None,
        }
    )
    path = root / POOL_OWNERSHIP_FILE
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    write_json_atomic(path, ownership)
    return ownership


def validate_pool_ownership(
    pool_dir: str | Path,
    pool_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(pool_dir)
    ownership = load_json_strict(root / POOL_OWNERSHIP_FILE)
    verify_manifest_hash(ownership)
    if (
        ownership.get("kind") != POOL_OWNERSHIP_KIND
        or ownership.get("pool_manifest_sha256") != pool_manifest["canonical_sha256"]
        or ownership.get("cleanup_authorized_after_verified_promotion") is not True
    ):
        raise TurkishCorpusError("filtered-pool run ownership is absent or stale")
    expected = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in pool_manifest["files"]
    ]
    if ownership.get("generated_fragment_files") != expected:
        raise TurkishCorpusError("run-owned fragment inventory differs from pool manifest")
    for item in expected:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "pool":
            raise TurkishCorpusError("run-owned fragment path escapes pool/ scope")
        path = root / relative
        if (
            path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise TurkishCorpusError(f"run-owned fragment drift: {item['path']}")
    return ownership


def _production_lid_ok(record: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    expected = policy["language_policy"]["independent_audit"]
    try:
        return (
            record["lid_label"] == expected["required_top_label"]
            and float(record["lid_probability"]) >= float(expected["document_min_probability"])
            and float(record["lid_margin"]) >= float(expected["document_min_margin"])
            and float(record["paragraph_min_probability"])
            >= float(expected["paragraph_min_probability"])
            and float(record["paragraph_min_margin"])
            >= float(expected["paragraph_min_margin"])
            and float(record["failed_long_paragraph_fraction"])
            <= float(expected["max_failed_long_paragraph_fraction"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _production_quality_ok(record: Mapping[str, Any]) -> bool:
    try:
        flags = json.loads(str(record.get("quality_filter_flags") or "[]"))
    except json.JSONDecodeError:
        return False
    return flags == []


def materialize_production_pool(
    policy: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    backend_receipt: Mapping[str, Any],
    output_dir: str | Path,
    *,
    git_commit: str,
    request_get: Any = requests.get,
) -> dict[str, Any]:
    """Import verified scalable-backend output into deterministic corpus splits."""

    validate_corpus_policy(policy)
    validate_backend_receipt(backend_receipt, policy, source_receipt)
    if not _SHA1_RE.fullmatch(git_commit) and not _SHA256_RE.fullmatch(git_commit):
        raise TurkishCorpusError("git_commit must be a full Git commit")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    build_dir = destination / ".build"
    build_dir.mkdir()
    writer = FragmentWriter(
        destination,
        rows_per_fragment=int(policy["materialization"]["rows_per_fragment"]),
        buckets=int(policy["materialization"]["shuffle_buckets"]),
        max_buffered_rows=int(policy["materialization"]["max_buffered_rows"]),
        rows_per_output_file=int(policy["materialization"]["rows_per_output_file"]),
    )
    counts: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_mixture: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_register: Counter[str] = Counter()
    source_policies = {source["id"]: source for source in policy["sources"]}
    required_columns = list(backend_receipt["columns"])
    qa_auditor = ProductionQAAuditor(policy)
    try:
        for item in backend_receipt["files"]:
            staged = stage_receipt_file(item, build_dir, request_get=request_get)
            input_rows = 0
            try:
                for record in iter_input_records(staged, columns=required_columns):
                    input_rows += 1
                    counts["seen"] += 1
                    source_id = str(record.get("source_id") or "unknown")
                    candidate = (
                        select_mixture_bucket(source_id, record, policy)
                        if source_id in source_policies
                        else None
                    )
                    candidate_mixture = candidate[0] if candidate else "unrouted"
                    if source_id not in source_policies:
                        counts["reject:unknown_source"] += 1
                        qa_auditor.observe(
                            record,
                            source_id=source_id,
                            mixture_id=candidate_mixture,
                            accepted=False,
                            reason="unknown_source",
                        )
                        continue
                    if record.get("dedup_keep") is not True:
                        counts["reject:backend_duplicate"] += 1
                        qa_auditor.observe(
                            record,
                            source_id=source_id,
                            mixture_id=candidate_mixture,
                            accepted=False,
                            reason="backend_duplicate",
                        )
                        continue
                    if not _production_lid_ok(record, policy):
                        counts["reject:production_lid"] += 1
                        qa_auditor.observe(
                            record,
                            source_id=source_id,
                            mixture_id=candidate_mixture,
                            accepted=False,
                            reason="production_lid",
                        )
                        continue
                    if not _production_quality_ok(record):
                        counts["reject:production_quality"] += 1
                        qa_auditor.observe(
                            record,
                            source_id=source_id,
                            mixture_id=candidate_mixture,
                            accepted=False,
                            reason="production_quality",
                        )
                        continue
                    source_policy = source_policies[source_id]
                    adapter = source_policy["adapter"]
                    source_label = record.get("source_lid_label")
                    source_probability = record.get("source_lid_probability")
                    source_lid_ok = source_label in set(
                        adapter.get("turkish_values", ["tur", "tur_Latn"])
                    )
                    if source_probability is not None:
                        try:
                            source_lid_ok = source_lid_ok and float(source_probability) >= float(
                                adapter.get("source_lid_min_probability", 0.0)
                            )
                        except (TypeError, ValueError):
                            source_lid_ok = False
                    audit = audit_document(
                        record.get("text"),
                        url=str(record.get("url") or ""),
                        source_lid_ok=source_lid_ok,
                        content_policy=policy["content_policy"],
                    )
                    if not audit.accepted:
                        counts[f"reject:secondary_{audit.reason}"] += 1
                        qa_auditor.observe(
                            record,
                            source_id=source_id,
                            mixture_id=candidate_mixture,
                            accepted=False,
                            reason=f"secondary_{audit.reason}",
                        )
                        continue
                    cluster_id = str(record.get("dedup_cluster_id") or "")
                    if not _SHA256_RE.fullmatch(cluster_id):
                        raise TurkishCorpusError("production dedup_cluster_id must be SHA-256")
                    if candidate is None:
                        counts["reject:mixture_selector"] += 1
                        qa_auditor.observe(
                            record,
                            source_id=source_id,
                            mixture_id="unrouted",
                            accepted=False,
                            reason="mixture_selector",
                        )
                        continue
                    mixture_id, selector_quality = candidate
                    document_id = str(record.get("document_id") or canonical_text_hash(audit.normalized_text))
                    split = assign_split(cluster_id, policy["splits"])
                    writer.add(
                        split,
                        mixture_id,
                        {
                            "text": audit.normalized_text,
                            "source_id": source_id,
                            "mixture_id": mixture_id,
                            "document_id": document_id,
                            "url": str(record.get("url") or ""),
                            "cluster_id": cluster_id,
                            "shuffle_key": stable_shuffle_key(document_id, policy["splits"]["seed"]),
                            "quality_score": max(
                                float(record.get("quality_score") or 0.0), selector_quality
                            ),
                            "register_bucket": dominant_register(record),
                        },
                    )
                    counts["accepted"] += 1
                    by_split[split] += 1
                    by_mixture[mixture_id] += 1
                    by_source[source_id] += 1
                    by_register[dominant_register(record)] += 1
                    qa_auditor.observe(
                        record,
                        source_id=source_id,
                        mixture_id=mixture_id,
                        accepted=True,
                        reason="accepted",
                    )
            finally:
                staged.unlink(missing_ok=True)
            if input_rows != item["rows"]:
                raise TurkishCorpusError(
                    f"backend output row-count drift for {item['uri']}: "
                    f"expected {item['rows']}, got {input_rows}"
                )
        files = writer.close()
        policy_hash = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
        qa_report = qa_auditor.write(
            destination,
            backend_receipt=backend_receipt,
            policy_sha256=policy_hash,
        )
        if qa_report["automated_gate_passed"] is not True:
            raise TurkishCorpusError(
                "stratified QA has no accepted audited sample for configured buckets: "
                f"{qa_report['missing_accepted_mixture_samples']}"
            )
        manifest = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": CORPUS_MANIFEST_KIND,
                "name": CORPUS_NAME,
                "stage": "filtered_pool",
                "backend_scope": "production_glotlid_datatrove",
                "policy_sha256": policy_hash,
                "source_receipt_sha256": backend_receipt["source_receipt_sha256"],
                "backend_receipt_sha256": backend_receipt["canonical_sha256"],
                "source_provenance": [
                    {
                        "id": source["id"],
                        "repo_id": source["repo_id"],
                        "resolved_revision": source["resolved_revision"],
                        "license_id": source["license_id"],
                    }
                    for source in policy["sources"]
                ],
                "independent_lid": dict(backend_receipt["lid"]),
                "lid_calibration": dict(backend_receipt["lid_calibration"]),
                "deduplication": dict(backend_receipt["dedup"]),
                "split_policy": policy["splits"],
                "accepted_documents_by_split": dict(sorted(by_split.items())),
                "accepted_documents_by_mixture": dict(sorted(by_mixture.items())),
                "accepted_documents_by_source": dict(sorted(by_source.items())),
                "accepted_documents_by_register": dict(sorted(by_register.items())),
                "audit_counts": dict(sorted(counts.items())),
                "quality_assurance": {
                    "report_path": "qa/qa_report.json",
                    "report_sha256": qa_report["canonical_sha256"],
                    "automated_gate_passed": True,
                    "manual_approval_required": True,
                },
                "writer_metrics": {
                    "max_buffered_rows": writer.max_buffered_rows,
                    "peak_buffered_rows": writer.peak_buffered_rows,
                },
                "files": files,
                "canonical_sha256": None,
            }
        )
        write_json_atomic(destination / "corpus_manifest.json", manifest)
        write_pool_ownership_manifest(destination, manifest)
        return manifest
    finally:
        try:
            build_dir.rmdir()
        except OSError:
            pass


def normalize_document(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_RE.sub("\n\n", text).strip()


def _script_metrics(text: str) -> tuple[int, int]:
    alphabetic = 0
    disallowed_script = 0
    for char in text:
        if not char.isalpha():
            continue
        alphabetic += 1
        name = unicodedata.name(char, "")
        if name and not any(tag in name for tag in ("LATIN", "COMBINING")):
            disallowed_script += 1
    return alphabetic, disallowed_script


def turkish_text_confidence(text: str) -> tuple[float, Mapping[str, float | int]]:
    words = [word.casefold() for word in _WORD_RE.findall(text)]
    if not words:
        return 0.0, {"word_count": 0, "turkish_function_hits": 0, "foreign_hits": 0}
    tr_hits = sum(word in _TR_FUNCTION_WORDS for word in words)
    foreign_hits = sum(word in _FOREIGN_FUNCTION_WORDS for word in words)
    suffix_hits = sum(bool(_TR_SUFFIX_RE.search(word)) for word in words if len(word) >= 5)
    turkish_letters = sum(char in "çğıöşüÇĞİÖŞÜ" for char in text)
    alphabetic, foreign_script = _script_metrics(text)
    lexical = min(1.0, (tr_hits + 0.35 * suffix_hits) / max(3.0, len(words) * 0.055))
    orthographic = min(1.0, turkish_letters / max(1.0, len(words) * 0.035))
    foreign_penalty = min(1.0, foreign_hits / max(2.0, tr_hits + 1.0))
    script_penalty = foreign_script / max(1, alphabetic)
    confidence = max(
        0.0,
        min(1.0, 0.72 * lexical + 0.28 * orthographic - 0.55 * foreign_penalty - script_penalty),
    )
    return confidence, {
        "word_count": len(words),
        "turkish_function_hits": tr_hits,
        "turkish_suffix_hits": suffix_hits,
        "turkish_letter_count": turkish_letters,
        "foreign_function_hits": foreign_hits,
        "foreign_script_fraction": script_penalty,
    }


def source_lid_result(
    record: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    strict_schema: bool = False,
) -> tuple[str, float, bool]:
    """Extract one source-LID decision without losing parallel-list alignment.

    HPLT3 stores ``lang`` and ``prob`` as parallel lists.  Treating either as a
    scalar (or accepting unequal lists) can attach a Turkish label to another
    language's probability.  Strict production callers raise on malformed
    schema; reference/audit callers fail the individual record closed.
    """

    def malformed(message: str) -> tuple[str, float, bool]:
        if strict_schema:
            raise TurkishCorpusError(message)
        return "malformed_source_lid", 0.0, False

    field = adapter.get("language_field")
    if not isinstance(field, str) or not field:
        return malformed("source LID adapter has no language field")
    value = (
        adapter.get("partition_language")
        if field == "$partition"
        else record.get(field)
    )
    allowed = set(adapter.get("turkish_values", ["tr", "tur", "tur_Latn"]))
    threshold = float(adapter.get("source_lid_min_probability", 0.8))
    if isinstance(value, str):
        probability_field = adapter.get("language_probability_field")
        if isinstance(probability_field, str) and probability_field:
            try:
                raw_probability = record.get(probability_field)
                if isinstance(raw_probability, (list, tuple, bool)):
                    return malformed("scalar source LID label has non-scalar probability")
                probability = float(raw_probability)
            except (TypeError, ValueError):
                return malformed("source LID probability is not numeric")
        else:
            probability = 1.0
        if (
            not math.isfinite(probability)
            or probability < -_FASTTEXT_PROBABILITY_EPSILON
            or probability > 1.0 + _FASTTEXT_PROBABILITY_EPSILON
        ):
            return malformed(
                f"source LID probability is outside the fastText tolerance: {probability!r}"
            )
        probability = min(1.0, max(0.0, probability))
        return value, probability, value in allowed and probability >= threshold
    if isinstance(value, list):
        probabilities = record.get(adapter.get("language_probability_field", "prob"))
        if not isinstance(probabilities, list) or len(probabilities) != len(value) or not value:
            return malformed("parallel source LID label/probability lists are misaligned")
        pairs: list[tuple[str, float]] = []
        for label, raw_probability in zip(value, probabilities, strict=True):
            if not isinstance(label, str) or isinstance(raw_probability, bool):
                return malformed("parallel source LID lists contain invalid element types")
            try:
                probability = float(raw_probability)
            except (TypeError, ValueError):
                return malformed("parallel source LID probability is not numeric")
            if (
                not math.isfinite(probability)
                or probability < -_FASTTEXT_PROBABILITY_EPSILON
                or probability > 1.0 + _FASTTEXT_PROBABILITY_EPSILON
            ):
                return malformed(
                    "parallel source LID probability is outside the fastText "
                    f"tolerance: {probability!r}"
                )
            probability = min(1.0, max(0.0, probability))
            pairs.append((label, probability))
        turkish = [pair for pair in pairs if pair[0] in allowed]
        if turkish:
            # Highest Turkish probability wins; original list position resolves ties.
            label, probability = max(turkish, key=lambda pair: pair[1])
            return label, probability, probability >= threshold
        label, probability = max(pairs, key=lambda pair: pair[1])
        return label, probability, False
    return malformed("source LID label is neither a string nor a parallel list")


def source_lid_is_turkish(record: Mapping[str, Any], adapter: Mapping[str, Any]) -> bool:
    return source_lid_result(record, adapter)[2]


def audit_document(
    text: Any,
    *,
    url: str = "",
    source_lid_ok: bool,
    content_policy: Mapping[str, Any],
) -> AuditDecision:
    """Apply the independent Turkish, quality, and no-code gates."""

    if not isinstance(text, str):
        return AuditDecision(False, "not_text", "", {})
    normalized = normalize_document(text)
    char_count = len(normalized)
    if char_count < int(content_policy["min_chars"]):
        return AuditDecision(False, "too_short", normalized, {"char_count": char_count})
    if char_count > int(content_policy["max_chars"]):
        return AuditDecision(False, "too_long", normalized, {"char_count": char_count})
    if not source_lid_ok:
        return AuditDecision(False, "source_lid", normalized, {"char_count": char_count})

    confidence, language_metrics = turkish_text_confidence(normalized)
    metrics: dict[str, float | int | str] = {"char_count": char_count, **language_metrics}
    metrics["turkish_confidence"] = confidence
    if confidence < float(content_policy["min_turkish_confidence"]):
        return AuditDecision(False, "independent_lid", normalized, metrics)

    words = [word.casefold() for word in _WORD_RE.findall(normalized)]
    if len(words) < int(content_policy["min_words"]):
        return AuditDecision(False, "too_few_words", normalized, metrics)
    alphabetic, _foreign_script = _script_metrics(normalized)
    alpha_fraction = alphabetic / max(1, char_count)
    metrics["alphabetic_fraction"] = alpha_fraction
    if alpha_fraction < float(content_policy["min_alphabetic_fraction"]):
        return AuditDecision(False, "low_alphabetic_fraction", normalized, metrics)

    lines = [line for line in normalized.split("\n") if line.strip()]
    code_lines = sum(bool(_CODE_LINE_RE.search(line)) for line in lines)
    brace_semicolon = sum(normalized.count(char) for char in "{};`") / max(1, char_count)
    programming_hits = len(_PROGRAMMING_RE.findall(normalized))
    code_line_fraction = code_lines / max(1, len(lines))
    metrics.update(
        {
            "code_line_fraction": code_line_fraction,
            "code_punctuation_fraction": brace_semicolon,
            "programming_term_hits": programming_hits,
        }
    )
    lowered_url = url.casefold()
    if any(host in lowered_url for host in _CODE_HOSTS):
        return AuditDecision(False, "code_domain", normalized, metrics)
    if (
        code_line_fraction > float(content_policy["max_code_line_fraction"])
        or brace_semicolon > float(content_policy["max_code_punctuation_fraction"])
        or programming_hits > int(content_policy["max_programming_term_hits"])
    ):
        return AuditDecision(False, "code_content", normalized, metrics)

    if len(lines) >= 4:
        duplicate_line_fraction = 1.0 - len(set(lines)) / len(lines)
    else:
        duplicate_line_fraction = 0.0
    metrics["duplicate_line_fraction"] = duplicate_line_fraction
    if duplicate_line_fraction > float(content_policy["max_duplicate_line_fraction"]):
        return AuditDecision(False, "repeated_lines", normalized, metrics)

    counts = Counter(words)
    most_common_fraction = max(counts.values()) / max(1, len(words))
    metrics["most_common_word_fraction"] = most_common_fraction
    if most_common_fraction > float(content_policy["max_common_word_fraction"]):
        return AuditDecision(False, "repetitive_text", normalized, metrics)
    boilerplate_hits = len(_BOILERPLATE_RE.findall(normalized))
    metrics["boilerplate_hits"] = boilerplate_hits
    if boilerplate_hits > int(content_policy["max_boilerplate_hits"]):
        return AuditDecision(False, "boilerplate", normalized, metrics)

    return AuditDecision(True, "accepted", normalized, metrics)


def infer_wds_bin(record: Mapping[str, Any]) -> int | None:
    value = record.get("wds_bin")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    path = str(record.get("_input_path", ""))
    match = re.search(r"(?:^|/)(10|[5-9])_\d+\.jsonl\.zst$", path)
    return int(match.group(1)) if match else None


def register_scores(record: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = record.get("web-register") or record.get("web_register") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, Mapping) else {}


def strict_hplt_register_scores(
    record: Mapping[str, Any], *, field: str = "web-register"
) -> dict[str, float]:
    """Parse HPLT's literal register field and fail on schema/probability drift."""

    if field != "web-register" or field not in record:
        raise TurkishCorpusError("HPLT record is missing literal 'web-register' field")
    raw: Any = record[field]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TurkishCorpusError("HPLT web-register is malformed JSON") from exc
    if not isinstance(raw, Mapping) or not raw:
        raise TurkishCorpusError("HPLT web-register must be a non-empty score mapping")
    scores: dict[str, float] = {}
    for raw_label, raw_probability in raw.items():
        label = str(raw_label).strip()
        if not label or isinstance(raw_probability, bool):
            raise TurkishCorpusError("HPLT web-register contains an invalid label/score")
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as exc:
            raise TurkishCorpusError("HPLT web-register score is not numeric") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise TurkishCorpusError("HPLT web-register score is outside [0,1]")
        scores[label] = probability
    return scores


def select_mixture_bucket(
    source_id: str,
    record: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[str, float] | None:
    """Route a post-audit document into the first matching disjoint bucket."""

    for bucket in policy["mixture"]:
        if bucket["source_id"] != source_id:
            continue
        selector = bucket["selector"]
        if "wds_bins" in selector and infer_wds_bin(record) not in selector["wds_bins"]:
            continue
        registers = register_scores(record)
        if "register_any" in selector:
            desired = max(float(registers.get(label, 0.0) or 0.0) for label in selector["register_any"])
            mt = float(registers.get("MT", 0.0) or 0.0)
            if desired < float(selector.get("register_min_probability", 0.0)):
                continue
            if mt > float(selector.get("max_machine_translated_probability", 1.0)):
                continue
            return bucket["id"], desired
        return bucket["id"], float(record.get("quality_score", 0.0) or 0.0)
    return None


def dominant_register(record: Mapping[str, Any]) -> str:
    """Return one auditable HPLT register label, or an explicit sentinel.

    The full score mapping remains in the production backend receipt.  The
    compact pool stores only the deterministic argmax needed for realized-mix
    accounting; non-HPLT sources are marked rather than guessed.
    """

    registers = register_scores(record)
    if not registers:
        return "not_applicable"
    scored: list[tuple[float, str]] = []
    for raw_label, raw_score in registers.items():
        label = str(raw_label).strip()
        if not label:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scored.append((score, label))
    if not scored:
        return "unclassified"
    # Higher probability wins; lexical order makes ties reproducible.
    return min(scored, key=lambda item: (-item[0], item[1]))[1]


def canonical_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_shingles(text: str, width: int = 5) -> frozenset[int]:
    words = [word.casefold() for word in _WORD_RE.findall(text)]
    if not words:
        return frozenset()
    pieces = words if len(words) < width else ["\x1f".join(words[i : i + width]) for i in range(len(words) - width + 1)]
    return frozenset(
        int.from_bytes(hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest(), "big")
        for piece in pieces
    )


def _minhash_signature(shingles: frozenset[int], num_hashes: int) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(num_hashes))
    mask = (1 << 64) - 1
    signature: list[int] = []
    # Fixed odd multipliers make the signature stable across Python versions.
    for index in range(num_hashes):
        a = (0x9E3779B185EBCA87 + 2 * index) & mask
        b = (0xC2B2AE3D27D4EB4F * (index + 1)) & mask
        signature.append(min(((a * value + b) & mask) for value in shingles))
    return tuple(signature)


def _signature_blob(signature: Sequence[int]) -> bytes:
    return b"".join(int(value).to_bytes(8, "big") for value in signature)


def _signature_similarity(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or len(left) % 8:
        return 0.0
    count = len(left) // 8
    matches = sum(left[offset : offset + 8] == right[offset : offset + 8] for offset in range(0, len(left), 8))
    return matches / max(1, count)


class SQLiteMinHashDeduper:
    """Bounded-memory exact + MinHash LSH deduper.

    Only hashes and signatures are retained in SQLite; document text is written
    directly to compressed output fragments.  The database is a generated build
    temporary and need not ship with the final corpus.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        num_hashes: int = 64,
        bands: int = 16,
        threshold: float = 0.82,
    ) -> None:
        if num_hashes <= 0 or bands <= 0 or num_hashes % bands:
            raise TurkishCorpusError("num_hashes must be positive and divisible by bands")
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows_per_band = num_hashes // bands
        self.threshold = threshold
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS representatives (
                text_sha256 TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL,
                signature BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lsh (
                band_key BLOB NOT NULL,
                text_sha256 TEXT NOT NULL,
                PRIMARY KEY (band_key, text_sha256)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS lsh_band ON lsh (band_key);
            """
        )

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "SQLiteMinHashDeduper":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _band_keys(self, signature: Sequence[int]) -> Iterator[bytes]:
        for band in range(self.bands):
            start = band * self.rows_per_band
            payload = _signature_blob(signature[start : start + self.rows_per_band])
            yield band.to_bytes(2, "big") + hashlib.blake2b(payload, digest_size=16).digest()

    def add(self, text: str) -> DedupDecision:
        text_hash = canonical_text_hash(text)
        row = self.connection.execute(
            "SELECT cluster_id FROM representatives WHERE text_sha256 = ?", (text_hash,)
        ).fetchone()
        if row is not None:
            return DedupDecision(False, str(row[0]), "exact")

        shingles = _word_shingles(text)
        signature = _minhash_signature(shingles, self.num_hashes)
        blob = _signature_blob(signature)
        candidates: set[str] = set()
        band_keys = tuple(self._band_keys(signature))
        for key in band_keys:
            candidates.update(
                row[0]
                for row in self.connection.execute(
                    "SELECT text_sha256 FROM lsh WHERE band_key = ?", (key,)
                )
            )
        best: tuple[float, str, str] | None = None
        for candidate in sorted(candidates):
            candidate_row = self.connection.execute(
                "SELECT cluster_id, signature FROM representatives WHERE text_sha256 = ?",
                (candidate,),
            ).fetchone()
            if candidate_row is None:
                continue
            similarity = _signature_similarity(blob, bytes(candidate_row[1]))
            if similarity >= self.threshold:
                match = (similarity, str(candidate_row[0]), candidate)
                if best is None or match > best:
                    best = match
        if best is not None:
            return DedupDecision(False, best[1], "near")

        cluster_id = text_hash
        self.connection.execute(
            "INSERT INTO representatives(text_sha256, cluster_id, signature) VALUES (?, ?, ?)",
            (text_hash, cluster_id, blob),
        )
        self.connection.executemany(
            "INSERT INTO lsh(band_key, text_sha256) VALUES (?, ?)",
            ((key, text_hash) for key in band_keys),
        )
        return DedupDecision(True, cluster_id, None)


def assign_split(cluster_id: str, split_policy: Mapping[str, Any]) -> str:
    fractions = split_policy["fractions"]
    seed = str(split_policy["seed"])
    value = int.from_bytes(
        hashlib.sha256(f"{seed}\x00{cluster_id}".encode("utf-8")).digest()[:8], "big"
    ) / float(1 << 64)
    train_end = float(fractions["train"])
    val_end = train_end + float(fractions["val"])
    return "train" if value < train_end else "val" if value < val_end else "test"


def stable_shuffle_key(document_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\x00{document_id}".encode("utf-8")).hexdigest()


def _nested_get(record: Mapping[str, Any], field: str | None, default: Any = "") -> Any:
    if not field:
        return default
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def iter_jsonl(stream: BinaryIO, *, input_path: str) -> Iterator[dict[str, Any]]:
    wrapper = io.TextIOWrapper(stream, encoding="utf-8")
    try:
        for line_number, line in enumerate(wrapper, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TurkishCorpusError(f"{input_path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise TurkishCorpusError(f"{input_path}:{line_number}: row must be an object")
            value["_input_path"] = input_path
            yield value
    finally:
        wrapper.detach()


def iter_input_records(
    path: str | Path, *, columns: Sequence[str] | None = None
) -> Iterator[dict[str, Any]]:
    source = Path(path)
    name = source.name
    if name.endswith(".parquet"):
        parquet = pq.ParquetFile(source)
        projected = None
        if columns is not None:
            projected = sorted(set(columns))
            missing = sorted(set(projected) - set(parquet.schema_arrow.names))
            if missing:
                raise TurkishCorpusError(
                    f"{source}: pinned adapter schema drift; missing columns {missing}"
                )
        for batch in parquet.iter_batches(batch_size=2_048, columns=projected):
            for row in batch.to_pylist():
                row["_input_path"] = name
                yield row
        return
    if name.endswith(".jsonl"):
        with source.open("rb") as handle:
            yield from iter_jsonl(handle, input_path=name)
        return
    if name.endswith(".jsonl.zst"):
        raw = pa.input_stream(str(source))
        compressed = pa.CompressedInputStream(raw, "zstd")
        try:
            yield from iter_jsonl(compressed, input_path=name)
        finally:
            compressed.close()
            raw.close()
        return
    raise TurkishCorpusError(f"unsupported input format: {source}")


def _copy_and_hash_response(response: requests.Response, destination: Path) -> str:
    digest = hashlib.sha256()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                digest.update(chunk)
                handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def stage_receipt_file(
    item: Mapping[str, Any],
    temporary_dir: str | Path,
    *,
    request_get: Any = requests.get,
) -> Path:
    """Stage exactly one receipt object and verify SHA-256 before parsing it."""

    uri = str(item["uri"])
    parsed = urllib.parse.urlparse(uri)
    expected = str(item["checksum"]["value"])
    suffixes = "".join(Path(parsed.path).suffixes[-2:]) or ".input"
    fd, name = tempfile.mkstemp(prefix="source-", suffix=suffixes, dir=temporary_dir)
    os.close(fd)
    destination = Path(name)
    try:
        if parsed.scheme == "file":
            source = Path(urllib.parse.unquote(parsed.path))
            digest = hashlib.sha256()
            with source.open("rb") as read_handle, destination.open("wb") as write_handle:
                for chunk in iter(lambda: read_handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    write_handle.write(chunk)
                write_handle.flush()
                os.fsync(write_handle.fileno())
            actual = digest.hexdigest()
        else:
            response = request_get(uri, stream=True, timeout=(30, 300))
            try:
                response.raise_for_status()
                actual = _copy_and_hash_response(response, destination)
            finally:
                response.close()
        if actual != expected:
            raise TurkishCorpusError(f"SHA-256 mismatch for {uri}: expected {expected}, got {actual}")
        expected_size = item.get("size_bytes")
        if isinstance(expected_size, int) and not isinstance(expected_size, bool):
            actual_size = destination.stat().st_size
            if actual_size != expected_size:
                raise TurkishCorpusError(
                    f"size mismatch for {uri}: expected {expected_size}, got {actual_size}"
                )
        return destination
    except Exception:
        destination.unlink(missing_ok=True)
        raise


class FragmentWriter:
    """Write bounded row groups into moderately sized run-owned fragments.

    A buffer per ``(split, mixture)`` keeps the number of live buffers small.
    The older implementation buffered every one of 256 shuffle buckets; under
    a strict global memory ceiling that degenerates into millions of tiny
    Parquet files.  We retain the deterministic bucket in the row order while
    rotating files only after many row groups.
    """

    _schema = pa.schema(
        [
            ("text", pa.string()),
            ("source_id", pa.string()),
            ("mixture_id", pa.string()),
            ("document_id", pa.string()),
            ("url", pa.string()),
            ("cluster_id", pa.string()),
            ("shuffle_key", pa.string()),
            ("quality_score", pa.float32()),
            ("register_bucket", pa.string()),
        ]
    )

    def __init__(
        self,
        root: str | Path,
        *,
        rows_per_fragment: int,
        buckets: int,
        max_buffered_rows: int,
        rows_per_output_file: int | None = None,
    ) -> None:
        if rows_per_fragment <= 0 or buckets <= 0 or max_buffered_rows <= 0:
            raise TurkishCorpusError("fragment-writer limits must be positive")
        self.root = Path(root)
        self.rows_per_fragment = min(rows_per_fragment, max_buffered_rows)
        self.buckets = buckets
        self.max_buffered_rows = max_buffered_rows
        self.rows_per_output_file = int(
            rows_per_output_file or max(self.rows_per_fragment, self.rows_per_fragment * 64)
        )
        if self.rows_per_output_file < self.rows_per_fragment:
            raise TurkishCorpusError("rows_per_output_file must cover at least one row group")
        self.buffers: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.counters: Counter[tuple[str, str]] = Counter()
        self.open_fragments: dict[tuple[str, str], dict[str, Any]] = {}
        self.files: list[dict[str, Any]] = []
        self.buffered_rows = 0
        self.peak_buffered_rows = 0

    def add(self, split: str, mixture_id: str, row: dict[str, Any]) -> None:
        key = (split, mixture_id)
        if self.buffered_rows >= self.max_buffered_rows:
            largest = max(self.buffers, key=lambda item: len(self.buffers[item]))
            self._flush_batch(largest)
        self.buffers[key].append(row)
        self.buffered_rows += 1
        self.peak_buffered_rows = max(self.peak_buffered_rows, self.buffered_rows)
        if len(self.buffers[key]) >= self.rows_per_fragment:
            self._flush_batch(key)
        while self.buffered_rows > self.max_buffered_rows:
            largest = max(self.buffers, key=lambda item: len(self.buffers[item]))
            if not self.buffers[largest]:
                raise TurkishCorpusError("fragment buffer accounting drift")
            self._flush_batch(largest)

    def _open_fragment(self, key: tuple[str, str]) -> dict[str, Any]:
        split, mixture_id = key
        index = self.counters[key]
        self.counters[key] += 1
        relative = Path("pool") / split / mixture_id / f"fragment-{index:05d}.parquet"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        state: dict[str, Any] = {
            "writer": pq.ParquetWriter(
                temporary,
                self._schema,
                compression="zstd",
                use_dictionary=True,
            ),
            "temporary": temporary,
            "destination": destination,
            "relative": relative,
            "split": split,
            "mixture_id": mixture_id,
            "rows": 0,
            "row_groups": 0,
            "bucket_min": None,
            "bucket_max": None,
        }
        self.open_fragments[key] = state
        return state

    def _close_fragment(self, key: tuple[str, str]) -> None:
        state = self.open_fragments.pop(key, None)
        if state is None:
            return
        state["writer"].close()
        temporary = state["temporary"]
        destination = state["destination"]
        os.replace(temporary, destination)
        self.files.append(
            {
                "path": state["relative"].as_posix(),
                "split": state["split"],
                "mixture_id": state["mixture_id"],
                "rows": state["rows"],
                "row_groups": state["row_groups"],
                "shuffle_bucket_min": state["bucket_min"],
                "shuffle_bucket_max": state["bucket_max"],
                "size_bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
            }
        )

    def _flush_batch(self, key: tuple[str, str]) -> None:
        rows = self.buffers[key]
        if not rows:
            return
        rows.sort(
            key=lambda row: (
                int(row["shuffle_key"][:8], 16) % self.buckets,
                row["shuffle_key"],
                row["document_id"],
            )
        )
        state = self.open_fragments.get(key)
        if state is not None and state["rows"] + len(rows) > self.rows_per_output_file:
            self._close_fragment(key)
            state = None
        if state is None:
            state = self._open_fragment(key)
        buckets = [int(row["shuffle_key"][:8], 16) % self.buckets for row in rows]
        state["writer"].write_table(pa.Table.from_pylist(rows, schema=self._schema))
        state["rows"] += len(rows)
        state["row_groups"] += 1
        batch_min, batch_max = min(buckets), max(buckets)
        state["bucket_min"] = (
            batch_min if state["bucket_min"] is None else min(state["bucket_min"], batch_min)
        )
        state["bucket_max"] = (
            batch_max if state["bucket_max"] is None else max(state["bucket_max"], batch_max)
        )
        self.buffered_rows -= len(rows)
        rows.clear()

    def close(self) -> list[dict[str, Any]]:
        for key in sorted(self.buffers):
            self._flush_batch(key)
        for key in sorted(tuple(self.open_fragments)):
            self._close_fragment(key)
        return sorted(self.files, key=lambda item: item["path"])


def materialize_filtered_pool(
    policy: Mapping[str, Any],
    receipt: Mapping[str, Any],
    output_dir: str | Path,
    *,
    git_commit: str,
    request_get: Any = requests.get,
    allow_reference_backend: bool = False,
) -> dict[str, Any]:
    """Reference-only materializer for fixtures and bounded smoke corpora.

    Production must import the output of the pinned GlotLID/DataTrove backend;
    the Python MinHash implementation is intentionally refused by default.
    """

    validate_corpus_policy(policy)
    validate_source_receipt(receipt, policy)
    if not allow_reference_backend:
        raise TurkishCorpusError(
            "SQLite/Python audit is a reference backend only; production requires "
            "materialize_production_pool with a sealed backend receipt"
        )
    if not _SHA1_RE.fullmatch(git_commit) and not _SHA256_RE.fullmatch(git_commit):
        raise TurkishCorpusError("git_commit must be a full Git commit")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    policy_hash = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
    source_policies = {source["id"]: source for source in policy["sources"]}
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_by_split: Counter[str] = Counter()
    accepted_by_mixture: Counter[str] = Counter()
    build_dir = destination / ".build"
    build_dir.mkdir()
    writer = FragmentWriter(
        destination,
        rows_per_fragment=int(policy["materialization"]["rows_per_fragment"]),
        buckets=int(policy["materialization"]["shuffle_buckets"]),
        max_buffered_rows=int(policy["materialization"]["max_buffered_rows"]),
        rows_per_output_file=int(policy["materialization"]["rows_per_output_file"]),
    )
    try:
        with SQLiteMinHashDeduper(
            build_dir / "dedup.sqlite3",
            num_hashes=int(policy["deduplication"]["reference_backend"]["num_hashes"]),
            bands=int(policy["deduplication"]["reference_backend"]["bands"]),
            threshold=float(
                policy["deduplication"]["reference_backend"]["similarity_threshold"]
            ),
        ) as deduper:
            receipt_sources = {source["id"]: source for source in receipt["sources"]}
            source_order = {
                source_id: index
                for index, source_id in enumerate(policy["deduplication"]["source_priority"])
            }
            for source_policy in sorted(
                policy["sources"], key=lambda source: source_order[source["id"]]
            ):
                source_id = source_policy["id"]
                adapter = source_policy["adapter"]
                projected_columns = [adapter["text_field"]]
                for field_name in (
                    "id_field",
                    "url_field",
                    "language_field",
                    "language_probability_field",
                ):
                    field = adapter.get(field_name)
                    if isinstance(field, str) and field and field != "$partition":
                        projected_columns.append(field.split(".")[0])
                if source_id == "hplt3_tr":
                    projected_columns.append("web-register")
                if source_id == "fineweb2_hq_tr":
                    projected_columns.append("quality_score")
                for item in receipt_sources[source_id]["files"]:
                    staged = stage_receipt_file(item, build_dir, request_get=request_get)
                    try:
                        for row_index, record in enumerate(
                            iter_input_records(staged, columns=projected_columns)
                        ):
                            counters[source_id]["seen"] += 1
                            # Preserve the upstream filename for HPLT WDS-bin routing;
                            # the staging filename is intentionally random.
                            record["_input_path"] = urllib.parse.urlparse(str(item["uri"])).path
                            source_lid_ok = source_lid_is_turkish(record, adapter)
                            text = _nested_get(record, adapter["text_field"])
                            url = str(_nested_get(record, adapter.get("url_field"), "") or "")
                            audit = audit_document(
                                text,
                                url=url,
                                source_lid_ok=source_lid_ok,
                                content_policy=policy["content_policy"],
                            )
                            if not audit.accepted:
                                counters[source_id][f"reject:{audit.reason}"] += 1
                                continue
                            selected = select_mixture_bucket(source_id, record, policy)
                            if selected is None:
                                counters[source_id]["reject:mixture_selector"] += 1
                                continue
                            mixture_id, quality_score = selected
                            dedup = deduper.add(audit.normalized_text)
                            if not dedup.accepted:
                                counters[source_id][f"reject:dedup_{dedup.duplicate_kind}"] += 1
                                continue
                            upstream_id = _nested_get(record, adapter.get("id_field"), "")
                            document_id = str(upstream_id or canonical_text_hash(audit.normalized_text))
                            split = assign_split(dedup.cluster_id, policy["splits"])
                            shuffle = stable_shuffle_key(document_id, policy["splits"]["seed"])
                            writer.add(
                                split,
                                mixture_id,
                                {
                                    "text": audit.normalized_text,
                                    "source_id": source_id,
                                    "mixture_id": mixture_id,
                                    "document_id": document_id,
                                    "url": url,
                                    "cluster_id": dedup.cluster_id,
                                    "shuffle_key": shuffle,
                                    "quality_score": quality_score,
                                    "register_bucket": dominant_register(record),
                                },
                            )
                            counters[source_id]["accepted"] += 1
                            accepted_by_split[split] += 1
                            accepted_by_mixture[mixture_id] += 1
                    finally:
                        # Only our verified staging copy is removed.  No pre-existing
                        # dataset or final artifact is ever a deletion target.
                        staged.unlink(missing_ok=True)
        files = writer.close()
        manifest = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": CORPUS_MANIFEST_KIND,
                "name": CORPUS_NAME,
                "stage": "filtered_pool",
                "backend_scope": "reference_smoke_only",
                "policy_sha256": policy_hash,
                "source_receipt_sha256": receipt["canonical_sha256"],
                "source_provenance": [
                    {
                        "id": source["id"],
                        "repo_id": source["repo_id"],
                        "resolved_revision": source["resolved_revision"],
                        "license_id": source["license_id"],
                    }
                    for source in policy["sources"]
                ],
                "created_by": {"git_commit": git_commit, "tool": "scripts.build_turkish_pretrain_corpus"},
                "language": "tur_Latn",
                "code_allowed": False,
                "deduplication": policy["deduplication"],
                "split_policy": policy["splits"],
                "accepted_documents_by_split": dict(sorted(accepted_by_split.items())),
                "accepted_documents_by_mixture": dict(sorted(accepted_by_mixture.items())),
                "audit_counts": {key: dict(sorted(value.items())) for key, value in sorted(counters.items())},
                "files": files,
                "writer_metrics": {
                    "max_buffered_rows": writer.max_buffered_rows,
                    "peak_buffered_rows": writer.peak_buffered_rows,
                },
                "canonical_sha256": None,
            }
        )
        write_json_atomic(destination / "corpus_manifest.json", manifest)
        write_pool_ownership_manifest(destination, manifest)
        return manifest
    finally:
        # SQLite/WAL are generated scratch.  Remove them safely, then the empty
        # build directory; leave it intact on unexpected foreign contents.
        for name in ("dedup.sqlite3", "dedup.sqlite3-wal", "dedup.sqlite3-shm"):
            (build_dir / name).unlink(missing_ok=True)
        try:
            build_dir.rmdir()
        except OSError:
            pass


def iter_pool_rows(
    pool_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    split: str,
    mixture_id: str,
) -> Iterator[dict[str, Any]]:
    root = Path(pool_dir)
    records = [
        item
        for item in manifest["files"]
        if item["split"] == split and item["mixture_id"] == mixture_id
    ]
    for item in sorted(records, key=lambda value: value["path"]):
        path = root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise TurkishCorpusError(f"filtered pool hash drift: {item['path']}")
        parquet = pq.ParquetFile(path)
        for row_group_index in range(parquet.num_row_groups):
            rows = parquet.read_row_group(row_group_index).to_pylist()
            rows.sort(key=lambda row: (row["shuffle_key"], row["document_id"]))
            yield from rows


def representative_sample(
    pool_dir: str | Path,
    policy: Mapping[str, Any],
    *,
    max_chars: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield a deterministic train-only tokenizer sample with mixture deficits."""

    manifest = load_json_strict(Path(pool_dir) / "corpus_manifest.json")
    verify_manifest_hash(manifest)
    if manifest.get("stage") != "filtered_pool":
        raise TurkishCorpusError("tokenizer sample requires a filtered pool")
    target = int(max_chars or policy["tokenizer_training"]["max_chars"])
    document_cap = int(policy["tokenizer_training"]["max_chars_per_document"])
    buckets = [bucket for bucket in policy["mixture"]]
    iterators = {
        bucket["id"]: iter_pool_rows(pool_dir, manifest, split="train", mixture_id=bucket["id"])
        for bucket in buckets
    }
    active = set(iterators)
    emitted: Counter[str] = Counter()
    total = 0
    while active:
        choice = max(
            active,
            key=lambda key: (
                next(bucket["weight"] for bucket in buckets if bucket["id"] == key) * max(1, total)
                - emitted[key],
                key,
            ),
        )
        try:
            row = next(iterators[choice])
        except StopIteration:
            active.remove(choice)
            continue
        # Apply the exact tok_train per-document cap before mixture accounting;
        # otherwise the sample receipt overstates what the trainer consumes.
        text = row["text"][:document_cap]
        if not text:
            continue
        yield {**row, "text": text}
        emitted[choice] += len(text)
        total += len(text)
        # Pinned upstream checks only after yielding the complete capped doc.
        if total > target:
            return


def write_tokenizer_sample(
    pool_dir: str | Path,
    policy: Mapping[str, Any],
    output_dir: str | Path,
    *,
    git_commit: str,
    max_chars: int | None = None,
    allow_reference_pool: bool = False,
) -> dict[str, Any]:
    """Write and seal a post-filter, train-only raw-BPE training sample."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty tokenizer sample directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    pool_manifest = load_json_strict(Path(pool_dir) / "corpus_manifest.json")
    verify_manifest_hash(pool_manifest)
    validate_pool_ownership(pool_dir, pool_manifest)
    qa_approval: dict[str, Any] | None = None
    if pool_manifest.get("backend_scope") == "production_glotlid_datatrove":
        qa_approval = validate_qa_gate(pool_dir, pool_manifest)
    elif not allow_reference_pool:
        raise TurkishCorpusError("tokenizer sample refuses a reference/smoke filtered pool")
    sample_path = destination / "train-00000.parquet"
    sample_writer = pq.ParquetWriter(
        sample_path, FragmentWriter._schema, compression="zstd", use_dictionary=True
    )
    sample_buffer: list[dict[str, Any]] = []
    sample_documents = 0
    sample_characters = 0
    try:
        for row in representative_sample(pool_dir, policy, max_chars=max_chars):
            sample_buffer.append(row)
            sample_documents += 1
            sample_characters += len(row["text"])
            if len(sample_buffer) >= 4_096:
                sample_writer.write_table(
                    pa.Table.from_pylist(sample_buffer, schema=FragmentWriter._schema)
                )
                sample_buffer.clear()
        if sample_buffer:
            sample_writer.write_table(
                pa.Table.from_pylist(sample_buffer, schema=FragmentWriter._schema)
            )
    finally:
        sample_writer.close()
    if sample_documents == 0:
        sample_path.unlink(missing_ok=True)
        raise TurkishCorpusError("filtered train pool produced an empty tokenizer sample")
    requested_characters = int(max_chars or policy["tokenizer_training"]["max_chars"])
    if (
        pool_manifest.get("backend_scope") == "production_glotlid_datatrove"
        and not (
            requested_characters < sample_characters
            <= requested_characters
            + int(policy["tokenizer_training"]["max_chars_per_document"])
        )
    ):
        raise TurkishCorpusError(
            "production train-only pool cannot cover pinned tokenizer threshold semantics: "
            f"requested={requested_characters}, realized={sample_characters}"
        )
    # A fixed validation file is required by nanochat's strict loader, but the
    # tokenizer trainer reads only train.  Copy deterministic val rows without
    # contaminating tokenizer training.
    val_rows: list[dict[str, Any]] = []
    for bucket in policy["mixture"]:
        iterator = iter_pool_rows(pool_dir, pool_manifest, split="val", mixture_id=bucket["id"])
        for _ in range(16):
            try:
                val_rows.append(next(iterator))
            except StopIteration:
                break
    if not val_rows:
        # Tiny synthetic fixtures can legitimately hash no row into val.  Emit a
        # schema-compatible empty file; production preflight rejects empty val.
        val_table = pa.Table.from_pylist([], schema=FragmentWriter._schema)
    else:
        val_table = pa.Table.from_pylist(val_rows, schema=FragmentWriter._schema)
    val_path = destination / "validation.parquet"
    pq.write_table(val_table, val_path, compression="zstd")
    ordered = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in (sample_path, val_path)
    ]
    synthetic_revision = hashlib.sha1(
        pool_manifest["canonical_sha256"].encode("ascii"), usedforsecurity=False
    ).hexdigest()
    compatibility = seal_manifest(
        {
            "schema_version": "1.0",
            "manifest_type": "dataset",
            "profile": "strict",
            "dataset": {
                "repo_id": f"local-composite/{CORPUS_NAME}",
                "path": "tokenizer_sample",
                "requested_revision": synthetic_revision,
                "resolved_revision": synthetic_revision,
                "repo_type": "dataset",
            },
            "text_column": "text",
            "ordered_files": ordered,
            "validation_file": val_path.name,
            "created_by": {"git_commit": git_commit, "tool": "scripts.build_turkish_pretrain_corpus"},
            "metadata": {
                "revision_semantics": "sha1_of_parent_corpus_manifest_not_hub_commit",
                "parent_corpus_manifest_sha256": pool_manifest["canonical_sha256"],
                "sample_scope": "post_filter_train_only",
                "max_chars_per_document": policy["tokenizer_training"][
                    "max_chars_per_document"
                ],
                "qa_approval_sha256": (
                    qa_approval["canonical_sha256"] if qa_approval else None
                ),
            },
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination / "fineweb2_manifest.json", compatibility)
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_raw_bpe_training_sample",
            "name": TOKENIZER_NAME,
            "vocab_size": VOCAB_SIZE,
            "parent_corpus_manifest_sha256": pool_manifest["canonical_sha256"],
            "nanochat_dataset_manifest_sha256": compatibility["canonical_sha256"],
            "characters": sample_characters,
            "trainer_visible_characters": sample_characters,
            "requested_max_characters": requested_characters,
            "terminal_overshoot_characters": sample_characters
            - requested_characters,
            "stop_rule": policy["tokenizer_training"]["stop_rule"],
            "documents": sample_documents,
            "max_chars_per_document": policy["tokenizer_training"][
                "max_chars_per_document"
            ],
            "qa_approval_sha256": (
                qa_approval["canonical_sha256"] if qa_approval else None
            ),
            "files": ordered,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination / "tokenizer_sample_manifest.json", receipt)
    return receipt


def _token_count(tokenizer: Any, text: str) -> int:
    # Pretraining prepends exactly one BOS per document.
    return 1 + len(tokenizer.encode(text))


def _batch_token_counts(tokenizer: Any, texts: list[str]) -> list[int]:
    try:
        encoded = tokenizer.encode(texts, num_threads=min(16, max(1, os.cpu_count() or 1)))
    except (TypeError, ValueError):
        return [_token_count(tokenizer, text) for text in texts]
    if not isinstance(encoded, list) or len(encoded) != len(texts):
        raise TurkishCorpusError("tokenizer batch encoding returned an invalid shape")
    counts: list[int] = []
    for row in encoded:
        if not isinstance(row, list):
            raise TurkishCorpusError("tokenizer batch encoding returned a non-list row")
        counts.append(1 + len(row))
    return counts


def _count_pool_fragment(
    pool_root: Path,
    record: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, int]]:
    path = pool_root / record["path"]
    if path.is_symlink() or path.stat().st_size != record["size_bytes"]:
        raise TurkishCorpusError(f"filtered pool size/symlink drift: {record['path']}")
    if file_sha256(path) != record["sha256"]:
        raise TurkishCorpusError(f"filtered pool hash drift: {record['path']}")
    parquet = pq.ParquetFile(path)
    missing = sorted(set(FragmentWriter._schema.names) - set(parquet.schema_arrow.names))
    if missing:
        raise TurkishCorpusError(f"filtered pool schema drift in {record['path']}: {missing}")
    tokens_by_source: Counter[str] = Counter()
    tokens_by_register: Counter[str] = Counter()
    documents_by_source: Counter[str] = Counter()
    documents_by_register: Counter[str] = Counter()
    total_tokens = 0
    total_rows = 0
    max_document_tokens = 0
    peak_rows = 0
    peak_utf8_bytes = 0
    for row_group_index in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            row_group_index,
            columns=["text", "source_id", "mixture_id", "register_bucket"],
        )
        rows = table.to_pylist()
        texts = [str(row["text"]) for row in rows]
        counts = _batch_token_counts(tokenizer, texts)
        peak_rows = max(peak_rows, len(rows))
        peak_utf8_bytes = max(
            peak_utf8_bytes,
            sum(len(text.encode("utf-8")) for text in texts),
        )
        for row, count in zip(rows, counts, strict=True):
            if row["mixture_id"] != record["mixture_id"]:
                raise TurkishCorpusError(f"mixture drift inside {record['path']}")
            source_id = str(row["source_id"])
            register = str(row.get("register_bucket") or "not_applicable")
            total_tokens += count
            total_rows += 1
            max_document_tokens = max(max_document_tokens, count)
            tokens_by_source[source_id] += count
            tokens_by_register[register] += count
            documents_by_source[source_id] += 1
            documents_by_register[register] += 1
    if total_rows != record["rows"]:
        raise TurkishCorpusError(f"filtered pool row-count drift: {record['path']}")
    return (
        {
            **dict(record),
            "encoded_tokens_with_bos": total_tokens,
            "max_document_tokens": max_document_tokens,
            "tokens_by_source": dict(sorted(tokens_by_source.items())),
            "tokens_by_register": dict(sorted(tokens_by_register.items())),
            "documents_by_source": dict(sorted(documents_by_source.items())),
            "documents_by_register": dict(sorted(documents_by_register.items())),
        },
        {"peak_rows": peak_rows, "peak_utf8_bytes": peak_utf8_bytes},
    )


def _exact_initial_quotas(
    policy: Mapping[str, Any],
    target_tokens: int,
    *,
    source_weights: Mapping[str, float] | None = None,
) -> dict[str, int]:
    from fractions import Fraction

    floors: dict[str, int] = {}
    remainders: list[tuple[Fraction, str]] = []
    exact_weights: dict[str, Fraction] = {}
    for bucket in policy["mixture"]:
        weight = (
            bucket["weight"]
            if source_weights is None
            else source_weights.get(bucket["id"])
        )
        if weight is None:
            raise TurkishCorpusError("packing plan omits a mixture source weight")
        parsed_weight = Fraction(str(weight))
        if parsed_weight <= 0:
            raise TurkishCorpusError("packing source weights must be positive")
        exact_weights[bucket["id"]] = parsed_weight
        exact = parsed_weight * target_tokens
        floor = exact.numerator // exact.denominator
        floors[bucket["id"]] = floor
        remainders.append((exact - floor, bucket["id"]))
    if sum(exact_weights.values()) != 1:
        raise TurkishCorpusError("packing source weights must sum exactly to one")
    missing = target_tokens - sum(floors.values())
    for _remainder, bucket_id in sorted(
        remainders, key=lambda item: (-item[0], item[1])
    )[:missing]:
        floors[bucket_id] += 1
    if sum(floors.values()) != target_tokens:
        raise TurkishCorpusError("initial mixture token quotas do not conserve target tokens")
    return floors


def allocate_fallback_quotas(
    policy: Mapping[str, Any],
    *,
    target_tokens: int,
    safe_capacity: Mapping[str, int],
    source_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, int]]:
    """Transfer unavailable quota once through ordered, cycle-safe fallbacks."""

    desired = _exact_initial_quotas(
        policy, target_tokens, source_weights=source_weights
    )
    fallback = {bucket["id"]: tuple(bucket["fallback"]) for bucket in policy["mixture"]}
    effective = {
        bucket_id: min(tokens, int(safe_capacity.get(bucket_id, 0)))
        for bucket_id, tokens in desired.items()
    }
    ledger: list[dict[str, Any]] = []
    unresolved: dict[str, int] = {}
    for origin in [bucket["id"] for bucket in policy["mixture"]]:
        remaining = desired[origin] - effective[origin]
        if remaining <= 0:
            continue
        queue = [origin]
        visited = {origin}
        while remaining > 0 and queue:
            current = queue.pop(0)
            for destination in fallback[current]:
                if destination in visited:
                    ledger.append(
                        {
                            "origin": origin,
                            "via": current,
                            "destination": destination,
                            "tokens": 0,
                            "event": "cycle_or_duplicate_skipped",
                        }
                    )
                    continue
                visited.add(destination)
                room = max(0, int(safe_capacity[destination]) - effective[destination])
                moved = min(remaining, room)
                if moved:
                    effective[destination] += moved
                    remaining -= moved
                    ledger.append(
                        {
                            "origin": origin,
                            "via": current,
                            "destination": destination,
                            "tokens": moved,
                            "event": "quota_transferred",
                        }
                    )
                if remaining > 0:
                    queue.append(destination)
                if remaining == 0:
                    break
        if remaining:
            unresolved[origin] = remaining
            ledger.append(
                {
                    "origin": origin,
                    "via": origin,
                    "destination": None,
                    "tokens": remaining,
                    "event": "unresolved_shortfall",
                }
            )
    if sum(effective.values()) + sum(unresolved.values()) != target_tokens:
        raise TurkishCorpusError("fallback ledger does not conserve target tokens")
    if unresolved:
        raise TurkishCorpusError(
            "unique Turkish pool cannot cover target through declared fallbacks: "
            f"{dict(sorted(unresolved.items()))}"
        )
    if sum(effective.values()) != target_tokens:
        raise TurkishCorpusError("fallback allocation did not conserve target tokens")
    return effective, ledger, desired


def _write_partial_fragment(
    source_path: Path,
    destination_path: Path,
    *,
    required_tokens: int,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Copy only a terminal fragment prefix with row-group-bounded memory."""

    if required_tokens <= 0:
        raise TurkishCorpusError("partial-fragment token request must be positive")
    temporary = destination_path.with_name(f".{destination_path.name}.tmp")
    writer = pq.ParquetWriter(
        temporary,
        FragmentWriter._schema,
        compression="zstd",
        use_dictionary=True,
    )
    totals: Counter[str] = Counter()
    tokens_by_source: Counter[str] = Counter()
    tokens_by_register: Counter[str] = Counter()
    documents_by_source: Counter[str] = Counter()
    documents_by_register: Counter[str] = Counter()
    selected_rows = 0
    selected_tokens = 0
    peak_rows = 0
    peak_utf8_bytes = 0
    finished = False
    try:
        parquet = pq.ParquetFile(source_path)
        for row_group_index in range(parquet.num_row_groups):
            rows = parquet.read_row_group(row_group_index).to_pylist()
            texts = [str(row["text"]) for row in rows]
            counts = _batch_token_counts(tokenizer, texts)
            output_rows: list[dict[str, Any]] = []
            for row, count in zip(rows, counts, strict=True):
                output_rows.append(row)
                selected_rows += 1
                selected_tokens += count
                source = str(row["source_id"])
                register = str(row.get("register_bucket") or "not_applicable")
                tokens_by_source[source] += count
                tokens_by_register[register] += count
                documents_by_source[source] += 1
                documents_by_register[register] += 1
                if selected_tokens >= required_tokens:
                    finished = True
                    break
            if output_rows:
                peak_rows = max(peak_rows, len(output_rows))
                peak_utf8_bytes = max(
                    peak_utf8_bytes,
                    sum(len(str(row["text"]).encode("utf-8")) for row in output_rows),
                )
                writer.write_table(pa.Table.from_pylist(output_rows, schema=FragmentWriter._schema))
            if finished:
                break
    finally:
        writer.close()
    if not finished:
        temporary.unlink(missing_ok=True)
        raise TurkishCorpusError("partial source fragment did not cover requested tokens")
    os.replace(temporary, destination_path)
    totals["rows"] = selected_rows
    totals["tokens"] = selected_tokens
    return (
        {
            "size_bytes": destination_path.stat().st_size,
            "sha256": file_sha256(destination_path),
            "rows": selected_rows,
            "encoded_tokens_with_bos": selected_tokens,
            "tokens_by_source": dict(sorted(tokens_by_source.items())),
            "tokens_by_register": dict(sorted(tokens_by_register.items())),
            "documents_by_source": dict(sorted(documents_by_source.items())),
            "documents_by_register": dict(sorted(documents_by_register.items())),
            "promotion_mode": "bounded_partial_copy",
        },
        {"peak_rows": peak_rows, "peak_utf8_bytes": peak_utf8_bytes},
    )


def _write_eval_split(
    pool_dir: str | Path,
    pool_manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    tokenizer: Any,
    destination: Path,
    split: str,
) -> tuple[
    dict[str, Any],
    Counter[str],
    Counter[str],
    dict[str, int],
    dict[str, Any],
]:
    token_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    path = destination
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    writer = pq.ParquetWriter(
        temporary,
        FragmentWriter._schema,
        compression="zstd",
        use_dictionary=True,
    )
    rows_written = 0
    rejected_documents: Counter[str] = Counter()
    rejected_tokens: Counter[str] = Counter()
    rejected_utf8_bytes: Counter[str] = Counter()
    max_accepted_tokens_with_bos = 0
    peak_rows = 0
    peak_utf8_bytes = 0
    try:
        records = sorted(
            (item for item in pool_manifest["files"] if item["split"] == split),
            key=lambda item: item["path"],
        )
        for item in records:
            source_path = Path(pool_dir) / item["path"]
            if file_sha256(source_path) != item["sha256"]:
                raise TurkishCorpusError(f"filtered pool hash drift: {item['path']}")
            parquet = pq.ParquetFile(source_path)
            for row_group_index in range(parquet.num_row_groups):
                rows = parquet.read_row_group(row_group_index).to_pylist()
                rows.sort(key=lambda row: (row["shuffle_key"], row["document_id"]))
                texts = [str(row["text"]) for row in rows]
                counts = _batch_token_counts(tokenizer, texts)
                accepted_rows: list[dict[str, Any]] = []
                for row, text, count in zip(rows, texts, counts, strict=True):
                    mixture_id = str(row["mixture_id"])
                    if count > D32_EVAL_ROW_CAPACITY:
                        rejected_documents[mixture_id] += 1
                        rejected_tokens[mixture_id] += count
                        rejected_utf8_bytes[mixture_id] += len(text.encode("utf-8"))
                        continue
                    accepted_rows.append(row)
                    max_accepted_tokens_with_bos = max(
                        max_accepted_tokens_with_bos, count
                    )
                    token_counts[row["mixture_id"]] += count
                    document_counts[row["mixture_id"]] += 1
                if accepted_rows:
                    writer.write_table(
                        pa.Table.from_pylist(
                            accepted_rows, schema=FragmentWriter._schema
                        )
                    )
                    rows_written += len(accepted_rows)
                    peak_rows = max(peak_rows, len(accepted_rows))
                    peak_utf8_bytes = max(
                        peak_utf8_bytes,
                        sum(
                            len(str(row["text"]).encode("utf-8"))
                            for row in accepted_rows
                        ),
                    )
    finally:
        writer.close()
    os.replace(temporary, path)
    if rows_written <= 0:
        raise TurkishCorpusError(
            f"{split} contains no whole documents fitting the 2049-token row contract"
        )
    return (
        {
            "path": path.name if path.parent == destination.parent else path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "rows": rows_written,
            "encoded_tokens_with_bos": sum(token_counts.values()),
        },
        token_counts,
        document_counts,
        {"peak_rows": peak_rows, "peak_utf8_bytes": peak_utf8_bytes},
        {
            "policy": "whole_document_no_crop",
            "tokenization": "exact_batched_encode_plus_document_bos",
            "max_payload_tokens": D32_EVAL_MAX_PAYLOAD_TOKENS,
            "max_encoded_tokens_with_bos": D32_EVAL_ROW_CAPACITY,
            "oversized_document_action": "excluded_before_exposure_selection",
            "accepted_documents": rows_written,
            "max_accepted_encoded_tokens_with_bos": max_accepted_tokens_with_bos,
            "rejected_long_documents": sum(rejected_documents.values()),
            "rejected_long_encoded_tokens_with_bos": sum(rejected_tokens.values()),
            "rejected_long_utf8_bytes": sum(rejected_utf8_bytes.values()),
            "rejected_long_documents_by_mixture": dict(
                sorted(rejected_documents.items())
            ),
            "rejected_long_encoded_tokens_by_mixture": dict(
                sorted(rejected_tokens.items())
            ),
            "rejected_long_utf8_bytes_by_mixture": dict(
                sorted(rejected_utf8_bytes.items())
            ),
        },
    )


PACKING_PREFLIGHT_REPORT_KIND = "turkish_packing_preflight_report"
PACKING_PREFLIGHT_APPROVAL_KIND = "turkish_packing_preflight_approval"


def _sampled_pool_rank_batches(
    pool_root: Path,
    units: Sequence[tuple[str, int]],
    tokenizer: Any,
    *,
    mixture_id: str,
    rank: int,
    world_size: int,
    max_documents: int,
    tokenizer_batch_size: int,
) -> Iterator[list[Any]]:
    from nanochat.packing_capacity import PackingDocument

    emitted = 0
    bos_token = tokenizer.get_bos_token_id()
    for relative_path, row_group_index in units[rank::world_size]:
        parquet = pq.ParquetFile(pool_root / relative_path)
        rows = parquet.read_row_group(
            row_group_index,
            columns=[
                "text",
                "document_id",
                "mixture_id",
                "source_id",
                "register_bucket",
            ],
        ).to_pylist()
        for offset in range(0, len(rows), tokenizer_batch_size):
            if emitted >= max_documents:
                return
            batch_rows = rows[offset : offset + tokenizer_batch_size]
            batch_rows = batch_rows[: max_documents - emitted]
            texts = [str(row["text"]) for row in batch_rows]
            encoded = tokenizer.encode(texts, prepend=bos_token, num_threads=4)
            if not isinstance(encoded, list) or len(encoded) != len(batch_rows):
                raise TurkishCorpusError("packing sample tokenizer batch shape drift")
            documents = []
            for row, tokens in zip(batch_rows, encoded, strict=True):
                if row["mixture_id"] != mixture_id:
                    raise TurkishCorpusError("packing sample mixture identity drift")
                documents.append(
                    PackingDocument(
                        tokens_with_bos=len(tokens),
                        document_id=str(row["document_id"]),
                        mixture_id=mixture_id,
                        source_id=str(row["source_id"]),
                        register_bucket=str(
                            row.get("register_bucket") or "not_applicable"
                        ),
                    )
                )
            emitted += len(documents)
            if documents:
                yield documents


def _normalized_measured_weights(raw: Mapping[str, float]) -> dict[str, float]:
    if not raw or any(not math.isfinite(value) or value <= 0 for value in raw.values()):
        raise TurkishCorpusError("measured source weights must be finite and positive")
    total = sum(raw.values())
    keys = sorted(raw)
    result: dict[str, float] = {}
    running = 0.0
    for key in keys[:-1]:
        value = round(raw[key] / total, 12)
        result[key] = value
        running += value
    result[keys[-1]] = round(1.0 - running, 12)
    if result[keys[-1]] <= 0 or abs(sum(result.values()) - 1.0) > 1e-12:
        raise TurkishCorpusError("measured source-weight normalization drift")
    return result


def build_packing_preflight_report(
    pool_dir: str | Path,
    policy: Mapping[str, Any],
    tokenizer_dir: str | Path,
    output_path: str | Path,
    *,
    max_documents_per_rank_and_mixture: int = 8192,
    projection_safety_factor: float = 1.08,
) -> dict[str, Any]:
    """Measure crop retention on a bounded, deterministic pool sample.

    This report is a planning estimate only.  Its approved target prevents the
    guaranteed 67B under-allocation; the final full-corpus simulation remains
    the hard gate.
    """

    validate_corpus_policy(policy)
    if max_documents_per_rank_and_mixture < 2048:
        raise TurkishCorpusError("packing sample needs at least 2048 documents per rank/mixture")
    if not math.isfinite(projection_safety_factor) or projection_safety_factor < 1.0:
        raise TurkishCorpusError("packing projection safety factor must be >=1")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite packing report: {destination}")
    pool_root = Path(pool_dir)
    pool_manifest = load_json_strict(pool_root / "corpus_manifest.json")
    pool_hash = verify_manifest_hash(pool_manifest)
    if pool_manifest.get("backend_scope") != "production_glotlid_datatrove":
        raise TurkishCorpusError("packing planning refuses a reference pool")
    validate_qa_gate(pool_root, pool_manifest)

    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.strict_tokenizer import verify_tokenizer_package
    from nanochat.packing_capacity import (
        run_upstream_loader_parity_fixture,
        simulate_bestfit_rank,
    )

    tokenizer_root = Path(tokenizer_dir)
    package = verify_tokenizer_package(
        tokenizer_root / "package_manifest.json",
        expected_name=TOKENIZER_NAME,
        expected_vocab_size=VOCAB_SIZE,
    ).manifest
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_root))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise TurkishCorpusError("packing preflight tokenizer vocabulary drift")

    records_by_mixture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in pool_manifest["files"]:
        if record["split"] == "train":
            records_by_mixture[str(record["mixture_id"])].append(record)
    expected_mixtures = {str(bucket["id"]) for bucket in policy["mixture"]}
    if set(records_by_mixture) != expected_mixtures:
        raise TurkishCorpusError("packing sample lacks one or more train mixtures")

    units_by_mixture: dict[str, list[tuple[str, int]]] = {}
    for mixture_id, records in records_by_mixture.items():
        units: list[tuple[str, int]] = []
        for record in records:
            path = pool_root / record["path"]
            if (
                path.is_symlink()
                or path.stat().st_size != record["size_bytes"]
                or file_sha256(path) != record["sha256"]
            ):
                raise TurkishCorpusError(f"packing sample pool drift: {record['path']}")
            parquet = pq.ParquetFile(path)
            units.extend((str(record["path"]), index) for index in range(parquet.num_row_groups))
        units.sort(
            key=lambda unit: hashlib.sha256(
                f"{pool_hash}\0packing-sample-v1\0{mixture_id}\0{unit[0]}\0{unit[1]}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        units_by_mixture[mixture_id] = units

    capacity_policy = policy["materialization"]["packing_capacity_gate"]
    worlds: dict[str, Any] = {}
    minimum_efficiency: dict[str, float] = {key: 1.0 for key in expected_mixtures}
    maximum_mean_document_tokens = 0.0
    for raw_world_size in capacity_policy["world_sizes"]:
        world_size = int(raw_world_size)
        mixture_results: dict[str, Any] = {}
        for mixture_id in sorted(expected_mixtures):
            ranks = [
                simulate_bestfit_rank(
                    _sampled_pool_rank_batches(
                        pool_root,
                        units_by_mixture[mixture_id],
                        tokenizer,
                        mixture_id=mixture_id,
                        rank=rank,
                        world_size=world_size,
                        max_documents=max_documents_per_rank_and_mixture,
                        tokenizer_batch_size=int(capacity_policy["tokenizer_batch_size"]),
                    ),
                    B=int(capacity_policy["device_batch_sequences"]),
                    T=int(capacity_policy["max_seq_len"]),
                    buffer_size=int(capacity_policy["buffer_size"]),
                )
                for rank in range(world_size)
            ]
            if any(int(rank["completed_microbatches"]) < 4 for rank in ranks):
                raise TurkishCorpusError(
                    f"packing sample is too small/starved for {mixture_id} at ws{world_size}"
                )
            retained = sum(
                int(rank["retained_positions"]["mixture"].get(mixture_id, 0))
                for rank in ranks
            )
            consumed = sum(
                int(rank["consumed_source_elements"]["mixture"].get(mixture_id, 0))
                for rank in ranks
            )
            cropped = sum(
                int(rank["cropped_tokens"]["mixture"].get(mixture_id, 0))
                for rank in ranks
            )
            loaded_documents = sum(int(rank["loaded_documents"]) for rank in ranks)
            loaded_tokens = sum(
                int(rank["loaded_source_tokens_with_bos"]) for rank in ranks
            )
            source_cost = consumed + cropped
            if retained <= 0 or source_cost <= 0 or loaded_documents <= 0:
                raise TurkishCorpusError("packing sample produced no measurable retention")
            efficiency = retained / source_cost
            minimum_efficiency[mixture_id] = min(
                minimum_efficiency[mixture_id], efficiency
            )
            mean_document_tokens = loaded_tokens / loaded_documents
            maximum_mean_document_tokens = max(
                maximum_mean_document_tokens, mean_document_tokens
            )
            mixture_results[mixture_id] = {
                "rank_completed_microbatches": [
                    int(rank["completed_microbatches"]) for rank in ranks
                ],
                "sample_loaded_documents": loaded_documents,
                "sample_loaded_source_tokens_with_bos": loaded_tokens,
                "retained_target_positions": retained,
                "consumed_source_elements": consumed,
                "cropped_source_tokens": cropped,
                "measured_retention_efficiency": efficiency,
                "mean_document_tokens_with_bos": mean_document_tokens,
            }
        worlds[str(world_size)] = {"mixtures": mixture_results}

    intended = {str(bucket["id"]): float(bucket["weight"]) for bucket in policy["mixture"]}
    raw_adjusted = {
        mixture: intended[mixture] / minimum_efficiency[mixture]
        for mixture in sorted(intended)
    }
    recommended_weights = _normalized_measured_weights(raw_adjusted)
    source_cost_per_retained_position = sum(raw_adjusted.values())
    required_steps_with_margin = math.ceil(
        int(capacity_policy["required_optimizer_steps"])
        * (1.0 + float(capacity_policy["safety_margin_fraction"]))
    )
    required_positions = required_steps_with_margin * int(
        capacity_policy["global_batch_tokens"]
    )
    terminal_buffer_reserve = math.ceil(
        maximum_mean_document_tokens
        * int(capacity_policy["buffer_size"])
        * max(int(value) for value in capacity_policy["world_sizes"])
    )
    projected = math.ceil(
        required_positions
        * source_cost_per_retained_position
        * projection_safety_factor
        + terminal_buffer_reserve
    )
    quantum = int(capacity_policy["global_batch_tokens"])
    recommended_target = max(
        int(policy["materialization"]["target_family_tokens"]),
        math.ceil(projected / quantum) * quantum,
    )
    parity = run_upstream_loader_parity_fixture()
    report = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": PACKING_PREFLIGHT_REPORT_KIND,
            "policy_sha256": hashlib.sha256(
                canonical_json(policy).encode("utf-8")
            ).hexdigest(),
            "pool_manifest_sha256": pool_hash,
            "tokenizer_package_sha256": package["canonical_sha256"],
            "sample_contract": {
                "algorithm": "hashed_row_groups_per_mixture_virtual_rank_v1",
                "max_documents_per_rank_and_mixture": max_documents_per_rank_and_mixture,
                "tokenizer_encode": "batched prepend=BOS num_threads=4",
                "tokenizer_batch_size": capacity_policy["tokenizer_batch_size"],
                "B": capacity_policy["device_batch_sequences"],
                "T": capacity_policy["max_seq_len"],
                "buffer_size": capacity_policy["buffer_size"],
                "world_sizes": capacity_policy["world_sizes"],
                "pool_units_by_mixture": {
                    key: len(value) for key, value in sorted(units_by_mixture.items())
                },
            },
            "upstream_fixture_parity": parity,
            "world_measurements": worlds,
            "minimum_measured_retention_efficiency_by_mixture": dict(
                sorted(minimum_efficiency.items())
            ),
            "intended_retained_weights": dict(sorted(intended.items())),
            "recommended_source_weights": recommended_weights,
            "source_cost_per_retained_position": source_cost_per_retained_position,
            "projection_safety_factor": projection_safety_factor,
            "required_positions_with_capacity_margin": required_positions,
            "terminal_buffer_reserve_source_tokens": terminal_buffer_reserve,
            "recommended_source_token_target": recommended_target,
            "planning_estimate_only_final_exact_gate_required": True,
            "manual_approval_required": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, report)
    return report


def seal_packing_preflight_approval(
    report_path: str | Path,
    output_path: str | Path,
    *,
    reviewer: str,
    reviewed_at_utc: str,
    decision: str,
    notes: str = "",
) -> dict[str, Any]:
    report = load_json_strict(report_path)
    report_hash = verify_manifest_hash(report)
    if report.get("kind") != PACKING_PREFLIGHT_REPORT_KIND:
        raise TurkishCorpusError("unexpected packing preflight report")
    if not reviewer.strip() or not _RFC3339_UTC_RE.fullmatch(reviewed_at_utc):
        raise TurkishCorpusError("packing approval requires reviewer and RFC3339 UTC time")
    if decision not in {"accepted", "rejected"}:
        raise TurkishCorpusError("packing approval decision must be accepted/rejected")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite packing approval: {destination}")
    approval = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": PACKING_PREFLIGHT_APPROVAL_KIND,
            "packing_report_sha256": report_hash,
            "policy_sha256": report["policy_sha256"],
            "pool_manifest_sha256": report["pool_manifest_sha256"],
            "tokenizer_package_sha256": report["tokenizer_package_sha256"],
            "approved_source_token_target": report["recommended_source_token_target"],
            "approved_source_weights": report["recommended_source_weights"],
            "reviewer": reviewer.strip(),
            "reviewed_at_utc": reviewed_at_utc,
            "decision": decision,
            "notes": notes,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, approval)
    return approval


def validate_packing_preflight_gate(
    directory: str | Path,
    *,
    policy: Mapping[str, Any],
    pool_manifest_sha256: str,
    tokenizer_package_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(directory)
    report = load_json_strict(root / "packing_preflight_report.json")
    report_hash = verify_manifest_hash(report)
    approval = load_json_strict(root / "packing_preflight_approval.json")
    verify_manifest_hash(approval)
    policy_hash = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
    if (
        report.get("kind") != PACKING_PREFLIGHT_REPORT_KIND
        or approval.get("kind") != PACKING_PREFLIGHT_APPROVAL_KIND
        or approval.get("decision") != "accepted"
        or approval.get("packing_report_sha256") != report_hash
        or report.get("policy_sha256") != policy_hash
        or report.get("pool_manifest_sha256") != pool_manifest_sha256
        or report.get("tokenizer_package_sha256") != tokenizer_package_sha256
        or approval.get("policy_sha256") != policy_hash
        or approval.get("pool_manifest_sha256") != pool_manifest_sha256
        or approval.get("tokenizer_package_sha256") != tokenizer_package_sha256
        or approval.get("approved_source_token_target")
        != report.get("recommended_source_token_target")
        or approval.get("approved_source_weights")
        != report.get("recommended_source_weights")
    ):
        raise TurkishCorpusError("packing preflight report/approval binding drift")
    return report, approval


def materialize_final_corpus(
    pool_dir: str | Path,
    policy: Mapping[str, Any],
    tokenizer_dir: str | Path,
    tokenizer_quality_dir: str | Path,
    output_dir: str | Path,
    *,
    target_tokens: int,
    shard_target_tokens: int,
    git_commit: str,
    quota_headroom_bytes: int,
    packing_preflight_dir: str | Path,
) -> dict[str, Any]:
    """Promote a weighted, unique corpus without ever storing a second full copy."""

    validate_corpus_policy(policy)
    if target_tokens <= 0 or shard_target_tokens <= 0 or quota_headroom_bytes <= 0:
        raise TurkishCorpusError("token targets must be positive")
    if target_tokens < int(policy["materialization"]["target_family_tokens"]):
        raise TurkishCorpusError(
            "source-token target is below the frozen initial 40x planning floor"
        )
    pool_root = Path(pool_dir).resolve()
    destination = Path(output_dir).resolve()
    if pool_root == destination or pool_root in destination.parents or destination in pool_root.parents:
        raise TurkishCorpusError("pool and final corpus directories must be disjoint")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty final corpus directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    pool_manifest = load_json_strict(pool_root / "corpus_manifest.json")
    verify_manifest_hash(pool_manifest)
    if pool_manifest.get("backend_scope") != "production_glotlid_datatrove":
        raise TurkishCorpusError("final production corpus refuses a reference/smoke pool")
    ownership = validate_pool_ownership(pool_root, pool_manifest)
    qa_approval = validate_qa_gate(pool_root, pool_manifest)
    if pool_manifest.get("policy_sha256") != hashlib.sha256(
        canonical_json(policy).encode("utf-8")
    ).hexdigest():
        raise TurkishCorpusError("filtered pool and final policy differ")

    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.strict_tokenizer import verify_tokenizer_package

    tokenizer_root = Path(tokenizer_dir)
    verified_package = verify_tokenizer_package(
        tokenizer_root / "package_manifest.json",
        expected_name=TOKENIZER_NAME,
        expected_vocab_size=VOCAB_SIZE,
    )
    package = verified_package.manifest
    from nanochat.tokenizer_quality import validate_tokenizer_quality_gate

    tokenizer_quality_report, tokenizer_quality_approval = validate_tokenizer_quality_gate(
        tokenizer_quality_dir,
        expected_package_sha256=package["canonical_sha256"],
    )
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_root))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise TurkishCorpusError(
            f"tokenizer must contain exactly {VOCAB_SIZE} tokens, got {tokenizer.get_vocab_size()}"
        )
    receipt_path = tokenizer_root / "training_receipt.json"
    if not receipt_path.is_file():
        raise TurkishCorpusError("tokenizer training_receipt.json is required")
    tokenizer_receipt = load_json_strict(receipt_path)
    verify_manifest_hash(tokenizer_receipt)
    packing_report, packing_approval = validate_packing_preflight_gate(
        packing_preflight_dir,
        policy=policy,
        pool_manifest_sha256=pool_manifest["canonical_sha256"],
        tokenizer_package_sha256=package["canonical_sha256"],
    )
    if target_tokens != int(packing_approval["approved_source_token_target"]):
        raise TurkishCorpusError(
            "--target-tokens must equal the manually approved measured packing target"
        )
    approved_source_weights = {
        str(key): float(value)
        for key, value in packing_approval["approved_source_weights"].items()
    }

    pool_device = pool_root.stat().st_dev
    if destination.stat().st_dev != pool_device:
        raise TurkishCorpusError("disk-safe promotion requires pool/final on one filesystem")
    pool_bytes = sum(int(item["size_bytes"]) for item in pool_manifest["files"])
    physical_free_before = shutil.disk_usage(destination).free

    inventory_by_mixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count_peak = {"peak_rows": 0, "peak_utf8_bytes": 0}
    for record in sorted(pool_manifest["files"], key=lambda item: item["path"]):
        if record["split"] != "train":
            continue
        counted, peak = _count_pool_fragment(pool_root, record, tokenizer)
        if counted["encoded_tokens_with_bos"] > shard_target_tokens:
            raise TurkishCorpusError(
                f"pool fragment {record['path']} exceeds --shard-target-tokens; "
                "rebuild the bounded pool with fewer rows_per_output_file"
            )
        inventory_by_mixture[record["mixture_id"]].append(counted)
        for key in count_peak:
            count_peak[key] = max(count_peak[key], peak[key])
    for mixture_id, records in inventory_by_mixture.items():
        records.sort(
            key=lambda item: hashlib.sha256(
                f"{pool_manifest['canonical_sha256']}\0{item['path']}".encode("utf-8")
            ).hexdigest()
        )

    bucket_policy = {bucket["id"]: bucket for bucket in policy["mixture"]}
    available = {
        mixture_id: sum(item["encoded_tokens_with_bos"] for item in inventory_by_mixture[mixture_id])
        for mixture_id in bucket_policy
    }
    max_documents = {
        mixture_id: max(
            (item["max_document_tokens"] for item in inventory_by_mixture[mixture_id]),
            default=0,
        )
        for mixture_id in bucket_policy
    }
    from fractions import Fraction

    caps = {
        mixture_id: (
            Fraction(str(bucket_policy[mixture_id]["source_cap"])) * target_tokens
        ).numerator
        // (Fraction(str(bucket_policy[mixture_id]["source_cap"])) * target_tokens).denominator
        for mixture_id in bucket_policy
    }
    safe_capacity = {
        mixture_id: max(
            0,
            min(available[mixture_id], caps[mixture_id]) - max_documents[mixture_id],
        )
        for mixture_id in bucket_policy
    }
    effective_quotas, transfer_ledger, initial_quotas = allocate_fallback_quotas(
        policy,
        target_tokens=target_tokens,
        safe_capacity=safe_capacity,
        source_weights=approved_source_weights,
    )

    eval_source_bytes = sum(
        int(item["size_bytes"])
        for item in pool_manifest["files"]
        if item["split"] in {"val", "test"}
    )
    partial_peak_bytes = sum(
        max(
            (item["size_bytes"] for item in inventory_by_mixture[mixture_id]),
            default=0,
        )
        for mixture_id in bucket_policy
    )
    estimated_incremental_peak = (
        partial_peak_bytes + math.ceil(eval_source_bytes * 1.5) + 1_073_741_824
    )
    estimated_live_peak = pool_bytes + estimated_incremental_peak
    if estimated_incremental_peak > min(quota_headroom_bytes, physical_free_before):
        raise TurkishCorpusError(
            "insufficient physical/quota headroom for bounded promotion: "
            f"need={estimated_incremental_peak}, quota={quota_headroom_bytes}, "
            f"physical={physical_free_before}"
        )
    if estimated_live_peak > int(policy["materialization"]["max_peak_disk_bytes"]):
        raise TurkishCorpusError(
            "estimated pool+temporary live bytes exceed frozen materialization ceiling"
        )

    build_dir = destination / ".promotion"
    build_dir.mkdir()
    plans_by_mixture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    promotion_peak = {"peak_rows": 0, "peak_utf8_bytes": 0}
    for mixture_id in [bucket["id"] for bucket in policy["mixture"]]:
        accumulated = 0
        quota = effective_quotas[mixture_id]
        for item in inventory_by_mixture[mixture_id]:
            if accumulated >= quota:
                break
            remaining = quota - accumulated
            if item["encoded_tokens_with_bos"] <= remaining:
                plans_by_mixture[mixture_id].append(
                    {"mode": "hardlink", "source_path": pool_root / item["path"], **item}
                )
                accumulated += item["encoded_tokens_with_bos"]
                continue
            partial_path = build_dir / f"partial-{mixture_id}.parquet"
            partial, peak = _write_partial_fragment(
                pool_root / item["path"],
                partial_path,
                required_tokens=remaining,
                tokenizer=tokenizer,
            )
            partial["mixture_id"] = mixture_id
            partial["source_path"] = partial_path
            partial["mode"] = "partial"
            plans_by_mixture[mixture_id].append(partial)
            accumulated += partial["encoded_tokens_with_bos"]
            for key in promotion_peak:
                promotion_peak[key] = max(promotion_peak[key], peak[key])
            break
        if accumulated < quota:
            raise TurkishCorpusError(
                f"selection unexpectedly failed safe capacity for {mixture_id}: "
                f"quota={quota}, selected={accumulated}"
            )
        if accumulated > caps[mixture_id]:
            raise TurkishCorpusError(f"selection exceeded source cap for {mixture_id}")

    active = {key for key, values in plans_by_mixture.items() if values}
    positions: Counter[str] = Counter()
    emitted: Counter[str] = Counter()
    total_emitted = 0
    ordered_plans: list[dict[str, Any]] = []
    while active:
        choice = max(
            active,
            key=lambda key: (
                effective_quotas[key] * max(1, total_emitted) / target_tokens - emitted[key],
                key,
            ),
        )
        item = plans_by_mixture[choice][positions[choice]]
        positions[choice] += 1
        ordered_plans.append(item)
        emitted[choice] += item["encoded_tokens_with_bos"]
        total_emitted += item["encoded_tokens_with_bos"]
        if positions[choice] >= len(plans_by_mixture[choice]):
            active.remove(choice)

    train_files: list[dict[str, Any]] = []
    tokens_by_mixture: Counter[str] = Counter()
    tokens_by_source: Counter[str] = Counter()
    tokens_by_register: Counter[str] = Counter()
    documents_by_mixture: Counter[str] = Counter()
    documents_by_source: Counter[str] = Counter()
    documents_by_register: Counter[str] = Counter()
    for index, item in enumerate(ordered_plans):
        relative = Path(f"train-{index:05d}.parquet")
        final_path = destination / relative
        if item["mode"] == "hardlink":
            os.link(item["source_path"], final_path, follow_symlinks=False)
            promotion_mode = "same_filesystem_hardlink"
        else:
            os.replace(item["source_path"], final_path)
            promotion_mode = "bounded_partial_copy"
        if file_sha256(final_path) != item["sha256"]:
            raise TurkishCorpusError(f"promoted file hash drift: {relative}")
        mixture_id = item["mixture_id"]
        tokens_by_mixture[mixture_id] += item["encoded_tokens_with_bos"]
        documents_by_mixture[mixture_id] += item["rows"]
        for key, value in item["tokens_by_source"].items():
            tokens_by_source[key] += value
        for key, value in item["tokens_by_register"].items():
            tokens_by_register[key] += value
        for key, value in item["documents_by_source"].items():
            documents_by_source[key] += value
        for key, value in item["documents_by_register"].items():
            documents_by_register[key] += value
        train_files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": final_path.stat().st_size,
                "sha256": item["sha256"],
                "rows": item["rows"],
                "encoded_tokens_with_bos": item["encoded_tokens_with_bos"],
                "mixture_id": mixture_id,
                "promotion_mode": promotion_mode,
            }
        )
    total = sum(tokens_by_mixture.values())
    if total < target_tokens:
        raise TurkishCorpusError("selected final corpus does not cover the scheduled horizon")

    validation_path = destination / "validation.parquet"
    (
        validation_record,
        validation_tokens,
        validation_docs,
        validation_peak,
        validation_no_crop,
    ) = _write_eval_split(
        pool_root, pool_manifest, policy, tokenizer, validation_path, "val"
    )
    test_path = destination / "test" / "test.parquet"
    (
        test_record,
        test_tokens,
        test_docs,
        test_peak,
        test_no_crop,
    ) = _write_eval_split(pool_root, pool_manifest, policy, tokenizer, test_path, "test")

    write_json_atomic(destination / "parent_pool_manifest.json", pool_manifest)
    write_json_atomic(destination / "parent_pool_ownership.json", ownership)
    qa_report = load_json_strict(pool_root / "qa" / "qa_report.json")
    final_qa_dir = destination / "qa"
    final_qa_dir.mkdir()
    write_json_atomic(final_qa_dir / "qa_approval.json", qa_approval)
    write_json_atomic(final_qa_dir / "qa_report.json", qa_report)
    for key in ("jsonl", "plaintext"):
        record = qa_report["examples"][key]
        source_path = pool_root / "qa" / record["path"]
        target_path = final_qa_dir / record["path"]
        shutil.copyfile(source_path, target_path)
        if (
            target_path.stat().st_size != record["size_bytes"]
            or file_sha256(target_path) != record["sha256"]
        ):
            raise TurkishCorpusError(f"archived QA {key} file hash drift")
    tokenizer_quality_archive = destination / "tokenizer_quality"
    tokenizer_quality_archive.mkdir()
    write_json_atomic(
        tokenizer_quality_archive / "quality_report.json", tokenizer_quality_report
    )
    write_json_atomic(
        tokenizer_quality_archive / "quality_approval.json", tokenizer_quality_approval
    )

    ordered_files = [
        {key: item[key] for key in ("path", "size_bytes", "sha256")}
        for item in train_files
    ] + [
        {
            "path": validation_path.name,
            "size_bytes": validation_record["size_bytes"],
            "sha256": validation_record["sha256"],
        }
    ]
    parent_hash = pool_manifest["canonical_sha256"]
    synthetic_revision = hashlib.sha1(
        f"{parent_hash}:{package['canonical_sha256']}:{target_tokens}".encode("ascii"),
        usedforsecurity=False,
    ).hexdigest()
    dataset_manifest = seal_manifest(
        {
            "schema_version": "1.0",
            "manifest_type": "dataset",
            "profile": "strict",
            "dataset": {
                "repo_id": f"local-composite/{CORPUS_NAME}",
                "path": "interleaved_final",
                "requested_revision": synthetic_revision,
                "resolved_revision": synthetic_revision,
                "repo_type": "dataset",
            },
            "text_column": "text",
            "ordered_files": ordered_files,
            "validation_file": validation_path.name,
            "created_by": {
                "git_commit": git_commit,
                "tool": "scripts.build_turkish_pretrain_corpus",
            },
            "metadata": {
                "revision_semantics": "sha1_of_pool_tokenizer_horizon_not_hub_commit",
                "corpus_name": CORPUS_NAME,
                "parent_pool_manifest_sha256": parent_hash,
                "tokenizer_package_sha256": package["canonical_sha256"],
                "scheduled_prefix_tokens": target_tokens,
                "materialized_tokens_with_terminal_overhang": total,
                "qa_approval_sha256": qa_approval["canonical_sha256"],
                "tokenizer_quality_approval_sha256": tokenizer_quality_approval[
                    "canonical_sha256"
                ],
                "packing_preflight_report_sha256": packing_report[
                    "canonical_sha256"
                ],
                "packing_preflight_approval_sha256": packing_approval[
                    "canonical_sha256"
                ],
                "approved_source_weights": dict(
                    sorted(approved_source_weights.items())
                ),
                "validation_policy": validation_no_crop,
            },
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination / "fineweb2_manifest.json", dataset_manifest)

    # A source-token sum is not a capacity proof because the pinned upstream
    # loader discards cropped document tails.  Simulate the exact finite epoch
    # after materialization and fail closed for either supported topology.
    from nanochat.packing_capacity import (
        seal_capacity_receipt,
        simulate_final_corpus_capacity,
    )

    capacity_policy = policy["materialization"]["packing_capacity_gate"]
    capacity_simulation = simulate_final_corpus_capacity(
        destination,
        train_files,
        tokenizer,
        world_sizes=capacity_policy["world_sizes"],
        B=int(capacity_policy["device_batch_sequences"]),
        T=int(capacity_policy["max_seq_len"]),
        buffer_size=int(capacity_policy["buffer_size"]),
        tokenizer_batch_size=int(capacity_policy["tokenizer_batch_size"]),
        global_batch_tokens=int(capacity_policy["global_batch_tokens"]),
        required_optimizer_steps=int(capacity_policy["required_optimizer_steps"]),
        safety_margin_fraction=float(capacity_policy["safety_margin_fraction"]),
    )
    capacity_receipt = seal_capacity_receipt(
        destination / "packing_capacity_receipt.json",
        simulation=capacity_simulation,
        dataset_manifest_sha256=dataset_manifest["canonical_sha256"],
        tokenizer_package_sha256=package["canonical_sha256"],
        intended_weights={
            bucket["id"]: float(bucket["weight"]) for bucket in policy["mixture"]
        },
        current_source_token_target=target_tokens,
        mix_absolute_tolerance=float(capacity_policy["mix_absolute_tolerance"]),
    )
    if capacity_receipt["gate_passed"] is not True:
        raise TurkishCorpusError(
            "exact best-fit capacity/mix gate failed; pool retained and cleanup forbidden; "
            "retry source target recommendation="
            f"{capacity_receipt['recommended_source_token_target_if_retry']}"
        )

    final_manifest = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": CORPUS_MANIFEST_KIND,
            "name": CORPUS_NAME,
            "stage": "final_interleaved",
            "policy_sha256": pool_manifest["policy_sha256"],
            "parent_pool_manifest_sha256": parent_hash,
            "source_receipt_sha256": pool_manifest["source_receipt_sha256"],
            "tokenizer": {
                "name": TOKENIZER_NAME,
                "vocab_size": VOCAB_SIZE,
                "package_sha256": package["canonical_sha256"],
                "training_receipt_sha256": tokenizer_receipt["canonical_sha256"],
                "quality_report_sha256": tokenizer_quality_report["canonical_sha256"],
                "quality_approval_sha256": tokenizer_quality_approval[
                    "canonical_sha256"
                ],
            },
            "created_by": {"git_commit": git_commit, "tool": "scripts.build_turkish_pretrain_corpus"},
            "language": "tur_Latn",
            "code_allowed": False,
            "split_policy": policy["splits"],
            "scheduled_train_tokens": target_tokens,
            "packing_preflight": {
                "report_sha256": packing_report["canonical_sha256"],
                "approval_sha256": packing_approval["canonical_sha256"],
                "approved_source_weights": dict(sorted(approved_source_weights.items())),
            },
            "materialized_train_tokens": total,
            "terminal_overhang_tokens": total - target_tokens,
            "terminal_overhang_upper_bound_tokens": sum(max_documents.values()),
            "initial_train_token_quotas": dict(sorted(initial_quotas.items())),
            "effective_train_token_quotas": dict(sorted(effective_quotas.items())),
            "fallback_transfer_ledger": transfer_ledger,
            "safe_capacity_by_mixture": dict(sorted(safe_capacity.items())),
            "available_tokens_by_mixture": dict(sorted(available.items())),
            "source_caps_by_mixture": dict(sorted(caps.items())),
            "train_tokens_by_mixture": dict(sorted(tokens_by_mixture.items())),
            "train_tokens_by_source": dict(sorted(tokens_by_source.items())),
            "train_tokens_by_register": dict(sorted(tokens_by_register.items())),
            "train_documents_by_mixture": dict(sorted(documents_by_mixture.items())),
            "train_documents_by_source": dict(sorted(documents_by_source.items())),
            "train_documents_by_register": dict(sorted(documents_by_register.items())),
            "validation_tokens_by_mixture": dict(sorted(validation_tokens.items())),
            "validation_documents_by_mixture": dict(sorted(validation_docs.items())),
            "validation_whole_document_no_crop": validation_no_crop,
            "test_tokens_by_mixture": dict(sorted(test_tokens.items())),
            "test_documents_by_mixture": dict(sorted(test_docs.items())),
            "test_whole_document_no_crop": test_no_crop,
            "quality_assurance": {
                "report_sha256": qa_report["canonical_sha256"],
                "approval_sha256": qa_approval["canonical_sha256"],
            },
            "resource_bounds": {
                "quota_headroom_bytes": quota_headroom_bytes,
                "physical_free_bytes_before": physical_free_before,
                "pool_fragment_bytes": pool_bytes,
                "estimated_incremental_peak_bytes": estimated_incremental_peak,
                "estimated_live_peak_bytes": estimated_live_peak,
                "token_count_peak_rows": count_peak["peak_rows"],
                "token_count_peak_utf8_bytes": count_peak["peak_utf8_bytes"],
                "partial_copy_peak_rows": promotion_peak["peak_rows"],
                "partial_copy_peak_utf8_bytes": promotion_peak["peak_utf8_bytes"],
                "validation_write_peak_rows": validation_peak["peak_rows"],
                "validation_write_peak_utf8_bytes": validation_peak["peak_utf8_bytes"],
                "test_write_peak_rows": test_peak["peak_rows"],
                "test_write_peak_utf8_bytes": test_peak["peak_utf8_bytes"],
            },
            "nanochat_dataset_manifest_sha256": dataset_manifest["canonical_sha256"],
            "packing_capacity": {
                "path": "packing_capacity_receipt.json",
                "sha256": capacity_receipt["canonical_sha256"],
                "all_worlds_pass": True,
                "cleanup_authorized": True,
            },
            "train_files": train_files,
            "validation_file": validation_record,
            "test_file": {
                **test_record,
                "path": test_path.relative_to(destination).as_posix(),
            },
            "test_data_accessed_by_training": False,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination / "corpus_manifest.json", final_manifest)

    from nanochat.experiment_manifest import verify_file_inventory

    verify_file_inventory(destination, dataset_manifest["ordered_files"])
    try:
        build_dir.rmdir()
    except OSError:
        pass
    promotion_receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_corpus_disk_safe_promotion",
            "parent_pool_manifest_sha256": pool_manifest["canonical_sha256"],
            "final_corpus_manifest_sha256": final_manifest["canonical_sha256"],
            "dataset_manifest_sha256": dataset_manifest["canonical_sha256"],
            "packing_capacity_receipt_sha256": capacity_receipt["canonical_sha256"],
            "cleanup_performed": False,
            "cleanup_requires_separate_verified_command": True,
            "retained_run_owned_fragment_files": len(
                ownership["generated_fragment_files"]
            ),
            "retained_run_owned_fragment_bytes": sum(
                item["size_bytes"] for item in ownership["generated_fragment_files"]
            ),
            "physical_free_bytes_after": shutil.disk_usage(destination).free,
            "no_user_inputs_removed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination / "promotion_receipt.json", promotion_receipt)
    return final_manifest


def cleanup_verified_pool(
    pool_dir: str | Path,
    final_corpus_dir: str | Path,
) -> dict[str, Any]:
    """Unlink only sealed run-owned pool fragments after every final gate passes.

    The intent record makes a partially interrupted cleanup safely resumable.
    Source downloads and arbitrary user files are never part of the authorized
    inventory and are never touched.
    """

    pool_root = Path(pool_dir).resolve()
    final_root = Path(final_corpus_dir).resolve()
    if (
        pool_root == final_root
        or pool_root in final_root.parents
        or final_root in pool_root.parents
    ):
        raise TurkishCorpusError("pool and final corpus cleanup roots must be disjoint")

    final_manifest = load_json_strict(final_root / "corpus_manifest.json")
    final_hash = verify_manifest_hash(final_manifest)
    dataset_manifest = load_json_strict(final_root / "fineweb2_manifest.json")
    dataset_hash = verify_manifest_hash(dataset_manifest)
    capacity = load_json_strict(final_root / "packing_capacity_receipt.json")
    capacity_hash = verify_manifest_hash(capacity)
    promotion = load_json_strict(final_root / "promotion_receipt.json")
    verify_manifest_hash(promotion)
    if (
        final_manifest.get("nanochat_dataset_manifest_sha256") != dataset_hash
        or final_manifest.get("packing_capacity", {}).get("sha256") != capacity_hash
        or promotion.get("final_corpus_manifest_sha256") != final_hash
        or promotion.get("dataset_manifest_sha256") != dataset_hash
        or promotion.get("packing_capacity_receipt_sha256") != capacity_hash
    ):
        raise TurkishCorpusError("final promotion/capacity binding drift")
    worlds = capacity.get("simulation", {}).get("worlds", {})
    if (
        capacity.get("kind") != "turkish_bestfit_capacity_receipt"
        or capacity.get("dataset_manifest_sha256") != dataset_hash
        or capacity.get("tokenizer_package_sha256")
        != final_manifest.get("tokenizer", {}).get("package_sha256")
        or capacity.get("gate_passed") is not True
        or capacity.get("cleanup_authorized") is not True
        or set(worlds) != {"8", "16"}
        or any(
            metrics.get("passes_40x_no_wrap_with_margin") is not True
            for metrics in worlds.values()
        )
    ):
        raise TurkishCorpusError(
            "pool cleanup requires passing ws8 and ws16 exact capacity receipts"
        )

    from nanochat.experiment_manifest import verify_file_inventory

    verify_file_inventory(final_root, dataset_manifest["ordered_files"])
    test_record = final_manifest["test_file"]
    test_path = final_root / test_record["path"]
    if (
        test_path.is_symlink()
        or test_path.stat().st_size != test_record["size_bytes"]
        or file_sha256(test_path) != test_record["sha256"]
    ):
        raise TurkishCorpusError("final test file drift forbids pool cleanup")

    pool_manifest = load_json_strict(pool_root / "corpus_manifest.json")
    pool_hash = verify_manifest_hash(pool_manifest)
    if final_manifest.get("parent_pool_manifest_sha256") != pool_hash:
        raise TurkishCorpusError("final corpus is bound to a different pool")
    ownership = load_json_strict(pool_root / POOL_OWNERSHIP_FILE)
    ownership_hash = verify_manifest_hash(ownership)
    expected_inventory = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in pool_manifest["files"]
    ]
    if (
        ownership.get("kind") != POOL_OWNERSHIP_KIND
        or ownership.get("pool_manifest_sha256") != pool_hash
        or ownership.get("generated_fragment_files") != expected_inventory
    ):
        raise TurkishCorpusError("pool cleanup ownership inventory drift")

    receipt_path = final_root / "pool_cleanup_receipt.json"
    if receipt_path.exists():
        prior = load_json_strict(receipt_path)
        verify_manifest_hash(prior)
        if (
            prior.get("final_corpus_manifest_sha256") != final_hash
            or prior.get("pool_manifest_sha256") != pool_hash
            or prior.get("capacity_receipt_sha256") != capacity_hash
            or prior.get("cleanup_completed") is not True
        ):
            raise TurkishCorpusError("existing pool cleanup receipt is stale")
        if any((pool_root / item["path"]).exists() for item in expected_inventory):
            raise TurkishCorpusError("cleanup receipt exists but pool fragments reappeared")
        return prior

    intent_path = final_root / "pool_cleanup_intent.json"
    if intent_path.exists():
        intent = load_json_strict(intent_path)
        verify_manifest_hash(intent)
        if (
            intent.get("final_corpus_manifest_sha256") != final_hash
            or intent.get("pool_manifest_sha256") != pool_hash
            or intent.get("pool_ownership_sha256") != ownership_hash
            or intent.get("capacity_receipt_sha256") != capacity_hash
            or intent.get("authorized_inventory") != expected_inventory
        ):
            raise TurkishCorpusError("existing cleanup intent is stale")
    else:
        # Before creating the durable intent, every owned fragment must still
        # exist and pass its inventory hash.
        validate_pool_ownership(pool_root, pool_manifest)
        intent = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": "turkish_pool_cleanup_intent",
                "final_corpus_manifest_sha256": final_hash,
                "pool_manifest_sha256": pool_hash,
                "pool_ownership_sha256": ownership_hash,
                "capacity_receipt_sha256": capacity_hash,
                "authorized_inventory": expected_inventory,
                "scope": "run_owned_pool_fragments_only",
                "canonical_sha256": None,
            }
        )
        write_json_atomic(intent_path, intent)

    removed_files = 0
    removed_bytes = 0
    for item in expected_inventory:
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "pool"
        ):
            raise TurkishCorpusError("cleanup inventory path escapes pool/ scope")
        path = pool_root / relative
        if not path.exists():
            # Missing files are permitted only after the durable intent exists;
            # this is the resumable-interruption case.
            continue
        if (
            path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise TurkishCorpusError(f"owned fragment drift forbids cleanup: {item['path']}")
        path.unlink()
        removed_files += 1
        removed_bytes += int(item["size_bytes"])

    pool_data_root = pool_root / "pool"
    if pool_data_root.exists() and not pool_data_root.is_symlink():
        directories = sorted(
            (path for path in pool_data_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            pool_data_root.rmdir()
        except OSError:
            pass
    if any((pool_root / item["path"]).exists() for item in expected_inventory):
        raise TurkishCorpusError("verified pool cleanup did not remove its full inventory")

    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_pool_cleanup_receipt",
            "final_corpus_manifest_sha256": final_hash,
            "pool_manifest_sha256": pool_hash,
            "pool_ownership_sha256": ownership_hash,
            "capacity_receipt_sha256": capacity_hash,
            "cleanup_intent_sha256": intent["canonical_sha256"],
            "cleanup_completed": True,
            "authorized_fragment_files": len(expected_inventory),
            "authorized_fragment_bytes": sum(
                int(item["size_bytes"]) for item in expected_inventory
            ),
            "removed_during_this_invocation_files": removed_files,
            "removed_during_this_invocation_bytes": removed_bytes,
            "user_inputs_removed": False,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(receipt_path, receipt)
    return receipt


def _iter_source_documents_from_parquet(
    root: Path, relative_path: str
) -> Iterator[Any]:
    from nanochat.exposure import SourceDocument

    parquet = pq.ParquetFile(root / relative_path)
    for row_group_index in range(parquet.num_row_groups):
        values = parquet.read_row_group(row_group_index, columns=["text"])["text"].to_pylist()
        for row_index, text in enumerate(values):
            yield SourceDocument(
                source_path=relative_path,
                row_group_index=row_group_index,
                row_index=row_index,
                text=str(text),
            )


def _verified_final_runtime_contract(
    root: Path, dataset_manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_hash = verify_manifest_hash(dataset_manifest)
    final_manifest = load_json_strict(root / "corpus_manifest.json")
    verify_manifest_hash(final_manifest)
    capacity = load_json_strict(root / "packing_capacity_receipt.json")
    capacity_hash = verify_manifest_hash(capacity)
    if (
        final_manifest.get("nanochat_dataset_manifest_sha256") != dataset_hash
        or final_manifest.get("packing_capacity", {}).get("sha256") != capacity_hash
        or capacity.get("dataset_manifest_sha256") != dataset_hash
        or capacity.get("tokenizer_package_sha256")
        != final_manifest.get("tokenizer", {}).get("package_sha256")
        or capacity.get("gate_passed") is not True
        or set(capacity.get("simulation", {}).get("worlds", {})) != {"8", "16"}
    ):
        raise TurkishCorpusError("final runtime contract lacks passing ws8/ws16 capacity")
    validation_policy = dataset_manifest.get("metadata", {}).get("validation_policy")
    if (
        not isinstance(validation_policy, Mapping)
        or validation_policy.get("policy") != "whole_document_no_crop"
        or validation_policy.get("max_payload_tokens")
        != D32_EVAL_MAX_PAYLOAD_TOKENS
        or validation_policy.get("max_encoded_tokens_with_bos")
        != D32_EVAL_ROW_CAPACITY
    ):
        raise TurkishCorpusError("validation whole-document no-crop contract is absent")
    return final_manifest, capacity


def write_runtime_exposure_manifests(
    final_corpus_dir: str | Path,
    *,
    study_sha256: str,
    tokenizer_sha256: str,
    seed: int,
    world_sizes: Sequence[int],
    target_token_positions: int,
    optimizer_steps: int,
    validation_target_bytes: int,
) -> dict[str, Any]:
    """Write fixed validation and world-size-bound equal-token plans."""

    from nanochat.exposure import build_exposure_manifest, build_training_exposure_plan

    root = Path(final_corpus_dir)
    dataset_manifest = load_json_strict(root / "fineweb2_manifest.json")
    dataset_hash = verify_manifest_hash(dataset_manifest)
    _final_manifest, capacity = _verified_final_runtime_contract(
        root, dataset_manifest
    )
    if dataset_manifest.get("validation_file") != "validation.parquet":
        raise TurkishCorpusError("final validation identity is not frozen")
    validation = build_exposure_manifest(
        _iter_source_documents_from_parquet(root, "validation.parquet"),
        mode="validation",
        target_value=validation_target_bytes,
        source_dataset_manifest_sha256=dataset_hash,
        study_sha256=study_sha256,
    )
    validation_path = root / "validation_exposure_manifest.json"
    if validation_path.exists():
        raise FileExistsError(f"refusing to overwrite {validation_path}")
    write_json_atomic(validation_path, validation)
    plans: dict[str, Any] = {}
    for world_size in world_sizes:
        plan = build_training_exposure_plan(
            estimand="equal_token",
            source_dataset_manifest_sha256=dataset_hash,
            study_sha256=study_sha256,
            tokenizer_sha256=tokenizer_sha256,
            seed=seed,
            world_size=int(world_size),
            target_token_positions=target_token_positions,
            derived_optimizer_steps=optimizer_steps,
        )
        path = root / f"training_exposure_ws{world_size}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        write_json_atomic(path, plan)
        plans[str(world_size)] = {
            "path": path.name,
            "canonical_sha256": plan["canonical_sha256"],
        }
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_pretrain_runtime_exposure_bundle",
            "dataset_manifest_sha256": dataset_hash,
            "study_sha256": study_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "validation": {
                "path": validation_path.name,
                "canonical_sha256": validation["canonical_sha256"],
            },
            "training_plans": plans,
            "packing_capacity_receipt_sha256": capacity["canonical_sha256"],
            "canonical_sha256": None,
        }
    )
    write_json_atomic(root / "runtime_exposure_receipt.json", receipt)
    return receipt


def write_d32_exposure_plan_index(
    final_corpus_dir: str | Path,
    *,
    family_id: str,
    study_manifest_sha256: str,
    tokenizer_artifact_sha256: str,
    validation_target_bytes: int = 16_777_216,
) -> dict[str, Any]:
    """Emit the complete d32 plan matrix and its single sealed index."""

    from nanochat.exposure import build_exposure_manifest, build_training_exposure_plan

    root = Path(final_corpus_dir)
    dataset_manifest = load_json_strict(root / "fineweb2_manifest.json")
    dataset_hash = verify_manifest_hash(dataset_manifest)
    _final_manifest, capacity = _verified_final_runtime_contract(
        root, dataset_manifest
    )
    validation = build_exposure_manifest(
        _iter_source_documents_from_parquet(root, "validation.parquet"),
        mode="validation",
        target_value=validation_target_bytes,
        source_dataset_manifest_sha256=dataset_hash,
        study_sha256=study_manifest_sha256,
    )
    validation_path = root / "validation_exposure_manifest.json"
    if validation_path.exists():
        raise FileExistsError(f"refusing to overwrite {validation_path}")
    write_json_atomic(validation_path, validation)
    plan_records: list[dict[str, Any]] = []
    for (
        key,
        purpose,
        world_size,
        seed,
        optimizer_steps,
        global_batch_tokens,
    ) in D32_EXPOSURE_MATRIX_V1:
        token_positions = optimizer_steps * global_batch_tokens
        plan = build_training_exposure_plan(
            estimand="equal_token",
            source_dataset_manifest_sha256=dataset_hash,
            study_sha256=study_manifest_sha256,
            tokenizer_sha256=tokenizer_artifact_sha256,
            seed=seed,
            world_size=world_size,
            target_token_positions=token_positions,
            derived_optimizer_steps=optimizer_steps,
        )
        filename = f"training_exposure_{key}.json"
        path = root / filename
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        write_json_atomic(path, plan)
        plan_records.append(
            {
                "key": key,
                "purpose": purpose,
                "path": filename,
                "world_size": world_size,
                "seed": seed,
                "optimizer_steps": optimizer_steps,
                "global_batch_tokens": global_batch_tokens,
                "token_positions": token_positions,
                "sha256": plan["canonical_sha256"],
            }
        )
    index = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_exposure_plan_index",
            "family_id": family_id,
            "study_manifest_sha256": study_manifest_sha256,
            "source_dataset_manifest_sha256": dataset_hash,
            "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
            "packing_capacity_receipt_sha256": capacity["canonical_sha256"],
            "validation": {
                "path": validation_path.name,
                "sha256": validation["canonical_sha256"],
            },
            "plans": plan_records,
            "canonical_sha256": None,
        }
    )
    index_path = root / "exposure_plan_index.json"
    if index_path.exists():
        raise FileExistsError(f"refusing to overwrite {index_path}")
    write_json_atomic(index_path, index)
    return index


__all__ = [
    "AuditDecision",
    "CORPUS_NAME",
    "DedupDecision",
    "D32_EXPOSURE_MATRIX_V1",
    "D32_GLOBAL_BATCH_TOKENS",
    "SQLiteMinHashDeduper",
    "TOKENIZER_NAME",
    "TurkishCorpusError",
    "VOCAB_SIZE",
    "assign_split",
    "audit_document",
    "build_packing_preflight_report",
    "canonical_text_hash",
    "cleanup_verified_pool",
    "load_corpus_policy",
    "materialize_filtered_pool",
    "materialize_final_corpus",
    "normalize_document",
    "representative_sample",
    "select_mixture_bucket",
    "seal_packing_preflight_approval",
    "source_lid_is_turkish",
    "source_lid_result",
    "stable_shuffle_key",
    "strict_hplt_register_scores",
    "turkish_text_confidence",
    "validate_corpus_policy",
    "validate_packing_preflight_gate",
    "validate_source_receipt",
    "write_tokenizer_sample",
    "write_runtime_exposure_manifests",
    "write_d32_exposure_plan_index",
]
