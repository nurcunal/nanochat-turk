"""
Measure tokenizer compression, fertility, boundary behavior, and throughput.

Tokenizer-only metrics cannot report model BPB; true BPB depends on a trained
model's loss. This script reports the tokenizer-side quantities we can measure
before pretraining: bytes/token, tokens/word fertility, word fragmentation,
round-trip safety, encode speed, vocabulary shape, and MorphBPE boundary
crossing rates when boundary-marked text is supplied.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from nanochat.dataset import DATA_DIR, parquets_iter_batched
from nanochat.morphology import MORPHEME_BOUNDARY, strip_morpheme_boundaries
from nanochat.morphology.morphbpe import strip_boundaries_with_offsets
from nanochat.tokenizer import (
    RustBPETokenizer,
    get_tokenizer,
    get_tokenizer_dir,
    get_tokenizer_name,
)
from tokenizers import BertWordPieceTokenizer
from tokenizers import Tokenizer as HFTokenizer


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def distribution_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def load_tokenizer_config(tokenizer_dir: str) -> dict[str, Any]:
    path = Path(tokenizer_dir) / "tokenizer_config.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(tokenizer_dir: str):
    if tokenizer_dir:
        return RustBPETokenizer.from_directory(tokenizer_dir)
    return get_tokenizer()


def hf_download_file(repo_id: str, filename: str) -> str:
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=True,
        )
    except Exception:
        return hf_hub_download(repo_id=repo_id, filename=filename)


class HuggingFaceTokenizerAdapter:
    """Small compatibility wrapper around Hugging Face `tokenizers` objects."""

    def __init__(
        self,
        backend: HFTokenizer,
        *,
        name: str,
        model_id: str,
        implementation: str,
        batch_size: int,
    ) -> None:
        self.backend = backend
        self.decode_strip = ""
        self.batch_size = batch_size
        self.tokenizer_config = {
            "name": name,
            "implementation": implementation,
            "source": "huggingface",
            "model_id": model_id,
            "vocab_size": backend.get_vocab_size(with_added_tokens=True),
        }

    @classmethod
    def from_identifier(
        cls,
        identifier: str,
        *,
        name: str,
        implementation: str,
        batch_size: int,
    ) -> "HuggingFaceTokenizerAdapter":
        path = Path(identifier)
        backend: HFTokenizer
        wants_wordpiece = "wordpiece" in implementation.lower()
        if path.is_file() and path.suffix == ".json":
            backend = HFTokenizer.from_file(str(path))
        elif path.is_dir() and (path / "tokenizer.json").is_file():
            backend = HFTokenizer.from_file(str(path / "tokenizer.json"))
        elif path.is_file() and path.name == "vocab.txt":
            backend = BertWordPieceTokenizer(str(path), lowercase=False, strip_accents=False)
        elif path.is_dir() and (path / "vocab.txt").is_file():
            backend = BertWordPieceTokenizer(
                str(path / "vocab.txt"),
                lowercase=False,
                strip_accents=False,
            )
        elif wants_wordpiece:
            vocab_path = hf_download_file(identifier, "vocab.txt")
            backend = BertWordPieceTokenizer(
                vocab_path,
                lowercase=False,
                strip_accents=False,
            )
        else:
            tokenizer_path = hf_download_file(identifier, "tokenizer.json")
            backend = HFTokenizer.from_file(tokenizer_path)
        resolved_name = name or identifier
        resolved_impl = implementation or type(backend.model).__name__.lower()
        return cls(
            backend,
            name=resolved_name,
            model_id=identifier,
            implementation=resolved_impl,
            batch_size=batch_size,
        )

    def encode(self, text_or_texts: str | list[str], num_threads: int = 0):
        del num_threads
        if isinstance(text_or_texts, str):
            return self.backend.encode(text_or_texts, add_special_tokens=False).ids
        return self.encode_with_offsets(text_or_texts)[0]

    def encode_with_offsets(
        self,
        texts: list[str],
    ) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
        all_ids: list[list[int]] = []
        all_offsets: list[list[tuple[int, int]]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            encodings = self.backend.encode_batch(chunk, add_special_tokens=False)
            all_ids.extend(encoding.ids for encoding in encodings)
            all_offsets.extend(encoding.offsets for encoding in encodings)
        return all_ids, all_offsets

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode(ids, skip_special_tokens=False)

    def get_vocab_size(self) -> int:
        return self.backend.get_vocab_size(with_added_tokens=True)


def char_offsets_to_byte_offsets(text: str, char_offsets: tuple[int, ...]) -> set[int]:
    wanted = set(char_offsets)
    out: set[int] = set()
    byte_pos = 0
    for idx, ch in enumerate(text):
        if idx in wanted:
            out.add(byte_pos)
        byte_pos += len(ch.encode("utf-8"))
    if len(text) in wanted:
        out.add(byte_pos)
    return out


def token_byte_lengths(tokenizer, ids: list[int]) -> list[int]:
    return [len(tokenizer.enc.decode_single_token_bytes(token_id)) for token_id in ids]


def boundary_violation_stats(
    tokenizer,
    encoded: list[list[int]],
    visible_docs: list[str],
    boundary_offsets_by_doc: list[tuple[int, ...]],
    token_offsets_by_doc: list[list[tuple[int, int]]] | None = None,
) -> dict[str, float | int]:
    token_len_cache: dict[int, int] = {}
    boundary_count = 0
    crossed_boundaries = 0
    crossing_tokens = 0
    docs_with_crossing = 0
    eligible_docs = 0

    for doc_idx, (ids, text, char_offsets) in enumerate(zip(encoded, visible_docs, boundary_offsets_by_doc)):
        if not char_offsets:
            continue
        eligible_docs += 1
        boundary_count += len(char_offsets)
        doc_crossing = False
        if token_offsets_by_doc is not None:
            sorted_offsets = sorted(char_offsets)
            token_offsets = token_offsets_by_doc[doc_idx]
            for start, end in token_offsets:
                left = bisect.bisect_right(sorted_offsets, start)
                right = bisect.bisect_left(sorted_offsets, end)
                crossed_here = right - left
                if crossed_here:
                    crossing_tokens += 1
                    crossed_boundaries += crossed_here
                    doc_crossing = True
        else:
            byte_offsets = sorted(char_offsets_to_byte_offsets(text, char_offsets))
            cursor = 0
            for token_id in ids:
                token_len = token_len_cache.get(token_id)
                if token_len is None:
                    token_len = len(tokenizer.enc.decode_single_token_bytes(token_id))
                    token_len_cache[token_id] = token_len
                start = cursor
                end = cursor + token_len
                left = bisect.bisect_right(byte_offsets, start)
                right = bisect.bisect_left(byte_offsets, end)
                crossed_here = right - left
                if crossed_here:
                    crossing_tokens += 1
                    crossed_boundaries += crossed_here
                    doc_crossing = True
                cursor = end
        if doc_crossing:
            docs_with_crossing += 1

    return {
        "eligible_docs": eligible_docs,
        "boundary_count": boundary_count,
        "crossing_tokens": crossing_tokens,
        "crossed_boundaries": crossed_boundaries,
        "docs_with_crossing": docs_with_crossing,
        "crossed_boundary_rate": safe_div(crossed_boundaries, boundary_count),
        "docs_with_crossing_rate": safe_div(docs_with_crossing, eligible_docs),
    }


def word_fertility_stats(
    tokenizer,
    docs: list[str],
    *,
    max_words: int,
    num_threads: int,
) -> dict[str, Any]:
    words: list[str] = []
    for doc in docs:
        words.extend(WORD_RE.findall(doc))
        if max_words > 0 and len(words) >= max_words:
            words = words[:max_words]
            break
    if not words:
        return {
            "sample_words": 0,
            "unique_words": 0,
            "tokens": 0,
            "tokens_per_word": 0.0,
            "single_token_word_rate": 0.0,
            "multi_token_word_rate": 0.0,
            "token_count_distribution": distribution_stats([]),
            "long_word_tokens_per_word": 0.0,
            "long_word_count": 0,
        }

    unique_words = list(dict.fromkeys(words))
    encoded_unique = tokenizer.encode(unique_words, num_threads=num_threads)
    lengths_by_word = {
        word: len(ids)
        for word, ids in zip(unique_words, encoded_unique)
    }
    lengths = [lengths_by_word[word] for word in words]
    long_lengths = [
        length for word, length in zip(words, lengths)
        if len(word) >= 8
    ]
    single = sum(1 for length in lengths if length == 1)
    return {
        "sample_words": len(words),
        "unique_words": len(unique_words),
        "tokens": sum(lengths),
        "tokens_per_word": safe_div(sum(lengths), len(words)),
        "single_token_word_rate": safe_div(single, len(words)),
        "multi_token_word_rate": safe_div(len(words) - single, len(words)),
        "token_count_distribution": distribution_stats(lengths),
        "long_word_tokens_per_word": safe_div(sum(long_lengths), len(long_lengths)),
        "long_word_count": len(long_lengths),
    }


def word_fertility_stats_from_words(
    tokenizer,
    words: list[str],
    *,
    num_threads: int,
) -> dict[str, Any]:
    if not words:
        return {
            "sample_words": 0,
            "unique_words": 0,
            "tokens": 0,
            "tokens_per_word": 0.0,
            "single_token_word_rate": 0.0,
            "multi_token_word_rate": 0.0,
            "token_count_distribution": distribution_stats([]),
            "long_word_tokens_per_word": 0.0,
            "long_word_count": 0,
        }

    unique_words = list(dict.fromkeys(words))
    encoded_unique = tokenizer.encode(unique_words, num_threads=num_threads)
    lengths_by_word = {
        word: len(ids)
        for word, ids in zip(unique_words, encoded_unique)
    }
    lengths = [lengths_by_word[word] for word in words]
    long_lengths = [
        length for word, length in zip(words, lengths)
        if len(word) >= 8
    ]
    single = sum(1 for length in lengths if length == 1)
    return {
        "sample_words": len(words),
        "unique_words": len(unique_words),
        "tokens": sum(lengths),
        "tokens_per_word": safe_div(sum(lengths), len(words)),
        "single_token_word_rate": safe_div(single, len(words)),
        "multi_token_word_rate": safe_div(len(words) - single, len(words)),
        "token_count_distribution": distribution_stats(lengths),
        "long_word_tokens_per_word": safe_div(sum(long_lengths), len(long_lengths)),
        "long_word_count": len(long_lengths),
    }


def vocabulary_stats(tokenizer) -> dict[str, Any]:
    if isinstance(tokenizer, HuggingFaceTokenizerAdapter):
        vocab = tokenizer.backend.get_vocab(with_added_tokens=True)
        decoded_lengths: list[int] = []
        utf8_decodable = 0
        for token_id in sorted(vocab.values()):
            decoded = tokenizer.decode([token_id])
            if not decoded:
                continue
            decoded_lengths.append(len(decoded.encode("utf-8")))
            utf8_decodable += 1
        return {
            "vocab_size": tokenizer.get_vocab_size(),
            "special_tokens": tokenizer.get_vocab_size() - len(tokenizer.backend.get_vocab(with_added_tokens=False)),
            "mergeable_tokens_seen": len(decoded_lengths),
            "utf8_decodable_token_rate": safe_div(utf8_decodable, len(decoded_lengths)),
            "token_byte_length_distribution": distribution_stats(decoded_lengths),
        }

    special_ids = {
        tokenizer.encode_special(token)
        for token in tokenizer.get_special_tokens()
    }
    byte_lengths: list[int] = []
    utf8_decodable = 0
    for token_id in range(tokenizer.get_vocab_size()):
        if token_id in special_ids:
            continue
        try:
            token_bytes = tokenizer.enc.decode_single_token_bytes(token_id)
        except KeyError:
            continue
        byte_lengths.append(len(token_bytes))
        try:
            token_bytes.decode("utf-8")
            utf8_decodable += 1
        except UnicodeDecodeError:
            pass
    return {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": len(special_ids),
        "mergeable_tokens_seen": len(byte_lengths),
        "utf8_decodable_token_rate": safe_div(utf8_decodable, len(byte_lengths)),
        "token_byte_length_distribution": distribution_stats(byte_lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer ablation metrics")
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="",
        help="Tokenizer directory. Default = active nanochat tokenizer.",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="",
        help=(
            "Hugging Face tokenizer repo id or local tokenizer.json/vocab.txt path. "
            "Uses tokenizer files only, never model weights."
        ),
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="",
        help="Optional display name for --hf-tokenizer metrics.",
    )
    parser.add_argument(
        "--tokenizer-implementation",
        type=str,
        default="",
        help="Optional implementation label for --hf-tokenizer metrics.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Optional parquet directory. Default = active nanochat dataset dir.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=os.environ.get("NANOCHAT_TEXT_COLUMN", "text"),
        help="Parquet text column to evaluate.",
    )
    parser.add_argument(
        "--input-has-morph-boundaries",
        action="store_true",
        help=(
            "Strip morpheme boundary markers before encoding and compute "
            "token-crosses-morpheme-boundary metrics."
        ),
    )
    parser.add_argument("--morph-boundary", type=str, default=MORPHEME_BOUNDARY)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10000,
        help="Maximum documents to evaluate. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--max-word-metrics",
        type=int,
        default=200000,
        help=(
            "Maximum word tokens for isolated word fertility metrics. "
            "Use 0 for all words in sampled docs."
        ),
    )
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--hf-batch-size", type=int, default=1000)
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--no-report", action="store_true", help="Do not write into nanochat.report.")
    args = parser.parse_args()

    if args.hf_tokenizer and args.tokenizer_dir:
        raise ValueError("Use only one of --hf-tokenizer or --tokenizer-dir")

    if args.hf_tokenizer:
        tokenizer = HuggingFaceTokenizerAdapter.from_identifier(
            args.hf_tokenizer,
            name=args.tokenizer_name,
            implementation=args.tokenizer_implementation,
            batch_size=args.hf_batch_size,
        )
        tokenizer_dir = args.hf_tokenizer
        tokenizer_config = tokenizer.tokenizer_config
    else:
        tokenizer_dir = args.tokenizer_dir or get_tokenizer_dir()
        tokenizer_config = load_tokenizer_config(tokenizer_dir)
        tokenizer = load_tokenizer(args.tokenizer_dir)
    decode_strip = getattr(tokenizer, "decode_strip", "")
    resolved_data_dir = args.data_dir or DATA_DIR

    docs_count = 0
    total_tokens = 0
    total_bytes = 0
    total_chars = 0
    total_words = 0
    unique_tokens: set[int] = set()
    roundtrip_failures = 0
    docs_with_decode_strip = 0
    encode_seconds = 0.0
    token_lengths: list[int] = []
    word_metric_words: list[str] = []
    boundary_totals = {
        "eligible_docs": 0,
        "boundary_count": 0,
        "crossing_tokens": 0,
        "crossed_boundaries": 0,
        "docs_with_crossing": 0,
    }

    for batch in parquets_iter_batched(
        split=args.split,
        data_dir=resolved_data_dir,
        text_column=args.text_column,
    ):
        if args.max_docs > 0:
            remaining_docs = args.max_docs - docs_count
            if remaining_docs <= 0:
                break
            batch = batch[:remaining_docs]
        if not batch:
            continue

        visible_docs: list[str] = []
        boundary_offsets_by_doc: list[tuple[int, ...]] = []
        for doc in batch:
            if decode_strip and decode_strip in doc:
                docs_with_decode_strip += 1
            if args.input_has_morph_boundaries:
                visible, offsets = strip_boundaries_with_offsets(
                    doc,
                    boundary=args.morph_boundary,
                )
                visible_docs.append(visible)
                boundary_offsets_by_doc.append(offsets)
            else:
                visible_docs.append(
                    strip_morpheme_boundaries(doc, decode_strip)
                    if decode_strip
                    else doc
                )
                boundary_offsets_by_doc.append(())

        t0 = time.time()
        if isinstance(tokenizer, HuggingFaceTokenizerAdapter):
            encoded, token_offsets_by_doc = tokenizer.encode_with_offsets(visible_docs)
        else:
            encoded = tokenizer.encode(visible_docs, num_threads=args.num_threads)
            token_offsets_by_doc = None
        encode_seconds += time.time() - t0

        batch_token_lengths = [len(ids) for ids in encoded]
        token_lengths.extend(batch_token_lengths)
        total_tokens += sum(batch_token_lengths)
        total_bytes += sum(len(doc.encode("utf-8")) for doc in visible_docs)
        total_chars += sum(len(doc) for doc in visible_docs)
        batch_words = [WORD_RE.findall(doc) for doc in visible_docs]
        total_words += sum(len(words) for words in batch_words)
        for ids in encoded:
            unique_tokens.update(ids)
        roundtrip_failures += sum(
            1 for doc, ids in zip(visible_docs, encoded)
            if tokenizer.decode(ids) != doc
        )

        if args.max_word_metrics == 0 or len(word_metric_words) < args.max_word_metrics:
            for words in batch_words:
                if args.max_word_metrics > 0:
                    remaining_words = args.max_word_metrics - len(word_metric_words)
                    if remaining_words <= 0:
                        break
                    word_metric_words.extend(words[:remaining_words])
                else:
                    word_metric_words.extend(words)

        batch_boundary_stats = boundary_violation_stats(
            tokenizer,
            encoded,
            visible_docs,
            boundary_offsets_by_doc,
            token_offsets_by_doc,
        )
        for key in boundary_totals:
            boundary_totals[key] += int(batch_boundary_stats.get(key, 0) or 0)

        docs_count += len(visible_docs)
        if args.max_docs > 0 and docs_count >= args.max_docs:
            break

    if docs_count == 0:
        raise RuntimeError("No documents found for tokenizer metrics")

    boundary_stats = dict(boundary_totals)
    boundary_stats["crossed_boundary_rate"] = safe_div(
        boundary_stats["crossed_boundaries"],
        boundary_stats["boundary_count"],
    )
    boundary_stats["docs_with_crossing_rate"] = safe_div(
        boundary_stats["docs_with_crossing"],
        boundary_stats["eligible_docs"],
    )
    boundary_stats["crossing_tokens_per_1k_tokens"] = (
        1000.0 * safe_div(boundary_stats["crossing_tokens"], total_tokens)
    )

    metrics = {
        "tokenizer_name": tokenizer_config.get("name") or get_tokenizer_name(),
        "tokenizer_dir": tokenizer_dir,
        "tokenizer_config": tokenizer_config,
        "split": args.split,
        "data_dir": resolved_data_dir,
        "text_column": args.text_column,
        "input_has_morph_boundaries": args.input_has_morph_boundaries,
        "morph_boundary_codepoint": (
            f"U+{ord(args.morph_boundary):04X}"
            if len(args.morph_boundary) == 1
            else ""
        ),
        "docs": docs_count,
        "bytes": total_bytes,
        "chars": total_chars,
        "words": total_words,
        "tokens": total_tokens,
        "unique_tokens_in_sample": len(unique_tokens),
        "unique_token_rate_in_sample": safe_div(len(unique_tokens), tokenizer.get_vocab_size()),
        "decode_strip": decode_strip,
        "docs_with_decode_strip": docs_with_decode_strip,
        "bytes_per_token": safe_div(total_bytes, total_tokens),
        "chars_per_token": safe_div(total_chars, total_tokens),
        "tokens_per_byte": safe_div(total_tokens, total_bytes),
        "tokens_per_char": safe_div(total_tokens, total_chars),
        "tokens_per_word": safe_div(total_tokens, total_words),
        "token_fertility": safe_div(total_tokens, total_words),
        "tokens_per_doc_distribution": distribution_stats(token_lengths),
        "word_fertility_isolated": word_fertility_stats_from_words(
            tokenizer,
            word_metric_words,
            num_threads=args.num_threads,
        ),
        "vocabulary": vocabulary_stats(tokenizer),
        "morph_boundary": boundary_stats,
        "roundtrip_failures": roundtrip_failures,
        "roundtrip_failure_rate": safe_div(roundtrip_failures, docs_count),
        "encode_seconds": encode_seconds,
        "encode_docs_per_sec": safe_div(docs_count, encode_seconds),
        "encode_tokens_per_sec": safe_div(total_tokens, encode_seconds),
    }

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if not args.no_report:
        from nanochat.report import get_report
        get_report().log(section="Tokenizer metrics", data=[metrics])


if __name__ == "__main__":
    main()
