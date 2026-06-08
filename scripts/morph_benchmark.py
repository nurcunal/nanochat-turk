"""Benchmark Turkish morphology segmenters on a sampled corpus.

This is a smoke/selection tool for tokenizer experiments. It samples word-like
tokens from FineWeb-2 Turkish parquet shards, segments unique word types, then
reports frequency-weighted segmentation statistics.

Examples:
    NANOCHAT_BASE_DIR=/private/tmp/nanochat-turk-morph-smoke \
      python -m scripts.morph_benchmark --max-words 1000000

    TRMORPH_SEGMENT_FST=/private/tmp/TRmorph/segment.fst \
      python -m scripts.morph_benchmark --backend trmorph --max-words 100000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

from nanochat.dataset import list_parquet_files
from nanochat.morphology import (
    SegmenterUnavailable,
    SegmentationError,
    create_segmenter,
    iter_word_spans,
)


def iter_parquet_texts(paths: Iterable[str], text_column: str) -> Iterable[str]:
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=[text_column])
            for value in rg.column(text_column).to_pylist():
                yield "" if value is None else str(value)


def collect_words(
    *,
    parquet_paths: list[str],
    text_column: str,
    max_words: int,
    max_docs: int,
    progress_every_docs: int,
) -> tuple[Counter[str], dict[str, int]]:
    counts: Counter[str] = Counter()
    total_words = 0
    total_docs = 0
    total_chars = 0
    total_bytes = 0

    for text in iter_parquet_texts(parquet_paths, text_column):
        total_docs += 1
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        for _start, _end, word in iter_word_spans(text):
            counts[word] += 1
            total_words += 1
            if max_words > 0 and total_words >= max_words:
                return counts, {
                    "docs": total_docs,
                    "words": total_words,
                    "chars": total_chars,
                    "bytes": total_bytes,
                }
        if max_docs > 0 and total_docs >= max_docs:
            break
        if progress_every_docs > 0 and total_docs % progress_every_docs == 0:
            print(
                f"... collected docs={total_docs:,} words={total_words:,} "
                f"unique={len(counts):,}",
                flush=True,
            )

    return counts, {
        "docs": total_docs,
        "words": total_words,
        "chars": total_chars,
        "bytes": total_bytes,
    }


def cache_metadata(
    *,
    parquet_paths: list[str],
    text_column: str,
    max_words: int,
    max_docs: int,
) -> dict:
    return {
        "parquet_files": [os.path.abspath(path) for path in parquet_paths],
        "text_column": text_column,
        "max_words": max_words,
        "max_docs": max_docs,
    }


def load_word_counts_cache(
    path: str,
    *,
    expected_metadata: dict,
) -> tuple[Counter[str], dict[str, int]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unsupported word-count cache payload in {path}")

    actual_metadata = payload.get("metadata")
    if actual_metadata != expected_metadata:
        raise RuntimeError(
            "Word-count cache metadata does not match this benchmark request. "
            f"Cache: {actual_metadata!r}; requested: {expected_metadata!r}. "
            "Use a different cache path or remove the stale cache."
        )

    counts = payload.get("counts")
    sample = payload.get("sample")
    if not isinstance(counts, Counter) or not isinstance(sample, dict):
        raise RuntimeError(f"Unsupported word-count cache format in {path}")
    return counts, sample


def save_word_counts_cache(
    path: str,
    *,
    counts: Counter[str],
    sample: dict[str, int],
    metadata: dict,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "metadata": metadata,
        "sample": sample,
        "counts": counts,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def select_words(
    counts: Counter[str],
    *,
    mode: str,
    max_unique_words: int,
) -> tuple[list[str], dict]:
    if mode == "lexical":
        words = sorted(counts)
    elif mode == "frequency":
        words = sorted(counts, key=lambda word: (-counts[word], word))
    elif mode == "hash":
        words = sorted(
            counts,
            key=lambda word: hashlib.blake2b(
                word.encode("utf-8"), digest_size=8
            ).digest(),
        )
    else:
        raise ValueError(f"Unknown word selection mode: {mode}")

    corpus_unique_words = len(words)
    corpus_weighted_words = sum(counts.values())
    if max_unique_words > 0:
        words = words[:max_unique_words]

    return words, {
        "mode": mode,
        "max_unique_words": max_unique_words,
        "processed_unique_words": len(words),
        "processed_weighted_words": sum(counts[word] for word in words),
        "corpus_unique_words": corpus_unique_words,
        "corpus_weighted_words": corpus_weighted_words,
    }


def fallback_reason_category(reason: str) -> str:
    if reason.startswith("analysis_output_overflow"):
        return reason
    if reason == "no_analysis":
        return reason
    if "timed out" in reason:
        return "timeout"
    if "Pieces do not reconstruct" in reason:
        return "surface_reconstruction_mismatch"
    if "Could not parse surface pieces" in reason:
        return "parse_failed"
    if "No pieces parsed" in reason:
        return "parse_failed"
    return reason.split(":", 1)[0] or "unknown"


def benchmark_backend(
    backend: str,
    words: list[str],
    counts: Counter[str],
    *,
    batch_size: int,
    strict: bool,
    timeout: float,
) -> dict:
    segmenter = create_segmenter(backend, strict=strict, timeout=timeout)
    t0 = time.time()

    weighted_words = sum(counts[word] for word in words)
    weighted_pieces = 0
    weighted_split_words = 0
    weighted_fallbacks = 0
    type_pieces = 0
    type_split_words = 0
    type_fallbacks = 0
    examples = []
    fallback_examples = []
    fallback_reasons: Counter[str] = Counter()
    fallback_reasons_weighted: Counter[str] = Counter()

    for i in range(0, len(words), batch_size):
        if i > 0 and i % (batch_size * 25) == 0:
            print(
                f"... {backend}: segmented {i:,}/{len(words):,} unique words",
                flush=True,
            )
        batch_words = words[i:i + batch_size]
        batch_segmentations = segmenter.segment_words(batch_words)
        if len(batch_segmentations) != len(batch_words):
            raise SegmentationError(
                f"{backend} returned {len(batch_segmentations)} segmentations "
                f"for {len(batch_words)} words"
            )

        for segmentation in batch_segmentations:
            freq = counts[segmentation.word]
            num_pieces = len(segmentation.pieces)
            weighted_pieces += freq * num_pieces
            type_pieces += num_pieces
            if num_pieces > 1:
                weighted_split_words += freq
                type_split_words += 1
                if len(examples) < 25:
                    examples.append({
                        "word": segmentation.word,
                        "pieces": list(segmentation.pieces),
                        "freq": freq,
                        "source": segmentation.source,
                    })
            if segmentation.fallback:
                weighted_fallbacks += freq
                type_fallbacks += 1
                reason = segmentation.metadata.get("fallback_reason", "unknown")
                reason_category = fallback_reason_category(reason)
                fallback_reasons[reason_category] += 1
                fallback_reasons_weighted[reason_category] += freq
                if len(fallback_examples) < 25:
                    fallback_examples.append({
                        "word": segmentation.word,
                        "pieces": list(segmentation.pieces),
                        "freq": freq,
                        "source": segmentation.source,
                        "reason": reason,
                        "reason_category": reason_category,
                    })

    elapsed = time.time() - t0

    return {
        "backend": backend,
        "status": "ok",
        "elapsed_seconds": elapsed,
        "unique_words": len(words),
        "weighted_words": weighted_words,
        "unique_words_per_sec": len(words) / elapsed if elapsed > 0 else 0.0,
        "weighted_words_per_sec": weighted_words / elapsed if elapsed > 0 else 0.0,
        "pieces_per_word_weighted": weighted_pieces / max(weighted_words, 1),
        "pieces_per_word_types": type_pieces / max(len(words), 1),
        "split_word_rate_weighted": weighted_split_words / max(weighted_words, 1),
        "split_word_rate_types": type_split_words / max(len(words), 1),
        "fallback_rate_weighted": weighted_fallbacks / max(weighted_words, 1),
        "fallback_rate_types": type_fallbacks / max(len(words), 1),
        "examples": examples,
        "fallback_examples": fallback_examples,
        "fallback_reasons_types": dict(fallback_reasons.most_common(25)),
        "fallback_reasons_weighted": dict(
            fallback_reasons_weighted.most_common(25)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Turkish morphology segmenters")
    parser.add_argument("--backend", action="append", default=[], help="Backend to run. Repeatable.")
    parser.add_argument("--data-dir", type=str, default="", help="Directory containing parquet shards")
    parser.add_argument("--text-column", type=str, default=os.environ.get("NANOCHAT_TEXT_COLUMN", "text"))
    parser.add_argument("--max-words", type=int, default=1_000_000)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=0, help="Use only the first N parquet files after manifest ordering.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--progress-every-docs", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--max-unique-words",
        type=int,
        default=0,
        help="Optional cap on unique word types segmented after collection/cache.",
    )
    parser.add_argument(
        "--word-selection",
        type=str,
        default="lexical",
        choices=["lexical", "frequency", "hash"],
        help="Deterministic word-type order before applying --max-unique-words.",
    )
    parser.add_argument(
        "--word-counts-cache",
        type=str,
        default="",
        help="Optional pickle cache for collected word counts.",
    )
    parser.add_argument(
        "--refresh-word-counts-cache",
        action="store_true",
        help="Rebuild --word-counts-cache even if it already exists.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect/cache word counts and skip segmentation backends.",
    )
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    backends = args.backend or ["identity", "trmorph", "zemberek", "tdelight"]

    parquet_paths = list_parquet_files(args.data_dir or None)
    if not parquet_paths:
        raise RuntimeError(
            "No parquet files found. Download a small sample first, for example:\n"
            "  NANOCHAT_BASE_DIR=/private/tmp/nanochat-turk-morph-smoke "
            "python -m nanochat.dataset -n 1 -w 1"
        )
    if args.max_files > 0:
        parquet_paths = parquet_paths[:args.max_files]

    metadata = cache_metadata(
        parquet_paths=parquet_paths,
        text_column=args.text_column,
        max_words=args.max_words,
        max_docs=args.max_docs,
    )
    if (
        args.word_counts_cache
        and os.path.exists(args.word_counts_cache)
        and not args.refresh_word_counts_cache
    ):
        print(f"Loading word-count cache from {args.word_counts_cache}", flush=True)
        counts, sample = load_word_counts_cache(
            args.word_counts_cache,
            expected_metadata=metadata,
        )
    else:
        counts, sample = collect_words(
            parquet_paths=parquet_paths,
            text_column=args.text_column,
            max_words=args.max_words,
            max_docs=args.max_docs,
            progress_every_docs=args.progress_every_docs,
        )
        if args.word_counts_cache:
            save_word_counts_cache(
                args.word_counts_cache,
                counts=counts,
                sample=sample,
                metadata=metadata,
            )
            print(f"Wrote word-count cache to {args.word_counts_cache}", flush=True)
    words, selection = select_words(
        counts,
        mode=args.word_selection,
        max_unique_words=args.max_unique_words,
    )

    results = {
        "sample": {
            **sample,
            "unique_words": selection["corpus_unique_words"],
            "parquet_files": parquet_paths,
        },
        "word_selection": selection,
        "backends": [],
    }

    if args.collect_only:
        print(json.dumps(results["sample"], ensure_ascii=False, indent=2))
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Wrote benchmark results to {args.output}")
        return

    for backend in backends:
        try:
            result = benchmark_backend(
                backend,
                words,
                counts,
                batch_size=args.batch_size,
                strict=args.strict,
                timeout=args.timeout,
            )
        except (SegmenterUnavailable, SegmentationError, RuntimeError, ImportError) as exc:
            result = {
                "backend": backend,
                "status": "unavailable",
                "error": str(exc),
            }
        results["backends"].append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote benchmark results to {args.output}")


if __name__ == "__main__":
    main()
