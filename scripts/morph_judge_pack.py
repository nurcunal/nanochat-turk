"""Create equal-sample segmenter metrics and blind LLM-judge packs.

This script runs the same deterministic word-type sample through multiple
segmenters, records aggregate metrics, and exports a small blind JSONL pack for
LLM or human judging.

Example:
    TRMORPH_SEGMENT_FST=/private/tmp/TRmorph/segment.fst \
    TRMORPH_FLOOKUP_FLAGS=-x \
    ZEMBEREK_SEGMENT_CMD="/private/tmp/zemberek-smoke-py312/bin/python scripts/zemberek_segment_cmd.py" \
    NANOCHAT_BASE_DIR=/Users/nurcunal/Documents/nanochat-turk/dev-ignore/morph-smoke \
    python3 -m scripts.morph_judge_pack \
      --backend trmorph \
      --backend zemberek \
      --backend tdelight \
      --include-identity \
      --max-unique-words 100000 \
      --judge-size 500 \
      --word-selection hash \
      --word-counts-cache dev-ignore/morph-smoke/benchmarks/word_counts_000_00000.pkl \
      --output-dir dev-ignore/morph-smoke/judge_packs/hash_100k_judge_500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from nanochat.dataset import list_parquet_files
from nanochat.morphology import (
    SegmenterUnavailable,
    SegmentationError,
    WordSegmentation,
    create_segmenter,
    iter_word_spans,
)
from scripts.morph_benchmark import (
    cache_metadata,
    collect_words,
    fallback_reason_category,
    load_word_counts_cache,
    save_word_counts_cache,
    select_words,
)


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def stable_int(text: str) -> int:
    import hashlib

    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def parse_backend_batch_sizes(values: list[str]) -> dict[str, int]:
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected BACKEND=N for --backend-batch-size, got {value!r}")
        backend, size = value.split("=", 1)
        out[backend.strip().lower()] = int(size)
    return out


def segment_backend(
    backend: str,
    words: list[str],
    *,
    batch_size: int,
    strict: bool,
    timeout: float,
) -> tuple[list[WordSegmentation], dict[str, Any]]:
    segmenter = create_segmenter(backend, strict=strict, timeout=timeout)
    started = time.time()
    out: list[WordSegmentation] = []
    for i in range(0, len(words), batch_size):
        if i > 0 and i % max(batch_size * 10, 1) == 0:
            print(
                f"... {backend}: segmented {i:,}/{len(words):,} judge-pack words",
                flush=True,
            )
        batch = words[i:i + batch_size]
        segmentations = segmenter.segment_words(batch)
        if len(segmentations) != len(batch):
            raise SegmentationError(
                f"{backend} returned {len(segmentations)} rows for {len(batch)} words"
            )
        out.extend(segmentations)
    elapsed = time.time() - started
    return out, {
        "backend": backend,
        "status": "ok",
        "elapsed_seconds": elapsed,
        "unique_words": len(words),
        "unique_words_per_sec": len(words) / elapsed if elapsed > 0 else 0.0,
    }


def summarize_segmentations(
    backend: str,
    segmentations: list[WordSegmentation],
    counts: Counter[str],
) -> dict[str, Any]:
    weighted_words = sum(counts[seg.word] for seg in segmentations)
    weighted_pieces = 0
    weighted_split_words = 0
    weighted_fallbacks = 0
    type_pieces = 0
    type_split_words = 0
    type_fallbacks = 0
    fallback_reasons: Counter[str] = Counter()
    fallback_reasons_weighted: Counter[str] = Counter()

    for segmentation in segmentations:
        freq = counts[segmentation.word]
        num_pieces = len(segmentation.pieces)
        weighted_pieces += freq * num_pieces
        type_pieces += num_pieces
        if num_pieces > 1:
            weighted_split_words += freq
            type_split_words += 1
        if segmentation.fallback:
            weighted_fallbacks += freq
            type_fallbacks += 1
            reason = segmentation.metadata.get("fallback_reason", "unknown")
            reason_category = fallback_reason_category(reason)
            fallback_reasons[reason_category] += 1
            fallback_reasons_weighted[reason_category] += freq

    return {
        "backend": backend,
        "status": "ok",
        "unique_words": len(segmentations),
        "weighted_words": weighted_words,
        "pieces_per_word_weighted": weighted_pieces / max(weighted_words, 1),
        "pieces_per_word_types": type_pieces / max(len(segmentations), 1),
        "split_word_rate_weighted": weighted_split_words / max(weighted_words, 1),
        "split_word_rate_types": type_split_words / max(len(segmentations), 1),
        "fallback_rate_weighted": weighted_fallbacks / max(weighted_words, 1),
        "fallback_rate_types": type_fallbacks / max(len(segmentations), 1),
        "fallback_reasons_types": dict(fallback_reasons.most_common(25)),
        "fallback_reasons_weighted": dict(fallback_reasons_weighted.most_common(25)),
    }


def disagreement_score(
    idx: int,
    words: list[str],
    counts: Counter[str],
    backend_segmentations: dict[str, list[WordSegmentation]],
) -> tuple[float, int]:
    segmentations = [
        rows[idx]
        for rows in backend_segmentations.values()
        if idx < len(rows)
    ]
    outputs = {seg.delimited(" ") for seg in segmentations}
    non_identity_outputs = {
        seg.delimited(" ")
        for seg in segmentations
        if len(seg.pieces) > 1 and not seg.fallback
    }
    has_split = any(len(seg.pieces) > 1 for seg in segmentations)
    has_fallback = any(seg.fallback for seg in segmentations)
    has_nonfallback = any(not seg.fallback for seg in segmentations)

    score = 0.0
    if len(outputs) > 1:
        score += 5.0
    if len(non_identity_outputs) > 1:
        score += 5.0
    if has_split:
        score += 2.0
    if has_fallback and has_nonfallback:
        score += 2.0
    score += min(math.log1p(counts[words[idx]]), 8.0) / 4.0
    return score, stable_int(words[idx])


def choose_judge_indices(
    words: list[str],
    counts: Counter[str],
    backend_segmentations: dict[str, list[WordSegmentation]],
    *,
    judge_size: int,
    strategy: str,
) -> list[int]:
    indices = list(range(len(words)))
    if judge_size <= 0 or judge_size >= len(indices):
        return indices

    if strategy == "hash":
        ordered = sorted(indices, key=lambda idx: stable_int(words[idx]))
    elif strategy == "frequency":
        ordered = sorted(indices, key=lambda idx: (-counts[words[idx]], words[idx]))
    elif strategy == "disagreement":
        ordered = sorted(
            indices,
            key=lambda idx: (
                -disagreement_score(idx, words, counts, backend_segmentations)[0],
                disagreement_score(idx, words, counts, backend_segmentations)[1],
            ),
        )
    else:
        raise ValueError(f"Unknown judge strategy: {strategy}")
    return ordered[:judge_size]


def find_contexts(
    parquet_paths: list[str],
    wanted_words: set[str],
    *,
    text_column: str,
    max_docs: int,
    context_chars: int,
) -> dict[str, str]:
    import pyarrow.parquet as pq

    contexts: dict[str, str] = {}
    docs_seen = 0
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=[text_column])
            for value in table.column(text_column).to_pylist():
                docs_seen += 1
                text = "" if value is None else str(value)
                for start, end, word in iter_word_spans(text):
                    if word in wanted_words and word not in contexts:
                        left = max(0, start - context_chars)
                        right = min(len(text), end + context_chars)
                        contexts[word] = text[left:right].replace("\n", " ")
                        if len(contexts) >= len(wanted_words):
                            return contexts
                if max_docs > 0 and docs_seen >= max_docs:
                    return contexts
    return contexts


def blind_candidates_for_word(
    word: str,
    idx: int,
    backend_segmentations: dict[str, list[WordSegmentation]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = []
    for backend, segmentations in backend_segmentations.items():
        seg = segmentations[idx]
        rows.append((stable_int(f"{word}:{backend}"), backend, seg))
    rows.sort(key=lambda item: item[0])

    candidates = []
    answer_key = {}
    for label, (_order, backend, seg) in zip(LABELS, rows):
        candidates.append({
            "label": label,
            "segmentation": seg.delimited(" "),
            "pieces": list(seg.pieces),
            "is_identity": len(seg.pieces) == 1,
        })
        answer_key[label] = backend
    return candidates, answer_key


def write_judge_prompt(path: Path, *, jsonl_name: str) -> None:
    path.write_text(
        f"""# Turkish Morphological Segmentation Judge Prompt

You are judging candidate morphological segmentations for Turkish words.

Input file: `{jsonl_name}`. Each JSONL row has:

- `id`: item id.
- `word`: original surface word.
- `frequency`: count in the sampled FineWeb-2 Turkish shard.
- `context`: optional corpus snippet.
- `candidates`: blind candidate segmentations labeled `A`, `B`, `C`, etc.

Judge criteria:

1. The best candidate must preserve the exact original surface word when its
   pieces are concatenated.
2. Prefer linguistically plausible Turkish morpheme boundaries.
3. Avoid over-segmentation when a split is not morphologically meaningful.
4. For proper names, foreign words, OCR/noisy tokens, and all-caps strings,
   identity/no-split can be the best answer.
5. Ignore candidate labels and do not infer which tool produced which candidate.

Return JSONL with one row per input item:

```json
{{"id":"morphjudge_000001","best_label":"A","acceptable_labels":["A","C"],"confidence":"medium","notes":"short reason"}}
```
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create blind LLM judge packs for Turkish segmenters")
    parser.add_argument("--backend", action="append", default=[], help="Segmenter backend. Repeatable.")
    parser.add_argument("--include-identity", action="store_true", help="Include identity/no-split baseline as a candidate.")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--text-column", type=str, default=os.environ.get("NANOCHAT_TEXT_COLUMN", "text"))
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=0)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--word-counts-cache", type=str, default="")
    parser.add_argument("--refresh-word-counts-cache", action="store_true")
    parser.add_argument("--max-unique-words", type=int, default=100000)
    parser.add_argument("--word-selection", choices=["hash", "frequency", "lexical"], default="hash")
    parser.add_argument("--judge-size", type=int, default=500)
    parser.add_argument("--judge-strategy", choices=["disagreement", "hash", "frequency"], default="disagreement")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--backend-batch-size", action="append", default=[], help="Override batch size as BACKEND=N.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--include-context", action="store_true")
    parser.add_argument("--context-max-docs", type=int, default=100000)
    parser.add_argument("--context-chars", type=int, default=160)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="segmenter_judge_pack")
    args = parser.parse_args()

    backends = list(args.backend)
    if args.include_identity and "identity" not in {b.lower() for b in backends}:
        backends.insert(0, "identity")
    if not backends:
        backends = ["identity", "trmorph", "zemberek", "tdelight"]

    parquet_paths = list_parquet_files(args.data_dir or None)
    if args.max_files > 0:
        parquet_paths = parquet_paths[:args.max_files]
    if not parquet_paths:
        raise RuntimeError("No parquet files found for judge pack")

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
        counts, sample = load_word_counts_cache(args.word_counts_cache, expected_metadata=metadata)
    else:
        counts, sample = collect_words(
            parquet_paths=parquet_paths,
            text_column=args.text_column,
            max_words=args.max_words,
            max_docs=args.max_docs,
            progress_every_docs=100000,
        )
        if args.word_counts_cache:
            save_word_counts_cache(
                args.word_counts_cache,
                counts=counts,
                sample=sample,
                metadata=metadata,
            )

    words, selection = select_words(
        counts,
        mode=args.word_selection,
        max_unique_words=args.max_unique_words,
    )

    batch_sizes = parse_backend_batch_sizes(args.backend_batch_size)
    backend_segmentations: dict[str, list[WordSegmentation]] = {}
    backend_metrics = []
    unavailable = []

    for backend in backends:
        normalized = backend.lower().replace("-", "_")
        batch_size = batch_sizes.get(normalized, args.batch_size)
        try:
            segmentations, timing = segment_backend(
                backend,
                words,
                batch_size=batch_size,
                strict=args.strict,
                timeout=args.timeout,
            )
            summary = summarize_segmentations(backend, segmentations, counts)
            summary.update({
                "elapsed_seconds": timing["elapsed_seconds"],
                "unique_words_per_sec": timing["unique_words_per_sec"],
                "batch_size": batch_size,
            })
            backend_segmentations[backend] = segmentations
            backend_metrics.append(summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        except (SegmenterUnavailable, SegmentationError, RuntimeError, ImportError) as exc:
            result = {
                "backend": backend,
                "status": "unavailable",
                "error": str(exc),
                "batch_size": batch_size,
            }
            backend_metrics.append(result)
            unavailable.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    if not backend_segmentations:
        raise RuntimeError("No segmenter produced judge-pack candidates")

    judge_indices = choose_judge_indices(
        words,
        counts,
        backend_segmentations,
        judge_size=args.judge_size,
        strategy=args.judge_strategy,
    )
    context_by_word = {}
    if args.include_context:
        context_by_word = find_contexts(
            parquet_paths,
            {words[idx] for idx in judge_indices},
            text_column=args.text_column,
            max_docs=args.context_max_docs,
            context_chars=args.context_chars,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{args.prefix}.jsonl"
    answer_key_path = output_dir / f"{args.prefix}.answer_key.json"
    metrics_path = output_dir / f"{args.prefix}.metrics.json"
    prompt_path = output_dir / f"{args.prefix}.prompt.md"

    answer_key = {}
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for out_idx, idx in enumerate(judge_indices, start=1):
            word = words[idx]
            item_id = f"morphjudge_{out_idx:06d}"
            candidates, key = blind_candidates_for_word(word, idx, backend_segmentations)
            answer_key[item_id] = key
            f.write(json.dumps({
                "id": item_id,
                "word": word,
                "frequency": counts[word],
                "context": context_by_word.get(word, ""),
                "candidates": candidates,
            }, ensure_ascii=False) + "\n")

    with open(answer_key_path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, ensure_ascii=False, indent=2)

    metrics = {
        "sample": {
            **sample,
            "unique_words": selection["corpus_unique_words"],
            "parquet_files": parquet_paths,
        },
        "word_selection": selection,
        "judge_selection": {
            "strategy": args.judge_strategy,
            "judge_size": len(judge_indices),
            "include_context": args.include_context,
            "context_found": len(context_by_word),
        },
        "backends": backend_metrics,
        "unavailable": unavailable,
        "outputs": {
            "judge_jsonl": str(jsonl_path),
            "answer_key": str(answer_key_path),
            "metrics": str(metrics_path),
            "prompt": str(prompt_path),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_judge_prompt(prompt_path, jsonl_name=jsonl_path.name)

    print(f"Wrote judge JSONL to {jsonl_path}")
    print(f"Wrote answer key to {answer_key_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote judge prompt to {prompt_path}")


if __name__ == "__main__":
    main()
