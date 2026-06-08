"""Materialize full segmented corpus shards for morphology-aware tokenizers.

Large outputs should be written under dev-ignore/, for example:

    TRMORPH_SEGMENT_FST=/private/tmp/TRmorph/segment.fst \
    TRMORPH_FLOOKUP_FLAGS=-x \
    NANOCHAT_BASE_DIR=/Users/nurcunal/Documents/nanochat-turk/dev-ignore/morph-smoke \
    python3 -m scripts.morph_segment_corpus \
      --backend trmorph \
      --max-files 1 \
      --output-dir dev-ignore/morph-smoke/segmented/trmorph
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.dataset import list_parquet_files
from nanochat.morphology import (
    SegmenterUnavailable,
    SegmentationError,
    WordSegmentation,
    create_segmenter,
    iter_word_spans,
)


class LRUWordCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._data: OrderedDict[str, WordSegmentation] = OrderedDict()

    def get(self, word: str) -> WordSegmentation | None:
        value = self._data.get(word)
        if value is not None:
            self._data.move_to_end(word)
        return value

    def set(self, word: str, segmentation: WordSegmentation) -> None:
        if self.max_size <= 0:
            return
        self._data[word] = segmentation
        self._data.move_to_end(word)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_segmentations(
    texts: list[str],
    *,
    segmenter,
    cache: LRUWordCache,
    segment_batch_size: int,
) -> dict[str, WordSegmentation]:
    lookup: dict[str, WordSegmentation] = {}
    missing: list[str] = []
    seen_missing = set()

    for text in texts:
        for _start, _end, word in iter_word_spans(text):
            cached = cache.get(word)
            if cached is not None:
                lookup[word] = cached
            elif word not in seen_missing:
                seen_missing.add(word)
                missing.append(word)

    for i in range(0, len(missing), segment_batch_size):
        batch = missing[i:i + segment_batch_size]
        segmentations = segmenter.segment_words(batch)
        if len(segmentations) != len(batch):
            raise SegmentationError(
                f"Segmenter returned {len(segmentations)} rows for {len(batch)} words"
            )
        for segmentation in segmentations:
            cache.set(segmentation.word, segmentation)
            lookup[segmentation.word] = segmentation

    return lookup


def segment_text_from_lookup(
    text: str,
    lookup: dict[str, WordSegmentation],
    *,
    delimiter: str,
) -> tuple[str, dict[str, int]]:
    chunks: list[str] = []
    cursor = 0
    word_count = 0
    split_words = 0
    fallback_words = 0

    for start, end, word in iter_word_spans(text):
        segmentation = lookup[word]
        chunks.append(text[cursor:start])
        chunks.append(segmentation.delimited(delimiter))
        cursor = end
        word_count += 1
        if len(segmentation.pieces) > 1:
            split_words += 1
        if segmentation.fallback:
            fallback_words += 1

    chunks.append(text[cursor:])
    return "".join(chunks), {
        "word_count": word_count,
        "split_words": split_words,
        "fallback_words": fallback_words,
    }


def output_path_for(input_path: str, output_dir: str) -> str:
    stem = Path(input_path).stem
    return str(Path(output_dir) / f"{stem}.segmented.parquet")


def write_segmented_shard(
    input_path: str,
    output_path: str,
    *,
    text_column: str,
    id_column: str,
    include_original: bool,
    backend: str,
    segmenter,
    cache: LRUWordCache,
    row_group_batch_size: int,
    segment_batch_size: int,
    delimiter: str,
    max_docs: int,
) -> dict[str, Any]:
    pf = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    total_docs = 0
    total_words = 0
    total_split_words = 0
    total_fallback_words = 0
    started = time.time()

    schema_fields = [
        pa.field("source_path", pa.string()),
        pa.field("source_row", pa.int64()),
        pa.field("id", pa.string()),
    ]
    if include_original:
        schema_fields.append(pa.field("text", pa.string()))
    schema_fields.extend([
        pa.field("segmented_text", pa.string()),
        pa.field("segmenter", pa.string()),
        pa.field("word_count", pa.int64()),
        pa.field("split_words", pa.int64()),
        pa.field("fallback_words", pa.int64()),
    ])
    schema = pa.schema(schema_fields)

    try:
        source_row_offset = 0
        for rg_idx in range(pf.num_row_groups):
            columns = [text_column]
            if id_column and id_column in pf.schema_arrow.names:
                columns.append(id_column)
            table = pf.read_row_group(rg_idx, columns=columns)
            texts = [
                "" if value is None else str(value)
                for value in table.column(text_column).to_pylist()
            ]
            ids = (
                ["" if value is None else str(value) for value in table.column(id_column).to_pylist()]
                if id_column and id_column in table.column_names
                else [""] * len(texts)
            )

            for start in range(0, len(texts), row_group_batch_size):
                if max_docs > 0 and total_docs >= max_docs:
                    break
                batch_texts = texts[start:start + row_group_batch_size]
                batch_ids = ids[start:start + row_group_batch_size]
                if max_docs > 0:
                    remaining = max_docs - total_docs
                    batch_texts = batch_texts[:remaining]
                    batch_ids = batch_ids[:remaining]

                lookup = ensure_segmentations(
                    batch_texts,
                    segmenter=segmenter,
                    cache=cache,
                    segment_batch_size=segment_batch_size,
                )

                rows: dict[str, list[Any]] = {
                    "source_path": [],
                    "source_row": [],
                    "id": [],
                    "segmented_text": [],
                    "segmenter": [],
                    "word_count": [],
                    "split_words": [],
                    "fallback_words": [],
                }
                if include_original:
                    rows["text"] = []

                for local_idx, (doc_id, text) in enumerate(zip(batch_ids, batch_texts)):
                    segmented_text, stats = segment_text_from_lookup(
                        text,
                        lookup,
                        delimiter=delimiter,
                    )
                    rows["source_path"].append(input_path)
                    rows["source_row"].append(source_row_offset + start + local_idx)
                    rows["id"].append(doc_id)
                    if include_original:
                        rows["text"].append(text)
                    rows["segmented_text"].append(segmented_text)
                    rows["segmenter"].append(backend)
                    rows["word_count"].append(stats["word_count"])
                    rows["split_words"].append(stats["split_words"])
                    rows["fallback_words"].append(stats["fallback_words"])

                    total_docs += 1
                    total_words += stats["word_count"]
                    total_split_words += stats["split_words"]
                    total_fallback_words += stats["fallback_words"]

                out_table = pa.table(rows, schema=schema)
                if writer is None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
                writer.write_table(out_table)

            source_row_offset += table.num_rows
            if max_docs > 0 and total_docs >= max_docs:
                break
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - started
    return {
        "input_path": input_path,
        "output_path": output_path,
        "docs": total_docs,
        "word_count": total_words,
        "split_words": total_split_words,
        "fallback_words": total_fallback_words,
        "elapsed_seconds": elapsed,
        "docs_per_sec": total_docs / elapsed if elapsed > 0 else 0.0,
        "words_per_sec": total_words / elapsed if elapsed > 0 else 0.0,
        "sha256": file_sha256(output_path) if os.path.exists(output_path) else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write segmented corpus parquet shards")
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--command", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--text-column", type=str, default=os.environ.get("NANOCHAT_TEXT_COLUMN", "text"))
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--row-group-batch-size", type=int, default=256)
    parser.add_argument("--segment-batch-size", type=int, default=2048)
    parser.add_argument("--word-cache-size", type=int, default=200000)
    parser.add_argument("--delimiter", type=str, default=" ")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Omit original text column from output.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    parquet_paths = list_parquet_files(args.data_dir or None)
    if args.max_files > 0:
        parquet_paths = parquet_paths[:args.max_files]
    if not parquet_paths:
        raise RuntimeError("No parquet shards found")

    segmenter = create_segmenter(
        args.backend,
        command=args.command or None,
        strict=args.strict,
        timeout=args.timeout,
    )
    cache = LRUWordCache(args.word_cache_size)
    started = time.time()
    outputs = []

    try:
        for input_path in parquet_paths:
            output_path = output_path_for(input_path, args.output_dir)
            if os.path.exists(output_path) and not args.overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use --overwrite to replace it."
                )
            print(f"Segmenting {input_path} -> {output_path}", flush=True)
            outputs.append(
                write_segmented_shard(
                    input_path,
                    output_path,
                    text_column=args.text_column,
                    id_column=args.id_column,
                    include_original=not args.compact,
                    backend=args.backend,
                    segmenter=segmenter,
                    cache=cache,
                    row_group_batch_size=args.row_group_batch_size,
                    segment_batch_size=args.segment_batch_size,
                    delimiter=args.delimiter,
                    max_docs=args.max_docs,
                )
            )
    except (SegmenterUnavailable, SegmentationError):
        raise

    manifest = {
        "backend": args.backend,
        "command": args.command,
        "output_dir": args.output_dir,
        "input_files": parquet_paths,
        "outputs": outputs,
        "text_column": args.text_column,
        "id_column": args.id_column,
        "include_original": not args.compact,
        "delimiter": args.delimiter,
        "row_group_batch_size": args.row_group_batch_size,
        "segment_batch_size": args.segment_batch_size,
        "word_cache_size": args.word_cache_size,
        "max_docs": args.max_docs,
        "git_commit": git_commit(),
        "elapsed_seconds": time.time() - started,
        "cache_entries_at_end": len(cache),
        "environment": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith(("TRMORPH_", "ZEMBEREK_", "TDELIGHT_"))
        },
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
