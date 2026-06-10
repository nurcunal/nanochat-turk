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
from collections.abc import Iterator
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
import threading
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.dataset import MANIFEST_FILE, list_parquet_files
from nanochat.morphology import (
    MORPHEME_BOUNDARY,
    SegmenterUnavailable,
    SegmentationError,
    WordSegmentation,
    create_segmenter,
    display_boundary,
    iter_word_spans,
)


_WORKER_SEGMENTER: Any | None = None
_WORKER_CACHE: LRUWordCache | None = None
_WORKER_LOCAL = threading.local()


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


def build_segmented_rows(
    *,
    input_path: str,
    source_row_start: int,
    ids: list[str],
    texts: list[str],
    include_original: bool,
    backend: str,
    segmenter,
    cache: LRUWordCache,
    segment_batch_size: int,
    delimiter: str,
) -> dict[str, Any]:
    lookup = ensure_segmentations(
        texts,
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

    total_words = 0
    total_split_words = 0
    total_fallback_words = 0

    for local_idx, (doc_id, text) in enumerate(zip(ids, texts)):
        segmented_text, stats = segment_text_from_lookup(
            text,
            lookup,
            delimiter=delimiter,
        )
        rows["source_path"].append(input_path)
        rows["source_row"].append(source_row_start + local_idx)
        rows["id"].append(doc_id)
        if include_original:
            rows["text"].append(text)
        rows["segmented_text"].append(segmented_text)
        rows["segmenter"].append(backend)
        rows["word_count"].append(stats["word_count"])
        rows["split_words"].append(stats["split_words"])
        rows["fallback_words"].append(stats["fallback_words"])

        total_words += stats["word_count"]
        total_split_words += stats["split_words"]
        total_fallback_words += stats["fallback_words"]

    return {
        "rows": rows,
        "docs": len(texts),
        "word_count": total_words,
        "split_words": total_split_words,
        "fallback_words": total_fallback_words,
        "cache_entries": len(cache),
    }


def init_segment_worker(
    backend: str,
    command: str,
    strict: bool,
    timeout: float,
    worker_cache_size: int,
) -> None:
    global _WORKER_SEGMENTER, _WORKER_CACHE
    _WORKER_SEGMENTER = create_segmenter(
        backend,
        command=command or None,
        strict=strict,
        timeout=timeout,
    )
    _WORKER_CACHE = LRUWordCache(worker_cache_size)
    _WORKER_LOCAL.segmenter = _WORKER_SEGMENTER
    _WORKER_LOCAL.cache = _WORKER_CACHE


def segment_batch_worker(task: dict[str, Any]) -> dict[str, Any]:
    segmenter = getattr(_WORKER_LOCAL, "segmenter", _WORKER_SEGMENTER)
    cache = getattr(_WORKER_LOCAL, "cache", _WORKER_CACHE)
    if segmenter is None or cache is None:
        raise RuntimeError("Segmentation worker was not initialized")
    result = build_segmented_rows(
        input_path=task["input_path"],
        source_row_start=task["source_row_start"],
        ids=task["ids"],
        texts=task["texts"],
        include_original=task["include_original"],
        backend=task["backend"],
        segmenter=segmenter,
        cache=cache,
        segment_batch_size=task["segment_batch_size"],
        delimiter=task["delimiter"],
    )
    result["batch_index"] = task["batch_index"]
    return result


def output_path_for(input_path: str, output_dir: str) -> str:
    stem = Path(input_path).stem
    return str(Path(output_dir) / f"{stem}.segmented.parquet")


def shard_done_path(output_path: str) -> str:
    return f"{output_path}.done.json"


def shard_tmp_path(output_path: str) -> str:
    return f"{output_path}.tmp.{os.getpid()}"


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def parquet_is_readable(path: str) -> bool:
    try:
        pq.ParquetFile(path)
    except Exception:
        return False
    return True


def load_completed_shard(output_path: str) -> dict[str, Any] | None:
    done_path = shard_done_path(output_path)
    if not os.path.isfile(output_path) or not os.path.isfile(done_path):
        return None
    try:
        with open(done_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if payload.get("output_path") != output_path:
        return None
    if payload.get("sha256") != file_sha256(output_path):
        return None
    if not parquet_is_readable(output_path):
        return None
    return payload


def remove_incomplete_shard(output_path: str) -> None:
    for path in Path(output_path).parent.glob(f"{Path(output_path).name}.tmp.*"):
        if path.is_file():
            path.unlink()
    for path in (output_path, shard_done_path(output_path)):
        if os.path.exists(path):
            os.remove(path)


def write_progress_manifests(
    output_dir: str,
    *,
    backend: str,
    command: str,
    parquet_paths: list[str],
    outputs: list[dict[str, Any]],
    text_column: str,
    id_column: str,
    include_original: bool,
    delimiter: str,
    row_group_batch_size: int,
    segment_batch_size: int,
    word_cache_size: int,
    num_workers: int,
    worker_mode: str,
    max_docs: int,
    started: float,
    cache_entries: int,
    complete: bool,
) -> dict[str, Any]:
    manifest = {
        "backend": backend,
        "command": command,
        "output_dir": output_dir,
        "input_files": parquet_paths,
        "outputs": outputs,
        "completed_files": [Path(output["output_path"]).name for output in outputs],
        "expected_files": [Path(path).name for path in parquet_paths],
        "complete": complete,
        "text_column": text_column,
        "id_column": id_column,
        "include_original": include_original,
        "delimiter": delimiter,
        "delimiter_codepoints": display_boundary(delimiter),
        "delimiter_semantics": (
            "internal_morpheme_boundary"
            if delimiter == MORPHEME_BOUNDARY
            else "custom"
        ),
        "row_group_batch_size": row_group_batch_size,
        "segment_batch_size": segment_batch_size,
        "word_cache_size": word_cache_size,
        "num_workers": num_workers,
        "worker_mode": worker_mode,
        "completed_shard_num_workers": sorted(
            {
                int(output["num_workers"])
                for output in outputs
                if str(output.get("num_workers", "")).isdigit()
            }
        ),
        "max_docs": max_docs,
        "git_commit": git_commit(),
        "elapsed_seconds": time.time() - started,
        "cache_entries_at_end": cache_entries,
        "environment": {
            key: os.environ[key]
            for key in sorted(os.environ)
            if key.startswith(("TRMORPH_", "ZEMBEREK_", "TDELIGHT_"))
        },
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_name = "manifest.json" if complete else "manifest.partial.json"
    write_json_atomic(Path(output_dir) / manifest_name, manifest)

    dataset_manifest = {
        "source": "morph_segment_corpus",
        "segmentation_manifest": str(Path(output_dir) / manifest_name),
        "backend": backend,
        "text_column": "segmented_text",
        "filenames": [Path(output["output_path"]).name for output in outputs],
        "complete": complete,
    }
    dataset_manifest_name = MANIFEST_FILE if complete else f"{MANIFEST_FILE}.partial"
    write_json_atomic(Path(output_dir) / dataset_manifest_name, dataset_manifest)
    return manifest


def iter_segment_tasks(
    input_path: str,
    *,
    text_column: str,
    id_column: str,
    include_original: bool,
    backend: str,
    row_group_batch_size: int,
    segment_batch_size: int,
    delimiter: str,
    max_docs: int,
) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(input_path)
    emitted_docs = 0
    source_row_offset = 0
    batch_index = 0

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
            if max_docs > 0 and emitted_docs >= max_docs:
                break
            batch_texts = texts[start:start + row_group_batch_size]
            batch_ids = ids[start:start + row_group_batch_size]
            if max_docs > 0:
                remaining = max_docs - emitted_docs
                batch_texts = batch_texts[:remaining]
                batch_ids = batch_ids[:remaining]
            if not batch_texts:
                continue

            yield {
                "batch_index": batch_index,
                "input_path": input_path,
                "source_row_start": source_row_offset + start,
                "ids": batch_ids,
                "texts": batch_texts,
                "include_original": include_original,
                "backend": backend,
                "segment_batch_size": segment_batch_size,
                "delimiter": delimiter,
            }
            batch_index += 1
            emitted_docs += len(batch_texts)

        source_row_offset += table.num_rows
        if max_docs > 0 and emitted_docs >= max_docs:
            break


def write_segmented_shard(
    input_path: str,
    output_path: str,
    *,
    text_column: str,
    id_column: str,
    include_original: bool,
    backend: str,
    command: str,
    strict: bool,
    timeout: float,
    segmenter,
    cache: LRUWordCache,
    row_group_batch_size: int,
    segment_batch_size: int,
    delimiter: str,
    max_docs: int,
    num_workers: int,
    worker_mode: str,
    worker_cache_size: int,
    max_in_flight: int,
) -> dict[str, Any]:
    writer: pq.ParquetWriter | None = None
    total_docs = 0
    total_words = 0
    total_split_words = 0
    total_fallback_words = 0
    cache_entries = 0
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

    def write_result(result: dict[str, Any]) -> None:
        nonlocal writer
        out_table = pa.table(result["rows"], schema=schema)
        if writer is None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(output_path, schema, compression="zstd")
        writer.write_table(out_table)

    tasks = iter_segment_tasks(
        input_path,
        text_column=text_column,
        id_column=id_column,
        include_original=include_original,
        backend=backend,
        row_group_batch_size=row_group_batch_size,
        segment_batch_size=segment_batch_size,
        delimiter=delimiter,
        max_docs=max_docs,
    )

    try:
        if num_workers <= 1:
            for task in tasks:
                result = build_segmented_rows(
                    input_path=task["input_path"],
                    source_row_start=task["source_row_start"],
                    ids=task["ids"],
                    texts=task["texts"],
                    include_original=task["include_original"],
                    backend=task["backend"],
                    segmenter=segmenter,
                    cache=cache,
                    segment_batch_size=segment_batch_size,
                    delimiter=delimiter,
                )
                result["batch_index"] = task["batch_index"]
                write_result(result)
                total_docs += result["docs"]
                total_words += result["word_count"]
                total_split_words += result["split_words"]
                total_fallback_words += result["fallback_words"]
                cache_entries = result["cache_entries"]
        else:
            pending_results: dict[int, dict[str, Any]] = {}
            next_to_write = 0
            in_flight = {}
            task_iter = iter(tasks)
            task_exhausted = False

            def submit_more(pool: ProcessPoolExecutor) -> None:
                nonlocal task_exhausted
                while not task_exhausted and len(in_flight) < max_in_flight:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        task_exhausted = True
                        return
                    future = pool.submit(segment_batch_worker, task)
                    in_flight[future] = task["batch_index"]

            executor_cls = (
                ProcessPoolExecutor
                if worker_mode == "process"
                else ThreadPoolExecutor
            )
            with executor_cls(
                max_workers=num_workers,
                initializer=init_segment_worker,
                initargs=(backend, command, strict, timeout, worker_cache_size),
            ) as pool:
                submit_more(pool)
                while in_flight:
                    done, _not_done = wait(
                        in_flight,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        batch_index = in_flight.pop(future)
                        result = future.result()
                        pending_results[batch_index] = result

                    while next_to_write in pending_results:
                        result = pending_results.pop(next_to_write)
                        write_result(result)
                        total_docs += result["docs"]
                        total_words += result["word_count"]
                        total_split_words += result["split_words"]
                        total_fallback_words += result["fallback_words"]
                        cache_entries = max(
                            cache_entries,
                            int(result.get("cache_entries", 0) or 0),
                        )
                        next_to_write += 1

                    submit_more(pool)
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
        "num_workers": num_workers,
        "worker_mode": worker_mode,
        "worker_cache_size": worker_cache_size if num_workers > 1 else cache.max_size,
        "max_in_flight": max_in_flight if num_workers > 1 else 1,
        "cache_entries_at_end": cache_entries,
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
    parser.add_argument("--shard-index", type=int, default=-1, help="Process one 0-based parquet shard index.")
    parser.add_argument("--shard-name", type=str, default="", help="Process one parquet shard by filename.")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Write final manifests from completed shard sidecars without segmenting.",
    )
    parser.add_argument("--row-group-batch-size", type=int, default=256)
    parser.add_argument("--segment-batch-size", type=int, default=2048)
    parser.add_argument("--word-cache-size", type=int, default=200000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SEGMENT_WORKERS", "1")),
        help=(
            "Parallel worker processes per shard. Each worker creates its own "
            "segmenter instance; the parent writes parquet rows in source order."
        ),
    )
    parser.add_argument(
        "--worker-mode",
        choices=("process", "thread"),
        default=os.environ.get("SEGMENT_WORKER_MODE", "process"),
        help=(
            "Parallel executor type. Use process workers for CPU-node runs; "
            "thread workers are useful for local smoke tests or network segmenters."
        ),
    )
    parser.add_argument(
        "--worker-cache-size",
        type=int,
        default=int(os.environ.get("SEGMENT_WORKER_CACHE_SIZE", "0")),
        help=(
            "Per-worker word cache size. Default 0 splits --word-cache-size "
            "across workers."
        ),
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=int(os.environ.get("SEGMENT_MAX_IN_FLIGHT", "0")),
        help="Maximum queued segmentation batches. Default is 2x --num-workers.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=MORPHEME_BOUNDARY,
        help=(
            "Delimiter inserted between surface morphemes. Default is the "
            f"internal MorphBPE marker {display_boundary(MORPHEME_BOUNDARY)}."
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Omit original text column from output.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip shards with a valid .done.json sidecar and rewrite "
            "incomplete/corrupt shard outputs."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    num_workers = max(1, args.num_workers)
    worker_mode = args.worker_mode
    worker_cache_size = (
        args.worker_cache_size
        if args.worker_cache_size > 0
        else max(1000, args.word_cache_size // num_workers)
    )
    max_in_flight = (
        args.max_in_flight
        if args.max_in_flight > 0
        else max(1, num_workers * 2)
    )

    all_parquet_paths = list_parquet_files(args.data_dir or None)
    if args.max_files > 0:
        all_parquet_paths = all_parquet_paths[:args.max_files]
    if not all_parquet_paths:
        raise RuntimeError("No parquet shards found")

    if args.shard_index >= 0 and args.shard_name:
        raise ValueError("Use --shard-index or --shard-name, not both")
    if args.shard_index >= 0:
        if args.shard_index >= len(all_parquet_paths):
            raise IndexError(
                f"--shard-index {args.shard_index} out of range for "
                f"{len(all_parquet_paths)} parquet shards"
            )
        parquet_paths = [all_parquet_paths[args.shard_index]]
    elif args.shard_name:
        matches = [path for path in all_parquet_paths if Path(path).name == args.shard_name]
        if not matches:
            raise FileNotFoundError(f"No parquet shard named {args.shard_name!r}")
        parquet_paths = matches
    else:
        parquet_paths = all_parquet_paths
    selected_all_shards = len(parquet_paths) == len(all_parquet_paths)

    started = time.time()
    if args.finalize_only:
        outputs = []
        missing = []
        for input_path in all_parquet_paths:
            output_path = output_path_for(input_path, args.output_dir)
            completed = load_completed_shard(output_path)
            if completed is None:
                missing.append(output_path)
            else:
                outputs.append(completed)
        if missing:
            raise RuntimeError(
                "Cannot finalize segmented corpus; missing or invalid completed "
                f"shards: {missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )
        write_progress_manifests(
            args.output_dir,
            backend=args.backend,
            command=args.command,
            parquet_paths=all_parquet_paths,
            outputs=outputs,
            text_column=args.text_column,
            id_column=args.id_column,
            include_original=not args.compact,
            delimiter=args.delimiter,
            row_group_batch_size=args.row_group_batch_size,
            segment_batch_size=args.segment_batch_size,
            word_cache_size=args.word_cache_size,
            num_workers=num_workers,
            worker_mode=worker_mode,
            max_docs=args.max_docs,
            started=started,
            cache_entries=0,
            complete=True,
        )
        print(f"Wrote manifest to {Path(args.output_dir) / 'manifest.json'}")
        print(f"Wrote dataset manifest to {Path(args.output_dir) / MANIFEST_FILE}")
        return

    segmenter = create_segmenter(
        args.backend,
        command=args.command or None,
        strict=args.strict,
        timeout=args.timeout,
    )
    cache = LRUWordCache(args.word_cache_size)
    outputs = []
    print(
        "Segmentation worker config: "
        f"num_workers={num_workers} "
        f"worker_mode={worker_mode} "
        f"worker_cache_size={worker_cache_size} "
        f"max_in_flight={max_in_flight}",
        flush=True,
    )

    try:
        for input_path in parquet_paths:
            output_path = output_path_for(input_path, args.output_dir)
            completed = load_completed_shard(output_path)
            if completed is not None and args.resume and not args.overwrite:
                print(f"Skipping completed shard {output_path}", flush=True)
                outputs.append(completed)
                if selected_all_shards:
                    write_progress_manifests(
                        args.output_dir,
                        backend=args.backend,
                        command=args.command,
                        parquet_paths=all_parquet_paths,
                        outputs=outputs,
                        text_column=args.text_column,
                        id_column=args.id_column,
                        include_original=not args.compact,
                        delimiter=args.delimiter,
                        row_group_batch_size=args.row_group_batch_size,
                        segment_batch_size=args.segment_batch_size,
                        word_cache_size=args.word_cache_size,
                        num_workers=num_workers,
                        worker_mode=worker_mode,
                        max_docs=args.max_docs,
                        started=started,
                        cache_entries=len(cache),
                        complete=False,
                    )
                continue
            if args.resume and completed is None and os.path.exists(output_path):
                print(f"Removing incomplete shard before resume: {output_path}", flush=True)
                remove_incomplete_shard(output_path)
            if os.path.exists(output_path) and not args.overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use --overwrite to replace it "
                    "or --resume to skip/rewrite shard outputs safely."
                )
            tmp_output_path = shard_tmp_path(output_path)
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
            print(f"Segmenting {input_path} -> {output_path}", flush=True)
            shard_stats = write_segmented_shard(
                input_path,
                tmp_output_path,
                text_column=args.text_column,
                id_column=args.id_column,
                include_original=not args.compact,
                backend=args.backend,
                command=args.command,
                strict=args.strict,
                timeout=args.timeout,
                segmenter=segmenter,
                cache=cache,
                row_group_batch_size=args.row_group_batch_size,
                segment_batch_size=args.segment_batch_size,
                delimiter=args.delimiter,
                max_docs=args.max_docs,
                num_workers=num_workers,
                worker_mode=worker_mode,
                worker_cache_size=worker_cache_size,
                max_in_flight=max_in_flight,
            )
            os.replace(tmp_output_path, output_path)
            shard_stats["output_path"] = output_path
            shard_stats["sha256"] = file_sha256(output_path)
            shard_stats["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_json_atomic(shard_done_path(output_path), shard_stats)
            outputs.append(shard_stats)
            if selected_all_shards:
                write_progress_manifests(
                    args.output_dir,
                    backend=args.backend,
                    command=args.command,
                    parquet_paths=all_parquet_paths,
                    outputs=outputs,
                    text_column=args.text_column,
                    id_column=args.id_column,
                    include_original=not args.compact,
                    delimiter=args.delimiter,
                    row_group_batch_size=args.row_group_batch_size,
                    segment_batch_size=args.segment_batch_size,
                    word_cache_size=args.word_cache_size,
                    num_workers=num_workers,
                    worker_mode=worker_mode,
                    max_docs=args.max_docs,
                    started=started,
                    cache_entries=len(cache),
                    complete=False,
                )
    except (SegmenterUnavailable, SegmentationError):
        raise

    if not selected_all_shards:
        print(
            "Shard selection complete. Run with --finalize-only after all "
            "expected shards have .done.json sidecars.",
            flush=True,
        )
        return

    write_progress_manifests(
        args.output_dir,
        backend=args.backend,
        command=args.command,
        parquet_paths=all_parquet_paths,
        outputs=outputs,
        text_column=args.text_column,
        id_column=args.id_column,
        include_original=not args.compact,
        delimiter=args.delimiter,
        row_group_batch_size=args.row_group_batch_size,
        segment_batch_size=args.segment_batch_size,
        word_cache_size=args.word_cache_size,
        num_workers=num_workers,
        worker_mode=worker_mode,
        max_docs=args.max_docs,
        started=started,
        cache_entries=len(cache),
        complete=True,
    )
    manifest_path = Path(args.output_dir) / "manifest.json"
    print(f"Wrote manifest to {manifest_path}")

    dataset_manifest_path = Path(args.output_dir) / MANIFEST_FILE
    print(f"Wrote dataset manifest to {dataset_manifest_path}")


if __name__ == "__main__":
    main()
