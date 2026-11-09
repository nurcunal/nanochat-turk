"""Materialize matched train/validation prefixes without altering source data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.dataset import list_parquet_files
from nanochat.morphology import MORPHEME_BOUNDARY


def prepare(source: Path, output: Path, max_docs: int) -> dict:
    if max_docs <= 0:
        raise ValueError("max_docs must be positive")
    output.mkdir(parents=True, exist_ok=False)
    source_files = list_parquet_files(str(source))
    if len(source_files) < 2:
        raise ValueError("Source must contain training shards and a separate final validation shard")
    manifest_path = source / "fineweb2_manifest.json"
    result = {
        "source_data_dir": str(source.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_manifest": json.loads(manifest_path.read_text()),
        "sample_limit_per_split": max_docs,
        "selection": "first documents in manifest order; train excludes the final source shard; val uses only that final shard",
        "document_hash_encoding": "concatenate 8-byte big-endian UTF-8 length followed by UTF-8 bytes for each document",
        "splits": {},
    }
    filenames = []
    for split, paths in (("train", source_files[:-1]), ("val", source_files[-1:])):
        filename = f"{split}.parquet"
        target = output / filename
        filenames.append(filename)
        annotated_hash = hashlib.sha256()
        raw_hash = hashlib.sha256()
        summary = {"documents": 0, "raw_bytes": 0, "raw_chars": 0, "regex_words": 0, "source_parquet_files_read": []}
        with pq.ParquetWriter(target, pa.schema([("segmented_text", pa.string())]), compression="zstd") as writer:
            for path in paths:
                if summary["documents"] >= max_docs:
                    break
                summary["source_parquet_files_read"].append(str(Path(path).resolve()))
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=250, columns=["segmented_text"]):
                    docs = batch.column(0).to_pylist()[:max_docs - summary["documents"]]
                    if not docs:
                        break
                    writer.write_table(pa.table({"segmented_text": docs}), row_group_size=250)
                    for doc in docs:
                        annotated = doc.encode("utf-8")
                        raw_text = doc.replace(MORPHEME_BOUNDARY, "")
                        raw = raw_text.encode("utf-8")
                        for digest, value in ((annotated_hash, annotated), (raw_hash, raw)):
                            digest.update(len(value).to_bytes(8, "big"))
                            digest.update(value)
                        summary["documents"] += 1
                        summary["raw_bytes"] += len(raw)
                        summary["raw_chars"] += len(raw_text)
                        summary["regex_words"] += len(re.findall(r"\w+", raw_text))
        summary["annotated_document_sequence_sha256"] = annotated_hash.hexdigest()
        summary["raw_document_sequence_sha256"] = raw_hash.hexdigest()
        summary["sample_parquet_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        result["splits"][split] = summary
    (output / "fineweb2_manifest.json").write_text(json.dumps({"filenames": filenames, "text_column": "segmented_text", "complete": True}, indent=2) + "\n")
    (output / "sample_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.max_docs), indent=2, ensure_ascii=False))
