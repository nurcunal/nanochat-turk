"""Materialize the 18 complete cancelled-run candidates without quality claims.

This is deliberately a separate salvage lineage.  It preserves every train
candidate exactly as stored, recomputes the exact-text hash used for splitting,
and never interprets candidate placeholder fields as evidence for deduplication
or quality filtering.  Evaluation rows whose whole document does not fit the
pinned 2,049-element validation row contract are omitted from that evaluation
file only and are counted explicitly; they are never moved into training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    load_json_strict,
    seal_manifest,
    validate_dataset_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.packing_capacity import (
    PackingDocument,
    run_upstream_loader_parity_fixture,
    simulate_bestfit_rank,
)
from nanochat.partial_tokenizer import (
    load_verified_partial_tokenizer,
    verify_partial_tokenizer_package,
)
from nanochat.strict_dataloader import verify_strict_dataset
from nanochat.turkish_corpus import (
    assign_split,
    canonical_text_hash,
    dominant_register,
    load_corpus_policy,
    select_mixture_bucket,
)
from nanochat.turkish_backend import (
    select_resource_sample_ranks,
    validate_source_plan,
)


DEFAULT_POLICY = Path("configs/pretrain/tr_d32_turkish_general_v4.json")
DEFAULT_EXPECTED_RANKS = (
    26,
    52,
    78,
    112,
    119,
    126,
    135,
    154,
    158,
    165,
    169,
    171,
    179,
    184,
    188,
    207,
    209,
    210,
)
DEFAULT_TOKENIZER_PACKAGE_SHA256 = (
    "909bfa20516c79b7349d3e35aacd655ef584aac431c055966366cf6e1545d871"
)
DEFAULT_TOKENIZER_RECEIPT_SHA256 = (
    "5014766c50fee069fde94806a9eb82de9c42e78d8d55632c4e37f2ebed94c445"
)
DEFAULT_SOURCE_JOB_ID = 512374
DEFAULT_TOKENIZER_JOB_ID = 512413
DEFAULT_SOURCE_PLAN_SHA256 = (
    "53ca06b8686049c4a52926bc4b1418b0380785c64f3f0ffe3925b9e2a4a09570"
)
INPUT_COLUMNS = (
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
    "genre",
    "pii_replacements",
    "harmful_signal_hits",
    "quality_filter_flags",
    "formatting_changes",
    "candidate_rank",
    "candidate_doc_index",
)
DERIVED_FIELDS = (
    pa.field("mixture_id", pa.string(), nullable=False),
    pa.field("split", pa.string(), nullable=False),
    pa.field("canonical_text_sha256", pa.string(), nullable=False),
    pa.field("register_bucket", pa.string(), nullable=False),
    pa.field("encoded_tokens_with_bos", pa.int64(), nullable=False),
    pa.field("shuffle_key", pa.string(), nullable=False),
    pa.field("shuffle_bucket", pa.int16(), nullable=False),
)
CORPUS_MANIFEST_FILE = "partial18_corpus_manifest.json"
CAPACITY_RECEIPT_FILE = "partial18_capacity_receipt.json"
MATERIALIZATION_RECEIPT_FILE = "partial18_materialization_receipt.json"
CORPUS_NAME = "tr_general_candidates_partial18_v1"
TRAIN_SHUFFLE_BUCKETS = 256
TRAIN_SHUFFLE_SEED_SUFFIX = "partial18-global-train-shuffle-v1"


class MultisetDigest:
    """Order-independent SHA-256 accumulator that preserves multiplicity."""

    _MODULUS = 1 << 256

    def __init__(self) -> None:
        self.count = 0
        self.modular_sum = 0

    def add(self, value: bytes) -> None:
        self.count += 1
        self.modular_sum = (
            self.modular_sum + int.from_bytes(hashlib.sha256(value).digest(), "big")
        ) % self._MODULUS

    def render(self) -> dict[str, Any]:
        return {
            "algorithm": "count_plus_modular_sum_of_sha256_v1",
            "count": self.count,
            "modular_sum_hex": f"{self.modular_sum:064x}",
        }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _parse_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(sorted({int(part) for part in value.split(",") if part}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer ranks") from exc
    if not ranks or any(rank < 0 for rank in ranks):
        raise argparse.ArgumentTypeError("expected at least one non-negative rank")
    return ranks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--sample-ranks", type=Path, required=True)
    parser.add_argument(
        "--expected-source-plan-sha256", default=DEFAULT_SOURCE_PLAN_SHA256
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-ranks", type=_parse_ranks, default=DEFAULT_EXPECTED_RANKS
    )
    parser.add_argument(
        "--expected-tokenizer-package-sha256",
        default=DEFAULT_TOKENIZER_PACKAGE_SHA256,
    )
    parser.add_argument(
        "--expected-tokenizer-receipt-sha256",
        default=DEFAULT_TOKENIZER_RECEIPT_SHA256,
    )
    parser.add_argument("--source-cancelled-job-id", type=int, default=DEFAULT_SOURCE_JOB_ID)
    parser.add_argument("--source-job-state", default="CANCELLED")
    parser.add_argument("--source-job-partition", default="cpu2dq")
    parser.add_argument("--source-job-allocated-cpus", type=int, default=128)
    parser.add_argument("--tokenizer-job-id", type=int, default=DEFAULT_TOKENIZER_JOB_ID)
    parser.add_argument("--rows-per-train-shard", type=int, default=262_144)
    parser.add_argument("--row-group-rows", type=int, default=4_096)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-threads", type=int, default=32)
    return parser


def _candidate_paths(input_root: Path, expected_ranks: tuple[int, ...]) -> list[Path]:
    paths = sorted(
        (input_root / "objects").glob(
            "[0-9][0-9][0-9][0-9][0-9]/candidates.parquet"
        )
    )
    actual = tuple(int(path.parent.name) for path in paths)
    if actual != expected_ranks:
        raise ValueError(
            f"candidate rank inventory differs: expected={expected_ranks}, actual={actual}"
        )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("candidate inventory contains a symlink or non-regular file")
    return paths


def build_input_inventory(
    input_root: Path, expected_ranks: tuple[int, ...]
) -> tuple[list[dict[str, Any]], pa.Schema]:
    inventory: list[dict[str, Any]] = []
    expected_schema: pa.Schema | None = None
    for path in _candidate_paths(input_root, expected_ranks):
        rank = int(path.parent.name)
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        if tuple(schema.names) != INPUT_COLUMNS:
            raise ValueError(
                f"rank {rank} candidate columns differ from the frozen candidate schema"
            )
        if expected_schema is None:
            expected_schema = schema
        elif not schema.equals(expected_schema, check_metadata=False):
            raise ValueError(f"rank {rank} candidate Arrow schema drift")
        if parquet.metadata.num_rows <= 0 or parquet.num_row_groups <= 0:
            raise ValueError(f"rank {rank} candidate Parquet is empty")
        inventory.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "object_receipt_present": (path.parent / "object_receipt.json").is_file(),
                "object_receipt_sha256": (
                    file_sha256(path.parent / "object_receipt.json")
                    if (path.parent / "object_receipt.json").is_file()
                    else None
                ),
            }
        )
    assert expected_schema is not None
    return inventory, expected_schema


def verify_source_evidence(
    source_plan_path: Path,
    sample_ranks_path: Path,
    *,
    policy: Mapping[str, Any],
    expected_ranks: tuple[int, ...],
    expected_source_plan_sha256: str,
) -> dict[str, Any]:
    if (
        source_plan_path.is_symlink()
        or not source_plan_path.is_file()
        or sample_ranks_path.is_symlink()
        or not sample_ranks_path.is_file()
    ):
        raise ValueError("source plan and sample-ranks evidence must be regular files")
    plan = load_json_strict(source_plan_path)
    validate_source_plan(plan, policy)
    plan_sha = verify_manifest_hash(plan)
    if plan_sha != expected_source_plan_sha256:
        raise ValueError("source-plan canonical SHA-256 differs from the pinned plan")
    payload = load_json_strict(sample_ranks_path)
    if not isinstance(payload, Mapping):
        raise ValueError("resource sample ranks must be a JSON object")
    ranks = payload.get("ranks")
    if (
        not isinstance(ranks, list)
        or any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks)
        or ranks != sorted(set(ranks))
        or tuple(ranks) != expected_ranks
        or ranks != select_resource_sample_ranks(plan)
        or payload.get("slurm_array") != ",".join(str(rank) for rank in ranks)
    ):
        raise ValueError("resource sample ranks differ from the sealed source-plan selector")
    rank_records = [dict(plan["objects"][rank]) for rank in ranks]
    if [record.get("rank") for record in rank_records] != ranks:
        raise ValueError("selected source-plan object rank identity drift")
    return {
        "source_plan_path": str(source_plan_path.resolve()),
        "source_plan_sha256": plan_sha,
        "source_plan_file_sha256": file_sha256(source_plan_path),
        "sample_ranks_path": str(sample_ranks_path.resolve()),
        "sample_ranks_file_sha256": file_sha256(sample_ranks_path),
        "sample_ranks_canonical_payload_sha256": hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest(),
        "expected_ranks": list(expected_ranks),
        "selected_rank_records": rank_records,
    }


def _validate_candidate_lid_fields(
    row: Mapping[str, Any], source_id: str, policy: Mapping[str, Any]
) -> None:
    source = next(
        (item for item in policy["sources"] if item["id"] == source_id), None
    )
    if source is None:
        raise ValueError(f"candidate source {source_id!r} is absent from policy")
    adapter = source["adapter"]
    source_label = row.get("source_lid_label")
    source_probability = row.get("source_lid_probability")
    try:
        source_probability_float = float(source_probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate source-LID probability is not numeric") from exc
    if (
        not isinstance(source_label, str)
        or source_label not in set(adapter.get("turkish_values", []))
        or not math.isfinite(source_probability_float)
        or source_probability_float
        < float(adapter.get("source_lid_min_probability", 0.8))
        or source_probability_float > 1.0 + 1e-6
    ):
        raise ValueError("candidate source-LID fields fail the frozen row thresholds")
    gate = policy["language_policy"]["independent_audit"]
    numeric = {}
    for field in (
        "lid_probability",
        "lid_margin",
        "paragraph_min_probability",
        "paragraph_min_margin",
        "failed_long_paragraph_fraction",
    ):
        try:
            numeric[field] = float(row.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate {field} is not numeric") from exc
        if not math.isfinite(numeric[field]):
            raise ValueError(f"candidate {field} is non-finite")
    if (
        row.get("lid_label") != gate["required_top_label"]
        or numeric["lid_probability"] < float(gate["document_min_probability"])
        or numeric["lid_margin"] < float(gate["document_min_margin"])
        or numeric["paragraph_min_probability"]
        < float(gate["paragraph_min_probability"])
        or numeric["paragraph_min_margin"] < float(gate["paragraph_min_margin"])
        or numeric["failed_long_paragraph_fraction"]
        > float(gate["max_failed_long_paragraph_fraction"])
    ):
        raise ValueError("candidate GlotLID fields fail the frozen row thresholds")


def validate_available_object_receipts(
    input_root: Path,
    inventory: Sequence[Mapping[str, Any]],
    source_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_by_rank = {
        int(item["rank"]): item
        for item in source_evidence["selected_rank_records"]
    }
    attestations: list[dict[str, Any]] = []
    for item in inventory:
        rank = int(item["rank"])
        path = input_root / "objects" / f"{rank:05d}" / "object_receipt.json"
        if item["object_receipt_present"] is not True:
            if path.exists():
                raise ValueError(f"rank {rank} object-receipt presence accounting drift")
            continue
        if path.is_symlink() or not path.is_file() or file_sha256(path) != item[
            "object_receipt_sha256"
        ]:
            raise ValueError(f"rank {rank} object receipt file identity drift")
        receipt = load_json_strict(path)
        receipt_sha = verify_manifest_hash(receipt)
        planned = source_by_rank[rank]
        candidate = receipt.get("candidate_file")
        expected_relative = Path(str(item["path"])).relative_to(input_root.resolve()).as_posix()
        if (
            receipt.get("sample_mode") is not True
            or receipt.get("rank") != rank
            or receipt.get("source_id") != planned.get("source_id")
            or receipt.get("source_uri") != planned.get("uri")
            or receipt.get("source_plan_sha256")
            != source_evidence["source_plan_sha256"]
            or not isinstance(candidate, Mapping)
            or candidate.get("path") != expected_relative
            or candidate.get("size_bytes") != item["size_bytes"]
            or candidate.get("sha256") != item["sha256"]
            or candidate.get("rows") != item["rows"]
        ):
            raise ValueError(f"rank {rank} object receipt/candidate/source-plan binding drift")
        attestations.append(
            {
                "rank": rank,
                "source_id": planned["source_id"],
                "source_uri": planned["uri"],
                "receipt_file_sha256": item["object_receipt_sha256"],
                "receipt_canonical_sha256": receipt_sha,
                "candidate_sha256": item["sha256"],
            }
        )
    return attestations


def verify_partial_tokenizer(
    tokenizer_dir: Path,
    *,
    expected_package_sha256: str,
    expected_receipt_sha256: str,
    expected_ranks: tuple[int, ...],
    expected_source_job_id: int,
    expected_tokenizer_job_id: int,
) -> tuple[Any, dict[str, Any]]:
    if tokenizer_dir.is_symlink() or not tokenizer_dir.is_dir():
        raise ValueError("tokenizer directory must be a real directory")
    verified = verify_partial_tokenizer_package(
        tokenizer_dir / "package_manifest.json",
        expected_sha256=expected_package_sha256,
        expected_training_receipt_sha256=expected_receipt_sha256,
    )
    receipt = verified.receipt
    if (
        tuple(receipt.get("expected_ranks", [])) != expected_ranks
        or int(receipt.get("cancelled_source_job_id", -1)) != expected_source_job_id
        or str(receipt.get("producer_slurm_job_id")) != str(expected_tokenizer_job_id)
    ):
        raise ValueError("partial tokenizer source/job provenance drift")
    tokenizer = load_verified_partial_tokenizer(verified)
    return tokenizer, {
        "tokenizer_package_sha256": verified.canonical_sha256,
        "tokenizer_training_receipt_sha256": verified.receipt_sha256,
        "tokenizer_name": verified.manifest["name"],
        "tokenizer_job_id": expected_tokenizer_job_id,
        "tokenizer_input_inventory_sha256": verified.manifest[
            "input_inventory_sha256"
        ],
        "tokenizer_input_inventory": verified.receipt["input_inventory"],
        "tokenizer_input_root": verified.receipt["input_root"],
        "tokenizer_policy_sha256": verified.receipt["policy_sha256"],
        "tokenizer_split_policy": verified.receipt["split_policy"],
        "tokenizer_producer_git_commit": verified.receipt["git_commit"],
        "tokenizer_production_eligible": False,
    }


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def verify_final_partial18_output(
    root: Path,
    *,
    data_files: Sequence[Mapping[str, Any]],
    test_file: Mapping[str, Any],
    expected_manifest_hashes: Mapping[str, str],
) -> None:
    """Fail closed on the complete temporary tree before atomic publication."""

    expected_paths = {
        str(record["path"])
        for record in data_files
    } | set(expected_manifest_hashes)
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("final partial18 output contains a symlink")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != expected_paths:
        raise ValueError("final partial18 output file inventory differs")

    for relative, expected_sha256 in expected_manifest_hashes.items():
        manifest = load_json_strict(root / relative)
        if verify_manifest_hash(manifest) != expected_sha256:
            raise ValueError(f"final manifest hash differs: {relative}")

    # This re-hashes every strict train/validation file after all receipts have
    # been written. The separately inventoried test file is checked explicitly.
    verify_strict_dataset(root, verify_bytes=True)
    final_test = _file_record(root / str(test_file["path"]), root)
    if final_test != {
        key: test_file[key] for key in ("path", "size_bytes", "sha256")
    }:
        raise ValueError("final test-file inventory differs")


class DistributionCounters:
    _FIELDS = ("documents", "characters", "utf8_bytes", "tokens_with_bos")

    def __init__(self) -> None:
        self.values: dict[str, dict[tuple[str, str, str], Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )

    def add(
        self,
        scope: str,
        *,
        split: str,
        source_id: str,
        mixture_id: str,
        text: str,
        tokens_with_bos: int,
    ) -> None:
        counter = self.values[scope][(split, source_id, mixture_id)]
        counter["documents"] += 1
        counter["characters"] += len(text)
        counter["utf8_bytes"] += len(text.encode("utf-8"))
        counter["tokens_with_bos"] += tokens_with_bos

    @staticmethod
    def _sum(records: Mapping[tuple[str, str, str], Counter[str]]) -> dict[str, int]:
        return {
            field: sum(int(counter[field]) for counter in records.values())
            for field in DistributionCounters._FIELDS
        }

    def render(self, scope: str) -> dict[str, Any]:
        records = self.values.get(scope, {})
        totals = self._sum(records)
        by_split: dict[str, dict[str, int]] = {}
        for split in ("train", "val", "test"):
            selected = {key: value for key, value in records.items() if key[0] == split}
            by_split[split] = self._sum(selected)
        rows = []
        for (split, source_id, mixture_id), counter in sorted(records.items()):
            rows.append(
                {
                    "split": split,
                    "source_id": source_id,
                    "mixture_id": mixture_id,
                    **{field: int(counter[field]) for field in self._FIELDS},
                }
            )
        return {"totals": totals, "by_split": by_split, "by_split_source_mixture": rows}


def _row_locator_bytes(row: Mapping[str, Any]) -> bytes:
    return (
        f"{row['candidate_rank']}\0{row['candidate_doc_index']}\0{row['source_id']}\0"
        f"{row['document_id']}\0{row['canonical_text_sha256']}\n"
    ).encode("utf-8")


class TrainHashBucketStager:
    """Bounded staging for one exact global permutation by a 256-way hash prefix."""

    def __init__(
        self,
        root: Path,
        schema: pa.Schema,
        *,
        fragment_rows: int,
        max_buffered_rows: int = 262_144,
        max_buffered_characters: int = 1_500_000_000,
    ) -> None:
        self.root = root
        self.schema = schema
        self.fragment_rows = fragment_rows
        self.max_buffered_rows = max_buffered_rows
        self.max_buffered_characters = max_buffered_characters
        self.buffers: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.buffer_characters: Counter[int] = Counter()
        self.buffered_rows = 0
        self.buffered_characters = 0
        self.peak_buffered_rows = 0
        self.peak_buffered_characters = 0
        self.fragment_indexes: Counter[int] = Counter()
        self.fragments: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def add(self, row: dict[str, Any]) -> None:
        bucket = int(row["shuffle_bucket"])
        if not 0 <= bucket < TRAIN_SHUFFLE_BUCKETS:
            raise ValueError("train shuffle bucket is outside the frozen range")
        self.buffers[bucket].append(row)
        characters = len(row["text"])
        self.buffer_characters[bucket] += characters
        self.buffered_rows += 1
        self.buffered_characters += characters
        self.peak_buffered_rows = max(self.peak_buffered_rows, self.buffered_rows)
        self.peak_buffered_characters = max(
            self.peak_buffered_characters, self.buffered_characters
        )
        if len(self.buffers[bucket]) >= self.fragment_rows:
            self._flush(bucket)
        while (
            self.buffered_rows > self.max_buffered_rows
            or self.buffered_characters > self.max_buffered_characters
        ):
            largest = max(
                (key for key, rows in self.buffers.items() if rows),
                key=lambda key: (self.buffer_characters[key], len(self.buffers[key]), key),
            )
            self._flush(largest)

    def _flush(self, bucket: int) -> None:
        rows = self.buffers[bucket]
        if not rows:
            return
        index = self.fragment_indexes[bucket]
        self.fragment_indexes[bucket] += 1
        path = (
            self.root
            / ".staging"
            / "train_hash_buckets"
            / f"bucket-{bucket:03d}"
            / f"fragment-{index:06d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=self.schema)
        pq.write_table(
            table,
            path,
            compression="zstd",
            use_dictionary=True,
            row_group_size=len(rows),
        )
        self.fragments[bucket].append(
            _file_record(path, self.root)
            | {
                "bucket": bucket,
                "rows": len(rows),
                "characters": self.buffer_characters[bucket],
                "tokens_with_bos": sum(
                    int(row["encoded_tokens_with_bos"]) for row in rows
                ),
            }
        )
        self.buffered_rows -= len(rows)
        self.buffered_characters -= self.buffer_characters[bucket]
        self.buffers[bucket] = []
        self.buffer_characters[bucket] = 0

    def finish(self) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
        for bucket in sorted(self.buffers):
            self._flush(bucket)
        if self.buffered_rows or self.buffered_characters:
            raise ValueError("train staging buffer accounting drift")
        return dict(self.fragments), {
            "peak_buffered_rows": self.peak_buffered_rows,
            "peak_buffered_characters": self.peak_buffered_characters,
            "fragments": sum(len(records) for records in self.fragments.values()),
        }


def emit_globally_shuffled_train(
    root: Path,
    schema: pa.Schema,
    fragments: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    rows_per_train_shard: int,
    row_group_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], MultisetDigest]:
    output_records: list[dict[str, Any]] = []
    bucket_receipts: list[dict[str, Any]] = []
    emitted_multiset = MultisetDigest()
    emitted_sequence = hashlib.sha256()
    previous_key: str | None = None
    shard_index = 0
    state: dict[str, Any] | None = None
    pending_rows: list[dict[str, Any]] = []

    def open_shard() -> dict[str, Any]:
        path = root / f"train-{shard_index:05d}.parquet"
        return {
            "path": path,
            "writer": pq.ParquetWriter(
                path, schema, compression="zstd", use_dictionary=True
            ),
            "rows": 0,
            "row_groups": 0,
            "characters": 0,
            "utf8_bytes": 0,
            "tokens_with_bos": 0,
        }

    def close_shard() -> None:
        nonlocal state
        if state is None:
            return
        state["writer"].close()
        if state["rows"] == rows_per_train_shard and (
            state["row_groups"] != rows_per_train_shard // row_group_rows
            or state["row_groups"] % 16
        ):
            raise ValueError("full train shard row-group topology is not ws16-balanced")
        output_records.append(
            _file_record(state["path"], root)
            | {
                key: int(state[key])
                for key in (
                    "rows",
                    "row_groups",
                    "characters",
                    "utf8_bytes",
                    "tokens_with_bos",
                )
            }
        )
        state = None

    def write_group(rows: list[dict[str, Any]]) -> None:
        nonlocal state, shard_index
        if not rows or len(rows) > row_group_rows:
            raise ValueError("invalid globally shuffled output row group")
        if state is not None and int(state["rows"]) == rows_per_train_shard:
            close_shard()
            shard_index += 1
        if state is None:
            state = open_shard()
        if int(state["rows"]) + len(rows) > rows_per_train_shard:
            raise ValueError("global shuffle row-group/shard divisibility drift")
        chunk = pa.Table.from_pylist(rows, schema=schema)
        state["writer"].write_table(chunk, row_group_size=chunk.num_rows)
        state["rows"] += chunk.num_rows
        state["row_groups"] += 1
        state["characters"] += sum(len(row["text"]) for row in rows)
        state["utf8_bytes"] += sum(len(row["text"].encode("utf-8")) for row in rows)
        state["tokens_with_bos"] += sum(
            int(row["encoded_tokens_with_bos"]) for row in rows
        )
        for row in rows:
            locator = _row_locator_bytes(row)
            emitted_multiset.add(locator)
            emitted_sequence.update(locator)

    for bucket in range(TRAIN_SHUFFLE_BUCKETS):
        records = list(fragments.get(bucket, ()))
        if not records:
            continue
        tables = []
        for record in records:
            path = root / str(record["path"])
            if (
                path.is_symlink()
                or path.stat().st_size != int(record["size_bytes"])
                or file_sha256(path) != record["sha256"]
            ):
                raise ValueError("train staging fragment identity drift")
            tables.append(pq.read_table(path))
        table = pa.concat_tables(tables)
        expected_rows = sum(int(record["rows"]) for record in records)
        expected_tokens = sum(int(record["tokens_with_bos"]) for record in records)
        if table.num_rows != expected_rows:
            raise ValueError("train staging bucket row accounting drift")
        table = table.sort_by([("shuffle_key", "ascending")])
        keys = table.column("shuffle_key").to_pylist()
        if any(
            int(str(key)[:2], 16) != bucket
            for key in keys
        ) or any(left >= right for left, right in zip(keys, keys[1:])):
            raise ValueError("train bucket is not an exact unique shuffle-key ordering")
        if previous_key is not None and previous_key >= keys[0]:
            raise ValueError("global train shuffle order drift across buckets")
        previous_key = keys[-1]
        bucket_sequence = hashlib.sha256()
        sorted_rows = table.to_pylist()
        bucket_tokens = sum(
            int(row["encoded_tokens_with_bos"]) for row in sorted_rows
        )
        for row in sorted_rows:
            bucket_sequence.update(_row_locator_bytes(row))
            pending_rows.append(row)
            if len(pending_rows) == row_group_rows:
                write_group(pending_rows)
                pending_rows = []
        if bucket_tokens != expected_tokens:
            raise ValueError("train staging bucket token accounting drift")
        bucket_receipts.append(
            {
                "bucket": bucket,
                "rows": expected_rows,
                "tokens_with_bos": expected_tokens,
                "fragments": len(records),
                "fragment_inventory_sha256": hashlib.sha256(
                    canonical_json(records).encode("utf-8")
                ).hexdigest(),
                "first_shuffle_key": keys[0],
                "last_shuffle_key": keys[-1],
                "emitted_row_sequence_sha256": bucket_sequence.hexdigest(),
            }
        )
    if pending_rows:
        write_group(pending_rows)
    close_shard()
    if not output_records:
        raise ValueError("global train shuffle emitted no train rows")
    staging = root / ".staging"
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("run-owned train staging directory is unsafe or missing")
    shutil.rmtree(staging)
    return output_records, {
        "algorithm": "sha256_prefix256_global_full_hash_sort_v1",
        "buckets": TRAIN_SHUFFLE_BUCKETS,
        "bucket_processing_order": "ascending_first_hash_byte_000_to_255",
        "within_bucket_order": "ascending_full_shuffle_key",
        "row_group_rows": row_group_rows,
        "all_assigned_train_rows_emitted_once": True,
        "emitted_row_sequence_sha256": emitted_sequence.hexdigest(),
        "bucket_receipts": bucket_receipts,
    }, emitted_multiset


class CorpusWriters:
    def __init__(
        self,
        root: Path,
        schema: pa.Schema,
        *,
        rows_per_train_shard: int,
        row_group_rows: int,
    ) -> None:
        self.root = root
        self.schema = schema
        self.rows_per_train_shard = rows_per_train_shard
        self.row_group_rows = row_group_rows
        self.buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.train_index = 0
        self.states: dict[str, dict[str, Any]] = {}
        self.records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _path_for(self, split: str) -> Path:
        if split == "train":
            return self.root / f"train-{self.train_index:05d}.parquet"
        if split == "val":
            return self.root / "validation.parquet"
        if split == "test":
            return self.root / "test" / "test.parquet"
        raise ValueError(f"unknown split {split!r}")

    def _open(self, split: str) -> dict[str, Any]:
        path = self._path_for(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "path": path,
            "writer": pq.ParquetWriter(
                path, self.schema, compression="zstd", use_dictionary=True
            ),
            "rows": 0,
            "row_groups": 0,
            "characters": 0,
            "utf8_bytes": 0,
            "tokens_with_bos": 0,
        }
        self.states[split] = state
        return state

    def add(self, split: str, row: dict[str, Any]) -> None:
        self.buffers[split].append(row)
        if len(self.buffers[split]) >= self.row_group_rows:
            self._flush(split)

    def _flush(self, split: str) -> None:
        buffer = self.buffers[split]
        while buffer:
            state = self.states.get(split) or self._open(split)
            limit = (
                self.rows_per_train_shard - int(state["rows"])
                if split == "train"
                else len(buffer)
            )
            chunk = buffer[: min(len(buffer), limit)]
            del buffer[: len(chunk)]
            state["writer"].write_table(
                pa.Table.from_pylist(chunk, schema=self.schema),
                row_group_size=len(chunk),
            )
            state["rows"] += len(chunk)
            state["row_groups"] += 1
            state["characters"] += sum(len(row["text"]) for row in chunk)
            state["utf8_bytes"] += sum(len(row["text"].encode("utf-8")) for row in chunk)
            state["tokens_with_bos"] += sum(
                int(row["encoded_tokens_with_bos"]) for row in chunk
            )
            if split == "train" and state["rows"] == self.rows_per_train_shard:
                self._close("train")
                self.train_index += 1

    def _close(self, split: str) -> None:
        state = self.states.pop(split, None)
        if state is None:
            return
        state["writer"].close()
        record = _file_record(state["path"], self.root) | {
            key: int(state[key])
            for key in (
                "rows",
                "row_groups",
                "characters",
                "utf8_bytes",
                "tokens_with_bos",
            )
        }
        self.records[split].append(record)

    def finish(
        self, required_splits: Sequence[str] = ("train", "val", "test")
    ) -> dict[str, list[dict[str, Any]]]:
        for split in ("train", "test", "val"):
            self._flush(split)
            self._close(split)
        for split in required_splits:
            if not self.records[split]:
                raise ValueError(f"partial corpus produced no materialized {split} rows")
        return {key: list(value) for key, value in self.records.items()}


def _stored_document_batches(
    root: Path,
    train_files: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    batch_size: int,
) -> Iterator[list[PackingDocument]]:
    for file_record in train_files:
        path = root / str(file_record["path"])
        parquet = pq.ParquetFile(path)
        row_group_index = rank
        while row_group_index < parquet.num_row_groups:
            rows = parquet.read_row_group(
                row_group_index,
                columns=[
                    "encoded_tokens_with_bos",
                    "document_id",
                    "mixture_id",
                    "source_id",
                    "register_bucket",
                ],
            ).to_pylist()
            for offset in range(0, len(rows), batch_size):
                yield [
                    PackingDocument(
                        tokens_with_bos=int(row["encoded_tokens_with_bos"]),
                        document_id=str(row["document_id"]),
                        mixture_id=str(row["mixture_id"]),
                        source_id=str(row["source_id"]),
                        register_bucket=str(row["register_bucket"]),
                    )
                    for row in rows[offset : offset + batch_size]
                ]
            row_group_index += world_size


def _merge_metrics(
    results: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    merged = {dimension: Counter() for dimension in ("mixture", "source", "register")}
    for result in results:
        for dimension, values in result[field].items():
            merged[dimension].update({str(key): int(value) for key, value in values.items()})
    return {
        dimension: dict(sorted(counter.items()))
        for dimension, counter in merged.items()
    }


def simulate_partial_first_epoch_capacity(
    root: Path,
    train_files: Sequence[Mapping[str, Any]],
    *,
    world_sizes: Sequence[int],
    B: int,
    T: int,
    buffer_size: int,
    tokenizer_batch_size: int,
    global_batch_tokens: int,
    verify_parity: bool = True,
) -> dict[str, Any]:
    if not train_files or min(B, T, buffer_size, tokenizer_batch_size) <= 0:
        raise ValueError("invalid first-epoch capacity simulation arguments")
    parity = run_upstream_loader_parity_fixture() if verify_parity else None
    worlds: dict[str, Any] = {}
    for world_size in world_sizes:
        denominator = int(world_size) * B * T
        if global_batch_tokens % denominator:
            raise ValueError("global batch is not divisible by rank-local microbatch")
        first_pass = [
            simulate_bestfit_rank(
                _stored_document_batches(
                    root,
                    train_files,
                    rank=rank,
                    world_size=int(world_size),
                    batch_size=tokenizer_batch_size,
                ),
                B=B,
                T=T,
                buffer_size=buffer_size,
            )
            for rank in range(int(world_size))
        ]
        common = min(int(item["completed_microbatches"]) for item in first_pass)
        if common <= 0:
            raise ValueError(f"world size {world_size} has no complete first-epoch microbatch")
        common_results = [
            simulate_bestfit_rank(
                _stored_document_batches(
                    root,
                    train_files,
                    rank=rank,
                    world_size=int(world_size),
                    batch_size=tokenizer_batch_size,
                ),
                B=B,
                T=T,
                buffer_size=buffer_size,
                max_microbatches=common,
            )
            for rank in range(int(world_size))
        ]
        if any(item["stop_reason"] != "requested_horizon_reached" for item in common_results):
            raise ValueError("common-prefix replay failed before its measured boundary")
        accumulation = global_batch_tokens // denominator
        optimizer_steps = common // accumulation
        retained = _merge_metrics(common_results, "retained_positions")
        cropped = _merge_metrics(common_results, "cropped_tokens")
        consumed = _merge_metrics(common_results, "consumed_source_elements")
        retained_total = sum(retained["mixture"].values())
        expected_retained = common * denominator
        if retained_total != expected_retained:
            raise ValueError("best-fit retained-position accounting drift")
        worlds[str(world_size)] = {
            "world_size": int(world_size),
            "gradient_accumulation": accumulation,
            "first_epoch_common_complete_microbatches_per_rank": common,
            "first_epoch_common_complete_optimizer_steps": optimizer_steps,
            "first_epoch_common_prefix_scheduled_positions": expected_retained,
            "first_epoch_optimizer_aligned_packed_positions": (
                optimizer_steps * global_batch_tokens
            ),
            "discarded_unaligned_microbatches_per_rank": common % accumulation,
            "retained_positions": retained,
            "cropped_tokens": cropped,
            "consumed_source_elements": consumed,
            "cropped_source_tokens": sum(cropped["mixture"].values()),
            "documents_completed": sum(
                int(item["documents_completed"]) for item in common_results
            ),
            "documents_cropped": sum(
                int(item["documents_cropped"]) for item in common_results
            ),
            "realized_retained_mix": {
                key: value / retained_total
                for key, value in sorted(retained["mixture"].items())
            },
            "rank_wrap_diagnostics": [
                {
                    "completed_microbatches": item["completed_microbatches"],
                    "loaded_documents": item["loaded_documents"],
                    "loaded_source_tokens_with_bos": item[
                        "loaded_source_tokens_with_bos"
                    ],
                    "buffered_documents_at_stop": item["buffered_documents_at_stop"],
                    "buffered_source_tokens_at_stop": item[
                        "buffered_source_tokens_at_stop"
                    ],
                    "stop_reason": item["stop_reason"],
                    "incomplete_microbatch_source_elements_at_wrap": item[
                        "incomplete_microbatch_source_elements_at_wrap"
                    ],
                    "incomplete_microbatch_cropped_tokens_at_wrap": item[
                        "incomplete_microbatch_cropped_tokens_at_wrap"
                    ],
                }
                for item in first_pass
            ],
        }
    return {
        "implementation": "partial18_stored_exact_token_bos_bestfit_first_epoch_v1",
        "token_length_authority": (
            "encoded_tokens_with_bos_counted_by_pinned_partial_tokenizer_during_"
            "materialization"
        ),
        "upstream_contract": {
            "device_batch_sequences": B,
            "max_seq_len": T,
            "row_source_elements": T + 1,
            "buffer_size": buffer_size,
            "tokenizer_batch_size": tokenizer_batch_size,
            "cropped_tail_policy": "discard",
            "rank_sharding": "parquet_row_group_index_mod_world_size",
            "tie_breaks": "first_largest_fit_else_first_shortest",
        },
        "world_sizes": list(world_sizes),
        "global_batch_tokens": global_batch_tokens,
        "fixture_parity": parity,
        "worlds": worlds,
    }


def _realized_train_mix(
    written: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[dict[str, Any]]:
    tokens = Counter()
    documents = Counter()
    for row in written["by_split_source_mixture"]:
        if row["split"] == "train":
            tokens[row["mixture_id"]] += int(row["tokens_with_bos"])
            documents[row["mixture_id"]] += int(row["documents"])
    total_tokens = sum(tokens.values())
    weights = {str(item["id"]): float(item["weight"]) for item in policy["mixture"]}
    return [
        {
            "mixture_id": key,
            "target_share": weights[key],
            "documents": documents[key],
            "tokens_with_bos": tokens[key],
            "realized_token_share": tokens[key] / total_tokens,
            "realized_minus_target": tokens[key] / total_tokens - weights[key],
        }
        for key in sorted(weights)
    ]


def materialize_partial18(
    *,
    input_root: Path,
    output_dir: Path,
    policy: Mapping[str, Any],
    policy_path: Path,
    tokenizer: Any,
    tokenizer_lineage: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    expected_ranks: tuple[int, ...],
    git_commit: str,
    source_job_provenance: Mapping[str, Any],
    rows_per_train_shard: int = 262_144,
    row_group_rows: int = 4_096,
    tokenizer_batch_size: int = 128,
    tokenizer_threads: int = 32,
    eval_max_tokens_with_bos: int = 2_049,
    capacity_world_sizes: Sequence[int] = (8, 16),
    capacity_B: int = 4,
    capacity_T: int = 2_048,
    capacity_buffer_size: int = 1_000,
    capacity_global_batch_tokens: int = 2_097_152,
    verify_capacity_parity: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output path: {output_dir}")
    if min(
        rows_per_train_shard,
        row_group_rows,
        tokenizer_batch_size,
        tokenizer_threads,
        eval_max_tokens_with_bos,
    ) <= 0:
        raise ValueError("materialization limits must be positive")
    if row_group_rows > rows_per_train_shard:
        raise ValueError("row-group rows cannot exceed train-shard rows")
    if (
        rows_per_train_shard % row_group_rows
        or (rows_per_train_shard // row_group_rows) % 16
    ):
        raise ValueError(
            "full train shards require an integer row-group count divisible by 16"
        )
    inventory, input_schema = build_input_inventory(input_root, expected_ranks)
    tokenizer_inventory = tokenizer_lineage.get("tokenizer_input_inventory")
    if not isinstance(tokenizer_inventory, list):
        raise ValueError("verified tokenizer lineage lacks its candidate inventory")
    tokenizer_by_rank = {int(item["rank"]): item for item in tokenizer_inventory}
    source_records = source_evidence.get("selected_rank_records")
    if not isinstance(source_records, list):
        raise ValueError("verified source evidence lacks selected rank records")
    source_by_rank = {int(item["rank"]): item for item in source_records}
    if tuple(sorted(tokenizer_by_rank)) != expected_ranks or tuple(
        sorted(source_by_rank)
    ) != expected_ranks:
        raise ValueError("candidate/tokenizer/source-plan rank inventory drift")
    for item in inventory:
        item["source_id"] = source_by_rank[int(item["rank"])]["source_id"]
        tokenizer_item = tokenizer_by_rank[int(item["rank"])]
        for field in (
            "rank",
            "source_id",
            "size_bytes",
            "sha256",
            "rows",
            "row_groups",
            "object_receipt_present",
            "object_receipt_sha256",
        ):
            if item[field] != tokenizer_item[field]:
                raise ValueError(
                    f"candidate rank {item['rank']} differs from tokenizer inventory at {field}"
                )
        if Path(str(tokenizer_item["path"])).resolve() != Path(str(item["path"])).resolve():
            raise ValueError(f"candidate rank {item['rank']} path differs from tokenizer inventory")
    inventory_sha = hashlib.sha256(canonical_json(inventory).encode("utf-8")).hexdigest()
    if inventory_sha != tokenizer_lineage.get("tokenizer_input_inventory_sha256"):
        raise ValueError("candidate inventory digest differs from verified tokenizer lineage")
    object_receipt_attestations = validate_available_object_receipts(
        input_root, inventory, source_evidence
    )
    policy_sha = hashlib.sha256(canonical_json(policy).encode("utf-8")).hexdigest()
    if (
        tokenizer_lineage.get("tokenizer_policy_sha256") != policy_sha
        or tokenizer_lineage.get("tokenizer_split_policy") != policy["splits"]
        or Path(str(tokenizer_lineage.get("tokenizer_input_root"))).resolve()
        != input_root.resolve()
    ):
        raise ValueError("tokenizer policy/split/input-root lineage differs from corpus")
    output_schema = pa.schema([*input_schema, *DERIVED_FIELDS])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / (
        f".{output_dir.name}.job{os.environ.get('SLURM_JOB_ID', 'local')}.tmp"
    )
    if temporary.exists():
        raise FileExistsError(f"refusing existing temporary output: {temporary}")
    temporary.mkdir()
    writers = CorpusWriters(
        temporary,
        output_schema,
        rows_per_train_shard=rows_per_train_shard,
        row_group_rows=row_group_rows,
    )
    train_stager = TrainHashBucketStager(
        temporary,
        output_schema,
        fragment_rows=row_group_rows,
    )
    eval_rows: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
    shuffle_seed = f"{policy['splits']['seed']}:{TRAIN_SHUFFLE_SEED_SUFFIX}"
    counters = DistributionCounters()
    input_sequence = hashlib.sha256()
    written_sequence = hashlib.sha256()
    excluded_sequence = hashlib.sha256()
    placeholder_dedup_keep = Counter()
    placeholder_quality_flags = Counter()
    assigned_train_multiset = MultisetDigest()
    lid_rows_validated = 0
    started = time.monotonic()
    for item in inventory:
        rank = int(item["rank"])
        path = Path(str(item["path"]))
        parquet = pq.ParquetFile(path)
        expected_index = 0
        observed_source: str | None = None
        for batch in parquet.iter_batches(
            batch_size=tokenizer_batch_size,
            columns=list(INPUT_COLUMNS),
            use_threads=True,
        ):
            rows = batch.to_pylist()
            texts = [row["text"] for row in rows]
            if any(not isinstance(text, str) or not text for text in texts):
                raise ValueError(f"rank {rank} contains empty/non-string candidate text")
            encoded = tokenizer.encode(texts, num_threads=tokenizer_threads)
            if not isinstance(encoded, list) or len(encoded) != len(rows):
                raise ValueError("tokenizer returned an invalid batch")
            for row, token_ids in zip(rows, encoded, strict=True):
                if (
                    row["candidate_rank"] != rank
                    or row["candidate_doc_index"] != expected_index
                ):
                    raise ValueError(f"rank {rank} candidate row identity/order drift")
                expected_index += 1
                source_id = row["source_id"]
                if not isinstance(source_id, str) or not source_id:
                    raise ValueError(f"rank {rank} contains invalid source_id")
                if observed_source is None:
                    observed_source = source_id
                elif source_id != observed_source:
                    raise ValueError(f"rank {rank} contains multiple source IDs")
                if source_id != source_by_rank[rank].get("source_id"):
                    raise ValueError(
                        f"rank {rank} candidate source differs from sealed source-plan object"
                    )
                _validate_candidate_lid_fields(row, source_id, policy)
                lid_rows_validated += 1
                text = row["text"]
                text_sha = canonical_text_hash(text)
                if row["dedup_cluster_id"] != text_sha:
                    raise ValueError(f"rank {rank} exact canonical-text hash drift")
                selected = select_mixture_bucket(source_id, row, policy)
                if selected is None:
                    raise ValueError(
                        f"rank {rank} candidate cannot be routed by the frozen mixture policy"
                    )
                mixture_id = str(selected[0])
                split = assign_split(text_sha, policy["splits"])
                register_bucket = dominant_register(row)
                tokens_with_bos = len(token_ids) + 1
                locator = (
                    f"{rank}\0{expected_index - 1}\0{source_id}\0{row['document_id']}\0"
                    f"{text_sha}\n"
                ).encode("utf-8")
                input_sequence.update(locator)
                placeholder_dedup_keep[str(row.get("dedup_keep"))] += 1
                placeholder_quality_flags[str(row.get("quality_filter_flags"))] += 1
                counters.add(
                    "assigned",
                    split=split,
                    source_id=source_id,
                    mixture_id=mixture_id,
                    text=text,
                    tokens_with_bos=tokens_with_bos,
                )
                if split == "train":
                    assigned_train_multiset.add(locator)
                if split in {"val", "test"} and tokens_with_bos > eval_max_tokens_with_bos:
                    counters.add(
                        "excluded_eval_oversize",
                        split=split,
                        source_id=source_id,
                        mixture_id=mixture_id,
                        text=text,
                        tokens_with_bos=tokens_with_bos,
                    )
                    excluded_sequence.update(locator)
                    continue
                output_row = dict(row)
                shuffle_key = hashlib.sha256(
                    f"{shuffle_seed}\0{rank}\0{expected_index - 1}\0{text_sha}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                output_row.update(
                    {
                        "mixture_id": mixture_id,
                        "split": split,
                        "canonical_text_sha256": text_sha,
                        "register_bucket": register_bucket,
                        "encoded_tokens_with_bos": tokens_with_bos,
                        "shuffle_key": shuffle_key,
                        "shuffle_bucket": int(shuffle_key[:2], 16),
                    }
                )
                if split == "train":
                    train_stager.add(output_row)
                else:
                    eval_rows[split].append(output_row)
                written_sequence.update(locator)
                counters.add(
                    "written",
                    split=split,
                    source_id=source_id,
                    mixture_id=mixture_id,
                    text=text,
                    tokens_with_bos=tokens_with_bos,
                )
        if expected_index != int(item["rows"]):
            raise ValueError(f"rank {rank} row count differs from its Parquet metadata")
        if observed_source != tokenizer_by_rank[rank].get("source_id"):
            raise ValueError(f"rank {rank} source differs from tokenizer inventory")
    fragments, staging_resources = train_stager.finish()
    train_files, train_shuffle, emitted_train_multiset = emit_globally_shuffled_train(
        temporary,
        output_schema,
        fragments,
        rows_per_train_shard=rows_per_train_shard,
        row_group_rows=row_group_rows,
    )
    if assigned_train_multiset.render() != emitted_train_multiset.render():
        raise ValueError("globally shuffled train output differs as a row multiset")
    for split in ("val", "test"):
        eval_rows[split].sort(key=lambda row: row["shuffle_key"])
        if any(
            left["shuffle_key"] >= right["shuffle_key"]
            for left, right in zip(eval_rows[split], eval_rows[split][1:])
        ):
            raise ValueError(f"{split} shuffle keys are not unique/stably sorted")
        for row in eval_rows[split]:
            writers.add(split, row)
    files = writers.finish(required_splits=("val", "test"))
    files["train"] = train_files
    assigned = counters.render("assigned")
    written = counters.render("written")
    excluded = counters.render("excluded_eval_oversize")
    if (
        assigned["totals"]["documents"]
        != written["totals"]["documents"] + excluded["totals"]["documents"]
        or written["by_split"]["train"]["documents"]
        != assigned["by_split"]["train"]["documents"]
    ):
        raise ValueError("partial row preservation accounting drift")
    strict_ordered = [
        {key: item[key] for key in ("path", "size_bytes", "sha256")}
        for item in sorted(files["train"], key=lambda record: record["path"])
    ] + [
        {
            key: files["val"][0][key]
            for key in ("path", "size_bytes", "sha256")
        }
    ]
    if [item["path"] for item in strict_ordered] != sorted(
        item["path"] for item in strict_ordered
    ):
        raise ValueError("strict dataset order is not lexically sorted")
    if strict_ordered[-1]["path"] != "validation.parquet":
        raise ValueError("validation.parquet must be last in the strict inventory")
    synthetic_revision = hashlib.sha1(
        f"{inventory_sha}:{tokenizer_lineage['tokenizer_package_sha256']}:{policy_sha}".encode(
            "ascii"
        ),
        usedforsecurity=False,
    ).hexdigest()
    dataset_manifest = seal_manifest(
        {
            "schema_version": "1.0",
            "manifest_type": "dataset",
            "profile": "strict",
            "dataset": {
                "repo_id": f"local-composite/{CORPUS_NAME}",
                "path": f"pretrain_data/{CORPUS_NAME}",
                "requested_revision": synthetic_revision,
                "resolved_revision": synthetic_revision,
                "repo_type": "dataset",
            },
            "text_column": "text",
            "ordered_files": strict_ordered,
            "validation_file": "validation.parquet",
            "created_by": {
                "git_commit": git_commit,
                "tool": "scripts.materialize_partial18_turkish_corpus",
                "tool_version": "1",
            },
            "metadata": {
                "profile_semantics": (
                    "strict_file_identity_and_loader_compatibility_only_not_quality_approval"
                ),
                "production_eligible": False,
                "corpus_scope": "18_complete_candidate_parquets_from_cancelled_sample_job",
                "input_inventory_sha256": inventory_sha,
                "policy_sha256": policy_sha,
                "tokenizer_package_sha256": tokenizer_lineage[
                    "tokenizer_package_sha256"
                ],
                "split_policy": policy["splits"],
                "split_hash_semantics": "sha256_of_exact_stored_utf8_text",
                "text_transform": "none_exact_candidate_text_preserved",
                "train_order": {
                    "algorithm": train_shuffle["algorithm"],
                    "seed": shuffle_seed,
                    "key": "sha256_seed_rank_candidate_index_exact_text_sha256",
                },
                "source_plan_sha256": source_evidence["source_plan_sha256"],
                "sample_ranks_file_sha256": source_evidence[
                    "sample_ranks_file_sha256"
                ],
                "validation_policy": {
                    "mode": "whole_document_no_crop",
                    "maximum_tokens_with_bos": eval_max_tokens_with_bos,
                    "oversize_documents": "excluded_from_original_eval_split_not_moved",
                    "excluded": excluded["by_split"]["val"],
                },
                "test_file": "test/test.parquet",
            },
            "canonical_sha256": None,
        }
    )
    validate_dataset_manifest(dataset_manifest, profile="strict")
    write_json_atomic(temporary / "fineweb2_manifest.json", dataset_manifest)
    verify_strict_dataset(temporary, verify_bytes=True)
    capacity_started = time.monotonic()
    capacity = simulate_partial_first_epoch_capacity(
        temporary,
        files["train"],
        world_sizes=capacity_world_sizes,
        B=capacity_B,
        T=capacity_T,
        buffer_size=capacity_buffer_size,
        tokenizer_batch_size=tokenizer_batch_size,
        global_batch_tokens=capacity_global_batch_tokens,
        verify_parity=verify_capacity_parity,
    )
    capacity_receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_partial18_first_epoch_bestfit_capacity",
            "production_eligible": False,
            "dataset_manifest_sha256": dataset_manifest["canonical_sha256"],
            "input_inventory_sha256": inventory_sha,
            "tokenizer_package_sha256": tokenizer_lineage[
                "tokenizer_package_sha256"
            ],
            "full_train_source_tokens_with_bos": written["by_split"]["train"][
                "tokens_with_bos"
            ],
            "full_train_documents": written["by_split"]["train"]["documents"],
            "simulation": capacity,
            "capacity_semantics": (
                "actual_pinned_bestfit_complete_first_epoch_common_prefix_with_"
                "cropped_tails_and_optimizer_alignment"
            ),
            "later_repetition_simulation_ready": True,
            "train_files": files["train"],
            "canonical_sha256": None,
        }
    )
    write_json_atomic(temporary / CAPACITY_RECEIPT_FILE, capacity_receipt)
    gates = {
        "source_language_id_end_to_end_attested": False,
        "independent_glotlid_end_to_end_attested": False,
        "global_minhash_completed": False,
        "official_gopher_filters_completed": False,
        "official_fineweb_filters_completed": False,
        "local_quality_filters_completed": False,
        "pii_filter_completed": False,
        "code_filter_completed": False,
        "manual_qa_completed": False,
        "benchmark_decontamination_completed": False,
    }
    data_files = sorted(
        [*files["train"], *files["val"], *files["test"]],
        key=lambda record: record["path"],
    )
    missing_object_receipts = [
        int(item["rank"])
        for item in inventory
        if item["object_receipt_present"] is not True
    ]
    tokenizer_lineage_public = {
        key: value
        for key, value in tokenizer_lineage.items()
        if key != "tokenizer_input_inventory"
    }
    corpus_manifest = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_partial18_as_is_materialized_corpus",
            "name": CORPUS_NAME,
            "production_eligible": False,
            "input_root": str(input_root.resolve()),
            "expected_ranks": list(expected_ranks),
            "input_inventory": inventory,
            "input_inventory_sha256": inventory_sha,
            "policy_path": str(policy_path.resolve()),
            "policy_file_sha256": file_sha256(policy_path),
            "policy_sha256": policy_sha,
            "source_evidence": dict(source_evidence),
            "source_job_provenance": dict(source_job_provenance),
            "tokenizer_lineage": tokenizer_lineage_public,
            "candidate_gate_status": gates,
            "candidate_stage_row_field_evidence": {
                "rows_validated": lid_rows_validated,
                "all_rows_match_frozen_source_lid_labels_and_thresholds": True,
                "all_rows_match_frozen_independent_glotlid_labels_and_thresholds": True,
                "semantics": (
                    "semantic_revalidation_of_stored_candidate_fields_not_a_"
                    "replacement_for_complete_sealed_object_receipts"
                ),
                "sealed_object_receipts_present": len(inventory)
                - len(missing_object_receipts),
                "sealed_object_receipts_missing_ranks": missing_object_receipts,
                "validated_object_receipt_attestations": object_receipt_attestations,
            },
            "candidate_placeholder_fields": {
                "dedup_keep": {
                    "used_for_admission_or_dedup": False,
                    "observed_value_counts": dict(sorted(placeholder_dedup_keep.items())),
                },
                "quality_filter_flags": {
                    "used_as_quality_evidence_or_filter": False,
                    "observed_value_counts": dict(
                        sorted(placeholder_quality_flags.items())
                    ),
                },
                "warning": (
                    "candidate placeholders do not prove global deduplication or any "
                    "quality/PII/code gate"
                ),
            },
            "row_policy": {
                "train": "all_exact-hash-assigned_rows_preserved_without_filter_or_dedup",
                "evaluation": (
                    "all_exact-hash-assigned_rows_preserved_except_explicit_whole-"
                    "document_token_limit_exclusions"
                ),
                "text_transform": "none",
                "document_character_cap": None,
                "split_hash": "sha256_of_exact_stored_utf8_text",
                "duplicates": "retained_and_exact_duplicates_share_a_split_by_hash",
                "train_order": {
                    **train_shuffle,
                    "seed": shuffle_seed,
                    "shuffle_key": (
                        "sha256(seed_nul_candidate_rank_nul_candidate_doc_index_"
                        "nul_exact_text_sha256)"
                    ),
                    "staging_resources": staging_resources,
                    "staging_removed_after_verified_emission": True,
                    "full_shard_row_groups_divisible_by_16": True,
                },
                "evaluation_order": "ascending_same_unique_shuffle_key",
            },
            "counts": {
                "assigned": assigned,
                "written": written,
                "excluded_eval_oversize": excluded,
                "realized_train_mix": _realized_train_mix(written, policy),
            },
            "sequence_hashes": {
                "all_assigned_rows": input_sequence.hexdigest(),
                "written_rows": written_sequence.hexdigest(),
                "excluded_eval_oversize_rows": excluded_sequence.hexdigest(),
                "semantics": "candidate_rank_index_source_document_id_text_sha256",
                "assigned_train_row_multiset": assigned_train_multiset.render(),
                "emitted_train_row_multiset": emitted_train_multiset.render(),
            },
            "output_data_files": data_files,
            "strict_dataset_manifest": _file_record(
                temporary / "fineweb2_manifest.json", temporary
            )
            | {"canonical_sha256": dataset_manifest["canonical_sha256"]},
            "capacity_receipt": _file_record(
                temporary / CAPACITY_RECEIPT_FILE, temporary
            )
            | {"canonical_sha256": capacity_receipt["canonical_sha256"]},
            "canonical_sha256": None,
        }
    )
    write_json_atomic(temporary / CORPUS_MANIFEST_FILE, corpus_manifest)
    materialization_receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_partial18_materialization_receipt",
            "production_eligible": False,
            "producer_git_commit": git_commit,
            "producer_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_job_provenance": dict(source_job_provenance),
            "input_inventory_sha256": inventory_sha,
            "tokenizer_package_sha256": tokenizer_lineage[
                "tokenizer_package_sha256"
            ],
            "strict_dataset_manifest_sha256": dataset_manifest["canonical_sha256"],
            "corpus_manifest": _file_record(
                temporary / CORPUS_MANIFEST_FILE, temporary
            )
            | {"canonical_sha256": corpus_manifest["canonical_sha256"]},
            "capacity_receipt_sha256": capacity_receipt["canonical_sha256"],
            "input_documents": assigned["totals"]["documents"],
            "written_documents": written["totals"]["documents"],
            "excluded_eval_oversize_documents": excluded["totals"]["documents"],
            "train_tokens_with_bos": written["by_split"]["train"]["tokens_with_bos"],
            "missing_production_gates": sorted(
                key for key, value in gates.items() if value is False
            ),
            "runtime_telemetry": {
                "materialization_seconds": time.monotonic() - started,
                "capacity_simulation_seconds": time.monotonic() - capacity_started,
                "staging_peak_buffered_rows": staging_resources[
                    "peak_buffered_rows"
                ],
                "staging_peak_buffered_characters": staging_resources[
                    "peak_buffered_characters"
                ],
            },
            "canonical_sha256": None,
        }
    )
    write_json_atomic(temporary / MATERIALIZATION_RECEIPT_FILE, materialization_receipt)
    verify_final_partial18_output(
        temporary,
        data_files=data_files,
        test_file=files["test"][0],
        expected_manifest_hashes={
            "fineweb2_manifest.json": dataset_manifest["canonical_sha256"],
            CAPACITY_RECEIPT_FILE: capacity_receipt["canonical_sha256"],
            CORPUS_MANIFEST_FILE: corpus_manifest["canonical_sha256"],
            MATERIALIZATION_RECEIPT_FILE: materialization_receipt["canonical_sha256"],
        },
    )
    os.replace(temporary, output_dir)
    return {
        "output_dir": str(output_dir),
        "corpus_manifest_sha256": corpus_manifest["canonical_sha256"],
        "dataset_manifest_sha256": dataset_manifest["canonical_sha256"],
        "capacity_receipt_sha256": capacity_receipt["canonical_sha256"],
        "materialization_receipt_sha256": materialization_receipt["canonical_sha256"],
        "input_documents": assigned["totals"]["documents"],
        "written_documents": written["totals"]["documents"],
        "train_tokens_with_bos": written["by_split"]["train"]["tokens_with_bos"],
        "production_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = load_corpus_policy(args.policy)
        if (
            args.output_dir.name != CORPUS_NAME
            or args.output_dir.parent.name != "pretrain_data"
        ):
            raise ValueError(
                f"partial corpus output must use pretrain_data/{CORPUS_NAME}"
            )
        source_evidence = verify_source_evidence(
            args.source_plan,
            args.sample_ranks,
            policy=policy,
            expected_ranks=args.expected_ranks,
            expected_source_plan_sha256=args.expected_source_plan_sha256,
        )
        tokenizer, tokenizer_lineage = verify_partial_tokenizer(
            args.tokenizer_dir,
            expected_package_sha256=args.expected_tokenizer_package_sha256,
            expected_receipt_sha256=args.expected_tokenizer_receipt_sha256,
            expected_ranks=args.expected_ranks,
            expected_source_job_id=args.source_cancelled_job_id,
            expected_tokenizer_job_id=args.tokenizer_job_id,
        )
        source_job_provenance = {
            "job_id": args.source_cancelled_job_id,
            "state": args.source_job_state,
            "partition": args.source_job_partition,
            "allocated_cpus": args.source_job_allocated_cpus,
            "verification_authority": "sacct_snapshot_validated_by_uhem_wrapper",
            "candidate_stage": "object_candidates_before_global_minhash",
        }
        if (
            not args.source_job_state.startswith("CANCELLED")
            or args.source_job_partition != "cpu2dq"
            or args.source_job_allocated_cpus != 128
        ):
            raise ValueError("cancelled source-job provenance differs from the pinned run")
        result = materialize_partial18(
            input_root=args.input_root,
            output_dir=args.output_dir,
            policy=policy,
            policy_path=args.policy,
            tokenizer=tokenizer,
            tokenizer_lineage=tokenizer_lineage,
            source_evidence=source_evidence,
            expected_ranks=args.expected_ranks,
            git_commit=_git_commit(),
            source_job_provenance=source_job_provenance,
            rows_per_train_shard=args.rows_per_train_shard,
            row_group_rows=args.row_group_rows,
            tokenizer_batch_size=args.tokenizer_batch_size,
            tokenizer_threads=args.tokenizer_threads,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
