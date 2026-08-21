"""Train an explicitly non-production Turkish BPE from cancelled sample outputs.

This is a salvage path for complete candidate Parquets left by a cancelled
``turkish_data_objects_packed_sample`` allocation.  It deliberately does not
pretend that the candidates passed the later global MinHash cluster, pool QA,
or production-corpus gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from nanochat.experiment_manifest import (
    canonical_json,
    file_sha256,
    seal_manifest,
    write_json_atomic,
)
from nanochat.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN, RustBPETokenizer
from nanochat.turkish_corpus import (
    assign_split,
    load_corpus_policy,
    select_mixture_bucket,
)
from scripts.train_turkish_raw_bpe import (
    _export_and_verify_canonical_tiktoken,
    _token_byte_lengths,
    _torch_save_atomic,
    _validate_tokenizer,
    run_pinned_iterator_parity_fixture,
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
PARTIAL_TRAINING_RECEIPT_KIND = "turkish_raw_bpe_partial_training_receipt"
PARTIAL_TOKENIZER_PACKAGE_KIND = "turkish_raw_bpe_partial_tokenizer_package"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RowGroupRef:
    path: Path
    rank: int
    row_group: int
    source_id: str


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--name", default="tr_general_raw_bpe_32k_partial18_v1"
    )
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--max-chars", type=int, default=2_000_000_000)
    parser.add_argument("--doc-cap", type=int, default=10_000)
    parser.add_argument(
        "--expected-ranks",
        type=_parse_ranks,
        default=DEFAULT_EXPECTED_RANKS,
    )
    parser.add_argument("--cancelled-job-id", type=int, required=True)
    return parser


def _candidate_paths(input_root: Path, expected_ranks: tuple[int, ...]) -> list[Path]:
    objects = input_root / "objects"
    paths = sorted(objects.glob("[0-9][0-9][0-9][0-9][0-9]/candidates.parquet"))
    ranks = tuple(int(path.parent.name) for path in paths)
    if ranks != expected_ranks:
        raise ValueError(
            f"candidate rank inventory differs: expected={expected_ranks}, actual={ranks}"
        )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("candidate inventory contains a symlink or non-regular file")
    return paths


def _source_for_file(parquet: pq.ParquetFile, *, rank: int) -> str:
    required = {
        "text",
        "source_id",
        "document_id",
        "dedup_cluster_id",
        "dedup_keep",
        "quality_filter_flags",
        "genre",
        "candidate_rank",
        "candidate_doc_index",
    }
    columns = set(parquet.schema_arrow.names)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"rank {rank} candidate schema is missing {missing}")
    if parquet.metadata.num_rows <= 0 or parquet.num_row_groups <= 0:
        raise ValueError(f"rank {rank} candidate is empty")
    first = parquet.read_row_group(0, columns=["source_id", "candidate_rank"]).slice(0, 1)
    row = first.to_pylist()[0]
    if row["candidate_rank"] != rank or not isinstance(row["source_id"], str):
        raise ValueError(f"rank {rank} candidate identity is invalid")
    return row["source_id"]


def build_inventory(
    input_root: Path,
    expected_ranks: tuple[int, ...],
    policy: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, list[RowGroupRef]]]:
    buckets = {str(item["id"]): item for item in policy["mixture"]}  # type: ignore[index]
    by_source: dict[str, list[str]] = {}
    for mixture_id, bucket in buckets.items():
        by_source.setdefault(str(bucket["source_id"]), []).append(mixture_id)
    inventory: list[dict[str, object]] = []
    schedule = {mixture_id: [] for mixture_id in buckets}
    for path in _candidate_paths(input_root, expected_ranks):
        rank = int(path.parent.name)
        parquet = pq.ParquetFile(path)
        source_id = _source_for_file(parquet, rank=rank)
        mixture_ids = by_source.get(source_id)
        if not mixture_ids:
            raise ValueError(f"rank {rank} has source outside policy: {source_id}")
        receipt_path = path.parent / "object_receipt.json"
        inventory.append(
            {
                "rank": rank,
                "source_id": source_id,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.num_row_groups,
                "object_receipt_present": receipt_path.is_file(),
                "object_receipt_sha256": (
                    file_sha256(receipt_path) if receipt_path.is_file() else None
                ),
            }
        )
        for mixture_id in mixture_ids:
            schedule[mixture_id].extend(
                RowGroupRef(path, rank, row_group, source_id)
                for row_group in range(parquet.num_row_groups)
            )
    seed = str(policy["tokenizer_training"]["sampling_seed"])  # type: ignore[index]
    for mixture_id, refs in schedule.items():
        refs.sort(
            key=lambda ref: hashlib.sha256(
                f"{seed}\0{mixture_id}\0{ref.rank}\0{ref.row_group}".encode()
            ).digest()
        )
    return inventory, schedule


def _mixture_rows(
    refs: list[RowGroupRef],
    mixture_id: str,
    policy: Mapping[str, object],
    counters: Counter[str],
) -> Iterator[tuple[str, str, int, int]]:
    columns = [
        "text",
        "source_id",
        "dedup_cluster_id",
        "dedup_keep",
        "quality_filter_flags",
        "genre",
        "candidate_rank",
        "candidate_doc_index",
    ]
    readers: dict[Path, pq.ParquetFile] = {}
    for ref in refs:
        reader = readers.get(ref.path)
        if reader is None:
            reader = pq.ParquetFile(ref.path)
            readers[ref.path] = reader
        rows = reader.read_row_group(ref.row_group, columns=columns).to_pylist()
        counters[f"row_groups_read:{mixture_id}"] += 1
        for row in rows:
            if row["source_id"] != ref.source_id or row["candidate_rank"] != ref.rank:
                raise ValueError(f"candidate row identity drift at rank {ref.rank}")
            if row["dedup_keep"] is not True or row["quality_filter_flags"] != "[]":
                raise ValueError(f"non-candidate row found at rank {ref.rank}")
            selected = select_mixture_bucket(ref.source_id, row, policy)
            if selected is None or selected[0] != mixture_id:
                continue
            cluster_id = str(row["dedup_cluster_id"])
            if not _SHA256_RE.fullmatch(cluster_id):
                raise ValueError(f"invalid exact-cluster ID at rank {ref.rank}")
            split = assign_split(cluster_id, policy["splits"])  # type: ignore[index]
            counters[f"encountered:{mixture_id}:{split}"] += 1
            if split != "train":
                continue
            text = row["text"]
            if not isinstance(text, str) or not text:
                raise ValueError(f"empty/non-string candidate text at rank {ref.rank}")
            yield text, cluster_id, ref.rank, int(row["candidate_doc_index"])


def weighted_training_texts(
    schedule: Mapping[str, list[RowGroupRef]],
    policy: Mapping[str, object],
    *,
    max_chars: int,
    doc_cap: int,
    stats: dict[str, object],
) -> Iterator[str]:
    weights = {
        str(bucket["id"]): float(bucket["weight"])
        for bucket in policy["mixture"]  # type: ignore[index]
    }
    counters: Counter[str] = stats["counters"]  # type: ignore[assignment]
    iterators = {
        mixture_id: _mixture_rows(refs, mixture_id, policy, counters)
        for mixture_id, refs in schedule.items()
    }
    active = set(iterators)
    emitted: Counter[str] = stats["characters_by_mixture"]  # type: ignore[assignment]
    documents: Counter[str] = stats["documents_by_mixture"]  # type: ignore[assignment]
    sequence: hashlib._Hash = stats["sequence_hash"]  # type: ignore[attr-defined,assignment]
    total = 0
    while active:
        choice = max(
            active,
            key=lambda key: (weights[key] * max(1, total) - emitted[key], key),
        )
        try:
            raw_text, cluster_id, rank, document_index = next(iterators[choice])
        except StopIteration:
            active.remove(choice)
            continue
        text = raw_text[:doc_cap]
        if not text:
            continue
        encoded_identity = f"{choice}\0{rank}\0{document_index}\0{cluster_id}\0{len(text)}\n"
        sequence.update(encoded_identity.encode("utf-8"))
        emitted[choice] += len(text)
        documents[choice] += 1
        total += len(text)
        stats["characters"] = total
        stats["documents"] = int(stats["documents"]) + 1
        yield text
        if total > max_chars:
            return
    raise ValueError(
        f"partial candidates exhausted before tokenizer threshold: {total} <= {max_chars}"
    )


def _distribution(stats: Mapping[str, object], weights: Mapping[str, float]) -> list[dict[str, object]]:
    chars: Counter[str] = stats["characters_by_mixture"]  # type: ignore[assignment]
    docs: Counter[str] = stats["documents_by_mixture"]  # type: ignore[assignment]
    total = int(stats["characters"])
    return [
        {
            "mixture_id": key,
            "target_share": weights[key],
            "documents": docs[key],
            "characters": chars[key],
            "realized_share": chars[key] / total,
        }
        for key in sorted(weights)
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.vocab_size != 32768:
            raise ValueError("this d32 salvage tokenizer requires vocab_size=32768")
        if args.max_chars <= 0 or args.doc_cap <= 0:
            raise ValueError("max-chars and doc-cap must be positive")
        policy = load_corpus_policy(args.policy)
        if args.output_dir.exists():
            raise FileExistsError(f"refusing existing output path: {args.output_dir}")
        inventory, schedule = build_inventory(args.input_root, args.expected_ranks, policy)
        weights = {
            str(bucket["id"]): float(bucket["weight"])
            for bucket in policy["mixture"]
        }
        stats: dict[str, object] = {
            "documents": 0,
            "characters": 0,
            "documents_by_mixture": Counter(),
            "characters_by_mixture": Counter(),
            "counters": Counter(),
            "sequence_hash": hashlib.sha256(),
        }
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_dir.parent / (
            f".{args.output_dir.name}.job{os.environ.get('SLURM_JOB_ID', 'local')}.tmp"
        )
        if temporary.exists():
            raise FileExistsError(f"refusing existing temporary output: {temporary}")
        temporary.mkdir()
        started = time.monotonic()
        tokenizer = RustBPETokenizer.train_from_iterator(
            weighted_training_texts(
                schedule,
                policy,
                max_chars=args.max_chars,
                doc_cap=args.doc_cap,
                stats=stats,
            ),
            args.vocab_size,
        )
        train_seconds = time.monotonic() - started
        realized = int(stats["characters"])
        if not args.max_chars < realized <= args.max_chars + args.doc_cap:
            raise ValueError("trainer-visible character threshold drift")
        validation = _validate_tokenizer(tokenizer)
        tokenizer.save(str(temporary))
        token_bytes = _token_byte_lengths(tokenizer)
        _torch_save_atomic(token_bytes, temporary / "token_bytes.pt")
        canonical_export = _export_and_verify_canonical_tiktoken(
            tokenizer, temporary, token_bytes, tokenizer_name=args.name
        )
        validation["canonical_export"] = canonical_export
        distribution = _distribution(stats, weights)
        if any(
            abs(float(item["realized_share"]) - float(item["target_share"]))
            > 0.0001
            for item in distribution
        ):
            raise ValueError("partial tokenizer sample mixture share drift")
        input_inventory_sha = hashlib.sha256(
            canonical_json(inventory).encode("utf-8")
        ).hexdigest()
        config = {
            "schema_version": "1.0",
            "name": args.name,
            "implementation": "raw_byte_bpe",
            "vocab_size": args.vocab_size,
            "split_pattern": SPLIT_PATTERN,
            "special_tokens": list(SPECIAL_TOKENS),
            "max_chars": args.max_chars,
            "realized_training_characters": realized,
            "doc_cap": args.doc_cap,
            "sample_distribution": distribution,
            "production_eligible": False,
            "requires_runtime_segmentation": False,
            "decode_strip": "",
            "canonical_export": canonical_export,
        }
        write_json_atomic(temporary / "tokenizer_config.json", config)
        payload_names = (
            "tokenizer.pkl",
            "tokenizer.tiktoken",
            "token_bytes.pt",
            "tokenizer_config.json",
        )
        payload = [
            {
                "path": name,
                "size_bytes": (temporary / name).stat().st_size,
                "sha256": file_sha256(temporary / name),
            }
            for name in payload_names
        ]
        receipt = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": PARTIAL_TRAINING_RECEIPT_KIND,
                "name": args.name,
                "vocab_size": args.vocab_size,
                "algorithm": "raw_byte_bpe",
                "git_commit": _git_commit(),
                "producer_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "cancelled_source_job_id": args.cancelled_job_id,
                "input_root": str(args.input_root.resolve()),
                "input_inventory": inventory,
                "input_inventory_sha256": input_inventory_sha,
                "expected_ranks": list(args.expected_ranks),
                "object_receipts_present": sum(
                    bool(item["object_receipt_present"]) for item in inventory
                ),
                "object_receipts_missing": [
                    item["rank"]
                    for item in inventory
                    if not item["object_receipt_present"]
                ],
                "training_documents": int(stats["documents"]),
                "training_characters": realized,
                "requested_max_characters": args.max_chars,
                "terminal_overshoot_characters": realized - args.max_chars,
                "max_chars_per_document": args.doc_cap,
                "sampling": "weighted_deficit_stable_rowgroup_shuffle_partial18_v1",
                "sampling_seed": policy["tokenizer_training"]["sampling_seed"],
                "training_sequence_sha256": stats["sequence_hash"].hexdigest(),
                "sample_distribution": distribution,
                "traversal_counters": dict(sorted(stats["counters"].items())),
                "split_policy": policy["splits"],
                "tokenizer_train_split": "train_only_by_exact_text_cluster_hash",
                "policy_sha256": hashlib.sha256(
                    canonical_json(policy).encode("utf-8")
                ).hexdigest(),
                "pinned_iterator_parity": run_pinned_iterator_parity_fixture(),
                "train_time_seconds": train_seconds,
                "validation": validation,
                "payload": payload,
                "production_eligible": False,
                "global_near_dedup_completed": False,
                "sample_cluster_completed": False,
                "manual_corpus_qa_completed": False,
                "caveat": (
                    "Complete object-level candidate Parquets from 18 sampled ranks; "
                    "the source job was cancelled before object receipts/global MinHash "
                    "clustering and manual corpus QA completed."
                ),
                "canonical_sha256": None,
            }
        )
        write_json_atomic(temporary / "partial_training_receipt.json", receipt)
        package = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": PARTIAL_TOKENIZER_PACKAGE_KIND,
                "name": args.name,
                "vocab_size": args.vocab_size,
                "training_receipt_sha256": receipt["canonical_sha256"],
                "input_inventory_sha256": input_inventory_sha,
                "production_eligible": False,
                "files": payload
                + [
                    {
                        "path": "partial_training_receipt.json",
                        "size_bytes": (temporary / "partial_training_receipt.json").stat().st_size,
                        "sha256": file_sha256(
                            temporary / "partial_training_receipt.json"
                        ),
                    }
                ],
                "canonical_sha256": None,
            }
        )
        write_json_atomic(temporary / "package_manifest.json", package)
        os.replace(temporary, args.output_dir)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "name": args.name,
                    "package_sha256": package["canonical_sha256"],
                    "training_characters": realized,
                    "production_eligible": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
