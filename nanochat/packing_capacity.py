"""Finite, bounded-memory simulation of Nanochat's upstream crop loader.

The production corpus is useful only if *model token positions*, rather than
raw encoded source tokens, remain unique through the 40x horizon.  Nanochat's
pinned BOS best-fit loader discards document tails, refills its 1,000-document
buffer in 128-document tokenizer batches, and shards Parquet row groups by
rank.  This module mirrors those details and stops at either the required
horizon or the first attempted epoch wrap.  It never retains per-microbatch
state in production; selection traces are opt-in and limited to parity tests.
"""

from __future__ import annotations

import hashlib
import ast
import math
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow.parquet as pq

from nanochat.experiment_manifest import (
    canonical_json_bytes,
    file_sha256,
    seal_manifest,
    write_json_atomic,
)


PACKING_SIMULATOR_ID = "nanochat_upstream_bos_bestfit_crop_capacity_v2"
PINNED_UPSTREAM_REVISION = "92d63d4e"
PINNED_UPSTREAM_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
PINNED_BESTFIT_FUNCTION_SHA256 = (
    "ed1d4997e3c407f242fbbcfa627f2987f8f34a3a0c340d792c4e10a202981990"
)


@dataclass(frozen=True)
class PackingDocument:
    tokens_with_bos: int
    document_id: str
    mixture_id: str
    source_id: str
    register_bucket: str

    def __post_init__(self) -> None:
        if self.tokens_with_bos <= 0:
            raise ValueError("packing documents require at least BOS")


def _metric_counters() -> dict[str, Counter[str]]:
    return {
        "mixture": Counter(),
        "source": Counter(),
        "register": Counter(),
    }


def _label_add(
    metrics: dict[str, Counter[str]], document: PackingDocument, value: int
) -> None:
    if value <= 0:
        return
    metrics["mixture"][document.mixture_id] += value
    metrics["source"][document.source_id] += value
    metrics["register"][document.register_bucket] += value


def _merge_metric_counters(
    target: dict[str, Counter[str]], source: Mapping[str, Mapping[str, int]]
) -> None:
    for dimension in target:
        for key, value in source[dimension].items():
            target[dimension][str(key)] += int(value)


def _plain_metrics(metrics: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {
        dimension: dict(sorted(counter.items()))
        for dimension, counter in metrics.items()
    }


def simulate_bestfit_rank(
    document_batches: Iterable[Sequence[PackingDocument]],
    *,
    B: int,
    T: int,
    buffer_size: int = 1000,
    max_microbatches: int | None = None,
    collect_trace: bool = False,
) -> dict[str, Any]:
    """Simulate one rank with memory independent of corpus size.

    ``max_microbatches`` right-censors a passing rank at the exact gate
    horizon.  With ``None``, the simulator runs until refilling would fetch an
    epoch-2 document.  Production callers do not request traces.
    """

    if min(B, T, buffer_size) <= 0:
        raise ValueError("B/T/buffer_size must be positive")
    if max_microbatches is not None and (
        isinstance(max_microbatches, bool)
        or not isinstance(max_microbatches, int)
        or max_microbatches < 0
    ):
        raise ValueError("max_microbatches must be a non-negative integer or None")

    batches = iter(document_batches)
    buffer: list[PackingDocument] = []
    retained = _metric_counters()
    cropped = _metric_counters()
    consumed = _metric_counters()
    row_leading = _metric_counters()
    completed = 0
    loaded_documents = 0
    loaded_source_tokens = 0
    refill_batches = 0
    retained_source_elements = 0
    cropped_tokens = 0
    row_leading_elements = 0
    documents_completed = 0
    documents_cropped = 0
    traces: list[dict[str, Any]] = []

    def result(
        stop_reason: str,
        *,
        incomplete_source_elements: int = 0,
        incomplete_cropped_tokens: int = 0,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "completed_microbatches": completed,
            "scheduled_token_positions": completed * B * T,
            "loaded_documents": loaded_documents,
            "loaded_source_tokens_with_bos": loaded_source_tokens,
            "buffered_documents_at_stop": len(buffer),
            "buffered_source_tokens_at_stop": sum(
                item.tokens_with_bos for item in buffer
            ),
            "refill_batches": refill_batches,
            "stop_reason": stop_reason,
            "first_wrap_before_microbatch": (
                completed if stop_reason == "first_epoch_exhausted_during_refill" else None
            ),
            "right_censored_at_requested_horizon": stop_reason == "requested_horizon_reached",
            "retained_source_elements": retained_source_elements,
            "cropped_source_tokens": cropped_tokens,
            "row_leading_source_elements_not_targets": row_leading_elements,
            "documents_completed": documents_completed,
            "documents_cropped": documents_cropped,
            "retained_positions": _plain_metrics(retained),
            "cropped_tokens": _plain_metrics(cropped),
            "consumed_source_elements": _plain_metrics(consumed),
            "row_leading_source_elements": _plain_metrics(row_leading),
            "incomplete_microbatch_source_elements_at_wrap": incomplete_source_elements,
            "incomplete_microbatch_cropped_tokens_at_wrap": incomplete_cropped_tokens,
        }
        if collect_trace:
            value["microbatches"] = traces
        return value

    if max_microbatches == 0:
        return result("requested_horizon_reached")

    while True:
        batch_retained = _metric_counters()
        batch_cropped = _metric_counters()
        batch_consumed = _metric_counters()
        batch_row_leading = _metric_counters()
        batch_source_elements = 0
        batch_cropped_tokens = 0
        batch_documents_completed = 0
        batch_documents_cropped = 0
        batch_row_leading_elements = 0
        batch_trace: list[dict[str, Any]] = []
        for row_index in range(B):
            position = 0
            while position < T + 1:
                # The upstream loader refills *before every selection* and a
                # tokenizer batch may overshoot buffer_size.
                while len(buffer) < buffer_size:
                    try:
                        incoming = list(next(batches))
                    except StopIteration:
                        return result(
                            "first_epoch_exhausted_during_refill",
                            incomplete_source_elements=batch_source_elements,
                            incomplete_cropped_tokens=batch_cropped_tokens,
                        )
                    if not incoming:
                        continue
                    buffer.extend(incoming)
                    refill_batches += 1
                    loaded_documents += len(incoming)
                    loaded_source_tokens += sum(
                        item.tokens_with_bos for item in incoming
                    )

                remaining = T + 1 - position
                best_index = -1
                best_length = 0
                for index, document in enumerate(buffer):
                    length = document.tokens_with_bos
                    # Strict > is intentional: equal-length ties select the
                    # first buffered document, exactly as pinned upstream.
                    if length <= remaining and length > best_length:
                        best_index = index
                        best_length = length
                if best_index >= 0:
                    selected = best_index
                else:
                    # Python min is stable, so equal shortest ties also select
                    # the first buffered document.
                    selected = min(
                        range(len(buffer)),
                        key=lambda index: buffer[index].tokens_with_bos,
                    )
                document = buffer.pop(selected)
                used = min(remaining, document.tokens_with_bos)
                discarded = document.tokens_with_bos - used
                target_positions = used - 1 if position == 0 else used
                if position == 0:
                    batch_row_leading_elements += 1
                    _label_add(batch_row_leading, document, 1)
                _label_add(batch_retained, document, target_positions)
                _label_add(batch_cropped, document, discarded)
                _label_add(batch_consumed, document, used)
                batch_source_elements += used
                batch_cropped_tokens += discarded
                if discarded:
                    batch_documents_cropped += 1
                else:
                    batch_documents_completed += 1
                if collect_trace:
                    batch_trace.append(
                        {
                            "document_id": document.document_id,
                            "row": row_index,
                            "position": position,
                            "tokens_with_bos": document.tokens_with_bos,
                            "used": used,
                            "discarded": discarded,
                        }
                    )
                position += used

        if sum(batch_retained["mixture"].values()) != B * T:
            raise ValueError("best-fit simulator target attribution drift")
        completed += 1
        retained_source_elements += batch_source_elements
        cropped_tokens += batch_cropped_tokens
        row_leading_elements += batch_row_leading_elements
        documents_completed += batch_documents_completed
        documents_cropped += batch_documents_cropped
        _merge_metric_counters(retained, batch_retained)
        _merge_metric_counters(cropped, batch_cropped)
        _merge_metric_counters(consumed, batch_consumed)
        _merge_metric_counters(row_leading, batch_row_leading)
        if collect_trace:
            traces.append(
                {
                    "scheduled_token_positions": B * T,
                    "selection_trace": batch_trace,
                }
            )
        if max_microbatches is not None and completed >= max_microbatches:
            return result("requested_horizon_reached")


def summarize_world(
    rank_results: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    B: int,
    T: int,
    global_batch_tokens: int,
    required_optimizer_steps: int,
    safety_margin_fraction: float,
    buffer_size: int = 1000,
) -> dict[str, Any]:
    if len(rank_results) != world_size:
        raise ValueError("packing world result count differs from world_size")
    local_microbatch_tokens = B * T
    denominator = world_size * local_microbatch_tokens
    if global_batch_tokens % denominator:
        raise ValueError("global batch is not divisible by rank-local microbatch")
    gradient_accumulation = global_batch_tokens // denominator
    required_with_margin = math.ceil(
        required_optimizer_steps * (1.0 + safety_margin_fraction)
    )
    requested_rank_microbatches = required_with_margin * gradient_accumulation
    completed_by_rank = [int(item["completed_microbatches"]) for item in rank_results]
    all_reached_horizon = all(
        value >= requested_rank_microbatches for value in completed_by_rank
    )
    common_microbatches = min(completed_by_rank)
    common_optimizer_steps = common_microbatches // gradient_accumulation

    retained = _metric_counters()
    cropped = _metric_counters()
    consumed = _metric_counters()
    row_leading = _metric_counters()
    for rank in rank_results:
        _merge_metric_counters(retained, rank["retained_positions"])
        _merge_metric_counters(cropped, rank["cropped_tokens"])
        _merge_metric_counters(consumed, rank["consumed_source_elements"])
        _merge_metric_counters(row_leading, rank["row_leading_source_elements"])
    observed_scheduled = sum(completed_by_rank) * local_microbatch_tokens
    if sum(retained["mixture"].values()) != observed_scheduled:
        raise ValueError("global best-fit retained-position attribution drift")
    retained_total = sum(retained["mixture"].values())
    retained_mix = {
        key: value / retained_total
        for key, value in sorted(retained["mixture"].items())
    } if retained_total else {}
    retention_efficiency: dict[str, float] = {}
    for key in sorted(set(consumed["mixture"]) | set(cropped["mixture"])):
        # Consumed includes retained targets plus one non-target leading source
        # element per row.  Cropped tails are source tokens unavailable to the
        # optimizer and therefore belong in the denominator.
        source_cost = consumed["mixture"][key] + cropped["mixture"][key]
        if source_cost:
            retention_efficiency[key] = retained["mixture"][key] / source_cost

    return {
        "world_size": world_size,
        "device_batch_sequences": B,
        "max_seq_len": T,
        "buffer_size": buffer_size,
        "preserve_document_tails": False,
        "row_capacity": T + 1,
        "rank_sharding": "parquet_row_group_index_mod_world_size",
        "gradient_accumulation_steps": gradient_accumulation,
        "requested_microbatches_per_rank": requested_rank_microbatches,
        "completed_microbatches_by_rank": completed_by_rank,
        "first_wrap_before_microbatch_by_rank": [
            item["first_wrap_before_microbatch"] for item in rank_results
        ],
        "first_wrap_observation": (
            "right_censored_at_required_horizon"
            if all_reached_horizon
            else "observed_on_at_least_one_rank"
        ),
        "common_no_wrap_microbatches_per_rank": common_microbatches,
        "common_no_wrap_optimizer_steps": common_optimizer_steps,
        "required_optimizer_steps": required_optimizer_steps,
        "safety_margin_fraction": safety_margin_fraction,
        "required_optimizer_steps_with_margin": required_with_margin,
        "required_positions_with_margin": required_with_margin * global_batch_tokens,
        "passes_40x_no_wrap_with_margin": all_reached_horizon,
        "observed_scheduled_positions_across_rank_horizons": observed_scheduled,
        "common_prefix_scheduled_positions": (
            common_optimizer_steps * global_batch_tokens
        ),
        "safe_global_scheduled_positions": (
            required_with_margin * global_batch_tokens
            if all_reached_horizon
            else common_optimizer_steps * global_batch_tokens
        ),
        "safe_global_scheduled_positions_semantics": (
            "right_censored_proven_lower_bound_at_required_horizon"
            if all_reached_horizon
            else "exact_complete_optimizer-step_prefix_before_first_wrap"
        ),
        "aggregate_scope": (
            "exact_common_required_horizon_all_ranks"
            if all_reached_horizon
            else "diagnostic_variable_rank_horizons_not_valid_for_mix_gate"
        ),
        "retained_source_elements": sum(
            int(item["retained_source_elements"]) for item in rank_results
        ),
        "row_leading_source_elements_not_targets": sum(
            int(item["row_leading_source_elements_not_targets"])
            for item in rank_results
        ),
        "cropped_source_tokens": sum(
            int(item["cropped_source_tokens"]) for item in rank_results
        ),
        "documents_completed": sum(
            int(item["documents_completed"]) for item in rank_results
        ),
        "documents_cropped": sum(
            int(item["documents_cropped"]) for item in rank_results
        ),
        "retained_positions_by_mixture": dict(sorted(retained["mixture"].items())),
        "retained_positions_by_source": dict(sorted(retained["source"].items())),
        "retained_positions_by_register": dict(sorted(retained["register"].items())),
        "cropped_tokens_by_mixture": dict(sorted(cropped["mixture"].items())),
        "cropped_tokens_by_source": dict(sorted(cropped["source"].items())),
        "cropped_tokens_by_register": dict(sorted(cropped["register"].items())),
        "consumed_source_elements_by_mixture": dict(
            sorted(consumed["mixture"].items())
        ),
        "consumed_source_elements_by_source": dict(sorted(consumed["source"].items())),
        "consumed_source_elements_by_register": dict(
            sorted(consumed["register"].items())
        ),
        "realized_retained_mix": retained_mix,
        "retention_efficiency_by_mixture": retention_efficiency,
        "rank_diagnostics": [
            {
                key: value
                for key, value in rank.items()
                if key
                in {
                    "completed_microbatches",
                    "loaded_documents",
                    "loaded_source_tokens_with_bos",
                    "buffered_documents_at_stop",
                    "buffered_source_tokens_at_stop",
                    "refill_batches",
                    "stop_reason",
                    "first_wrap_before_microbatch",
                    "incomplete_microbatch_source_elements_at_wrap",
                    "incomplete_microbatch_cropped_tokens_at_wrap",
                }
            }
            for rank in rank_results
        ],
    }


def _rank_document_batches(
    root: Path,
    train_files: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    rank: int,
    world_size: int,
    tokenizer_batch_size: int,
) -> Iterator[list[PackingDocument]]:
    bos_token = tokenizer.get_bos_token_id()
    for file_record in train_files:
        path = root / str(file_record["path"])
        parquet = pq.ParquetFile(path)
        needed = {
            "text",
            "document_id",
            "mixture_id",
            "source_id",
            "register_bucket",
        }
        missing = needed - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"packing simulation schema drift: {sorted(missing)}")
        row_group_index = rank
        while row_group_index < parquet.num_row_groups:
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
                batch_rows = rows[offset : offset + tokenizer_batch_size]
                texts = [str(row["text"]) for row in batch_rows]
                # This is deliberately the exact pinned-upstream call shape.
                encoded = tokenizer.encode(
                    texts,
                    prepend=bos_token,
                    num_threads=4,
                )
                if not isinstance(encoded, list) or len(encoded) != len(batch_rows):
                    raise ValueError("packing tokenizer returned an invalid batch")
                yield [
                    PackingDocument(
                        tokens_with_bos=len(tokens),
                        document_id=str(row["document_id"]),
                        mixture_id=str(row["mixture_id"]),
                        source_id=str(row["source_id"]),
                        register_bucket=str(
                            row.get("register_bucket") or "not_applicable"
                        ),
                    )
                    for row, tokens in zip(batch_rows, encoded, strict=True)
                ]
            row_group_index += world_size


def run_upstream_loader_parity_fixture() -> dict[str, Any]:
    """Execute a tiny trace against the byte-exact function at the pinned commit."""

    import torch

    repository = Path(__file__).resolve().parents[1]
    try:
        module_source = subprocess.check_output(
            [
                "git",
                "-C",
                str(repository),
                "show",
                f"{PINNED_UPSTREAM_COMMIT}:nanochat/dataloader.py",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "pinned upstream commit is unavailable for packing parity"
        ) from exc
    tree = ast.parse(module_source)
    function_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name
            == "tokenizing_distributed_data_loader_with_state_bos_bestfit"
        ),
        None,
    )
    if function_node is None:
        raise ValueError("pinned upstream best-fit function is absent")
    function_source = ast.get_source_segment(module_source, function_node)
    if not isinstance(function_source, str):
        raise ValueError("cannot extract pinned upstream best-fit source")
    function_sha256 = hashlib.sha256(function_source.encode("utf-8")).hexdigest()
    if function_sha256 != PINNED_BESTFIT_FUNCTION_SHA256:
        raise ValueError("pinned upstream best-fit source hash drift")

    lengths = [4, 3, 3, 7, 2, 6, 4, 8, 2, 5, 3, 9, 2, 4, 7, 3, 2, 6, 5, 4]
    texts = [f"fixture-{index}" for index in range(len(lengths))]
    token_map = {
        text: [1, *([10 + index] * (length - 1))]
        for index, (text, length) in enumerate(zip(texts, lengths, strict=True))
    }
    source_batches = [texts[index : index + 3] for index in range(0, len(texts), 3)]

    class FixtureTokenizer:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        @staticmethod
        def get_bos_token_id() -> int:
            return 1

        def encode(self, batch, *, prepend, num_threads):
            self.calls.append(
                {
                    "documents": list(batch),
                    "prepend": prepend,
                    "num_threads": num_threads,
                }
            )
            return [list(token_map[text]) for text in batch]

    def actual_batches(*_args, **_kwargs):
        for index, batch in enumerate(source_batches):
            yield batch, (0, index, 1)

    namespace: dict[str, Any] = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[function_node], type_ignores=[]),
            filename=f"{PINNED_UPSTREAM_COMMIT}:nanochat/dataloader.py",
            mode="exec",
        ),
        namespace,
    )
    namespace["_document_batches"] = actual_batches
    pinned_loader = namespace[
        "tokenizing_distributed_data_loader_with_state_bos_bestfit"
    ]
    tokenizer = FixtureTokenizer()
    with patch.dict(pinned_loader.__globals__, {"_document_batches": actual_batches}):
        actual_loader = pinned_loader(
            tokenizer,
            2,
            5,
            "train",
            tokenizer_threads=4,
            tokenizer_batch_size=128,
            device="cpu",
            buffer_size=4,
        )
        actual = []
        for _ in range(3):
            inputs, targets, _state = next(actual_loader)
            actual.append(
                {
                    "inputs": inputs.detach().cpu().tolist(),
                    "targets": targets.detach().cpu().tolist(),
                }
            )

    packing_batches = [
        [
            PackingDocument(lengths[int(text.split("-")[1])], text, "m", "s", "r")
            for text in batch
        ]
        for batch in source_batches
    ]
    simulated = simulate_bestfit_rank(
        packing_batches,
        B=2,
        T=5,
        buffer_size=4,
        max_microbatches=3,
        collect_trace=True,
    )
    reconstructed = []
    for microbatch in simulated["microbatches"]:
        rows: list[list[int]] = [[], []]
        for selection in microbatch["selection_trace"]:
            rows[int(selection["row"])].extend(
                token_map[str(selection["document_id"])][ : int(selection["used"])]
            )
        reconstructed.append(
            {
                "inputs": [row[:-1] for row in rows],
                "targets": [row[1:] for row in rows],
            }
        )
    if actual != reconstructed:
        raise ValueError("packing simulator differs from imported upstream loader fixture")
    if any(
        call["prepend"] != 1 or call["num_threads"] != 4
        for call in tokenizer.calls
    ):
        raise ValueError("upstream tokenizer call contract drifted")
    fixture = {
        "fixture_id": "bos_bestfit_ties_refill_overshoot_crop_v1",
        "B": 2,
        "T": 5,
        "buffer_size": 4,
        "tokenizer_batch_documents": 3,
        "document_lengths_with_bos": lengths,
        "microbatches_compared": 3,
    }
    return {
        "passed": True,
        "fixture": fixture,
        "fixture_sha256": hashlib.sha256(canonical_json_bytes(fixture)).hexdigest(),
        "actual_output_sha256": hashlib.sha256(
            canonical_json_bytes(actual)
        ).hexdigest(),
        "simulated_output_sha256": hashlib.sha256(
            canonical_json_bytes(reconstructed)
        ).hexdigest(),
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
        "upstream_loader_source_sha256": function_sha256,
    }


def simulate_final_corpus_capacity(
    root: str | Path,
    train_files: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    world_sizes: Sequence[int] = (8, 16),
    B: int = 4,
    T: int = 2048,
    buffer_size: int = 1000,
    tokenizer_batch_size: int = 128,
    global_batch_tokens: int = 2_097_152,
    required_optimizer_steps: int = 32_000,
    safety_margin_fraction: float = 0.02,
) -> dict[str, Any]:
    corpus_root = Path(root)
    total_wall_start = time.monotonic()
    total_cpu_start = time.process_time()
    hash_wall_start = time.monotonic()
    hash_cpu_start = time.process_time()
    # File identity is invariant across rank/world simulations.  Verify it once
    # rather than re-reading and hashing the entire corpus 24 times.
    for file_record in train_files:
        path = corpus_root / str(file_record["path"])
        if (
            path.is_symlink()
            or path.stat().st_size != int(file_record["size_bytes"])
            or file_sha256(path) != file_record["sha256"]
        ):
            raise ValueError(f"packing simulation file drift: {file_record['path']}")
    hash_telemetry = {
        "wall_seconds": time.monotonic() - hash_wall_start,
        "cpu_seconds": time.process_time() - hash_cpu_start,
        "files_verified": len(train_files),
        "bytes_verified": sum(int(item["size_bytes"]) for item in train_files),
    }
    parity = run_upstream_loader_parity_fixture()
    results: dict[str, Any] = {}
    world_telemetry: dict[str, Any] = {}
    for raw_world_size in world_sizes:
        world_size = int(raw_world_size)
        world_wall_start = time.monotonic()
        world_cpu_start = time.process_time()
        denominator = world_size * B * T
        if global_batch_tokens % denominator:
            raise ValueError("global batch is not divisible by rank-local microbatch")
        accumulation = global_batch_tokens // denominator
        required_with_margin = math.ceil(
            required_optimizer_steps * (1.0 + safety_margin_fraction)
        )
        requested_microbatches = required_with_margin * accumulation
        ranks = [
            simulate_bestfit_rank(
                _rank_document_batches(
                    corpus_root,
                    train_files,
                    tokenizer,
                    rank=rank,
                    world_size=world_size,
                    tokenizer_batch_size=tokenizer_batch_size,
                ),
                B=B,
                T=T,
                buffer_size=buffer_size,
                max_microbatches=requested_microbatches,
            )
            for rank in range(world_size)
        ]
        results[str(world_size)] = summarize_world(
            ranks,
            world_size=world_size,
            B=B,
            T=T,
            global_batch_tokens=global_batch_tokens,
            required_optimizer_steps=required_optimizer_steps,
            safety_margin_fraction=safety_margin_fraction,
            buffer_size=buffer_size,
        )
        world_telemetry[str(world_size)] = {
            "wall_seconds": time.monotonic() - world_wall_start,
            "cpu_seconds": time.process_time() - world_cpu_start,
            "documents_loaded": sum(int(item["loaded_documents"]) for item in ranks),
            "source_tokens_with_bos_loaded": sum(
                int(item["loaded_source_tokens_with_bos"]) for item in ranks
            ),
        }
    return {
        "implementation": PACKING_SIMULATOR_ID,
        "implementation_file_sha256": file_sha256(Path(__file__)),
        "upstream_contract": {
            "nanochat_revision": PINNED_UPSTREAM_REVISION,
            "encode_call": "tokenizer.encode(doc_batch, prepend=bos_token, num_threads=4)",
            "tokenizer_batch_size": tokenizer_batch_size,
            "tokenizer_threads": 4,
            "refill_buffer_size": buffer_size,
            "tie_breaks": "first_largest_fit_else_first_shortest",
            "cropped_tail_policy": "discard",
            "rank_sharding": "row_group_index_mod_world_size",
        },
        "fixture_parity": parity,
        "resource_measurement": {
            "hash_inventory": hash_telemetry,
            "worlds": world_telemetry,
            "total_wall_seconds": time.monotonic() - total_wall_start,
            "total_cpu_seconds": time.process_time() - total_cpu_start,
            "token_lengths_recomputed_once_per_requested_world_size": True,
            "file_hashes_verified_once_across_all_ranks_and_worlds": True,
        },
        "worlds": results,
        "all_worlds_pass": all(
            item["passes_40x_no_wrap_with_margin"] for item in results.values()
        ),
    }


def _recommended_source_weights(
    intended_weights: Mapping[str, float], simulation: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    recommendations: dict[str, dict[str, float]] = {}
    for world, metrics in simulation["worlds"].items():
        efficiency = metrics["retention_efficiency_by_mixture"]
        raw: dict[str, float] = {}
        for mixture, intended in intended_weights.items():
            measured = float(efficiency.get(mixture, 0.0))
            if measured <= 0:
                raw[mixture] = 0.0
            else:
                raw[mixture] = float(intended) / measured
        denominator = sum(raw.values())
        recommendations[world] = {
            key: (value / denominator if denominator else 0.0)
            for key, value in sorted(raw.items())
        }
    return recommendations


def seal_capacity_receipt(
    output_path: str | Path,
    *,
    simulation: Mapping[str, Any],
    dataset_manifest_sha256: str,
    tokenizer_package_sha256: str,
    intended_weights: Mapping[str, float],
    current_source_token_target: int,
    mix_absolute_tolerance: float = 0.03,
) -> dict[str, Any]:
    deviations: dict[str, dict[str, float]] = {}
    mix_pass = bool(simulation["all_worlds_pass"])
    for world, metrics in simulation["worlds"].items():
        world_deviation: dict[str, float] = {}
        realized = metrics["realized_retained_mix"]
        for mixture, intended in intended_weights.items():
            delta = float(realized.get(mixture, 0.0)) - float(intended)
            world_deviation[mixture] = delta
            if abs(delta) > mix_absolute_tolerance:
                mix_pass = False
        deviations[world] = world_deviation

    minimum_positions = min(
        int(item["common_prefix_scheduled_positions"])
        for item in simulation["worlds"].values()
    )
    required_positions = max(
        int(item["required_positions_with_margin"])
        for item in simulation["worlds"].values()
    )
    recommended = (
        current_source_token_target
        if minimum_positions >= required_positions
        else math.ceil(
            current_source_token_target
            * required_positions
            / max(1, minimum_positions)
            * 1.01
        )
    )
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_bestfit_capacity_receipt",
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "tokenizer_package_sha256": tokenizer_package_sha256,
            "simulation": dict(simulation),
            "intended_mixture_weights": dict(sorted(intended_weights.items())),
            "realized_minus_intended_by_world": deviations,
            "mix_absolute_tolerance": mix_absolute_tolerance,
            "mix_gate_evaluated_on_common_horizon": bool(simulation["all_worlds_pass"]),
            "mix_gate_passed": mix_pass,
            "no_wrap_gate_passed": simulation["all_worlds_pass"],
            "gate_passed": bool(simulation["all_worlds_pass"] and mix_pass),
            "current_source_token_target": current_source_token_target,
            "recommended_source_token_target_if_retry": recommended,
            "recommended_source_weights_from_measured_retention": (
                _recommended_source_weights(intended_weights, simulation)
            ),
            "recommendation_requires_fresh_simulation": True,
            "cleanup_authorized": bool(simulation["all_worlds_pass"] and mix_pass),
            "canonical_sha256": None,
        }
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite capacity receipt: {destination}")
    write_json_atomic(destination, receipt)
    return receipt


__all__ = [
    "PACKING_SIMULATOR_ID",
    "PackingDocument",
    "run_upstream_loader_parity_fixture",
    "seal_capacity_receipt",
    "simulate_bestfit_rank",
    "simulate_final_corpus_capacity",
    "summarize_world",
]
