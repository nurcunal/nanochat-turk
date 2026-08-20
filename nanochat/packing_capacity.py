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
import tempfile
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
REPETITION_PACKING_SIMULATOR_ID = "nanochat_upstream_bos_bestfit_repeat_capacity_v3"
D32_REPETITION_WORLD_SIZES = (8, 16)
D32_REPETITION_DEVICE_BATCH_SEQUENCES = 4
D32_REPETITION_MAX_SEQ_LEN = 2048
D32_REPETITION_TOKENIZER_BATCH_SIZE = 128
D32_REPETITION_BUFFER_SIZE = 1000
D32_REPETITION_GLOBAL_BATCH_TOKENS = 2_097_152
D32_REPETITION_MIX_ABSOLUTE_TOLERANCE = 0.03
D32_REPETITION_HORIZON_OPTIMIZER_STEPS = {
    "s12": 9_600,
    "s20": 16_000,
    "s40": 32_000,
    "s40_margin": 32_640,
}
PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS = 34_225_520_640
HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS = 17_112_760_320
PREFERRED_MAX_LOADED_EPOCH = 2
PREFERRED_MAX_CONSUMED_EPOCH = 2
HARD_MAX_LOADED_EPOCH = 4
HARD_MAX_CONSUMED_EPOCH = 4
PINNED_PARITY_FIXTURE_SHA256 = (
    "42d90d4935fd7308acaef01634847f88653cdc509496cd6945e73070ecfe3b97"
)
PINNED_PARITY_OUTPUT_SHA256 = (
    "fbc66bb43681b69ae6f3de516e4b9e75f6aac9c96c2f0be1cb1775c0d8fc34f7"
)
PINNED_REPEAT_PARITY_FIXTURE_SHA256 = (
    "fa4865e88faa2dbc3d8ddf61c58d5ba5c84034a0c1649ddb7fc6b5a50bf5db03"
)
PINNED_REPEAT_PARITY_OUTPUT_SHA256 = (
    "deb284d3292f645603495abc6e6c367052e172e4d7cb4b45efab980cbe4616c7"
)
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
    # v3 repeats the *whole* rank-sharded pool.  The base locator identifies
    # one physical corpus instance independently of its epoch.  Defaults keep
    # every v2 caller and its no-wrap semantics byte-for-byte compatible.
    base_locator: str | None = None
    epoch: int = 1

    def __post_init__(self) -> None:
        if self.tokens_with_bos <= 0:
            raise ValueError("packing documents require at least BOS")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 1:
            raise ValueError("packing document epoch must be a positive integer")

    @property
    def locator_ignoring_epoch(self) -> str:
        return self.document_id if self.base_locator is None else self.base_locator


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
    result = {
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
    if (
        result["fixture_sha256"] != PINNED_PARITY_FIXTURE_SHA256
        or result["actual_output_sha256"] != PINNED_PARITY_OUTPUT_SHA256
        or result["simulated_output_sha256"] != PINNED_PARITY_OUTPUT_SHA256
    ):
        raise ValueError("pinned upstream packing parity output drift")
    return result


def run_upstream_repeated_epoch_parity_fixture() -> dict[str, Any]:
    """Cross two epoch boundaries against the byte-exact upstream loader."""

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
            "pinned upstream commit is unavailable for repeat packing parity"
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

    token_map: dict[str, list[int]] = {}
    for epoch in range(1, 5):
        for index in range(2):
            text = f"epoch-{epoch}-document-{index}"
            token_map[text] = [1, *([10 + epoch * 2 + index] * 3)]

    class FixtureTokenizer:
        @staticmethod
        def get_bos_token_id() -> int:
            return 1

        @staticmethod
        def encode(batch, *, prepend, num_threads):
            if prepend != 1 or num_threads != 4:
                raise ValueError("repeat parity tokenizer call drift")
            return [list(token_map[text]) for text in batch]

    def actual_batches(*_args, **_kwargs):
        epoch = 1
        while True:
            yield [
                f"epoch-{epoch}-document-0",
                f"epoch-{epoch}-document-1",
            ], (0, 0, epoch)
            epoch += 1

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
    with patch.dict(pinned_loader.__globals__, {"_document_batches": actual_batches}):
        actual_loader = pinned_loader(
            FixtureTokenizer(),
            1,
            3,
            "train",
            tokenizer_threads=4,
            tokenizer_batch_size=128,
            device="cpu",
            buffer_size=2,
        )
        actual: list[dict[str, Any]] = []
        for _ in range(6):
            inputs, targets, state = next(actual_loader)
            actual.append(
                {
                    "inputs": inputs.detach().cpu().tolist(),
                    "targets": targets.detach().cpu().tolist(),
                    "state": dict(state),
                }
            )

    def packing_batches():
        epoch = 1
        while True:
            yield [
                PackingDocument(
                    4,
                    f"epoch-{epoch}-document-{index}",
                    "m",
                    "s",
                    "r",
                    base_locator=f"document-{index}",
                    epoch=epoch,
                )
                for index in range(2)
            ]
            epoch += 1

    horizon_names = {f"microbatch_{index}": index for index in range(1, 7)}
    simulated = simulate_bestfit_rank_repeated(
        packing_batches(),
        B=1,
        T=3,
        buffer_size=2,
        snapshot_microbatches=horizon_names,
        collect_trace=True,
    )
    reconstructed: list[dict[str, Any]] = []
    for index, microbatch in enumerate(simulated["microbatches"], start=1):
        row: list[int] = []
        for selection in microbatch["selection_trace"]:
            row.extend(
                token_map[str(selection["document_id"])][: int(selection["used"])]
            )
        loaded_epoch = simulated["snapshots"][f"microbatch_{index}"][
            "max_loaded_epoch"
        ]
        reconstructed.append(
            {
                "inputs": [row[:-1]],
                "targets": [row[1:]],
                "state": {"pq_idx": 0, "rg_idx": 0, "epoch": loaded_epoch},
            }
        )
    if actual != reconstructed:
        raise ValueError("repeat simulator differs from pinned loader at epoch boundary")
    fixture = {
        "fixture_id": "bos_bestfit_repeat_refill_epoch_boundary_v1",
        "B": 1,
        "T": 3,
        "buffer_size": 2,
        "documents_per_epoch": 2,
        "document_lengths_with_bos": [4, 4],
        "microbatches_compared": 6,
        "epoch_boundaries_crossed": 2,
    }
    result = {
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
    if (
        result["fixture_sha256"] != PINNED_REPEAT_PARITY_FIXTURE_SHA256
        or result["actual_output_sha256"] != PINNED_REPEAT_PARITY_OUTPUT_SHA256
        or result["simulated_output_sha256"] != PINNED_REPEAT_PARITY_OUTPUT_SHA256
    ):
        raise ValueError("pinned repeated-epoch packing parity output drift")
    return result


def _repeat_document_fingerprint(document: PackingDocument) -> bytes:
    """Hash the epoch-independent fields used by the packing decision."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "base_locator": document.locator_ignoring_epoch,
                "tokens_with_bos": document.tokens_with_bos,
                "mixture_id": document.mixture_id,
                "source_id": document.source_id,
                "register_bucket": document.register_bucket,
            }
        )
    ).digest()


class _WholePoolPrefixVerifier:
    """Exact disk-backed epoch/prefix equality proof with bounded RAM."""

    _DIGEST_BYTES = hashlib.sha256().digest_size

    def __init__(self) -> None:
        self._reference = tempfile.TemporaryFile(mode="w+b")
        self.epoch = 0
        self.epoch1_documents: int | None = None
        self.current_documents = 0
        self.verified_complete_epochs = 0
        self._epoch1_hasher = hashlib.sha256()

    def begin_epoch(self, epoch: int) -> None:
        if self.epoch == 0:
            if epoch != 1:
                raise ValueError("whole-pool repeat simulation must begin at epoch 1")
        else:
            if epoch != self.epoch + 1:
                raise ValueError("whole-pool repeat epochs must be contiguous")
            self.finish_epoch()
        self.epoch = epoch
        self.current_documents = 0
        if epoch > 1:
            self._reference.seek(0)

    def observe(self, document: PackingDocument) -> None:
        digest = _repeat_document_fingerprint(document)
        if self.epoch == 1:
            self._reference.write(digest)
            self._epoch1_hasher.update(digest)
        else:
            expected = self._reference.read(self._DIGEST_BYTES)
            if expected != digest:
                raise ValueError("whole-pool repeat prefix differs from epoch 1")
        self.current_documents += 1

    def finish_epoch(self) -> None:
        if self.epoch == 0:
            return
        if self.epoch == 1:
            self.epoch1_documents = self.current_documents
            if self.epoch1_documents <= 0:
                raise ValueError("whole-pool repeat epoch 1 is empty")
            self._reference.flush()
        elif (
            self.current_documents != self.epoch1_documents
            or self._reference.read(1) != b""
        ):
            raise ValueError("whole-pool repeat epoch differs from epoch 1")
        self.verified_complete_epochs = self.epoch

    @property
    def epoch1_sha256(self) -> str | None:
        return self._epoch1_hasher.hexdigest() if self.epoch1_documents is not None else None

    @property
    def external_state_bytes(self) -> int:
        return int(self.epoch1_documents or self.current_documents) * self._DIGEST_BYTES

    def close(self) -> None:
        self._reference.close()


def classify_repetition_tier(
    *,
    first_epoch_packed_positions: int,
    max_loaded_epoch: int,
    max_consumed_epoch: int,
    preferred_min_first_epoch_packed_positions: int,
    hard_min_first_epoch_packed_positions: int,
    preferred_max_loaded_epoch: int,
    preferred_max_consumed_epoch: int,
    hard_max_loaded_epoch: int,
    hard_max_consumed_epoch: int,
) -> str:
    # Loading epoch 5 is disqualifying even if it happened only because the
    # 1,000-document buffer prefetched it while completing an epoch-4 batch.
    if (
        first_epoch_packed_positions < hard_min_first_epoch_packed_positions
        or max_loaded_epoch > hard_max_loaded_epoch
        or max_consumed_epoch > hard_max_consumed_epoch
        or max_loaded_epoch >= 5
    ):
        return "failed"
    if (
        first_epoch_packed_positions >= preferred_min_first_epoch_packed_positions
        and max_loaded_epoch <= preferred_max_loaded_epoch
        and max_consumed_epoch <= preferred_max_consumed_epoch
    ):
        return "preferred"
    return "manual_risk"


def simulate_bestfit_rank_repeated(
    document_batches: Iterable[Sequence[PackingDocument]],
    *,
    B: int,
    T: int,
    snapshot_microbatches: Mapping[str, int],
    buffer_size: int = 1000,
    collect_trace: bool = False,
) -> dict[str, Any]:
    """Simulate one rank across exact whole-pool epochs with bounded memory.

    Epochs must contain the same documents in the same order.  A disk-backed
    digest stream validates every complete epoch and the currently loaded
    partial prefix against epoch 1 without retaining a corpus-sized RAM table.
    Under that invariant, stable best-fit ties
    imply that an epoch-N instance cannot be consumed before older instances
    of the same base locator; consequently the greatest loaded/consumed epoch
    is exactly the greatest per-base-locator load/consume count.
    """

    if min(B, T, buffer_size) <= 0:
        raise ValueError("B/T/buffer_size must be positive")
    if not snapshot_microbatches:
        raise ValueError("repeat simulation requires at least one horizon")
    normalized_horizons: dict[str, int] = {}
    for raw_name, raw_value in snapshot_microbatches.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value <= 0
        ):
            raise ValueError("repeat horizons require named positive microbatch counts")
        normalized_horizons[raw_name] = raw_value
    if len(set(normalized_horizons.values())) != len(normalized_horizons):
        raise ValueError("repeat horizon microbatch counts must be distinct")
    target_to_name = {value: key for key, value in normalized_horizons.items()}
    max_microbatches = max(target_to_name)

    batches = iter(document_batches)
    buffer: list[PackingDocument] = []
    retained = _metric_counters()
    cropped = _metric_counters()
    consumed = _metric_counters()
    row_leading = _metric_counters()
    loaded_epoch_histogram: Counter[int] = Counter()
    consumed_epoch_histogram: Counter[int] = Counter()
    first_load_microbatch_by_epoch: dict[int, int] = {}
    first_consume_microbatch_by_epoch: dict[int, int] = {}
    completed = 0
    loaded_documents = 0
    loaded_source_tokens = 0
    refill_batches = 0
    retained_source_elements = 0
    cropped_tokens = 0
    row_leading_elements = 0
    documents_completed = 0
    documents_cropped = 0
    max_loaded_epoch = 0
    max_consumed_epoch = 0
    first_epoch2_load_before_microbatch: int | None = None
    snapshots: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []

    order_verifier = _WholePoolPrefixVerifier()

    def ingest(incoming: Sequence[PackingDocument]) -> None:
        nonlocal loaded_documents, loaded_source_tokens, refill_batches, max_loaded_epoch
        nonlocal first_epoch2_load_before_microbatch
        if not incoming:
            return
        epochs = {item.epoch for item in incoming}
        if len(epochs) != 1:
            raise ValueError("one tokenizer refill batch may not cross repeat epochs")
        incoming_epoch = next(iter(epochs))
        if incoming_epoch != order_verifier.epoch:
            order_verifier.begin_epoch(incoming_epoch)
        if incoming_epoch == 2 and first_epoch2_load_before_microbatch is None:
            first_epoch2_load_before_microbatch = completed
        first_load_microbatch_by_epoch.setdefault(incoming_epoch, completed)
        max_loaded_epoch = max(max_loaded_epoch, incoming_epoch)
        for document in incoming:
            order_verifier.observe(document)
        buffer.extend(incoming)
        refill_batches += 1
        loaded_documents += len(incoming)
        loaded_source_tokens += sum(item.tokens_with_bos for item in incoming)
        loaded_epoch_histogram[incoming_epoch] += len(incoming)

    def snapshot() -> dict[str, Any]:
        buffer_epochs = Counter(item.epoch for item in buffer)
        return {
            "completed_microbatches": completed,
            "scheduled_token_positions": completed * B * T,
            "loaded_documents": loaded_documents,
            "loaded_source_tokens_with_bos": loaded_source_tokens,
            "buffered_documents": len(buffer),
            "buffered_source_tokens": sum(item.tokens_with_bos for item in buffer),
            "buffer_epoch_histogram": {
                str(key): value for key, value in sorted(buffer_epochs.items())
            },
            "loaded_documents_by_epoch": {
                str(key): value for key, value in sorted(loaded_epoch_histogram.items())
            },
            "consumed_documents_by_epoch": {
                str(key): value for key, value in sorted(consumed_epoch_histogram.items())
            },
            "first_load_before_microbatch_by_epoch": {
                str(key): value
                for key, value in sorted(first_load_microbatch_by_epoch.items())
            },
            "first_consume_in_microbatch_by_epoch": {
                str(key): value
                for key, value in sorted(first_consume_microbatch_by_epoch.items())
            },
            "max_loaded_epoch": max_loaded_epoch,
            "max_consumed_epoch": max_consumed_epoch,
            "max_base_locator_load_count": max_loaded_epoch,
            "max_base_locator_consume_count": max_consumed_epoch,
            "epoch5_loaded_including_prefetch": max_loaded_epoch >= 5,
            "retained_source_elements": retained_source_elements,
            "row_leading_source_elements_not_targets": row_leading_elements,
            "cropped_source_tokens": cropped_tokens,
            "documents_completed": documents_completed,
            "documents_cropped": documents_cropped,
            "retained_positions": _plain_metrics(retained),
            "cropped_tokens": _plain_metrics(cropped),
            "consumed_source_elements": _plain_metrics(consumed),
            "row_leading_source_elements": _plain_metrics(row_leading),
        }

    while completed < max_microbatches:
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
        for _row_index in range(B):
            position = 0
            while position < T + 1:
                while len(buffer) < buffer_size:
                    try:
                        incoming = list(next(batches))
                    except StopIteration as exc:
                        raise ValueError(
                            "whole-pool repeat iterator ended before the requested horizon"
                        ) from exc
                    ingest(incoming)
                remaining = T + 1 - position
                best_index = -1
                best_length = 0
                for index, document in enumerate(buffer):
                    length = document.tokens_with_bos
                    if length <= remaining and length > best_length:
                        best_index = index
                        best_length = length
                selected = (
                    best_index
                    if best_index >= 0
                    else min(
                        range(len(buffer)),
                        key=lambda index: buffer[index].tokens_with_bos,
                    )
                )
                document = buffer.pop(selected)
                first_consume_microbatch_by_epoch.setdefault(document.epoch, completed)
                max_consumed_epoch = max(max_consumed_epoch, document.epoch)
                consumed_epoch_histogram[document.epoch] += 1
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
                            "epoch": document.epoch,
                            "tokens_with_bos": document.tokens_with_bos,
                            "used": used,
                            "discarded": discarded,
                        }
                    )
                position += used

        if sum(batch_retained["mixture"].values()) != B * T:
            raise ValueError("repeat best-fit simulator target attribution drift")
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
        horizon_name = target_to_name.get(completed)
        if horizon_name is not None:
            snapshots[horizon_name] = snapshot()

    validation = {
        "method": "bounded_disk_exact_epoch_and_partial_prefix_sha256_v2",
        "epoch1_sha256": order_verifier.epoch1_sha256,
        "epoch1_documents": order_verifier.epoch1_documents,
        "verified_complete_epochs": order_verifier.verified_complete_epochs,
        "current_partial_epoch": order_verifier.epoch,
        "current_partial_epoch_documents": order_verifier.current_documents,
        "current_partial_prefix_verified_against_epoch1": (
            order_verifier.epoch <= 1 or order_verifier.current_documents > 0
        ),
        "external_state_bytes": order_verifier.external_state_bytes,
        "digest_bytes_per_epoch1_document": _WholePoolPrefixVerifier._DIGEST_BYTES,
        "fixed_rank_assignment_and_stable_ties_prove_base_locator_maxima": True,
    }
    order_verifier.close()
    result = {
        "completed_microbatches": completed,
        "first_epoch2_load_before_microbatch": first_epoch2_load_before_microbatch,
        "first_epoch2_load_observation": (
            "observed_including_prefetch"
            if first_epoch2_load_before_microbatch is not None
            else "right_censored_at_maximum_horizon"
        ),
        "whole_pool_order_validation": validation,
        "snapshots": snapshots,
    }
    if collect_trace:
        result["microbatches"] = traces
    return result


def _rank_repeated_document_batches(
    root: Path,
    train_files: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    rank: int,
    world_size: int,
    tokenizer_batch_size: int,
) -> Iterator[list[PackingDocument]]:
    """Yield the entire fixed rank shard repeatedly; never cycle a source."""

    bos_token = tokenizer.get_bos_token_id()
    epoch = 1
    while True:
        yielded = False
        for file_record in train_files:
            relative_path = str(file_record["path"])
            path = root / relative_path
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
                    encoded = tokenizer.encode(
                        [str(row["text"]) for row in batch_rows],
                        prepend=bos_token,
                        num_threads=4,
                    )
                    if not isinstance(encoded, list) or len(encoded) != len(batch_rows):
                        raise ValueError("packing tokenizer returned an invalid batch")
                    yielded = True
                    yield [
                        PackingDocument(
                            tokens_with_bos=len(tokens),
                            document_id=str(row["document_id"]),
                            mixture_id=str(row["mixture_id"]),
                            source_id=str(row["source_id"]),
                            register_bucket=str(
                                row.get("register_bucket") or "not_applicable"
                            ),
                            base_locator=(
                                f"{relative_path}\0{row_group_index}\0{offset + index}"
                            ),
                            epoch=epoch,
                        )
                        for index, (row, tokens) in enumerate(
                            zip(batch_rows, encoded, strict=True)
                        )
                    ]
                row_group_index += world_size
        if not yielded:
            raise ValueError(f"rank {rank} has no documents in the whole repeat pool")
        epoch += 1


def _require_exact_nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def _validate_repetition_rank_snapshot(
    snapshot: Mapping[str, Any],
    *,
    expected_microbatches: int,
    B: int,
    T: int,
) -> None:
    required = {
        "completed_microbatches",
        "scheduled_token_positions",
        "loaded_documents",
        "loaded_source_tokens_with_bos",
        "buffered_documents",
        "buffered_source_tokens",
        "buffer_epoch_histogram",
        "loaded_documents_by_epoch",
        "consumed_documents_by_epoch",
        "first_load_before_microbatch_by_epoch",
        "first_consume_in_microbatch_by_epoch",
        "max_loaded_epoch",
        "max_consumed_epoch",
        "max_base_locator_load_count",
        "max_base_locator_consume_count",
        "epoch5_loaded_including_prefetch",
        "retained_source_elements",
        "row_leading_source_elements_not_targets",
        "cropped_source_tokens",
        "documents_completed",
        "documents_cropped",
        "retained_positions",
        "cropped_tokens",
        "consumed_source_elements",
        "row_leading_source_elements",
    }
    if set(snapshot) != required:
        raise ValueError("repeat rank snapshot fields differ from frozen evidence schema")
    completed = _require_exact_nonnegative_int(
        snapshot["completed_microbatches"], "completed microbatches"
    )
    if completed != expected_microbatches:
        raise ValueError("repeat rank snapshot is not at its exact horizon")
    scheduled = _require_exact_nonnegative_int(
        snapshot["scheduled_token_positions"], "scheduled token positions"
    )
    if scheduled != completed * B * T:
        raise ValueError("repeat rank scheduled positions drift")

    def histogram(name: str) -> dict[int, int]:
        raw = snapshot[name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} must be a mapping")
        parsed: dict[int, int] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.isdigit() or int(key) < 1:
                raise ValueError(f"{name} has an invalid epoch key")
            parsed[int(key)] = _require_exact_nonnegative_int(value, name)
        if any(value == 0 for value in parsed.values()):
            raise ValueError(f"{name} may not retain zero-count epochs")
        return parsed

    loaded_hist = histogram("loaded_documents_by_epoch")
    consumed_hist = histogram("consumed_documents_by_epoch")
    buffer_hist = histogram("buffer_epoch_histogram")
    if not loaded_hist or set(loaded_hist) != set(range(1, max(loaded_hist) + 1)):
        raise ValueError("loaded repeat epochs must be contiguous from one")
    if consumed_hist and set(consumed_hist) != set(range(1, max(consumed_hist) + 1)):
        raise ValueError("consumed repeat epochs must be contiguous from one")
    for epoch, loaded in loaded_hist.items():
        if loaded - consumed_hist.get(epoch, 0) != buffer_hist.get(epoch, 0):
            raise ValueError("repeat epoch load/consume/buffer accounting drift")
    if set(buffer_hist) - set(loaded_hist):
        raise ValueError("buffer contains an unloaded repeat epoch")
    loaded_documents = _require_exact_nonnegative_int(
        snapshot["loaded_documents"], "loaded documents"
    )
    buffered_documents = _require_exact_nonnegative_int(
        snapshot["buffered_documents"], "buffered documents"
    )
    documents_completed = _require_exact_nonnegative_int(
        snapshot["documents_completed"], "documents completed"
    )
    documents_cropped = _require_exact_nonnegative_int(
        snapshot["documents_cropped"], "documents cropped"
    )
    if loaded_documents != sum(loaded_hist.values()):
        raise ValueError("loaded repeat document histogram drift")
    if buffered_documents != sum(buffer_hist.values()):
        raise ValueError("buffered repeat document histogram drift")
    if documents_completed + documents_cropped != sum(consumed_hist.values()):
        raise ValueError("consumed repeat document histogram drift")
    if loaded_documents != buffered_documents + documents_completed + documents_cropped:
        raise ValueError("repeat document conservation drift")

    max_loaded = max(loaded_hist)
    max_consumed = max(consumed_hist) if consumed_hist else 0
    for field, expected in (
        ("max_loaded_epoch", max_loaded),
        ("max_consumed_epoch", max_consumed),
        ("max_base_locator_load_count", max_loaded),
        ("max_base_locator_consume_count", max_consumed),
    ):
        if _require_exact_nonnegative_int(snapshot[field], field) != expected:
            raise ValueError(f"{field} differs from exact epoch evidence")
    if snapshot["epoch5_loaded_including_prefetch"] is not (max_loaded >= 5):
        raise ValueError("epoch-5 prefetch evidence drift")

    def first_batches(name: str, epochs: set[int]) -> dict[int, int]:
        raw = snapshot[name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} must be a mapping")
        parsed = {
            int(key): _require_exact_nonnegative_int(value, name)
            for key, value in raw.items()
            if isinstance(key, str) and key.isdigit() and int(key) >= 1
        }
        if len(parsed) != len(raw) or set(parsed) != epochs:
            raise ValueError(f"{name} epoch coverage drift")
        if any(value >= completed for value in parsed.values()):
            raise ValueError(f"{name} points outside completed microbatches")
        return parsed

    first_load = first_batches(
        "first_load_before_microbatch_by_epoch", set(loaded_hist)
    )
    first_consume = first_batches(
        "first_consume_in_microbatch_by_epoch", set(consumed_hist)
    )
    if first_load.get(1) != 0 or first_consume.get(1) != 0:
        raise ValueError("repeat epoch 1 must begin in microbatch zero")
    if list(first_load.values()) != sorted(first_load.values()):
        raise ValueError("repeat first-load batches are not monotonic")
    if list(first_consume.values()) != sorted(first_consume.values()):
        raise ValueError("repeat first-consume batches are not monotonic")

    metrics_by_name: dict[str, dict[str, Mapping[str, Any]]] = {}
    for name in (
        "retained_positions",
        "cropped_tokens",
        "consumed_source_elements",
        "row_leading_source_elements",
    ):
        raw = snapshot[name]
        if not isinstance(raw, Mapping) or set(raw) != {"mixture", "source", "register"}:
            raise ValueError(f"{name} dimensions drift")
        metrics_by_name[name] = {}
        for dimension, values in raw.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"{name}.{dimension} must be a mapping")
            parsed_values: dict[str, int] = {}
            for key, value in values.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{name}.{dimension} has an invalid label")
                parsed_values[key] = _require_exact_nonnegative_int(
                    value, f"{name}.{dimension}"
                )
            metrics_by_name[name][str(dimension)] = parsed_values
        totals = [sum(values.values()) for values in metrics_by_name[name].values()]
        if len(set(totals)) != 1:
            raise ValueError(f"{name} dimension totals disagree")

    retained_total = sum(metrics_by_name["retained_positions"]["mixture"].values())
    cropped_total = sum(metrics_by_name["cropped_tokens"]["mixture"].values())
    consumed_total = sum(
        metrics_by_name["consumed_source_elements"]["mixture"].values()
    )
    row_leading_total = sum(
        metrics_by_name["row_leading_source_elements"]["mixture"].values()
    )
    retained_source_elements = _require_exact_nonnegative_int(
        snapshot["retained_source_elements"], "retained source elements"
    )
    cropped_source_tokens = _require_exact_nonnegative_int(
        snapshot["cropped_source_tokens"], "cropped source tokens"
    )
    row_leading = _require_exact_nonnegative_int(
        snapshot["row_leading_source_elements_not_targets"], "row-leading elements"
    )
    if (
        retained_total != scheduled
        or consumed_total != retained_source_elements
        or cropped_total != cropped_source_tokens
        or row_leading_total != row_leading
        or row_leading != completed * B
        or retained_source_elements != scheduled + row_leading
    ):
        raise ValueError("repeat token-position accounting drift")
    buffered_source_tokens = _require_exact_nonnegative_int(
        snapshot["buffered_source_tokens"], "buffered source tokens"
    )
    loaded_source_tokens = _require_exact_nonnegative_int(
        snapshot["loaded_source_tokens_with_bos"], "loaded source tokens"
    )
    if loaded_source_tokens != consumed_total + cropped_total + buffered_source_tokens:
        raise ValueError("repeat loaded-token conservation drift")


def _validate_repetition_rank_evidence(
    evidence: Mapping[str, Any],
    *,
    snapshot_microbatches: Mapping[str, int],
    B: int,
    T: int,
) -> None:
    if set(evidence) != {
        "completed_microbatches",
        "first_epoch2_load_before_microbatch",
        "first_epoch2_load_observation",
        "whole_pool_order_validation",
        "snapshots",
    }:
        raise ValueError("repeat rank evidence fields differ from frozen schema")
    if _require_exact_nonnegative_int(
        evidence["completed_microbatches"], "rank completed microbatches"
    ) != max(snapshot_microbatches.values()):
        raise ValueError("repeat rank did not reach the maximum horizon")
    snapshots = evidence["snapshots"]
    if not isinstance(snapshots, Mapping) or set(snapshots) != set(snapshot_microbatches):
        raise ValueError("repeat rank horizon snapshot coverage drift")
    for name, expected in snapshot_microbatches.items():
        snapshot = snapshots[name]
        if not isinstance(snapshot, Mapping):
            raise ValueError("repeat rank snapshot must be a mapping")
        _validate_repetition_rank_snapshot(
            snapshot,
            expected_microbatches=expected,
            B=B,
            T=T,
        )
    margin = snapshots["s40_margin"]
    first_epoch2 = evidence["first_epoch2_load_before_microbatch"]
    expected_first_epoch2 = margin["first_load_before_microbatch_by_epoch"].get("2")
    if first_epoch2 != expected_first_epoch2:
        raise ValueError("repeat first epoch-2 load evidence drift")
    expected_observation = (
        "observed_including_prefetch"
        if first_epoch2 is not None
        else "right_censored_at_maximum_horizon"
    )
    if evidence["first_epoch2_load_observation"] != expected_observation:
        raise ValueError("repeat epoch-2 observation status drift")
    validation = evidence["whole_pool_order_validation"]
    if not isinstance(validation, Mapping) or set(validation) != {
        "method",
        "epoch1_sha256",
        "epoch1_documents",
        "verified_complete_epochs",
        "current_partial_epoch",
        "current_partial_epoch_documents",
        "current_partial_prefix_verified_against_epoch1",
        "external_state_bytes",
        "digest_bytes_per_epoch1_document",
        "fixed_rank_assignment_and_stable_ties_prove_base_locator_maxima",
    }:
        raise ValueError("whole-pool validation evidence schema drift")
    if (
        validation["method"]
        != "bounded_disk_exact_epoch_and_partial_prefix_sha256_v2"
        or validation["current_partial_prefix_verified_against_epoch1"] is not True
        or validation["fixed_rank_assignment_and_stable_ties_prove_base_locator_maxima"]
        is not True
        or validation["digest_bytes_per_epoch1_document"]
        != _WholePoolPrefixVerifier._DIGEST_BYTES
    ):
        raise ValueError("whole-pool exact-prefix proof is absent")
    current_epoch = int(margin["max_loaded_epoch"])
    current_documents = int(margin["loaded_documents_by_epoch"][str(current_epoch)])
    if (
        validation["current_partial_epoch"] != current_epoch
        or validation["current_partial_epoch_documents"] != current_documents
        or validation["verified_complete_epochs"] != current_epoch - 1
    ):
        raise ValueError("whole-pool epoch validation coverage drift")
    epoch1_documents = validation["epoch1_documents"]
    if current_epoch > 1:
        if (
            isinstance(epoch1_documents, bool)
            or not isinstance(epoch1_documents, int)
            or epoch1_documents <= 0
            or not isinstance(validation["epoch1_sha256"], str)
            or len(validation["epoch1_sha256"]) != 64
            or validation["external_state_bytes"]
            != epoch1_documents * _WholePoolPrefixVerifier._DIGEST_BYTES
        ):
            raise ValueError("whole-pool epoch-1 reference evidence drift")
        loaded_hist = {
            int(epoch): int(count)
            for epoch, count in margin["loaded_documents_by_epoch"].items()
        }
        if (
            epoch1_documents != loaded_hist[1]
            or any(
                loaded_hist[epoch] != epoch1_documents
                for epoch in range(1, current_epoch)
            )
            or loaded_hist[current_epoch] > epoch1_documents
        ):
            raise ValueError("whole-pool epoch document cardinality drift")
    elif (
        validation["epoch1_sha256"] is not None
        or validation["epoch1_documents"] is not None
        or validation["verified_complete_epochs"] != 0
        or validation["external_state_bytes"]
        != current_documents * _WholePoolPrefixVerifier._DIGEST_BYTES
    ):
        raise ValueError("right-censored epoch-1 validation evidence drift")


def summarize_repetition_world(
    rank_results: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    B: int,
    T: int,
    global_batch_tokens: int,
    horizon_optimizer_steps: Mapping[str, int],
    buffer_size: int,
    preferred_min_first_epoch_packed_positions: int,
    hard_min_first_epoch_packed_positions: int,
    preferred_max_loaded_epoch: int,
    preferred_max_consumed_epoch: int,
    hard_max_loaded_epoch: int,
    hard_max_consumed_epoch: int,
) -> dict[str, Any]:
    if len(rank_results) != world_size:
        raise ValueError("repeat world result count differs from world_size")
    denominator = world_size * B * T
    if global_batch_tokens % denominator:
        raise ValueError("global batch is not divisible by rank-local microbatch")
    accumulation = global_batch_tokens // denominator
    rank_horizon_microbatches = {
        name: steps * accumulation for name, steps in horizon_optimizer_steps.items()
    }
    for rank in rank_results:
        if not isinstance(rank, Mapping):
            raise ValueError("repeat rank evidence must be a mapping")
        _validate_repetition_rank_evidence(
            rank,
            snapshot_microbatches=rank_horizon_microbatches,
            B=B,
            T=T,
        )
    maximum_microbatches = max(horizon_optimizer_steps.values()) * accumulation
    first_wrap_by_rank = [
        (
            maximum_microbatches
            if item["first_epoch2_load_before_microbatch"] is None
            else int(item["first_epoch2_load_before_microbatch"])
        )
        for item in rank_results
    ]
    common_first_epoch_microbatches = min(first_wrap_by_rank)
    first_epoch_optimizer_steps = common_first_epoch_microbatches // accumulation
    first_epoch_packed_positions = first_epoch_optimizer_steps * global_batch_tokens

    horizons: dict[str, Any] = {}
    for name, optimizer_steps in horizon_optimizer_steps.items():
        snapshots = [item["snapshots"][name] for item in rank_results]
        retained = _metric_counters()
        cropped = _metric_counters()
        consumed = _metric_counters()
        for snapshot in snapshots:
            _merge_metric_counters(retained, snapshot["retained_positions"])
            _merge_metric_counters(cropped, snapshot["cropped_tokens"])
            _merge_metric_counters(consumed, snapshot["consumed_source_elements"])
        retained_total = sum(retained["mixture"].values())
        expected_positions = optimizer_steps * global_batch_tokens
        if retained_total != expected_positions:
            raise ValueError("repeat world retained-position attribution drift")
        realized_mix = {
            key: value / retained_total
            for key, value in sorted(retained["mixture"].items())
        }
        retention_efficiency: dict[str, float] = {}
        for key in sorted(set(consumed["mixture"]) | set(cropped["mixture"])):
            source_cost = consumed["mixture"][key] + cropped["mixture"][key]
            if source_cost:
                retention_efficiency[key] = retained["mixture"][key] / source_cost
        max_loaded_epoch = max(int(item["max_loaded_epoch"]) for item in snapshots)
        max_consumed_epoch = max(int(item["max_consumed_epoch"]) for item in snapshots)
        tier = classify_repetition_tier(
            first_epoch_packed_positions=first_epoch_packed_positions,
            max_loaded_epoch=max_loaded_epoch,
            max_consumed_epoch=max_consumed_epoch,
            preferred_min_first_epoch_packed_positions=(
                preferred_min_first_epoch_packed_positions
            ),
            hard_min_first_epoch_packed_positions=hard_min_first_epoch_packed_positions,
            preferred_max_loaded_epoch=preferred_max_loaded_epoch,
            preferred_max_consumed_epoch=preferred_max_consumed_epoch,
            hard_max_loaded_epoch=hard_max_loaded_epoch,
            hard_max_consumed_epoch=hard_max_consumed_epoch,
        )
        horizons[name] = {
            "optimizer_steps": optimizer_steps,
            "scheduled_token_positions": expected_positions,
            "microbatches_per_rank": optimizer_steps * accumulation,
            "max_loaded_epoch": max_loaded_epoch,
            "max_consumed_epoch": max_consumed_epoch,
            "max_base_locator_load_count": max_loaded_epoch,
            "max_base_locator_consume_count": max_consumed_epoch,
            "epoch5_loaded_including_prefetch": max_loaded_epoch >= 5,
            "repetition_tier": tier,
            "manual_repetition_risk_approval_required": tier == "manual_risk",
            "capacity_floor_passed": tier != "failed",
            "realized_retained_mix": realized_mix,
            "retention_efficiency_by_mixture": retention_efficiency,
            "retained_positions_by_mixture": dict(sorted(retained["mixture"].items())),
            "retained_positions_by_source": dict(sorted(retained["source"].items())),
            "retained_positions_by_register": dict(sorted(retained["register"].items())),
            "rank_snapshots": snapshots,
        }
    margin = horizons["s40_margin"]
    return {
        "world_size": world_size,
        "device_batch_sequences": B,
        "max_seq_len": T,
        "buffer_size": buffer_size,
        "gradient_accumulation_steps": accumulation,
        "accumulation_evidence": {
            "global_batch_tokens": global_batch_tokens,
            "world_size": world_size,
            "rank_microbatch_tokens": B * T,
            "world_microbatch_tokens": denominator,
            "gradient_accumulation_steps": accumulation,
            "identity": "global_batch_tokens=world_size*rank_microbatch_tokens*gradient_accumulation_steps",
        },
        "rank_count": len(rank_results),
        "whole_pool_repetition_only": True,
        "source_specific_repetition": False,
        "rank_sharding": "parquet_row_group_index_mod_world_size_fixed_across_epochs",
        "first_epoch2_load_before_microbatch_by_rank": [
            item["first_epoch2_load_before_microbatch"] for item in rank_results
        ],
        "first_epoch_common_complete_microbatches_per_rank": (
            common_first_epoch_microbatches
        ),
        "first_epoch_common_complete_optimizer_steps": first_epoch_optimizer_steps,
        "first_epoch_packed_positions": first_epoch_packed_positions,
        "first_epoch_packed_positions_semantics": (
            "complete_global_optimizer_step_prefix_before_any_rank_loads_epoch_2_"
            "including_refill_prefetch"
        ),
        "horizons": horizons,
        "repetition_tier": margin["repetition_tier"],
        "manual_repetition_risk_approval_required": margin[
            "manual_repetition_risk_approval_required"
        ],
        "capacity_floor_passed": margin["capacity_floor_passed"],
        "max_base_locator_load_count": margin["max_base_locator_load_count"],
        "max_base_locator_consume_count": margin["max_base_locator_consume_count"],
        "rank_evidence": [dict(item) for item in rank_results],
    }


def simulate_final_corpus_repetition_capacity(
    root: str | Path,
    train_files: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    world_sizes: Sequence[int] = D32_REPETITION_WORLD_SIZES,
    B: int = D32_REPETITION_DEVICE_BATCH_SEQUENCES,
    T: int = D32_REPETITION_MAX_SEQ_LEN,
    buffer_size: int = D32_REPETITION_BUFFER_SIZE,
    tokenizer_batch_size: int = D32_REPETITION_TOKENIZER_BATCH_SIZE,
    global_batch_tokens: int = D32_REPETITION_GLOBAL_BATCH_TOKENS,
    horizon_optimizer_steps: Mapping[str, int] = D32_REPETITION_HORIZON_OPTIMIZER_STEPS,
    preferred_min_first_epoch_packed_positions: int = (
        PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS
    ),
    hard_min_first_epoch_packed_positions: int = HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS,
    preferred_max_loaded_epoch: int = PREFERRED_MAX_LOADED_EPOCH,
    preferred_max_consumed_epoch: int = PREFERRED_MAX_CONSUMED_EPOCH,
    hard_max_loaded_epoch: int = HARD_MAX_LOADED_EPOCH,
    hard_max_consumed_epoch: int = HARD_MAX_CONSUMED_EPOCH,
) -> dict[str, Any]:
    """Run the v3 exact bounded-memory whole-pool repeat capacity gate."""

    frozen_arguments = {
        "world_sizes": list(world_sizes),
        "device_batch_sequences": B,
        "max_seq_len": T,
        "buffer_size": buffer_size,
        "tokenizer_batch_size": tokenizer_batch_size,
        "global_batch_tokens": global_batch_tokens,
        "horizon_optimizer_steps": dict(horizon_optimizer_steps),
        "preferred_min_first_epoch_packed_positions": (
            preferred_min_first_epoch_packed_positions
        ),
        "hard_min_first_epoch_packed_positions": hard_min_first_epoch_packed_positions,
        "preferred_max_loaded_epoch": preferred_max_loaded_epoch,
        "preferred_max_consumed_epoch": preferred_max_consumed_epoch,
        "hard_max_loaded_epoch": hard_max_loaded_epoch,
        "hard_max_consumed_epoch": hard_max_consumed_epoch,
    }
    expected_arguments = {
        "world_sizes": list(D32_REPETITION_WORLD_SIZES),
        "device_batch_sequences": D32_REPETITION_DEVICE_BATCH_SEQUENCES,
        "max_seq_len": D32_REPETITION_MAX_SEQ_LEN,
        "buffer_size": D32_REPETITION_BUFFER_SIZE,
        "tokenizer_batch_size": D32_REPETITION_TOKENIZER_BATCH_SIZE,
        "global_batch_tokens": D32_REPETITION_GLOBAL_BATCH_TOKENS,
        "horizon_optimizer_steps": D32_REPETITION_HORIZON_OPTIMIZER_STEPS,
        "preferred_min_first_epoch_packed_positions": (
            PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS
        ),
        "hard_min_first_epoch_packed_positions": HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS,
        "preferred_max_loaded_epoch": PREFERRED_MAX_LOADED_EPOCH,
        "preferred_max_consumed_epoch": PREFERRED_MAX_CONSUMED_EPOCH,
        "hard_max_loaded_epoch": HARD_MAX_LOADED_EPOCH,
        "hard_max_consumed_epoch": HARD_MAX_CONSUMED_EPOCH,
    }
    if canonical_json_bytes(frozen_arguments) != canonical_json_bytes(expected_arguments):
        raise ValueError("repeat simulator arguments differ from the frozen d32 contract")
    if not train_files:
        raise ValueError("repeat simulator requires at least one train file")
    corpus_root = Path(root)
    for file_record in train_files:
        path = corpus_root / str(file_record["path"])
        if (
            path.is_symlink()
            or path.stat().st_size != int(file_record["size_bytes"])
            or file_sha256(path) != file_record["sha256"]
        ):
            raise ValueError(f"packing simulation file drift: {file_record['path']}")
    parity = run_upstream_loader_parity_fixture()
    repeated_epoch_parity = run_upstream_repeated_epoch_parity_fixture()
    worlds: dict[str, Any] = {}
    for raw_world_size in world_sizes:
        world_size = raw_world_size
        denominator = world_size * B * T
        if global_batch_tokens % denominator:
            raise ValueError("global batch is not divisible by rank-local microbatch")
        accumulation = global_batch_tokens // denominator
        rank_horizons = {
            name: steps * accumulation
            for name, steps in horizon_optimizer_steps.items()
        }
        ranks = [
            simulate_bestfit_rank_repeated(
                _rank_repeated_document_batches(
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
                snapshot_microbatches=rank_horizons,
            )
            for rank in range(world_size)
        ]
        worlds[str(world_size)] = summarize_repetition_world(
            ranks,
            world_size=world_size,
            B=B,
            T=T,
            global_batch_tokens=global_batch_tokens,
            horizon_optimizer_steps=horizon_optimizer_steps,
            buffer_size=buffer_size,
            preferred_min_first_epoch_packed_positions=(
                preferred_min_first_epoch_packed_positions
            ),
            hard_min_first_epoch_packed_positions=(
                hard_min_first_epoch_packed_positions
            ),
            preferred_max_loaded_epoch=preferred_max_loaded_epoch,
            preferred_max_consumed_epoch=preferred_max_consumed_epoch,
            hard_max_loaded_epoch=hard_max_loaded_epoch,
            hard_max_consumed_epoch=hard_max_consumed_epoch,
        )
    tier_order = {"preferred": 0, "manual_risk": 1, "failed": 2}
    overall_tier = max(
        (item["repetition_tier"] for item in worlds.values()),
        key=tier_order.__getitem__,
    )
    return {
        "implementation": REPETITION_PACKING_SIMULATOR_ID,
        "implementation_file_sha256": file_sha256(Path(__file__)),
        "world_sizes": list(D32_REPETITION_WORLD_SIZES),
        "device_batch_sequences": B,
        "max_seq_len": T,
        "tokenizer_batch_size": tokenizer_batch_size,
        "buffer_size": buffer_size,
        "global_batch_tokens": global_batch_tokens,
        "whole_pool_repetition_only": True,
        "source_specific_repetition": False,
        "manual_repetition_risk_approval_between_tiers": True,
        "upstream_contract": {
            "nanochat_revision": PINNED_UPSTREAM_REVISION,
            "tokenizer_batch_size": tokenizer_batch_size,
            "tokenizer_threads": 4,
            "refill_buffer_size": buffer_size,
            "tie_breaks": "first_largest_fit_else_first_shortest",
            "cropped_tail_policy": "discard",
            "rank_sharding": "parquet_row_group_index_mod_world_size",
            "epoch_order": "repeat_identical_whole_rank_shard",
            "whole_pool_repetition_only": True,
            "source_specific_repetition": False,
        },
        "fixture_parity": parity,
        "repeated_epoch_fixture_parity": repeated_epoch_parity,
        "horizon_optimizer_steps": dict(horizon_optimizer_steps),
        "thresholds": {
            "preferred_min_first_epoch_packed_positions": (
                preferred_min_first_epoch_packed_positions
            ),
            "hard_min_first_epoch_packed_positions": (
                hard_min_first_epoch_packed_positions
            ),
            "preferred_max_loaded_epoch": preferred_max_loaded_epoch,
            "preferred_max_consumed_epoch": preferred_max_consumed_epoch,
            "hard_max_loaded_epoch": hard_max_loaded_epoch,
            "hard_max_consumed_epoch": hard_max_consumed_epoch,
        },
        "worlds": worlds,
        "repetition_tier": overall_tier,
        "manual_repetition_risk_approval_required": overall_tier == "manual_risk",
        "capacity_floor_passed": overall_tier != "failed",
        "all_worlds_pass": overall_tier != "failed",
        "max_base_locator_load_count": max(
            int(item["max_base_locator_load_count"]) for item in worlds.values()
        ),
        "max_base_locator_consume_count": max(
            int(item["max_base_locator_consume_count"])
            for item in worlds.values()
        ),
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


def validate_repetition_capacity_simulation(
    simulation: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute and validate every frozen v3 world and horizon summary."""

    expected_fields = {
        "implementation",
        "implementation_file_sha256",
        "world_sizes",
        "device_batch_sequences",
        "max_seq_len",
        "tokenizer_batch_size",
        "buffer_size",
        "global_batch_tokens",
        "whole_pool_repetition_only",
        "source_specific_repetition",
        "manual_repetition_risk_approval_between_tiers",
        "upstream_contract",
        "fixture_parity",
        "repeated_epoch_fixture_parity",
        "horizon_optimizer_steps",
        "thresholds",
        "worlds",
        "repetition_tier",
        "manual_repetition_risk_approval_required",
        "capacity_floor_passed",
        "all_worlds_pass",
        "max_base_locator_load_count",
        "max_base_locator_consume_count",
    }
    if set(simulation) != expected_fields:
        raise ValueError("repetition simulation fields differ from frozen v3 schema")
    if simulation["implementation"] != REPETITION_PACKING_SIMULATOR_ID:
        raise ValueError("unexpected repetition capacity implementation")
    if simulation["implementation_file_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("repetition simulator implementation file hash drift")
    frozen_scalars = {
        "world_sizes": list(D32_REPETITION_WORLD_SIZES),
        "device_batch_sequences": D32_REPETITION_DEVICE_BATCH_SEQUENCES,
        "max_seq_len": D32_REPETITION_MAX_SEQ_LEN,
        "tokenizer_batch_size": D32_REPETITION_TOKENIZER_BATCH_SIZE,
        "buffer_size": D32_REPETITION_BUFFER_SIZE,
        "global_batch_tokens": D32_REPETITION_GLOBAL_BATCH_TOKENS,
        "whole_pool_repetition_only": True,
        "source_specific_repetition": False,
        "manual_repetition_risk_approval_between_tiers": True,
    }
    for key, expected in frozen_scalars.items():
        if canonical_json_bytes(simulation[key]) != canonical_json_bytes(expected):
            raise ValueError(f"repetition simulation {key} drift")
    expected_upstream_contract = {
        "nanochat_revision": PINNED_UPSTREAM_REVISION,
        "tokenizer_batch_size": D32_REPETITION_TOKENIZER_BATCH_SIZE,
        "tokenizer_threads": 4,
        "refill_buffer_size": D32_REPETITION_BUFFER_SIZE,
        "tie_breaks": "first_largest_fit_else_first_shortest",
        "cropped_tail_policy": "discard",
        "rank_sharding": "parquet_row_group_index_mod_world_size",
        "epoch_order": "repeat_identical_whole_rank_shard",
        "whole_pool_repetition_only": True,
        "source_specific_repetition": False,
    }
    if canonical_json_bytes(simulation["upstream_contract"]) != canonical_json_bytes(
        expected_upstream_contract
    ):
        raise ValueError("repetition upstream loader contract drift")
    if canonical_json_bytes(simulation["fixture_parity"]) != canonical_json_bytes(
        run_upstream_loader_parity_fixture()
    ):
        raise ValueError("legacy upstream parity evidence drift")
    if (
        canonical_json_bytes(simulation["repeated_epoch_fixture_parity"])
        != canonical_json_bytes(run_upstream_repeated_epoch_parity_fixture())
    ):
        raise ValueError("repeated epoch-boundary parity evidence drift")
    if canonical_json_bytes(simulation["horizon_optimizer_steps"]) != canonical_json_bytes(
        D32_REPETITION_HORIZON_OPTIMIZER_STEPS
    ):
        raise ValueError("repetition optimizer horizons drift")
    expected_thresholds = {
        "preferred_min_first_epoch_packed_positions": (
            PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS
        ),
        "hard_min_first_epoch_packed_positions": HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS,
        "preferred_max_loaded_epoch": PREFERRED_MAX_LOADED_EPOCH,
        "preferred_max_consumed_epoch": PREFERRED_MAX_CONSUMED_EPOCH,
        "hard_max_loaded_epoch": HARD_MAX_LOADED_EPOCH,
        "hard_max_consumed_epoch": HARD_MAX_CONSUMED_EPOCH,
    }
    if canonical_json_bytes(simulation["thresholds"]) != canonical_json_bytes(
        expected_thresholds
    ):
        raise ValueError("repetition thresholds drift")

    worlds = simulation["worlds"]
    if not isinstance(worlds, Mapping) or set(worlds) != {"8", "16"}:
        raise ValueError("repetition simulation requires exactly ws8 and ws16")
    recomputed_worlds: dict[str, dict[str, Any]] = {}
    observed_tiers: list[str] = []
    for world_size in D32_REPETITION_WORLD_SIZES:
        key = str(world_size)
        world = worlds[key]
        if not isinstance(world, Mapping):
            raise ValueError("repetition world evidence must be a mapping")
        rank_evidence = world.get("rank_evidence")
        if not isinstance(rank_evidence, list) or len(rank_evidence) != world_size:
            raise ValueError(f"ws{world_size} requires complete per-rank evidence")
        recomputed = summarize_repetition_world(
            rank_evidence,
            world_size=world_size,
            B=D32_REPETITION_DEVICE_BATCH_SEQUENCES,
            T=D32_REPETITION_MAX_SEQ_LEN,
            global_batch_tokens=D32_REPETITION_GLOBAL_BATCH_TOKENS,
            horizon_optimizer_steps=D32_REPETITION_HORIZON_OPTIMIZER_STEPS,
            buffer_size=D32_REPETITION_BUFFER_SIZE,
            preferred_min_first_epoch_packed_positions=(
                PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS
            ),
            hard_min_first_epoch_packed_positions=(
                HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS
            ),
            preferred_max_loaded_epoch=PREFERRED_MAX_LOADED_EPOCH,
            preferred_max_consumed_epoch=PREFERRED_MAX_CONSUMED_EPOCH,
            hard_max_loaded_epoch=HARD_MAX_LOADED_EPOCH,
            hard_max_consumed_epoch=HARD_MAX_CONSUMED_EPOCH,
        )
        if canonical_json_bytes(world) != canonical_json_bytes(recomputed):
            raise ValueError(f"ws{world_size} summary differs from rank evidence")
        expected_accumulation = 32 if world_size == 8 else 16
        if (
            recomputed["gradient_accumulation_steps"] != expected_accumulation
            or recomputed["rank_count"] != world_size
            or recomputed["accumulation_evidence"]["gradient_accumulation_steps"]
            != expected_accumulation
        ):
            raise ValueError(f"ws{world_size} accumulation evidence drift")
        recomputed_worlds[key] = recomputed
        observed_tiers.append(recomputed["repetition_tier"])

    tier_order = {"preferred": 0, "manual_risk": 1, "failed": 2}
    overall_tier = max(observed_tiers, key=tier_order.__getitem__)
    max_load = max(
        int(world["max_base_locator_load_count"])
        for world in recomputed_worlds.values()
    )
    max_consume = max(
        int(world["max_base_locator_consume_count"])
        for world in recomputed_worlds.values()
    )
    expected_top = {
        "repetition_tier": overall_tier,
        "manual_repetition_risk_approval_required": overall_tier == "manual_risk",
        "capacity_floor_passed": overall_tier != "failed",
        "all_worlds_pass": overall_tier != "failed",
        "max_base_locator_load_count": max_load,
        "max_base_locator_consume_count": max_consume,
    }
    for key, expected in expected_top.items():
        if simulation[key] != expected:
            raise ValueError(f"top-level repetition {key} drift")
    return {
        "simulation_sha256": hashlib.sha256(canonical_json_bytes(simulation)).hexdigest(),
        "repetition_tier": overall_tier,
        "capacity_floor_passed": overall_tier != "failed",
        "worlds": recomputed_worlds,
        "max_base_locator_load_count": max_load,
        "max_base_locator_consume_count": max_consume,
    }


def seal_repetition_risk_approval(
    output_path: str | Path,
    *,
    dataset_manifest_sha256: str,
    tokenizer_package_sha256: str,
    simulation_sha256: str,
    approver: str,
    rationale: str,
) -> dict[str, Any]:
    """Seal a distinct, reviewable approval for a manual-risk simulation."""

    for location, value in (
        ("dataset_manifest_sha256", dataset_manifest_sha256),
        ("tokenizer_package_sha256", tokenizer_package_sha256),
        ("simulation_sha256", simulation_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{location} must be a lowercase SHA-256")
    if not isinstance(approver, str) or not approver.strip():
        raise ValueError("manual-risk approval requires a named approver")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("manual-risk approval requires a rationale")
    approval = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "turkish_repetition_risk_approval",
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "tokenizer_package_sha256": tokenizer_package_sha256,
            "simulation_sha256": simulation_sha256,
            "decision": "approved",
            "approver": approver.strip(),
            "rationale": rationale.strip(),
            "canonical_sha256": None,
        }
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite repetition approval: {destination}")
    write_json_atomic(destination, approval)
    return approval


def validate_repetition_risk_approval(
    approval: Mapping[str, Any],
    *,
    dataset_manifest_sha256: str,
    tokenizer_package_sha256: str,
    simulation_sha256: str,
) -> str:
    """Validate approval identity and return its canonical artifact hash."""

    if not isinstance(approval, Mapping):
        raise ValueError("manual-risk approval must be a sealed mapping artifact")
    if set(approval) != {
        "schema_version",
        "kind",
        "dataset_manifest_sha256",
        "tokenizer_package_sha256",
        "simulation_sha256",
        "decision",
        "approver",
        "rationale",
        "canonical_sha256",
    }:
        raise ValueError("manual-risk approval fields differ from frozen schema")
    from nanochat.experiment_manifest import verify_manifest_hash

    verify_manifest_hash(approval)
    if (
        approval["schema_version"] != "1.0"
        or approval["kind"] != "turkish_repetition_risk_approval"
        or approval["decision"] != "approved"
        or approval["dataset_manifest_sha256"] != dataset_manifest_sha256
        or approval["tokenizer_package_sha256"] != tokenizer_package_sha256
        or approval["simulation_sha256"] != simulation_sha256
        or not isinstance(approval["approver"], str)
        or not approval["approver"].strip()
        or not isinstance(approval["rationale"], str)
        or not approval["rationale"].strip()
    ):
        raise ValueError("manual-risk approval is invalid or bound to other artifacts")
    return str(approval["canonical_sha256"])


def seal_repetition_capacity_receipt(
    output_path: str | Path,
    *,
    simulation: Mapping[str, Any],
    dataset_manifest_sha256: str,
    tokenizer_package_sha256: str,
    intended_weights: Mapping[str, float],
    manual_repetition_risk_approval: Mapping[str, Any] | None = None,
    mix_absolute_tolerance: float = D32_REPETITION_MIX_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Seal a v3 receipt from recomputed evidence and an optional approval."""

    payload = _recompute_repetition_capacity_receipt_payload(
        simulation=simulation,
        dataset_manifest_sha256=dataset_manifest_sha256,
        tokenizer_package_sha256=tokenizer_package_sha256,
        intended_weights=intended_weights,
        manual_repetition_risk_approval=manual_repetition_risk_approval,
        mix_absolute_tolerance=mix_absolute_tolerance,
    )
    receipt = seal_manifest({**payload, "canonical_sha256": None})
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite capacity receipt: {destination}")
    write_json_atomic(destination, receipt)
    return receipt


def _recompute_repetition_capacity_receipt_payload(
    *,
    simulation: Mapping[str, Any],
    dataset_manifest_sha256: str,
    tokenizer_package_sha256: str,
    intended_weights: Mapping[str, float],
    manual_repetition_risk_approval: Mapping[str, Any] | None,
    mix_absolute_tolerance: float,
) -> dict[str, Any]:
    for location, value in (
        ("dataset_manifest_sha256", dataset_manifest_sha256),
        ("tokenizer_package_sha256", tokenizer_package_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{location} must be a lowercase SHA-256")
    if (
        type(mix_absolute_tolerance) is not float
        or mix_absolute_tolerance != D32_REPETITION_MIX_ABSOLUTE_TOLERANCE
    ):
        raise ValueError("repetition mix tolerance differs from frozen contract")
    if not isinstance(intended_weights, Mapping) or not intended_weights:
        raise ValueError("intended repetition weights must be a non-empty mapping")
    normalized_weights: dict[str, float] = {}
    for mixture, raw_weight in intended_weights.items():
        if (
            not isinstance(mixture, str)
            or not mixture
            or isinstance(raw_weight, bool)
            or not isinstance(raw_weight, (int, float))
            or not math.isfinite(float(raw_weight))
            or float(raw_weight) < 0.0
        ):
            raise ValueError("intended repetition weights are invalid")
        normalized_weights[mixture] = float(raw_weight)
    if abs(sum(normalized_weights.values()) - 1.0) > 1e-12:
        raise ValueError("intended repetition weights must sum to one")

    validated = validate_repetition_capacity_simulation(simulation)
    worlds = validated["worlds"]
    deviations: dict[str, dict[str, dict[str, float]]] = {}
    mix_pass = True
    for world, world_metrics in worlds.items():
        deviations[world] = {}
        for horizon, metrics in world_metrics["horizons"].items():
            realized = metrics["realized_retained_mix"]
            if set(realized) != set(normalized_weights):
                raise ValueError("realized repetition mixture label coverage drift")
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in realized.values()
            ) or abs(sum(float(value) for value in realized.values()) - 1.0) > 1e-12:
                raise ValueError("realized repetition mixture is invalid")
            horizon_deviation = {
                mixture: float(realized[mixture]) - intended
                for mixture, intended in sorted(normalized_weights.items())
            }
            if any(
                abs(delta) > D32_REPETITION_MIX_ABSOLUTE_TOLERANCE
                for delta in horizon_deviation.values()
            ):
                mix_pass = False
            deviations[world][horizon] = horizon_deviation

    tier = validated["repetition_tier"]
    approval_required = tier == "manual_risk"
    simulation_sha256 = validated["simulation_sha256"]
    approval_sha256: str | None = None
    if manual_repetition_risk_approval is not None:
        if not isinstance(manual_repetition_risk_approval, Mapping):
            raise ValueError("manual-risk approval must be a sealed mapping artifact")
        if not approval_required:
            raise ValueError("manual-risk approval is forbidden outside manual-risk tier")
        approval_sha256 = validate_repetition_risk_approval(
            manual_repetition_risk_approval,
            dataset_manifest_sha256=dataset_manifest_sha256,
            tokenizer_package_sha256=tokenizer_package_sha256,
            simulation_sha256=simulation_sha256,
        )
    approval_satisfied = not approval_required or approval_sha256 is not None
    gate_passed = validated["capacity_floor_passed"] and mix_pass and approval_satisfied
    return {
        "schema_version": "3.0",
        "kind": "turkish_bestfit_repeat_capacity_receipt",
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "tokenizer_package_sha256": tokenizer_package_sha256,
        "simulation_sha256": simulation_sha256,
        "simulation": dict(simulation),
        "intended_mixture_weights": dict(sorted(normalized_weights.items())),
        "realized_minus_intended_by_world_and_horizon": deviations,
        "mix_absolute_tolerance": D32_REPETITION_MIX_ABSOLUTE_TOLERANCE,
        "mix_gate_passed": mix_pass,
        "repetition_tier": tier,
        "max_base_locator_load_count": validated["max_base_locator_load_count"],
        "max_base_locator_consume_count": validated[
            "max_base_locator_consume_count"
        ],
        "capacity_floor_passed": validated["capacity_floor_passed"],
        "approval_required": approval_required,
        "manual_repetition_risk_approval_sha256": approval_sha256,
        "approval_satisfied": approval_satisfied,
        "gate_passed": gate_passed,
        "cleanup_authorized": gate_passed,
    }


def validate_repetition_capacity_receipt(
    receipt: Mapping[str, Any],
    *,
    dataset_manifest_sha256: str,
    tokenizer_package_sha256: str,
    manual_repetition_risk_approval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and recompute a sealed receipt; return its gate summary."""

    from nanochat.experiment_manifest import verify_manifest_hash

    verify_manifest_hash(receipt)
    expected_fields = {
        "schema_version",
        "kind",
        "dataset_manifest_sha256",
        "tokenizer_package_sha256",
        "simulation_sha256",
        "simulation",
        "intended_mixture_weights",
        "realized_minus_intended_by_world_and_horizon",
        "mix_absolute_tolerance",
        "mix_gate_passed",
        "repetition_tier",
        "max_base_locator_load_count",
        "max_base_locator_consume_count",
        "capacity_floor_passed",
        "approval_required",
        "manual_repetition_risk_approval_sha256",
        "approval_satisfied",
        "gate_passed",
        "cleanup_authorized",
        "canonical_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("repetition receipt fields differ from frozen schema")
    if (
        receipt["schema_version"] != "3.0"
        or receipt["kind"] != "turkish_bestfit_repeat_capacity_receipt"
        or receipt["dataset_manifest_sha256"] != dataset_manifest_sha256
        or receipt["tokenizer_package_sha256"] != tokenizer_package_sha256
    ):
        raise ValueError("repetition receipt identity drift")
    payload = _recompute_repetition_capacity_receipt_payload(
        simulation=receipt["simulation"],
        dataset_manifest_sha256=dataset_manifest_sha256,
        tokenizer_package_sha256=tokenizer_package_sha256,
        intended_weights=receipt["intended_mixture_weights"],
        manual_repetition_risk_approval=manual_repetition_risk_approval,
        mix_absolute_tolerance=receipt["mix_absolute_tolerance"],
    )
    observed_payload = {
        key: value for key, value in receipt.items() if key != "canonical_sha256"
    }
    if canonical_json_bytes(observed_payload) != canonical_json_bytes(payload):
        raise ValueError("repetition receipt differs from recomputed evidence")
    return {
        "canonical_sha256": receipt["canonical_sha256"],
        "simulation_sha256": receipt["simulation_sha256"],
        "repetition_tier": receipt["repetition_tier"],
        "gate_passed": receipt["gate_passed"],
        "cleanup_authorized": receipt["cleanup_authorized"],
        "approval_required": receipt["approval_required"],
        "approval_satisfied": receipt["approval_satisfied"],
        "manual_repetition_risk_approval_sha256": receipt[
            "manual_repetition_risk_approval_sha256"
        ],
    }


__all__ = [
    "D32_REPETITION_BUFFER_SIZE",
    "D32_REPETITION_DEVICE_BATCH_SEQUENCES",
    "D32_REPETITION_GLOBAL_BATCH_TOKENS",
    "D32_REPETITION_HORIZON_OPTIMIZER_STEPS",
    "D32_REPETITION_MAX_SEQ_LEN",
    "D32_REPETITION_MIX_ABSOLUTE_TOLERANCE",
    "D32_REPETITION_TOKENIZER_BATCH_SIZE",
    "D32_REPETITION_WORLD_SIZES",
    "HARD_MAX_CONSUMED_EPOCH",
    "HARD_MAX_LOADED_EPOCH",
    "HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS",
    "PACKING_SIMULATOR_ID",
    "PREFERRED_MAX_CONSUMED_EPOCH",
    "PREFERRED_MAX_LOADED_EPOCH",
    "PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS",
    "REPETITION_PACKING_SIMULATOR_ID",
    "PackingDocument",
    "classify_repetition_tier",
    "run_upstream_loader_parity_fixture",
    "run_upstream_repeated_epoch_parity_fixture",
    "seal_capacity_receipt",
    "seal_repetition_capacity_receipt",
    "seal_repetition_risk_approval",
    "simulate_bestfit_rank",
    "simulate_bestfit_rank_repeated",
    "simulate_final_corpus_capacity",
    "simulate_final_corpus_repetition_capacity",
    "summarize_repetition_world",
    "summarize_world",
    "validate_repetition_capacity_receipt",
    "validate_repetition_capacity_simulation",
    "validate_repetition_risk_approval",
]
