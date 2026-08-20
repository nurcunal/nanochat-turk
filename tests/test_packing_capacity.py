from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.experiment_manifest import file_sha256, verify_manifest_hash
from nanochat.packing_capacity import (
    PackingDocument,
    run_upstream_loader_parity_fixture,
    seal_capacity_receipt,
    simulate_bestfit_rank,
    simulate_final_corpus_capacity,
    summarize_world,
)


def _doc(index: int, length: int = 4, mixture: str = "m") -> PackingDocument:
    return PackingDocument(length, f"d{index}", mixture, "s", "r")


def test_simulator_executes_against_byte_exact_pinned_upstream_trace():
    parity = run_upstream_loader_parity_fixture()
    assert parity["passed"] is True
    assert parity["actual_output_sha256"] == parity["simulated_output_sha256"]
    assert parity["upstream_commit"] == "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
    assert parity["upstream_loader_source_sha256"] == (
        "ed1d4997e3c407f242fbbcfa627f2987f8f34a3a0c340d792c4e10a202981990"
    )


def test_first_wrap_is_before_incomplete_batch_and_default_memory_is_bounded():
    result = simulate_bestfit_rank(
        [[_doc(0), _doc(1)]], B=1, T=3, buffer_size=2
    )
    assert result["completed_microbatches"] == 1
    assert result["first_wrap_before_microbatch"] == 1
    assert result["stop_reason"] == "first_epoch_exhausted_during_refill"
    assert "microbatches" not in result
    assert result["retained_positions"]["mixture"] == {"m": 3}


def test_margin_and_gradient_accumulation_require_every_rank():
    short = simulate_bestfit_rank(
        [[_doc(0), _doc(1)]], B=1, T=3, buffer_size=2
    )
    long = simulate_bestfit_rank(
        [[_doc(i) for i in range(8)]],
        B=1,
        T=3,
        buffer_size=2,
        max_microbatches=2,
    )
    failed = summarize_world(
        [long, short],
        world_size=2,
        B=1,
        T=3,
        global_batch_tokens=12,
        required_optimizer_steps=1,
        safety_margin_fraction=0.0,
        buffer_size=2,
    )
    assert failed["gradient_accumulation_steps"] == 2
    assert failed["passes_40x_no_wrap_with_margin"] is False
    assert failed["safe_global_scheduled_positions"] == 0

    passed = summarize_world(
        [long, long],
        world_size=2,
        B=1,
        T=3,
        global_batch_tokens=12,
        required_optimizer_steps=1,
        safety_margin_fraction=0.0,
        buffer_size=2,
    )
    assert passed["passes_40x_no_wrap_with_margin"] is True
    assert passed["safe_global_scheduled_positions"] == 12


class _LengthTokenizer:
    @staticmethod
    def get_bos_token_id() -> int:
        return 1

    @staticmethod
    def encode(texts, *, prepend, num_threads):
        assert prepend == 1
        assert num_threads == 4
        return [[prepend, *([2] * len(text))] for text in texts]


def test_row_group_starvation_fails_topology_gate(tmp_path: Path):
    schema = pa.schema(
        [
            ("text", pa.string()),
            ("document_id", pa.string()),
            ("mixture_id", pa.string()),
            ("source_id", pa.string()),
            ("register_bucket", pa.string()),
        ]
    )
    path = tmp_path / "train.parquet"
    rows = [
        {
            "text": "abc",
            "document_id": f"d{i}",
            "mixture_id": "m",
            "source_id": "s",
            "register_bucket": "r",
        }
        for i in range(12)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=12)
    simulation = simulate_final_corpus_capacity(
        tmp_path,
        [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        ],
        _LengthTokenizer(),
        world_sizes=(2,),
        B=1,
        T=3,
        buffer_size=2,
        tokenizer_batch_size=2,
        global_batch_tokens=6,
        required_optimizer_steps=1,
        safety_margin_fraction=0.0,
    )
    assert simulation["all_worlds_pass"] is False
    assert simulation["worlds"]["2"]["completed_microbatches_by_rank"][1] == 0


def test_capacity_receipt_never_authorizes_cleanup_if_one_world_fails(tmp_path: Path):
    base = {
        "realized_retained_mix": {"m": 1.0},
        "retention_efficiency_by_mixture": {"m": 0.8},
        "common_prefix_scheduled_positions": 100,
        "required_positions_with_margin": 100,
    }
    simulation = {
        "implementation": "fixture",
        "all_worlds_pass": False,
        "worlds": {
            "8": {**base, "passes_40x_no_wrap_with_margin": True},
            "16": {
                **base,
                "common_prefix_scheduled_positions": 90,
                "passes_40x_no_wrap_with_margin": False,
            },
        },
    }
    path = tmp_path / "capacity.json"
    receipt = seal_capacity_receipt(
        path,
        simulation=simulation,
        dataset_manifest_sha256="a" * 64,
        tokenizer_package_sha256="b" * 64,
        intended_weights={"m": 1.0},
        current_source_token_target=100,
    )
    verify_manifest_hash(json.loads(path.read_text()))
    assert receipt["gate_passed"] is False
    assert receipt["cleanup_authorized"] is False
    assert receipt["mix_gate_evaluated_on_common_horizon"] is False
