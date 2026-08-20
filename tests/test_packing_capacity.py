from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.experiment_manifest import file_sha256, verify_manifest_hash
from nanochat.packing_capacity import (
    D32_REPETITION_BUFFER_SIZE,
    D32_REPETITION_DEVICE_BATCH_SEQUENCES,
    D32_REPETITION_GLOBAL_BATCH_TOKENS,
    D32_REPETITION_HORIZON_OPTIMIZER_STEPS,
    D32_REPETITION_MAX_SEQ_LEN,
    D32_REPETITION_TOKENIZER_BATCH_SIZE,
    HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS,
    PackingDocument,
    PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS,
    REPETITION_PACKING_SIMULATOR_ID,
    classify_repetition_tier,
    run_upstream_loader_parity_fixture,
    run_upstream_repeated_epoch_parity_fixture,
    seal_capacity_receipt,
    seal_repetition_capacity_receipt,
    seal_repetition_risk_approval,
    simulate_bestfit_rank,
    simulate_bestfit_rank_repeated,
    simulate_final_corpus_capacity,
    simulate_final_corpus_repetition_capacity,
    summarize_repetition_world,
    summarize_world,
    validate_repetition_capacity_receipt,
    validate_repetition_capacity_simulation,
)


def _doc(index: int, length: int = 4, mixture: str = "m") -> PackingDocument:
    return PackingDocument(length, f"d{index}", mixture, "s", "r")


def test_frozen_d32_repeat_horizons_and_thresholds_are_exact():
    assert D32_REPETITION_HORIZON_OPTIMIZER_STEPS == {
        "s12": 9_600,
        "s20": 16_000,
        "s40": 32_000,
        "s40_margin": 32_640,
    }
    assert PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS == 34_225_520_640
    assert HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS == 17_112_760_320


def test_simulator_executes_against_byte_exact_pinned_upstream_trace():
    parity = run_upstream_loader_parity_fixture()
    assert parity["passed"] is True
    assert parity["actual_output_sha256"] == parity["simulated_output_sha256"]
    assert parity["upstream_commit"] == "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
    assert parity["upstream_loader_source_sha256"] == (
        "ed1d4997e3c407f242fbbcfa627f2987f8f34a3a0c340d792c4e10a202981990"
    )


def test_repeat_simulator_crosses_epoch_boundaries_against_pinned_upstream():
    parity = run_upstream_repeated_epoch_parity_fixture()
    assert parity["passed"] is True
    assert parity["fixture_sha256"] == (
        "fa4865e88faa2dbc3d8ddf61c58d5ba5c84034a0c1649ddb7fc6b5a50bf5db03"
    )
    assert parity["actual_output_sha256"] == (
        "deb284d3292f645603495abc6e6c367052e172e4d7cb4b45efab980cbe4616c7"
    )
    assert parity["simulated_output_sha256"] == parity["actual_output_sha256"]


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


def _repeat_batches(*, documents: int = 2, changed_epoch: int | None = None):
    epoch = 1
    while True:
        order = list(range(documents))
        if epoch == changed_epoch:
            order.reverse()
        yield [
            PackingDocument(
                4,
                f"d{index}",
                "m",
                "s",
                "r",
                base_locator=f"locator-{index}",
                epoch=epoch,
            )
            for index in order
        ]
        epoch += 1


def test_repeat_simulator_models_prefetch_and_reports_bounded_exact_metrics():
    result = simulate_bestfit_rank_repeated(
        _repeat_batches(),
        B=1,
        T=3,
        buffer_size=2,
        snapshot_microbatches={"first": 1, "prefetched": 2, "epoch5": 8},
    )
    first = result["snapshots"]["first"]
    assert first["buffer_epoch_histogram"] == {"1": 1}
    assert first["max_loaded_epoch"] == 1

    # Epoch 2 is loaded before microbatch 1 while the remaining epoch-1
    # document is still buffered, then consumed only in microbatch 2.
    prefetched = result["snapshots"]["prefetched"]
    assert result["first_epoch2_load_before_microbatch"] == 1
    assert prefetched["first_load_before_microbatch_by_epoch"]["2"] == 1
    assert prefetched["max_loaded_epoch"] == 2
    assert prefetched["max_consumed_epoch"] == 1
    assert prefetched["buffer_epoch_histogram"] == {"2": 2}

    epoch5 = result["snapshots"]["epoch5"]
    assert epoch5["max_loaded_epoch"] == 5
    assert epoch5["max_base_locator_load_count"] == 5
    assert epoch5["epoch5_loaded_including_prefetch"] is True
    assert result["whole_pool_order_validation"]["method"] == (
        "bounded_disk_exact_epoch_and_partial_prefix_sha256_v2"
    )
    assert result["whole_pool_order_validation"][
        "current_partial_prefix_verified_against_epoch1"
    ] is True


def test_repeat_simulator_rejects_source_specific_or_reordered_epoch():
    try:
        simulate_bestfit_rank_repeated(
            _repeat_batches(changed_epoch=2),
            B=1,
            T=3,
            buffer_size=2,
            snapshot_microbatches={"after_second_epoch": 5},
        )
    except ValueError as exc:
        assert "differs from epoch 1" in str(exc)
    else:
        raise AssertionError("a reordered repeat epoch must fail closed")


def test_repeat_horizon_snapshots_do_not_change_final_state():
    with_intermediate = simulate_bestfit_rank_repeated(
        _repeat_batches(),
        B=1,
        T=3,
        buffer_size=2,
        snapshot_microbatches={"resume_boundary": 3, "final": 6},
    )
    direct = simulate_bestfit_rank_repeated(
        _repeat_batches(),
        B=1,
        T=3,
        buffer_size=2,
        snapshot_microbatches={"final": 6},
    )
    assert with_intermediate["snapshots"]["final"] == direct["snapshots"]["final"]
    assert with_intermediate["whole_pool_order_validation"] == direct[
        "whole_pool_order_validation"
    ]


def test_repeat_world_uses_the_earliest_rank_wrap_for_topology_capacity():
    horizon_steps = {"s12": 1, "s20": 2, "s40": 3, "s40_margin": 4}
    rank_horizons = dict(horizon_steps)  # accumulation is one in this fixture
    short = simulate_bestfit_rank_repeated(
        _repeat_batches(documents=2),
        B=1,
        T=3,
        buffer_size=2,
        snapshot_microbatches=rank_horizons,
    )
    long = simulate_bestfit_rank_repeated(
        _repeat_batches(documents=4),
        B=1,
        T=3,
        buffer_size=2,
        snapshot_microbatches=rank_horizons,
    )
    world = summarize_repetition_world(
        [long, short],
        world_size=2,
        B=1,
        T=3,
        global_batch_tokens=6,
        horizon_optimizer_steps=horizon_steps,
        buffer_size=2,
        preferred_min_first_epoch_packed_positions=12,
        hard_min_first_epoch_packed_positions=6,
        preferred_max_loaded_epoch=2,
        preferred_max_consumed_epoch=2,
        hard_max_loaded_epoch=4,
        hard_max_consumed_epoch=4,
    )
    assert world["first_epoch2_load_before_microbatch_by_rank"] == [3, 1]
    assert world["first_epoch_common_complete_optimizer_steps"] == 1
    assert world["first_epoch_packed_positions"] == 6
    assert world["repetition_tier"] == "manual_risk"


def test_repetition_tier_boundaries_and_epoch5_are_fail_closed():
    common = {
        "preferred_min_first_epoch_packed_positions": (
            PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS
        ),
        "hard_min_first_epoch_packed_positions": HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS,
        "preferred_max_loaded_epoch": 2,
        "preferred_max_consumed_epoch": 2,
        "hard_max_loaded_epoch": 4,
        "hard_max_consumed_epoch": 4,
    }
    assert classify_repetition_tier(
        first_epoch_packed_positions=PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS,
        max_loaded_epoch=2,
        max_consumed_epoch=2,
        **common,
    ) == "preferred"
    assert classify_repetition_tier(
        first_epoch_packed_positions=PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS - 1,
        max_loaded_epoch=2,
        max_consumed_epoch=2,
        **common,
    ) == "manual_risk"
    assert classify_repetition_tier(
        first_epoch_packed_positions=HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS,
        max_loaded_epoch=4,
        max_consumed_epoch=4,
        **common,
    ) == "manual_risk"
    assert classify_repetition_tier(
        first_epoch_packed_positions=HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS - 1,
        max_loaded_epoch=1,
        max_consumed_epoch=1,
        **common,
    ) == "failed"
    assert classify_repetition_tier(
        first_epoch_packed_positions=PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS,
        max_loaded_epoch=5,
        max_consumed_epoch=4,
        **common,
    ) == "failed"


def _synthetic_rank_evidence(world_size: int) -> dict:
    accumulation = 32 if world_size == 8 else 16
    first_wrap_microbatch = 8_200 * accumulation
    pool_documents = first_wrap_microbatch * D32_REPETITION_DEVICE_BATCH_SEQUENCES + 1_000
    snapshots = {}
    for name, optimizer_steps in D32_REPETITION_HORIZON_OPTIMIZER_STEPS.items():
        completed = optimizer_steps * accumulation
        consumed_documents = completed * D32_REPETITION_DEVICE_BATCH_SEQUENCES
        loaded_documents = consumed_documents + 1_000
        max_epoch = (loaded_documents + pool_documents - 1) // pool_documents
        loaded_hist = {
            epoch: pool_documents for epoch in range(1, max_epoch)
        }
        loaded_hist[max_epoch] = loaded_documents - pool_documents * (max_epoch - 1)
        consumed_hist = dict(loaded_hist)
        consumed_hist[max_epoch] -= 1_000
        scheduled = completed * D32_REPETITION_DEVICE_BATCH_SEQUENCES * D32_REPETITION_MAX_SEQ_LEN
        row_leading = completed * D32_REPETITION_DEVICE_BATCH_SEQUENCES
        consumed_source = scheduled + row_leading
        snapshots[name] = {
            "completed_microbatches": completed,
            "scheduled_token_positions": scheduled,
            "loaded_documents": loaded_documents,
            "loaded_source_tokens_with_bos": consumed_source + 4_000,
            "buffered_documents": 1_000,
            "buffered_source_tokens": 4_000,
            "buffer_epoch_histogram": {str(max_epoch): 1_000},
            "loaded_documents_by_epoch": {
                str(epoch): count for epoch, count in loaded_hist.items()
            },
            "consumed_documents_by_epoch": {
                str(epoch): count for epoch, count in consumed_hist.items()
            },
            "first_load_before_microbatch_by_epoch": {
                str(epoch): (epoch - 1) * first_wrap_microbatch
                for epoch in range(1, max_epoch + 1)
            },
            "first_consume_in_microbatch_by_epoch": {
                str(epoch): (
                    0 if epoch == 1 else (epoch - 1) * first_wrap_microbatch + 1
                )
                for epoch in range(1, max_epoch + 1)
            },
            "max_loaded_epoch": max_epoch,
            "max_consumed_epoch": max_epoch,
            "max_base_locator_load_count": max_epoch,
            "max_base_locator_consume_count": max_epoch,
            "epoch5_loaded_including_prefetch": False,
            "retained_source_elements": consumed_source,
            "row_leading_source_elements_not_targets": row_leading,
            "cropped_source_tokens": 0,
            "documents_completed": consumed_documents,
            "documents_cropped": 0,
            "retained_positions": {
                dimension: {"m": scheduled}
                for dimension in ("mixture", "source", "register")
            },
            "cropped_tokens": {
                dimension: {} for dimension in ("mixture", "source", "register")
            },
            "consumed_source_elements": {
                dimension: {"m": consumed_source}
                for dimension in ("mixture", "source", "register")
            },
            "row_leading_source_elements": {
                dimension: {"m": row_leading}
                for dimension in ("mixture", "source", "register")
            },
        }
    margin = snapshots["s40_margin"]
    return {
        "completed_microbatches": margin["completed_microbatches"],
        "first_epoch2_load_before_microbatch": first_wrap_microbatch,
        "first_epoch2_load_observation": "observed_including_prefetch",
        "whole_pool_order_validation": {
            "method": "bounded_disk_exact_epoch_and_partial_prefix_sha256_v2",
            "epoch1_sha256": "c" * 64,
            "epoch1_documents": pool_documents,
            "verified_complete_epochs": margin["max_loaded_epoch"] - 1,
            "current_partial_epoch": margin["max_loaded_epoch"],
            "current_partial_epoch_documents": margin[
                "loaded_documents_by_epoch"
            ][str(margin["max_loaded_epoch"])],
            "current_partial_prefix_verified_against_epoch1": True,
            "external_state_bytes": pool_documents * 32,
            "digest_bytes_per_epoch1_document": 32,
            "fixed_rank_assignment_and_stable_ties_prove_base_locator_maxima": True,
        },
        "snapshots": snapshots,
    }


def _receipt_simulation() -> dict:
    worlds = {}
    for world_size in (8, 16):
        evidence = _synthetic_rank_evidence(world_size)
        worlds[str(world_size)] = summarize_repetition_world(
            [json.loads(json.dumps(evidence)) for _ in range(world_size)],
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
            preferred_max_loaded_epoch=2,
            preferred_max_consumed_epoch=2,
            hard_max_loaded_epoch=4,
            hard_max_consumed_epoch=4,
        )
    return {
        "implementation": REPETITION_PACKING_SIMULATOR_ID,
        "implementation_file_sha256": file_sha256(
            Path(__file__).parents[1] / "nanochat" / "packing_capacity.py"
        ),
        "world_sizes": [8, 16],
        "device_batch_sequences": D32_REPETITION_DEVICE_BATCH_SEQUENCES,
        "max_seq_len": D32_REPETITION_MAX_SEQ_LEN,
        "tokenizer_batch_size": D32_REPETITION_TOKENIZER_BATCH_SIZE,
        "buffer_size": D32_REPETITION_BUFFER_SIZE,
        "global_batch_tokens": D32_REPETITION_GLOBAL_BATCH_TOKENS,
        "whole_pool_repetition_only": True,
        "source_specific_repetition": False,
        "manual_repetition_risk_approval_between_tiers": True,
        "upstream_contract": {
            "nanochat_revision": "92d63d4e",
            "tokenizer_batch_size": D32_REPETITION_TOKENIZER_BATCH_SIZE,
            "tokenizer_threads": 4,
            "refill_buffer_size": D32_REPETITION_BUFFER_SIZE,
            "tie_breaks": "first_largest_fit_else_first_shortest",
            "cropped_tail_policy": "discard",
            "rank_sharding": "parquet_row_group_index_mod_world_size",
            "epoch_order": "repeat_identical_whole_rank_shard",
            "whole_pool_repetition_only": True,
            "source_specific_repetition": False,
        },
        "fixture_parity": run_upstream_loader_parity_fixture(),
        "repeated_epoch_fixture_parity": run_upstream_repeated_epoch_parity_fixture(),
        "horizon_optimizer_steps": dict(D32_REPETITION_HORIZON_OPTIMIZER_STEPS),
        "thresholds": {
            "preferred_min_first_epoch_packed_positions": (
                PREFERRED_MIN_FIRST_EPOCH_PACKED_POSITIONS
            ),
            "hard_min_first_epoch_packed_positions": (
                HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS
            ),
            "preferred_max_loaded_epoch": 2,
            "preferred_max_consumed_epoch": 2,
            "hard_max_loaded_epoch": 4,
            "hard_max_consumed_epoch": 4,
        },
        "worlds": worlds,
        "repetition_tier": "manual_risk",
        "manual_repetition_risk_approval_required": True,
        "capacity_floor_passed": True,
        "all_worlds_pass": True,
        "max_base_locator_load_count": 4,
        "max_base_locator_consume_count": 4,
    }


def test_repetition_receipt_requires_manual_approval_and_rejects_tampering(
    tmp_path: Path,
):
    receipt = seal_repetition_capacity_receipt(
        tmp_path / "manual.json",
        simulation=_receipt_simulation(),
        dataset_manifest_sha256="a" * 64,
        tokenizer_package_sha256="b" * 64,
        intended_weights={"m": 1.0},
    )
    verify_manifest_hash(receipt)
    assert receipt["repetition_tier"] == "manual_risk"
    assert receipt["approval_required"] is True
    assert receipt["gate_passed"] is False
    assert receipt["cleanup_authorized"] is False

    simulation = _receipt_simulation()
    validation = validate_repetition_capacity_simulation(simulation)
    approval = seal_repetition_risk_approval(
        tmp_path / "approval.json",
        dataset_manifest_sha256="a" * 64,
        tokenizer_package_sha256="b" * 64,
        simulation_sha256=validation["simulation_sha256"],
        approver="test-reviewer",
        rationale="bounded four-pass proof-of-concept training",
    )
    approved = seal_repetition_capacity_receipt(
        tmp_path / "approved.json",
        simulation=simulation,
        dataset_manifest_sha256="a" * 64,
        tokenizer_package_sha256="b" * 64,
        intended_weights={"m": 1.0},
        manual_repetition_risk_approval=approval,
    )
    assert approved["gate_passed"] is True
    validated_receipt = validate_repetition_capacity_receipt(
        approved,
        dataset_manifest_sha256="a" * 64,
        tokenizer_package_sha256="b" * 64,
        manual_repetition_risk_approval=approval,
    )
    assert validated_receipt["gate_passed"] is True
    assert validated_receipt["manual_repetition_risk_approval_sha256"] == approval[
        "canonical_sha256"
    ]

    tampered = _receipt_simulation()
    tampered["worlds"]["8"]["max_base_locator_load_count"] = 2
    try:
        seal_repetition_capacity_receipt(
            tmp_path / "tampered.json",
            simulation=tampered,
            dataset_manifest_sha256="a" * 64,
            tokenizer_package_sha256="b" * 64,
            intended_weights={"m": 1.0},
        )
    except ValueError as exc:
        assert "summary" in str(exc) or "drift" in str(exc)
    else:
        raise AssertionError("tampered repetition maxima must fail closed")


def test_frozen_ws8_ws16_simulation_recomputes_complete_rank_evidence():
    simulation = _receipt_simulation()
    validated = validate_repetition_capacity_simulation(simulation)
    assert set(validated["worlds"]) == {"8", "16"}
    for world_size, accumulation in ((8, 32), (16, 16)):
        world = validated["worlds"][str(world_size)]
        assert world["rank_count"] == world_size
        assert len(world["rank_evidence"]) == world_size
        assert world["gradient_accumulation_steps"] == accumulation
        assert world["accumulation_evidence"]["gradient_accumulation_steps"] == accumulation
        for name, steps in D32_REPETITION_HORIZON_OPTIMIZER_STEPS.items():
            assert world["horizons"][name]["optimizer_steps"] == steps
            assert world["horizons"][name]["scheduled_token_positions"] == (
                steps * D32_REPETITION_GLOBAL_BATCH_TOKENS
            )


def test_final_repeat_simulator_executes_both_frozen_topologies(tmp_path: Path):
    data_path = tmp_path / "train.parquet"
    data_path.write_bytes(b"frozen-topology-fixture")
    train_files = [
        {
            "path": data_path.name,
            "size_bytes": data_path.stat().st_size,
            "sha256": file_sha256(data_path),
        }
    ]

    def fake_batches(*_args, rank, world_size, **_kwargs):
        return world_size, rank

    def fake_rank(source, *, B, T, snapshot_microbatches, buffer_size, **_kwargs):
        world_size, _rank = source
        accumulation = 32 if world_size == 8 else 16
        assert B == D32_REPETITION_DEVICE_BATCH_SEQUENCES
        assert T == D32_REPETITION_MAX_SEQ_LEN
        assert buffer_size == D32_REPETITION_BUFFER_SIZE
        assert snapshot_microbatches == {
            name: steps * accumulation
            for name, steps in D32_REPETITION_HORIZON_OPTIMIZER_STEPS.items()
        }
        return json.loads(json.dumps(_synthetic_rank_evidence(world_size)))

    legacy_parity = run_upstream_loader_parity_fixture()
    repeat_parity = run_upstream_repeated_epoch_parity_fixture()
    with patch(
        "nanochat.packing_capacity._rank_repeated_document_batches",
        side_effect=fake_batches,
    ), patch(
        "nanochat.packing_capacity.simulate_bestfit_rank_repeated",
        side_effect=fake_rank,
    ), patch(
        "nanochat.packing_capacity.run_upstream_loader_parity_fixture",
        return_value=legacy_parity,
    ), patch(
        "nanochat.packing_capacity.run_upstream_repeated_epoch_parity_fixture",
        return_value=repeat_parity,
    ):
        simulation = simulate_final_corpus_repetition_capacity(
            tmp_path,
            train_files,
            tokenizer=object(),
        )
    validated = validate_repetition_capacity_simulation(simulation)
    assert set(validated["worlds"]) == {"8", "16"}
    assert validated["worlds"]["8"]["gradient_accumulation_steps"] == 32
    assert validated["worlds"]["16"]["gradient_accumulation_steps"] == 16


def test_repetition_simulation_rejects_adversarial_world_horizon_and_hash_drift():
    mutations = []

    def mutate(description, callback):
        candidate = json.loads(json.dumps(_receipt_simulation()))
        callback(candidate)
        mutations.append((description, candidate))

    mutate("missing world", lambda item: item["worlds"].pop("16"))
    mutate("extra world", lambda item: item["worlds"].update({"4": {}}))
    mutate("file hash", lambda item: item.update({"implementation_file_sha256": "0" * 64}))
    mutate("flag", lambda item: item.update({"source_specific_repetition": True}))
    mutate(
        "threshold",
        lambda item: item["thresholds"].update(
            {"hard_min_first_epoch_packed_positions": HARD_MIN_FIRST_EPOCH_PACKED_POSITIONS - 1}
        ),
    )
    mutate(
        "legacy parity",
        lambda item: item["fixture_parity"].update({"actual_output_sha256": "0" * 64}),
    )
    mutate(
        "repeat parity",
        lambda item: item["repeated_epoch_fixture_parity"].update(
            {"actual_output_sha256": "0" * 64}
        ),
    )
    mutate(
        "incomplete ws8 evidence",
        lambda item: item["worlds"]["8"]["rank_evidence"].pop(),
    )
    mutate(
        "rank scheduled positions",
        lambda item: item["worlds"]["16"]["rank_evidence"][0]["snapshots"][
            "s20"
        ].update(
            {
                "scheduled_token_positions": item["worlds"]["16"]["rank_evidence"][0][
                    "snapshots"
                ]["s20"]["scheduled_token_positions"]
                + 1
            }
        ),
    )
    mutate(
        "world optimizer horizon",
        lambda item: item["worlds"]["8"]["horizons"]["s40"].update(
            {"optimizer_steps": 31_999}
        ),
    )
    mutate(
        "rank max locator",
        lambda item: item["worlds"]["8"]["rank_evidence"][0]["snapshots"][
            "s40_margin"
        ].update({"max_base_locator_load_count": 3}),
    )
    mutate(
        "retained mixture",
        lambda item: item["worlds"]["16"]["rank_evidence"][0]["snapshots"][
            "s12"
        ]["retained_positions"]["mixture"].update(
            {
                "m": item["worlds"]["16"]["rank_evidence"][0]["snapshots"][
                    "s12"
                ]["retained_positions"]["mixture"]["m"]
                - 1
            }
        ),
    )
    for description, candidate in mutations:
        try:
            validate_repetition_capacity_simulation(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"adversarial {description} mutation passed")


def test_manual_risk_rejects_boolean_or_misbound_approval(tmp_path: Path):
    simulation = _receipt_simulation()
    simulation_sha256 = validate_repetition_capacity_simulation(simulation)[
        "simulation_sha256"
    ]
    approval = seal_repetition_risk_approval(
        tmp_path / "misbound.json",
        dataset_manifest_sha256="a" * 64,
        tokenizer_package_sha256="b" * 64,
        simulation_sha256=simulation_sha256,
        approver="reviewer",
        rationale="test",
    )
    for supplied, dataset_sha in ((True, "a" * 64), (approval, "d" * 64)):
        try:
            seal_repetition_capacity_receipt(
                tmp_path / f"rejected-{dataset_sha[0]}-{type(supplied).__name__}.json",
                simulation=simulation,
                dataset_manifest_sha256=dataset_sha,
                tokenizer_package_sha256="b" * 64,
                intended_weights={"m": 1.0},
                manual_repetition_risk_approval=supplied,
            )
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("unsealed or misbound manual approval passed")
