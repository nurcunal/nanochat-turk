from __future__ import annotations

import copy
from pathlib import Path

import pytest

from nanochat.experiment_manifest import seal_manifest, verify_manifest_hash, write_json_atomic
from nanochat.turkish_corpus import TurkishCorpusError
from scripts import turkish_packed_sample as packed


ROOT = Path(__file__).resolve().parents[1]


def _fixture_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ranks: list[int]
) -> dict[str, Path]:
    policy = {"schema_version": "2.0", "name": packed.V2_POLICY_NAME}
    source_plan = {
        "policy_sha256": "a" * 64,
        "canonical_sha256": "b" * 64,
        "objects": [
            {
                "rank": rank,
                "source_id": "fixture",
                "uri": f"https://example.test/part-{rank:05d}.jsonl.zst",
                "size_bytes": 1_000,
            }
            for rank in range(max(ranks) + 1)
        ],
    }
    calibration = {"canonical_sha256": "c" * 64}
    paths = {
        "policy": tmp_path / "policy.json",
        "source_plan": tmp_path / "source_plan.json",
        "calibration": tmp_path / "calibration.json",
        "sample_ranks": tmp_path / "sample_ranks.json",
        "lane_plan": tmp_path / "lane_plan.json",
    }
    write_json_atomic(paths["policy"], policy)
    write_json_atomic(paths["source_plan"], source_plan)
    write_json_atomic(paths["calibration"], calibration)
    write_json_atomic(
        paths["sample_ranks"],
        {"ranks": ranks, "slurm_array": ",".join(str(rank) for rank in ranks)},
    )
    monkeypatch.setattr(packed, "load_corpus_policy", lambda _path: policy)
    monkeypatch.setattr(packed, "validate_source_plan", lambda *_args: None)
    monkeypatch.setattr(packed, "validate_backend_calibration", lambda *_args: None)
    monkeypatch.setattr(
        packed, "select_resource_sample_ranks", lambda _plan: list(ranks)
    )
    return paths


def _slurm_env(lane_id: int, tmp_path: Path) -> dict[str, str]:
    return {
        "SLURM_NTASKS": "8",
        "SLURM_CPUS_PER_TASK": "16",
        "SLURM_NNODES": "1",
        "SLURM_PROCID": str(lane_id),
        "SLURM_LOCALID": str(lane_id),
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "0",
        "SLURMD_NODENAME": "cpu-node-01",
        "SLURM_TMPDIR": str(tmp_path / "scratch"),
        **{name: "1" for name in packed.THREAD_CAPS},
    }


def test_lane_plan_is_sealed_disjoint_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ranks = list(range(19))
    paths = _fixture_inputs(tmp_path, monkeypatch, ranks)

    first = packed.seal_lane_plan(
        paths["policy"],
        paths["source_plan"],
        paths["calibration"],
        paths["sample_ranks"],
        paths["lane_plan"],
    )
    second = packed.seal_lane_plan(
        paths["policy"],
        paths["source_plan"],
        paths["calibration"],
        paths["sample_ranks"],
        paths["lane_plan"],
    )

    assert first == second
    assert verify_manifest_hash(first) == first["canonical_sha256"]
    assert first["sample_mode"] is True
    assert first["lane_count"] == 8
    assert first["cpus_per_lane"] == 16
    assert first["totals"]["allocated_cpus"] == 128
    assert [lane["lane_id"] for lane in first["lanes"]] == list(range(8))
    assert [lane["ranks"] for lane in first["lanes"]] == [
        ranks[lane_id::8] for lane_id in range(8)
    ]
    flattened = [rank for lane in first["lanes"] for rank in lane["ranks"]]
    assert sorted(flattened) == ranks
    assert len(flattened) == len(set(flattened))


def test_lane_plan_rejects_resource_rank_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ranks = list(range(12))
    paths = _fixture_inputs(tmp_path, monkeypatch, ranks)
    lane_plan = packed.seal_lane_plan(
        paths["policy"],
        paths["source_plan"],
        paths["calibration"],
        paths["sample_ranks"],
        paths["lane_plan"],
    )
    write_json_atomic(
        paths["sample_ranks"],
        {"ranks": ranks[:-1], "slurm_array": ",".join(map(str, ranks[:-1]))},
    )

    with pytest.raises(TurkishCorpusError, match="deterministic v2 selector"):
        packed.validate_lane_plan(
            lane_plan,
            policy={"schema_version": "2.0", "name": packed.V2_POLICY_NAME},
            source_plan={
                "policy_sha256": "a" * 64,
                "canonical_sha256": "b" * 64,
                "objects": [],
            },
            calibration={"canonical_sha256": "c" * 64},
            sample_ranks_path=paths["sample_ranks"],
        )


def test_lane_worker_is_sample_only_and_finalizes_collectively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ranks = list(range(11))
    paths = _fixture_inputs(tmp_path, monkeypatch, ranks)
    packed.seal_lane_plan(
        paths["policy"],
        paths["source_plan"],
        paths["calibration"],
        paths["sample_ranks"],
        paths["lane_plan"],
    )
    run_root = tmp_path / "sample-run"
    lane_root = tmp_path / "launch" / "lanes"
    calls = []

    def fake_process(
        _policy,
        source_plan,
        calibration,
        _model_path,
        run_dir,
        *,
        rank,
        sample_mode,
        resource_approval_path,
        scratch_dir,
    ):
        calls.append(
            {
                "rank": rank,
                "sample_mode": sample_mode,
                "resource_approval_path": resource_approval_path,
                "scratch_dir": Path(scratch_dir),
            }
        )
        receipt = seal_manifest(
            {
                "rank": rank,
                "sample_mode": sample_mode,
                "source_plan_sha256": source_plan["canonical_sha256"],
                "calibration_sha256": calibration["canonical_sha256"],
                "canonical_sha256": None,
            }
        )
        path = Path(run_dir) / "objects" / f"{rank:05d}" / "object_receipt.json"
        write_json_atomic(path, receipt)
        return receipt

    monkeypatch.setattr(packed, "process_source_object", fake_process)
    for lane_id in range(8):
        packed.run_lane(
            paths["policy"],
            paths["source_plan"],
            paths["calibration"],
            paths["sample_ranks"],
            paths["lane_plan"],
            tmp_path / "model.bin",
            run_root,
            lane_root,
            env=_slurm_env(lane_id, tmp_path),
        )

    launch = packed.seal_launch_receipt(
        paths["policy"],
        paths["source_plan"],
        paths["calibration"],
        paths["sample_ranks"],
        paths["lane_plan"],
        run_root,
        lane_root,
        tmp_path / "launch" / "launch_receipt.json",
        job_id="12345",
    )

    assert [call["rank"] for call in calls] == [
        rank for lane_id in range(8) for rank in ranks[lane_id::8]
    ]
    assert all(call["sample_mode"] is True for call in calls)
    assert all(call["resource_approval_path"] is None for call in calls)
    assert launch["all_lanes_completed"] is True
    assert launch["allocation"] == {
        "slurm_job_id": "12345",
        "slurm_step_id": "0",
        "slurm_node": "cpu-node-01",
        "nodes": 1,
        "tasks": 8,
        "cpus_per_task": 16,
        "allocated_cpus": 128,
    }
    assert [item["rank"] for item in launch["object_receipts"]] == ranks


def test_lane_worker_rejects_wrong_topology_or_thread_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ranks = [0]
    paths = _fixture_inputs(tmp_path, monkeypatch, ranks)
    packed.seal_lane_plan(
        paths["policy"],
        paths["source_plan"],
        paths["calibration"],
        paths["sample_ranks"],
        paths["lane_plan"],
    )
    monkeypatch.setattr(
        packed,
        "process_source_object",
        lambda *_args, **_kwargs: pytest.fail("invalid topology reached processing"),
    )
    bad_topology = _slurm_env(0, tmp_path) | {"SLURM_NTASKS": "7"}
    with pytest.raises(TurkishCorpusError, match="one node, eight tasks"):
        packed.run_lane(
            paths["policy"],
            paths["source_plan"],
            paths["calibration"],
            paths["sample_ranks"],
            paths["lane_plan"],
            tmp_path / "model.bin",
            tmp_path / "run-a",
            tmp_path / "receipts-a",
            env=bad_topology,
        )
    bad_threads = _slurm_env(0, tmp_path) | {"OPENBLAS_NUM_THREADS": "16"}
    with pytest.raises(TurkishCorpusError, match="thread caps"):
        packed.run_lane(
            paths["policy"],
            paths["source_plan"],
            paths["calibration"],
            paths["sample_ranks"],
            paths["lane_plan"],
            tmp_path / "model.bin",
            tmp_path / "run-b",
            tmp_path / "receipts-b",
            env=bad_threads,
        )


def test_packed_sbatch_is_direct_sample_only_and_collective():
    source = (
        ROOT / "runs" / "uhem_turkish_data_objects_packed_sample.sbatch"
    ).read_text(encoding="utf-8")
    required = (
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=8",
        "#SBATCH --ntasks-per-node=8",
        "#SBATCH --cpus-per-task=16",
        "srun --nodes=1 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=16",
        "--cpu-bind=cores --kill-on-bad-exit=1",
        "-m scripts.turkish_packed_sample run-lane",
        "-m scripts.turkish_packed_sample finalize",
        'if [ "$srun_rc" -ne 0 ]',
    )
    assert all(fragment in source for fragment in required)
    assert "RESOURCE_APPROVAL" not in source
    assert "SAMPLE=" not in source
    for name in packed.THREAD_CAPS:
        assert f"export {name}=1" in source

    parser = packed.build_parser()
    run_lane_parser = next(
        action for action in parser._actions if action.dest == "command"
    ).choices["run-lane"]
    options = {
        option
        for action in run_lane_parser._actions
        for option in action.option_strings
    }
    assert "--sample" not in options
    assert "--resource-approval" not in options
