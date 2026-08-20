from __future__ import annotations

import copy
from pathlib import Path

import pytest

from nanochat.experiment_manifest import (
    file_sha256,
    load_json_strict,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
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
        "SLURM_NTASKS": "32",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_NNODES": "1",
        "SLURM_PROCID": str(lane_id),
        "SLURM_LOCALID": str(lane_id),
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "0",
        "SLURMD_NODENAME": "cpu-node-01",
        "SLURM_TMPDIR": str(tmp_path / "scratch"),
        **{name: "1" for name in packed.THREAD_CAPS},
    }


def _slurm_bucket_env(rank: int) -> dict[str, str]:
    return {
        "SLURM_NTASKS": "14",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_NNODES": "1",
        "SLURM_PROCID": str(rank),
        "SLURM_LOCALID": str(rank),
        "SLURM_JOB_ID": "23456",
        "SLURM_STEP_ID": "0",
        "SLURMD_NODENAME": "cpu-node-02",
        **{name: "1" for name in packed.THREAD_CAPS},
    }


def _write_object_launch_fixture(paths: dict[str, Path], run_root: Path) -> Path:
    source_plan = load_json_strict(paths["source_plan"])
    calibration = load_json_strict(paths["calibration"])
    lane_plan = load_json_strict(paths["lane_plan"])
    records = []
    for position, rank in enumerate(lane_plan["resource_sample_ranks"]["ranks"]):
        receipt = seal_manifest(
            {
                "rank": rank,
                "sample_mode": True,
                "source_plan_sha256": source_plan["canonical_sha256"],
                "calibration_sha256": calibration["canonical_sha256"],
                "canonical_sha256": None,
            }
        )
        path = run_root / "objects" / f"{rank:05d}" / "object_receipt.json"
        write_json_atomic(path, receipt)
        records.append(
            {
                "lane_id": position % 32,
                "rank": rank,
                "path": f"objects/{rank:05d}/object_receipt.json",
                "canonical_sha256": receipt["canonical_sha256"],
                "disposition": "produced_by_allocation",
            }
        )
    launch = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": packed.LAUNCH_RECEIPT_KIND,
            "sample_mode": True,
            "lane_plan_sha256": lane_plan["canonical_sha256"],
            "policy_sha256": source_plan["policy_sha256"],
            "source_plan_sha256": source_plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "allocation": {
                "slurm_job_id": "12345",
                "slurm_step_id": "0",
                "slurm_node": "cpu-node-01",
                "nodes": 1,
                "tasks": 32,
                "cpus_per_task": 4,
                "allocated_cpus": 128,
            },
            "object_receipts": records,
            "all_lanes_completed": True,
            "canonical_sha256": None,
        }
    )
    path = run_root / "packed_sample_launches" / "job12345" / "launch_receipt.json"
    write_json_atomic(path, launch)
    return path


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
    assert first["lane_count"] == 32
    assert first["cpus_per_lane"] == 4
    assert first["totals"]["allocated_cpus"] == 128
    assert [lane["lane_id"] for lane in first["lanes"]] == list(range(32))
    assert [lane["ranks"] for lane in first["lanes"]] == [
        ranks[lane_id::32] for lane_id in range(32)
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
    for lane_id in range(32):
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
        rank for lane_id in range(32) for rank in ranks[lane_id::32]
    ]
    assert all(call["sample_mode"] is True for call in calls)
    assert all(call["resource_approval_path"] is None for call in calls)
    assert launch["all_lanes_completed"] is True
    assert launch["allocation"] == {
        "slurm_job_id": "12345",
        "slurm_step_id": "0",
        "slurm_node": "cpu-node-01",
        "nodes": 1,
        "tasks": 32,
        "cpus_per_task": 4,
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
    with pytest.raises(TurkishCorpusError, match="one node, thirty-two tasks"):
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
        "#SBATCH --ntasks=32",
        "#SBATCH --ntasks-per-node=32",
        "#SBATCH --cpus-per-task=4",
        "srun --nodes=1 --ntasks=32 --ntasks-per-node=32 --cpus-per-task=4",
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


def test_bucket_workers_map_one_to_one_and_finalize_collectively(
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
    object_launch = _write_object_launch_fixture(paths, run_root)
    task_root = tmp_path / "bucket-launch" / "tasks"
    calls = []

    def fake_bucket(
        _policy,
        source_plan,
        calibration,
        run_dir,
        *,
        rank,
        sample_mode,
        resource_approval_path,
    ):
        calls.append(
            {
                "rank": rank,
                "sample_mode": sample_mode,
                "resource_approval_path": resource_approval_path,
            }
        )
        output_path = Path(run_dir) / "bucket_matches" / f"{rank:05d}_00.dups"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(rank.to_bytes(8, "little") * 2)
        object_hashes = [
            item["canonical_sha256"]
            for item in load_json_strict(object_launch)["object_receipts"]
        ]
        receipt = seal_manifest(
            {
                "schema_version": "1.0",
                "kind": packed.BACKEND_BUCKET_RECEIPT_KIND,
                "sample_mode": sample_mode,
                "rank": rank,
                "world_size": 14,
                "source_plan_sha256": source_plan["canonical_sha256"],
                "calibration_sha256": calibration["canonical_sha256"],
                "object_receipt_sha256": object_hashes,
                "output": {
                    "path": f"bucket_matches/{rank:05d}_00.dups",
                    "size_bytes": 16,
                    "sha256": file_sha256(output_path),
                    "duplicate_edges": 1,
                },
                "canonical_sha256": None,
            }
        )
        write_json_atomic(
            Path(run_dir) / "bucket_receipts" / f"{rank:05d}.json", receipt
        )
        return receipt

    monkeypatch.setattr(packed, "run_datatrove_bucket", fake_bucket)
    for rank in range(14):
        packed.run_bucket_task(
            paths["policy"],
            paths["source_plan"],
            paths["calibration"],
            paths["sample_ranks"],
            paths["lane_plan"],
            object_launch,
            run_root,
            task_root,
            env=_slurm_bucket_env(rank),
        )

    launch = packed.seal_bucket_launch_receipt(
        paths["policy"],
        paths["source_plan"],
        paths["calibration"],
        paths["sample_ranks"],
        paths["lane_plan"],
        object_launch,
        run_root,
        task_root,
        tmp_path / "bucket-launch" / "launch_receipt.json",
        job_id="23456",
    )

    assert [item["rank"] for item in calls] == list(range(14))
    assert all(item["sample_mode"] is True for item in calls)
    assert all(item["resource_approval_path"] is None for item in calls)
    assert launch["assignment"] == {
        "algorithm": packed.BUCKET_ASSIGNMENT_ALGORITHM,
        "bucket_ranks": list(range(14)),
        "world_size": 14,
    }
    assert launch["allocation"] == {
        "slurm_job_id": "23456",
        "slurm_step_id": "0",
        "slurm_node": "cpu-node-02",
        "nodes": 1,
        "tasks": 14,
        "cpus_per_task": 8,
        "allocated_cpus": 112,
    }
    assert launch["all_buckets_completed"] is True
    assert [item["bucket_rank"] for item in launch["backend_bucket_receipts"]] == list(
        range(14)
    )


def test_bucket_context_rejects_topology_mapping_and_thread_drift():
    with pytest.raises(TurkishCorpusError, match="fourteen tasks"):
        packed._slurm_bucket_context(
            _slurm_bucket_env(0) | {"SLURM_NTASKS": "13"}
        )
    with pytest.raises(TurkishCorpusError, match="one-to-one"):
        packed._slurm_bucket_context(
            _slurm_bucket_env(3) | {"SLURM_LOCALID": "2"}
        )
    with pytest.raises(TurkishCorpusError, match="thread caps"):
        packed._slurm_bucket_context(
            _slurm_bucket_env(0) | {"MKL_NUM_THREADS": "8"}
        )


def test_packed_bucket_sbatch_is_direct_sample_only_and_collective():
    source = (
        ROOT / "runs" / "uhem_turkish_data_buckets_packed_sample.sbatch"
    ).read_text(encoding="utf-8")
    required = (
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=14",
        "#SBATCH --ntasks-per-node=14",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=240G",
        "srun --nodes=1 --ntasks=14 --ntasks-per-node=14 --cpus-per-task=8",
        "--cpu-bind=cores --kill-on-bad-exit=1",
        "-m scripts.turkish_packed_sample run-bucket",
        "-m scripts.turkish_packed_sample finalize-buckets",
        'if [ "$srun_rc" -ne 0 ]',
    )
    assert all(fragment in source for fragment in required)
    assert "#SBATCH --array" not in source
    assert "RESOURCE_APPROVAL" not in source
    assert "SAMPLE=" not in source
    for name in packed.THREAD_CAPS:
        assert f"export {name}=1" in source

    parser = packed.build_parser()
    bucket_parser = next(
        action for action in parser._actions if action.dest == "command"
    ).choices["run-bucket"]
    options = {
        option for action in bucket_parser._actions for option in action.option_strings
    }
    assert "--sample" not in options
    assert "--resource-approval" not in options
