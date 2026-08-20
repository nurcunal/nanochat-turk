"""Seal and execute packed Turkish v2 object and MinHash resource samples.

The launchers are deliberately sample-only.  Object ranks are assigned
round-robin to thirty-two Slurm workers; the fourteen fixed MinHash buckets map
one-to-one to fourteen workers.  There is no production-mode switch in this
interface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nanochat.experiment_manifest import (
    file_sha256,
    load_json_strict,
    seal_manifest,
    verify_manifest_hash,
    write_json_atomic,
)
from nanochat.turkish_backend import (
    process_source_object,
    run_datatrove_bucket,
    select_resource_sample_ranks,
    validate_backend_calibration,
    validate_source_plan,
)
from nanochat.turkish_corpus import TurkishCorpusError, load_corpus_policy


LANE_COUNT = 32
CPUS_PER_LANE = 4
BUCKET_COUNT = 14
CPUS_PER_BUCKET_TASK = 8
V2_POLICY_NAME = "tr_general_clean_v2"
LANE_PLAN_KIND = "turkish_packed_resource_sample_lane_plan"
LANE_RECEIPT_KIND = "turkish_packed_resource_sample_lane_receipt"
LAUNCH_RECEIPT_KIND = "turkish_packed_resource_sample_launch_receipt"
ASSIGNMENT_ALGORITHM = "sorted_rank_round_robin_v1"
BUCKET_TASK_RECEIPT_KIND = "turkish_packed_sample_bucket_task_receipt"
BUCKET_LAUNCH_RECEIPT_KIND = "turkish_packed_sample_bucket_launch_receipt"
BUCKET_ASSIGNMENT_ALGORITHM = "slurm_procid_equals_minhash_bucket_rank_v1"
BACKEND_BUCKET_RECEIPT_KIND = "turkish_datatrove_bucket_result"
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _load_bound_inputs(
    policy_path: str | Path,
    source_plan_path: str | Path,
    calibration_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = load_corpus_policy(policy_path)
    if (
        policy.get("schema_version") != "2.0"
        or policy.get("name") != V2_POLICY_NAME
    ):
        raise TurkishCorpusError(
            "the packed object sample accepts only the frozen Turkish v2 policy"
        )
    source_plan = load_json_strict(source_plan_path)
    calibration = load_json_strict(calibration_path)
    validate_source_plan(source_plan, policy)
    validate_backend_calibration(calibration, policy)
    return policy, source_plan, calibration


def _load_deterministic_sample_ranks(
    path: str | Path, source_plan: Mapping[str, Any]
) -> tuple[list[int], str]:
    payload = load_json_strict(path)
    if not isinstance(payload, Mapping):
        raise TurkishCorpusError("resource sample ranks must be a JSON object")
    ranks = payload.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        raise TurkishCorpusError("resource sample ranks are missing")
    if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in ranks):
        raise TurkishCorpusError("resource sample ranks must be integers")
    if ranks != sorted(set(ranks)):
        raise TurkishCorpusError("resource sample ranks must be sorted and unique")
    expected = select_resource_sample_ranks(source_plan)
    if ranks != expected:
        raise TurkishCorpusError(
            "resource sample ranks do not match the deterministic v2 selector"
        )
    expected_array = ",".join(str(rank) for rank in expected)
    if payload.get("slurm_array") != expected_array:
        raise TurkishCorpusError("resource sample Slurm array rendering drift")
    return expected, file_sha256(path)


def _expected_lane_plan(
    policy: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    ranks: list[int],
    ranks_file_sha256: str,
) -> dict[str, Any]:
    lanes = [
        {"lane_id": lane_id, "ranks": ranks[lane_id::LANE_COUNT]}
        for lane_id in range(LANE_COUNT)
    ]
    return seal_manifest(
        {
            "schema_version": "1.0",
            "kind": LANE_PLAN_KIND,
            "sample_mode": True,
            "policy_name": policy["name"],
            "policy_sha256": source_plan["policy_sha256"],
            "source_plan_sha256": source_plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "resource_sample_ranks": {
                "file_sha256": ranks_file_sha256,
                "ranks": ranks,
            },
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "lane_count": LANE_COUNT,
            "cpus_per_lane": CPUS_PER_LANE,
            "lanes": lanes,
            "totals": {
                "source_objects": len(source_plan["objects"]),
                "sample_ranks": len(ranks),
                "workers": LANE_COUNT,
                "allocated_cpus": LANE_COUNT * CPUS_PER_LANE,
            },
            "canonical_sha256": None,
        }
    )


def validate_lane_plan(
    lane_plan: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    sample_ranks_path: str | Path,
) -> str:
    digest = verify_manifest_hash(lane_plan)
    ranks, ranks_file_sha256 = _load_deterministic_sample_ranks(
        sample_ranks_path, source_plan
    )
    expected = _expected_lane_plan(
        policy, source_plan, calibration, ranks, ranks_file_sha256
    )
    if dict(lane_plan) != expected:
        raise TurkishCorpusError(
            "packed sample lane plan does not match its sealed inputs"
        )
    return digest


def seal_lane_plan(
    policy_path: str | Path,
    source_plan_path: str | Path,
    calibration_path: str | Path,
    sample_ranks_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    policy, source_plan, calibration = _load_bound_inputs(
        policy_path, source_plan_path, calibration_path
    )
    ranks, ranks_file_sha256 = _load_deterministic_sample_ranks(
        sample_ranks_path, source_plan
    )
    expected = _expected_lane_plan(
        policy, source_plan, calibration, ranks, ranks_file_sha256
    )
    destination = Path(output_path)
    if destination.exists():
        existing = load_json_strict(destination)
        validate_lane_plan(
            existing,
            policy=policy,
            source_plan=source_plan,
            calibration=calibration,
            sample_ranks_path=sample_ranks_path,
        )
        return dict(existing)
    write_json_atomic(destination, expected)
    return expected


def _positive_slurm_int(env: Mapping[str, str], name: str) -> int:
    raw = env.get(name)
    try:
        value = int(raw) if raw is not None else 0
    except ValueError as exc:
        raise TurkishCorpusError(f"{name} must be an integer") from exc
    if value <= 0:
        raise TurkishCorpusError(f"{name} must be positive")
    return value


def _slurm_lane_context(env: Mapping[str, str]) -> dict[str, Any]:
    tasks = _positive_slurm_int(env, "SLURM_NTASKS")
    cpus_per_task = _positive_slurm_int(env, "SLURM_CPUS_PER_TASK")
    nodes = _positive_slurm_int(env, "SLURM_NNODES")
    if tasks != LANE_COUNT or cpus_per_task != CPUS_PER_LANE or nodes != 1:
        raise TurkishCorpusError(
            "packed sample requires one node, thirty-two tasks, and four CPUs per task"
        )
    try:
        proc_id = int(env.get("SLURM_PROCID", ""))
        local_id = int(env.get("SLURM_LOCALID", ""))
    except ValueError as exc:
        raise TurkishCorpusError("Slurm task coordinates must be integers") from exc
    if proc_id not in range(LANE_COUNT) or local_id != proc_id:
        raise TurkishCorpusError(
            "packed sample tasks must map one-to-one to local lane IDs 0..7"
        )
    job_id = env.get("SLURM_JOB_ID", "")
    step_id = env.get("SLURM_STEP_ID", "")
    node_name = env.get("SLURMD_NODENAME", "")
    if not job_id.isdigit() or not step_id.isdigit() or not node_name:
        raise TurkishCorpusError("packed sample must run in a numeric Slurm job step")
    thread_caps = {name: env.get(name) for name in THREAD_CAPS}
    if any(value != "1" for value in thread_caps.values()):
        raise TurkishCorpusError("OMP/BLAS thread caps must all be exactly one")
    return {
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "slurm_node": node_name,
        "nodes": nodes,
        "tasks": tasks,
        "cpus_per_task": cpus_per_task,
        "proc_id": proc_id,
        "local_id": local_id,
        "allocated_cpus": tasks * cpus_per_task,
        "thread_caps": thread_caps,
    }


def _object_receipt_record(
    receipt: Mapping[str, Any], rank: int, disposition: str
) -> dict[str, Any]:
    digest = verify_manifest_hash(receipt)
    if (
        receipt.get("rank") != rank
        or receipt.get("sample_mode") is not True
        or disposition not in {"produced_by_allocation", "reused_verified"}
    ):
        raise TurkishCorpusError("packed lane received an invalid object receipt")
    return {
        "rank": rank,
        "path": f"objects/{rank:05d}/object_receipt.json",
        "canonical_sha256": digest,
        "disposition": disposition,
    }


def run_lane(
    policy_path: str | Path,
    source_plan_path: str | Path,
    calibration_path: str | Path,
    sample_ranks_path: str | Path,
    lane_plan_path: str | Path,
    model_path: str | Path,
    run_dir: str | Path,
    receipt_dir: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    policy, source_plan, calibration = _load_bound_inputs(
        policy_path, source_plan_path, calibration_path
    )
    lane_plan = load_json_strict(lane_plan_path)
    lane_plan_sha256 = validate_lane_plan(
        lane_plan,
        policy=policy,
        source_plan=source_plan,
        calibration=calibration,
        sample_ranks_path=sample_ranks_path,
    )
    execution_env = os.environ if env is None else env
    context = _slurm_lane_context(execution_env)
    lane_id = int(context["proc_id"])
    assignment = lane_plan["lanes"][lane_id]
    destination = Path(receipt_dir) / f"lane_{lane_id:05d}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite packed lane receipt: {destination}")

    run_root = Path(run_dir)
    object_records = []
    for rank in assignment["ranks"]:
        object_receipt_path = run_root / "objects" / f"{rank:05d}" / "object_receipt.json"
        preexisting = object_receipt_path.is_file()
        receipt = process_source_object(
            policy,
            source_plan,
            calibration,
            model_path,
            run_root,
            rank=rank,
            sample_mode=True,
            resource_approval_path=None,
            scratch_dir=Path(execution_env.get("SLURM_TMPDIR", run_root / ".scratch"))
            / f"lane-{lane_id:02d}",
        )
        object_records.append(
            _object_receipt_record(
                receipt,
                rank,
                "reused_verified" if preexisting else "produced_by_allocation",
            )
        )

    lane_receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": LANE_RECEIPT_KIND,
            "sample_mode": True,
            "lane_plan_sha256": lane_plan_sha256,
            "policy_sha256": source_plan["policy_sha256"],
            "source_plan_sha256": source_plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "allocation": context,
            "rank_assignment": dict(assignment),
            "object_receipts": object_records,
            "all_assigned_ranks_completed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, lane_receipt)
    return lane_receipt


def _validate_lane_receipt(
    receipt: Mapping[str, Any],
    *,
    lane_id: int,
    lane_plan: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_root: Path,
    job_id: str,
) -> str:
    digest = verify_manifest_hash(receipt)
    assignment = lane_plan["lanes"][lane_id]
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("kind") != LANE_RECEIPT_KIND
        or receipt.get("sample_mode") is not True
        or receipt.get("all_assigned_ranks_completed") is not True
        or receipt.get("lane_plan_sha256") != lane_plan["canonical_sha256"]
        or receipt.get("policy_sha256") != source_plan["policy_sha256"]
        or receipt.get("source_plan_sha256") != source_plan["canonical_sha256"]
        or receipt.get("calibration_sha256") != calibration["canonical_sha256"]
        or receipt.get("rank_assignment") != assignment
    ):
        raise TurkishCorpusError(f"packed lane {lane_id} receipt binding drift")
    allocation = receipt.get("allocation")
    if not isinstance(allocation, Mapping):
        raise TurkishCorpusError(f"packed lane {lane_id} allocation is missing")
    if (
        allocation.get("slurm_job_id") != job_id
        or allocation.get("nodes") != 1
        or allocation.get("tasks") != LANE_COUNT
        or allocation.get("cpus_per_task") != CPUS_PER_LANE
        or allocation.get("allocated_cpus") != LANE_COUNT * CPUS_PER_LANE
        or allocation.get("proc_id") != lane_id
        or allocation.get("local_id") != lane_id
        or allocation.get("thread_caps") != {name: "1" for name in THREAD_CAPS}
        or not str(allocation.get("slurm_step_id", "")).isdigit()
        or not allocation.get("slurm_node")
    ):
        raise TurkishCorpusError(f"packed lane {lane_id} Slurm allocation drift")
    object_records = receipt.get("object_receipts")
    if not isinstance(object_records, list) or [
        item.get("rank") if isinstance(item, Mapping) else None
        for item in object_records
    ] != assignment["ranks"]:
        raise TurkishCorpusError(f"packed lane {lane_id} object inventory drift")
    for item in object_records:
        rank = int(item["rank"])
        expected_path = f"objects/{rank:05d}/object_receipt.json"
        if (
            item.get("path") != expected_path
            or item.get("disposition")
            not in {"produced_by_allocation", "reused_verified"}
        ):
            raise TurkishCorpusError(f"packed lane {lane_id} object record drift")
        object_path = run_root / expected_path
        if object_path.is_symlink() or not object_path.is_file():
            raise TurkishCorpusError(f"packed object receipt {rank} is unsafe or missing")
        object_receipt = load_json_strict(object_path)
        object_sha256 = verify_manifest_hash(object_receipt)
        if (
            item.get("canonical_sha256") != object_sha256
            or object_receipt.get("rank") != rank
            or object_receipt.get("sample_mode") is not True
            or object_receipt.get("source_plan_sha256")
            != source_plan["canonical_sha256"]
            or object_receipt.get("calibration_sha256")
            != calibration["canonical_sha256"]
        ):
            raise TurkishCorpusError(f"packed object receipt {rank} binding drift")
    return digest


def seal_launch_receipt(
    policy_path: str | Path,
    source_plan_path: str | Path,
    calibration_path: str | Path,
    sample_ranks_path: str | Path,
    lane_plan_path: str | Path,
    run_dir: str | Path,
    lane_receipt_dir: str | Path,
    output_path: str | Path,
    *,
    job_id: str,
) -> dict[str, Any]:
    if not job_id.isdigit():
        raise TurkishCorpusError("Slurm job ID must be numeric")
    policy, source_plan, calibration = _load_bound_inputs(
        policy_path, source_plan_path, calibration_path
    )
    lane_plan = load_json_strict(lane_plan_path)
    lane_plan_sha256 = validate_lane_plan(
        lane_plan,
        policy=policy,
        source_plan=source_plan,
        calibration=calibration,
        sample_ranks_path=sample_ranks_path,
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite packed launch receipt: {destination}"
        )
    run_root = Path(run_dir)
    lane_root = Path(lane_receipt_dir)
    lane_records = []
    object_records = []
    step_ids: set[str] = set()
    nodes: set[str] = set()
    for lane_id in range(LANE_COUNT):
        path = lane_root / f"lane_{lane_id:05d}.json"
        if path.is_symlink() or not path.is_file():
            raise TurkishCorpusError(f"packed lane {lane_id} receipt is missing")
        receipt = load_json_strict(path)
        digest = _validate_lane_receipt(
            receipt,
            lane_id=lane_id,
            lane_plan=lane_plan,
            source_plan=source_plan,
            calibration=calibration,
            run_root=run_root,
            job_id=job_id,
        )
        step_ids.add(str(receipt["allocation"]["slurm_step_id"]))
        nodes.add(str(receipt["allocation"]["slurm_node"]))
        lane_records.append(
            {
                "lane_id": lane_id,
                "path": f"lanes/lane_{lane_id:05d}.json",
                "canonical_sha256": digest,
            }
        )
        object_records.extend(
            {"lane_id": lane_id, **dict(item)}
            for item in receipt["object_receipts"]
        )
    if len(step_ids) != 1 or len(nodes) != 1:
        raise TurkishCorpusError("packed lanes do not share one Slurm step and node")
    expected_ranks = lane_plan["resource_sample_ranks"]["ranks"]
    if sorted(item["rank"] for item in object_records) != expected_ranks:
        raise TurkishCorpusError("packed launch does not cover every sample rank exactly once")

    launch_receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": LAUNCH_RECEIPT_KIND,
            "sample_mode": True,
            "lane_plan_sha256": lane_plan_sha256,
            "policy_sha256": source_plan["policy_sha256"],
            "source_plan_sha256": source_plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "allocation": {
                "slurm_job_id": job_id,
                "slurm_step_id": next(iter(step_ids)),
                "slurm_node": next(iter(nodes)),
                "nodes": 1,
                "tasks": LANE_COUNT,
                "cpus_per_task": CPUS_PER_LANE,
                "allocated_cpus": LANE_COUNT * CPUS_PER_LANE,
            },
            "lane_receipts": lane_records,
            "object_receipts": sorted(object_records, key=lambda item: item["rank"]),
            "all_lanes_completed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, launch_receipt)
    return launch_receipt


def _validate_object_sample_launch_receipt(
    path: str | Path,
    *,
    lane_plan: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    run_root: Path,
) -> tuple[str, list[str]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TurkishCorpusError("packed object launch receipt is unsafe or missing")
    receipt = load_json_strict(source)
    digest = verify_manifest_hash(receipt)
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("kind") != LAUNCH_RECEIPT_KIND
        or receipt.get("sample_mode") is not True
        or receipt.get("all_lanes_completed") is not True
        or receipt.get("lane_plan_sha256") != lane_plan["canonical_sha256"]
        or receipt.get("policy_sha256") != source_plan["policy_sha256"]
        or receipt.get("source_plan_sha256") != source_plan["canonical_sha256"]
        or receipt.get("calibration_sha256") != calibration["canonical_sha256"]
    ):
        raise TurkishCorpusError("packed object launch receipt binding drift")
    allocation = receipt.get("allocation")
    if (
        not isinstance(allocation, Mapping)
        or allocation.get("nodes") != 1
        or allocation.get("tasks") != LANE_COUNT
        or allocation.get("cpus_per_task") != CPUS_PER_LANE
        or allocation.get("allocated_cpus") != LANE_COUNT * CPUS_PER_LANE
        or not str(allocation.get("slurm_job_id", "")).isdigit()
        or not str(allocation.get("slurm_step_id", "")).isdigit()
        or not allocation.get("slurm_node")
    ):
        raise TurkishCorpusError("packed object launch allocation drift")
    records = receipt.get("object_receipts")
    expected_ranks = lane_plan["resource_sample_ranks"]["ranks"]
    if not isinstance(records, list) or [
        item.get("rank") if isinstance(item, Mapping) else None for item in records
    ] != expected_ranks:
        raise TurkishCorpusError("packed object launch inventory drift")
    object_hashes = []
    for item in records:
        rank = int(item["rank"])
        expected_path = f"objects/{rank:05d}/object_receipt.json"
        if item.get("path") != expected_path:
            raise TurkishCorpusError("packed object launch path drift")
        object_path = run_root / expected_path
        if object_path.is_symlink() or not object_path.is_file():
            raise TurkishCorpusError(f"packed object receipt {rank} is unsafe or missing")
        object_receipt = load_json_strict(object_path)
        object_sha256 = verify_manifest_hash(object_receipt)
        if (
            item.get("canonical_sha256") != object_sha256
            or object_receipt.get("rank") != rank
            or object_receipt.get("sample_mode") is not True
            or object_receipt.get("source_plan_sha256")
            != source_plan["canonical_sha256"]
            or object_receipt.get("calibration_sha256")
            != calibration["canonical_sha256"]
        ):
            raise TurkishCorpusError(f"packed object receipt {rank} binding drift")
        object_hashes.append(object_sha256)
    return digest, object_hashes


def _slurm_bucket_context(env: Mapping[str, str]) -> dict[str, Any]:
    tasks = _positive_slurm_int(env, "SLURM_NTASKS")
    cpus_per_task = _positive_slurm_int(env, "SLURM_CPUS_PER_TASK")
    nodes = _positive_slurm_int(env, "SLURM_NNODES")
    if (
        tasks != BUCKET_COUNT
        or cpus_per_task != CPUS_PER_BUCKET_TASK
        or nodes != 1
    ):
        raise TurkishCorpusError(
            "packed buckets require one node, fourteen tasks, and eight CPUs per task"
        )
    try:
        proc_id = int(env.get("SLURM_PROCID", ""))
        local_id = int(env.get("SLURM_LOCALID", ""))
    except ValueError as exc:
        raise TurkishCorpusError("Slurm bucket coordinates must be integers") from exc
    if proc_id not in range(BUCKET_COUNT) or local_id != proc_id:
        raise TurkishCorpusError(
            "packed bucket tasks must map one-to-one to MinHash ranks 0..13"
        )
    job_id = env.get("SLURM_JOB_ID", "")
    step_id = env.get("SLURM_STEP_ID", "")
    node_name = env.get("SLURMD_NODENAME", "")
    if not job_id.isdigit() or not step_id.isdigit() or not node_name:
        raise TurkishCorpusError("packed buckets must run in a numeric Slurm job step")
    thread_caps = {name: env.get(name) for name in THREAD_CAPS}
    if any(value != "1" for value in thread_caps.values()):
        raise TurkishCorpusError("OMP/BLAS thread caps must all be exactly one")
    return {
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "slurm_node": node_name,
        "nodes": nodes,
        "tasks": tasks,
        "cpus_per_task": cpus_per_task,
        "proc_id": proc_id,
        "local_id": local_id,
        "allocated_cpus": tasks * cpus_per_task,
        "thread_caps": thread_caps,
    }


def _validate_backend_bucket_receipt(
    receipt: Mapping[str, Any],
    *,
    rank: int,
    source_plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    object_receipt_hashes: list[str],
    run_root: Path,
) -> str:
    digest = verify_manifest_hash(receipt)
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("kind") != BACKEND_BUCKET_RECEIPT_KIND
        or receipt.get("sample_mode") is not True
        or receipt.get("rank") != rank
        or receipt.get("world_size") != BUCKET_COUNT
        or receipt.get("source_plan_sha256") != source_plan["canonical_sha256"]
        or receipt.get("calibration_sha256")
        != calibration["canonical_sha256"]
        or receipt.get("object_receipt_sha256") != object_receipt_hashes
    ):
        raise TurkishCorpusError(f"backend bucket {rank} receipt binding drift")
    output = receipt.get("output")
    expected_path = f"bucket_matches/{rank:05d}_00.dups"
    if not isinstance(output, Mapping) or output.get("path") != expected_path:
        raise TurkishCorpusError(f"backend bucket {rank} output path drift")
    size_bytes = output.get("size_bytes")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
        or size_bytes % 16
        or output.get("duplicate_edges") != size_bytes // 16
    ):
        raise TurkishCorpusError(f"backend bucket {rank} output structure drift")
    output_path = run_root / expected_path
    if (
        output_path.is_symlink()
        or not output_path.is_file()
        or output_path.stat().st_size != size_bytes
        or file_sha256(output_path) != output.get("sha256")
    ):
        raise TurkishCorpusError(f"backend bucket {rank} output hash drift")
    return digest


def run_bucket_task(
    policy_path: str | Path,
    source_plan_path: str | Path,
    calibration_path: str | Path,
    sample_ranks_path: str | Path,
    lane_plan_path: str | Path,
    object_launch_receipt_path: str | Path,
    run_dir: str | Path,
    receipt_dir: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    policy, source_plan, calibration = _load_bound_inputs(
        policy_path, source_plan_path, calibration_path
    )
    lane_plan = load_json_strict(lane_plan_path)
    validate_lane_plan(
        lane_plan,
        policy=policy,
        source_plan=source_plan,
        calibration=calibration,
        sample_ranks_path=sample_ranks_path,
    )
    run_root = Path(run_dir)
    object_launch_sha256, object_receipt_hashes = (
        _validate_object_sample_launch_receipt(
            object_launch_receipt_path,
            lane_plan=lane_plan,
            source_plan=source_plan,
            calibration=calibration,
            run_root=run_root,
        )
    )
    context = _slurm_bucket_context(os.environ if env is None else env)
    rank = int(context["proc_id"])
    destination = Path(receipt_dir) / f"bucket_{rank:05d}.json"
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite packed bucket task receipt: {destination}"
        )
    backend_path = run_root / "bucket_receipts" / f"{rank:05d}.json"
    preexisting = backend_path.is_file()
    backend_receipt = run_datatrove_bucket(
        policy,
        source_plan,
        calibration,
        run_root,
        rank=rank,
        sample_mode=True,
        resource_approval_path=None,
    )
    backend_sha256 = _validate_backend_bucket_receipt(
        backend_receipt,
        rank=rank,
        source_plan=source_plan,
        calibration=calibration,
        object_receipt_hashes=object_receipt_hashes,
        run_root=run_root,
    )
    task_receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": BUCKET_TASK_RECEIPT_KIND,
            "sample_mode": True,
            "policy_sha256": source_plan["policy_sha256"],
            "source_plan_sha256": source_plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "object_sample_launch_receipt_sha256": object_launch_sha256,
            "assignment": {
                "algorithm": BUCKET_ASSIGNMENT_ALGORITHM,
                "bucket_rank": rank,
                "world_size": BUCKET_COUNT,
            },
            "allocation": context,
            "backend_bucket_receipt": {
                "rank": rank,
                "path": f"bucket_receipts/{rank:05d}.json",
                "canonical_sha256": backend_sha256,
                "disposition": (
                    "reused_verified" if preexisting else "produced_by_allocation"
                ),
            },
            "bucket_completed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, task_receipt)
    return task_receipt


def _validate_bucket_task_receipt(
    receipt: Mapping[str, Any],
    *,
    rank: int,
    source_plan: Mapping[str, Any],
    calibration: Mapping[str, Any],
    object_launch_sha256: str,
    object_receipt_hashes: list[str],
    run_root: Path,
    job_id: str,
) -> tuple[str, str]:
    digest = verify_manifest_hash(receipt)
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("kind") != BUCKET_TASK_RECEIPT_KIND
        or receipt.get("sample_mode") is not True
        or receipt.get("bucket_completed") is not True
        or receipt.get("policy_sha256") != source_plan["policy_sha256"]
        or receipt.get("source_plan_sha256") != source_plan["canonical_sha256"]
        or receipt.get("calibration_sha256")
        != calibration["canonical_sha256"]
        or receipt.get("object_sample_launch_receipt_sha256")
        != object_launch_sha256
        or receipt.get("assignment")
        != {
            "algorithm": BUCKET_ASSIGNMENT_ALGORITHM,
            "bucket_rank": rank,
            "world_size": BUCKET_COUNT,
        }
    ):
        raise TurkishCorpusError(f"packed bucket task {rank} binding drift")
    allocation = receipt.get("allocation")
    if (
        not isinstance(allocation, Mapping)
        or allocation.get("slurm_job_id") != job_id
        or allocation.get("nodes") != 1
        or allocation.get("tasks") != BUCKET_COUNT
        or allocation.get("cpus_per_task") != CPUS_PER_BUCKET_TASK
        or allocation.get("allocated_cpus")
        != BUCKET_COUNT * CPUS_PER_BUCKET_TASK
        or allocation.get("proc_id") != rank
        or allocation.get("local_id") != rank
        or allocation.get("thread_caps") != {name: "1" for name in THREAD_CAPS}
        or not str(allocation.get("slurm_step_id", "")).isdigit()
        or not allocation.get("slurm_node")
    ):
        raise TurkishCorpusError(f"packed bucket task {rank} allocation drift")
    backend_record = receipt.get("backend_bucket_receipt")
    expected_path = f"bucket_receipts/{rank:05d}.json"
    if (
        not isinstance(backend_record, Mapping)
        or backend_record.get("rank") != rank
        or backend_record.get("path") != expected_path
        or backend_record.get("disposition")
        not in {"produced_by_allocation", "reused_verified"}
    ):
        raise TurkishCorpusError(f"packed bucket task {rank} inventory drift")
    backend_path = run_root / expected_path
    if backend_path.is_symlink() or not backend_path.is_file():
        raise TurkishCorpusError(f"backend bucket {rank} receipt is unsafe or missing")
    backend_receipt = load_json_strict(backend_path)
    backend_sha256 = _validate_backend_bucket_receipt(
        backend_receipt,
        rank=rank,
        source_plan=source_plan,
        calibration=calibration,
        object_receipt_hashes=object_receipt_hashes,
        run_root=run_root,
    )
    if backend_record.get("canonical_sha256") != backend_sha256:
        raise TurkishCorpusError(f"packed bucket task {rank} receipt hash drift")
    return digest, backend_sha256


def seal_bucket_launch_receipt(
    policy_path: str | Path,
    source_plan_path: str | Path,
    calibration_path: str | Path,
    sample_ranks_path: str | Path,
    lane_plan_path: str | Path,
    object_launch_receipt_path: str | Path,
    run_dir: str | Path,
    task_receipt_dir: str | Path,
    output_path: str | Path,
    *,
    job_id: str,
) -> dict[str, Any]:
    if not job_id.isdigit():
        raise TurkishCorpusError("Slurm job ID must be numeric")
    policy, source_plan, calibration = _load_bound_inputs(
        policy_path, source_plan_path, calibration_path
    )
    lane_plan = load_json_strict(lane_plan_path)
    validate_lane_plan(
        lane_plan,
        policy=policy,
        source_plan=source_plan,
        calibration=calibration,
        sample_ranks_path=sample_ranks_path,
    )
    run_root = Path(run_dir)
    object_launch_sha256, object_receipt_hashes = (
        _validate_object_sample_launch_receipt(
            object_launch_receipt_path,
            lane_plan=lane_plan,
            source_plan=source_plan,
            calibration=calibration,
            run_root=run_root,
        )
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite packed bucket launch receipt: {destination}"
        )
    task_root = Path(task_receipt_dir)
    task_records = []
    bucket_records = []
    step_ids: set[str] = set()
    nodes: set[str] = set()
    for rank in range(BUCKET_COUNT):
        path = task_root / f"bucket_{rank:05d}.json"
        if path.is_symlink() or not path.is_file():
            raise TurkishCorpusError(f"packed bucket task {rank} receipt is missing")
        receipt = load_json_strict(path)
        task_sha256, backend_sha256 = _validate_bucket_task_receipt(
            receipt,
            rank=rank,
            source_plan=source_plan,
            calibration=calibration,
            object_launch_sha256=object_launch_sha256,
            object_receipt_hashes=object_receipt_hashes,
            run_root=run_root,
            job_id=job_id,
        )
        step_ids.add(str(receipt["allocation"]["slurm_step_id"]))
        nodes.add(str(receipt["allocation"]["slurm_node"]))
        task_records.append(
            {
                "bucket_rank": rank,
                "path": f"tasks/bucket_{rank:05d}.json",
                "canonical_sha256": task_sha256,
            }
        )
        bucket_records.append(
            {
                "bucket_rank": rank,
                "path": f"bucket_receipts/{rank:05d}.json",
                "canonical_sha256": backend_sha256,
            }
        )
    if len(step_ids) != 1 or len(nodes) != 1:
        raise TurkishCorpusError(
            "packed bucket tasks do not share one Slurm step and node"
        )
    launch_receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": BUCKET_LAUNCH_RECEIPT_KIND,
            "sample_mode": True,
            "policy_sha256": source_plan["policy_sha256"],
            "source_plan_sha256": source_plan["canonical_sha256"],
            "calibration_sha256": calibration["canonical_sha256"],
            "object_sample_launch_receipt_sha256": object_launch_sha256,
            "assignment": {
                "algorithm": BUCKET_ASSIGNMENT_ALGORITHM,
                "bucket_ranks": list(range(BUCKET_COUNT)),
                "world_size": BUCKET_COUNT,
            },
            "allocation": {
                "slurm_job_id": job_id,
                "slurm_step_id": next(iter(step_ids)),
                "slurm_node": next(iter(nodes)),
                "nodes": 1,
                "tasks": BUCKET_COUNT,
                "cpus_per_task": CPUS_PER_BUCKET_TASK,
                "allocated_cpus": BUCKET_COUNT * CPUS_PER_BUCKET_TASK,
            },
            "task_receipts": task_records,
            "backend_bucket_receipts": bucket_records,
            "all_buckets_completed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, launch_receipt)
    return launch_receipt


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--sample-ranks", type=Path, required=True)
    parser.add_argument("--lane-plan", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("seal-plan", help="seal the exact thirty-two-lane rank plan")
    _common_inputs(plan)

    lane = sub.add_parser("run-lane", help="run one sample-only Slurm lane")
    _common_inputs(lane)
    lane.add_argument("--model", type=Path, required=True)
    lane.add_argument("--run-dir", type=Path, required=True)
    lane.add_argument("--receipt-dir", type=Path, required=True)

    finalize = sub.add_parser(
        "finalize", help="seal unanimous lane and object completion evidence"
    )
    _common_inputs(finalize)
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--receipt-dir", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--job-id", required=True)

    bucket = sub.add_parser(
        "run-bucket", help="run one fixed sample-only MinHash bucket task"
    )
    _common_inputs(bucket)
    bucket.add_argument("--object-launch-receipt", type=Path, required=True)
    bucket.add_argument("--run-dir", type=Path, required=True)
    bucket.add_argument("--receipt-dir", type=Path, required=True)

    finalize_buckets = sub.add_parser(
        "finalize-buckets", help="seal unanimous completion of all 14 buckets"
    )
    _common_inputs(finalize_buckets)
    finalize_buckets.add_argument(
        "--object-launch-receipt", type=Path, required=True
    )
    finalize_buckets.add_argument("--run-dir", type=Path, required=True)
    finalize_buckets.add_argument("--receipt-dir", type=Path, required=True)
    finalize_buckets.add_argument("--output", type=Path, required=True)
    finalize_buckets.add_argument("--job-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "seal-plan":
            result = seal_lane_plan(
                args.policy,
                args.source_plan,
                args.calibration,
                args.sample_ranks,
                args.lane_plan,
            )
        elif args.command == "run-lane":
            result = run_lane(
                args.policy,
                args.source_plan,
                args.calibration,
                args.sample_ranks,
                args.lane_plan,
                args.model,
                args.run_dir,
                args.receipt_dir,
            )
        elif args.command == "finalize":
            result = seal_launch_receipt(
                args.policy,
                args.source_plan,
                args.calibration,
                args.sample_ranks,
                args.lane_plan,
                args.run_dir,
                args.receipt_dir,
                args.output,
                job_id=args.job_id,
            )
        elif args.command == "run-bucket":
            result = run_bucket_task(
                args.policy,
                args.source_plan,
                args.calibration,
                args.sample_ranks,
                args.lane_plan,
                args.object_launch_receipt,
                args.run_dir,
                args.receipt_dir,
            )
        else:
            result = seal_bucket_launch_receipt(
                args.policy,
                args.source_plan,
                args.calibration,
                args.sample_ranks,
                args.lane_plan,
                args.object_launch_receipt,
                args.run_dir,
                args.receipt_dir,
                args.output,
                job_id=args.job_id,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
