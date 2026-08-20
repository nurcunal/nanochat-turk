"""Seal and execute the one-node packed Turkish v2 object resource sample.

The launcher is deliberately sample-only.  It accepts the deterministic ranks
written by ``uhem_turkish_data_bootstrap.sbatch``, binds them to the sealed v2
source plan and calibration, and assigns them round-robin to exactly eight
Slurm workers.  There is no production-mode switch in this interface.
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
    select_resource_sample_ranks,
    validate_backend_calibration,
    validate_source_plan,
)
from nanochat.turkish_corpus import TurkishCorpusError, load_corpus_policy


LANE_COUNT = 8
CPUS_PER_LANE = 16
V2_POLICY_NAME = "tr_general_clean_v2"
LANE_PLAN_KIND = "turkish_packed_resource_sample_lane_plan"
LANE_RECEIPT_KIND = "turkish_packed_resource_sample_lane_receipt"
LAUNCH_RECEIPT_KIND = "turkish_packed_resource_sample_launch_receipt"
ASSIGNMENT_ALGORITHM = "sorted_rank_round_robin_v1"
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
            "packed sample requires one node, eight tasks, and sixteen CPUs per task"
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


def _common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--sample-ranks", type=Path, required=True)
    parser.add_argument("--lane-plan", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("seal-plan", help="seal the exact eight-lane rank plan")
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
        else:
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
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
