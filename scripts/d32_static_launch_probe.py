"""Exercise the exact srun-native distributed topology without torchrun.

Every rank initializes NCCL from Slurm-derived ``env://`` variables, performs
an all-reduce and all-gather, destroys the process group, and only then writes
its immutable rank receipt.  The batch launcher aggregates those receipts, so
a missing rank or an unclean rendezvous shutdown can never pass silently.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from nanochat.experiment_manifest import seal_manifest, write_json_atomic
from scripts.d32_family_workflow import load_recipe


def _required_int(name: str) -> int:
    value = os.environ.get(name, "")
    try:
        result = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, found {value!r}") from exc
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()

    recipe, recipe_sha = load_recipe(args.recipe)
    rank = _required_int("RANK")
    torch_local_rank = _required_int("LOCAL_RANK")
    local_rank = _required_int("NODE_LOCAL_RANK")
    world_size = _required_int("WORLD_SIZE")
    local_world_size = _required_int("LOCAL_WORLD_SIZE")
    if world_size not in {4, 8, 16} or local_world_size != 4:
        raise RuntimeError("static probe requires 4, 8, or 16 ranks and four ranks per node")
    if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise RuntimeError("Slurm-derived rank mapping is outside the declared world")
    if torch_local_rank != 0 or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "each --gpus-per-task=1 static rank must see one GPU at local index zero; "
            f"found {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(torch_local_rank)
    device = torch.device("cuda", torch_local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=3),
    )
    reduced = torch.tensor(float(rank), device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    expected_sum = world_size * (world_size - 1) / 2
    if reduced.item() != expected_sum:
        raise RuntimeError(
            f"NCCL all-reduce mismatch: expected {expected_sum}, found {reduced.item()}"
        )
    local_identity = {
        "rank": rank,
        "local_rank": local_rank,
        "torch_local_rank": torch_local_rank,
        "node": socket.gethostname(),
        "device_index": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(device),
        "visible_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    gathered: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_identity)
    if sorted(int(record["rank"]) for record in gathered if record is not None) != list(
        range(world_size)
    ):
        raise RuntimeError("NCCL all-gather did not return every rank exactly once")
    dist.barrier()
    dist.destroy_process_group()

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    slurm_step_id = os.environ.get("SLURM_STEP_ID", "")
    if not slurm_job_id or not slurm_step_id:
        raise RuntimeError("static probe requires Slurm job and step IDs")
    receipt = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_static_srun_probe_rank",
            "family_id": recipe["family_id"],
            "recipe_sha256": recipe_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "launcher": "slurm_srun_direct_python_env_v1",
            "run_id": args.run_id,
            "phase": args.phase,
            "slurm_job_id": slurm_job_id,
            "slurm_step_id": slurm_step_id,
            "node": local_identity["node"],
            "rank": rank,
            "local_rank": local_rank,
            "torch_local_rank": torch_local_rank,
            "world_size": world_size,
            "local_world_size": local_world_size,
            "device_index": local_identity["device_index"],
            "device_name": local_identity["device_name"],
            "visible_device_count": local_identity["visible_device_count"],
            "cuda_visible_devices": local_identity["cuda_visible_devices"],
            "collective": {
                "backend": "nccl",
                "all_reduce_expected": expected_sum,
                "all_reduce_observed": reduced.item(),
                "all_gather_world_size": len(gathered),
                "final_barrier_completed": True,
                "process_group_destroyed_before_receipt": True,
            },
            "child_exit_code": 0,
            "termination": "clean",
            "canonical_sha256": None,
        }
    )
    output = args.receipt_dir / f"rank_{rank:05d}.json"
    write_json_atomic(output, receipt)
    if rank == 0:
        print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
