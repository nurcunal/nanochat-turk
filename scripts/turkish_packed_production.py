"""Execute the sealed Turkish production backend with packed cpu2dq allocations.

Object work is submitted as one single-node Slurm array task per pack-plan
node. Each allocation runs the exact local lanes from that sealed plan. The
fourteen MinHash bands then run concurrently in one separate node allocation.
Every production entry point requires the accepted resource/quality decisions
and the post-safety d32 storage gate; this module has no sample-mode switch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    validate_backend_calibration,
    validate_mixture_quality_approval,
    validate_resource_approval,
    validate_source_plan,
)
from nanochat.turkish_corpus import TurkishCorpusError, load_corpus_policy
from scripts.d32_family_workflow import (
    CPU2DQ_BILLABLE_CPUS,
    DATA_PREP_FIXED_CPU2DQ_CEILINGS,
    DATA_PREP_FUTURE_CPU_COMPONENTS,
    _data_prep_policy_sha256,
    _live_beegfs_storage,
    _live_uhem_cpu_saat,
    _validate_data_prep_storage_gate_receipt,
    _validate_production_pack_plan,
    load_recipe,
)


OBJECT_LANE_KIND = "turkish_packed_production_object_lane_receipt"
OBJECT_NODE_KIND = "turkish_packed_production_object_node_receipt"
BUCKET_TASK_KIND = "turkish_packed_production_bucket_task_receipt"
BUCKET_LAUNCH_KIND = "turkish_packed_production_bucket_launch_receipt"
CLUSTER_INPUT_KIND = "turkish_packed_production_cluster_input_receipt"
CLUSTER_LAUNCH_KIND = "turkish_packed_production_cluster_launch_receipt"
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

# These sources may remain in historical/sample policies so their audit evidence
# stays reproducible, but they must never cross the production launch boundary.
# FinePDF was disqualified after manual Turkish review found OCR/read-order/layout
# corruption even in rows whose declared extractor was not the OCR extractor.
# Raw FineWeb2 was disqualified after its structural gates retained 188/192
# representative documents while manual review found 7/24 materially bad or
# off-purpose rows; global deduplication cannot repair that semantic failure.
PRODUCTION_DISQUALIFIED_SOURCE_IDS = frozenset(
    {"finepdfs_edu_tr", "fineweb2_tr"}
)
PRODUCTION_ALLOWED_TEXT_ORIGINS = frozenset(
    {"born_digital_text", "structured_text"}
)
PRODUCTION_TEXT_INTEGRITY_LIMITS = {
    "max_unicode_replacement_characters": 0,
    "max_mojibake_sequence_hits": 0,
    "max_c1_control_characters": 0,
    "max_unicode_surrogate_characters": 0,
}


def _validate_production_source_eligibility(policy: Mapping[str, Any]) -> None:
    sources = policy.get("sources")
    if not isinstance(sources, list):
        raise TurkishCorpusError("production policy has no valid source inventory")
    source_inventory = {
        str(source.get("id", "")): source
        for source in sources
        if isinstance(source, Mapping) and str(source.get("id", ""))
    }
    mixture = policy.get("mixture")
    if not isinstance(mixture, list) or not mixture:
        raise TurkishCorpusError("production policy has no valid mixture")
    if any(
        not isinstance(bucket, Mapping)
        or not isinstance(bucket.get("source_id"), str)
        or not bucket["source_id"]
        for bucket in mixture
    ):
        raise TurkishCorpusError("production mixture has an invalid source_id")
    selected = {
        str(bucket["source_id"])
        for bucket in mixture
    }
    disqualified = sorted(selected & PRODUCTION_DISQUALIFIED_SOURCE_IDS)
    if disqualified:
        raise TurkishCorpusError(
            "production policy selects manually disqualified sources: "
            f"{disqualified}"
        )
    missing = sorted(selected - set(source_inventory))
    if missing:
        raise TurkishCorpusError(
            f"production policy selects sources absent from its inventory: {missing}"
        )
    invalid_origins: dict[str, Any] = {}
    for source_id in sorted(selected):
        origin = source_inventory[source_id].get("text_origin")
        if not isinstance(origin, str) or origin not in PRODUCTION_ALLOWED_TEXT_ORIGINS:
            invalid_origins[source_id] = origin
    if invalid_origins:
        raise TurkishCorpusError(
            "every production source must explicitly declare text_origin as "
            "born_digital_text or structured_text; PDF-extracted, OCR-derived, "
            f"mixed, missing, and unknown origins are forbidden: {invalid_origins}"
        )
    content_policy = policy.get("content_policy")
    if not isinstance(content_policy, Mapping):
        raise TurkishCorpusError("production policy has no valid content_policy")
    for key, expected in PRODUCTION_TEXT_INTEGRITY_LIMITS.items():
        value = content_policy.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise TurkishCorpusError(
                f"production content_policy.{key} must be exactly {expected}"
            )


def _positive_env(env: Mapping[str, str], name: str) -> int:
    try:
        value = int(env.get(name, ""))
    except ValueError as exc:
        raise TurkishCorpusError(f"{name} must be an integer") from exc
    if value <= 0:
        raise TurkishCorpusError(f"{name} must be positive")
    return value


def _load_inputs(
    recipe_path: Path,
    policy_path: Path,
    source_plan_path: Path,
    calibration_path: Path,
    pack_plan_path: Path,
    resource_approval_path: Path,
    mixture_quality_approval_path: Path,
    storage_gate_path: Path,
) -> dict[str, Any]:
    recipe, recipe_sha = load_recipe(recipe_path)
    policy = load_corpus_policy(policy_path)
    _validate_production_source_eligibility(policy)
    source_plan = load_json_strict(source_plan_path)
    calibration = load_json_strict(calibration_path)
    validate_source_plan(source_plan, policy)
    validate_backend_calibration(calibration, policy)
    policy_sha = _data_prep_policy_sha256(policy)
    source_plan_sha = str(source_plan["canonical_sha256"])
    calibration_sha = str(calibration["canonical_sha256"])

    pack_plan = load_json_strict(pack_plan_path)
    pack_plan_sha = verify_manifest_hash(pack_plan)
    _validate_production_pack_plan(
        pack_plan,
        recipe=recipe,
        recipe_sha=recipe_sha,
        policy_sha=policy_sha,
        source_plan=source_plan,
        source_plan_sha=source_plan_sha,
    )
    mixture_approval = load_json_strict(mixture_quality_approval_path)
    mixture_approval_sha = validate_mixture_quality_approval(
        mixture_approval,
        policy=policy,
        plan=source_plan,
        calibration=calibration,
        approval_path=mixture_quality_approval_path,
    )
    resource_approval = load_json_strict(resource_approval_path)
    resource_approval_sha = verify_manifest_hash(resource_approval)
    validate_resource_approval(
        resource_approval,
        plan=source_plan,
        policy=policy,
        calibration=calibration,
        approval_path=resource_approval_path,
    )
    if resource_approval.get("mixture_quality_approval_sha256") != mixture_approval_sha:
        raise TurkishCorpusError(
            "resource approval does not bind the supplied mixture-quality approval"
        )

    gate = load_json_strict(storage_gate_path)
    gate_sha = verify_manifest_hash(gate)
    _validate_data_prep_storage_gate_receipt(
        gate, recipe=recipe, recipe_sha=recipe_sha
    )
    if (
        gate.get("schema_version") != "3.0"
        or gate.get("kind") != "d32_data_prep_storage_gate"
        or gate.get("family_id") != recipe["family_id"]
        or gate.get("recipe_sha256") != recipe_sha
        or gate.get("policy_sha256") != policy_sha
        or gate.get("source_plan_sha256") != source_plan_sha
        or gate.get("calibration_sha256") != calibration_sha
        or gate.get("production_pack_plan_sha256") != pack_plan_sha
        or gate.get("resource_approval_sha256") != resource_approval_sha
        or gate.get("mixture_quality_approval_sha256") != mixture_approval_sha
        or gate.get("backend_resource_report_sha256")
        != resource_approval.get("resource_report_sha256")
        or gate.get("sample_cluster_receipt_sha256")
        != mixture_approval.get("sample_cluster_receipt_sha256")
        or gate.get("sample_cluster_receipt_sha256")
        != resource_approval.get("sample_cluster_receipt_sha256")
        or gate.get("safety_factor") != 1.35
        or gate.get("safety_factor_application_count") != 1
        or gate.get("never_auto_delete_existing_artifacts") is not True
    ):
        raise TurkishCorpusError("post-safety d32 storage gate binding drift")
    future = gate.get("future_resource_projection")
    if not isinstance(future, Mapping) or set(future.get("components", {})) != set(
        DATA_PREP_FUTURE_CPU_COMPONENTS
    ):
        raise TurkishCorpusError("storage gate future CPU inventory drift")
    details = future.get("allocation_details")
    if not isinstance(details, Mapping):
        raise TurkishCorpusError("storage gate allocation ceilings are missing")
    for stage, ceiling in DATA_PREP_FIXED_CPU2DQ_CEILINGS.items():
        component = future["components"].get(stage)
        expected_details = {
            "allocation_contract": "one_exclusive_128cpu_cpu2dq_node",
            "maximum_wall_hours": ceiling / CPU2DQ_BILLABLE_CPUS,
            "projected_cpu_saat_before_safety": float(ceiling),
            "submission_must_not_exceed_ceiling": True,
        }
        if (
            not isinstance(component, Mapping)
            or float(component.get("projected_cpu_saat_before_safety", -1))
            != float(ceiling)
            or details.get(stage) != expected_details
        ):
            raise TurkishCorpusError(f"storage gate {stage} ceiling drift")
    return {
        "recipe": recipe,
        "recipe_sha": recipe_sha,
        "policy": policy,
        "policy_sha": policy_sha,
        "source_plan": source_plan,
        "source_plan_sha": source_plan_sha,
        "calibration": calibration,
        "calibration_sha": calibration_sha,
        "pack_plan": pack_plan,
        "pack_plan_sha": pack_plan_sha,
        "resource_approval_sha": resource_approval_sha,
        "mixture_approval_sha": mixture_approval_sha,
        "storage_gate_sha": gate_sha,
        "storage_gate": gate,
        "sample_cluster_receipt_sha": gate[
            "sample_cluster_receipt_sha256"
        ],
    }


def validate_launch_gate(*args: Path) -> dict[str, Any]:
    """Public read-only validation used by downstream bounded CPU launchers."""

    return _load_inputs(*args)


def validate_live_launch_gate(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Re-query the same UHeM storage/quota constraints immediately before launch."""

    gate = inputs["storage_gate"]
    recipe = inputs["recipe"]
    raw_work_dir = Path(str(gate["work_dir"])).expanduser()
    if raw_work_dir.is_symlink() or not raw_work_dir.is_dir():
        raise TurkishCorpusError("gated BeeGFS work directory is unsafe or missing")
    work_dir = raw_work_dir.resolve()
    if work_dir.stat().st_dev != int(gate["work_dir_filesystem_device"]):
        raise TurkishCorpusError("gated BeeGFS filesystem device drifted")
    storage_policy = recipe["storage"]["uhem_live_quota"]
    free, storage_audit = _live_beegfs_storage(
        Path.cwd(),
        uid=int(storage_policy["uid"]),
        storage_pool_id=int(storage_policy["storage_pool_id"]),
        path=work_dir,
    )
    required_free = int(gate["required_free_bytes_including_headroom"])
    if free < required_free:
        raise TurkishCorpusError("live BeeGFS headroom fell below the sealed gate")
    budget = recipe["uhem_budget"]
    remaining, quota_sha, quota_audit = _live_uhem_cpu_saat(
        Path.cwd(), str(budget["account"]), str(budget["user"])
    )
    required_cpu = int(gate["total_project_operational_ceiling_cpu_saat"])
    if remaining < required_cpu:
        raise TurkishCorpusError("live UHeM CPU quota fell below the sealed gate")
    return {
        "work_dir": str(work_dir),
        "work_dir_filesystem_device": int(gate["work_dir_filesystem_device"]),
        "required_free_bytes": required_free,
        "observed_free_bytes": free,
        "remaining_cpu_saat": remaining,
        "required_cpu_saat": required_cpu,
        "quota_output_sha256": quota_sha,
        "storage": storage_audit,
        "quota": quota_audit,
    }


def validate_gated_write_dir(inputs: Mapping[str, Any], path: Path) -> str:
    """Require a downstream output directory on the exact gated BeeGFS tree."""

    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise TurkishCorpusError("gated write directory is unsafe or missing")
    resolved = raw.resolve()
    gated = Path(str(inputs["storage_gate"]["work_dir"])).expanduser().resolve()
    if (
        resolved != gated
        and gated not in resolved.parents
        or resolved.stat().st_dev
        != int(inputs["storage_gate"]["work_dir_filesystem_device"])
    ):
        raise TurkishCorpusError("write directory is outside the gated BeeGFS tree")
    return str(resolved)


def validate_downstream_lineage(
    inputs: Mapping[str, Any],
    cluster_launch_path: Path,
    *,
    pool_dir: Path | None = None,
    tokenizer_sample_dir: Path | None = None,
    tokenizer_dir: Path | None = None,
    tokenizer_quality_dir: Path | None = None,
    packing_preflight_dir: Path | None = None,
) -> dict[str, Any]:
    """Bind every downstream data artifact to this exact production launch."""

    launch = load_json_strict(cluster_launch_path)
    launch_sha = verify_manifest_hash(launch)
    expected_bindings = _receipt_bindings(inputs)
    if (
        launch.get("schema_version") != "2.0"
        or launch.get("kind") != CLUSTER_LAUNCH_KIND
        or launch.get("cluster_completed") is not True
        or any(launch.get(key) != value for key, value in expected_bindings.items())
    ):
        raise TurkishCorpusError("downstream cluster-launch lineage drift")
    expected_chain = {
        "cluster_launch_receipt_sha256": launch_sha,
        "production_pack_plan_sha256": inputs["pack_plan_sha"],
        "resource_approval_sha256": inputs["resource_approval_sha"],
        "mixture_quality_approval_sha256": inputs["mixture_approval_sha"],
        "data_prep_storage_gate_sha256": inputs["storage_gate_sha"],
        "sample_cluster_receipt_sha256": inputs["sample_cluster_receipt_sha"],
    }

    def require(path: Path, label: str) -> dict[str, Any]:
        receipt = load_json_strict(path)
        verify_manifest_hash(receipt)
        if (
            receipt.get("policy_sha256") != inputs["policy_sha"]
            or receipt.get("production_chain") != expected_chain
        ):
            raise TurkishCorpusError(f"{label} production lineage drift")
        return receipt

    checked: dict[str, str] = {"cluster_launch_receipt_sha256": launch_sha}
    pool_manifest: dict[str, Any] | None = None
    pool_qa_sha: str | None = None
    tokenizer_training_sha: str | None = None
    if pool_dir is not None:
        pool_manifest = require(pool_dir / "corpus_manifest.json", "filtered pool")
        if pool_manifest.get("stage") != "filtered_pool":
            raise TurkishCorpusError("downstream pool is not a filtered pool")
        qa_approval = load_json_strict(pool_dir / "qa" / "qa_approval.json")
        pool_qa_sha = verify_manifest_hash(qa_approval)
        if qa_approval.get("decision") != "accepted":
            raise TurkishCorpusError("downstream pool QA approval is not accepted")
        checked["pool_manifest_sha256"] = pool_manifest["canonical_sha256"]
        checked["pool_qa_approval_sha256"] = pool_qa_sha

    def require_pool_binding(receipt: Mapping[str, Any], label: str) -> None:
        if pool_manifest is not None and (
            receipt.get("parent_corpus_manifest_sha256")
            != pool_manifest["canonical_sha256"]
            or receipt.get("qa_approval_sha256") != pool_qa_sha
        ):
            raise TurkishCorpusError(f"{label} parent-pool/QA binding drift")

    if tokenizer_sample_dir is not None:
        receipt = require(
            tokenizer_sample_dir / "tokenizer_sample_manifest.json",
            "tokenizer sample",
        )
        require_pool_binding(receipt, "tokenizer sample")
        checked["tokenizer_sample_sha256"] = receipt["canonical_sha256"]
    if tokenizer_dir is not None:
        package = require(tokenizer_dir / "package_manifest.json", "tokenizer package")
        training = require(
            tokenizer_dir / "training_receipt.json", "tokenizer training receipt"
        )
        if package.get("training_receipt_sha256") != training["canonical_sha256"]:
            raise TurkishCorpusError("tokenizer package/training receipt binding drift")
        require_pool_binding(package, "tokenizer package")
        require_pool_binding(training, "tokenizer training receipt")
        tokenizer_training_sha = training["canonical_sha256"]
        checked["tokenizer_package_sha256"] = package["canonical_sha256"]
    if tokenizer_quality_dir is not None:
        if tokenizer_training_sha is None:
            raise TurkishCorpusError(
                "tokenizer quality validation requires the verified tokenizer receipt"
            )
        report = require(
            tokenizer_quality_dir / "quality_report.json", "tokenizer quality report"
        )
        require_pool_binding(report, "tokenizer quality report")
        if report.get("training_receipt_sha256") != tokenizer_training_sha:
            raise TurkishCorpusError(
                "tokenizer quality report/training receipt binding drift"
            )
        approval_path = tokenizer_quality_dir / "quality_approval.json"
        if approval_path.is_file():
            approval = require(approval_path, "tokenizer quality approval")
            require_pool_binding(approval, "tokenizer quality approval")
            if approval.get("quality_report_sha256") != report["canonical_sha256"]:
                raise TurkishCorpusError("tokenizer quality report/approval binding drift")
            if approval.get("training_receipt_sha256") != tokenizer_training_sha:
                raise TurkishCorpusError(
                    "tokenizer quality approval/training receipt binding drift"
                )
        checked["tokenizer_quality_report_sha256"] = report["canonical_sha256"]
    if packing_preflight_dir is not None:
        report = require(
            packing_preflight_dir / "packing_preflight_report.json",
            "packing preflight report",
        )
        approval = require(
            packing_preflight_dir / "packing_preflight_approval.json",
            "packing preflight approval",
        )
        if approval.get("packing_report_sha256") != report["canonical_sha256"]:
            raise TurkishCorpusError("packing report/approval binding drift")
        if pool_manifest is not None and (
            report.get("pool_manifest_sha256")
            != pool_manifest["canonical_sha256"]
            or approval.get("pool_manifest_sha256")
            != pool_manifest["canonical_sha256"]
            or report.get("qa_approval_sha256") != pool_qa_sha
            or approval.get("qa_approval_sha256") != pool_qa_sha
        ):
            raise TurkishCorpusError("packing parent-pool/QA binding drift")
        checked["packing_preflight_report_sha256"] = report["canonical_sha256"]
    return {"production_chain": expected_chain, "checked": checked}


def _object_context(inputs: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    plan = inputs["pack_plan"]
    workers = int(plan["workers_per_node"])
    cpus = int(plan["cpus_per_worker"])
    nodes = int(plan["node_count"])
    if (
        _positive_env(env, "SLURM_NTASKS") != workers
        or _positive_env(env, "SLURM_CPUS_PER_TASK") != cpus
        or _positive_env(env, "SLURM_NNODES") != 1
        or workers * cpus != CPU2DQ_BILLABLE_CPUS
    ):
        raise TurkishCorpusError("object allocation does not match the sealed pack plan")
    node_index = int(env.get("SLURM_ARRAY_TASK_ID", "-1"))
    local_lane = int(env.get("SLURM_PROCID", "-1"))
    local_id = int(env.get("SLURM_LOCALID", "-1"))
    if (
        node_index not in range(nodes)
        or local_lane not in range(workers)
        or local_id != local_lane
        or int(env.get("SLURM_ARRAY_TASK_COUNT", "-1")) != nodes
        or int(env.get("SLURM_ARRAY_TASK_MIN", "-1")) != 0
        or int(env.get("SLURM_ARRAY_TASK_MAX", "-1")) != nodes - 1
        or int(env.get("SLURM_ARRAY_TASK_STEP", "-1")) != 1
    ):
        raise TurkishCorpusError("object Slurm array/lane coordinates drifted")
    array_job_id = env.get("SLURM_ARRAY_JOB_ID", "")
    job_id = env.get("SLURM_JOB_ID", "")
    step_id = env.get("SLURM_STEP_ID", "")
    node_name = env.get("SLURMD_NODENAME", "")
    if not array_job_id.isdigit() or not job_id.isdigit() or not step_id.isdigit() or not node_name:
        raise TurkishCorpusError("object worker lacks exact Slurm identity")
    thread_caps = {name: env.get(name) for name in THREAD_CAPS}
    if thread_caps != {name: "1" for name in THREAD_CAPS}:
        raise TurkishCorpusError("object worker thread caps must all equal one")
    lane_id = node_index * workers + local_lane
    lane = plan["lanes"][lane_id]
    if (
        lane["lane_id"] != lane_id
        or lane["node_index"] != node_index
        or lane["node_local_lane_id"] != local_lane
    ):
        raise TurkishCorpusError("object worker assignment drifted from pack plan")
    return {
        "slurm_job_id": job_id,
        "slurm_array_job_id": array_job_id,
        "slurm_array_task_id": node_index,
        "slurm_allocation_id": f"{array_job_id}_{node_index}",
        "slurm_step_id": step_id,
        "slurm_node": node_name,
        "nodes": 1,
        "tasks": workers,
        "cpus_per_task": cpus,
        "allocated_cpus": CPU2DQ_BILLABLE_CPUS,
        "proc_id": local_lane,
        "local_id": local_id,
        "lane_id": lane_id,
        "thread_caps": thread_caps,
    }


def _object_record(
    receipt: Mapping[str, Any], rank: int, inputs: Mapping[str, Any], disposition: str
) -> dict[str, Any]:
    digest = verify_manifest_hash(receipt)
    if (
        receipt.get("rank") != rank
        or receipt.get("sample_mode") is not False
        or receipt.get("source_plan_sha256") != inputs["source_plan_sha"]
        or receipt.get("calibration_sha256") != inputs["calibration_sha"]
    ):
        raise TurkishCorpusError(f"production object receipt {rank} binding drift")
    return {
        "rank": rank,
        "path": f"objects/{rank:05d}/object_receipt.json",
        "canonical_sha256": digest,
        "disposition": disposition,
    }


def run_object_lane(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    inputs = _load_inputs(*_input_paths(args))
    context = _object_context(inputs, os.environ if env is None else env)
    lane = inputs["pack_plan"]["lanes"][context["lane_id"]]
    destination = args.receipt_dir / f"lane_{context['lane_id']:05d}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite production lane receipt: {destination}")
    raw_scratch = args.scratch_dir.expanduser()
    if raw_scratch.is_symlink():
        raise TurkishCorpusError("production scratch must not be a symlink")
    scratch = raw_scratch.resolve()
    gated_work_dir = Path(str(inputs["storage_gate"]["work_dir"])).resolve()
    if (
        not scratch.is_dir()
        or (scratch != gated_work_dir and gated_work_dir not in scratch.parents)
        or scratch.stat().st_dev
        != int(inputs["storage_gate"]["work_dir_filesystem_device"])
    ):
        raise TurkishCorpusError("production scratch must be an existing BeeGFS directory")
    records = []
    for rank in lane["object_ranks"]:
        object_path = args.run_dir / "objects" / f"{rank:05d}" / "object_receipt.json"
        preexisting = object_path.is_file()
        receipt = process_source_object(
            inputs["policy"],
            inputs["source_plan"],
            inputs["calibration"],
            args.model,
            args.run_dir,
            rank=rank,
            sample_mode=False,
            resource_approval_path=args.resource_approval,
            scratch_dir=scratch / f"lane-{context['lane_id']:05d}",
        )
        records.append(
            _object_record(
                receipt,
                rank,
                inputs,
                "reused_verified" if preexisting else "produced_by_allocation",
            )
        )
    result = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": OBJECT_LANE_KIND,
            "sample_mode": False,
            **_receipt_bindings(inputs),
            "allocation": context,
            "rank_assignment": dict(lane),
            "object_receipts": records,
            "all_assigned_ranks_completed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, result)
    return result


def _validate_lane_receipt(
    receipt: Mapping[str, Any], lane_id: int, inputs: Mapping[str, Any], run_root: Path
) -> tuple[str, list[dict[str, Any]]]:
    digest = verify_manifest_hash(receipt)
    lane = inputs["pack_plan"]["lanes"][lane_id]
    if (
        receipt.get("kind") != OBJECT_LANE_KIND
        or receipt.get("sample_mode") is not False
        or receipt.get("rank_assignment") != lane
        or receipt.get("all_assigned_ranks_completed") is not True
        or any(receipt.get(key) != value for key, value in _receipt_bindings(inputs).items())
    ):
        raise TurkishCorpusError(f"production lane {lane_id} binding drift")
    allocation = receipt.get("allocation")
    if (
        not isinstance(allocation, Mapping)
        or allocation.get("lane_id") != lane_id
        or allocation.get("slurm_array_task_id") != lane["node_index"]
        or allocation.get("proc_id") != lane["node_local_lane_id"]
        or allocation.get("tasks") != inputs["pack_plan"]["workers_per_node"]
        or allocation.get("cpus_per_task") != inputs["pack_plan"]["cpus_per_worker"]
        or allocation.get("allocated_cpus") != CPU2DQ_BILLABLE_CPUS
    ):
        raise TurkishCorpusError(f"production lane {lane_id} allocation drift")
    records = receipt.get("object_receipts")
    if not isinstance(records, list) or [item.get("rank") for item in records] != lane["object_ranks"]:
        raise TurkishCorpusError(f"production lane {lane_id} object inventory drift")
    for record in records:
        rank = int(record["rank"])
        path = run_root / f"objects/{rank:05d}/object_receipt.json"
        if path.is_symlink() or not path.is_file():
            raise TurkishCorpusError(f"production object receipt {rank} is missing")
        actual = load_json_strict(path)
        validated = _object_record(actual, rank, inputs, str(record.get("disposition")))
        if validated != dict(record):
            raise TurkishCorpusError(f"production object receipt {rank} hash drift")
    return digest, [dict(item) for item in records]


def finalize_object_node(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _load_inputs(*_input_paths(args))
    workers = int(inputs["pack_plan"]["workers_per_node"])
    node_index = args.node_index
    if node_index not in range(int(inputs["pack_plan"]["node_count"])):
        raise TurkishCorpusError("production object node index is outside pack plan")
    lane_records = []
    object_records = []
    allocation_ids = set()
    nodes = set()
    for local_lane in range(workers):
        lane_id = node_index * workers + local_lane
        path = args.receipt_dir / f"lane_{lane_id:05d}.json"
        receipt = load_json_strict(path)
        digest, objects = _validate_lane_receipt(receipt, lane_id, inputs, args.run_dir)
        allocation_ids.add(receipt["allocation"]["slurm_allocation_id"])
        nodes.add(receipt["allocation"]["slurm_node"])
        lane_records.append({"lane_id": lane_id, "canonical_sha256": digest})
        object_records.extend({"lane_id": lane_id, **item} for item in objects)
    expected = sorted(
        rank
        for lane in inputs["pack_plan"]["lanes"]
        if lane["node_index"] == node_index
        for rank in lane["object_ranks"]
    )
    if sorted(item["rank"] for item in object_records) != expected or len(allocation_ids) != 1 or len(nodes) != 1:
        raise TurkishCorpusError("production object node did not complete its exact ranks")
    result = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": OBJECT_NODE_KIND,
            "sample_mode": False,
            **_receipt_bindings(inputs),
            "node_index": node_index,
            "slurm_allocation_id": next(iter(allocation_ids)),
            "slurm_node": next(iter(nodes)),
            "lane_receipts": lane_records,
            "object_receipts": sorted(object_records, key=lambda item: item["rank"]),
            "all_node_ranks_completed": True,
            "canonical_sha256": None,
        }
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite production node receipt: {args.output}")
    write_json_atomic(args.output, result)
    return result


def _all_object_nodes(
    inputs: Mapping[str, Any], node_receipt_dir: Path, run_root: Path
) -> tuple[list[str], list[str]]:
    node_hashes = []
    objects: list[dict[str, Any]] = []
    allocation_ids: set[str] = set()
    array_job_ids: set[str] = set()
    for node_index in range(int(inputs["pack_plan"]["node_count"])):
        path = node_receipt_dir / f"node_{node_index:05d}.json"
        receipt = load_json_strict(path)
        digest = verify_manifest_hash(receipt)
        if (
            receipt.get("kind") != OBJECT_NODE_KIND
            or receipt.get("node_index") != node_index
            or receipt.get("all_node_ranks_completed") is not True
            or any(receipt.get(key) != value for key, value in _receipt_bindings(inputs).items())
        ):
            raise TurkishCorpusError(f"production object node {node_index} binding drift")
        allocation_id = str(receipt.get("slurm_allocation_id", ""))
        match = re.fullmatch(r"([0-9]+)_([0-9]+)", allocation_id)
        if match is None or int(match.group(2)) != node_index:
            raise TurkishCorpusError(
                f"production object node {node_index} allocation identity drift"
            )
        allocation_ids.add(allocation_id)
        array_job_ids.add(match.group(1))
        node_hashes.append(digest)
        objects.extend(receipt["object_receipts"])
    if (
        len(allocation_ids) != int(inputs["pack_plan"]["node_count"])
        or len(array_job_ids) != 1
    ):
        raise TurkishCorpusError(
            "production object nodes must be unique allocations from one exact array"
        )
    if [item.get("rank") for item in sorted(objects, key=lambda item: item["rank"])] != list(
        range(len(inputs["source_plan"]["objects"]))
    ):
        raise TurkishCorpusError("production node receipts do not cover every source rank once")
    object_hashes = []
    for item in sorted(objects, key=lambda value: value["rank"]):
        rank = int(item["rank"])
        path = run_root / f"objects/{rank:05d}/object_receipt.json"
        receipt = load_json_strict(path)
        record = _object_record(receipt, rank, inputs, str(item.get("disposition")))
        if record != {key: item[key] for key in record}:
            raise TurkishCorpusError(f"production object {rank} inventory drift")
        object_hashes.append(record["canonical_sha256"])
    return node_hashes, object_hashes


def _bucket_context(inputs: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    contract = inputs["pack_plan"]["minhash_bucket_execution_contract"]
    tasks = int(contract["bucket_tasks"])
    cpus = int(contract["cpus_per_task"])
    if (
        _positive_env(env, "SLURM_NTASKS") != tasks
        or _positive_env(env, "SLURM_CPUS_PER_TASK") != cpus
        or _positive_env(env, "SLURM_NNODES") != 1
    ):
        raise TurkishCorpusError("bucket allocation does not match the sealed plan")
    rank = int(env.get("SLURM_PROCID", "-1"))
    if rank not in range(tasks) or int(env.get("SLURM_LOCALID", "-1")) != rank:
        raise TurkishCorpusError("bucket task coordinates drifted")
    job_id = env.get("SLURM_JOB_ID", "")
    step_id = env.get("SLURM_STEP_ID", "")
    node = env.get("SLURMD_NODENAME", "")
    if not job_id.isdigit() or not step_id.isdigit() or not node:
        raise TurkishCorpusError("bucket task lacks exact Slurm identity")
    caps = {name: env.get(name) for name in THREAD_CAPS}
    if caps != {name: "1" for name in THREAD_CAPS}:
        raise TurkishCorpusError("bucket worker thread caps must all equal one")
    return {
        "slurm_job_id": job_id,
        "slurm_step_id": step_id,
        "slurm_node": node,
        "nodes": 1,
        "tasks": tasks,
        "cpus_per_task": cpus,
        "allocated_task_cpus": tasks * cpus,
        "billable_cpus": CPU2DQ_BILLABLE_CPUS,
        "proc_id": rank,
        "local_id": rank,
        "thread_caps": caps,
    }


def run_bucket_task(args: argparse.Namespace, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    inputs = _load_inputs(*_input_paths(args))
    node_hashes, object_hashes = _all_object_nodes(
        inputs, args.object_node_receipt_dir, args.run_dir
    )
    context = _bucket_context(inputs, os.environ if env is None else env)
    rank = int(context["proc_id"])
    destination = args.receipt_dir / f"bucket_{rank:05d}.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite production bucket receipt: {destination}")
    backend_path = args.run_dir / "bucket_receipts" / f"{rank:05d}.json"
    preexisting = backend_path.is_file()
    receipt = run_datatrove_bucket(
        inputs["policy"],
        inputs["source_plan"],
        inputs["calibration"],
        args.run_dir,
        rank=rank,
        sample_mode=False,
        resource_approval_path=args.resource_approval,
    )
    backend_sha = verify_manifest_hash(receipt)
    if (
        receipt.get("sample_mode") is not False
        or receipt.get("rank") != rank
        or receipt.get("source_plan_sha256") != inputs["source_plan_sha"]
        or receipt.get("calibration_sha256") != inputs["calibration_sha"]
        or receipt.get("object_receipt_sha256") != object_hashes
    ):
        raise TurkishCorpusError(f"production backend bucket {rank} binding drift")
    result = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": BUCKET_TASK_KIND,
            "sample_mode": False,
            **_receipt_bindings(inputs),
            "object_node_receipt_sha256": node_hashes,
            "allocation": context,
            "bucket_rank": rank,
            "backend_bucket_receipt": {
                "path": f"bucket_receipts/{rank:05d}.json",
                "canonical_sha256": backend_sha,
                "disposition": "reused_verified" if preexisting else "produced_by_allocation",
            },
            "bucket_completed": True,
            "canonical_sha256": None,
        }
    )
    write_json_atomic(destination, result)
    return result


def finalize_buckets(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _load_inputs(*_input_paths(args))
    node_hashes, _object_hashes = _all_object_nodes(
        inputs, args.object_node_receipt_dir, args.run_dir
    )
    tasks = []
    backend_records = []
    steps = set()
    nodes = set()
    jobs = set()
    for rank in range(14):
        receipt = load_json_strict(args.receipt_dir / f"bucket_{rank:05d}.json")
        digest = verify_manifest_hash(receipt)
        if (
            receipt.get("kind") != BUCKET_TASK_KIND
            or receipt.get("bucket_rank") != rank
            or receipt.get("bucket_completed") is not True
            or receipt.get("object_node_receipt_sha256") != node_hashes
            or any(receipt.get(key) != value for key, value in _receipt_bindings(inputs).items())
        ):
            raise TurkishCorpusError(f"production bucket task {rank} binding drift")
        allocation = receipt["allocation"]
        steps.add(allocation["slurm_step_id"])
        nodes.add(allocation["slurm_node"])
        jobs.add(allocation["slurm_job_id"])
        backend = receipt["backend_bucket_receipt"]
        actual = load_json_strict(args.run_dir / backend["path"])
        if verify_manifest_hash(actual) != backend["canonical_sha256"]:
            raise TurkishCorpusError(f"production backend bucket {rank} hash drift")
        tasks.append({"bucket_rank": rank, "canonical_sha256": digest})
        backend_records.append(
            {"bucket_rank": rank, "path": backend["path"], "canonical_sha256": backend["canonical_sha256"]}
        )
    if len(steps) != 1 or len(nodes) != 1 or len(jobs) != 1:
        raise TurkishCorpusError("production bucket tasks did not share one allocation")
    result = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": BUCKET_LAUNCH_KIND,
            "sample_mode": False,
            **_receipt_bindings(inputs),
            "object_node_receipt_sha256": node_hashes,
            "allocation": {
                "slurm_job_id": next(iter(jobs)),
                "slurm_step_id": next(iter(steps)),
                "slurm_node": next(iter(nodes)),
                "nodes": 1,
                "tasks": 14,
                "cpus_per_task": 8,
                "allocated_task_cpus": 112,
                "billable_cpus": CPU2DQ_BILLABLE_CPUS,
            },
            "task_receipts": tasks,
            "backend_bucket_receipts": backend_records,
            "all_buckets_completed": True,
            "canonical_sha256": None,
        }
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite bucket launch receipt: {args.output}")
    write_json_atomic(args.output, result)
    return result


def validate_cluster_inputs(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _load_inputs(*_input_paths(args))
    node_hashes, _object_hashes = _all_object_nodes(
        inputs, args.object_node_receipt_dir, args.run_dir
    )
    buckets = load_json_strict(args.bucket_launch_receipt)
    bucket_sha = verify_manifest_hash(buckets)
    if (
        buckets.get("kind") != BUCKET_LAUNCH_KIND
        or buckets.get("all_buckets_completed") is not True
        or buckets.get("object_node_receipt_sha256") != node_hashes
        or any(buckets.get(key) != value for key, value in _receipt_bindings(inputs).items())
    ):
        raise TurkishCorpusError("production bucket launch binding drift")
    result = seal_manifest(
        {
            "schema_version": "1.0",
            "kind": CLUSTER_INPUT_KIND,
            **_receipt_bindings(inputs),
            "object_node_receipt_sha256": node_hashes,
            "bucket_launch_receipt_sha256": bucket_sha,
            "backend_bucket_receipt_sha256": [
                item["canonical_sha256"] for item in buckets["backend_bucket_receipts"]
            ],
            "all_cluster_inputs_verified": True,
            "canonical_sha256": None,
        }
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cluster input receipt: {args.output}")
    write_json_atomic(args.output, result)
    return result


def seal_cluster_launch(
    args: argparse.Namespace, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    inputs = _load_inputs(*_input_paths(args))
    cluster_input = load_json_strict(args.cluster_input_receipt)
    input_sha = verify_manifest_hash(cluster_input)
    if (
        cluster_input.get("kind") != CLUSTER_INPUT_KIND
        or cluster_input.get("all_cluster_inputs_verified") is not True
        or any(cluster_input.get(key) != value for key, value in _receipt_bindings(inputs).items())
    ):
        raise TurkishCorpusError("production cluster input receipt drift")
    cluster = load_json_strict(args.run_dir / "cluster_receipt.json")
    cluster_sha = verify_manifest_hash(cluster)
    if (
        cluster.get("sample_mode") is not False
        or cluster.get("source_plan_sha256") != inputs["source_plan_sha"]
        or cluster.get("calibration_sha256") != inputs["calibration_sha"]
        or cluster.get("bucket_receipt_sha256")
        != cluster_input["backend_bucket_receipt_sha256"]
    ):
        raise TurkishCorpusError("production cluster receipt binding drift")
    execution_env = os.environ if env is None else env
    job_id = execution_env.get("SLURM_JOB_ID", "")
    partition = execution_env.get("SLURM_JOB_PARTITION", "")
    node_list = execution_env.get("SLURM_NODELIST", "")
    if (
        not job_id.isdigit()
        or partition != "cpu2dq"
        or not node_list
        or _positive_env(execution_env, "SLURM_NNODES") != 1
        or _positive_env(execution_env, "SLURM_NTASKS") != 1
        or _positive_env(execution_env, "SLURM_CPUS_PER_TASK") != 16
    ):
        raise TurkishCorpusError("production cluster Slurm allocation drifted")
    result = seal_manifest(
        {
            "schema_version": "2.0",
            "kind": CLUSTER_LAUNCH_KIND,
            **_receipt_bindings(inputs),
            "cluster_input_receipt_sha256": input_sha,
            "cluster_receipt_sha256": cluster_sha,
            "allocation": {
                "slurm_job_id": job_id,
                "partition": partition,
                "node_list": node_list,
                "nodes": 1,
                "tasks": 1,
                "cpus_per_task": 16,
                "billable_cpus": CPU2DQ_BILLABLE_CPUS,
                "memory_bytes": 192 * 1024**3,
                "maximum_wall_seconds": 172_800,
            },
            "cluster_completed": True,
            "canonical_sha256": None,
        }
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cluster launch receipt: {args.output}")
    write_json_atomic(args.output, result)
    return result


def _receipt_bindings(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family_id": inputs["recipe"]["family_id"],
        "recipe_sha256": inputs["recipe_sha"],
        "policy_sha256": inputs["policy_sha"],
        "source_plan_sha256": inputs["source_plan_sha"],
        "calibration_sha256": inputs["calibration_sha"],
        "production_pack_plan_sha256": inputs["pack_plan_sha"],
        "resource_approval_sha256": inputs["resource_approval_sha"],
        "mixture_quality_approval_sha256": inputs["mixture_approval_sha"],
        "data_prep_storage_gate_sha256": inputs["storage_gate_sha"],
        "sample_cluster_receipt_sha256": inputs["sample_cluster_receipt_sha"],
    }


def _input_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    return (
        args.recipe,
        args.policy,
        args.source_plan,
        args.calibration,
        args.pack_plan,
        args.resource_approval,
        args.mixture_quality_approval,
        args.storage_gate,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--pack-plan", type=Path, required=True)
    parser.add_argument("--resource-approval", type=Path, required=True)
    parser.add_argument("--mixture-quality-approval", type=Path, required=True)
    parser.add_argument("--storage-gate", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("validate-gate")
    _common(gate)
    gate.add_argument("--write-dir", type=Path, action="append", default=[])
    gate.add_argument("--cluster-launch-receipt", type=Path)
    gate.add_argument("--pool-dir", type=Path)
    gate.add_argument("--tokenizer-sample-dir", type=Path)
    gate.add_argument("--tokenizer-dir", type=Path)
    gate.add_argument("--tokenizer-quality-dir", type=Path)
    gate.add_argument("--packing-preflight-dir", type=Path)
    lane = sub.add_parser("run-object-lane")
    _common(lane)
    lane.add_argument("--model", type=Path, required=True)
    lane.add_argument("--run-dir", type=Path, required=True)
    lane.add_argument("--receipt-dir", type=Path, required=True)
    lane.add_argument("--scratch-dir", type=Path, required=True)
    node = sub.add_parser("finalize-object-node")
    _common(node)
    node.add_argument("--run-dir", type=Path, required=True)
    node.add_argument("--receipt-dir", type=Path, required=True)
    node.add_argument("--node-index", type=int, required=True)
    node.add_argument("--output", type=Path, required=True)
    bucket = sub.add_parser("run-bucket")
    _common(bucket)
    bucket.add_argument("--run-dir", type=Path, required=True)
    bucket.add_argument("--object-node-receipt-dir", type=Path, required=True)
    bucket.add_argument("--receipt-dir", type=Path, required=True)
    bucket_final = sub.add_parser("finalize-buckets")
    _common(bucket_final)
    bucket_final.add_argument("--run-dir", type=Path, required=True)
    bucket_final.add_argument("--object-node-receipt-dir", type=Path, required=True)
    bucket_final.add_argument("--receipt-dir", type=Path, required=True)
    bucket_final.add_argument("--output", type=Path, required=True)
    cluster = sub.add_parser("validate-cluster-inputs")
    _common(cluster)
    cluster.add_argument("--run-dir", type=Path, required=True)
    cluster.add_argument("--object-node-receipt-dir", type=Path, required=True)
    cluster.add_argument("--bucket-launch-receipt", type=Path, required=True)
    cluster.add_argument("--output", type=Path, required=True)
    cluster_final = sub.add_parser("seal-cluster-launch")
    _common(cluster_final)
    cluster_final.add_argument("--run-dir", type=Path, required=True)
    cluster_final.add_argument("--cluster-input-receipt", type=Path, required=True)
    cluster_final.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-gate":
            inputs = _load_inputs(*_input_paths(args))
            artifact_paths = (
                args.pool_dir,
                args.tokenizer_sample_dir,
                args.tokenizer_dir,
                args.tokenizer_quality_dir,
                args.packing_preflight_dir,
            )
            if any(path is not None for path in artifact_paths) and args.cluster_launch_receipt is None:
                raise TurkishCorpusError(
                    "downstream artifact validation requires --cluster-launch-receipt"
                )
            result = {
                "valid": True,
                **_receipt_bindings(inputs),
                "live": validate_live_launch_gate(inputs),
                "write_dirs": [
                    validate_gated_write_dir(inputs, path) for path in args.write_dir
                ],
                "downstream_lineage": (
                    validate_downstream_lineage(
                        inputs,
                        args.cluster_launch_receipt,
                        pool_dir=args.pool_dir,
                        tokenizer_sample_dir=args.tokenizer_sample_dir,
                        tokenizer_dir=args.tokenizer_dir,
                        tokenizer_quality_dir=args.tokenizer_quality_dir,
                        packing_preflight_dir=args.packing_preflight_dir,
                    )
                    if args.cluster_launch_receipt is not None
                    else None
                ),
            }
        elif args.command == "run-object-lane":
            result = run_object_lane(args)
        elif args.command == "finalize-object-node":
            result = finalize_object_node(args)
        elif args.command == "run-bucket":
            result = run_bucket_task(args)
        elif args.command == "finalize-buckets":
            result = finalize_buckets(args)
        elif args.command == "validate-cluster-inputs":
            result = validate_cluster_inputs(args)
        else:
            result = seal_cluster_launch(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
