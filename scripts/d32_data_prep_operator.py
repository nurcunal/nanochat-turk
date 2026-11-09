"""Read-only operator helpers for the active Turkish d32 data workflow.

The commands here deliberately call the same parsers, receipt validators, lane
balancer, and projection arithmetic as ``d32_family_workflow``.  They do not
submit Slurm jobs and never create production pack plans or storage gates.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.experiment_manifest import write_json_atomic
from scripts import d32_family_workflow as workflow


def _fail(message: str) -> None:
    raise workflow.FamilyWorkflowError(message)


def _temporary_pack_plan(
    *,
    recipe_path: Path,
    policy_path: Path,
    source_plan_path: Path,
    calibration_path: Path,
    nodes: int,
) -> dict[str, Any]:
    """Build and validate an ephemeral plan through the authoritative command."""

    with tempfile.TemporaryDirectory(prefix="d32-node-selection-") as temporary:
        output = Path(temporary) / "production_source_pack_plan.json"
        with contextlib.redirect_stdout(io.StringIO()):
            workflow.command_seal_data_prep_pack_plan(
                SimpleNamespace(
                    recipe=recipe_path,
                    policy=policy_path,
                    source_plan=source_plan_path,
                    calibration=calibration_path,
                    nodes=nodes,
                    output=output,
                )
            )
        return workflow._load_object(output, "ephemeral production pack plan")


def _select_first_passing(evaluations: Sequence[Mapping[str, Any]]) -> int:
    passing = sorted(
        int(item["node_count"])
        for item in evaluations
        if item.get("passes_gate_limits") is True
    )
    if not passing:
        _fail("no production data node count passes the existing gate arithmetic")
    return passing[0]


def _node_evaluations(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    (
        recipe,
        recipe_sha,
        _policy,
        policy_sha,
        source_plan,
        source_plan_sha,
        _calibration,
        calibration_sha,
    ) = workflow._load_data_prep_inputs(
        recipe_path=args.recipe,
        policy_path=args.policy,
        source_plan_path=args.source_plan,
        calibration_path=args.calibration,
    )
    try:
        from nanochat.turkish_backend import (
            select_resource_sample_ranks,
            validate_resource_projection,
        )
    except ImportError as exc:  # pragma: no cover - exercised on UHeM setup failure
        raise workflow.FamilyWorkflowError(
            "Turkish data environment is unavailable"
        ) from exc

    backend_report = workflow._load_object(
        args.backend_resource_report, "backend resource report"
    )
    backend_report_sha = validate_resource_projection(
        backend_report, plan=source_plan
    )
    expected_bindings = {
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
    }
    for field, expected in expected_bindings.items():
        if backend_report.get(field) != expected:
            _fail(f"backend resource report {field} binding mismatch")
    report_projection = workflow._mapping(
        backend_report.get("projection"), "backend resource projection"
    )
    if (
        report_projection.get("safety_factor") != 1.0
        or backend_report.get("automated_gate_passed") is not True
    ):
        _fail("node selection requires the passed pre-safety backend report")

    expected_ranks = select_resource_sample_ranks(source_plan)
    sample_run_source = args.sample_run_dir.expanduser()
    if sample_run_source.is_symlink():
        _fail("sample run directory must exist and not be a symlink")
    sample_run_dir = sample_run_source.resolve()
    if not sample_run_dir.is_dir():
        _fail("sample run directory must exist and not be a symlink")
    objects, buckets, cluster = workflow._load_sample_receipt_inventory(
        sample_run_dir,
        object_ranks=expected_ranks,
        backend_report=backend_report,
    )
    sample_documents = sum(
        workflow._positive_int(
            workflow._mapping(receipt.get("candidate_file"), "candidate file").get(
                "rows"
            ),
            "sample candidate rows",
        )
        for receipt in objects.values()
    )
    estimated_total_documents = math.ceil(
        workflow._positive_number(
            report_projection.get("candidate_documents"),
            "projected candidate documents",
        )
    )
    if estimated_total_documents < sample_documents:
        _fail("backend report projects fewer documents than its sample")

    writer_probe, writer_probe_sha = workflow._verify_sealed(
        args.writer_probe, "post-cluster writer probe"
    )
    _writer_components, writer_projected_cpu = workflow._validate_writer_probe(
        writer_probe,
        recipe=recipe,
        recipe_sha=recipe_sha,
        policy_sha=policy_sha,
        source_plan_sha=source_plan_sha,
        calibration_sha=calibration_sha,
        backend_report_sha=backend_report_sha,
        cluster_sha=cluster["canonical_sha256"],
        sample_documents=sample_documents,
        estimated_total_documents=estimated_total_documents,
    )

    source_objects = workflow._sequence(source_plan.get("objects"), "source-plan objects")
    maximum_valid_nodes = len(source_objects) // workflow.PRODUCTION_WORKERS_PER_NODE
    if maximum_valid_nodes < 1:
        _fail("source plan has fewer objects than one production node has worker lanes")
    safety = float(
        recipe["storage"]["data_preparation_peak_gate"][
            "extrapolation_safety_factor"
        ]
    )
    memory_limit = 192 * 1024**3
    evaluations: list[dict[str, Any]] = []
    for nodes in range(1, maximum_valid_nodes + 1):
        pack_plan = _temporary_pack_plan(
            recipe_path=args.recipe,
            policy_path=args.policy,
            source_plan_path=args.source_plan,
            calibration_path=args.calibration,
            nodes=nodes,
        )
        backend_cpu, details = workflow._packed_production_backend_cpu_projection(
            source_plan=source_plan,
            pack_plan=pack_plan,
            backend_report=backend_report,
            sample_bucket_receipts=buckets,
        )
        node_walls = workflow._mapping(
            details.get("projected_node_wall_seconds_before_safety"),
            "projected production node walls",
        )
        object_wall = max(float(value) for value in node_walls.values())
        bucket_wall = float(
            details["projected_packed_bucket_node_wall_seconds_before_safety"]
        )
        cluster_wall = (
            float(details["projected_priority_cluster_cpu_saat_before_safety"])
            * 3600.0
            / workflow.CPU2DQ_BILLABLE_CPUS
        )
        pool_wall = (
            writer_projected_cpu * 3600.0 / workflow.CPU2DQ_BILLABLE_CPUS
        )
        sample_rss = int(details["sample_priority_cluster_peak_rss_bytes"])
        projected_rss = float(
            details[
                "projected_priority_cluster_peak_rss_bytes_before_safety"
            ]
        )
        failures = []
        if object_wall * safety > 172_800:
            failures.append("object_node_wall_over_48h")
        if bucket_wall * safety > 86_400:
            failures.append("bucket_node_wall_over_24h")
        if cluster_wall * safety > 172_800:
            failures.append("cluster_wall_over_48h")
        if pool_wall * safety > 172_800:
            failures.append("pool_wall_over_48h")
        if sample_rss >= memory_limit:
            failures.append("sample_cluster_rss_not_below_192gib")
        if projected_rss * safety >= memory_limit:
            failures.append("projected_cluster_rss_not_below_192gib")
        projected_future_cpu = (
            backend_cpu
            + writer_projected_cpu
            + sum(workflow.DATA_PREP_FIXED_CPU2DQ_CEILINGS.values())
        )
        evaluations.append(
            {
                "node_count": nodes,
                "passes_gate_limits": not failures,
                "failures": failures,
                "safety_factor": safety,
                "object_node_max_wall_seconds_before_safety": object_wall,
                "bucket_node_wall_seconds_before_safety": bucket_wall,
                "cluster_wall_seconds_before_safety": cluster_wall,
                "pool_wall_seconds_before_safety": pool_wall,
                "sample_cluster_peak_rss_bytes": sample_rss,
                "projected_cluster_peak_rss_bytes_before_safety": projected_rss,
                "projected_future_cpu_saat_before_safety": projected_future_cpu,
                "projected_future_cpu_saat_with_safety": math.ceil(
                    projected_future_cpu * safety
                ),
            }
        )

    bindings = {
        "family_id": recipe["family_id"],
        "recipe_sha256": recipe_sha,
        "policy_sha256": policy_sha,
        "source_plan_sha256": source_plan_sha,
        "calibration_sha256": calibration_sha,
        "backend_resource_report_sha256": backend_report_sha,
        "writer_probe_sha256": writer_probe_sha,
        "sample_cluster_receipt_sha256": cluster["canonical_sha256"],
    }
    return bindings, evaluations


def command_select_production_nodes(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        _fail(f"refusing to overwrite node-selection receipt: {args.output}")
    if not args.output.parent.is_dir() or args.output.parent.is_symlink():
        _fail("node-selection output parent must exist and not be a symlink")
    bindings, evaluations = _node_evaluations(args)
    selected = _select_first_passing(evaluations)
    receipt = workflow.seal_manifest(
        {
            "schema_version": "1.0",
            "kind": "d32_data_prep_production_node_selection",
            **bindings,
            "selection_policy": (
                "smallest_positive_node_count_passing_exact_existing_"
                "storage_gate_walltime_and_rss_arithmetic"
            ),
            "selected_nodes": selected,
            "maximum_evaluated_nodes": evaluations[-1]["node_count"],
            "evaluations": evaluations,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_sha256": None,
        }
    )
    write_json_atomic(args.output, receipt)
    print(selected)


def command_validate_production_nodes(args: argparse.Namespace) -> None:
    receipt, _receipt_sha = workflow._verify_sealed(
        args.node_selection, "production data node-selection receipt"
    )
    bindings, evaluations = _node_evaluations(args)
    selected = _select_first_passing(evaluations)
    expected_policy = (
        "smallest_positive_node_count_passing_exact_existing_"
        "storage_gate_walltime_and_rss_arithmetic"
    )
    expected_keys = {
        "schema_version",
        "kind",
        *bindings,
        "selection_policy",
        "selected_nodes",
        "maximum_evaluated_nodes",
        "evaluations",
        "created_at_utc",
        "canonical_sha256",
    }
    created_at = receipt.get("created_at_utc")
    try:
        created_at_value = datetime.fromisoformat(str(created_at))
    except ValueError:
        created_at_value = None
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != "1.0"
        or receipt.get("kind") != "d32_data_prep_production_node_selection"
        or receipt.get("selection_policy") != expected_policy
        or receipt.get("selected_nodes") != selected
        or receipt.get("maximum_evaluated_nodes") != evaluations[-1]["node_count"]
        or receipt.get("evaluations") != evaluations
        or any(receipt.get(field) != expected for field, expected in bindings.items())
        or created_at_value is None
        or created_at_value.utcoffset() is None
        or created_at_value.utcoffset().total_seconds() != 0
    ):
        _fail("production data node-selection receipt drifted from current evidence")
    print(selected)


def command_live_beegfs_headroom(args: argparse.Namespace) -> None:
    recipe, _recipe_sha = workflow.load_recipe(args.recipe)
    work_dir_source = args.work_dir.expanduser()
    if work_dir_source.is_symlink():
        _fail("BeeGFS work directory must exist and not be a symlink")
    work_dir = work_dir_source.resolve()
    if not work_dir.is_dir():
        _fail("BeeGFS work directory must exist and not be a symlink")
    policy = recipe["storage"]["uhem_live_quota"]
    effective_free, _audit = workflow._live_beegfs_storage(
        workflow.REPO_ROOT,
        uid=int(policy["uid"]),
        storage_pool_id=int(policy["storage_pool_id"]),
        path=work_dir,
    )
    if effective_free <= 0:
        _fail("live effective BeeGFS headroom is zero")
    print(effective_free)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select-production-nodes",
        help="select the smallest node count that passes the existing gate arithmetic",
    )
    select.add_argument("--recipe", type=Path, default=workflow.DEFAULT_RECIPE)
    select.add_argument("--policy", type=Path, default=workflow.DEFAULT_POLICY)
    select.add_argument("--source-plan", type=Path, required=True)
    select.add_argument("--calibration", type=Path, required=True)
    select.add_argument("--sample-run-dir", type=Path, required=True)
    select.add_argument("--backend-resource-report", type=Path, required=True)
    select.add_argument("--writer-probe", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(func=command_select_production_nodes)

    validate = subparsers.add_parser(
        "validate-production-nodes",
        help="recompute and validate a sealed production-node selection",
    )
    validate.add_argument("--recipe", type=Path, default=workflow.DEFAULT_RECIPE)
    validate.add_argument("--policy", type=Path, default=workflow.DEFAULT_POLICY)
    validate.add_argument("--source-plan", type=Path, required=True)
    validate.add_argument("--calibration", type=Path, required=True)
    validate.add_argument("--sample-run-dir", type=Path, required=True)
    validate.add_argument("--backend-resource-report", type=Path, required=True)
    validate.add_argument("--writer-probe", type=Path, required=True)
    validate.add_argument("--node-selection", type=Path, required=True)
    validate.set_defaults(func=command_validate_production_nodes)

    quota = subparsers.add_parser(
        "live-beegfs-headroom",
        help="print min(physical free bytes, finite BeeGFS user-quota remaining)",
    )
    quota.add_argument("--recipe", type=Path, default=workflow.DEFAULT_RECIPE)
    quota.add_argument("--work-dir", type=Path, required=True)
    quota.set_defaults(func=command_live_beegfs_headroom)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, workflow.FamilyWorkflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
