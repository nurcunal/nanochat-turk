"""Operate the resumable Turkish-only source/LID/DataTrove CPU backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nanochat.experiment_manifest import load_json_strict
from nanochat.turkish_backend import (
    build_resource_projection,
    fetch_glotlid_model,
    prepare_macocu_genre,
    process_source_object,
    resolve_source_plan,
    run_backend_calibration,
    run_datatrove_bucket,
    run_priority_cluster_merge,
    seal_backend_receipt_from_cluster,
    seal_resource_approval,
    seal_source_receipt_from_objects,
    select_resource_sample_ranks,
)
from nanochat.turkish_corpus import load_corpus_policy


DEFAULT_POLICY = Path("configs/pretrain/tr_d32_turkish_general_v2.json")
DEFAULT_CALIBRATION_FIXTURE = Path("configs/pretrain/glotlid_calibration_tr_v1.jsonl")


def _rank(value: int | None) -> int:
    if value is not None:
        return value
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError("--rank or SLURM_ARRAY_TASK_ID is required")
    return int(raw)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)


def _run_inputs(parser: argparse.ArgumentParser) -> None:
    _common(parser)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--resource-approval", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    resolve = sub.add_parser("resolve", help="resolve all immutable source objects")
    _common(resolve)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.add_argument("--macocu-manifest", type=Path)

    macocu = sub.add_parser(
        "prepare-macocu", help="verify the official gzip and create sealed zstd shards"
    )
    _common(macocu)
    macocu.add_argument("--output-dir", type=Path, required=True)
    macocu.add_argument(
        "--target-uncompressed-bytes", type=int, default=512 * 1024 * 1024
    )

    model = sub.add_parser("fetch-glotlid", help="fetch and verify the pinned model")
    _common(model)
    model.add_argument("--output-dir", type=Path, required=True)

    calibrate = sub.add_parser("calibrate", help="run mandatory LID/LSH/tokenizer probes")
    _common(calibrate)
    calibrate.add_argument("--model", type=Path, required=True)
    calibrate.add_argument("--fixture", type=Path, default=DEFAULT_CALIBRATION_FIXTURE)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--lsh-trials", type=int, default=1024)

    ranks = sub.add_parser("sample-ranks", help="print deterministic sample array ranks")
    _common(ranks)
    ranks.add_argument("--source-plan", type=Path, required=True)

    process = sub.add_parser("process-object", help="run one acquisition/LID/signature rank")
    _run_inputs(process)
    process.add_argument("--model", type=Path, required=True)
    process.add_argument("--rank", type=int)
    process.add_argument("--scratch-dir", type=Path)

    bucket = sub.add_parser("bucket", help="run one of the fourteen DataTrove LSH buckets")
    _run_inputs(bucket)
    bucket.add_argument("--rank", type=int)

    cluster = sub.add_parser("cluster", help="priority cluster and apply quality/PII stages")
    _run_inputs(cluster)

    resources = sub.add_parser("resource-report", help="project CPU/storage from sample run")
    _common(resources)
    resources.add_argument("--source-plan", type=Path, required=True)
    resources.add_argument("--calibration", type=Path, required=True)
    resources.add_argument("--sample-run-dir", type=Path, required=True)
    resources.add_argument("--quota-headroom-bytes", type=int, required=True)
    resources.add_argument("--billable-cpus-per-job", type=int, required=True)
    resources.add_argument("--safety-factor", type=float, default=1.5)
    resources.add_argument("--output", type=Path, required=True)

    approve = sub.add_parser("approve-resources", help="seal manual resource decision")
    approve.add_argument("--report", type=Path, required=True)
    approve.add_argument("--mixture-quality-approval", type=Path, required=True)
    approve.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    approve.add_argument("--source-plan", type=Path, required=True)
    approve.add_argument("--calibration", type=Path, required=True)
    approve.add_argument("--output", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--reviewed-at-utc", required=True)
    approve.add_argument("--decision", choices=("accepted", "rejected"), required=True)
    approve.add_argument("--notes", default="")

    source = sub.add_parser("seal-source", help="seal full source receipt")
    _run_inputs(source)
    source.add_argument("--cluster-launch-receipt", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)

    backend = sub.add_parser("seal-backend", help="seal production backend receipt")
    _run_inputs(backend)
    backend.add_argument("--cluster-launch-receipt", type=Path, required=True)
    backend.add_argument("--source-receipt", type=Path, required=True)
    backend.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "approve-resources":
            result = seal_resource_approval(
                args.report,
                args.mixture_quality_approval,
                args.output,
                policy_path=args.policy,
                source_plan_path=args.source_plan,
                calibration_path=args.calibration,
                reviewer=args.reviewer,
                reviewed_at_utc=args.reviewed_at_utc,
                decision=args.decision,
                notes=args.notes,
            )
        else:
            policy = load_corpus_policy(args.policy)
            if args.command == "prepare-macocu":
                result = prepare_macocu_genre(
                    policy,
                    args.output_dir,
                    target_uncompressed_bytes=args.target_uncompressed_bytes,
                )
            elif args.command == "resolve":
                result = resolve_source_plan(
                    policy,
                    args.output,
                    macocu_manifest_path=args.macocu_manifest,
                )
            elif args.command == "fetch-glotlid":
                result = fetch_glotlid_model(policy, args.output_dir)
            elif args.command == "calibrate":
                result = run_backend_calibration(
                    policy,
                    args.model,
                    args.fixture,
                    args.output,
                    lsh_trials=args.lsh_trials,
                )
            else:
                plan = load_json_strict(args.source_plan)
                if args.command == "sample-ranks":
                    sample = select_resource_sample_ranks(plan)
                    result = {
                        "ranks": sample,
                        "slurm_array": ",".join(str(rank) for rank in sample),
                    }
                else:
                    calibration = load_json_strict(args.calibration)
                    if args.command == "process-object":
                        result = process_source_object(
                            policy,
                            plan,
                            calibration,
                            args.model,
                            args.run_dir,
                            rank=_rank(args.rank),
                            sample_mode=args.sample,
                            resource_approval_path=args.resource_approval,
                            scratch_dir=args.scratch_dir,
                        )
                    elif args.command == "bucket":
                        result = run_datatrove_bucket(
                            policy,
                            plan,
                            calibration,
                            args.run_dir,
                            rank=_rank(args.rank),
                            sample_mode=args.sample,
                            resource_approval_path=args.resource_approval,
                        )
                    elif args.command == "cluster":
                        result = run_priority_cluster_merge(
                            policy,
                            plan,
                            calibration,
                            args.run_dir,
                            sample_mode=args.sample,
                            resource_approval_path=args.resource_approval,
                        )
                    elif args.command == "resource-report":
                        result = build_resource_projection(
                            policy,
                            plan,
                            calibration,
                            args.sample_run_dir,
                            args.output,
                            quota_headroom_bytes=args.quota_headroom_bytes,
                            billable_cpus_per_job=args.billable_cpus_per_job,
                            safety_factor=args.safety_factor,
                        )
                    elif args.command == "seal-source":
                        if args.sample:
                            raise ValueError("seal-source is production-only")
                        result = seal_source_receipt_from_objects(
                            policy,
                            plan,
                            calibration,
                            args.run_dir,
                            args.cluster_launch_receipt,
                            args.output,
                        )
                    else:
                        if args.sample:
                            raise ValueError("seal-backend is production-only")
                        source_receipt = load_json_strict(args.source_receipt)
                        result = seal_backend_receipt_from_cluster(
                            policy,
                            plan,
                            source_receipt,
                            calibration,
                            args.run_dir,
                            args.cluster_launch_receipt,
                            args.output,
                        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
