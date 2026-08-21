"""Build sealed Turkish-only d32 corpus artifacts without submitting jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nanochat.experiment_manifest import load_json_strict
from nanochat.turkish_corpus import (
    build_packing_preflight_report,
    cleanup_verified_pool,
    load_corpus_policy,
    materialize_filtered_pool,
    materialize_final_corpus,
    materialize_production_pool,
    seal_qa_approval,
    seal_packing_preflight_approval,
    validate_backend_receipt,
    validate_source_receipt,
    write_d32_exposure_plan_index,
    write_runtime_exposure_manifests,
    write_tokenizer_sample,
)


DEFAULT_POLICY = Path("configs/pretrain/tr_d32_turkish_general_v4.json")


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate policy/optional receipts")
    validate.add_argument("--source-receipt", type=Path)
    validate.add_argument("--backend-receipt", type=Path)

    reference = subparsers.add_parser(
        "reference-pool", help="small fixture/smoke backend; forbidden for production"
    )
    reference.add_argument("--source-receipt", type=Path, required=True)
    reference.add_argument("--output-dir", type=Path, required=True)
    reference.add_argument(
        "--i-understand-reference-only", action="store_true", required=True
    )

    production = subparsers.add_parser(
        "production-pool", help="import pinned GlotLID/DataTrove output"
    )
    production.add_argument("--source-receipt", type=Path, required=True)
    production.add_argument("--backend-receipt", type=Path, required=True)
    production.add_argument("--cluster-launch-receipt", type=Path, required=True)
    production.add_argument("--output-dir", type=Path, required=True)

    sample = subparsers.add_parser(
        "tokenizer-sample", help="write deterministic post-filter train-only sample"
    )
    sample.add_argument("--pool-dir", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--max-chars", type=int)
    sample.add_argument("--allow-reference-pool", action="store_true")

    approve = subparsers.add_parser(
        "approve-qa", help="seal a manual decision after reviewing QA JSONL/plaintext"
    )
    approve.add_argument("--pool-dir", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--reviewed-at-utc", required=True)
    approve.add_argument("--decision", choices=("accepted", "rejected"), required=True)
    approve.add_argument("--notes", default="")

    packing = subparsers.add_parser(
        "packing-preflight",
        help="measure bounded exact-tokenizer crop retention before full materialization",
    )
    packing.add_argument("--pool-dir", type=Path, required=True)
    packing.add_argument("--tokenizer-dir", type=Path, required=True)
    packing.add_argument("--output", type=Path, required=True)
    packing.add_argument("--max-documents-per-rank-and-mixture", type=int, default=8192)
    packing.add_argument("--projection-safety-factor", type=float, default=1.08)

    packing_approve = subparsers.add_parser(
        "approve-packing-preflight",
        help="seal a manual decision on the measured source target/weights",
    )
    packing_approve.add_argument("--report", type=Path, required=True)
    packing_approve.add_argument("--output", type=Path, required=True)
    packing_approve.add_argument("--reviewer", required=True)
    packing_approve.add_argument("--reviewed-at-utc", required=True)
    packing_approve.add_argument(
        "--decision", choices=("accepted", "rejected"), required=True
    )
    packing_approve.add_argument("--notes", default="")

    final = subparsers.add_parser(
        "finalize", help="token-count and interleave the frozen 40x physical corpus"
    )
    final.add_argument("--pool-dir", type=Path, required=True)
    final.add_argument("--tokenizer-dir", type=Path, required=True)
    final.add_argument("--tokenizer-quality-dir", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    final.add_argument("--target-tokens", type=int, required=True)
    final.add_argument("--packing-preflight-dir", type=Path, required=True)
    final.add_argument("--shard-target-tokens", type=int, default=250_000_000)
    final.add_argument(
        "--quota-headroom-bytes",
        type=int,
        required=True,
        help="scheduler-reported bytes still available inside the project quota",
    )

    cleanup = subparsers.add_parser(
        "cleanup-pool",
        help="unlink only sealed run-owned pool fragments after capacity passes",
    )
    cleanup.add_argument("--pool-dir", type=Path, required=True)
    cleanup.add_argument("--final-corpus-dir", type=Path, required=True)

    exposure = subparsers.add_parser(
        "exposure", help="write fixed validation and one equal-token plan per world size"
    )
    exposure.add_argument("--final-corpus-dir", type=Path, required=True)
    exposure.add_argument("--study-sha256", required=True)
    exposure.add_argument("--tokenizer-sha256", required=True)
    exposure.add_argument("--seed", type=int, default=42)
    exposure.add_argument("--world-size", type=int, action="append", required=True)
    exposure.add_argument("--target-token-positions", type=int, required=True)
    exposure.add_argument("--optimizer-steps", type=int, required=True)
    exposure.add_argument("--validation-target-bytes", type=int, default=16_777_216)

    index = subparsers.add_parser(
        "exposure-index", help="write the complete frozen d32 exposure-plan matrix"
    )
    index.add_argument("--final-corpus-dir", type=Path, required=True)
    index.add_argument("--family-id", required=True)
    index.add_argument("--study-manifest-sha256", required=True)
    index.add_argument("--tokenizer-artifact-sha256", required=True)
    index.add_argument("--validation-target-bytes", type=int, default=16_777_216)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = load_corpus_policy(args.policy)
        if args.command == "validate":
            output: dict[str, object] = {"policy": policy["name"], "valid": True}
            if args.source_receipt:
                receipt = load_json_strict(args.source_receipt)
                validate_source_receipt(receipt, policy)
                output["source_receipt_sha256"] = receipt["canonical_sha256"]
            if args.backend_receipt:
                if not args.source_receipt:
                    raise ValueError("--source-receipt is required with --backend-receipt")
                source_receipt = load_json_strict(args.source_receipt)
                receipt = load_json_strict(args.backend_receipt)
                validate_backend_receipt(receipt, policy, source_receipt)
                output["backend_receipt_sha256"] = receipt["canonical_sha256"]
        elif args.command == "reference-pool":
            receipt = load_json_strict(args.source_receipt)
            output = materialize_filtered_pool(
                policy,
                receipt,
                args.output_dir,
                git_commit=_git_commit(),
                allow_reference_backend=args.i_understand_reference_only,
            )
        elif args.command == "production-pool":
            source_receipt = load_json_strict(args.source_receipt)
            receipt = load_json_strict(args.backend_receipt)
            cluster_launch_receipt = load_json_strict(args.cluster_launch_receipt)
            output = materialize_production_pool(
                policy,
                source_receipt,
                receipt,
                cluster_launch_receipt,
                args.output_dir,
                git_commit=_git_commit(),
            )
        elif args.command == "tokenizer-sample":
            output = write_tokenizer_sample(
                args.pool_dir,
                policy,
                args.output_dir,
                git_commit=_git_commit(),
                max_chars=args.max_chars,
                allow_reference_pool=args.allow_reference_pool,
            )
        elif args.command == "approve-qa":
            output = seal_qa_approval(
                args.pool_dir,
                reviewer=args.reviewer,
                reviewed_at_utc=args.reviewed_at_utc,
                decision=args.decision,
                notes=args.notes,
            )
        elif args.command == "packing-preflight":
            output = build_packing_preflight_report(
                args.pool_dir,
                policy,
                args.tokenizer_dir,
                args.output,
                max_documents_per_rank_and_mixture=(
                    args.max_documents_per_rank_and_mixture
                ),
                projection_safety_factor=args.projection_safety_factor,
            )
        elif args.command == "approve-packing-preflight":
            output = seal_packing_preflight_approval(
                args.report,
                args.output,
                reviewer=args.reviewer,
                reviewed_at_utc=args.reviewed_at_utc,
                decision=args.decision,
                notes=args.notes,
            )
        elif args.command == "finalize":
            output = materialize_final_corpus(
                args.pool_dir,
                policy,
                args.tokenizer_dir,
                args.tokenizer_quality_dir,
                args.output_dir,
                target_tokens=args.target_tokens,
                shard_target_tokens=args.shard_target_tokens,
                git_commit=_git_commit(),
                quota_headroom_bytes=args.quota_headroom_bytes,
                packing_preflight_dir=args.packing_preflight_dir,
            )
        elif args.command == "cleanup-pool":
            output = cleanup_verified_pool(args.pool_dir, args.final_corpus_dir)
        elif args.command == "exposure":
            output = write_runtime_exposure_manifests(
                args.final_corpus_dir,
                study_sha256=args.study_sha256,
                tokenizer_sha256=args.tokenizer_sha256,
                seed=args.seed,
                world_sizes=args.world_size,
                target_token_positions=args.target_token_positions,
                optimizer_steps=args.optimizer_steps,
                validation_target_bytes=args.validation_target_bytes,
            )
        else:
            output = write_d32_exposure_plan_index(
                args.final_corpus_dir,
                family_id=args.family_id,
                study_manifest_sha256=args.study_manifest_sha256,
                tokenizer_artifact_sha256=args.tokenizer_artifact_sha256,
                validation_target_bytes=args.validation_target_bytes,
            )
        print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
