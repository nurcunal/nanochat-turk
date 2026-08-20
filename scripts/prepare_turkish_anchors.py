"""Offline preparation of the pinned MOT v1.11 and ParlaMint-TR v5.0 anchors.

The production workflow is intentionally explicit: seal an acquisition receipt,
run a discovery preparation, manually accept its exact counts and logical hashes,
then rerun the same preparation with that count-acceptance receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.turkish_anchor_preparation import (
    DEFAULT_EVIDENCE_TARGET_BYTES,
    DEFAULT_SHARD_TARGET_BYTES,
    MOT_SOURCE_ID,
    MOT_V1_11_CONTRACT,
    PARLAMINT_SOURCE_ID,
    PARLAMINT_TR_V5_CONTRACT,
    prepare_mot_v1_11,
    prepare_parlamint_tr_v5,
    seal_anchor_acquisition_receipt,
    seal_anchor_count_acceptance,
    validate_anchor_preparation,
)


def _add_attestation_arguments(
    parser: argparse.ArgumentParser, *, timestamp_flag: str
) -> None:
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(timestamp_flag, required=True)
    parser.add_argument(
        "--decision", choices=("accepted", "rejected"), default="accepted"
    )
    parser.add_argument("--notes", default="")


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--discovery",
        action="store_true",
        help="emit an audited but production-ineligible discovery preparation",
    )
    mode.add_argument(
        "--count-acceptance",
        type=Path,
        help="rerun as production and require this exact accepted discovery receipt",
    )
    parser.add_argument(
        "--shard-target-bytes", type=int, default=DEFAULT_SHARD_TARGET_BYTES
    )
    parser.add_argument(
        "--evidence-target-bytes",
        type=int,
        default=DEFAULT_EVIDENCE_TARGET_BYTES,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mot_receipt = subparsers.add_parser(
        "seal-mot-acquisition",
        help="hash the two staged MOT assets and seal their acquisition receipt",
    )
    mot_receipt.add_argument("--amerikaninsesi-tgz", type=Path, required=True)
    mot_receipt.add_argument("--voaturkce-tgz", type=Path, required=True)
    mot_receipt.add_argument("--output", type=Path, required=True)
    _add_attestation_arguments(mot_receipt, timestamp_flag="--acquired-at-utc")

    parlamint_receipt = subparsers.add_parser(
        "seal-parlamint-acquisition",
        help="verify the staged ParlaMint asset against its official MD5 and seal receipt",
    )
    parlamint_receipt.add_argument("--archive-tgz", type=Path, required=True)
    parlamint_receipt.add_argument("--output", type=Path, required=True)
    _add_attestation_arguments(parlamint_receipt, timestamp_flag="--acquired-at-utc")

    accept = subparsers.add_parser(
        "accept-counts",
        help="seal manual acceptance of one discovery output's exact counts and hashes",
    )
    accept.add_argument("--discovery-output-dir", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)
    _add_attestation_arguments(accept, timestamp_flag="--reviewed-at-utc")

    mot = subparsers.add_parser(
        "mot-v1.11",
        help="prepare exactly the two official local Turkish MOT v1.11 tgz assets",
    )
    mot.add_argument("--amerikaninsesi-tgz", type=Path, required=True)
    mot.add_argument("--voaturkce-tgz", type=Path, required=True)
    _add_prepare_arguments(mot)

    parlamint = subparsers.add_parser(
        "parlamint-tr-v5.0",
        help="prepare the official local native Turkish ParlaMint 5.0 tgz",
    )
    parlamint.add_argument("--archive-tgz", type=Path, required=True)
    _add_prepare_arguments(parlamint)

    validate = subparsers.add_parser(
        "validate", help="revalidate one completed standalone preparation"
    )
    validate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "seal-mot-acquisition":
        result = seal_anchor_acquisition_receipt(
            MOT_SOURCE_ID,
            [args.amerikaninsesi_tgz, args.voaturkce_tgz],
            args.output,
            reviewer=args.reviewer,
            acquired_at_utc=args.acquired_at_utc,
            decision=args.decision,
            notes=args.notes,
            contract=MOT_V1_11_CONTRACT,
        )
    elif args.command == "seal-parlamint-acquisition":
        result = seal_anchor_acquisition_receipt(
            PARLAMINT_SOURCE_ID,
            [args.archive_tgz],
            args.output,
            reviewer=args.reviewer,
            acquired_at_utc=args.acquired_at_utc,
            decision=args.decision,
            notes=args.notes,
            contract=PARLAMINT_TR_V5_CONTRACT,
        )
    elif args.command == "accept-counts":
        result = seal_anchor_count_acceptance(
            args.discovery_output_dir,
            args.output,
            reviewer=args.reviewer,
            reviewed_at_utc=args.reviewed_at_utc,
            decision=args.decision,
            notes=args.notes,
        )
    elif args.command == "mot-v1.11":
        result = prepare_mot_v1_11(
            args.amerikaninsesi_tgz,
            args.voaturkce_tgz,
            args.output_dir,
            acquisition_receipt_path=args.acquisition_receipt,
            discovery=args.discovery,
            count_acceptance_path=args.count_acceptance,
            shard_target_bytes=args.shard_target_bytes,
            evidence_target_bytes=args.evidence_target_bytes,
        )
    elif args.command == "parlamint-tr-v5.0":
        result = prepare_parlamint_tr_v5(
            args.archive_tgz,
            args.output_dir,
            acquisition_receipt_path=args.acquisition_receipt,
            discovery=args.discovery,
            count_acceptance_path=args.count_acceptance,
            shard_target_bytes=args.shard_target_bytes,
            evidence_target_bytes=args.evidence_target_bytes,
        )
    else:
        result = validate_anchor_preparation(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
