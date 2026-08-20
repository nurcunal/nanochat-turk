"""Seal a human decision for the fixed held-out Turkish tokenizer report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nanochat.tokenizer_quality import seal_tokenizer_quality_approval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at-utc", required=True)
    parser.add_argument("--decision", choices=("accepted", "rejected"), required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    try:
        result = seal_tokenizer_quality_approval(
            args.quality_dir,
            reviewer=args.reviewer,
            reviewed_at_utc=args.reviewed_at_utc,
            decision=args.decision,
            notes=args.notes,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
