"""Report frozen d32 code-pin compatibility without changing any recipe or gate.

This diagnostic always reports drift as data, not as a new launch restriction.
Production launchers retain their existing strict validation. Matching these
file hashes alone does not establish production readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MAIN = "955c3809c94c04731bf35a9695cc0a04c28e35e9"
FROZEN_D32 = "0ea4231e26208be92bce765e4fdde94eeedb21fe"
RECIPE = "configs/pretrain/tr_d32_turkish_general_wsd_v4.json"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inspect(revision: str | None = None) -> dict:
    checked = git("rev-parse", "--verify", f"{revision}^{{commit}}" if revision else "HEAD").decode().strip()
    recipe_bytes = git("show", f"{FROZEN_D32}:{RECIPE}")
    recipe = json.loads(recipe_bytes)
    policy = recipe["code_provenance"]
    pins = dict(policy["exact_file_sha256"])
    pins["pyproject.toml"] = policy["training_environment"]["pyproject_sha256"]
    pins["uv.lock"] = policy["training_environment"]["uv_lock_sha256"]

    def read_target(relative: str) -> bytes | None:
        if revision:
            try:
                return git("show", f"{checked}:{relative}")
            except subprocess.CalledProcessError:
                return None
        path = ROOT / relative
        return path.read_bytes() if path.is_file() else None

    rows = []
    for relative, expected in sorted(pins.items()):
        frozen = digest(git("show", f"{FROZEN_D32}:{relative}"))
        value = read_target(relative)
        actual = digest(value) if value is not None else None
        rows.append({
            "path": relative, "expected_sha256": expected,
            "frozen_d32_sha256": frozen, "frozen_d32_matches": frozen == expected,
            "checked_sha256": actual, "matches": actual == expected,
        })
    current_recipe = read_target(RECIPE)
    return {
        "mode": "report_only_no_mutation_no_new_launch_gate",
        "historical_main_revision": HISTORICAL_MAIN,
        "frozen_d32_revision": FROZEN_D32,
        "pinned_upstream_revision": policy["upstream_base_revision"],
        "checked_revision": checked,
        "checked_source": "committed_revision" if revision else "current_worktree_bytes",
        "recipe_path": RECIPE,
        "recipe_unchanged_from_frozen_d32": current_recipe == recipe_bytes,
        "frozen_d32_all_pins_match": all(row["frozen_d32_matches"] for row in rows),
        "checked_all_pins_match": all(row["matches"] for row in rows),
        "files": rows,
        "limitations": [
            "Report-only diagnostic; drift does not change its successful process exit status.",
            "Existing strict preflight/runtime checks still reject incompatible code or environments.",
            "Matching file pins is necessary but insufficient: clean code, ancestry, Python, sealed data/tokenizer/receipts, topology, quota and other gates remain applicable.",
            "The historical-main revision is a recovery snapshot, not an assertion that all archived runs used that exact commit.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Inspect immutable bytes at this local Git revision instead of the working tree")
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable report")
    args = parser.parse_args()
    report = inspect(args.revision)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        matches = sum(row["matches"] for row in report["files"])
        print(f'Frozen d32 recipe: {RECIPE} at {FROZEN_D32}')
        print(f'Checked: {report["checked_source"]} (HEAD/revision {report["checked_revision"]})')
        print(f'Matching file pins: {matches}/{len(report["files"])}')
        for row in report["files"]:
            print(f'{"MATCH" if row["matches"] else "DRIFT":5}  {row["path"]}')
        print("Report only. No pins changed; existing strict production validation remains authoritative.")


if __name__ == "__main__":
    main()
