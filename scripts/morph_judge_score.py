"""Score blind LLM/human judgments for morphology segmenter judge packs.

The judge pack keeps candidate labels blind. This script maps judged labels back
to segmenter names using the answer key emitted by scripts.morph_judge_pack.

Expected judgment JSONL row:

    {"id":"morphjudge_000001","best_label":"A","acceptable_labels":["A","C"],"confidence":"medium","notes":"..."}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {exc}") from exc


def normalize_labels(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Score blind morphology judge outputs")
    parser.add_argument("--judgments", required=True, help="Judge output JSONL.")
    parser.add_argument("--answer-key", required=True, help="Judge-pack answer key JSON.")
    parser.add_argument("--output", default="", help="Optional summary JSON output path.")
    args = parser.parse_args()

    with open(args.answer_key, "r", encoding="utf-8") as f:
        answer_key = json.load(f)

    best_counts: Counter[str] = Counter()
    acceptable_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    invalid = []
    total = 0

    for line_no, row in iter_jsonl(args.judgments):
        total += 1
        item_id = str(row.get("id", ""))
        key = answer_key.get(item_id)
        if not isinstance(key, dict):
            invalid.append({
                "line": line_no,
                "id": item_id,
                "reason": "unknown_id",
            })
            continue

        best_label = str(row.get("best_label", "")).strip()
        best_backend = key.get(best_label)
        if best_backend:
            best_counts[best_backend] += 1
        else:
            invalid.append({
                "line": line_no,
                "id": item_id,
                "reason": "invalid_best_label",
                "best_label": best_label,
            })

        for label in normalize_labels(row.get("acceptable_labels", [])):
            backend = key.get(label.strip())
            if backend:
                acceptable_counts[backend] += 1
            else:
                invalid.append({
                    "line": line_no,
                    "id": item_id,
                    "reason": "invalid_acceptable_label",
                    "label": label,
                })

        confidence = str(row.get("confidence", "")).strip() or "unspecified"
        confidence_counts[confidence] += 1

    denominator = max(total - len([x for x in invalid if x["reason"] == "unknown_id"]), 1)
    summary = {
        "judgments": args.judgments,
        "answer_key": args.answer_key,
        "total_rows": total,
        "best_counts": dict(best_counts.most_common()),
        "best_rates": {
            backend: count / denominator for backend, count in best_counts.items()
        },
        "acceptable_counts": dict(acceptable_counts.most_common()),
        "confidence_counts": dict(confidence_counts.most_common()),
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:25],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
