"""Smoke-test and apply morphology segmenters.

Examples:
    python -m scripts.morph_segment --backend identity --text "evlerimizden geldik"
    TRMORPH_SEGMENT_CMD="..." python -m scripts.morph_segment --backend trmorph --input docs/sample.txt
"""

from __future__ import annotations

import argparse
import json
import sys

from nanochat.morphology import create_segmenter, iter_word_spans, segment_text, segment_text_spans


def _read_text(args) -> str:
    if args.text:
        return args.text
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Turkish morphology segmenter")
    parser.add_argument("--backend", type=str, default="identity", help="identity|trmorph|zemberek|tdelight|command")
    parser.add_argument("--command", type=str, default="", help="External segmenter command. Overrides backend env var.")
    parser.add_argument("--text", type=str, default="", help="Inline text to segment")
    parser.add_argument("--input", type=str, default="", help="UTF-8 text file to segment. Default reads stdin.")
    parser.add_argument("--format", type=str, default="text", choices=["text", "jsonl", "spans"])
    parser.add_argument("--delimiter", type=str, default=" ", help="Delimiter for --format text")
    parser.add_argument("--strict", action="store_true", help="Fail instead of identity-fallback on invalid backend output")
    parser.add_argument("--timeout", type=float, default=60.0, help="External command timeout in seconds")
    args = parser.parse_args()

    text = _read_text(args)
    segmenter = create_segmenter(
        args.backend,
        command=args.command or None,
        strict=args.strict,
        timeout=args.timeout,
    )

    if args.format == "text":
        print(segment_text(text, segmenter, delimiter=args.delimiter), end="" if text.endswith("\n") else "\n")
    elif args.format == "spans":
        rows = [
            {
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "word": span.word,
                "source": span.source,
            }
            for span in segment_text_spans(text, segmenter)
        ]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.format == "jsonl":
        words = [word for _start, _end, word in iter_word_spans(text)]
        for segmentation in segmenter.segment_words(words):
            print(json.dumps({
                "word": segmentation.word,
                "pieces": list(segmentation.pieces),
                "source": segmentation.source,
                "fallback": segmentation.fallback,
                "metadata": dict(segmentation.metadata),
            }, ensure_ascii=False))
    else:
        raise AssertionError(f"Unhandled format: {args.format}")


if __name__ == "__main__":
    main()
