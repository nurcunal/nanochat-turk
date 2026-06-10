"""Build a compact comparison table from tokenizer metric JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(payload: dict[str, Any], path: str, default: Any = "") -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(payload: dict[str, Any], path: str) -> dict[str, Any]:
    word = payload.get("word_fertility_isolated", {})
    boundary = payload.get("morph_boundary", {})
    vocab = payload.get("vocabulary", {})
    row = {
        "tokenizer": payload.get("tokenizer_name", Path(path).stem),
        "implementation": get_nested(payload, "tokenizer_config.implementation", ""),
        "docs": payload.get("docs", 0),
        "bytes": payload.get("bytes", 0),
        "tokens": payload.get("tokens", 0),
        "bytes_per_token": payload.get("bytes_per_token", 0.0),
        "tokens_per_word": payload.get("tokens_per_word", 0.0),
        "isolated_word_tokens_per_word": word.get("tokens_per_word", 0.0),
        "single_token_word_rate": word.get("single_token_word_rate", 0.0),
        "long_word_tokens_per_word": word.get("long_word_tokens_per_word", 0.0),
        "boundary_crossed_rate": boundary.get("crossed_boundary_rate", 0.0),
        "crossing_tokens_per_1k_tokens": boundary.get("crossing_tokens_per_1k_tokens", 0.0),
        "roundtrip_failure_rate": payload.get("roundtrip_failure_rate", 0.0),
        "unique_token_rate": payload.get("unique_token_rate_in_sample", 0.0),
        "vocab_utf8_decodable_rate": vocab.get("utf8_decodable_token_rate", 0.0),
        "encode_tokens_per_sec": payload.get("encode_tokens_per_sec", 0.0),
        "metrics_path": path,
    }
    return row


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Tokenizer Metrics Comparison",
        "",
        "Tokenizer-only metrics are measured before model training. True BPB is "
        "model-dependent and should be reported from validation loss after each "
        "pretraining run; this table reports tokenizer compression, fertility, "
        "boundary behavior, reversibility, and throughput.",
        "",
        "| Tokenizer | Impl. | Docs | Bytes | Tokens | Bytes/token ↑ | Tokens/word ↓ | Isolated word fertility ↓ | Single-token words ↑ | Long-word fertility ↓ | Boundary crossed ↓ | Crossing tok/1k ↓ | Roundtrip fail ↓ | Encode tok/s ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {tokenizer} | {implementation} | {docs} | {bytes} | {tokens} | {bpt} | {tpw} | {itpw} | {single} | {long} | {boundary} | {crossing} | {rt} | {speed} |".format(
                tokenizer=row["tokenizer"],
                implementation=row["implementation"],
                docs=fmt(row["docs"], 0),
                bytes=fmt(row["bytes"], 0),
                tokens=fmt(row["tokens"], 0),
                bpt=fmt(row["bytes_per_token"]),
                tpw=fmt(row["tokens_per_word"]),
                itpw=fmt(row["isolated_word_tokens_per_word"]),
                single=fmt(row["single_token_word_rate"]),
                long=fmt(row["long_word_tokens_per_word"]),
                boundary=fmt(row["boundary_crossed_rate"]),
                crossing=fmt(row["crossing_tokens_per_1k_tokens"]),
                rt=fmt(row["roundtrip_failure_rate"]),
                speed=fmt(row["encode_tokens_per_sec"], 0),
            )
        )

    lines.extend([
        "",
        "## Metric Notes",
        "",
        "- `Bytes/token`: compression proxy; higher means fewer tokens for the same raw bytes.",
        "- `Tokens/word`: corpus-level fertility; lower is usually better for Turkish.",
        "- `Isolated word fertility`: token count when each sampled word is encoded alone.",
        "- `Boundary crossed`: share of TRmorph morpheme boundaries crossed by at least one tokenizer token.",
        "- `Crossing tok/1k`: tokenizer tokens per 1,000 sample tokens that cross a TRmorph boundary.",
        "- `Roundtrip fail`: fraction of sampled documents where decode(encode(text)) differs from text.",
        "- High `Roundtrip fail` is expected for some encoder/seq2seq tokenizers that normalize or do not preserve raw text exactly.",
        "",
        "## Source Files",
        "",
    ])
    for row in rows:
        lines.append(f"- `{row['tokenizer']}`: `{row['metrics_path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tokenizer metric JSONs")
    parser.add_argument("--metric", action="append", required=True, help="Metric JSON path. Can be repeated.")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows = [summarize(load_json(path), path) for path in args.metric]
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"tokenizers": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
