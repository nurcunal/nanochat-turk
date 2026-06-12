"""Build a compact comparison table from tokenizer metric JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def as_float(value: Any, default: float) -> float:
    if value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def paper_style_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    """MorphBPE-paper-style ordering without a synthetic weighted score."""

    return (
        as_float(row["morph_edit_distance"], math.inf),
        -as_float(row["morph_consistency_f1"], -math.inf),
        as_float(row["tokens_per_word"], math.inf),
        as_float(row["boundary_crossed_rate"], math.inf),
        row["tokenizer"],
    )


def rank_rows_paper_style(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for rank, row in enumerate(sorted(rows, key=paper_style_sort_key), start=1):
        ranked.append({"paper_style_rank": rank, **row})
    return ranked


def summarize(payload: dict[str, Any], path: str) -> dict[str, Any]:
    word = payload.get("word_fertility_isolated", {})
    boundary = payload.get("morph_boundary", {})
    morphology = payload.get("morphology", {})
    morph_alignment = morphology.get("alignment", {})
    morph_consistency = morphology.get("consistency", {})
    vocab = payload.get("vocabulary", {})
    tokenizer_config = payload.get("tokenizer_config", {})
    row = {
        "tokenizer": payload.get("tokenizer_name", Path(path).stem),
        "implementation": tokenizer_config.get("implementation", ""),
        "source": tokenizer_config.get("source", "local"),
        "model_id": tokenizer_config.get("model_id", ""),
        "vocab_size": vocab.get("vocab_size", tokenizer_config.get("vocab_size", "")),
        "docs": payload.get("docs", 0),
        "bytes": payload.get("bytes", 0),
        "tokens": payload.get("tokens", 0),
        "bytes_per_token": payload.get("bytes_per_token", 0.0),
        "tokens_per_word": payload.get("tokens_per_word", 0.0),
        "isolated_word_tokens_per_word": word.get("tokens_per_word", 0.0),
        "single_token_word_rate": word.get("single_token_word_rate", 0.0),
        "long_word_tokens_per_word": word.get("long_word_tokens_per_word", 0.0),
        "morph_edit_distance": morph_alignment.get("morphological_edit_distance", ""),
        "morph_edit_distance_norm": morph_alignment.get("morphological_edit_distance_normalized", ""),
        "morph_exact_sequence_rate": morph_alignment.get("exact_morpheme_sequence_rate", ""),
        "morph_consistency_precision": morph_consistency.get("precision_mean", ""),
        "morph_consistency_recall": morph_consistency.get("recall_mean", ""),
        "morph_consistency_f1": morph_consistency.get("f1_mean", ""),
        "morph_consistency_clustering": morph_consistency.get("clustering", ""),
        "morph_sample_words": morph_alignment.get("sample_word_occurrences", morphology.get("sample_word_occurrences", "")),
        "morph_consistency_words": morph_consistency.get("sample_unique_words", ""),
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
        "MorphBPE paper morphology metrics, boundary behavior, reversibility, "
        "and throughput.",
        "",
        "Rows are sorted by MorphBPE-paper-style intrinsic quality: lower "
        "`mu_e`, then higher `mu_c` F1, then lower fertility `phi` as an "
        "efficiency tie-breaker. This ordering is not a custom weighted score.",
        "",
        "| Rank | Tokenizer | Source | Impl. | Vocab | Docs | Bytes/token ↑ | Tokens/word phi ↓ | Isolated fertility ↓ | Morph edit mu_e ↓ | Morph edit norm ↓ | Morph exact ↑ | Morph consistency P ↑ | Morph consistency R ↑ | Morph consistency F1 mu_c ↑ | Boundary crossed ↓ | Roundtrip fail ↓ | Encode tok/s ↑ |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {tokenizer} | {source} | {implementation} | {vocab} | {docs} | {bpt} | {tpw} | {itpw} | {med} | {medn} | {mex} | {mcp} | {mcr} | {mcf1} | {boundary} | {rt} | {speed} |".format(
                rank=row["paper_style_rank"],
                tokenizer=row["tokenizer"],
                source=row["model_id"] or row["source"],
                implementation=row["implementation"],
                vocab=fmt(row["vocab_size"], 0),
                docs=fmt(row["docs"], 0),
                bpt=fmt(row["bytes_per_token"]),
                tpw=fmt(row["tokens_per_word"]),
                itpw=fmt(row["isolated_word_tokens_per_word"]),
                med=fmt(row["morph_edit_distance"]),
                medn=fmt(row["morph_edit_distance_norm"]),
                mex=fmt(row["morph_exact_sequence_rate"]),
                mcp=fmt(row["morph_consistency_precision"]),
                mcr=fmt(row["morph_consistency_recall"]),
                mcf1=fmt(row["morph_consistency_f1"]),
                boundary=fmt(row["boundary_crossed_rate"]),
                rt=fmt(row["roundtrip_failure_rate"]),
                speed=fmt(row["encode_tokens_per_sec"], 0),
            )
        )

    lines.extend([
        "",
        "## Metric Notes",
        "",
        "- `Bytes/token`: compression proxy; higher means fewer tokens for the same raw bytes.",
        "- `Tokens/word phi`: corpus-level fertility, matching the MorphBPE paper's fertility metric.",
        "- `Isolated word fertility`: token count when each sampled word is encoded alone.",
        "- `Morph edit mu_e`: MorphBPE paper's raw average edit distance between gold morpheme sequence and tokenizer-piece sequence.",
        "- `Morph edit norm`: same edit distance divided by the number of gold morphemes.",
        "- `Morph exact`: share of sampled word occurrences whose tokenizer pieces exactly match gold morphemes.",
        "- `Morph consistency P/R/F1 mu_c`: binary shared-token/shared-morpheme precision, recall, and harmonic mean; defaults follow the MorphBPE paper (`k=100`, `C=50`, `N=10`).",
        "- `Boundary crossed`: share of TRmorph morpheme boundaries crossed by at least one tokenizer token.",
        "- `Roundtrip fail`: fraction of sampled documents where decode(encode(text)) differs from text.",
        "- High `Roundtrip fail` is expected for some encoder/seq2seq tokenizers that normalize or do not preserve raw text exactly.",
        "",
        "## Morphology Metric Sample Sizes",
        "",
        "| Tokenizer | Morph word occurrences | Consistency unique words | Consistency clustering |",
        "|---|---:|---:|---|",
    ])
    for row in rows:
        lines.append(
            "| {tokenizer} | {words} | {consistency_words} | {clustering} |".format(
                tokenizer=row["tokenizer"],
                words=fmt(row["morph_sample_words"], 0),
                consistency_words=fmt(row["morph_consistency_words"], 0),
                clustering=row["morph_consistency_clustering"],
            )
        )

    lines.extend([
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

    rows = rank_rows_paper_style([summarize(load_json(path), path) for path in args.metric])
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
