"""Refresh the per-tokenizer overview from the consolidated evidence reports.

Run after build_tokenizer_study_results and build_model_study_results.
Writes a private section consumed by build_study_report.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.study_table_highlights import highlight_columns, map_tables
from scripts.study_report_paths import DATA_DIR, write_section

ROOT = Path(__file__).resolve().parents[1]
OUT = DATA_DIR


def main():
    tokenizers = json.loads((OUT / "tokenizer_full_corpus.json").read_text())["tokenizers"]
    models = {r["tokenizer"]: r for r in json.loads((OUT / "model_results.json").read_text())["runs"]}
    with (OUT / "model_training_performance.csv").open(newline="") as f:
        speed = {r["tokenizer"]: float(r["median_last100_tokens_per_second"]) for r in csv.DictReader(f)}
    if len(tokenizers) != 12 or len(models) != 12 or len(speed) != 11:
        raise ValueError("Expected twelve tokenizer cells and eleven measured models")
    families = {"bpe": "Raw BPE", "trmorph": "MorphBPE / TRmorph", "zemberek": "MorphBPE / Zemberek", "tdelight": "MorphBPE / TurkishDelight"}
    tier = {32768: "32k / d20", 65536: "64k / d16", 131072: "128k / d12"}
    lines = [
        "| Vocabulary / depth ↔ | Tokenizer | Corpus tokens/word ↓ | Morph consistency F1 ↑ | Val BPB ↓ | Legacy core-11 ↑ | Training k tokens/s ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    def fmt(value, digits=4):
        return "—" if value is None else f"{value:.{digits}f}"
    for t in tokenizers:
        name = t["tokenizer"]
        m = models[name]
        family = "bpe" if name.startswith("bpe_") else name.split("_")[1]
        values = [tier[t["vocab_size"]], families[family], fmt(t["tokens_per_word"]), fmt(t["morphological_consistency_f1"]), fmt(m["validation_bpb"]), fmt(m["macro_core11"]), fmt(speed[name] / 1000, 1) if name in speed else "—"]
        lines.append("| " + " | ".join(values) + " |")
    table = map_tables(
        "\n".join(lines),
        lambda headers, rows: highlight_columns(
            rows, {2: "min", 3: "max", 4: "min", 5: "max", 6: "max"},
            group_keys=[row[0] for row in rows],
        ),
    )
    write_section("overview.md", table + "\n")
    print("Built twelve-tokenizer overview from intrinsic, model and training-performance reports")


if __name__ == "__main__":
    main()
