"""Refresh the per-tokenizer overview from the consolidated evidence reports.

Run after build_tokenizer_study_results and build_model_study_results.
Only the delimited table in docs/study_results/README.md is changed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/study_results"
START = "<!-- BEGIN GENERATED TOKENIZER MODEL OVERVIEW -->"
END = "<!-- END GENERATED TOKENIZER MODEL OVERVIEW -->"


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
        "| Vocabulary / depth | Tokenizer | Corpus tokens/word ↓ | Morph consistency F1 ↑ | Val BPB ↓ | Core-11 ↑ | Training k tokens/s |",
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
    path = OUT / "README.md"
    text = path.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("Overview markers must occur exactly once")
    before, remainder = text.split(START)
    _, after = remainder.split(END)
    path.write_text(before + START + "\n\n" + "\n".join(lines) + "\n\n" + END + after)
    print("Built twelve-tokenizer overview from intrinsic, model and training-performance reports")


if __name__ == "__main__":
    main()
