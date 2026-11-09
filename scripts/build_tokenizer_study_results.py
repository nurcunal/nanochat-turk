"""Build auditable presentation tables from archived tokenizer measurements.

This command never estimates missing measurements. The original JSON files are
preserved, with hashes and complete metrics copied into the consolidated JSON.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/tokenizer_tests/tokenizer_metrics"
OUT = ROOT / "docs/study_results"
FAMILIES = ("bpe", "morphbpe_trmorph", "morphbpe_zemberek", "morphbpe_tdelight")
LABELS = {"bpe": "Raw BPE", "morphbpe_trmorph": "MorphBPE / TRmorph",
          "morphbpe_zemberek": "MorphBPE / Zemberek", "morphbpe_tdelight": "MorphBPE / TurkishDelight"}


def row_from_file(path: Path, *, external: bool = False) -> dict:
    raw = path.read_bytes()
    d = json.loads(raw)
    w = d["word_fertility_isolated"]
    a = d["morphology"]["alignment"]
    c = d["morphology"]["consistency"]
    b = d["morph_boundary"]
    # Integer denominators are authoritative; do not round before deriving rates.
    boundary_preservation = ((b["boundary_count"] - b["crossed_boundaries"]) /
                             b["boundary_count"] if b["boundary_count"] else None)
    return {
        "tokenizer": d["tokenizer_name"], "external_reference": external,
        "vocab_size": d["vocabulary"]["vocab_size"],
        "special_tokens": d["vocabulary"]["special_tokens"],
        "split": d["split"], "documents": d["docs"], "raw_bytes": d["bytes"],
        "corpus_words": d["words"], "corpus_tokens": d["tokens"],
        "tokens_per_word": d["tokens_per_word"],
        "isolated_tokens_per_word": w["tokens_per_word"],
        "isolated_word_occurrences": w["sample_words"],
        "whole_word_single_token_rate": w["single_token_word_rate"],
        "character_only_word_rate": None,
        "unk_token_count": None,
        "unk_status": "not_applicable_byte_complete_bpe" if not external else "not_recorded",
        "roundtrip_success_rate": 1 - d["roundtrip_failure_rate"],
        "roundtrip_failures": d["roundtrip_failures"],
        "exact_morpheme_sequence_rate": a["exact_morpheme_sequence_rate"],
        "intact_morpheme_token_rate": None,
        "morphology_word_occurrences": a["sample_word_occurrences"],
        "morphological_edit_distance": a["morphological_edit_distance"],
        "morphological_edit_distance_normalized": a["morphological_edit_distance_normalized"],
        "morphological_consistency_precision": c["precision_mean"],
        "morphological_consistency_recall": c["recall_mean"],
        "morphological_consistency_f1": c["f1_mean"],
        "morphological_consistency_f1_resample_std": c["f1_std"],
        "reference_boundaries": b["boundary_count"],
        "crossed_reference_boundaries": b["crossed_boundaries"],
        "boundary_preservation_rate": boundary_preservation,
        "crossed_boundary_rate": b["crossed_boundary_rate"],
        "boundary_precision": None, "boundary_f1": None,
        "bytes_per_token": d["bytes_per_token"], "chars_per_token": d["chars_per_token"],
        "encode_tokens_per_second": d["encode_tokens_per_sec"],
        "vocab_used_rate": d["unique_token_rate_in_sample"],
        "source_file": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_metrics": d,
    }


def fmt(v, digits=3):
    return "not measured" if v is None else f"{v:.{digits}f}"


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row_from_file(SOURCE / f"{f}_{v}_metrics.json")
            for v in (32768, 65536, 131072) for f in FAMILIES]
    external = [row_from_file(SOURCE / f"{n}_metrics.json", external=True)
                for n in ("kumru_2b", "cosmos_turkish_gpt2")]
    payload = {
        "schema_version": 1, "report_date": "2026-11-09",
        "evidence_status": "archived_measurements_not_a_new_evaluation",
        "notes": [
            "All rows use the historical 50,000-document TRAIN corpus sample, not held-out test data.",
            "Word-level caps differ: 200,000 for three 32k rows and 500,000 for the others.",
            "TRmorph generates the morphology reference; it is not independent human annotation.",
            "Boundary preservation is derived from archived integer counts, not a new run.",
            "Exact morpheme-sequence rate and boundary preservation are distinct from intact-morpheme-token rate.",
            "Morphological consistency is the original shared-token/shared-morpheme pair metric, not boundary F1.",
            "Missing character-only, intact-morpheme and boundary precision measurements remain null.",
            "The 12 local BPE tokenizers have byte fallback and no UNK mechanism; UNK is N/A, not an empirical zero.",
            "Encoding throughput is historical and settings differ; do not treat it as a controlled speed benchmark.",
        ], "tokenizers": rows, "external_references": external,
    }
    (OUT / "tokenizer_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    columns = [k for k in rows[0] if k != "original_metrics"]
    with (OUT / "tokenizer_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in rows + external)
    lines = ["# Tokenizer results", "", "Archived measurements, consolidated 9 November 2026. All twelve local tokenizers are included. Full precision, original metrics and SHA-256 source hashes are in [JSON](tokenizer_results.json); flat values are in [CSV](tokenizer_results.csv).", "",
        "## Requested metrics", "", "**Scope:** 50,000 FineWeb-2 Turkish **training** documents, 186,328,421 bytes and 23,042,031 regex-defined words. These are corpus diagnostics, not held-out generalization results. Values below are percentages where marked `%`.", "",
        "| Vocabulary | Tokenizer | Tokens/word ↓ | Isolated tokens/word ↓ | Whole word: one token % | Exact morpheme sequence % ↑ | Preserved boundaries % ↑ | Morph consistency F1 ↑ | Word sample |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        f = r["tokenizer"].rsplit("_", 1)[0]
        lines.append(f'| {r["vocab_size"]:,} | {LABELS[f]} | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {100*r["exact_morpheme_sequence_rate"]:.2f} | {100*r["boundary_preservation_rate"]:.2f} | {r["morphological_consistency_f1"]:.4f} | {r["isolated_word_occurrences"]:,} |')
    lines += ["", "### Definitions and coverage", "",
        "- **Vocabulary size:** actual vocabulary including nine special tokens for every local tokenizer.",
        "- **Subword efficiency:** total encoded document tokens divided by regex word count (`\\w+`). This includes punctuation/spacing tokens in the numerator. Isolated fertility encodes each sampled word separately and is reported alongside it.",
        "- **Whole-word coverage:** fraction of sampled word occurrences represented by exactly one token when encoded in isolation. It is sensitive to whitespace conventions and differs from context-aware word coverage.",
        "- **Morpheme preservation:** the archived **exact morpheme-sequence rate** counts words whose token pieces exactly match the automatic morpheme sequence. It is a strict word-level proxy. The requested fraction of individual morphemes preserved as intact tokens was not stored and needs the new evaluator.",
        "- **Boundary preservation:** `(reference boundaries - crossed boundaries) / reference boundaries`, using all 12,706,795 reference boundaries. This measures retained reference boundaries; it does not penalize extra token splits, so it is not boundary precision or F1.",
        "- **Morphological consistency:** retain the original MorphBPE shared-token/shared-morpheme pair precision, recall and F1 (`k=100`, `C=50`, `N=10`). It must not be relabelled as boundary F1. The new evaluator reports explicit boundary P/R/F1 separately.",
        "- **Character-only coverage:** fraction of words split entirely into single Unicode-character tokens; it was not measured in the archived outputs. A UTF-8 byte fragment is not a character token. The added evaluator also separates single-byte-only segmentation.",
        "- **UNK:** N/A for the twelve local byte-complete BPE/MorphBPE tokenizers, which have no UNK mechanism. For tokenizers with an UNK token, the evaluator records count and token/word incidence; no historical count is fabricated.",
        "- **Roundtrip coverage:** all twelve local tokenizers had zero failures in 50,000 documents (100% exact decode/encode success). This establishes observed representability, not whole-word compression or morphology quality.",
        "", "## Previously reported metrics", "",
        "| Vocabulary | Tokenizer | Bytes/token ↑ | Morph edit distance ↓ | Normalized edit ↓ | Consistency P ↑ | Consistency R ↑ | Boundary crossed % ↓ | Used vocab % | Encode M tokens/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        f = r["tokenizer"].rsplit("_", 1)[0]
        lines.append(f'| {r["vocab_size"]:,} | {LABELS[f]} | {r["bytes_per_token"]:.4f} | {r["morphological_edit_distance"]:.4f} | {r["morphological_edit_distance_normalized"]:.4f} | {r["morphological_consistency_precision"]:.4f} | {r["morphological_consistency_recall"]:.4f} | {100*r["crossed_boundary_rate"]:.2f} | {100*r["vocab_used_rate"]:.2f} | {r["encode_tokens_per_second"]/1e6:.3f} |')
    lines += ["", "The JSON also preserves all original distributions, unique-word measurements, resampling standard deviations, vocabulary shape, worker/thread counts and timing. The original [comparison](../tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md) remains available.", "",
        "## External tokenizer references", "", "These are tokenizer-only diagnostics; their pretrained models are **not** part of the controlled LLM ablation.", "",
        "| Tokenizer | Vocabulary | Tokens/word | Whole word: one token % | Exact morpheme sequence % | Preserved boundaries % | Morph consistency F1 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in external:
        lines.append(f'| {r["tokenizer"]} | {r["vocab_size"]:,} | {r["tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {100*r["exact_morpheme_sequence_rate"]:.2f} | {100*r["boundary_preservation_rate"]:.2f} | {r["morphological_consistency_f1"]:.4f} |')
    lines += ["", "## Interpretation and limits", "",
        "At every vocabulary size, MorphBPE preserves more TRmorph reference boundaries and has better recorded morphology edit distance and consistency than raw BPE, while requiring more document tokens per word. TRmorph training and evaluation share an analyzer, so these are automatic-reference results. The 32k morphology and isolated-word caps differ across treatments, preventing a fully matched word-level estimate. Corpus-level fertility and boundary counts use the common 50,000 documents.", "",
        "Isolated-word fertility often improves with MorphBPE even though document tokens/word rises; the two measures capture different contexts. Job 496882 has now been verified COMPLETED and recovered separately in the [full-corpus report](tokenizer_full_corpus.md). Fresh measurements retain corpus/split identity, equal sample caps and separate provenance. No significance or causal advantage is claimed from these single-run diagnostics.", "",
        "Rebuild: `python -m scripts.build_tokenizer_study_results`."]
    (OUT / "tokenizer_results.md").write_text("\n".join(lines) + "\n")

    full_dir = ROOT / "artifacts/tokenizer_metrics_full_2026-06-23"
    if full_dir.exists():
        full = [row_from_file(full_dir / f"{f}_{v}_metrics.json")
                for v in (32768, 65536, 131072) for f in FAMILIES]
        assert len({(r["documents"], r["raw_bytes"], r["corpus_words"], r["reference_boundaries"]) for r in full}) == 1
        assert all(r["isolated_word_occurrences"] == r["morphology_word_occurrences"] == 500000 for r in full)
        result = {"schema_version": 1, "job_id": 496882, "job_state": "COMPLETED", "tokenizers": full,
                  "manifest": str((full_dir / "manifest.json").relative_to(ROOT)),
                  "note": "Document metrics cover the complete train split; isolated word and morphology alignment metrics use 500k occurrences; consistency uses 50k unique words."}
        (OUT / "tokenizer_full_corpus.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        with (OUT / "tokenizer_full_corpus.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows({k: r[k] for k in columns} for r in full)
        lines = ["# Recovered full-corpus tokenizer results", "",
            "UHEM job **496882** completed on 23 June 2026 (exit 0:0; 29h 20m 05s). Results were recovered on 9 November. [Original files and scheduler evidence](../../artifacts/tokenizer_metrics_full_2026-06-23/) are preserved unchanged; [JSON](tokenizer_full_corpus.json) and [CSV](tokenizer_full_corpus.csv) contain full precision and source hashes.", "",
            "All twelve rows cover the same **93,211,827 training documents**, **36,699,889,764 regex words**, and **299,173,651,770 UTF-8 bytes**. Full-corpus totals apply to compression, boundary crossing and roundtrip. Word-level metrics use **500,000 occurrences** for all rows, and consistency uses **50,000 unique words**. This is not a held-out test.", "",
            "| Vocab | Tokenizer | Corpus tokens/word ↓ | Isolated tokens/word ↓ | One-token words % | Bytes/token ↑ | Exact morph sequence % ↑ | Boundaries preserved % ↑ | Morph edit ↓ | Morph consistency F1 ↑ |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in full:
            f = r["tokenizer"].rsplit("_", 1)[0]
            lines.append(f'| {r["vocab_size"]:,} | {LABELS[f]} | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {r["bytes_per_token"]:.4f} | {100*r["exact_morpheme_sequence_rate"]:.2f} | {100*r["boundary_preservation_rate"]:.2f} | {r["morphological_edit_distance"]:.4f} | {r["morphological_consistency_f1"]:.4f} |')
        lines += ["", "All twelve rows record zero roundtrip failures. Per-morpheme intact-token preservation, character-only coverage and explicit boundary P/R/F1 were not present in these June outputs; see the fresh matched evaluation for those measurements. The full-corpus word caps resolve the earlier 32k 200k/500k cap mismatch, while retaining the automatic TRmorph-reference limitation. Original throughput is not a controlled tokenizer speed comparison.", "",
                  "The recovered generated comparison uses a diagnostic sorting order, not an official paper ranking. Compare each treatment with raw BPE of the same vocabulary size; avoid a single synthetic overall winner."]
        (OUT / "tokenizer_full_corpus.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    build()
