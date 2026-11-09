"""Summarize measured matched train/validation tokenizer results, fail on gaps."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.build_tokenizer_study_results import explain_metric_directions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/study_results"
SOURCE = OUT / "fresh_tokenizer_metrics"
FAMILIES = ("bpe", "morphbpe_trmorph", "morphbpe_zemberek", "morphbpe_tdelight")
LABELS = {"bpe": "Raw BPE", "morphbpe_trmorph": "TRmorph", "morphbpe_zemberek": "Zemberek", "morphbpe_tdelight": "TurkishDelight"}


def percent(v):
    return "N/A" if v is None else f"{100*v:.2f}"


def build():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    result = {"schema_version": 1, "evidence": "fresh_matched_measurements",
              "manifest": "docs/study_results/fresh_tokenizer_metrics/manifest.json",
              "job_id": str(manifest["environment"]["slurm_job_id"]), "splits": {}}
    all_rows = []
    for split in ("val", "train"):
        rows = []
        samples = set()
        for size in (32768, 65536, 131072):
            for family in FAMILIES:
                name = f"{family}_{size}"
                path = SOURCE / split / f"{name}_metrics.json"
                raw = path.read_bytes()
                d = json.loads(raw)
                c = d["coverage"]
                p = d["morphology"]["preservation"]
                assert d["vocabulary"]["vocab_size"] == size
                assert d["tokenizer_name"] == name
                assert d["split"] == split
                assert hashlib.sha256(raw).hexdigest() == manifest["splits"][split]["files"][path.name]
                assert c["sample_words"] == manifest["word_cap_per_tokenizer_per_split"]
                assert p["sample_word_occurrences"] == manifest["morphology_cap_per_tokenizer_per_split"]
                assert d["morphology"]["consistency"]["clustering"] == "sklearn_minibatch_kmeans"
                samples.add((d["docs"], d["bytes"], d["words"], c["sample_words"],
                             c["word_sequence_sha256"], p["reference_sequence_sha256"]))
                rows.append({
                    "split": split, "tokenizer": name, "family": family, "vocab_size": size,
                    "documents": d["docs"], "raw_bytes": d["bytes"], "regex_words": d["words"],
                    "document_tokens_per_word": d["tokens_per_word"],
                    "isolated_tokens_per_word": d["word_fertility_isolated"]["tokens_per_word"],
                    "one_token_word_rate": c["one_token_word_rate"],
                    "character_only_word_rate": c["character_only_word_rate"],
                    "multi_character_character_only_word_rate": c["multi_character_character_only_word_rate"],
                    "single_byte_only_word_rate": c["single_byte_only_word_rate"],
                    "unk_applicable": c["unk_applicable"], "unk_token_count": c["unk_token_count"],
                    "corpus_unk_token_count": c["corpus_unk_token_count"],
                    "morpheme_preservation_rate": p["morpheme_preservation_rate"],
                    "boundary_precision": p["boundary_precision"], "boundary_recall": p["boundary_recall"],
                    "boundary_f1": p["boundary_f1"], "exact_boundary_word_rate": p["exact_boundary_word_rate"],
                    "sample_word_occurrences": c["sample_words"],
                    "morphology_scored_word_occurrences": p["scored_word_occurrences"],
                    "morphology_unalignable_word_occurrences": p["unalignable_word_occurrences"],
                    "morpheme_count": p["morpheme_count"], "preserved_morpheme_count": p["preserved_morpheme_count"],
                    "roundtrip_failures": d["roundtrip_failures"],
                    "roundtrip_failure_rate": d["roundtrip_failure_rate"],
                    "bytes_per_token": d["bytes_per_token"],
                    "morphological_edit_distance": d["morphology"]["alignment"]["morphological_edit_distance"],
                    "morphological_consistency_f1": d["morphology"]["consistency"]["f1_mean"],
                    "source": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(raw).hexdigest(),
                })
        assert len(rows) == 12 and len(samples) == 1, f"Unmatched {split} results"
        sample = next(iter(samples))
        result["splits"][split] = {"documents": sample[0], "raw_bytes": sample[1], "regex_words": sample[2],
                                  "sample_word_occurrences": sample[3], "word_sequence_sha256": sample[4],
                                  "reference_sequence_sha256": sample[5], "rows": rows}
        all_rows.extend(rows)
    assert result["splits"]["train"]["word_sequence_sha256"] != result["splits"]["val"]["word_sequence_sha256"]
    (OUT / "tokenizer_fresh_results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (OUT / "tokenizer_fresh_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    lines = ["# Fresh matched tokenizer measurements", "",
             "New measurements on UHEM, 9 November 2026. All twelve original tokenizer bundles receive identical raw text within each split; segmentation is reference-only. [Run/sample provenance](fresh_tokenizer_metrics/manifest.json), [full precision JSON](tokenizer_fresh_results.json), [CSV](tokenizer_fresh_results.csv), and [complete evaluator outputs](fresh_tokenizer_metrics/) are retained.", "",
             "The validation file is withheld by the repository's configured file split; training and validation sample hashes differ. This is an automatic TRmorph-reference diagnostic, not independently annotated human morphology. No model was retrained for this evaluation. Reserved validation refers to the original tokenizer-study split; the later d32 all-shard training run includes this source text, so it is not held out from every model in the repository. The [archival matched report](tokenizer_matched.md) presents the same job 558060 measurements with additional recovery receipts; it is not an independent replication."]
    for split in ("val", "train"):
        sample = result["splits"][split]
        lines += ["", "## " + ("Validation" if split == "val" else "Training sample"), "",
                  f'{sample["documents"]:,} documents; {sample["raw_bytes"]:,} UTF-8 bytes; {sample["regex_words"]:,} regex words. Word-level metrics use {sample["sample_word_occurrences"]:,} occurrences per tokenizer. Identical word/reference hashes are required for all twelve rows.', "",
                  "| Vocab | Tokenizer | Corpus tok/word ↓ | Isolated tok/word ↓ | One-token words % | Character-only words % | Morphemes intact % ↑ | Isolated boundary P % ↑ | Isolated boundary R % ↑ | Isolated boundary F1 % ↑ |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sample["rows"]:
            lines.append(f'| {r["vocab_size"]:,} | {LABELS[r["family"]]} | {r["document_tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {percent(r["one_token_word_rate"])} | {percent(r["character_only_word_rate"])} | {percent(r["morpheme_preservation_rate"])} | {percent(r["boundary_precision"])} | {percent(r["boundary_recall"])} | {percent(r["boundary_f1"])} |')
        lines += ["", "| Vocab | Tokenizer | Multi-character words split only into characters % | Byte-only words % | Roundtrip failures | UNK count | Scored / excluded morphology words |",
                  "|---:|---|---:|---:|---:|---|---|"]
        for r in sample["rows"]:
            lines.append(f'| {r["vocab_size"]:,} | {LABELS[r["family"]]} | {percent(r["multi_character_character_only_word_rate"])} | {percent(r["single_byte_only_word_rate"])} | {r["roundtrip_failures"]} | {r["unk_token_count"] if r["unk_applicable"] else "N/A (byte-complete)"} | {r["morphology_scored_word_occurrences"]:,} / {r["morphology_unalignable_word_occurrences"]:,} |')
    lines += ["", "## Definitions", "",
        "- **Vocabulary size (↔)** includes the nine special tokens in each original bundle. Larger vocabularies can shorten sequences while increasing model embedding/output size, so size alone is not a quality score.",
        "- **Corpus fertility (↓ compression)** is total document tokens / regex word count; punctuation and spacing tokens are included. **Isolated fertility (↓ compression)** is sampled subword count / sampled word occurrences. A lower value means fewer units, not necessarily better morphology.",
        "- **Morpheme preservation (↑ reference agreement)** counts reference morpheme spans that coincide exactly with one token span, divided by scored reference morphemes. It is distinct from the earlier whole-word exact-morpheme-sequence rate; both compare with automatic annotations.",
        "- **Boundary consistency (↑ reference agreement)** reports micro precision, recall and F1 from isolated-word internal token boundaries versus internal reference morpheme boundaries. Precision measures how many predicted boundaries are correct; recall measures how many reference boundaries are found; F1 is their harmonic mean. Extra splits lower precision; merged morphemes lower recall. Byte boundaries inside a Unicode character are extra token boundaries. These isolated-word rates differ from contextual document-boundary preservation. The earlier pairwise morphology-consistency metric is retained separately in the raw outputs.",
        "- **One-token coverage (↑ isolated-word compression only)** counts losslessly represented, non-UNK words encoded as one token in isolation. **Character-only coverage (↔ diagnostic)** counts words represented entirely by one-Unicode-character tokens, including one-character words. The additional **multi-character-only rate (↓ fragmentation)** excludes one-character words from both numerator and denominator; a lower rate means fewer longer words split entirely into characters, not proven linguistic quality.",
        "- **Byte-only coverage (↔ diagnostic)** instead requires each token to represent one UTF-8 byte. Unicode character and byte rates are not interchangeable; short ASCII words can contribute without indicating a tokenizer failure.",
        "- **Roundtrip failures (↓; zero ideal)** count documents for which decoding the encoding does not reproduce the original text exactly. This checks text recovery, separately from whole-word coverage.",
        "- **UNK (↓ when applicable)** is architecture-specific: all twelve local tokenizers contain the full 256-byte alphabet and no UNK mechanism, so counts are N/A. For an UNK-enabled tokenizer, the evaluator measures counts and incidence instead of treating a one-UNK word as whole-word coverage.",
        "- **Scored words (↔ sample size) / excluded words (↓ alignment failures):** unalignable words are excluded from morphology denominators and reported explicitly. Fewer exclusions improve evaluation coverage but do not by themselves improve morphology quality. Undefined denominators produce null/N/A. Rates are occurrence-weighted, not unique-type weighted.", "",
        "Source sample preparation records ordered raw/annotated document hashes. The same morphology analyzer supplies reference labels and TRmorph tokenizer training supervision; interpretation must retain that bias limitation. Earlier train-corpus diagnostics and the full-corpus archive remain separate in [the index](README.md).", "",
        "Rebuild: `python -m scripts.build_fresh_tokenizer_results`."]
    (OUT / "tokenizer_fresh_results.md").write_text("\n".join(explain_metric_directions(lines, scope="fresh"))+"\n")


if __name__ == "__main__":
    build()
