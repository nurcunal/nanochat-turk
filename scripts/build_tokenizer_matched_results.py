"""Rebuild the archival mirror of matched job 558060 from completed receipts.

This reads saved measurements only. It refuses partial jobs, mismatched samples,
changed source/tokenizer artifacts, and inconsistent derived rates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.build_tokenizer_study_results import FAMILIES, OUT, ROOT, label, row_from_file, verify_rows
from scripts.tokenizer_table_interpretation import interpret_tables

ARTIFACT = ROOT / "artifacts/tokenizer_metrics_matched_2026-11-09"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_ratio(actual: float | None, numerator: int, denominator: int) -> None:
    expected = numerator / denominator if denominator else None
    if expected is None:
        if actual is not None:
            raise ValueError("A zero-denominator metric must be null")
    elif actual is None or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Rate {actual} does not match {numerator}/{denominator}")


def build() -> None:
    results = ARTIFACT / "results"
    recovery = json.loads((ARTIFACT / "recovery_manifest.json").read_text())
    for record in recovery["files"]:
        path = ARTIFACT / record["file"]
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Recovered artifact is missing or differs: {record['file']}")
    manifest = json.loads((results / "manifest.json").read_text())
    source = json.loads((ARTIFACT / "source_manifest.json").read_text())
    sample = json.loads((results / "sample_manifest.json").read_text())
    environment = json.loads((results / "environment.json").read_text())
    if (manifest["source"], manifest["sample"], manifest["environment"]) != (source, sample, environment):
        raise ValueError("Final receipt disagrees with source, sample or environment manifests")
    job_id = str(environment["slurm_job_id"])
    scheduler_lines = (ARTIFACT / "slurm_accounting.txt").read_text().splitlines()
    accounting = [line.split("|") for line in scheduler_lines if line.startswith(job_id + "|")]
    if len(accounting) != 1 or accounting[0][2:4] != ["COMPLETED", "0:0"]:
        raise ValueError("Successful terminal scheduler state is required")
    if manifest["word_cap_per_tokenizer_per_split"] != 500000 or manifest["morphology_cap_per_tokenizer_per_split"] != 500000:
        raise ValueError("Unexpected word metric caps")
    expected_names = [f"{family}_{size}" for size in (32768, 65536, 131072) for family in FAMILIES]
    for name in expected_names:
        for filename in ("tokenizer.pkl", "tokenizer_config.json"):
            relative = f"artifacts/tokenizers/{name}/{filename}"
            if digest(ROOT / relative) != source["files"][relative]:
                raise ValueError(f"Tokenizer bytes differ from measured source: {relative}")
    # The evaluator was an isolated source snapshot. Require its archived copy;
    # do not imply that current main contains the same runtime.
    source_files = {}
    for relative, expected in source["files"].items():
        if relative.startswith("artifacts/tokenizers/"):
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Canonical tokenizer artifact differs: {relative}")
            continue
        archived = ARTIFACT / "source" / relative
        if not archived.is_file() or digest(archived) != expected:
            raise ValueError(f"Archived evaluator source is missing or differs: {relative}")
        source_files[relative] = str(archived.relative_to(ROOT))
    all_rows, groups = [], {}
    for split in ("train", "val"):
        receipt = manifest["splits"][split]
        paths = [path for path in (results / split).glob("*_metrics.json")
                 if path.name != "requested_metrics.json"]
        if {p.stem.removesuffix("_metrics") for p in paths} != set(expected_names):
            raise ValueError(f"Missing or unexpected result files in {split}")
        if receipt["measurements"] != 12 or set(receipt["files"]) != {p.name for p in paths}:
            raise ValueError(f"Incomplete final receipt in {split}")
        if digest(results / split / "requested_metrics.json") != receipt["requested_metrics_sha256"]:
            raise ValueError(f"Final requested-metric summary hash differs in {split}")
        rows = []
        for name in expected_names:
            path = results / split / f"{name}_metrics.json"
            if digest(path) != receipt["files"][path.name]:
                raise ValueError(f"Final metric hash differs: {path}")
            r = row_from_file(path)
            d, c, p = r["original_metrics"], r["original_metrics"]["coverage"], r["original_metrics"]["morphology"]["preservation"]
            if d["tokenizer_name"] != name or d["vocabulary"]["vocab_size"] != int(name.rsplit("_", 1)[1]):
                raise ValueError(f"Tokenizer identity differs in {split}/{name}")
            if c["unk_applicable"] is not False or c["unk_token_count"] is not None:
                raise ValueError(f"Unexpected UNK mechanism in {split}/{name}")
            m = sample["splits"][split]
            if m["documents"] != 50000 or d["metric_workers"] != 1 or d["metric_tokenizer_threads_per_worker"] != 8:
                raise ValueError(f"Unexpected sample size or execution settings in {split}/{name}")
            if (d["split"], d["docs"], d["bytes"], d["chars"], d["words"]) != (split, m["documents"], m["raw_bytes"], m["raw_chars"], m["regex_words"]):
                raise ValueError(f"Sample totals differ in {split}/{name}")
            if c["word_sequence_sha256"] != receipt["word_sequence_sha256"] or p["reference_sequence_sha256"] != receipt["reference_sequence_sha256"]:
                raise ValueError(f"Word/reference sample differs in {split}/{name}")
            if c["sample_words"] != 500000 or p["sample_word_occurrences"] != 500000 or r["morphology_word_occurrences"] != 500000 or r["isolated_word_occurrences"] != 500000:
                raise ValueError(f"Word metric caps differ in {split}/{name}")
            if d["morphology"]["consistency"]["clustering"] != "sklearn_minibatch_kmeans" or d["morphology"]["consistency"]["sample_unique_words"] != 50000:
                raise ValueError(f"Consistency sample/protocol differs in {split}/{name}")
            for key, numerator in (("one_token_word_rate", "one_token_word_count"), ("character_only_word_rate", "character_only_word_count"), ("single_byte_only_word_rate", "single_byte_only_word_count")):
                check_ratio(c[key], c[numerator], c["sample_words"])
            check_ratio(c["multi_character_character_only_word_rate"], c["multi_character_character_only_word_count"], c["multi_character_word_count"])
            for key, numerator, denominator in (("morpheme_preservation_rate", "preserved_morpheme_count", "morpheme_count"), ("boundary_precision", "matched_boundary_count", "token_boundary_count"), ("boundary_recall", "matched_boundary_count", "reference_boundary_count")):
                check_ratio(p[key], p[numerator], p[denominator])
            check_ratio(p["boundary_f1"], 2*p["matched_boundary_count"], p["token_boundary_count"] + p["reference_boundary_count"])
            if p["scored_word_occurrences"] + p["unalignable_word_occurrences"] != p["sample_word_occurrences"]:
                raise ValueError(f"Morpheme scoring/exclusion counts differ in {split}/{name}")
            r.update({
                "whole_word_single_token_rate": c["one_token_word_rate"],
                "character_only_word_rate": c["character_only_word_rate"],
                "single_byte_only_word_rate": c["single_byte_only_word_rate"],
                "multi_character_character_only_word_rate": c["multi_character_character_only_word_rate"],
                "intact_morpheme_token_rate": p["morpheme_preservation_rate"],
                "boundary_precision": p["boundary_precision"], "boundary_f1": p["boundary_f1"],
                "isolated_boundary_recall": p["boundary_recall"],
                "coverage_unalignable_words": c["unalignable_word_count"],
                "morphology_unalignable_words": p["unalignable_word_occurrences"],
                "morphology_scored_words": p["scored_word_occurrences"],
                "unk_token_count": c["unk_token_count"], "unk_applicable": c["unk_applicable"],
                "word_sequence_sha256": c["word_sequence_sha256"],
                "reference_sequence_sha256": p["reference_sequence_sha256"],
            })
            rows.append(r)
        verify_rows(rows, expected_count=12)
        groups[split] = rows
        all_rows += rows
    OUT.mkdir(exist_ok=True, parents=True)
    receipt_paths = [ARTIFACT / "source_manifest.json", ARTIFACT / "recovery_manifest.json", results / "manifest.json", results / "sample_manifest.json", results / "environment.json", ARTIFACT / "slurm_accounting.txt"]
    payload = {"schema_version": 1, "evidence_status": "completed_fresh_matched_tokenizer_evaluation", "report_date": "2026-11-09",
               "job_id": job_id, "canonical_report": "docs/study_results/tokenizer_fresh_results.json",
               "same_measurement_run_as_canonical": True, "scheduler_state": "COMPLETED", "completed_utc": manifest["completed_utc"], "groups": groups,
               "receipts": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in receipt_paths],
               "source_manifest": source, "sample_manifest": sample, "environment": environment, "archived_source_files": source_files}
    (OUT / "tokenizer_matched.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    columns = [key for key in all_rows[0] if key != "original_metrics"]
    with (OUT / "tokenizer_matched.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in all_rows)
    lines = ["# Matched tokenizer evaluation: archival mirror", "",
        "This is an archival view of the same 24 measurements from job 558060 in the [canonical fresh tokenizer report](tokenizer_fresh_results.md). It preserves the recovered source snapshot and receipt checks; it is not an additional experiment or independent replication.", "",
        f'Fresh UHeM job **{job_id}** completed successfully on 9 November 2026 (exit 0:0; scheduler elapsed {accounting[0][4]}). This separate evaluation adds individual-morpheme preservation, literal boundary precision/recall/F1, character-only coverage and a reserved validation-shard sample for all twelve tokenizers. It does not create new language-model results. [Original measurements and receipts](../../artifacts/tokenizer_metrics_matched_2026-11-09/), [full JSON](tokenizer_matched.json), [flat CSV](tokenizer_matched.csv).', "",
        "Each split uses 50,000 documents, the same first 500,000 isolated word occurrences and morphology-reference occurrences across all tokenizers, and 50,000 unique words for the original consistency metric. The final receipt verifies matching word/reference hashes within each split. Tokenizer bytes match the twelve checked-in bundles. The train prefix and validation prefix come from distinct source shards; the latter is the final reserved source shard, not an independently human-annotated morphology dataset. Here reserved validation describes this tokenizer evaluation's sample split: the later d32 all-shard training run includes this source text, so it is not held out from every model in the repository.", "",
        "## Metric definitions", "",
        "- **Corpus tokens/word:** encoded document tokens divided by regex words; punctuation/spacing tokens contribute to the numerator.",
        "- **One-token and character-only coverage:** occurrence-weighted isolated words. Character-only requires every token to decode to one Unicode character; UTF-8 byte fragments are counted separately.",
        "- **Intact morphemes:** fraction of reference morpheme spans that coincide exactly with one token span. Splitting a morpheme or merging multiple morphemes loses that exact match.",
        "- **Boundary P/R/F1:** micro agreement of internal token boundaries and reference boundaries on isolated words; word edges are excluded. This differs from document-context boundary preservation and the original shared-token/shared-morpheme consistency F1.",
        "- **Alignment exclusions:** unalignable word encodings are counted; morphology denominators use scored words. The tables report the exclusions instead of treating them as successful alignment.",
        "- **UNK:** local byte-complete tokenizers have no UNK mechanism, so UNK is N/A, not an observed zero.", "",
        "Automatic TRmorph annotations may favor the TRmorph-trained treatment. These are single-run diagnostic measurements; consistency resample SD is not across-training-seed uncertainty or a confidence interval."]
    for split in ("train", "val"):
        rows, m = groups[split], sample["splits"][split]
        title = "Training prefix" if split == "train" else "Reserved validation-shard prefix"
        lines += ["", f"## {title}", "", f'{m["documents"]:,} documents; {m["raw_bytes"]:,} UTF-8 bytes; {m["regex_words"]:,} regex words. Source: `{Path(m["source_parquet_files_read"][0]).name}`. Percentages below use the denominators defined above.', "",
            "| Vocab | Tokenizer | Corpus tokens/word ↓ | Isolated tokens/word ↓ | One-token words % | Character-only words % | Intact morphemes % ↑ | Boundary P % ↑ | Boundary R % ↑ | Boundary F1 % ↑ |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f'| {r["vocab_size"]:,} | {label(r)} | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {100*r["character_only_word_rate"]:.2f} | {100*r["intact_morpheme_token_rate"]:.2f} | {100*r["boundary_precision"]:.2f} | {100*r["isolated_boundary_recall"]:.2f} | {100*r["boundary_f1"]:.2f} |')
        lines += ["", "### Compression and original morphology diagnostics", "",
            "| Vocab | Tokenizer | Bytes/token ↑ | Morph edit ↓ | Normalized edit ↓ | Exact morph sequence % | Consistency P | Consistency R | Consistency F1 | Document boundaries preserved % |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f'| {r["vocab_size"]:,} | {label(r)} | {r["bytes_per_token"]:.4f} | {r["morphological_edit_distance"]:.4f} | {r["morphological_edit_distance_normalized"]:.4f} | {100*r["exact_morpheme_sequence_rate"]:.2f} | {r["morphological_consistency_precision"]:.4f} | {r["morphological_consistency_recall"]:.4f} | {r["morphological_consistency_f1"]:.4f} | {100*r["boundary_preservation_rate"]:.2f} |')
        lines += ["", "### Coverage exclusions and recorded runtime", "",
            "All use one metric worker and eight tokenizer threads per process, with four tokenizer processes scheduled concurrently. Timing is a recorded runtime diagnostic rather than an isolated serial speed benchmark. Multi-character character-only coverage excludes one-character words from its denominator.", "",
            "| Vocab | Tokenizer | Byte-only words % | Multi-char character-only % | Coverage unaligned | Morphology unaligned | Roundtrip failures | Encode M tok/s | Evaluator wall s |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f'| {r["vocab_size"]:,} | {label(r)} | {100*r["single_byte_only_word_rate"]:.2f} | {100*r["multi_character_character_only_word_rate"]:.2f} | {r["coverage_unalignable_words"]} | {r["morphology_unalignable_words"]} | {r["roundtrip_failures"]} | {r["encode_tokens_per_second"]/1e6:.3f} | {r["metric_wall_seconds"]:.2f} |')
    lines += ["", "## Provenance and rebuild", "",
        "The evaluated runtime is an isolated source snapshot with per-file hashes, based on a prior repository revision. It is not a claim that the current `main` implementation ran this experiment. The source manifest records the evaluator, tokenizer and sample-preparation code hashes; environment and sample manifests preserve package versions, selected shards, document hashes and word/reference hashes. Original per-tokenizer outputs remain unchanged.", "",
        "Rebuild these tables from completed measurements with `python -m scripts.build_tokenizer_matched_results`. The builder requires terminal successful scheduler accounting, all 24 row hashes, final receipt agreement, matching document/word/reference samples, matching tokenizer bytes, and valid rate/count arithmetic. Rebuilding is separate from rerunning the evaluator.", "",
        "| Split | Tokenizer | Original JSON |", "|---|---|---|"]
    for r in all_rows:
        lines.append(f'| {r["split"]} | {r["tokenizer"]} | [{Path(r["source_file"]).name}](../../{r["source_file"]}) |')
    (OUT / "tokenizer_matched.md").write_text("\n".join(interpret_tables(lines, scope="matched")) + "\n")


if __name__ == "__main__":
    build()
