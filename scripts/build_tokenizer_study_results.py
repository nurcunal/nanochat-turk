"""Build auditable presentation tables from archived tokenizer measurements.

This command never estimates missing measurements. The original JSON files are
preserved, with hashes and complete metrics copied into the consolidated JSON.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.study_table_highlights import highlight_columns, map_tables
from scripts.tokenizer_table_interpretation import interpret_tables

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/tokenizer_tests/tokenizer_metrics"
OUT = ROOT / "docs/study_results"
FAMILIES = ("bpe", "morphbpe_trmorph", "morphbpe_zemberek", "morphbpe_tdelight")
LABELS = {"bpe": "Raw BPE", "morphbpe_trmorph": "MorphBPE / TRmorph",
          "morphbpe_zemberek": "MorphBPE / Zemberek", "morphbpe_tdelight": "MorphBPE / TurkishDelight"}


def highlight_tokenizer_scores(text: str, *, archived_rows: list[dict] | None = None) -> str:
    """Bold objective-specific extrema only within comparable table groups."""
    samples = {(f'{r["vocab_size"]:,}', label(r)): r for r in archived_rows or []}
    minimize = {"Tokens/word", "Corpus tokens/word", "Corpus tok/word", "Isolated tokens/word", "Isolated tok/word",
                "Morph edit", "Morph edit distance", "Normalized edit", "Normalized morph edit", "Boundary crossed %",
                "Crossed boundaries %", "Multi-character words split only into characters %", "Multi-char character-only %",
                "Roundtrip failures", "UNK count", "Coverage unaligned", "Morphology unaligned"}
    maximize = {"Bytes/token", "One-token words %", "Whole word: one token %", "Morphemes intact %", "Intact morphemes %",
                "Isolated boundary P %", "Isolated boundary R %", "Isolated boundary F1 %", "Boundary P %", "Boundary R %",
                "Boundary F1 %", "Exact morph sequence %", "Exact morpheme sequence %", "Boundaries preserved %",
                "Preserved boundaries %", "Document boundaries preserved %", "Morph consistency F1", "Consistency P",
                "Consistency R", "Consistency F1"}
    isolated = {"Isolated tokens/word", "Isolated tok/word", "One-token words %", "Whole word: one token %"}
    alignment = {"Exact morph sequence %", "Exact morpheme sequence %", "Morph edit", "Morph edit distance",
                 "Normalized edit", "Normalized morph edit"}
    consistency = {"Morph consistency F1", "Consistency P", "Consistency R", "Consistency F1"}

    def transform(headers: list[str], rows: list[list[str]]) -> list[list[str]]:
        names = [h.split(" ↑")[0].split(" ↓")[0].split(" ↔")[0] for h in headers]
        if any(name.startswith("Recorded ") for name in names) or "Original JSON" in names:
            return rows
        if "Analyzer" in names:
            if "Best count" in names:
                return highlight_columns(rows, {i: "max" for i, name in enumerate(names)
                                                if name in {"Best count", "Best %", "Acceptable count", "Acceptable %"}})
            sample_index, analyzer_index = names.index("Sample"), names.index("Analyzer")
            # Only the common hash-100k analyzer screen has matched inputs.
            groups = ["hash-100k" if row[sample_index].startswith("hash-100k")
                      else (row[sample_index], row[analyzer_index]) for row in rows]
            return highlight_columns(rows, {i: "min" for i, name in enumerate(names)
                                            if name in {"Fallback % (weighted)", "Fallback % (types)"}}, groups)
        vocab_index = next((i for i, name in enumerate(names) if name in {"Vocab", "Vocabulary"}), None)
        if vocab_index is None or "Tokenizer" not in names:
            # Different train/validation workloads and single-row diagnostics
            # must not acquire a winner merely because an arrow is present.
            return rows
        tokenizer_index = names.index("Tokenizer")
        output = rows
        for index, name in enumerate(names):
            direction = "min" if name in minimize else "max" if name in maximize else None
            if direction is None:
                continue
            groups = []
            for row in rows:
                key = row[vocab_index]
                if samples and name in isolated | alignment | consistency:
                    record = samples.get((key, row[tokenizer_index]))
                    if record is None:
                        key = (key, row[tokenizer_index])
                    elif name in isolated:
                        key = (key, record["isolated_word_occurrences"])
                    elif name in alignment:
                        key = (key, record["morphology_word_occurrences"])
                    else:
                        c = record["original_metrics"]["morphology"]["consistency"]
                        key = (key, c["sample_unique_words"], c["clustering"], c["clusters_requested"],
                               c["pairs_per_cluster"], c["resamples"])
                groups.append(key)
            output = highlight_columns(output, {index: direction}, groups)
        return output

    return map_tables(text, transform)


def explain_metric_directions(lines: list[str], *, scope: str, archived_rows: list[dict] | None = None) -> list[str]:
    """Label display headers after interpreting their original metric names.

    Keep the interpretation helper's exact header keys and every measured cell
    unchanged. These labels describe objectives, not an overall quality score.
    """
    output = interpret_tables(lines, scope=scope)
    output[2:2] = [
        "**Direction legend:** ↑ means higher is better for the stated objective; ↓ means lower is better; ↔ means descriptive or a tradeoff, with no universally better direction. See the [metric guide](metric_guide.md) for every definition, denominator and limitation.", "",
        "**Bold scores** mark the best comparable value for that metric within the same vocabulary and table's split/sample; all ties at displayed precision are bold. They do not identify an overall winner. Descriptive fields, uncontrolled timings, normalizing references and groups with only one measured value stay unbolded. Analyzer fallback is compared only on the common hash-100k screen; judge scores only within the shared scripted rubric.", "",
        "**Objectives:** lower tokens/word and higher bytes/token mean more compact lossless encoding. Higher one-token coverage helps isolated-word compression only. Higher morphology preservation, boundary agreement and consistency, or lower morphology edit distance, mean closer agreement with the automatic reference; they do not establish better LLM performance. Lower multi-character character-only coverage means less character-level fragmentation.", "",
        "**Diagnostics:** vocabulary size, vocabulary use, sample counts, inclusive character-only/byte-only coverage and resample SD have no universal optimum. Zero roundtrip failures is ideal for exact text recovery; fewer UNKs and alignment exclusions are better when applicable. Speed ↑ and time ↓ apply only to matched workloads and hardware; recorded timings here do not establish a controlled speed ranking. Normalizing references' apparent compression and boundary figures remain descriptive.", "",
    ]
    directions = {}
    for suffix, names in (
        ("↔", ("Vocab", "Vocabulary", "Word sample", "Unique forms", "Documents", "UTF-8 bytes", "Regex words", "Workers × threads", "Used vocab %", "F1 resample SD", "Character-only words %", "Byte-only words %", "Pieces/word (weighted)", "Split % (weighted)", "Recorded tokens/word", "Recorded one-token words %", "Recorded boundary crossed %")),
        ("↓", ("Tokens/word", "Corpus tokens/word", "Corpus tok/word", "Isolated tokens/word", "Isolated tok/word", "Morph edit", "Morph edit distance", "Normalized edit", "Normalized morph edit", "Boundary crossed %", "Crossed boundaries %", "UNK count", "Coverage unaligned", "Morphology unaligned")),
        ("↑", ("Bytes/token", "Morphemes intact %", "Intact morphemes %", "Isolated boundary P %", "Isolated boundary R %", "Isolated boundary F1 %", "Boundary P %", "Boundary R %", "Boundary F1 %", "Exact morph sequence %", "Exact morpheme sequence %", "Boundaries preserved %", "Preserved boundaries %", "Document boundaries preserved %", "Morph consistency F1", "Consistency P", "Consistency R", "Consistency F1")),
        ("↑ compression", ("One-token words %", "Whole word: one token %")),
        ("↓ fragmentation", ("Multi-character words split only into characters %", "Multi-char character-only %")),
        ("↓ (0 ideal)", ("Roundtrip failures", "Failure %")),
        ("↑ speed", ("Encode M tok/s", "Encode M tokens/s", "Forms/s")),
        ("↓ time", ("Encoding s (summed)", "Evaluator wall s", "Runtime s")),
        ("↓ fallback", ("Fallback % (weighted)", "Fallback % (types)")),
        ("↑ rubric", ("Best count", "Best %", "Acceptable count", "Acceptable %")),
        ("↔ / ↓", ("Scored / excluded morphology words",)),
    ):
        directions.update({name: suffix for name in names})
    labels = {"Tokenizer", "Analyzer", "Sample", "Split", "Scope", "Original JSON"}
    for index, line in enumerate(output[:-1]):
        if not line.startswith("|") or not output[index + 1].startswith("|---"):
            continue
        headers = [cell.strip().rstrip(" ↑↓↔") for cell in line.strip("|").split("|")]
        normalizing = "Recorded tokens/word" in headers
        annotated = []
        for header in headers:
            if header in labels:
                annotated.append(header)
                continue
            suffix = directions[header]
            if normalizing and header in {"Isolated tokens/word", "Bytes/token"}:
                suffix = "↔"
            display = "Document-pass wall s" if header == "Evaluator wall s" else header
            annotated.append(f"{display} {suffix}")
        output[index] = "| " + " | ".join(annotated) + " |"
    if archived_rows is not None:
        output[6:6] = ["**Historical sample groups:** at 32k, isolated-word and morphology-edit/exact-sequence bolding compares only the three 200,000-word rows; TurkishDelight's 500,000-word row has no matched peer and stays unbolded for those metrics. Corpus metrics compare all four rows. Pairwise consistency also compares all four because each uses the same 50,000-word-type prefix and clustering settings.", ""]
    return highlight_tokenizer_scores("\n".join(output), archived_rows=archived_rows).splitlines()


def row_from_file(path: Path, *, external: bool = False) -> dict:
    raw = path.read_bytes()
    d = json.loads(raw)
    w = d["word_fertility_isolated"]
    a = d.get("morphology", {}).get("alignment", {})
    c = d.get("morphology", {}).get("consistency", {})
    b = d["morph_boundary"]
    # Integer denominators are authoritative; do not round before deriving rates.
    boundary_preservation = ((b["boundary_count"] - b["crossed_boundaries"]) /
                             b["boundary_count"] if b["boundary_count"] else None)
    return {
        "tokenizer": d["tokenizer_name"], "external_reference": external,
        "evaluation_scope": "morphology_reference" if a else "raw_text_diagnostic",
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
        "exact_morpheme_sequence_rate": a.get("exact_morpheme_sequence_rate"),
        "intact_morpheme_token_rate": None,
        "morphology_word_occurrences": a.get("sample_word_occurrences"),
        "morphological_edit_distance": a.get("morphological_edit_distance"),
        "morphological_edit_distance_normalized": a.get("morphological_edit_distance_normalized"),
        "morphological_consistency_precision": c.get("precision_mean"),
        "morphological_consistency_recall": c.get("recall_mean"),
        "morphological_consistency_f1": c.get("f1_mean"),
        "morphological_consistency_f1_resample_std": c.get("f1_std"),
        "reference_boundaries": b["boundary_count"],
        "crossed_reference_boundaries": b["crossed_boundaries"],
        "boundary_preservation_rate": boundary_preservation,
        "crossed_boundary_rate": b["crossed_boundary_rate"] if b["boundary_count"] else None,
        "boundary_precision": None, "boundary_f1": None,
        "bytes_per_token": d["bytes_per_token"], "chars_per_token": d["chars_per_token"],
        "encode_tokens_per_second": d["encode_tokens_per_sec"],
        "encode_seconds_accumulated": d["encode_seconds"],
        "metric_wall_seconds": d.get("metric_wall_seconds"),
        "metric_workers": d.get("metric_workers"),
        "tokenizer_threads_per_worker": d.get("metric_tokenizer_threads_per_worker"),
        "vocab_used_rate": d["unique_token_rate_in_sample"],
        "source_file": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "original_metrics": d,
    }


def fmt(v, digits=3):
    return "not measured" if v is None else f"{v:.{digits}f}"


def label(row: dict) -> str:
    return LABELS.get(row["tokenizer"].rsplit("_", 1)[0], row["tokenizer"])


def verify_rows(rows: list[dict], *, expected_count: int, matched_docs: bool = True) -> None:
    if len(rows) != expected_count or len({r["tokenizer"] for r in rows}) != expected_count:
        raise ValueError("Missing or duplicate tokenizer rows")
    signatures = {(r["split"], r["documents"], r["raw_bytes"], r["corpus_words"], r["reference_boundaries"]) for r in rows}
    if matched_docs and len(signatures) != 1:
        raise ValueError("Tokenizer rows use different document samples")
    for r in rows:
        d = r["original_metrics"]
        ratios = [(r["tokens_per_word"], r["corpus_tokens"] / r["corpus_words"]),
                  (r["bytes_per_token"], r["raw_bytes"] / r["corpus_tokens"]),
                  (r["roundtrip_success_rate"], 1 - r["roundtrip_failures"] / r["documents"])]
        if r["reference_boundaries"]:
            ratios.append((r["boundary_preservation_rate"], 1 - d["morph_boundary"]["crossed_boundary_rate"]))
        if any(not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12) for actual, expected in ratios):
            raise ValueError(f"Inconsistent archived ratios: {r['tokenizer']}")


def systems_table(rows: list[dict]) -> list[str]:
    lines = ["", "## Archived encoding and evaluator performance", "",
        "These are recorded timings, not a controlled tokenizer speed benchmark. `Encode M tok/s` divides token count by accumulated encoding time across workers (↑ token throughput under matched conditions). `Document-pass wall s` is elapsed time for the document pass (↓ time), including document I/O, roundtrip and boundary accounting but excluding the later isolated-word, coverage and morphology computations; parallel word-sample collection also precedes this timer. The raw field remains `metric_wall_seconds`. The two time measures have different denominators. Worker/thread settings and vocabulary usage are descriptive; different work allocations prevent like-for-like speed rankings.", "",
        "| Vocab | Tokenizer | Encode M tok/s | Encoding s (summed) | Evaluator wall s | Workers × threads | Used vocab % | Roundtrip failures |",
        "|---:|---|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        settings = (f'{r["metric_workers"]} × {r["tokenizer_threads_per_worker"]}'
                    if r["metric_workers"] is not None else "not recorded")
        lines.append(f'| {r["vocab_size"]:,} | {label(r)} | {r["encode_tokens_per_second"]/1e6:.3f} | {r["encode_seconds_accumulated"]:.2f} | {fmt(r["metric_wall_seconds"], 2)} | {settings} | {100*r["vocab_used_rate"]:.2f} | {r["roundtrip_failures"]:,} |')
    return lines


def source_table(rows: list[dict]) -> list[str]:
    lines = ["", "## Row-level source files", "",
             "Every file below is copied in full into the report JSON with its SHA-256 hash; the CSV carries the same path and hash. Original result files are preserved unchanged.", "",
             "| Vocabulary | Tokenizer | Original JSON |", "|---:|---|---|"]
    for r in rows:
        lines.append(f'| {r["vocab_size"]:,} | {label(r)} | [{Path(r["source_file"]).name}](../../{r["source_file"]}) |')
    return lines


def segmenter_archive() -> dict:
    """Preserve the earlier analyzer screen separately from final BPE artifacts."""
    screening = ROOT / "docs/tokenizer_tests/segmenter_benchmark_status.md"
    judge = ROOT / "docs/tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.scores.json"
    # Rounded observations transcribed from the archived screening document;
    # its original large JSON files are not present in this evidence bundle.
    columns = ("backend", "sample", "unique_types", "weighted_words", "elapsed_seconds",
               "types_per_second", "pieces_per_weighted_word", "split_rate_weighted",
               "fallback_rate_weighted", "fallback_rate_types")
    values = [
        ("identity", "full first shard", 10829544, 1282426655, 35.03, 309135, 1.000, 0.000, 0.000, None),
        ("TRmorph", "full first shard, pre-repair", 10829544, 1282426655, 1286.69, 8417, 1.474, 0.292, 0.390, 0.874),
        ("TRmorph", "hash-100k, corrected wrapper", 100000, 12095210, 10.84, 9223, 1.542, 0.338, 0.145, 0.817),
        ("Zemberek", "hash-1k smoke", 1000, 76484, 2.00, 500, 1.468, 0.247, 0.426, 0.250),
        ("Zemberek", "hash-100k, corrected wrapper", 100000, 12095210, 29.99, 3335, 1.447, 0.316, 0.425, 0.233),
        ("TurkishDelight", "hash-100k", 100000, 12095210, 65.29, 1532, 1.503, 0.362, 0.000, 0.000),
    ]
    return {
        "screening_source": str(screening.relative_to(ROOT)),
        "screening_source_sha256": hashlib.sha256(screening.read_bytes()).hexdigest(),
        "screening_precision": "rounded values preserved from archived Markdown; not a new benchmark",
        "screening_rows": [dict(zip(columns, row)) for row in values],
        "judge_source": str(judge.relative_to(ROOT)),
        "judge_source_sha256": hashlib.sha256(judge.read_bytes()).hexdigest(),
        "judge_original_metrics": json.loads(judge.read_text()),
    }


def normalizing_references() -> list[dict]:
    directory = ROOT / "artifacts/tokenizer_metrics_historical_normalizing_references"
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = []
    for record in manifest["files"]:
        path = directory / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Git-history reference hash differs: {path}")
        row = row_from_file(path, external=True)
        row["evaluation_scope"] = "historical_normalizing_reference_not_lossless_comparator"
        rows.append(row)
    verify_rows(rows, expected_count=3)
    return rows


def segmenter_tables(archive: dict) -> list[str]:
    lines = ["", "## Earlier morphology analyzer screening", "",
        "These measurements evaluate the upstream analyzers that supplied training boundaries. They are separate from the finished tokenizer and LLM evaluations above. Rounded screening values come from the [archived benchmark log](../tokenizer_tests/segmenter_benchmark_status.md); its large original JSON outputs are not included in this repository bundle. Unique forms is the number of distinct inputs (↔ sample size); forms/s (↑ speed) and runtime (↓ time) describe processing cost only for matched scopes. Pieces/word and split rate describe how often/how much an analyzer segments (↔, no universal optimum). Lower fallback means the analyzer handles more inputs through its analysis path; it does not prove those analyses are correct. Weighted rates count occurrences, while type rates count distinct forms.", "",
        "| Analyzer | Sample | Unique forms | Runtime s | Forms/s | Pieces/word (weighted) | Split % (weighted) | Fallback % (weighted) | Fallback % (types) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in archive["screening_rows"]:
        fallback_types = "not reported" if r["fallback_rate_types"] is None else f'{100*r["fallback_rate_types"]:.1f}'
        lines.append(f'| {r["backend"]} | {r["sample"]} | {r["unique_types"]:,} | {r["elapsed_seconds"]:.2f} | {r["types_per_second"]:,} | {r["pieces_per_weighted_word"]:.3f} | {100*r["split_rate_weighted"]:.1f} | {100*r["fallback_rate_weighted"]:.1f} | {fallback_types} |')
    lines += ["", "The common hash-100k screen represents 12,095,210 weighted word occurrences; the full first shard represents 1,282,426,655 and the 1k smoke sample 76,484. The full TRmorph row predates case-only surface repair and analysis-tag filtering; its fallback rate is not a current-wrapper result. Different wrappers and scopes prevent a single speed ranking.", "",
        "### Archived 500-item blind segmentation judge", "",
        "The [judge report](../tokenizer_tests/codex_local_judge_results.md) used a disagreement-focused sample, blind candidate labels and a local scripted rubric. Best counts/rates measure selections as the preferred candidate; acceptable counts/rates measure candidates passing the rubric. Higher is better within that rubric only. It is not a human-gold or corpus-representative score. Original [scores](../tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.scores.json) and [item judgments](../tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.judgments.jsonl) remain available.", "",
        "| Analyzer | Best count | Best % | Acceptable count | Acceptable % |",
        "|---|---:|---:|---:|---:|"]
    j = archive["judge_original_metrics"]
    for backend, name in (("trmorph", "TRmorph"), ("tdelight", "TurkishDelight"), ("zemberek", "Zemberek"), ("identity", "Identity")):
        best, acceptable = j["best_counts"].get(backend, 0), j["acceptable_counts"].get(backend, 0)
        lines.append(f'| {name} | {best} | {100*best/j["total_rows"]:.1f} | {acceptable} | {100*acceptable/j["total_rows"]:.1f} |')
    confidence = ", ".join(f'{name} {j["confidence_counts"][name]}' for name in ("high", "medium", "low"))
    lines += ["", f'Confidence counts: {confidence}; invalid judgments: {j["invalid_count"]}. Screening source hashes and the full judge summary are retained in the consolidated JSON.']
    return lines


def build_additional() -> None:
    directory = ROOT / "artifacts/tokenizer_metrics_additional_2026-11-09"
    if not directory.exists():
        return
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["files"]:
        raw = (directory / record["file"]).read_bytes()
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Additional source checksum mismatch: {record['file']}")
    scheduler = (directory / manifest["scheduler_file"]).read_bytes()
    if hashlib.sha256(scheduler).hexdigest() != manifest["scheduler_sha256"]:
        raise ValueError("Additional scheduler evidence checksum mismatch")
    sections, all_rows = {}, []
    lines = ["# Additional archived tokenizer measurements", "",
        "This supplement adds previously uncollected UHeM results: two complete twelve-tokenizer training-prefix comparisons and raw-BPE-only train/validation diagnostics used by d32 preflight. They are archived measurements recovered on 9 November 2026, not new model training or evaluation. [Original JSONs and scheduler evidence](../../artifacts/tokenizer_metrics_additional_2026-11-09/) are preserved byte-for-byte. Download [JSON](tokenizer_additional.json) or [CSV](tokenizer_additional.csv).", "",
        "The 100k, 1m and [full-corpus](tokenizer_full_corpus.md) passes overlap. Their isolated-word/alignment metrics remain capped at 500,000 occurrences and consistency at 50,000 word types; repeated word-level results are not independent replications. Every comparison uses the automatic TRmorph reference, with its associated analyzer bias."]
    for group, title in (("sample_100k", "100,000-document training prefix"), ("sample_1m", "1,000,000-document training prefix")):
        rows = [row_from_file(directory / group / f"{family}_{size}_metrics.json")
                for size in (32768, 65536, 131072) for family in FAMILIES]
        verify_rows(rows, expected_count=12)
        for row in rows:
            row["result_group"] = group
            if row["isolated_word_occurrences"] != 500000 or row["morphology_word_occurrences"] != 500000:
                raise ValueError(f"Unmatched word caps in {group}")
        sections[group] = rows
        all_rows += rows
        d, job = rows[0], manifest["groups"][group]
        if job["job_state"] != "COMPLETED" or job["exit_code"] != "0:0":
            raise ValueError(f"No successful completion evidence for {group}")
        lines += ["", f"## {title}", "",
            f'Job **{job["job_id"]}**: COMPLETED, exit 0:0, elapsed {job["elapsed"]}. Every row covers {d["documents"]:,} training documents, {d["raw_bytes"]:,} UTF-8 bytes and {d["corpus_words"]:,} regex words. All use 32 metric workers and one tokenizer thread per worker.', "",
            "| Vocab | Tokenizer | Tokens/word ↓ | Isolated tokens/word ↓ | One-token words % | Bytes/token ↑ | Exact morph sequence % | Boundaries preserved % | Morph edit ↓ | Consistency F1 ↑ |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f'| {r["vocab_size"]:,} | {label(r)} | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {r["bytes_per_token"]:.4f} | {100*r["exact_morpheme_sequence_rate"]:.2f} | {100*r["boundary_preservation_rate"]:.2f} | {r["morphological_edit_distance"]:.4f} | {r["morphological_consistency_f1"]:.4f} |')
        lines += ["", "| Vocab | Tokenizer | Normalized morph edit | Consistency P | Consistency R | F1 resample SD |", "|---:|---|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f'| {r["vocab_size"]:,} | {label(r)} | {r["morphological_edit_distance_normalized"]:.4f} | {r["morphological_consistency_precision"]:.4f} | {r["morphological_consistency_recall"]:.4f} | {r["morphological_consistency_f1_resample_std"]:.4f} |')
        # Scope headings one level below the corresponding document-cap section.
        lines += [line.replace("## Archived", "### Archived") for line in systems_table(rows)]
    baseline = [row_from_file(directory / "d32_preflight" / f"bpe_32768_{suffix}_metrics.json")
                for suffix in ("train_raw", "val")]
    for r in baseline:
        verify_rows([r], expected_count=1)
        r["result_group"] = "d32_preflight"
    sections["d32_preflight"] = baseline
    all_rows += baseline
    lines += ["", "## Raw BPE32k baseline train/validation diagnostics", "",
        "The d32 preflight records name the original `nanochat-turk-d20-bpe32k/tokenizers/bpe_32768` tokenizer path. They evaluate raw text with no morphology reference; boundary and morphology fields are not applicable. These are tokenizer-only diagnostics and do not add a new model cell or matched multi-tokenizer validation comparison. The full-train token/byte counts match the recovered full-corpus BPE32k row exactly.", "",
        "| Split | Documents | UTF-8 bytes | Regex words | Tokens/word | Isolated tokens/word | One-token words % | Bytes/token | Roundtrip failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in baseline:
        lines.append(f'| {r["split"]} | {r["documents"]:,} | {r["raw_bytes"]:,} | {r["corpus_words"]:,} | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {r["bytes_per_token"]:.4f} | {r["roundtrip_failures"]} |')
    lines += ["", "Both use 500,000 isolated word occurrences. Training uses 112 workers × 1 thread; validation uses 56 × 1. Their timing is separate from the twelve-tokenizer prefix comparisons.", "",
        "| Split | Encode M tok/s | Encoding s (summed) | Evaluator wall s | Used vocab % |",
        "|---|---:|---:|---:|---:|"]
    for r in baseline:
        lines.append(f'| {r["split"]} | {r["encode_tokens_per_second"]/1e6:.3f} | {r["encode_seconds_accumulated"]:.2f} | {r["metric_wall_seconds"]:.2f} | {100*r["vocab_used_rate"]:.2f} |')
    lines += ["", "## Source index", "", "| Scope | Tokenizer | Original JSON |", "|---|---|---|"]
    for r in all_rows:
        scope = r["result_group"] + (" / " + r["split"] if r["result_group"] == "d32_preflight" else "")
        lines.append(f'| {scope} | {r["tokenizer"]} | [{Path(r["source_file"]).name}](../../{r["source_file"]}) |')
    lines += ["", "Intact-morpheme-token preservation, character-only coverage and literal boundary P/R/F1 were absent from these archives. They are now measured separately for all twelve local tokenizers in the [completed matched train/validation evaluation](tokenizer_fresh_results.md); historical missing values remain null. All original distributions, vocabulary shape, source paths, timing fields and hashes are preserved in JSON. Rebuild with `python -m scripts.build_tokenizer_study_results`."]
    payload = {"schema_version": 1, "evidence_status": "recovered_archived_measurements", "report_date": "2026-11-09",
               "manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "groups": sections}
    (OUT / "tokenizer_additional.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (OUT / "tokenizer_additional.md").write_text("\n".join(explain_metric_directions(lines, scope="additional")) + "\n")
    columns = [k for k in all_rows[0] if k != "original_metrics"]
    with (OUT / "tokenizer_additional.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in all_rows)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row_from_file(SOURCE / f"{f}_{v}_metrics.json")
            for v in (32768, 65536, 131072) for f in FAMILIES]
    external = [row_from_file(SOURCE / f"{n}_metrics.json", external=True)
                for n in ("kumru_2b", "cosmos_turkish_gpt2")]
    raw_diagnostic = row_from_file(SOURCE / "morphbpe_zemberek_32768_raw_metrics.json")
    analyzers = segmenter_archive()
    normalizing = normalizing_references()
    verify_rows(rows + external, expected_count=14)
    verify_rows([raw_diagnostic], expected_count=1)
    payload = {
        "schema_version": 2, "report_date": "2026-11-09",
        "evidence_status": "archived_measurements_not_a_new_evaluation",
        "notes": [
            "The main and external rows use the historical 50,000-document TRAIN corpus sample, not held-out test data. A separate raw-text diagnostic uses 10,000 documents.",
            "Word-level caps differ: 200,000 for three 32k rows and 500,000 for the others.",
            "TRmorph generates the morphology reference; it is not independent human annotation.",
            "Boundary preservation is derived from archived integer counts, not a new run.",
            "Exact morpheme-sequence rate and boundary preservation are distinct from intact-morpheme-token rate.",
            "Morphological consistency is the original shared-token/shared-morpheme pair metric, not boundary F1.",
            "Missing character-only, intact-morpheme and boundary precision measurements remain null.",
            "The 12 local BPE tokenizers have byte fallback and no UNK mechanism; UNK is N/A, not an empirical zero.",
            "Encoding throughput is historical and settings differ; do not treat it as a controlled speed benchmark.",
        ], "tokenizers": rows, "external_references": external,
        "separate_raw_text_diagnostics": [raw_diagnostic],
        "upstream_analyzer_archive": analyzers,
        "historical_normalizing_references": normalizing,
    }
    (OUT / "tokenizer_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    columns = [k for k in rows[0] if k != "original_metrics"]
    with (OUT / "tokenizer_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in rows + external + [raw_diagnostic] + normalizing)
    lines = ["# Archived tokenizer and morphology screening results", "", "Archived measurements, consolidated 9 November 2026. This report retains the earlier twelve-tokenizer sample, two lossless external references, three historical normalizing references, a separate raw-text diagnostic and upstream analyzer screening. Full precision, original metrics and SHA-256 source hashes are in [JSON](tokenizer_results.json); flat tokenizer values are in [CSV](tokenizer_results.csv). For the subsequently recovered complete training-corpus pass, use the [full-corpus report](tokenizer_full_corpus.md).", "",
        "## Historical 50,000-document intrinsic comparison", "", "**Scope:** 50,000 FineWeb-2 Turkish **training** documents, 186,328,421 bytes and 23,042,031 regex-defined words. These are corpus diagnostics, not held-out generalization results. Values below are percentages where marked `%`.", "",
        "| Vocabulary | Tokenizer | Tokens/word ↓ | Isolated tokens/word ↓ | Whole word: one token % | Exact morpheme sequence % ↑ | Preserved boundaries % ↑ | Morph consistency F1 ↑ | Word sample |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        f = r["tokenizer"].rsplit("_", 1)[0]
        lines.append(f'| {r["vocab_size"]:,} | {LABELS[f]} | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {100*r["exact_morpheme_sequence_rate"]:.2f} | {100*r["boundary_preservation_rate"]:.2f} | {r["morphological_consistency_f1"]:.4f} | {r["isolated_word_occurrences"]:,} |')
    lines += ["", "### Definitions and coverage", "",
        "- **Vocabulary size (↔):** actual vocabulary including nine special tokens for every local tokenizer. Larger vocabularies can shorten sequences but increase embedding/output table size; vocabulary use is the fraction of IDs seen in the sample (↔ scope-dependent diagnostic).",
        "- **Subword efficiency (↓ compression):** total encoded document tokens divided by regex word count (`\\w+`). This includes punctuation/spacing tokens in the numerator. Isolated fertility encodes each sampled word separately and is reported alongside it. **Bytes/token (↑ compression)** is original UTF-8 bytes divided by encoded document tokens.",
        "- **Whole-word coverage (↑ isolated-word compression only):** fraction of sampled word occurrences represented by exactly one token when encoded in isolation. It is sensitive to whitespace conventions and differs from context-aware word coverage; it is not a morphology-quality score.",
        "- **Morpheme preservation (↑ automatic-reference agreement):** the archived **exact morpheme-sequence rate** counts words whose token pieces exactly match the automatic morpheme sequence. It is a strict word-level proxy. Individual-morpheme intact-token preservation was absent from these June measurements and is now separately measured in the [matched evaluation](tokenizer_fresh_results.md).",
        "- **Boundary preservation (↑ reference agreement):** `(reference boundaries - crossed boundaries) / reference boundaries`, using all 12,706,795 reference boundaries. The complementary crossed-boundary rate is better when lower. Neither penalizes extra token splits, so neither is boundary precision or F1.",
        "- **Morphological consistency (↑ reference agreement):** retain the original MorphBPE shared-token/shared-morpheme pair precision, recall and F1 (`k=100`, `C=50`, `N=10`). It must not be relabelled as boundary F1. **Morph edit / normalized edit (↓ reference disagreement)** is token-piece versus reference-piece sequence edit distance, averaged over word occurrences; the normalized version first divides each distance by its reference morpheme count. Explicit boundary precision and F1 were not measured here.",
        "- **Character-only coverage (↔ diagnostic):** fraction of words split entirely into single Unicode-character tokens, including single-character words; it was not measured in the archived outputs. A UTF-8 byte fragment is not a character token.",
        "- **UNK (↓ when applicable):** N/A for the twelve local byte-complete BPE/MorphBPE tokenizers, which have no UNK mechanism. Historical UNK counts were not recorded for the external references and remain unmeasured.",
        "- **Roundtrip failures (↓; zero ideal):** documents whose decoded encoding differs from the original. All twelve local tokenizers had zero failures in 50,000 documents (100% exact decode/encode success). This establishes observed representability, not whole-word compression or morphology quality.",
        "", "## Previously reported metrics", "",
        "| Vocabulary | Tokenizer | Bytes/token ↑ | Morph edit distance ↓ | Normalized edit ↓ | Consistency P ↑ | Consistency R ↑ | Boundary crossed % ↓ | Used vocab % | Encode M tokens/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        f = r["tokenizer"].rsplit("_", 1)[0]
        lines.append(f'| {r["vocab_size"]:,} | {LABELS[f]} | {r["bytes_per_token"]:.4f} | {r["morphological_edit_distance"]:.4f} | {r["morphological_edit_distance_normalized"]:.4f} | {r["morphological_consistency_precision"]:.4f} | {r["morphological_consistency_recall"]:.4f} | {100*r["crossed_boundary_rate"]:.2f} | {100*r["vocab_used_rate"]:.2f} | {r["encode_tokens_per_second"]/1e6:.3f} |')
    lines += ["", "The JSON also preserves all original distributions, unique-word measurements, resampling standard deviations, vocabulary shape, worker/thread counts and timing. The original [comparison](../tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md) remains available.", "",
        "## External tokenizer references", "", "These are tokenizer-only diagnostics; their pretrained models are **not** part of the controlled LLM ablation. Kumru (`vngrs-ai/Kumru-2B`) and Cosmos (`ytu-ce-cosmos/turkish-gpt2`) use the same 50,000 documents, 200,000 isolated/alignment word occurrences and 50,000 consistency word types. The archived external-reference job is 494176. Their vocabularies and runtime settings differ from the local tokenizers.", "",
        "| Tokenizer | Vocabulary | Tokens/word | Isolated tokens/word | Whole word: one token % | Exact morph sequence % | Boundaries preserved % | Morph consistency F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in external:
        lines.append(f'| {r["tokenizer"]} | {r["vocab_size"]:,} | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {100*r["exact_morpheme_sequence_rate"]:.2f} | {100*r["boundary_preservation_rate"]:.2f} | {r["morphological_consistency_f1"]:.4f} |')
    lines += ["", "| Tokenizer | Bytes/token | Morph edit ↓ | Normalized edit ↓ | Consistency P | Consistency R | Crossed boundaries % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in external:
        lines.append(f'| {r["tokenizer"]} | {r["bytes_per_token"]:.4f} | {r["morphological_edit_distance"]:.4f} | {r["morphological_edit_distance_normalized"]:.4f} | {r["morphological_consistency_precision"]:.4f} | {r["morphological_consistency_recall"]:.4f} | {100*r["crossed_boundary_rate"]:.2f} |')
    lines += systems_table(rows + external)
    lines += ["", "## Earlier normalizing-tokenizer references from Git history", "",
        "BERTurk, TURNA and VBART were published in [commit 2ff5305](https://github.com/nurcunal/nanochat-turk/commit/2ff53058540b5d2d00a9e832e7f02a19582faed4), then removed when the primary comparison narrowed to lossless raw-text BPE. The [three original files and immutable source links](../../artifacts/tokenizer_metrics_historical_normalizing_references/) are restored separately for completeness. All use the same 50,000 training documents and 500,000 isolated word occurrences.", "",
        "**These are not lossless-tokenizer comparators.** Their observed roundtrip failures reflect normalization/decoding behavior; UNK counts were not recorded. A word producing one token need not preserve that word. Thus the recorded token counts below do not establish equivalent compression, coverage or morphology quality. No model-weight evaluation was performed for these references.", "",
        "| Tokenizer | Vocab | Recorded tokens/word | Isolated tokens/word | Recorded one-token words % | Bytes/token | Roundtrip failures | Failure % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in normalizing:
        lines.append(f'| {r["tokenizer"]} | {r["vocab_size"]:,} | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {r["bytes_per_token"]:.4f} | {r["roundtrip_failures"]:,} | {100*(1-r["roundtrip_success_rate"]):.3f} |')
    lines += ["", "| Tokenizer | Recorded boundary crossed % | Encode M tok/s | Used vocab % |", "|---|---:|---:|---:|"]
    for r in normalizing:
        lines.append(f'| {r["tokenizer"]} | {100*r["crossed_boundary_rate"]:.2f} | {r["encode_tokens_per_second"]/1e6:.3f} | {100*r["vocab_used_rate"]:.2f} |')
    lines += ["", "Boundary-crossing fields are preserved as historical diagnostics, not used in the lossless morphology ranking. Morphological edit/consistency measurements and worker/thread settings were absent. Original IDs: `dbmdz/bert-base-turkish-cased`, `boun-tabi-LMG/TURNA`, `vngrs-ai/VBART-Large-Base`."]
    r = raw_diagnostic
    lines += ["", "## Separate raw-text diagnostic", "",
        "The earlier Zemberek 32k diagnostic uses **10,000 raw training documents**, 48,975,401 bytes and 6,077,942 words. It has no morphology reference; zero boundary counts in its original file mean **not applicable**, not perfect boundary preservation. Its 200,000-word isolated sample is distinct from the corpus denominator. Runtime worker/thread settings were not recorded.", "",
        "| Tokenizer | Tokens/word | Isolated tokens/word | One-token words % | Bytes/token | Encode M tok/s | Roundtrip failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f'| {label(r)} 32k | {r["tokens_per_word"]:.4f} | {r["isolated_tokens_per_word"]:.4f} | {100*r["whole_word_single_token_rate"]:.2f} | {r["bytes_per_token"]:.4f} | {r["encode_tokens_per_second"]/1e6:.3f} | {r["roundtrip_failures"]} |']
    lines += segmenter_tables(analyzers)
    lines += source_table(rows + external + [raw_diagnostic] + normalizing)
    lines += ["", "## Interpretation and limits", "",
        "At every vocabulary size, MorphBPE preserves more TRmorph reference boundaries and has better recorded morphology edit distance and consistency than raw BPE, while requiring more document tokens per word. TRmorph training and evaluation share an analyzer, so these are automatic-reference results. The 32k morphology and isolated-word caps differ across treatments, preventing a fully matched word-level estimate. Corpus-level fertility and boundary counts use the common 50,000 documents.", "",
        "Isolated-word fertility often improves with MorphBPE even though document tokens/word rises; the two measures capture different contexts. Job 496882 has now been verified COMPLETED and recovered separately in the [full-corpus report](tokenizer_full_corpus.md). The [completed matched evaluation](tokenizer_fresh_results.md) adds separate training/validation results and the previously absent coverage/boundary metrics for all twelve local tokenizers; no historical null is overwritten. No significance or causal advantage is claimed from these single-run diagnostics.", "",
        "Rebuild: `python -m scripts.build_tokenizer_study_results`."]
    (OUT / "tokenizer_results.md").write_text("\n".join(explain_metric_directions(lines, scope="archived_sample", archived_rows=rows)) + "\n")

    full_dir = ROOT / "artifacts/tokenizer_metrics_full_2026-06-23"
    if full_dir.exists():
        manifest = json.loads((full_dir / "manifest.json").read_text())
        if manifest["job_state"] != "COMPLETED" or manifest["exit_code"] != "0:0":
            raise ValueError("Full-corpus recovery manifest does not establish successful completion")
        for record in manifest["files"]:
            raw = (full_dir / record["file"]).read_bytes()
            if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Recovered source checksum mismatch: {record['file']}")
        full = [row_from_file(full_dir / f"{f}_{v}_metrics.json")
                for v in (32768, 65536, 131072) for f in FAMILIES]
        verify_rows(full, expected_count=12)
        if not all(r["isolated_word_occurrences"] == r["morphology_word_occurrences"] == 500000 for r in full):
            raise ValueError("Full-corpus word metric caps differ from 500,000")
        result = {"schema_version": 2, "job_id": 496882, "job_state": "COMPLETED", "tokenizers": full,
                  "manifest": str((full_dir / "manifest.json").relative_to(ROOT)),
                  "manifest_sha256": hashlib.sha256((full_dir / "manifest.json").read_bytes()).hexdigest(),
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
        lines += ["", "All twelve rows record zero roundtrip failures. Per-morpheme intact-token preservation, character-only coverage and explicit boundary P/R/F1 were absent from these June outputs; they are now separately measured in the [completed matched train/validation evaluation](tokenizer_fresh_results.md), while the historical missing fields remain null. The full-corpus word caps resolve the earlier 32k 200k/500k cap mismatch, while retaining the automatic TRmorph-reference limitation. The repeated consistency values are expected because this metric remains capped at the same 50,000 word types; they are not independent replications.", "",
            "## Additional morphology diagnostics", "",
            "Morph consistency is shared-token/shared-morpheme pair precision/recall/F1. `F1 resample SD` is variability over the ten sampled pair sets, not a confidence interval or across-training-seed uncertainty. Boundary preservation is 1 minus crossed-boundary rate and does not penalize extra splits.", "",
            "| Vocab | Tokenizer | Normalized morph edit ↓ | Consistency P | Consistency R | F1 resample SD | Crossed boundaries % |",
            "|---:|---|---:|---:|---:|---:|---:|"]
        for r in full:
            lines.append(f'| {r["vocab_size"]:,} | {label(r)} | {r["morphological_edit_distance_normalized"]:.4f} | {r["morphological_consistency_precision"]:.4f} | {r["morphological_consistency_recall"]:.4f} | {r["morphological_consistency_f1_resample_std"]:.4f} | {100*r["crossed_boundary_rate"]:.2f} |')
        lines += systems_table(full)
        lines += source_table(full)
        lines += ["", "## Interpretation and reproduction", "",
            "The recovered generated comparison uses a diagnostic sorting order, not an official paper ranking. Compare each treatment with raw BPE of the same vocabulary size; avoid a single synthetic overall winner. All nine MorphBPE variants preserve more automatic reference boundaries and improve recorded morphology edit/consistency diagnostics while using more corpus tokens per word than same-size raw BPE. These training-corpus diagnostics do not establish a downstream model advantage.", "",
            "The report JSON retains original distributions, unique-word metrics, vocabulary shape, elapsed times and actual consistency clusters/pairs. Rebuild with `python -m scripts.build_tokenizer_study_results`; the builder verifies all recovered source hashes, sample counts and derived ratios before producing tables."]
        (OUT / "tokenizer_full_corpus.md").write_text("\n".join(explain_metric_directions(lines, scope="full")) + "\n")
    build_additional()


if __name__ == "__main__":
    build()
