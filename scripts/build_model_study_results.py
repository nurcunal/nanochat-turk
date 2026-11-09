"""Rebuild presentation tables from the archived tokenizer-ablation evidence.

Run from the repository root: python -m scripts.build_model_study_results
Only the standard library is required. This does not rerun model evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/study_results"
FINAL = "artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/metrics_summary.json"
EARLIER = "artifacts/cetvel_core12_model_comparison_2026-06-12/metrics_summary.json"
BASELINE = "artifacts/cetvel_base_subset_2026-06-09_job493293/metrics_summary.json"
SMOKE = "artifacts/uhem_smoke_2026-06-07_job492393/metrics.json"
TIERS = {"32k": (32768, 20, 896.5), "64k": (65536, 16, 872.4), "128k": (131072, 12, 890.2)}


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build():
    source, earlier, baseline, smoke = map(read_json, (FINAL, EARLIER, BASELINE, SMOKE))
    runs = []
    task_rows = []
    tasks = {t["task"]: t for t in source["task_metrics"]}
    baselines = {r["vocab"]: r for r in source["runs"] if r["segmenter"] == "none"}
    baseline_tasks = {t["task"]: t for t in baseline}
    for task in source["task_metrics"]:
        raw_value = baseline_tasks[task["task"]]["metrics"][task["metric"]]
        assert round(raw_value, 4) == task["values"]["raw_bpe_d20"]
    historical_checks = []
    for old in earlier["runs"]:
        new = next(r for r in source["runs"] if r["run_id"] == old["run_id"])
        fields = ("validation_bpb", "lowest_validation_bpb", "final_train_loss", "macro_core11", "xquad_tr_f1")
        assert all(round(old[f], 4) == new[f] for f in fields), old["run_id"]
        for task in earlier["task_metrics"]:
            assert task[old["run_id"]] == tasks[task["task"]]["values"][old["run_id"]]
        historical_checks.append({"run_id": old["run_id"], "summaries_match_at_four_decimals_and_task_values_unchanged": True})
    for original in source["runs"]:
        row = dict(original)
        vocab_size, depth, params = TIERS[row["vocab"]]
        reference = baselines[row["vocab"]]
        values = [tasks[t]["values"][row["run_id"]] for t in source["macro_core11_tasks"]]
        reconstructed = mean(values)
        discrepancy = reconstructed - row["macro_core11"]
        # Source task values AND macros were rounded independently to 4 decimals.
        # Their difference can be as large as 0.0001 without a contradiction.
        assert len(values) == 11 and abs(discrepancy) <= 0.0001
        assert row["model_step"] == 17100
        assert row["validation_bpb"] == row["lowest_validation_bpb"]
        assert abs(row["core12_examples_per_second"] - 39441 / row["core12_elapsed_seconds"]) <= 0.0001
        assert row["xquad_tr_f1"] == tasks["xquad_tr"]["values"][row["run_id"]]
        row.update(
            status="completed_base_model_with_archived_core12_results",
            vocab_size=vocab_size,
            transformer_depth=depth,
            approximate_parameters_millions=params,
            training_token_positions=17100 * 1048576,
            delta_validation_bpb_vs_tier_bpe=round(row["validation_bpb"] - reference["validation_bpb"], 4),
            delta_macro_core11_percentage_points=round(100 * row["delta_macro_core11_vs_vocab_raw_bpe"], 4),
            reported_macro_audit={
                "mean_of_archived_rounded_task_values": reconstructed,
                "difference_from_archived_macro": discrepancy,
                "compatible_with_independent_four_decimal_rounding": True,
            },
            source_artifact=FINAL,
        )
        old = next((r for r in earlier["runs"] if r["run_id"] == row["run_id"]), None)
        if old is not None:
            row["higher_precision_pretraining_archive"] = {
                "source_artifact": EARLIER,
                **{f: old[f] for f in ("validation_bpb", "lowest_validation_bpb", "final_train_loss")},
            }
        runs.append(row)
        for task in source["task_metrics"]:
            value = task["values"][row["run_id"]]
            task_rows.append({
                "tokenizer": row["tokenizer"], "vocab": row["vocab"], "depth": row["depth"],
                "task": task["task"], "metric": task["metric"], "value": value,
                "scale": "0-100" if task["task"] == "xquad_tr" else "0-1",
                "delta_vs_tier_bpe_from_rounded_values": round(value - task["values"][reference["run_id"]], 4),
                "source_artifact": FINAL,
            })
    missing = {
        "tokenizer": "morphbpe_tdelight_32768", "vocab": "32k", "vocab_size": 32768,
        "depth": "d20", "transformer_depth": 20, "segmenter": "TurkishDelightNLP",
        "status": "tokenizer_only_no_full_model_checkpoint_in_archived_inventory",
        "model_step": None, "validation_bpb": None, "lowest_validation_bpb": None,
        "final_train_loss": None, "macro_core11": None, "xquad_tr_f1": None,
        "source_artifact": "docs/model_bpb_inventory.md",
    }
    runs.insert(3, missing)
    paths = [FINAL, EARLIER, BASELINE, SMOKE, "docs/tokenizer_ablation_plan.md", "docs/model_bpb_inventory.md"]
    result = {
        "schema_version": 1,
        "report_date": "2026-11-09",
        "evidence_cutoff": source["generated_at_utc"],
        "description": "Historical pretraining/CETVEL scores plus recovered UHeM metadata, full-precision metrics and performance logs; regeneration is not a new experiment.",
        "sources": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in paths],
        "metric_definitions": {
            "validation_bpb": "Sum of non-special target-token negative log probabilities in nats / (ln(2) * represented UTF-8 bytes); lower is better.",
            "final_train_loss": "Last printed token-level training loss at step 17099/17100; vocabulary-dependent, not directly comparable across tokenizers.",
            "macro_core11": "Unweighted mean of 11 selected classification/log-likelihood task scores, with acc_norm preferred except XCOPA acc; scale 0-1, higher is better.",
            "xquad_tr_f1": "XQuAD token-overlap F1 on 0-100 scale; separate from the core-11 mean.",
            "core12_examples_per_second": "39441 expanded effective examples divided by logged end-to-end core-12 elapsed seconds; hardware/harness throughput proxy, not isolated tokenization speed.",
        },
        "controls": {
            "design_source": "docs/tokenizer_ablation_plan.md",
            "matched_within_tier": ["vocabulary size", "model shape", "FineWeb-2 Turkish source/manifest/document order", "token-position budget", "batch schedule", "training recipe and evaluation harness"],
            "token_positions_per_step": 1048576, "optimizer_steps": 17100,
            "total_training_token_positions": 17930649600,
            "tier_parameters": [{"vocab": k, "vocab_size": v[0], "depth": v[1], "approximate_parameters_millions": v[2]} for k, v in TIERS.items()],
            "hardware_recipe": "4 x A100, launcher profile; all eleven recovered training log excerpts identify NVIDIA A100-SXM4-80GB",
            "context_recipe": "2048 token positions, launcher/default recipe; not equal characters of context",
            "seed_evidence": "Source initialization uses torch seed 42; individual archived run summaries do not contain a complete seed/config manifest.",
            "audit_scope": "Controls are the documented design/launcher recipe; recovered model and user configurations verify ten final metadata records. The original raw-BPE d20 final metadata path was absent at recovery. All eleven training logs were recovered.",
        },
        "limitations": [
            "Cross-vocabulary comparisons change depth and approximate parameter count; tokenizer effects should be compared within a tier.",
            "Fixed token positions do not fix raw bytes/documents consumed. Per-run raw exposure accounting is absent from the compact summaries.",
            "Validation uses a fixed token-position budget; byte normalization does not prove identical raw-text evaluation spans across tokenizers.",
            "Per-token training loss is vocabulary-dependent; use BPB as the loss comparison, subject to evaluation-span caveats.",
            "Single runs without seed replications, confidence intervals or paired significance tests; small task differences are descriptive.",
            "Base-model CETVEL results do not establish instruction-following or post-SFT assistant quality.",
            "Archived BPB, macro and task cells are rounded to four decimals; recomputing a macro from rounded task cells may differ in the last digit.",
            "The 128k TurkishDelight evaluation result was complete while its evaluation job was finishing uploads at the original import; the recovered complete training log is a distinct job.",
        ],
        "audit": {"completed_models": 11, "missing_models": 1, "tasks_per_completed_model": 12, "baseline_high_precision_task_values_match_at_four_decimals": True, "historical_32k_checks": historical_checks},
        "macro_core11_tasks": source["macro_core11_tasks"],
        "runs": runs,
        "task_metrics": source["task_metrics"],
        "baseline_only_archive": {"source_artifact": BASELINE, "results": baseline},
        "pipeline_smoke_archive": {"source_artifact": SMOKE, "result": smoke, "in_controlled_comparison": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts.model_study_recovery import enrich
    recovered_sections = enrich(ROOT, OUT, result)
    (OUT / "model_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (OUT / "model_results.csv").write_text(csv_text(runs, ["tokenizer", "vocab_size", "depth", "status", "model_step", "approximate_parameters_millions", "training_token_positions", "validation_bpb", "lowest_validation_bpb", "final_train_loss", "macro_core11", "delta_macro_core11_percentage_points", "xquad_tr_f1", "delta_xquad_tr_f1_vs_vocab_raw_bpe", "core12_elapsed_seconds", "core12_examples_per_second", "speed_vs_vocab_raw_bpe", "cetvel_job_id", "source_meta_path", "source_results_path"]))
    (OUT / "model_task_metrics.csv").write_text(csv_text(task_rows, list(task_rows[0])))
    (OUT / "model_results.md").write_text(markdown(result) + recovered_sections)
    print(f"Verified 11 models × 12 tasks; wrote model_results.json/.md/.csv and model_task_metrics.csv to {OUT}")


def markdown(data):
    lines = [
        "# Pretraining and model benchmarks",
        "",
        "**The tokenizer effect is task-dependent.** Raw BPE has the lowest archived validation BPB in all three tiers. Zemberek MorphBPE improves the 32k core-11 mean by **1.04 percentage points**; raw BPE leads that mean at 64k and 128k. MorphBPE improves XQuAD in selected tiers. These are single-run base-model results, not evidence of a universal MorphBPE advantage.",
        "",
        "Historical tables below preserve results imported 2026-06-22; a 9 November 2026 recovery adds full-precision metrics, checkpoint settings and training performance. Eleven of twelve tokenizer-model cells have results; 32k TurkishDelight has no full model in this evidence. Additional continuation and d32 results are reported separately. No new evaluation is claimed.",
        "",
        "Download [model summary CSV](model_results.csv), [all 132 task scores CSV](model_task_metrics.csv), or [full JSON with source paths and hashes](model_results.json). Original evidence: [June 22 benchmark artifact](../../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/metrics_summary.json).",
        "",
        "## Experiment controls",
        "",
        "Compare tokenizer families **within a vocabulary tier**. The documented design holds the raw FineWeb-2 Turkish source/manifest/order, model shape, optimizer recipe, token-position budget and evaluation harness fixed within that tier. Across tiers, depth changes to keep parameter counts near 0.9B.",
        "",
        "| Vocab size | Transformer depth | Approx. parameters | Steps | Token positions/step | Total token positions |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier in data["controls"]["tier_parameters"]:
        lines.append(f'| {tier["vocab_size"]:,} | {tier["depth"]} | {tier["approximate_parameters_millions"]:.1f}M | 17,100 | 1,048,576 | 17,930,649,600 |')
    lines += [
        "",
        "The launch recipe uses four A100 GPUs and 2,048-token context. The recovered section below verifies effective configuration fields for ten final metadata records. The original raw-BPE d20 final metadata path was absent. Source initialization uses seed 42; repeated training seeds are not available. See the [design](../tokenizer_ablation_plan.md) and [launch record](../tokenizer_tests/vocab64_128_uhem_launch.md).",
        "",
        "Fixed token positions do **not** mean equal raw bytes, documents, or characters of context. The compact results lack per-run exposure counts. Validation also uses a fixed token budget, so BPB normalization alone does not guarantee the exact same raw-text evaluation spans. Cross-tier results combine vocabulary allocation and model-depth effects.",
        "",
        "## Pretraining results",
        "",
        "BPB is bits per UTF-8 byte after excluding special tokens; lower is better. Final BPB equals the recorded minimum in all eleven completed runs. Training loss is the last printed token-level loss; it is preserved for completeness but cannot be compared directly across different tokenizations. All completed models end at step 17,100.",
        "",
        "| Tier | Tokenizer | Final/minimum val BPB ↓ | Δ BPB vs raw | Final token loss | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    names = {"none": "Raw BPE", "TRmorph": "MorphBPE · TRmorph", "Zemberek": "MorphBPE · Zemberek", "TurkishDelightNLP": "MorphBPE · TurkishDelight"}
    for r in data["runs"]:
        if r["model_step"] is None:
            lines.append("| 32k/d20 | MorphBPE · TurkishDelight | — | — | — | No full model in archived inventory |")
        else:
            lines.append(f'| {r["vocab"]}/{r["depth"]} | {names[r["segmenter"]]} | {r["validation_bpb"]:.4f} | {r["delta_validation_bpb_vs_tier_bpe"]:+.4f} | {r["final_train_loss"]:.4f} | Complete |')
    lines += [
        "",
        "## Common CETVEL benchmark",
        "",
        "Core-11 is the unweighted mean of eleven classification/log-likelihood task scores on a 0–1 scale. XQuAD F1 uses a 0–100 scale and is excluded from that mean. Core-11 deltas below are **percentage points**; XQuAD deltas are F1 points. Timing divides 39,441 expanded effective examples by the logged core-12 elapsed time. It is an end-to-end harness throughput proxy, not isolated tokenizer speed.",
        "",
        "| Tier | Tokenizer | Core-11 ↑ | Δ pp | XQuAD F1 ↑ | Δ F1 | Elapsed | ex/s ↑ | Speed/raw | Job |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for r in data["runs"]:
        if r["model_step"] is None:
            lines.append("| 32k/d20 | MorphBPE · TurkishDelight | — | — | — | — | — | — | — | — |")
        else:
            lines.append(f'| {r["vocab"]}/{r["depth"]} | {names[r["segmenter"]]} | {r["macro_core11"]:.4f} | {r["delta_macro_core11_percentage_points"]:+.2f} | {r["xquad_tr_f1"]:.4f} | {r["delta_xquad_tr_f1_vs_vocab_raw_bpe"]:+.4f} | {r["core12_elapsed"]} | {r["core12_examples_per_second"]:.2f} | {r["speed_vs_vocab_raw_bpe"]:.3f}× | {r["cetvel_job_id"]} |')
    lines += [
        "",
        "Legacy-generation XQuAD scores are low (about 2.25–3.48/100). A later corrected-generation raw-BPE d20 reference reaches 11.3369, reported separately below; the old scores should be interpreted with their legacy generation implementation. No confidence intervals or paired significance tests are available. The 128k TurkishDelight result file was complete at import time while Slurm was still finishing uploads; that archived job-state qualification is retained in JSON.",
    ]
    for vocab in TIERS:
        rows = [r for r in data["runs"] if r["vocab"] == vocab and r["model_step"] is not None]
        lines += ["", f"## Per-task results: {vocab}", "", "Classification metrics below use 0–1; the final XQuAD row uses 0–100. Higher is better. `acc_norm` is the benchmark's length-normalized choice score; it is not F1.", "", "| Task | Metric | " + " | ".join(names[r["segmenter"]] for r in rows) + " |", "| --- | --- | " + " | ".join("---:" for _ in rows) + " |"]
        for task in data["task_metrics"]:
            lines.append(f'| `{task["task"]}` | `{task["metric"].split(",")[0]}` | ' + " | ".join(f'{task["values"][r["run_id"]]:.4f}' for r in rows) + " |")
    lines += [
        "",
        "## Historical evidence retained",
        "",
        "The [June 12 comparison](../../artifacts/cetvel_core12_model_comparison_2026-06-12/) contains the same three 32k models. Shared summaries match the June 22 artifact at four decimal places, and all 36 task cells match exactly; these are not additional independent runs. Its higher-precision 32k BPB and train-loss values are preserved separately in JSON (raw-BPE BPB 0.6232126545832695, TRmorph 0.6265726039416546, Zemberek 0.6249791391940578).",
        "",
        "The earlier [raw-BPE-only archive](../../artifacts/cetvel_base_subset_2026-06-09_job493293/metrics_summary.json) preserves additional metrics and subtasks. The following are baseline-only and must not be used as a tokenizer-family comparison:",
        "",
        "| Baseline-only task | Metric | Value |",
        "| --- | --- | ---: |",
    ]
    for t in data["baseline_only_archive"]["results"]:
        if t["task"] in {"check_worthiness", "relevance_judgment", "tquad"}:
            for metric, value in t["metrics"].items():
                if metric in {"acc,none", "acc_norm,none", "f1,none", "exact,none"}:
                    lines.append(f'| `{t["task"]}` | `{metric.split(",")[0]}` | {value:.6f} |')
    lines += [
        "",
        "The [June 7 pipeline smoke test](../../artifacts/uhem_smoke_2026-06-07_job492393/metrics.json) is also retained: depth 2, 12,976,182 parameters, 259,522,560 training token positions, final/minimum validation BPB 1.239557, separate base-eval validation BPB 1.239951. It validates the training pipeline and is excluded from the controlled model comparison.",
        "",
        "The later [SFT configuration screens](sft_configuration_screens.md) compare post-training settings on the d32 raw-BPE model; they are separate from this matched tokenizer ablation. The completed FineWeb-2 d32 experiment and d20 continuation are reported in [additional model configurations](model_configuration_results.md), separately from this controlled tokenizer comparison and from the general-corpus d32/WSD production plan.",
        "",
        "## Reproduce and interpret",
        "",
        "Run `python -m scripts.build_model_study_results` from the repository root. The builder checks all 132 task cells, historical carry-forward consistency, final/minimum BPB equality, throughput arithmetic, and macro compatibility with four-decimal source rounding. It records SHA-256 hashes and preserves UHeM source paths in JSON.",
        "",
        "Reported macros are preserved rather than recomputed as new measurements. In particular, the 64k TurkishDelight reported macro is 0.4567; the mean of its already-rounded task cells is 0.4567545. The difference is compatible with independent rounding of the source macro and task scores. The recovered full-precision task values reproduce 0.4567 when their mean is rounded, resolving the apparent discrepancy; they remain separate from the untouched historical task cells.",
        "",
        "For a presentation: emphasize the within-tier trade-off, report BPB alongside the common benchmark, show the missing 32k TurkishDelight model explicitly, and treat small differences as descriptive. Strong causal claims require matching raw-text exposure and evaluation spans, replicated seeds, and paired statistical analysis.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    build()
