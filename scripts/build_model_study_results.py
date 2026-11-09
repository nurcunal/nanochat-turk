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
import re
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/study_results"
FINAL = "artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/metrics_summary.json"
EARLIER = "artifacts/cetvel_core12_model_comparison_2026-06-12/metrics_summary.json"
BASELINE = "artifacts/cetvel_base_subset_2026-06-09_job493293/metrics_summary.json"
SMOKE = "artifacts/uhem_smoke_2026-06-07_job492393/metrics.json"
RECOVERED = "artifacts/uhem_model_audit_2026-11-09/model-metadata-and-benchmarks.json"
LATER = "artifacts/uhem_model_audit_2026-11-09/later-model-metadata-and-benchmarks.json"
TRAIN_REPORTS = "artifacts/uhem_model_audit_2026-11-09/historical-training-reports.json"
JOB_ACCOUNTING = "artifacts/uhem_model_audit_2026-11-09/model-job-accounting.txt"
TIERS = {"32k": (32768, 20, 896.5), "64k": (65536, 16, 872.4), "128k": (131072, 12, 890.2)}


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def recovered_evidence(source):
    recovered, later_source = read_json(RECOVERED), read_json(LATER)
    tasks = {row["task"]: row for row in source["task_metrics"]}
    checked = []
    for record in recovered["records"]:
        row = next(r for r in source["runs"] if r["run_id"] == record["run_id"])
        raw = record["source_results_path"]["data"]["results"]
        for task, values in tasks.items():
            assert round(raw[task][values["metric"]], 4) == values["values"][row["run_id"]]
        original_meta = record["source_meta_path"]
        meta = original_meta.get("data", {})
        if meta:
            assert meta["step"] == row["model_step"]
            assert round(meta["val_bpb"], 4) == row["validation_bpb"]
            assert meta["total_batch_size"] == 1048576
        checked.append({
            "run_id": row["run_id"], "raw_cetvel_source_found": True,
            "source_results_sha256": record["source_results_path"]["sha256"],
            "original_checkpoint_metadata_found": original_meta["exists"],
            "checkpoint_source_status": "recovered_2026_11_09" if meta else "archived_only_original_path_absent_2026_11_09",
            "model_config": meta.get("model_config"), "user_config": meta.get("user_config"),
            "validation_bpb_full_precision": meta.get("val_bpb"),
            "core11_mean_full_precision_legacy_adapter": mean(raw[t][tasks[t]["metric"]] for t in source["macro_core11_tasks"]),
            "training_git_environment_fingerprint_present": False,
        })
    later_records = later_source["records"]

    def find(fragment):
        return next(r for r in later_records if fragment in r["path"] and "data" in r)

    launch = find("/launch/")
    models = []
    for fragment, validation_status in (("meta_034200", "continuation_diagnostic_separate_budget"), ("meta_028000", "train_overlap_not_held_out")):
        rec = find(fragment)
        meta = rec["data"]
        models.append({
            "model_tag": meta["user_config"]["model_tag"], "step": meta["step"],
            "validation_bpb": meta["val_bpb"], "lowest_validation_bpb": meta["loop_state"]["min_val_bpb"],
            "validation_status": validation_status, "depth": meta["model_config"]["n_layer"],
            "vocab_size": meta["model_config"]["vocab_size"], "total_batch_size": meta["total_batch_size"],
            "scheduled_token_positions": meta["step"] * meta["total_batch_size"],
            "in_controlled_tokenizer_ablation": False, "source_path": rec["path"], "source_sha256": rec["sha256"],
        })
    qa = []
    for fragment, label, protocol in (
        ("reference_d20_xquad_fixed_step017100", "d20 BPE32k, step 17100", "generation-fixed, CLI cap 128"),
        ("/xquad_fixed_step028000", "d32 BPE32k, step 28000", "generation-fixed, CLI cap 128"),
        ("/kumru20_qa_fixed_step028000", "d32 BPE32k, step 28000", "comparison runner, CLI cap 256"),
    ):
        rec = find(fragment)
        metric = rec["data"]["results"]["xquad_tr"]
        qa.append({"model": label, "protocol": protocol, "f1": metric["f1,none"], "exact_match": metric["exact,none"],
                   "source_path": rec["path"], "source_sha256": rec["sha256"]})
    receipt_rec = find("cetvel_eacl2026_official_scoring_receipt.json")
    receipt = receipt_rec["data"]
    assert receipt["complete"] is True
    assert abs(mean(receipt["category_scores"].values()) - receipt["overall_score"]) < 1e-12
    legacy = find("core11_legacy_step028000")["data"]["results"]
    evidence = {
        "retrieved_at_utc": recovered["retrieved_at_utc"], "source_artifact": RECOVERED,
        "all_132_archived_task_cells_match_recovered_raw_results_at_four_decimals": True,
        "original_metadata_found_count": sum(r["original_checkpoint_metadata_found"] for r in checked),
        "runs": checked,
    }
    later = {
        "schema_version": 1, "report_date": "2026-11-09", "source_artifact": LATER,
        "retrieved_at_utc": later_source["retrieved_at_utc"], "in_controlled_tokenizer_ablation": False,
        "pretraining": models, "fixed_xquad": qa,
        "d32_legacy_core11_mean": mean(legacy[t][tasks[t]["metric"]] for t in source["macro_core11_tasks"]),
        "d32_launch": launch["data"],
        "eacl_table_compatible_receipt": {"source_path": receipt_rec["path"], "source_sha256": receipt_rec["sha256"], **receipt},
        "excluded_records": [{"path": r["path"], "reason": r.get("error", "two-example smoke run, not a benchmark result")} for r in later_records if "error" in r or "smoke" in r["path"]],
        "protocol_warning": "June adapter ignored generation kwargs and did not validate exact continuation token boundaries. Fixed results do not exist for the complete eleven-model ablation; legacy model ranking remains provisional.",
    }
    return evidence, later


def recovered_training_stats(source):
    reports = {r["run_id"]: r for r in read_json(TRAIN_REPORTS)["records"]}
    accounting = (ROOT / JOB_ACCOUNTING).read_text()
    lines = accounting.splitlines()
    header = lines.index("JobID|JobName|State|ExitCode|Elapsed|Start|End|AllocCPUS|AllocTRES")
    jobs = {r["JobID"]: r for r in csv.DictReader(lines[header:], delimiter="|") if "." not in r["JobID"]}
    result = {}
    for row in source["runs"]:
        job_id = re.search(r"-(\d+)\.out$", row["source_training_log"]).group(1)
        if row["run_id"] == "raw_bpe_d20":
            text = "\n".join(lines[:header])
            fields = {"parameters": int(re.search(r"total\s+: ([\d,]+)", text).group(1).replace(",", "")),
                      "training_minutes": float(re.search(r"Total training time: ([\d.]+)m", text).group(1)),
                      "estimated_training_flops": float(re.search(r"Total training FLOPs estimate: ([\de+.]+)", text).group(1)),
                      "peak_memory_mib": float(re.search(r"Peak memory usage: ([\d.]+)MiB", text).group(1)),
                      "mfu_percent": None, "source_artifact": JOB_ACCOUNTING}
        else:
            report = next(r for r in reports[row["run_id"]]["reports"] if r["path"].endswith("base-model-training.md"))
            values = dict(line[2:].split(": ", 1) for line in report["text"].splitlines() if line.startswith("- ") and ": " in line)
            assert int(values["Calculated number of iterations"].replace(",", "")) == 17100
            assert values["model_tag"] == row["model_tag"]
            fields = {"parameters": int(values["Number of parameters"].replace(",", "")),
                      "training_minutes": float(values["Total training time"].removesuffix("m")),
                      "estimated_training_flops": float(values["Total training flops"]),
                      "peak_memory_mib": float(values["Peak memory usage"].removesuffix("MiB")),
                      "mfu_percent": float(values["MFU %"].removesuffix("%")),
                      "source_artifact": TRAIN_REPORTS, "source_report_path": report["path"], "source_report_sha256": report["sha256"]}
        fields.update(training_job_id=job_id, scheduler_state=jobs[job_id]["State"], scheduler_elapsed=jobs[job_id]["Elapsed"], gpu_count=4)
        assert fields["scheduler_state"] == "COMPLETED" and jobs[job_id]["ExitCode"] == "0:0"
        result[row["run_id"]] = fields
    return result


def build():
    source, earlier, baseline, smoke = map(read_json, (FINAL, EARLIER, BASELINE, SMOKE))
    live_audit, later = recovered_evidence(source)
    training_stats = recovered_training_stats(source)
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
            benchmark_protocol_status="legacy_adapter_known_generation_and_continuation_boundary_deviations",
        )
        row["recovery_audit"] = next(r for r in live_audit["runs"] if r["run_id"] == row["run_id"])
        row["training_statistics"] = training_stats[row["run_id"]]
        row["training_minutes"] = row["training_statistics"]["training_minutes"]
        row["total_parameters"] = row["training_statistics"]["parameters"]
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
    paths = [FINAL, EARLIER, BASELINE, SMOKE, RECOVERED, LATER, TRAIN_REPORTS, JOB_ACCOUNTING, "docs/tokenizer_ablation_plan.md", "docs/model_bpb_inventory.md"]
    result = {
        "schema_version": 1,
        "report_date": "2026-11-09",
        "evidence_cutoff": later["retrieved_at_utc"],
        "original_ablation_import": source["generated_at_utc"],
        "description": "Archived pretraining/CETVEL results audited against UHeM on 2026-11-09. Original eleven-model CETVEL values use a legacy evaluator with known deviations; later corrected results are separate.",
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
            "hardware_recipe": "4 x A100, launcher profile; per-run full hardware logs remain on UHeM",
            "context_recipe": "2048 token positions, launcher/default recipe; not equal characters of context",
            "seed_evidence": "Source initialization uses torch seed 42; individual archived run summaries do not contain a complete seed/config manifest.",
            "audit_scope": "Ten original effective model/user configurations and training reports were recovered. The original raw-BPE32k metadata is absent, but its training log survives. No complete per-run training Git/environment fingerprint is established.",
        },
        "limitations": [
            "June CETVEL ignored generation stop/max-length kwargs and did not validate exact context/continuation token boundaries. Original XQuAD claims are superseded by the correction audit; complete corrected tokenizer-ablation evaluation remains missing.",
            "Cross-vocabulary comparisons change depth and approximate parameter count; tokenizer effects should be compared within a tier.",
            "Fixed token positions do not fix raw bytes/documents consumed. Per-run raw exposure accounting is absent from the compact summaries.",
            "Validation uses a fixed token-position budget; byte normalization does not prove identical raw-text evaluation spans across tokenizers.",
            "Per-token training loss is vocabulary-dependent; use BPB as the loss comparison, subject to evaluation-span caveats.",
            "Single runs without seed replications or paired significance tests; task standard errors in recovered raw files are not confidence intervals for tokenizer differences.",
            "Base-model CETVEL results do not establish instruction-following or post-SFT assistant quality.",
            "Archived BPB, macro and task cells are rounded to four decimals; recomputing a macro from rounded task cells may differ in the last digit.",
            "The 128k TurkishDelight result was complete while Slurm was finishing uploads at import time; current scheduler state was not reverified.",
        ],
        "audit": {"completed_models": 11, "missing_models": 1, "tasks_per_completed_model": 12, "baseline_high_precision_task_values_match_at_four_decimals": True, "historical_32k_checks": historical_checks},
        "macro_core11_tasks": source["macro_core11_tasks"],
        "runs": runs,
        "task_metrics": source["task_metrics"],
        "baseline_only_archive": {"source_artifact": BASELINE, "results": baseline},
        "pipeline_smoke_archive": {"source_artifact": SMOKE, "result": smoke, "in_controlled_comparison": False},
        "uhem_recovery_audit": live_audit,
        "later_results": later,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    # The additional recovery preserves full-precision metrics, step-level
    # performance and broader configurations while retaining this report's
    # original schema and evaluator-audit fields used by the presentation.
    from scripts.model_study_recovery import enrich
    recovered_sections = enrich(ROOT, OUT, result)
    result["supplementary_reports"] = {
        "model_configurations": "model_configuration_results.md",
        "all_task_metrics": "model_all_task_metrics.csv",
        "training_performance": "model_training_performance.csv",
        "sft_configuration_screens": "sft_configuration_screens.md",
        "sft_scope": "Post-training settings on one raw-BPE d32 model; not a matched tokenizer ablation.",
    }
    later["supplementary_configuration_report"] = result["configuration_experiments"]
    (OUT / "model_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (OUT / "model_results.csv").write_text(csv_text(runs, ["tokenizer", "vocab_size", "depth", "status", "model_step", "approximate_parameters_millions", "training_token_positions", "validation_bpb", "lowest_validation_bpb", "final_train_loss", "macro_core11", "delta_macro_core11_percentage_points", "xquad_tr_f1", "delta_xquad_tr_f1_vs_vocab_raw_bpe", "core12_elapsed_seconds", "core12_examples_per_second", "speed_vs_vocab_raw_bpe", "cetvel_job_id", "source_meta_path", "source_results_path"]))
    (OUT / "model_task_metrics.csv").write_text(csv_text(task_rows, list(task_rows[0])))
    training_rows = [{"run_id": row["run_id"], "tokenizer": row["tokenizer"], "depth": row["depth"], **row["training_statistics"]} for row in runs if row["model_step"] is not None]
    training_fields = ["run_id", "tokenizer", "depth", "parameters", "training_minutes", "estimated_training_flops", "peak_memory_mib", "mfu_percent", "gpu_count", "training_job_id", "scheduler_state", "scheduler_elapsed", "source_artifact", "source_report_path", "source_report_sha256"]
    (OUT / "model_training_statistics.csv").write_text(csv_text(training_rows, training_fields))
    (OUT / "model_results.md").write_text(markdown(result) + recovered_sections)
    (OUT / "later_model_results.json").write_text(json.dumps(later, indent=2, ensure_ascii=False) + "\n")
    (OUT / "later_model_results.md").write_text(later_markdown(later))
    print(f"Verified 11 models × 12 tasks; wrote model_results.json/.md/.csv and model_task_metrics.csv to {OUT}")


def markdown(data):
    lines = [
        "# Pretraining and model benchmarks",
        "",
        "**The original model benchmark table needs a protocol qualification.** Raw BPE has the lowest archived validation BPB in all three tiers, subject to unmatched raw-text evaluation spans. The saved core-11 table puts Zemberek ahead by 1.04 percentage points at 32k, but recovered evaluator code documents generation and continuation-boundary errors. Treat those benchmark rankings as legacy measurements pending a corrected eleven-model rerun; do not present them as established tokenizer effects.",
        "",
        "Report assembled and audited against UHeM on 2026-11-09. All 132 archived task cells match recovered raw results at four decimals. Ten original step-17,100 metadata files remain; the raw-BPE 32k metadata at its original path is absent, so that BPB retains archived provenance. Eleven of twelve model cells have results; 32k TurkishDelight remains tokenizer-only. No new model training or benchmark evaluation was performed for this report.",
        "",
        "**Correction evidence:** the same d20/raw-BPE step-17,100 checkpoint scores XQuAD F1 **11.3369** with corrected generation handling, versus the archived **3.0985**. Corrected results for all MorphBPE models were not recovered. See [later model results and evaluator audit](later_model_results.md) for the separate d20 continuation, d32 training, and protocol-specific benchmarks.",
        "",
        "Download [model summary CSV](model_results.csv), [training statistics CSV](model_training_statistics.csv), [all 132 task scores CSV](model_task_metrics.csv), or [full JSON with source paths and hashes](model_results.json). Original evidence: [June 22 benchmark artifact](../../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/metrics_summary.json).",
        "",
        "Additional recovery supplies [all available task metrics](model_all_task_metrics.csv), [training throughput and memory](model_training_performance.csv), [expanded model configurations](model_configuration_results.md), and [SFT configuration screens](sft_configuration_screens.md). Those screens vary post-training settings on one raw-BPE model and are not additional controlled tokenizer cells.",
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
        "Recovered original reports/configurations document four A100 GPUs and 2,048-token context. Ten original checkpoint metadata files survive; the raw-BPE32k run is supported by its historical training log and archived metadata values. The ten original report headers record Python 3.12.4 and PyTorch 2.9.1+cu128, but their Git revision is unknown. Source initialization uses seed 42; complete per-run code/environment fingerprints and repeated-seed experiments remain absent. See the [design](../tokenizer_ablation_plan.md) and [launch record](../tokenizer_tests/vocab64_128_uhem_launch.md).",
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
        "## Recovered training cost and hardware metrics",
        "",
        "Training-loop time is separate from Slurm job elapsed time and CETVEL benchmark time. All eleven original jobs were recovered as `COMPLETED`, exit `0:0`; original training used four A100 GPUs. FLOPs and MFU are implementation estimates. Cross-tier speed also changes model depth and cannot isolate tokenizer efficiency.",
        "",
        "| Tier | Tokenizer | Parameters | Train minutes | Slurm elapsed | Estimated FLOPs | Peak MiB | MFU % |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for r in data["runs"]:
        if r["model_step"] is None:
            continue
        s = r["training_statistics"]
        mfu = "—" if s["mfu_percent"] is None else f'{s["mfu_percent"]:.2f}'
        lines.append(f'| {r["vocab"]}/{r["depth"]} | {names[r["segmenter"]]} | {s["parameters"]:,} | {s["training_minutes"]:.2f} | {s["scheduler_elapsed"]} | {s["estimated_training_flops"]:.6e} | {s["peak_memory_mib"]:.2f} | {mfu} |')
    lines += [
        "",
        "The raw-BPE32k report directory was overwritten by its later continuation. Its original 1,267.66-minute training time and memory/FLOP values come from job 492421's surviving log, not that overwritten report. The recovered excerpt does not include its final aggregate MFU, so that cell remains unavailable. Source text and job accounting are retained in the [November audit artifact](../../artifacts/uhem_model_audit_2026-11-09/).",
        "",
        "## Archived common CETVEL benchmark — legacy evaluator",
        "",
        "These original values are retained, not silently replaced with results from a different evaluator. The legacy adapter ignored generation kwargs and used a separately tokenized context length without validating the continuation boundary. Both can affect comparisons across tokenizers; the complete matrix needs a corrected rerun. Core-11 is the unweighted mean of eleven classification/log-likelihood task scores on a 0–1 scale. XQuAD F1 uses a 0–100 scale and is excluded from that mean. Core-11 deltas below are **percentage points**; XQuAD deltas are F1 points. Timing divides 39,441 expanded effective examples by the logged core-12 elapsed time.",
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
        "The low legacy XQuAD scores are affected by the generation bug and should not be used to judge absolute QA ability or tokenizer gains. Recovered raw task standard errors do not provide paired confidence intervals for model differences. The 128k TurkishDelight result file was complete at import time while Slurm was still finishing uploads; that archived job-state qualification is retained in JSON.",
    ]
    for vocab in TIERS:
        rows = [r for r in data["runs"] if r["vocab"] == vocab and r["model_step"] is not None]
        lines += ["", f"## Per-task legacy results: {vocab}", "", "Legacy adapter values. Classification metrics use 0–1; the final XQuAD row uses 0–100. `acc_norm` is the benchmark's length-normalized choice score; it is not F1. Interpretation requires the evaluator caveat above.", "", "| Task | Metric | " + " | ".join(names[r["segmenter"]] for r in rows) + " |", "| --- | --- | " + " | ".join("---:" for _ in rows) + " |"]
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
        "The repository has Atlas SFT preparation/launcher documentation, but no matched multi-tokenizer SFT result table in this evidence bundle. The d32 production work is a separate experiment and is excluded from the d20/d16/d12 ablation.",
        "",
        "## Reproduce and interpret",
        "",
        "Run `python -m scripts.build_model_study_results` from the repository root. The builder checks all 132 task cells, historical carry-forward consistency, final/minimum BPB equality, throughput arithmetic, and macro compatibility with four-decimal source rounding. It records SHA-256 hashes and preserves UHeM source paths in JSON.",
        "",
        "Reported macros are preserved rather than recomputed as new measurements. In particular, the 64k TurkishDelight reported macro is 0.4567; the mean of its already-rounded task cells is 0.4567545. The difference is compatible with independent rounding of the source macro and task scores. The recovered full-precision mean confirms the archived macro when rounded to four decimals.",
        "",
        "For a presentation: report the tokenizer diagnostics and archived BPB, label the CETVEL matrix as legacy, show the evaluator correction and missing 32k TurkishDelight model, and separate later model-scale/budget experiments. Tokenizer-effect claims require a corrected common evaluator, matched raw-text exposure/evaluation spans, replicated seeds, and paired statistical analysis.",
        "",
    ]
    return "\n".join(lines)


def later_markdown(data):
    lines = [
        "# Later model results and evaluator audit",
        "",
        "**Recovered on 2026-11-09; measured in August 2026.** These results supplement the original tokenizer-ablation table. They change model depth, training budget and/or evaluation protocol, so they are not additional controlled tokenizer cells.",
        "",
        "Sources: [compact UHeM audit](../../artifacts/uhem_model_audit_2026-11-09/), [derived JSON](later_model_results.json), and [runtime/reproduction notes](reproduction.md). Per-source paths, byte counts, timestamps and SHA-256 hashes are preserved. No model training or benchmark inference was rerun during recovery.",
        "",
        "The [expanded configuration report](model_configuration_results.md) preserves additional training-log throughput, memory, scheduler outcomes and all task-level metrics. The separate [SFT configuration screens](sft_configuration_screens.md) concern post-training choices on a single raw-BPE d32 model.",
        "",
        "## A corrected evaluator changes the interpretation",
        "",
        "The recovered [generation runner](../../artifacts/uhem_model_audit_2026-11-09/evaluator_evidence/cetvel_generation_fixed_runner.py) documents that the June adapter treated lm-eval's generation-kwargs mapping as a list of stop strings, ignoring actual `until` strings and task generation limits. The later [comparison runner](../../artifacts/uhem_model_audit_2026-11-09/evaluator_evidence/cetvel_comparison_runner.py) also moves trailing context whitespace to the continuation and checks that joint tokenization preserves an exact continuation boundary. Its code is archived as evidence; these files are not installed by the report builder.",
        "",
        "The old eleven-model benchmark values remain visible as **legacy evaluator measurements**. The generation bug invalidates the previous absolute-XQuAD interpretation; the likelihood-boundary deviation can be tokenizer-dependent, so classification rankings also need a corrected common rerun. A single corrected raw-BPE row cannot repair the complete matrix.",
        "",
        "| Model | Protocol | XQuAD F1 /100 | Exact match /100 |",
        "| --- | --- | ---: | ---: |",
        "| d20 BPE32k, step 17100 | June legacy adapter, cap 128 | 3.0985 | 0.0000 |",
    ]
    for row in data["fixed_xquad"]:
        lines.append(f'| {row["model"]} | {row["protocol"]} | {row["f1"]:.4f} | {row["exact_match"]:.4f} |')
    lines += [
        "",
        "The two d32 F1 values use distinct recorded runner/generation settings and must not be mixed. The d20-versus-d32 rows also differ in scale and training exposure; they do not isolate a tokenizer effect. The fixed d20 run records the original `tr_d20_bpe_32768_chinchilla20` step-17,100 model in its evaluator arguments; its original checkpoint metadata path is now absent from UHeM, while the evaluation record remains.",
        "",
        "## Additional pretraining",
        "",
        "| Model | Depth | Step | Scheduled token positions | Final/minimum BPB | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in data["pretraining"]:
        interpretation = "Continued d20 run; doubled budget" if row["depth"] == 20 else "Train-overlap diagnostic; not held out"
        lines.append(f'| `{row["model_tag"]}` | {row["depth"]} | {row["step"]:,} | {row["scheduled_token_positions"]:,} | {row["validation_bpb"]:.6f} | {interpretation} |')
    lines += [
        "",
        "The d20 continuation uses a distinct learning-rate continuation policy and step 34,200; it must not replace the step-17,100 baseline in the original matrix. Its final checkpoint was saved, but the later recovered Slurm record reports job 512018 FAILED (exit 1); checkpoint availability does not establish a cleanly completed job. Its training-time counter is cumulative and must not be attributed only to the continuation phase.",
        "",
        "The completed d32 model has 2,818,575,450 parameters, 2,048-token context, raw BPE32k and 58,720,256,000 scheduled positions. Its launch manifest explicitly trains on **all 30** FineWeb-2 Turkish source shards and also reuses the final source shard as validation. Consequently BPB **0.519040 is a train-overlap diagnostic**, not a held-out generalization result and not comparable as a held-out improvement over the original ablation. The launch describes the exposure schedule as a coverage-oriented heuristic, not a proof of exactly-once document/token consumption.",
        "",
        "This completed run is `tr_d32_bpe_32768_fineweb2_all_v1`, deployed from a content-addressed source snapshot based on `955c3809c94c04731bf35a9695cc0a04c28e35e9` (snapshot SHA-256 `37503db09be2617d53d3f201d2fc776f7eb96844c7115818f02f38a3e0f83a97`). It records `strict_mode=false` and `lr_schedule=nanochat_linear`. It is **not** the frozen Turkish-general WSD v4 family whose source was subsequently consolidated into main.",
        "",
        f'The d32 legacy core-11 mean is **{data["d32_legacy_core11_mean"]:.4f}**, calculated from the same eleven selected metrics, but retains the legacy continuation-scoring caveat. It is not pooled into the tokenizer-ablation matrix.',
        "",
        "## d32 CETVEL EACL table-compatible receipt",
        "",
    ]
    receipt = data["eacl_table_compatible_receipt"]
    lines += [
        f'The recovered scoring receipt reports **{receipt["overall_score"]:.4f}/100**, the unweighted mean of seven category scores, and marks the result complete. This is the archived `cetvel-eacl2026-table4-compatible-v1` protocol result for d32 step 28,000; it is not the core-11 macro and is not an independent verification against a public leaderboard.',
        "",
        "| Category | Score /100 |",
        "| --- | ---: |",
    ]
    for key in receipt["protocol"]["category_order"]:
        lines.append(f'| {key} | {receipt["category_scores"][key]:.4f} |')
    lines += [
        "",
        "The receipt pins CETVEL `6119c517e06cce23aaeac103ce3504c9380655e3`, lm-eval `6e49b1f6910931882a4b3b105794c6faf96b74e5`, and the metric provenance. Its compatibility warnings are retained: GECTurk uses stripped exact match; MCQA includes check-worthiness and relevance-judgment; summarization BERTScore uses pinned TabiBERT F1/final layer 22 with no baseline rescaling. These metrics differ from the legacy table's normalized accuracy, XQuAD F1 and ROUGE; do not compare their numerical scales directly.",
        "",
        "All task scores in that receipt are preserved below. The receipt's QA values use exact match, not the F1 values in the correction table.",
        "",
        "| Receipt task | Protocol score /100 |",
        "| --- | ---: |",
    ]
    for task, value in sorted(receipt["task_scores"].items()):
        lines.append(f'| `{task}` | {value:.4f} |')
    lines += [
        "",
        "Intermediate `kumru20_*` directories describe evaluation task suites, not a recovered Kumru model run: their model arguments identify the nanochat d32 checkpoint. Their full extracted metrics, including QA F1, GEC exact match, translation WER/BLEU and partial summarization ROUGE, remain in the source JSON; the table-compatible receipt uses its own documented aggregations.",
        "",
        "## Excluded or incomplete evidence",
        "",
        "Two-example smoke outputs are retained only as operations evidence. Their 50% XQuAD and 100% Belebele scores are not full benchmarks. The raw d20 `cetvel_out/full/cetvel_full_results.json` failed JSON parsing during recovery and is not accepted as a completed result. A `kumru20_sum_fixed` file is explicitly partial; it does not establish a completed suite by itself. The later complete table-compatible receipt is reported separately.",
        "",
        "The original ten surviving step-17,100 metadata files contain effective user/model configuration but no complete training-code/environment fingerprint. All eleven original raw CETVEL result files survive and match the archived table at published precision. Their task standard errors are retained; they do not replace seed replication or paired uncertainty estimates for tokenizer differences.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    build()
