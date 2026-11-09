"""Render publication-friendly charts from archived and fresh study summaries.

Run: MPLCONFIGDIR=/tmp/nanochat-mpl python -m scripts.plot_study_results
Requires matplotlib and numpy; no network, model inference, or GPU is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nanochat-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


from scripts.study_report_paths import ROOT, DATA_DIR, REPORT_DIR

DATA = DATA_DIR
FAMILIES = ("bpe", "trmorph", "zemberek", "tdelight")
LABELS = {"bpe": "Raw BPE", "trmorph": "TRmorph", "zemberek": "Zemberek", "tdelight": "TurkishDelight"}
COLORS = {"bpe": "#344563", "trmorph": "#168674", "zemberek": "#487CB1", "tdelight": "#CE8233"}
TIERS = ((32768, "32k", "d20", "896.5M"), (65536, "64k", "d16", "872.4M"), (131072, "128k", "d12", "890.2M"))
INK = "#14283D"
MUTED = "#526170"


def family(name):
    return "bpe" if name.startswith("bpe_") else name.split("_")[1]


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": "#C6CED6", "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "none", "svg.hashsalt": "nanochat-turk-study-2026-11-09",
        "savefig.facecolor": "white", "savefig.dpi": 220,
    })


def legend(fig, *, y, x=0.08):
    handles = [Line2D([], [], color=COLORS[k], marker="o", linestyle="none", markersize=8,
                      label=LABELS[k] if k == "bpe" else f"MorphBPE · {LABELS[k]}") for k in FAMILIES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncols=4,
               frameon=False, fontsize=10, handletextpad=0.5, columnspacing=2)


def heading(fig, title, subtitle):
    fig.text(0.055, 0.965, title, fontsize=23, fontweight="bold", va="top")
    fig.text(0.055, 0.917, subtitle, fontsize=11.5, color=MUTED, va="top")


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", metadata={"Software": "nanochat-turk study chart builder"})
    svg_path = out / f"{stem}.svg"
    fig.savefig(svg_path, metadata={"Date": "2026-11-09", "Creator": "nanochat-turk study chart builder"})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def model_comparison(data, out):
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(left=0.15, right=0.965, bottom=0.20, top=0.745, hspace=0.65, wspace=0.20)
    heading(fig, "Archived pretraining and legacy benchmarks", "Raw BPE has lower archived BPB; core-11 rankings need a corrected common evaluator.")
    legend(fig, y=0.875, x=0.047)
    for col, (vocab, label, depth, params) in enumerate(TIERS):
        rows = {family(r["tokenizer"]): r for r in data["runs"] if r["vocab_size"] == vocab}
        axs[0, col].set_title(f"{label} vocabulary · {depth}\n≈ {params} parameters", loc="left", fontsize=13, fontweight="bold", pad=15)
        for row_idx, (metric, scale, axis_label) in enumerate((
            ("validation_bpb", 1, "Validation bits per byte  ·  lower is better"),
            ("macro_core11", 100, "Legacy core-11 mean (%)  ·  higher is better"),
        )):
            ax = axs[row_idx, col]
            vals = [r[metric] * scale for r in rows.values() if r.get(metric) is not None]
            span = max(vals) - min(vals)
            pad = max(span * 0.45, 0.002 if row_idx == 0 else 0.45)
            low, high = min(vals) - pad * 0.45, max(vals) + pad * 1.8
            ax.set_xlim(low, high)
            raw = rows["bpe"][metric] * scale
            ax.axvline(raw, color=COLORS["bpe"], linestyle=(0, (3, 3)), linewidth=1, alpha=0.4, zorder=1)
            ax.set_ylim(3.55, -0.55)
            ax.set_yticks(range(4), [LABELS[k] for k in FAMILIES] if col == 0 else [])
            ax.tick_params(axis="y", length=0, pad=12)
            ax.grid(axis="x", color="#E6EBEF", linewidth=0.8, zorder=0)
            ax.set_xlabel(axis_label, fontsize=9.5, labelpad=9)
            ax.tick_params(axis="x", labelsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            for idx, key in enumerate(FAMILIES):
                r = rows.get(key)
                if r is None or r.get(metric) is None:
                    ax.text(low + (high - low) * 0.035, idx, "No full model", va="center", fontsize=10, color="#89939B", style="italic")
                    continue
                value = r[metric] * scale
                ax.plot([raw, value], [idx, idx], color=COLORS[key], linewidth=2, alpha=0.45, zorder=2)
                ax.scatter([value], [idx], color=COLORS[key], s=85, zorder=3, edgecolor="white", linewidth=0.8)
                ax.annotate(f'{value:.4f}' if row_idx == 0 else f'{value:.2f}', (value, idx), xytext=(10, 0),
                            textcoords="offset points", va="center", fontsize=10, fontweight="bold", color=COLORS[key],
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.95})
    fig.text(0.055, 0.105, "COMPARE WITHIN EACH COLUMN", fontsize=10, fontweight="bold", color=INK)
    fig.text(0.055, 0.065,
             "Depth changes across tiers. All completed runs: 17,100 steps / 17.93B token positions; equal tokens do not imply equal raw-text exposure.\n"
             "Legacy CETVEL has known continuation-boundary deviations; model rankings are provisional. Single runs; no paired uncertainty estimates.",
             fontsize=9.5, color=MUTED, linespacing=1.7, va="top")
    save(fig, out, "model_outcomes_by_tier")


def tokenizer_tradeoff(data, out):
    rows = [r for r in data["tokenizers"] if not r["external_reference"]]
    assert len(rows) == 12
    assert len({(r["documents"], r["raw_bytes"], r["corpus_words"], r["reference_boundaries"]) for r in rows}) == 1
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.27, top=0.77, wspace=0.12)
    heading(fig, "Morphology comes with a compression trade-off", "MorphBPE preserves more automatic reference boundaries while using more tokens per word.")
    legend(fig, y=0.87, x=0.047)
    for col, (vocab, label, _, _) in enumerate(TIERS):
        ax = axs[col]
        ax.set_title(f"{label} vocabulary", loc="left", fontsize=13, fontweight="bold", pad=16)
        ax.set_xlim(1.30, 2.15)
        ax.set_ylim(0, 65)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(color="#E6EBEF", linewidth=0.8, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Corpus tokens per word  ↓", labelpad=13, fontsize=11)
        if col == 0:
            ax.set_ylabel("Reference boundaries preserved (%)  ↑", labelpad=15, fontsize=11)
        else:
            ax.set_yticklabels([])
        for r in (r for r in rows if r["vocab_size"] == vocab):
            key = family(r["tokenizer"])
            x, y = r["tokens_per_word"], r["boundary_preservation_rate"] * 100
            ax.scatter([x], [y], s=125, color=COLORS[key], edgecolor="white", linewidth=1.5, zorder=3)
            label_text = f'{LABELS[key]}\n{x:.2f} / {y:.1f}%'
            ax.annotate(label_text, (x, y), xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=9.3, color=COLORS[key], linespacing=1.55, fontweight="medium")
    fig.text(0.055, 0.15, "ARCHIVED TRAIN-CORPUS DIAGNOSTIC · NOT HELD-OUT TEST PERFORMANCE", fontsize=10, fontweight="bold")
    fig.text(0.055, 0.11,
             "Both axes use the same 50,000-document sample (23.04M words). Preservation = 1 − crossed reference boundaries / reference boundaries.\n"
             "Reference: automatic TRmorph segmentation; this can favor TRmorph-trained tokenizers and is not independent human annotation.\n"
             "Separate word-level metrics in the report use 200k/500k occurrence caps; those metrics are not plotted here.",
             fontsize=9.5, color=MUTED, linespacing=1.8, va="top")
    save(fig, out, "tokenizer_efficiency_boundary_tradeoff")


def task_deltas(data, out):
    names = {"exams_tr": "EXAMS", "belebele_tr": "Belebele", "turkish_plu": "Turkish PLU", "cetvel_xcopa_tr": "XCOPA",
             "cetvel_xnli_tr": "XNLI", "mnli_tr": "MNLI", "snli_tr": "SNLI", "news_cat": "News category",
             "offenseval_tr": "OffensEval", "trclaim19": "TRClaim19", "xfact_tr": "XFact"}
    tasks = [t for t in data["task_metrics"] if t["task"] in data["macro_core11_tasks"]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 9))
    fig.subplots_adjust(left=0.145, right=0.90, top=0.80, bottom=0.205, wspace=0.13)
    heading(fig, "Legacy benchmark deltas", "Archived classification-task changes within tiers; a corrected matched rerun remains pending.")
    fig.text(0.055, 0.858, "Blue = higher score than raw BPE     Red = lower score     Values = percentage-point difference", fontsize=10.5, color=MUTED)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#F0F2F4")
    norm = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12)
    for col, (vocab, label, depth, _) in enumerate(TIERS):
        ax = axs[col]
        rows = {family(r["tokenizer"]): r for r in data["runs"] if r["vocab_size"] == vocab}
        raw_id = rows["bpe"]["run_id"]
        values = np.full((len(tasks), 3), np.nan)
        for j, key in enumerate(FAMILIES[1:]):
            r = rows.get(key)
            if r is not None and r.get("model_step") is not None:
                for i, task in enumerate(tasks):
                    values[i, j] = 100 * (task["values"][r["run_id"]] - task["values"][raw_id])
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"{label} vocabulary · {depth}", fontsize=13, fontweight="bold", pad=18)
        ax.set_xticks(range(3), ["TRmorph", "Zemberek", "TurkishDelight"], fontsize=10)
        ax.set_yticks(range(len(tasks)), [names[t["task"]] for t in tasks] if col == 0 else [])
        ax.tick_params(length=0, axis="both", pad=10)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.spines["bottom"].set_visible(False)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                text = "—" if np.isnan(value) else ("0.0" if abs(value) < 0.05 else f"{value:+.1f}")
                color = "#8D98A1" if np.isnan(value) else ("white" if abs(value) >= 7 else INK)
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10.5)
    cb_ax = fig.add_axes([0.927, 0.28, 0.014, 0.45])
    cb = fig.colorbar(im, cax=cb_ax, ticks=[-12, -6, 0, 6, 12])
    cb.outline.set_visible(False)
    cb.ax.set_ylabel("Δ percentage points", labelpad=12, fontsize=10)
    fig.text(0.055, 0.13, "LEGACY EVALUATOR · PROVISIONAL RANKINGS", fontsize=10, fontweight="bold")
    fig.text(0.055, 0.09,
             "Known continuation-boundary scoring deviation can affect tokenizer comparisons. No paired significance tests; deltas use rounded scores.\n"
             "32k TurkishDelight has no full model result (—). XQuAD uses a different F1 scale and is reported separately.\n"
             "Metric: acc_norm for all tasks except XCOPA (acc). Same-tier controls apply; depth changes across tiers.",
             fontsize=9.5, color=MUTED, linespacing=1.7, va="top")
    save(fig, out, "cetvel_task_deltas")


def evaluator_correction(data, out):
    later = data["later_results"]
    corrected = later["fixed_xquad"][0]
    original = next(r for r in data["runs"] if r.get("run_id") == "raw_bpe_d20")
    original_qa = next(r for r in data["baseline_only_archive"]["results"] if r["task"] == "xquad_tr")["metrics"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.29, top=0.73, wspace=0.25)
    heading(fig, "An evaluator fix changes the same model’s score", "d20 · raw BPE32k · checkpoint 17,100 · XQuAD · recorded CLI generation cap 128")
    for i, (metric, legacy, fixed, upper) in enumerate((
        ("F1", original["xquad_tr_f1"], corrected["f1"], 14),
        ("Exact match", original_qa["exact,none"], corrected["exact_match"], 6),
    )):
        ax = axs[i]
        ax.set_title(metric, fontsize=15, fontweight="bold", loc="left", pad=15)
        ax.barh([1, 0], [legacy, fixed], height=0.43, color=["#9AA4AF", COLORS["bpe"]], zorder=2)
        ax.set_xlim(0, upper)
        ax.set_ylim(-0.55, 1.55)
        ax.set_yticks([1, 0], ["June legacy\nadapter", "August generation\nfix"] if i == 0 else [])
        ax.tick_params(axis="y", length=0, pad=12)
        ax.grid(axis="x", color="#E6EBEF", zorder=0)
        ax.set_xlabel("Score /100", labelpad=12)
        for y, value in ((1, legacy), (0, fixed)):
            ax.text(value + upper * 0.025, y, f"{value:.4f}", va="center", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.055, 0.16, "THIS IS AN EVALUATOR CORRECTION, NOT A TOKENIZER OR TRAINING GAIN", fontsize=11, fontweight="bold")
    fig.text(0.055, 0.11,
             "The legacy generation adapter ignored task stop strings and generation limits. Corrected results for all MorphBPE models were not recovered.\n"
             "Retain the original eleven-model table as legacy evidence; a common corrected evaluation is needed before claiming tokenizer effects.\n"
             "Sources: archived June results and August reference-d20 fixed-generation run, recovered from UHeM on 9 Nov 2026.",
             fontsize=9.5, color=MUTED, linespacing=1.7, va="top")
    save(fig, out, "evaluator_correction_same_checkpoint")


def matched_validation_tokenizers(data, out):
    """Plot the measured common validation sample, never archived substitutes."""
    assert data["evidence"] == "fresh_matched_measurements"
    sample = data["splits"]["val"]
    rows = {(r["vocab_size"], family(r["tokenizer"])): r for r in sample["rows"]}
    assert len(rows) == 12
    assert set(rows) == {(vocab, key) for vocab, *_ in TIERS for key in FAMILIES}
    assert all(r["split"] == "val" and r["sample_word_occurrences"] == sample["sample_word_occurrences"]
               for r in rows.values())
    panels = (
        ("isolated_tokens_per_word", 1, "Isolated subword units per word  ↓", "Subword units / word", ".2f"),
        ("morpheme_preservation_rate", 100, "Individual morphemes preserved  ↑", "Reference morphemes (%)", ".1f"),
        ("boundary_f1", 100, "Isolated morpheme-boundary F1  ↑", "Boundary F1 (%)", ".1f"),
        ("one_token_word_rate", 100, "Words represented by one token", "Word occurrences (%)", ".1f"),
    )
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.22, top=0.785, hspace=0.55, wspace=0.20)
    heading(fig, "Twelve tokenizers on the same validation text",
            "Fresh measurements · identical words within the sample · comparison within each vocabulary tier")
    legend(fig, y=0.873, x=0.047)
    x = np.arange(3)
    width = 0.19
    for ax, (metric, scale, title, ylabel, fmt) in zip(axs.flat, panels):
        values = [r[metric] * scale for r in rows.values() if r[metric] is not None]
        upper = max(values) * 1.20 if values else 1
        ax.set_ylim(0, upper)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6EBEF", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel(ylabel, fontsize=10.5, labelpad=9)
        ax.set_xticks(x, [label for _, label, _, _ in TIERS])
        ax.set_xlabel("Vocabulary size", fontsize=10, labelpad=7)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        for i, key in enumerate(FAMILIES):
            positions = x + (i - 1.5) * width
            for pos, (vocab, *_) in zip(positions, TIERS):
                raw_value = rows[vocab, key][metric]
                if raw_value is None:
                    ax.text(pos, upper * 0.025, "N/A", ha="center", fontsize=8.5, color=MUTED)
                    continue
                value = raw_value * scale
                ax.bar(pos, value, width=width * 0.88, color=COLORS[key], zorder=3)
                ax.annotate(format(value, fmt), (pos, value), xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.7, color=COLORS[key], fontweight="bold")
    fig.text(0.055, 0.145, "HELD-OUT FILE SPLIT · AUTOMATIC MORPHOLOGY REFERENCE", fontsize=10, fontweight="bold")
    fig.text(0.055, 0.105,
             f'{sample["documents"]:,} validation documents; {sample["sample_word_occurrences"]:,} sampled word occurrences per tokenizer. '
             "Reference: automatic TRmorph, with no human annotation.\n"
             "Morpheme preservation counts exact one-token reference spans; boundary F1 measures isolated-word internal boundary agreement.\n"
             "TRmorph reference may favor TRmorph-trained tokenizers. Unalignable words are excluded from morphology denominators and reported separately.",
             fontsize=9.5, color=MUTED, linespacing=1.8, va="top")
    save(fig, out, "matched_validation_tokenizers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR / "figures")
    parser.add_argument("--fresh-only", action="store_true", help="Render only the optional fresh validation chart; retain existing chart manifest entries.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path, tok_path = DATA / "model_results.json", DATA / "tokenizer_results.json"
    fresh_path = DATA / "tokenizer_fresh_results.json"
    if args.fresh_only and not fresh_path.exists():
        parser.error("Fresh validation measurements are not available yet.")
    setup()
    chart_sources = {}
    if not args.fresh_only:
        models, tokenizers = json.loads(model_path.read_text()), json.loads(tok_path.read_text())
        model_comparison(models, args.output_dir)
        tokenizer_tradeoff(tokenizers, args.output_dir)
        task_deltas(models, args.output_dir)
        evaluator_correction(models, args.output_dir)
        chart_sources.update({
            "model_outcomes_by_tier": (model_path, "archived_pretraining_and_legacy_benchmarks"),
            "tokenizer_efficiency_boundary_tradeoff": (tok_path, "archived_train_corpus_diagnostic"),
            "cetvel_task_deltas": (model_path, "archived_legacy_benchmarks"),
            "evaluator_correction_same_checkpoint": (model_path, "archived_evaluator_correction"),
        })
    if fresh_path.exists():
        matched_validation_tokenizers(json.loads(fresh_path.read_text()), args.output_dir)
        chart_sources["matched_validation_tokenizers"] = (fresh_path, "fresh_matched_validation_measurements")
    manifest_path = args.output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if args.fresh_only and manifest_path.exists() else {}
    charts = {c["png"].removesuffix(".png"): c for c in previous.get("charts", [])}
    sources = {s["path"]: s for s in previous.get("sources", [])}
    for stem, (path, evidence) in chart_sources.items():
        source = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        sources[source["path"]] = source
        charts[stem] = {"png": stem + ".png", "svg": stem + ".svg", "evidence": evidence, "source": source}
    manifest = {
        "report_date": "2026-11-09", "script": "scripts/plot_study_results.py",
        "sources": list(sources.values()), "charts": list(charts.values()),
        "notes": ["Historical model and corpus charts retain archived measurements; the optional matched validation chart uses fresh tokenizer measurements. Each chart records its own evidence and source hash.", "SVG preserves editable text.", "Error bars omitted because model seed replication is unavailable; these charts do not represent statistical significance."],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rendered {len(chart_sources)} charts as PNG + SVG in {args.output_dir}")


if __name__ == "__main__":
    main()
