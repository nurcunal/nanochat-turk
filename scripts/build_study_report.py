"""Build the one public study page from measured data and private section drafts.

Run ``python -m scripts.build_study_report``. ``--plots`` also refreshes figures.
Old report filenames are mapped to stable anchors; exports and evidence have
dedicated subfolders. Rebuilding never launches training or evaluation jobs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit, urlunsplit

from scripts.study_report_paths import CONTENT_DIR, DATA_DIR, REPORT_DIR, ROOT, SECTION_DIR

PAGE = "docs/study_results/README.md"
REPORT_IDS = {
    "README.md": "study-results",
    "tokenizer_fresh_results.md": "fresh-tokenizers",
    "tokenizer_matched.md": "fresh-tokenizers",
    "tokenizer_full_corpus.md": "full-corpus-tokenizers",
    "model_results.md": "original-models",
    "model_configuration_results.md": "model-configurations",
    "later_model_results.md": "evaluator-audit",
    "sft_configuration_screens.md": "sft-configurations",
    "tokenizer_results.md": "historical-tokenizers",
    "tokenizer_additional.md": "additional-tokenizers",
    "metric_guide.md": "metric-guide",
    "reproduction.md": "reproduction",
}
SECTIONS = [
    ("fresh-tokenizers", "Fresh intrinsic tokenizer assessment", "tokenizer_fresh_results.md", False),
    ("original-models", "Original model pretraining and extrinsic benchmarks", "model_results.md", False),
    ("model-configurations", "Later d20/d32 configurations and benchmarks", "model_configuration_results.md", False),
    ("evaluator-audit", "Evaluator corrections and model provenance", "later_model_results.md", True),
    ("sft-configurations", "SFT configuration experiments", "sft_configuration_screens.md", True),
    ("full-corpus-tokenizers", "Recovered full-corpus tokenizer measurements", "tokenizer_full_corpus.md", True),
    ("historical-tokenizers", "Earlier tokenizer measurements and external references", "tokenizer_results.md", True),
    ("additional-tokenizers", "Additional 100k/1M and diagnostic runs", "tokenizer_additional.md", True),
    ("metric-guide", "Metric definitions and better directions", "metric_guide.md", True),
    ("reproduction", "Reproduction, evidence and historical source revisions", "reproduction.md", True),
]


def slug(title: str) -> str:
    return re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")


def load_section(name: str) -> str:
    source = CONTENT_DIR if name in {"metric_guide.md", "reproduction.md", "summary.md"} else SECTION_DIR
    return (source / name).read_text()


def audit_prose(text: str) -> str:
    """Keep unique audit details; four repeated result tables live in configurations."""
    skip = ("**Direction legend:", "**Bold numeric table cells", "The [expanded configuration report]",
            "All task scores in that receipt are preserved below.", "Rebuild:")
    lines = [line for line in text.splitlines() if not line.startswith("|") and not line.startswith(skip)]
    # The old EM=0 cell was unique to the removed comparison table.
    first = lines.index("## A corrected evaluator changes the interpretation")
    lines[first + 1:first + 1] = ["", "For the unchanged d20 step-17,100 checkpoint, the archived generation correction changes XQuAD F1 from **3.0985 to 11.3369** and exact match from **0.0000 to 4.5378**. These are evaluator changes, not training gains. The complete corrected-model tables appear in [model configurations](model_configuration_results.md)."]
    return "\n".join(lines).replace(
        "This table describes one d32 model",
        "The [configuration task table](model_configuration_results.md) describes one d32 model",
    )


def section_anchors() -> dict[tuple[str, str], str]:
    mapping = {}
    for section_id, _, name, _ in SECTIONS:
        counts = Counter()
        for line in load_section(name).splitlines():
            match = re.match(r"^#{2,6} (.+)$", line)
            if match:
                base = slug(match[1])
                suffix = f"-{counts[base]}" if counts[base] else ""
                counts[base] += 1
                mapping[(name, base + suffix)] = section_id + "--" + base + suffix
    # Duplicate matched report now resolves to the complete fresh section.
    for anchor in ("validation", "training-sample", "definitions"):
        mapping[("tokenizer_matched.md", anchor)] = mapping.get(("tokenizer_fresh_results.md", anchor), "fresh-tokenizers")
    for anchor, target in {
        "findings-that-can-be-defended": "overview", "original-model-matrix-beside-tokenizer-diagnostics": "overview-comparison",
        "requested-metrics-and-interpretation": "metric-guide", "sunum-için-kısa-sonuç": "overview",
        "read-or-present": "study-results", "rebuild-and-source-evidence": "reproduction",
    }.items():
        mapping[("README.md", anchor)] = target
    return mapping


def migrated_target(path: str, fragment: str, anchors: dict) -> tuple[str, str]:
    prefix = "docs/study_results/"
    if not path.startswith(prefix):
        return path, fragment
    relative = path[len(prefix):]
    if relative in REPORT_IDS:
        if relative == "README.md" and fragment in set(REPORT_IDS.values()) | set(anchors.values()) | {"overview", "overview-comparison", "downloads", "figures"}:
            return PAGE, fragment
        target = anchors.get((relative, fragment), REPORT_IDS[relative]) if fragment else REPORT_IDS[relative]
        return PAGE, target
    if relative == "fresh_tokenizer_metrics" or relative.startswith("fresh_tokenizer_metrics/"):
        return prefix + "provenance/" + relative, fragment
    if "/" not in relative and relative.endswith((".json", ".csv")):
        if relative.startswith("tokenizer_matched."):
            relative = relative.replace("tokenizer_matched.", "tokenizer_fresh_results.")
        return prefix + "data/" + relative, fragment
    return path, fragment


def rewrite_markdown(text: str, source: str, *, output: str | None = None, anchors: dict | None = None) -> str:
    """Migrate repository links, including links from old report names to anchors."""
    anchors = section_anchors() if anchors is None else anchors
    output = output or source
    github = "https://github.com/nurcunal/nanochat-turk/blob/main/"

    def replace(match):
        original = match[2]
        if original.startswith(github):
            parts = urlsplit(original)
            path, fragment = migrated_target(unquote(parts.path.split("/blob/main/", 1)[1]), parts.fragment, anchors)
            return match[1] + github + path + ("#" + fragment if fragment else "") + match[3]
        if re.match(r"[a-zA-Z][\w+.-]*:", original) or original.startswith("//"):
            return match[0]
        parts = urlsplit(original)
        source_path = Path(source)
        target = os.path.normpath(str(source_path.parent / unquote(parts.path))) if parts.path else source
        old_target = target
        target, fragment = migrated_target(target, parts.fragment, anchors)
        if target == old_target and fragment == parts.fragment and source == output:
            return match[0]
        relative = "" if target == output else os.path.relpath(target, Path(output).parent)
        return match[1] + urlunsplit(("", "", relative, parts.query, fragment)) + match[3]

    return re.sub(r"(!?\[[^\]]*\]\()([^\n)]+)(\))", replace, text)


def render_section(section_id: str, name: str, text: str, anchors: dict) -> str:
    rendered = []
    counts = Counter()
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("# "):
            continue
        match = re.match(r"^(#{2,6}) (.+)$", line) if not in_code else None
        if match:
            base = slug(match[2])
            suffix = f"-{counts[base]}" if counts[base] else ""
            counts[base] += 1
            rendered.extend([f'<a id="{section_id}--{base}{suffix}"></a>', "", "#" + line])
        elif not line.startswith("Rebuild:"):
            rendered.append(line)
    return rewrite_markdown("\n".join(rendered).strip(), f"docs/study_results/{name}", output=PAGE, anchors=anchors)


def generate() -> None:
    from scripts.build_tokenizer_study_results import build as historical
    from scripts.build_fresh_tokenizer_results import build as fresh
    from scripts.build_model_study_results import build as models
    from scripts.build_sft_configuration_results import main as sft
    from scripts.build_study_overview import main as overview
    for builder in (historical, fresh, models, sft, overview):
        builder()


def assemble() -> None:
    anchors = section_anchors()
    text = [
        '<a id="study-results"></a>', "", "# nanochat-turk — complete study report", "",
        "**One page for the study, updated 9 November 2026.** The significant intrinsic, extrinsic, later-model and historical results are collected below. Supporting exports are in [`data/`](data/), raw run evidence in [`provenance/`](provenance/), and figures in [`figures/`](figures/). Older file versions remain in Git history.", "",
        "**Reading the tables:** ↑ higher is better, ↓ lower is better for the stated objective, ↔ descriptive/trade-off. **Bold scores** are the best comparable displayed values, including ties, within vocabulary/sample/protocol groups. They are not significance claims. Legacy benchmark rankings remain provisional.", "",
        "## On this page", "",
        "- [Study summary and comparison overview](#overview)",
    ]
    text.extend(f"- [{title}](#{section_id})" for section_id, title, _, _ in SECTIONS)
    text += ["- [Presentation figures](#figures)", "- [Download data](#downloads)", "",
             "Historical and supporting sections expand in place: use **Show section** beneath their headings. You do not need to switch report files.", "",
             '<a id="overview"></a>', "", "## Study summary", "",
             render_section("overview", "README.md", load_section("summary.md"), anchors), "",
             '<a id="overview-comparison"></a>', "", "### Original model matrix beside tokenizer diagnostics", "",
             "Intrinsic columns below use the recovered full training corpus with capped morphology analysis. Models use the original step-17,100 matrix. Training speed is the recorded final-100-step median on four A100s. Compare within each tier; these different objectives do not form a composite score.", "",
             load_section("overview.md").strip(), "",
             "Raw BPE leads document compression and archived BPB in every tier. TRmorph leads this historical pairwise consistency metric at 32k/64k; TurkishDelight leads at 128k. The legacy core-11 leaders are Zemberek at 32k and raw BPE at 64k/128k, pending the corrected common rerun. Within-tier speed differences are under 1% and do not establish a tokenizer speed advantage.", ""]
    manifest_sections = []
    for section_id, title, filename, collapsed in SECTIONS:
        body = load_section(filename)
        if filename == "later_model_results.md":
            body = audit_prose(body)
        text += [f'<a id="{section_id}"></a>', "", f"## {title}", ""]
        if collapsed:
            text += ["<details>", "<summary>Show section</summary>", ""]
        text += [render_section(section_id, filename, body, anchors), ""]
        if collapsed:
            text += ["</details>", ""]
        text += ["[Back to contents](#study-results)", ""]
        manifest_sections.append({"anchor": section_id, "title": title, "collapsed": collapsed,
                                  "section_sha256": hashlib.sha256(body.encode()).hexdigest()})
    text += ['<a id="figures"></a>', "", "## Presentation figures", "", "<details>", "<summary>Show all five figures</summary>", ""]
    for stem, caption in [
        ("matched_validation_tokenizers", "Fresh matched validation: compression, morpheme preservation, boundary F1 and whole-word coverage"),
        ("model_outcomes_by_tier", "Archived pretraining and provisional legacy model benchmarks"),
        ("cetvel_task_deltas", "Archived task deltas under the legacy evaluator"),
        ("tokenizer_efficiency_boundary_tradeoff", "Earlier 50k-document compression and boundary-preservation trade-off"),
        ("evaluator_correction_same_checkpoint", "Evaluator correction on the same checkpoint"),
    ]:
        text += [f"**{caption}**", "", f"![{caption}](figures/{stem}.png)", ""]
    text += ["</details>", "", "[Back to contents](#study-results)", "", '<a id="downloads"></a>', "", "## Download data", "",
             "CSV files are for analysis; JSON preserves full precision, source hashes and protocol details. These are exports supporting this page, not separate report versions. The duplicated matched-tokenizer export has been folded into the fresh export, including its additional diagnostic fields.", "",
             "| Evidence | CSV | JSON |", "| --- | --- | --- |"]
    for stem, title in [("tokenizer_fresh_results", "Fresh train/validation tokenizer metrics"),
                        ("tokenizer_full_corpus", "Recovered full-corpus tokenizer metrics"),
                        ("tokenizer_results", "Historical tokenizer and reference metrics"),
                        ("tokenizer_additional", "100k/1M and other tokenizer diagnostics"),
                        ("model_results", "Original eleven-model matrix"),
                        ("model_configuration_results", "Later model configurations"),
                        ("sft_configuration_screens", "SFT configuration screens")]:
        text.append(f"| {title} | [CSV](data/{stem}.csv) | [JSON](data/{stem}.json) |")
    text += ["", "Detailed exports: [132 original task cells](data/model_task_metrics.csv), [all recovered task metrics](data/model_all_task_metrics.csv), [d32 table-compatible receipt scores](data/model_official_task_metrics.csv), [training summary statistics](data/model_training_statistics.csv), [detailed training performance](data/model_training_performance.csv), and [later-model audit JSON](data/later_model_results.json).", "",
             "Figures: [matched validation](figures/matched_validation_tokenizers.png), [archived model comparison](figures/model_outcomes_by_tier.png), [legacy task deltas](figures/cetvel_task_deltas.png), [historical compression/boundary trade-off](figures/tokenizer_efficiency_boundary_tradeoff.png), and [same-checkpoint evaluator correction](figures/evaluator_correction_same_checkpoint.png). Editable SVGs and source hashes are in [`figures/`](figures/).", "",
             "[Nine-page PDF brief](../../output/pdf/nanochat_turk_study_results.pdf) · [Fresh evaluation source/output manifests](provenance/fresh_tokenizer_metrics/manifest.json) · [Report assembly manifest](data/report_manifest.json)", ""]
    output = "\n".join(text)
    (REPORT_DIR / "README.md").write_text(output)
    exports = [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in sorted(DATA_DIR.iterdir()) if path.is_file() and path.name != "report_manifest.json"]
    manifest = {"schema_version": 1, "canonical_report": PAGE, "sections": manifest_sections,
                "data_exports": exports, "report_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "notes": ["All significant report sections are assembled on one public page.",
                          "Matched tokenizer diagnostics are merged into the fresh report/export; no new measurements.",
                          "Repeated later-model tables are shown once in model configurations; unique evaluator/provenance prose is retained."]}
    (DATA_DIR / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built one report: {REPORT_DIR / 'README.md'} ({len(output.encode()):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assemble-only", action="store_true", help="Use existing private section drafts")
    parser.add_argument("--plots", action="store_true", help="Also refresh all figures")
    args = parser.parse_args()
    if not args.assemble_only:
        generate()
    if args.plots:
        subprocess.run([sys.executable, "-m", "scripts.plot_study_results"], cwd=ROOT, check=True)
    assemble()


if __name__ == "__main__":
    main()
