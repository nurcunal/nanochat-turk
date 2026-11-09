"""Create the shareable PDF brief after measured study results are assembled.

Requires reportlab, matplotlib. Run with the bundled artifact Python runtime or
an environment containing those packages. Source tables stay in docs/study_results.
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/study_results"
OUTPUT = ROOT / "output/pdf/nanochat_turk_study_results.pdf"
INK = colors.HexColor("#14283D")
MUTED = colors.HexColor("#526170")
TEAL = colors.HexColor("#168674")
FAMILIES = ("bpe", "morphbpe_trmorph", "morphbpe_zemberek", "morphbpe_tdelight")
LABELS = {"bpe": "Raw BPE", "morphbpe_trmorph": "TRmorph", "morphbpe_zemberek": "Zemberek", "morphbpe_tdelight": "TurkishDelight"}


def load(name):
    return json.loads((DATA / name).read_text())


def percent(v):
    return "N/A" if v is None else f"{100*v:.2f}"


def build():
    models = load("model_results.json")
    later = load("later_model_results.json")
    full = load("tokenizer_full_corpus.json")["tokenizers"]
    fresh = {}
    for split in ("train", "val"):
        fresh[split] = [load(f"fresh_tokenizer_metrics/{split}/{family}_{size}_metrics.json")
                        for size in (32768, 65536, 131072) for family in FAMILIES]
    pdfmetrics.registerFont(TTFont("Study", font_manager.findfont("DejaVu Sans")))
    pdfmetrics.registerFont(TTFont("StudyBold", font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="StudyTitle", fontName="StudyBold", fontSize=26, leading=34, textColor=INK, spaceAfter=16))
    styles.add(ParagraphStyle(name="StudyH", fontName="StudyBold", fontSize=18, leading=24, textColor=INK, spaceAfter=13))
    styles.add(ParagraphStyle(name="StudyBody", fontName="Study", fontSize=10.5, leading=16, textColor=INK, spaceAfter=11))
    styles.add(ParagraphStyle(name="StudySmall", fontName="Study", fontSize=8.3, leading=12.5, textColor=MUTED, spaceAfter=8))
    styles.add(ParagraphStyle(name="StudyCell", fontName="Study", fontSize=8.2, leading=11, textColor=INK, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="StudyHead", fontName="StudyBold", fontSize=7.8, leading=10.5, textColor=colors.white))
    story = []
    width, height = landscape(A4)
    available_width = width - 36*mm

    def para(text, style="StudyBody"):
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        para(text, "StudyH")

    def table(headers, rows, widths):
        content = [[Paragraph(str(x), styles["StudyHead"]) for x in headers]]
        content += [[Paragraph(str(x), styles["StudyCell"]) for x in row] for row in rows]
        t = Table(content, colWidths=[available_width*x/sum(widths) for x in widths], repeatRows=1, hAlign="LEFT")
        commands = [("BACKGROUND", (0,0), (-1,0), INK), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                    ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                    ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
                    ("LINEBELOW", (0,0), (-1,0), 0.6, INK)]
        for i in range(1, len(content)):
            if i % 2 == 0:
                commands.append(("BACKGROUND", (0,i), (-1,i), colors.HexColor("#F0F4F6")))
        t.setStyle(TableStyle(commands))
        story.extend([t, Spacer(1, 12)])

    para("nanochat-turk<br/>Tokenizer study results", "StudyTitle")
    para("Research brief · 9 November 2026 · Consolidated on main", "StudySmall")
    para("<b>12 tokenizers. 11 matched base models. Three vocabulary tiers.</b><br/>Raw byte-level BPE versus MorphBPE trained with TRmorph, Zemberek and TurkishDelightNLP boundaries.")
    table(["What the evidence supports", "Result"], [
        ["Morphology / compression", "MorphBPE preserves more automatic TRmorph boundaries; raw BPE uses fewer tokens per document word."],
        ["Pretraining", "Raw BPE has the lowest archived validation bits per byte in every vocabulary tier."],
        ["Downstream evaluation", "The legacy table records a +1.04-point 32k Zemberek difference. Generation and likelihood bugs make tokenizer-effect rankings provisional until a corrected matched rerun."],
        ["Missing model", "32k TurkishDelight has a tokenizer, but no complete model in the recovered ablation inventory."],
    ], [1, 3])
    para("<b>Compare within each tier.</b> The architecture is constant within a vocabulary tier. Across tiers, depth changes (d20 / d16 / d12) to keep parameters near 0.9B. All completed ablation runs use 17,100 steps and 17.93B token positions. Equal token positions do not guarantee equal raw-text exposure or evaluation spans.")
    para("The package preserves the earlier 50k-document metrics, recovers a completed 93.21M-document training-corpus pass, and adds fresh matched train/validation measurements. Morphology references are automatic, not human gold annotation. Model differences are single-run estimates without confidence intervals.", "StudySmall")
    story.append(PageBreak())

    for split in ("val", "train"):
        rows = fresh[split]
        sample = rows[0]
        heading("New tokenizer measurements / " + ("validation split" if split == "val" else "matched training sample"))
        para(f'{sample["docs"]:,} documents; {sample["coverage"]["sample_words"]:,} isolated word occurrences per tokenizer. All rows share the same word/reference sample hashes. Percentages use occurrence counts.', "StudySmall")
        grid = []
        for d in rows:
            p = d["morphology"]["preservation"]
            c = d["coverage"]
            family = d["tokenizer_name"].rsplit("_", 1)[0]
            grid.append([f'{d["vocabulary"]["vocab_size"]:,}', LABELS[family],
                         f'{d["tokens_per_word"]:.4f}', f'{d["word_fertility_isolated"]["tokens_per_word"]:.4f}',
                         percent(c["one_token_word_rate"]), percent(c["character_only_word_rate"]),
                         percent(p["morpheme_preservation_rate"]), percent(p["boundary_precision"]),
                         percent(p["boundary_recall"]), percent(p["boundary_f1"])])
        table(["Vocab", "Tokenizer", "Corpus tok/word", "Isolated tok/word", "One-token words %", "Char-only words %", "Morphemes intact %", "Isolated boundary P %", "Isolated boundary R %", "Isolated boundary F1 %"], grid, [0.8,1.2,0.9,0.9,0.9,0.9,1,0.8,0.8,0.8])
        para("<b>Morpheme preservation:</b> fraction of reference morpheme spans that coincide exactly with one token span. <b>Boundary consistency:</b> micro precision/recall/F1 over isolated-word internal token and reference boundaries; these differ from contextual document-boundary preservation. <b>Character-only:</b> every token is one Unicode character; UTF-8 byte fragments are treated separately in JSON.", "StudySmall")
        para("All twelve tokenizers are byte-complete and have no UNK mechanism, so UNK counts are N/A. Whole-word coverage is measured in isolation. Corpus fertility includes punctuation/spacing tokens. Automatic TRmorph references may favor TRmorph-trained tokenizers. Consult the JSON for alignment exclusions, denominators, byte-only coverage and roundtrip counts.", "StudySmall")
        story.append(PageBreak())

    heading("Earlier metrics / recovered complete training corpus")
    para("Job 496882: COMPLETED, exit 0:0. All 12 tokenizers: 93,211,827 documents, 36.70B regex words, 299.17GB raw UTF-8 bytes. Isolated-word/alignment metrics use 500k occurrences; consistency uses 50k unique words.", "StudySmall")
    grid = []
    for r in full:
        f = r["tokenizer"].rsplit("_",1)[0]
        grid.append([f'{r["vocab_size"]:,}', LABELS[f], f'{r["tokens_per_word"]:.4f}',
                     f'{r["bytes_per_token"]:.4f}', f'{r["morphological_edit_distance"]:.4f}',
                     percent(r["exact_morpheme_sequence_rate"]), percent(r["boundary_preservation_rate"]),
                     f'{r["morphological_consistency_f1"]:.4f}'])
    table(["Vocab", "Tokenizer", "Tok/word", "Bytes/token", "Morph edit", "Exact morph sequence %", "Boundaries preserved %", "Morph consistency F1"], grid, [0.8,1.25,0.85,0.85,0.85,1.2,1.2,1.1])
    para("All twelve record zero document roundtrip failures. Preserved boundaries = 1 - crossed-reference-boundary rate; it does not penalize extra splits and differs from boundary precision/F1. The original morphology-consistency statistic uses shared-token/shared-morpheme word pairs. It is distinct from the new literal boundary metric.", "StudySmall")
    para("This is a training-corpus diagnostic. The companion report retains the earlier 50k table, additional 100k/1M runs, external and normalizing tokenizer references, all distributions and timing data. It also includes extended model benchmarks and separate SFT configuration screens.", "StudySmall")
    story.append(PageBreak())

    heading("Pretraining and legacy CETVEL outcomes")
    para("Within-tier controls: same documented data source/order, architecture, optimizer recipe and 17.93B token positions. The historical recipe is retained in the pre-consolidation snapshot; original per-run Git revisions were not recorded. The consolidated d32 optimizer is a later recipe. Cross-tier comparisons change model depth.", "StudySmall")
    grid = []
    for r in models["runs"]:
        if r.get("validation_bpb") is None:
            continue
        family = "bpe" if r["tokenizer"].startswith("bpe_") else r["tokenizer"].rsplit("_",1)[0]
        grid.append([r["vocab"] + "/d" + str(r["transformer_depth"]), LABELS[family],
                     f'{r["validation_bpb"]:.4f}', f'{r["final_train_loss"]:.4f}',
                     f'{r["macro_core11"]:.4f}', f'{r["delta_macro_core11_percentage_points"]:+.2f}',
                     f'{r["core12_examples_per_second"]:.2f}'])
    table(["Tier", "Tokenizer", "Validation BPB ↓", "Token loss*", "Core-11 ↑", "Delta pp", "CETVEL ex/s"], grid, [0.9,1.35,1,1,1,0.8,1])
    para("* Token loss uses different prediction units across tokenizers and is not a comparable loss measure. BPB is byte-normalized, but equal evaluation token budgets need not cover identical raw-text spans. Core-11 is the unweighted mean of eleven classification/likelihood tasks; its legacy values may be affected by continuation-boundary scoring. XQuAD is excluded and retained in the detailed legacy/fixed-harness reports. 32k TurkishDelight remains missing.", "StudySmall")
    para("Later d32 production, continued d20 pretraining and changed benchmark harnesses are reported separately in the companion model report. They cannot be added as tokenizer-only treatments in this matched matrix.", "StudySmall")
    story.append(PageBreak())

    heading("Later training and corrected benchmark evidence")
    para("These recovered August results change model size, training budget or evaluator settings. They are separate from the eleven-model tokenizer ablation.", "StudySmall")
    grid = []
    for r in later["pretraining"]:
        grid.append([f'd{r["depth"]} BPE32k', f'{r["step"]:,}', f'{r["scheduled_token_positions"]/1e9:.3f}B',
                     f'{r["validation_bpb"]:.6f}', "Train-overlap diagnostic" if r["depth"] == 32 else "Continued training; doubled budget"])
    table(["Model", "Step", "Token positions", "Recorded BPB", "Qualification"], grid, [1,0.8,1,1,2.2])
    para("<b>The d32 model trained on all 30 source shards, including its validation shard.</b> Its 0.519040 BPB cannot be interpreted as held-out generalization. This completed 2.819B-parameter FineWeb2 run is separate from the frozen WSD production family. The d20 continuation saved its final checkpoint, but job 512018 later ended FAILED; its saved training-time counter is cumulative.", "StudySmall")
    grid = [["d20 BPE32k, step 17100", "June legacy adapter, cap 128", "3.0985", "0.0000"]]
    grid += [[r["model"], r["protocol"], f'{r["f1"]:.4f}', f'{r["exact_match"]:.4f}'] for r in later["fixed_xquad"]]
    table(["Model", "XQuAD protocol", "F1 /100", "Exact match /100"], grid, [1.8,2.4,1,1])
    receipt = later["eacl_table_compatible_receipt"]
    para(f'The d32 EACL table-compatible receipt reports <b>{receipt["overall_score"]:.4f}/100</b>, the mean of seven category scores. It has its own scoring conventions and cannot be compared directly with core-11. Full task/category tables and protocol pins are in later_model_results.md.', "StudySmall")
    para("The adapter now handles generation kwargs, stops and exact continuation boundaries, with regression tests and an output protocol version. A corrected common rerun for all eleven model cells is still needed; no tokenizer advantage is inferred from a single corrected baseline.", "StudySmall")
    story.append(PageBreak())

    chart_stems = ("matched_validation_tokenizers", "model_outcomes_by_tier", "evaluator_correction_same_checkpoint")
    for i, stem in enumerate(chart_stems):
        img = Image(str(DATA / "figures" / f"{stem}.png"))
        scale = min(available_width/img.imageWidth, (height-42*mm)/img.imageHeight)
        img.drawWidth, img.drawHeight = img.imageWidth*scale, img.imageHeight*scale
        story.append(img)
        if i < len(chart_stems)-1:
            story.append(PageBreak())

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D8E0E5"))
        canvas.line(18*mm, 14*mm, width-18*mm, 14*mm)
        canvas.setFont("Study", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(18*mm, 9*mm, "nanochat-turk | Sources, full tables and checksums: docs/study_results/README.md")
        canvas.drawRightString(width-18*mm, 9*mm, str(doc.page))
        canvas.restoreState()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(OUTPUT), pagesize=landscape(A4), rightMargin=18*mm, leftMargin=18*mm,
                                 topMargin=16*mm, bottomMargin=20*mm, title="nanochat-turk tokenizer study results",
                                 author="nanochat-turk research project")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
