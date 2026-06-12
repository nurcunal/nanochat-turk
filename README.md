# nanochat-turkish

This branch adapts current upstream nanochat for Turkish foundation-model
training. The base pipeline uses FineWeb-2 Turkish (`tur_Latn`) parquet files,
raw Rust BPE by default, Chinchilla-style `20` tokens per total parameter in the
Turkish presets, and CETVEL for Turkish evaluation. See
[docs/turkish_foundation.md](docs/turkish_foundation.md) for the active Turkish
workflow, A100/Slurm launch notes, and ablation plan. The running project/report
memory is in [docs/project_report_readme.md](docs/project_report_readme.md),
which summarizes what we have done, how the tokenizer study is being
implemented, and which claims are ready for the final report. Folder-level
indexes live in [docs/README.md](docs/README.md) and
[artifacts/README.md](artifacts/README.md).

The project has moved from a generic nanochat fork into a controlled Turkish LLM
tokenizer ablation study. The central question is:

> With model parameter size, pretraining budget, and FineWeb-2 Turkish data held
> fixed, which tokenizer choice - especially MorphBPE tokenizers trained from
> different morphological segmenters - produces the best tokenizer diagnostics,
> validation BPB, and CETVEL performance?

Turkish is agglutinative, so a frequency-only BPE tokenizer can learn pieces
that cut across productive morpheme boundaries. Our main experiment therefore
compares raw BPE against MorphBPE-style tokenizers: a morphological segmenter is
used while training BPE merges, but the final tokenizer still encodes and
decodes raw Turkish text without requiring a runtime segmenter.

## Current State

| Area | Status | Main artifacts |
| --- | --- | --- |
| Turkish data pipeline | FineWeb-2 Turkish download, parquet reading, UHeM runs, checkpointing, and artifact capture work end to end. | [docs/turkish_foundation.md](docs/turkish_foundation.md), [artifacts/uhem_smoke_2026-06-07_job492393](artifacts/uhem_smoke_2026-06-07_job492393) |
| Segmenter screening | TRmorph, Zemberek, TurkishDelightNLP, and identity control benchmarked on deterministic FineWeb-2 Turkish samples. | [docs/tokenizer_tests/segmenter_benchmark_status.md](docs/tokenizer_tests/segmenter_benchmark_status.md), [docs/tokenizer_tests/codex_local_judge_results.md](docs/tokenizer_tests/codex_local_judge_results.md) |
| MorphBPE implementation | Raw-text MorphBPE tokenizer training implemented and tested. Segmentation constrains merge learning only. | [docs/tokenizer_tests/morphbpe_framework.md](docs/tokenizer_tests/morphbpe_framework.md), [tests/test_morphbpe_tokenizer.py](tests/test_morphbpe_tokenizer.py) |
| Tokenizer artifacts | Raw BPE, TRmorph MorphBPE, and Zemberek MorphBPE 32k tokenizers archived; public Turkish tokenizer baselines measured. | [artifacts/tokenizers](artifacts/tokenizers), [docs/tokenizer_tests/tokenizer_metrics](docs/tokenizer_tests/tokenizer_metrics) |
| Base LLM benchmark | Raw-BPE, TRmorph MorphBPE, and Zemberek MorphBPE d20 base models evaluated on the common CETVEL core slice before SFT. | [docs/cetvel_model_comparison.md](docs/cetvel_model_comparison.md), [artifacts/cetvel_core12_model_comparison_2026-06-12](artifacts/cetvel_core12_model_comparison_2026-06-12) |

## Data Ground

The tokenizer work begins from FineWeb-2 Turkish (`tur_Latn`). The first large
segmenter inventory was computed from the first FineWeb-2 Turkish shard:

| Field | Value |
| --- | ---: |
| Documents | 3,381,000 |
| Word-like tokens | 1,282,426,655 |
| Unique word forms | 10,829,544 |
| UTF-8 bytes | 10,981,374,221 |

This matters because our tokenizer metrics are not just vocabulary statistics.
We measure tokenizer behavior on real Turkish web text and, when a segmented
reference is available, whether tokenizer pieces cross morpheme boundaries.

## Segmenter Screening

We evaluated four segmentation backends:

| Backend | Role |
| --- | --- |
| `identity` | No-segmentation control. |
| `trmorph` | TRmorph finite-state analyzer via `flookup` and `segment.fst`. |
| `zemberek` | Zemberek morphology through a Python wrapper. |
| `tdelight` | TurkishDelightNLP through a command wrapper or REST endpoint. |

The first screening pass segments unique word types, then weights results by
their corpus frequency. Segmentations are accepted only if the pieces reconstruct
the exact original surface word.

### Deterministic Hash-100k Screen

| Backend | Unique/s | Weighted split | Weighted fallback | Type fallback |
| --- | ---: | ---: | ---: | ---: |
| identity | ~510k | 0.000 | 0.000 | 0.000 |
| TRmorph | ~9.3k | 0.338 | 0.145 | 0.817 |
| Zemberek | ~3.3k | 0.316 | 0.425 | 0.233 |
| TurkishDelightNLP | ~1.5k | 0.362 | 0.000 | 0.000 |

Read this table as engineering screening, not final linguistic truth. TRmorph
is sharp when it produces usable analyses, TurkishDelightNLP has the best
coverage in the current wrapper, and Zemberek remains a useful conservative
control.

### Blind Local Judge

We also built a 500-item disagreement-focused judge pack from the same
hash-100k sample. Backend names were hidden behind labels and decoded only after
all judgments were written.

| Backend | Best count | Best rate | Acceptable count | Acceptable rate |
| --- | ---: | ---: | ---: | ---: |
| TRmorph | 241 | 48.2% | 255 | 51.0% |
| TurkishDelightNLP | 173 | 34.6% | 219 | 43.8% |
| Zemberek | 86 | 17.2% | 200 | 40.0% |
| identity | 0 | 0.0% | 4 | 0.8% |

The result supported carrying at least TRmorph and TurkishDelightNLP forward.
TRmorph became the first full MorphBPE tokenizer target because it won the blind
best-label rate, while TurkishDelightNLP remains a strong coverage-oriented
candidate and fallback ingredient.

## Tokenizer Optimization

The main tokenizer design is raw-text MorphBPE:

1. Materialize a boundary-marked training corpus from a segmenter.
2. Train BPE while preventing merges from crossing morpheme boundaries.
3. Save a normal raw-text tokenizer.
4. Use the tokenizer without runtime segmentation for training, CETVEL, and
   user prompts.

This is different from pre-segmented BPE. Pre-segmented BPE would put boundary
markers or segmented text into the model stream. MorphBPE uses segmentation only
to constrain the merge table.

The checked-in tokenizer comparison uses the same first `50,000` documents from
the TRmorph-segmented FineWeb-2 Turkish corpus. The boundary marker is stripped
before encoding, so all tokenizers receive identical raw Turkish text. True BPB
is model-dependent, so the diagnostic score below remains tokenizer-only. For
tokenizers that have completed matched d20 model runs, the table also reports
final validation BPB from `meta_017100.json`; public/tokenizer-only baselines
are marked `-`.

Ranking is calculated from the checked-in comparable metrics only. Zemberek is
now ranked because it has the same `50,000`-document TRmorph-reference metric as
the other rows; its earlier raw `10,000`-document diagnostic remains archived in
[docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_32768_raw_metrics.json](docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_32768_raw_metrics.json).
TurkishDelightNLP is still excluded from this table until its matching
TRmorph-reference metric is checked into the repo. The score is intentionally
transparent and documented in
[docs/tokenizer_tests/tokenizer_metrics/ranking_methodology.md](docs/tokenizer_tests/tokenizer_metrics/ranking_methodology.md):

1. For each metric, tokenizers are ranked best-to-worst; ties receive the average
   rank.
2. The diagnostic score uses a weighted average rank over morphology
   preservation, compression, word fertility, and throughput:

   | Group | Metric | Weight |
   | --- | --- | ---: |
   | Morphology | Boundary crossed down | 28% |
   | Morphology | Crossing tokens per 1k down | 17% |
   | Compression | Bytes/token up | 15% |
   | Word fertility | Corpus tokens/word down | 8% |
   | Word fertility | Isolated word fertility down | 8% |
   | Word fertility | Single-token words up | 6% |
   | Word fertility | Long-word fertility down | 6% |
   | Throughput | Encode tokens/sec up | 12% |

3. `Diagnostic score = 100 * (1 - (weighted average rank - 1) / (N - 1))`.
4. `Primary score = diagnostic score * (1 - roundtrip failure rate)`.

The round-trip factor is a raw-text safety gate: a tokenizer that normalizes or
cannot decode back to the original document may remain a useful public baseline,
but it should not outrank lossless candidates for nanochat pretraining.

These weights are a project-specific morphology-prioritized heuristic, not a
universal tokenizer benchmark. Morphology preservation receives the largest
share (`45%`) because the central hypothesis is that Turkish BPE should avoid
crossing productive morpheme boundaries. Word fertility (`28%`) captures
fragmentation of Turkish forms, compression (`15%`) captures raw-text exposure
under a fixed token budget, and throughput (`12%`) is included but kept below
the linguistic and compression terms because it is implementation- and
hardware-dependent. Sensitivity checks are reported in the methodology note:
equal or compression-heavy weights favor public compact tokenizers more, while
morphology-heavy weights favor the MorphBPE candidates. Therefore this table is
a screening view for tokenizer candidates; validation BPB and CETVEL decide the
final model-facing claim.

| Rank | Tokenizer | Impl. | Primary score | Diagnostic score | Val BPB d20 down | Roundtrip fail | Bytes/token up | Tokens/word down | Boundary crossed down | Crossing tok/1k down | Encode tok/s up | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `morphbpe_trmorph_32768` | morphbpe | 60.7 | 60.7 | 0.6266 | 0.0000 | 4.4514 | 1.8166 | 0.4569 | 125.3 | 13,419,184 | trained d20 |
| 2 | `morphbpe_zemberek_32768` | morphbpe | 55.3 | 55.3 | 0.6250 | 0.0000 | 4.4959 | 1.7986 | 0.5906 | 138.4 | 1,472,567 | trained d20 |
| 2 | `kumru_2b` | bpe | 55.3 | 55.3 | - | 0.0000 | 4.8488 | 1.6677 | 0.7745 | 189.9 | 818,251 | tokenizer baseline |
| 4 | `cosmos_turkish_gpt2` | bpe | 35.0 | 35.0 | - | 0.0000 | 5.1938 | 1.5570 | 0.8694 | 220.6 | 681,476 | tokenizer baseline |
| 5 | `bpe_32768` | bpe | 26.6 | 26.6 | 0.6232 | 0.0000 | 5.0051 | 1.6157 | 0.8395 | 210.4 | 12,127,686 | trained d20 |
| 6 | `vbart_large_base` | unigram | 2.9 | 60.4 | - | 0.9521 | 5.1827 | 1.5603 | 0.8015 | 203.4 | 579,818 | lossy baseline |
| 7 | `turna` | unigram | 2.9 | 59.6 | - | 0.9521 | 5.1827 | 1.5603 | 0.8015 | 203.4 | 540,988 | lossy baseline |
| 8 | `berturk_cased` | wordpiece | 0.3 | 47.1 | - | 0.9929 | 5.0897 | 1.5888 | 0.8045 | 205.0 | 863,345 | lossy baseline |

Full source metrics live in
[docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md](docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md).
The complete metric files include token counts, isolated-word fertility,
single-token word rate, long-word fertility, vocabulary diagnostics, and source
paths for each row.

### Tokenizer Takeaways

- `morphbpe_trmorph_32768` is ranked first because it is lossless, fastest in
  this harness, and sharply improves morphology preservation: boundary-crossed
  rate drops from `0.8395` for raw BPE to `0.4569`.
- `morphbpe_zemberek_32768` ties `kumru_2b` under the current weighted-rank
  score. Zemberek is better on TRmorph-reference boundary preservation, while
  Kumru is more compressive and has lower word fertility.
- `kumru_2b` is a strong lossless public-tokenizer baseline and stays close in
  the diagnostic score, but it crosses more TRmorph reference boundaries than
  either checked-in MorphBPE tokenizer.
- `vbart_large_base`, `turna`, and `berturk_cased` look strong on some fertility
  metrics, but their high round-trip failure rates make them lossy baselines
  rather than drop-in raw-text nanochat tokenizers.
- The trade-off is still real: TRmorph MorphBPE spends more tokens on the same
  raw text, and the first d20 validation BPB values still slightly favor raw
  BPE. The final project claim must use validation BPB and CETVEL, not only
  tokenizer-only optimization.

## Matched LLM Training Plan

The full study keeps corpus source, document order, training recipe, optimizer,
and evaluation fixed. At the 32k vocabulary tier, the primary model cells are:

| Vocab | Depth | Tokenizer | Model tag | Current status |
| ---: | ---: | --- | --- | --- |
| 32,768 | d20 | raw BPE | `tr_d20_bpe_32768_chinchilla20` | Trained; CETVEL tasks 01-13 archived and common core slice compared. |
| 32,768 | d20 | MorphBPE + TRmorph | `tr_d20_morphbpe_trmorph_32768_chinchilla20` | Trained; CETVEL core tasks 01-12 complete. |
| 32,768 | d20 | MorphBPE + Zemberek | `tr_d20_morphbpe_zemberek_32768_chinchilla20` | Trained; CETVEL core tasks 01-12 complete. |
| 32,768 | d20 | MorphBPE + TurkishDelightNLP | `tr_d20_morphbpe_tdelight_32768_chinchilla20` | Pipeline scripts/preflight in progress. |

The larger-vocabulary plan is documented in
[docs/tokenizer_ablation_plan.md](docs/tokenizer_ablation_plan.md): 65,536-vocab
models use d16, and 131,072-vocab models use d12, keeping total parameters near
the current approximately 1B-parameter budget.

## Current Base-Model CETVEL Comparison

The completed model-facing comparison currently uses CETVEL core tasks 01-12.
All rows are d20 base models evaluated before SFT at model step `17100`. The raw
BPE run has a larger tasks 01-13 archive, but the table below uses the common
tasks 01-12 slice so the raw baseline and MorphBPE variants are comparable.

Core-11 macro is the mean over the classification/loglikelihood tasks; `xquad_tr`
F1 is reported separately because it is on a different scale. CETVEL speed is
the logged core-12 wall-clock time divided by `39,441` expanded effective
examples, so it is an end-to-end benchmark throughput proxy rather than a pure
GPU-kernel measurement.

| Run | Tokenizer | Segmenter | CETVEL job | Core-12 elapsed | CETVEL ex/s up | Final train loss | Core-11 macro | Delta vs raw BPE | XQuAD F1 | Delta vs raw BPE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw BPE d20 | `bpe_32768` | none | `493293` | 50m20s | 13.06 | 2.4899 | 0.4514 | +0.0000 | 3.0985 | +0.0000 |
| MorphBPE + TRmorph d20 | `morphbpe_trmorph_32768` | TRmorph | `494056` | 52m38s | 12.49 | 2.0106 | 0.4541 | +0.0027 | 3.4786 | +0.3801 |
| MorphBPE + Zemberek d20 | `morphbpe_zemberek_32768` | Zemberek | `494057` | 50m30s | 13.02 | 2.3227 | 0.4618 | +0.0104 | 3.2633 | +0.1648 |

Detailed task metrics and source result paths live in
[docs/cetvel_model_comparison.md](docs/cetvel_model_comparison.md) and the
compact artifact summary
[artifacts/cetvel_core12_model_comparison_2026-06-12](artifacts/cetvel_core12_model_comparison_2026-06-12).
The raw-BPE tasks 01-13 archive remains in
[artifacts/cetvel_base_subset_2026-06-09_job493293](artifacts/cetvel_base_subset_2026-06-09_job493293).

Early model-facing evidence is mixed rather than uniformly pro-MorphBPE:
TRmorph MorphBPE has the best XQuAD F1, Zemberek MorphBPE has the strongest
core-11 macro, and raw BPE still wins several individual tasks. These results
should be reported as base-model evidence before SFT, not as a final
instruction-following or generation-quality claim. Final train loss is included
as run telemetry from the last printed training step; because token units differ
across tokenizers, validation BPB is the comparable loss metric. Benchmark
throughput is nearly tied for raw BPE and Zemberek MorphBPE in this core-12
harness, while TRmorph MorphBPE is about 4% slower end-to-end.

## Reproduction Pointers

| Goal | Entry point |
| --- | --- |
| Train the Turkish raw-BPE baseline on UHeM | [runs/uhem_nakane_a100x4_d20_bpe32k.sbatch](runs/uhem_nakane_a100x4_d20_bpe32k.sbatch) |
| Launch multi-node raw-BPE d20 training | [runs/uhem_nakane_a100x4_multinode_d20_bpe32k.sbatch](runs/uhem_nakane_a100x4_multinode_d20_bpe32k.sbatch) |
| Prepare TRmorph MorphBPE tokenizer | [runs/uhem_nakane_finalize_morphbpe_trmorph_32k.sbatch](runs/uhem_nakane_finalize_morphbpe_trmorph_32k.sbatch) |
| Train TRmorph MorphBPE d20 model | [runs/uhem_nakane_a100x4_morphbpe_trmorph_32k.sbatch](runs/uhem_nakane_a100x4_morphbpe_trmorph_32k.sbatch) |
| Compare tokenizer-only metrics | [runs/uhem_tokenizer_metrics_compare_32k.sbatch](runs/uhem_tokenizer_metrics_compare_32k.sbatch) |
| Run CETVEL | [runs/uhem_cetvel_full_final.sbatch](runs/uhem_cetvel_full_final.sbatch) |

For local development:

```bash
uv sync --extra cpu --group dev
source .venv/bin/activate
pytest tests/test_morphbpe_tokenizer.py
```

For GPU/UHeM work, the Slurm scripts are the source of truth because they set
the cache directories, tokenizer names, model tags, and CETVEL environment
variables that keep ablations comparable.

## Repository Map

| Path | Purpose |
| --- | --- |
| [nanochat](nanochat) | Core model, dataloader, tokenizer, checkpoint, and evaluation code. |
| [nanochat/morphology](nanochat/morphology) | Boundary markers, MorphBPE helpers, and segmenter adapters. |
| [scripts](scripts) | Training, tokenizer, morphology, CETVEL, publishing, and reporting scripts. |
| [runs](runs) | UHeM/Slurm launchers for smoke tests, tokenizer prep, pretraining, CETVEL, and uploads. |
| [docs](docs) | Study design, project memory, tokenizer notes, segmenter results, and Turkish workflow docs. |
| [artifacts](artifacts) | Compact, checked-in result artifacts and provenance. |

## License

MIT. This repository is a Turkish research fork of
[karpathy/nanochat](https://github.com/karpathy/nanochat).
