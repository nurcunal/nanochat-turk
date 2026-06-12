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
| Tokenizer artifacts | Raw BPE, TRmorph MorphBPE, and Zemberek MorphBPE 32k tokenizers archived; paper-style intrinsic metrics computed. | [artifacts/tokenizers](artifacts/tokenizers), [docs/tokenizer_tests/tokenizer_metrics](docs/tokenizer_tests/tokenizer_metrics) |
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
before encoding, so every tokenizer receives identical raw Turkish text.
MorphBPE-paper metrics are computed over `200,000` segmented word occurrences,
with Morph-Consistency using the paper defaults: `k=100` clusters, `C=50` word
pairs per cluster, and `N=10` resamples.

Following the MorphBPE paper, this table does not use a custom weighted ranking
formula. The paper-facing intrinsic ordering is based on:

- fertility `phi`: tokens per whitespace word, lower is more compact;
- Morphological Edit Distance `mu_e`: lower means token pieces align better with
  gold morpheme pieces;
- Morphological Consistency `mu_c`: higher means words that share morphemes also
  share tokenizer pieces, and shared tokenizer pieces more often correspond to
  shared morphemes.

For Turkish, the paper-style interpretation is not "lowest fertility wins."
MorphBPE is expected to spend more tokens on agglutinative forms if that buys
better morpheme alignment and consistency. Therefore the rank below prioritizes
`mu_e` and `mu_c`, with `phi` reported as the efficiency cost. Extra engineering
diagnostics are kept beside the paper metrics because they matter for actual
nanochat pretraining: boundary-crossing rate, bytes/token, isolated-word
fertility, reversibility, encode speed, and d20 validation BPB where a matched
model exists.

| Paper-style rank | Tokenizer | Impl. | Segmenter | phi down | mu_e down | mu_c F1 up | Morph exact up | Boundary crossed down | Bytes/token up | Isolated fertility down | Roundtrip fail | Encode tok/s up | Val BPB d20 down | Status |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `morphbpe_trmorph_32768` | morphbpe | TRmorph | 1.8166 | 1.4126 | 0.5129 | 0.4258 | 0.4569 | 4.4514 | 1.9821 | 0.0000 | 5,111,947 | 0.6266 | trained d20 |
| 2 | `morphbpe_zemberek_32768` | morphbpe | Zemberek | 1.7986 | 1.4817 | 0.4357 | 0.4040 | 0.5906 | 4.4959 | 1.9748 | 0.0000 | 5,075,946 | 0.6250 | trained d20 |
| 3 | `bpe_32768` | bpe | none | 1.6157 | 1.6836 | 0.3241 | 0.3342 | 0.8395 | 5.0051 | 2.0312 | 0.0000 | 4,922,601 | 0.6232 | trained d20 |

Full source metrics live in
[docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md](docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md).
The complete metric files also include token counts, normalized edit distance,
Morph-Consistency precision/recall/std, vocabulary diagnostics, and source paths
for each row.

### Public Tokenizer References

`kumru_2b` and `cosmos_turkish_gpt2` were recomputed with the same
MorphBPE-paper metric implementation by UHeM job `494176`
(`nanochat-tokenizer-extrefs32k`). They are loaded from public Hugging Face
tokenizer files only; no model weights are downloaded or used. In other words,
we have access to their tokenizer definitions, not their training recipe or a
matched nanochat model.

These rows are external references because their vocabularies are about `50k`,
not the controlled `32k` vocabulary used by our current ablations. They are
useful for tokenizer diagnostics, but they should not replace same-vocab model
comparisons.

| External tokenizer | Source | Vocab | phi down | mu_e down | mu_c F1 up | Morph exact up | Boundary crossed down | Bytes/token up | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `kumru_2b` | `vngrs-ai/Kumru-2B` tokenizer files | 50,176 | 1.6677 | 1.3500 | 0.3436 | 0.4515 | 0.7745 | 4.8488 | public BPE reference |
| `cosmos_turkish_gpt2` | `ytu-ce-cosmos/turkish-gpt2` tokenizer files | 50,257 | 1.5570 | 1.4839 | 0.3001 | 0.3965 | 0.8694 | 5.1938 | public BPE reference |

The combined local-plus-external table is checked in at
[docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison_with_external.md](docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison_with_external.md).
Kumru is especially instructive: its larger vocabulary gives it the best
`mu_e` and exact morpheme-sequence rate in this diagnostic, but TRmorph MorphBPE
still has much better Morph-Consistency (`mu_c`) and far fewer crossed morpheme
boundaries.

`vbart_large_base`, `turna`, and `berturk_cased` were removed from the main
README comparison because their earlier runs were lossy/normalizing or
architecture-specific references rather than plausible raw-text nanochat
pretraining tokenizers.

### Tokenizer Takeaways

- TRmorph MorphBPE ranks first by the MorphBPE paper logic: it has the lowest
  `mu_e`, highest `mu_c`, highest exact morpheme-sequence rate, and lowest
  boundary-crossing rate among checked-in 32k tokenizers.
- Zemberek MorphBPE is second: it also improves morphology alignment and
  consistency over raw BPE, but less strongly than TRmorph on the TRmorph
  reference segmentation.
- Raw BPE remains the most compact checked-in trained tokenizer by `phi`,
  bytes/token, and current d20 validation BPB. That is the central trade-off:
  MorphBPE improves morphology metrics while spending more tokens.
- The final project claim must still combine tokenizer-only metrics with matched
  validation BPB and CETVEL. The paper-style intrinsic metrics justify carrying
  TRmorph and Zemberek forward; they do not by themselves prove the best model.

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
