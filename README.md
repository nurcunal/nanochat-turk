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
optimization study. The central question is:

> Given an approximately fixed Turkish pretraining budget, which data and
> tokenizer choices produce the best tokenizer diagnostics, validation BPB, and
> CETVEL performance?

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
| Tokenizer artifacts | Raw BPE and TRmorph MorphBPE 32k tokenizers archived; public Turkish tokenizer baselines measured. | [artifacts/tokenizers](artifacts/tokenizers), [docs/tokenizer_tests/tokenizer_metrics](docs/tokenizer_tests/tokenizer_metrics) |
| Base LLM benchmark | Raw-BPE d20 base model trained and evaluated on CETVEL tasks 01-13 before SFT. | [artifacts/cetvel_base_subset_2026-06-09_job493293](artifacts/cetvel_base_subset_2026-06-09_job493293) |

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
is model-dependent, so this table reports tokenizer-only diagnostics.

| Tokenizer | Impl. | Docs | Bytes | Tokens | Bytes/token up | Tokens/word down | Isolated word fertility down | Single-token words up | Long-word fertility down | Boundary crossed down | Crossing tok/1k down | Roundtrip fail down | Encode tok/s up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bpe_32768` | bpe | 50,000 | 186,328,421 | 37,228,014 | 5.0051 | 1.6157 | 2.0472 | 0.3105 | 2.8417 | 0.8395 | 210.3501 | 0.0000 | 12,127,686 |
| `morphbpe_trmorph_32768` | morphbpe | 50,000 | 186,328,421 | 41,858,836 | 4.4514 | 1.8166 | 1.9890 | 0.3409 | 2.8046 | 0.4569 | 125.2956 | 0.0000 | 13,419,184 |
| `kumru_2b` | bpe | 50,000 | 186,328,421 | 38,427,427 | 4.8488 | 1.6677 | 1.6804 | 0.5003 | 2.1697 | 0.7745 | 189.9358 | 0.0000 | 818,251 |
| `berturk_cased` | wordpiece | 50,000 | 186,328,421 | 36,609,010 | 5.0897 | 1.5888 | 1.4201 | 0.7003 | 1.8625 | 0.8045 | 205.0115 | 0.9929 | 863,345 |
| `cosmos_turkish_gpt2` | bpe | 50,000 | 186,328,421 | 35,875,345 | 5.1938 | 1.5570 | 1.8836 | 0.3740 | 2.6116 | 0.8694 | 220.6239 | 0.0000 | 681,476 |
| `turna` | unigram | 50,000 | 186,328,421 | 35,952,018 | 5.1827 | 1.5603 | 1.3966 | 0.7181 | 1.7958 | 0.8015 | 203.4204 | 0.9521 | 540,988 |
| `vbart_large_base` | unigram | 50,000 | 186,328,421 | 35,952,025 | 5.1827 | 1.5603 | 1.3966 | 0.7181 | 1.7958 | 0.8015 | 203.4204 | 0.9521 | 579,818 |

Full source metrics live in
[docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md](docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md).
High `Roundtrip fail` is expected for some encoder/seq2seq tokenizers that
normalize text or do not preserve raw text exactly.

### Tokenizer Takeaways

- `morphbpe_trmorph_32768` is the clear boundary-behavior winner in the current
  tokenizer-only table: boundary-crossed rate drops from `0.8395` for raw BPE to
  `0.4569`.
- Public Turkish tokenizers such as cosmosGPT/TURNA/VBART are strong on
  corpus-level compression, but they cross the TRmorph reference boundaries more
  often in this measurement.
- The trade-off is real: TRmorph MorphBPE spends more tokens on the same raw
  text, so the final decision must use model validation BPB and CETVEL, not only
  tokenizer compression.

## Matched LLM Training Plan

The full study keeps corpus source, document order, training recipe, optimizer,
and evaluation fixed. At the 32k vocabulary tier, the primary model cells are:

| Vocab | Depth | Tokenizer | Model tag | Current status |
| ---: | ---: | --- | --- | --- |
| 32,768 | d20 | raw BPE | `tr_d20_bpe_32768_chinchilla20` | Trained; CETVEL tasks 01-13 archived. |
| 32,768 | d20 | MorphBPE + TRmorph | `tr_d20_morphbpe_trmorph_32768_chinchilla20` | Tokenizer trained and archived; model run script ready. |
| 32,768 | d20 | MorphBPE + Zemberek | `tr_d20_morphbpe_zemberek_32768_chinchilla20` | Pipeline scripts/preflight in progress. |
| 32,768 | d20 | MorphBPE + TurkishDelightNLP | `tr_d20_morphbpe_tdelight_32768_chinchilla20` | Pipeline scripts/preflight in progress. |

The larger-vocabulary plan is documented in
[docs/tokenizer_ablation_plan.md](docs/tokenizer_ablation_plan.md): 65,536-vocab
models use d16, and 131,072-vocab models use d12, keeping total parameters near
the current approximately 1B-parameter budget.

## Current Base-Model CETVEL Result

The completed base-model benchmark currently checked into the repo is the raw
BPE d20 run before SFT:

| Field | Value |
| --- | --- |
| Model tag | `tr_d20_bpe_32768_chinchilla20` |
| Tokenizer | `bpe_32768` |
| Step | `17100` |
| Training job | UHeM `492421` |
| CETVEL job | UHeM `493293` |
| Artifact | [artifacts/cetvel_base_subset_2026-06-09_job493293](artifacts/cetvel_base_subset_2026-06-09_job493293) |

The run was intentionally stopped after `tquad`. Tasks 01-13 cover the
base-model selection and extractive-QA slice. The remaining CETVEL generation
and instruction-style tasks are better reserved for SFT models or for later
base-model diagnostics.

| Task | n | Metric | Value |
| --- | ---: | --- | ---: |
| `belebele_tr` | 900 | `acc_norm` | 0.2522 |
| `cetvel_xcopa_tr` | 500 | `acc` | 0.6180 |
| `cetvel_xnli_tr` | 5,010 | `acc_norm` | 0.3335 |
| `check_worthiness` | 2,188 | `acc_norm` | 0.4287 |
| `exams_tr` | 393 | `acc_norm` | 0.3104 |
| `mnli_tr` | 10,000 | `acc_norm` | 0.3210 |
| `news_cat` | 250 | `acc_norm` | 0.6760 |
| `offenseval_tr` | 3,528 | `acc_norm` | 0.7971 |
| `relevance_judgment` | 2,188 | `acc_norm` | 0.5590 |
| `snli_tr` | 10,000 | `acc_norm` | 0.3234 |
| `tquad` | 892 | `f1` | 5.2603 |
| `trclaim19` | - | `acc_norm` | 0.4938 |
| `turkish_plu` | - | `acc_norm` | 0.5027 |
| `turkish_plu_goal_inference` | 837 | `acc_norm` | 0.4241 |
| `turkish_plu_next_event_prediction` | 655 | `acc_norm` | 0.5344 |
| `turkish_plu_step_inference` | 612 | `acc_norm` | 0.4771 |
| `turkish_plu_step_ordering` | 1,021 | `acc_norm` | 0.5622 |
| `xfact_tr` | 169 | `acc_norm` | 0.3373 |
| `xquad_tr` | 1,190 | `f1` | 3.0985 |

These numbers are a baseline, not the final claim. The tokenizer study becomes
meaningful only after the MorphBPE-tokenized models are trained with the same
budget and evaluated through the same CETVEL adapter.

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
