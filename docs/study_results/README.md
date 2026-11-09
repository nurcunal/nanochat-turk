# nanochat-turk study results

**Presentation and evidence package, 9 November 2026.** `main` is the development
and results entry point. All historical branch commits and original measurements
are retained. The d32 production code is consolidated without rewriting its
sealed experiment contracts.

The study compares **12 original tokenizers**: raw BPE and MorphBPE with TRmorph,
Zemberek or TurkishDelightNLP boundaries, at **32,768 / 65,536 / 131,072** vocabulary
sizes. There are **11 completed base-model cells** in the original controlled
matrix; the 32k TurkishDelight model remains missing. Later continued-d20 and
d32 runs are reported separately.

## Read or present

**How to read the scores:** **↑ higher is better**, **↓ lower is better** for
the stated objective; **↔ descriptive / trade-off** has no universal better
direction. See the [metric guide](metric_guide.md) for plain-language meanings,
formulas, units, worked examples and comparison limits. A compression advantage
does not automatically imply better morphology or LLM performance.

**Bold scores identify the best displayed value within a comparable group**;
ties at the shown precision are all highlighted. Tokenizer/model matrices compare
within each vocabulary tier and sample; task tables compare models on the same
task and protocol. Descriptive settings, missing results and incomparable scores
are not ranked. Bold legacy benchmark values remain provisional, and bold
runtime values indicate observed speed or cost rather than statistical superiority.

| Deliverable | Contents |
| --- | --- |
| [Metric meanings and better directions](metric_guide.md) | Definitions for tokenizer, morphology, model, benchmark and runtime columns |
| [Shareable PDF brief](../../output/pdf/nanochat_turk_study_results.pdf) | Main tables, controls, caveats and figures |
| [Fresh matched tokenizer results](tokenizer_fresh_results.md) | All requested metrics on equal train and validation samples; 500k word occurrences per tokenizer |
| [Recovered full-corpus metrics](tokenizer_full_corpus.md) | All 12 tokenizers over 93,211,827 training documents; job 496882 verified complete |
| [Earlier tokenizer metrics](tokenizer_results.md) | Original 50k-document results, external Kumru/Cosmos tokenizer references, previous metrics and definitions |
| [Additional tokenizer measurements](tokenizer_additional.md) | Completed 100k/1M-document runs and raw-BPE train/validation diagnostics |
| [Pretraining and original model benchmarks](model_results.md) | Eleven-model BPB, token loss, training statistics and all 132 legacy CETVEL task scores |
| [Later models and evaluation corrections](later_model_results.md) | Continued d20, completed d32, corrected XQuAD and EACL table-compatible receipt |
| [Extended model configuration appendix](model_configuration_results.md) | Full-precision later-model configurations, training performance and broader task/category scores |
| [SFT configuration screens](sft_configuration_screens.md) | Learning-rate, curriculum and checkpoint selection diagnostics for d32/raw-BPE32k |
| [Reproduction and provenance](reproduction.md) | Exact recovery revisions, optimizer changes and unchanged strict d32 contracts |
| [PNG and editable SVG figures](figures/) | Presentation figures with source hashes and explicit evaluation scope |

Machine-readable exports: [fresh tokenizer CSV](tokenizer_fresh_results.csv),
[fresh JSON](tokenizer_fresh_results.json), [full-corpus CSV](tokenizer_full_corpus.csv),
[model summary CSV](model_results.csv), [all model task scores](model_task_metrics.csv),
and [training statistics CSV](model_training_statistics.csv). Original detailed
JSONs preserve counts, distributions and source hashes.

Additional exports include [all recovered task/submetric rows](model_all_task_metrics.csv),
[training performance](model_training_performance.csv), and
[SFT screen records](sft_configuration_screens.csv). The historical tokenizer
appendix also retains BERTurk, TURNA and VBART normalization diagnostics,
segmenter throughput/coverage and the blind local judge. Failed roundtrips are
separated from the lossless tokenizer comparison. Earlier June model reports
partly describe the same checkpoints and are not independent repetitions.

The [archival matched report](tokenizer_matched.md) and its frozen-source folder
mirror the same job `558060` as the fresh report; they are not another experiment.
SFT screens use one tokenizer and selection probes, so they do not establish a
multi-tokenizer SFT effect or independent held-out capability improvements.

## Findings that can be defended

**Best for morphology: TRmorph MorphBPE.** On the matched validation sample it has the highest intact-morpheme preservation, internal-boundary F1 and one-token word coverage at every vocabulary size. For example, at 32k it preserves **44.30%** of morphemes versus **31.57%** for raw BPE, and reaches **38.38%** boundary F1 versus **24.96%**. This is performance against automatic TRmorph annotations, which can favor the same analyzer used in training. The older pairwise-consistency metric has an exception: **TurkishDelight wins at 128k on the training/full-corpus diagnostic** (0.3971 versus TRmorph's 0.3786); TRmorph leads that metric on the matched validation sample. The metric and split determine the winner.

**Best for document compression: raw BPE.** It uses the fewest document tokens per word and the most bytes per token within every vocabulary tier. Among MorphBPE treatments, TurkishDelight is the most compact. Larger vocabulary improves compression, but that alone does not establish a better language model. Isolated-word fertility is a different objective: on validation, Zemberek is slightly best at 32k, while TRmorph is best at 64k and 128k.

MorphBPE improves the recorded automatic morphology diagnostics relative to
same-vocabulary raw BPE. This comes with higher **document** token fertility;
isolated-word fertility can move in the opposite direction because whitespace
and context change segmentation. TRmorph is also the reference analyzer, so
these results do not establish agreement with independent human annotation.

Raw BPE has the lowest **archived validation BPB** within every vocabulary tier.
The design holds model shape and training token positions fixed within a tier,
but it does not prove identical raw-text exposure or identical validation text
spans. Across tiers, depth changes from d20 to d16 to d12 to keep parameters
near 0.9B; cross-tier results do not isolate vocabulary size alone.

**The original downstream rankings are provisional.** Recovered evaluation
code documents generation-kwargs and continuation-boundary errors in the
legacy CETVEL adapter. The same d20 checkpoint's XQuAD F1 changes from **3.0985**
to **11.3369** under corrected generation handling. This is an evaluator
correction, not a training or tokenizer gain. The original eleven-model table
is preserved, but a complete rerun under one corrected protocol is still needed
before making downstream tokenizer-effect claims. The adapter fixes and
regression tests are included in the consolidated source.

**Recorded model leaders:** raw BPE has the lowest validation BPB at 32k, 64k and 128k (**0.6232 / 0.6409 / 0.6749**). In the legacy downstream tables, Zemberek is the 32k core-11 leader (**0.4618**, +1.04 percentage points over raw BPE); raw BPE leads at 64k (**0.4590**) and 128k (**0.4651**). These are archived rankings, not confirmed downstream tokenizer advantages, because the corrected common benchmark matrix is still missing.

**What the configuration studies show:** continuing raw-BPE d20 from 17,100 to 34,200 steps lowers recorded BPB from **0.62321 to 0.59145** (about 5.1%), with a changed learning-rate schedule; no continuation benchmark establishes a downstream gain. The larger d32 has higher corrected XQuAD F1 (**15.8833 versus 11.3369** for d20 under the 128-token protocol), but model size, data exposure and budget also change. Its training-overlap BPB cannot establish held-out superiority. The shallower d16/d12 configurations train faster than d20, while also changing vocabulary and width; this is a configuration trade-off, not an isolated vocabulary effect. SFT screens show that lower SFT BPB or a higher total probe score can still fail behavior/category gates; the recorded decisions retain the source checkpoint.

The later d32 run reached 28,000 steps and 58.72B scheduled token positions.
Its **0.519040 BPB is a train-overlap diagnostic** because its validation shard
was also used for training. It must not be presented as a held-out improvement
over the original ablation. Its distinct benchmark protocols and scores are
fully listed in the later-model appendix.

## Original model matrix beside tokenizer diagnostics

Intrinsic columns use the recovered full training corpus with capped morphology
analysis. Model scores use step 17,100; speed is the median of the final 100
printed steps on the recorded four-A100 setup. Compare within each vocabulary
tier. The legacy core-11 scores require a corrected common rerun. These columns
measure different quantities and are not a composite ranking. `—` means a
missing completed model cell.

<!-- BEGIN GENERATED TOKENIZER MODEL OVERVIEW -->

| Vocabulary / depth ↔ | Tokenizer | Corpus tokens/word ↓ | Morph consistency F1 ↑ | Val BPB ↓ | Legacy core-11 ↑ | Training k tokens/s ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32k / d20 | Raw BPE | **1.5553** | 0.3241 | **0.6232** | 0.4514 | 235.8 |
| 32k / d20 | MorphBPE / TRmorph | 1.7698 | **0.5129** | 0.6266 | 0.4541 | **236.1** |
| 32k / d20 | MorphBPE / Zemberek | 1.7551 | 0.4357 | 0.6250 | **0.4618** | 235.3 |
| 32k / d20 | MorphBPE / TurkishDelight | 1.6865 | 0.4795 | — | — | — |
| 64k / d16 | Raw BPE | **1.4328** | 0.2863 | **0.6409** | **0.4590** | 329.4 |
| 64k / d16 | MorphBPE / TRmorph | 1.6712 | **0.4421** | 0.6521 | 0.4532 | 329.9 |
| 64k / d16 | MorphBPE / Zemberek | 1.6640 | 0.3743 | 0.6514 | 0.4568 | 329.9 |
| 64k / d16 | MorphBPE / TurkishDelight | 1.5651 | 0.4296 | 0.6510 | 0.4567 | **331.1** |
| 128k / d12 | Raw BPE | **1.3473** | 0.2412 | **0.6749** | **0.4651** | 451.9 |
| 128k / d12 | MorphBPE / TRmorph | 1.5952 | 0.3786 | 0.6917 | 0.4503 | 454.6 |
| 128k / d12 | MorphBPE / Zemberek | 1.5916 | 0.3330 | 0.6940 | 0.4618 | **455.0** |
| 128k / d12 | MorphBPE / TurkishDelight | 1.4715 | **0.3971** | 0.6820 | 0.4481 | 453.7 |

<!-- END GENERATED TOKENIZER MODEL OVERVIEW -->

**Interpretation.** Raw BPE wins this table's document compression and validation-BPB columns in every tier. TRmorph has the highest historical pairwise-consistency F1 at 32k/64k, while TurkishDelight leads it at 128k. The recorded legacy core-11 leaders are Zemberek at 32k and raw BPE at 64k/128k, subject to the evaluator caveat above. Training-throughput differences between tokenizer families inside a tier are small (under 1% in these final-step medians); the much larger speed change across tiers accompanies a different depth/width/vocabulary configuration. No single tokenizer wins all objectives.

## Requested metrics and interpretation

| Requested quantity | Reported definition | Better direction |
| --- | --- | --- |
| Vocabulary size | Exact total vocabulary, including nine special tokens | ↔ Capacity/cost trade-off; larger is not automatically better |
| Subword efficiency | Both document tokens/regex word and isolated-word subword units/word | ↓ Lower means fewer tokens for the stated text unit |
| Morpheme preservation rate | Reference morpheme spans exactly coinciding with one token span; isolated-word measurement | ↑ Higher means more reference morphemes remain intact |
| Morpheme boundary consistency | Isolated-word internal-boundary micro precision, recall and F1; separate from older pairwise morphological consistency | ↑ Higher means better agreement with reference boundaries |
| Whole-word coverage | Lossless non-UNK words represented by one token in isolation | ↑ Higher for whole-word compactness; does not establish morphology quality |
| Character-only coverage | Words represented entirely by single Unicode-character tokens, with a second rate excluding trivial one-character words | ↔ Inclusive diagnostic; ↓ lower multi-character-only rate means less character-level fragmentation |
| Byte-only coverage | Words represented entirely by single-byte tokens; distinct from Unicode characters | ↔ Byte-granularity diagnostic, including trivial single-byte words |
| Unknown tokens | Architecture-aware count/incidence where UNK exists; N/A for these twelve byte-complete BPE tokenizers | ↓ Lower is better where UNK exists; N/A is not zero |

**Interpretation.** “Best coverage” needs a definition: TRmorph leads one-token coverage on the matched validation sample, all twelve local tokenizers tie on zero observed roundtrip failures, and UNK does not apply to these byte-complete tokenizers. Character-only segmentation is a fragmentation diagnostic, not a separate measure of whether the tokenizer can represent a word. Boundary F1 and the older pairwise morphology-consistency F1 must be interpreted separately.

Previously reported edit distance, normalized edit distance, exact morpheme
sequence match, pairwise consistency P/R/F1, contextual boundary crossing,
compression, roundtrip, vocabulary usage/shape, throughput and distributions
remain available. Contextual boundary preservation and isolated-word boundary
recall are not interchangeable. Full-corpus document statistics still use
capped samples for word-level morphology.

## Sunum için kısa sonuç

Morfoloji duyarlı tokenizasyon, otomatik TRmorph referansına göre morfem ve sınır
korunumunu artırıyor; buna karşılık belge düzeyinde kelime başına token sayısı
artıyor. Ham BPE, her kelime haznesi grubunda kayıtlı doğrulama BPB değerinde
önde. Ancak eski CETVEL değerlendirme kodunda hata bulunduğu için LLM görev
başarımı sıralamaları düzeltilmiş ortak protokolle yeniden ölçülmeden kesin
sonuç olarak sunulmamalı. Yeni tokenizer ölçümleri ile eski ve düzeltilmiş model
sonuçları ayrı tablolar halinde verilmiştir.

## Rebuild and source evidence

```bash
python -m scripts.build_tokenizer_study_results
python -m scripts.build_fresh_tokenizer_results
python -m scripts.build_tokenizer_matched_results
python -m scripts.build_model_study_results
python -m scripts.build_sft_configuration_results
python -m scripts.build_study_overview
MPLCONFIGDIR=/tmp/nanochat-mpl python -m scripts.plot_study_results
python -m scripts.build_study_brief  # requires reportlab 4.4.9 and matplotlib
```

Fresh measurements are produced with
[`runs/uhem_tokenizer_metrics_study.sbatch`](../../runs/uhem_tokenizer_metrics_study.sbatch)
in an isolated evaluation directory. Sample preparation records ordered raw
and annotated document hashes; finalization requires all 24 outputs to match
their respective split and rejects the consistency metric's non-sklearn fallback.
The [fresh manifest](fresh_tokenizer_metrics/manifest.json) records the source
snapshot, tokenizer hashes, environment, corpus samples and result hashes.

Recovered evidence is preserved under
[`artifacts/tokenizer_metrics_full_2026-06-23`](../../artifacts/tokenizer_metrics_full_2026-06-23/)
and [`artifacts/uhem_model_audit_2026-11-09`](../../artifacts/uhem_model_audit_2026-11-09/).
Historical GPU training and model benchmarks were recovered; this update runs
new CPU tokenizer evaluations only.
