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

| Deliverable | Contents |
| --- | --- |
| [Shareable PDF brief](../../output/pdf/nanochat_turk_study_results.pdf) | Main tables, controls, caveats and figures |
| [Fresh matched tokenizer results](tokenizer_fresh_results.md) | All requested metrics on equal train and validation samples; 500k word occurrences per tokenizer |
| [Recovered full-corpus metrics](tokenizer_full_corpus.md) | All 12 tokenizers over 93,211,827 training documents; job 496882 verified complete |
| [Earlier tokenizer metrics](tokenizer_results.md) | Original 50k-document results, external Kumru/Cosmos tokenizer references, previous metrics and definitions |
| [Pretraining and original model benchmarks](model_results.md) | Eleven-model BPB, token loss, training statistics and all 132 legacy CETVEL task scores |
| [Later models and evaluation corrections](later_model_results.md) | Continued d20, completed d32, corrected XQuAD and EACL table-compatible receipt |
| [Reproduction and provenance](reproduction.md) | Exact recovery revisions, optimizer changes and unchanged strict d32 contracts |
| [PNG and editable SVG figures](figures/) | Presentation figures with source hashes and explicit evaluation scope |

Machine-readable exports: [fresh tokenizer CSV](tokenizer_fresh_results.csv),
[fresh JSON](tokenizer_fresh_results.json), [full-corpus CSV](tokenizer_full_corpus.csv),
[model summary CSV](model_results.csv), [all model task scores](model_task_metrics.csv),
and [training statistics CSV](model_training_statistics.csv). Original detailed
JSONs preserve counts, distributions and source hashes.

## Findings that can be defended

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

The later d32 run reached 28,000 steps and 58.72B scheduled token positions.
Its **0.519040 BPB is a train-overlap diagnostic** because its validation shard
was also used for training. It must not be presented as a held-out improvement
over the original ablation. Its distinct benchmark protocols and scores are
fully listed in the later-model appendix.

## Requested metrics and interpretation

| Requested quantity | Reported definition |
| --- | --- |
| Vocabulary size | Exact total vocabulary, including nine special tokens |
| Subword efficiency | Both document tokens/regex word and isolated-word subword units/word |
| Morpheme preservation rate | Reference morpheme spans exactly coinciding with one token span; isolated-word measurement |
| Morpheme boundary consistency | Isolated-word internal-boundary micro precision, recall and F1; separate from older pairwise morphological consistency |
| Whole-word coverage | Lossless non-UNK words represented by one token in isolation |
| Character-only coverage | Words represented entirely by single Unicode-character tokens, with a second rate excluding trivial one-character words |
| Byte-only coverage | Words represented entirely by single-byte tokens; distinct from Unicode characters |
| Unknown tokens | Architecture-aware count/incidence where UNK exists; N/A for these twelve byte-complete BPE tokenizers |

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
python -m scripts.build_model_study_results
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
