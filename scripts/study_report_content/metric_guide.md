# How to read the study metrics

This guide explains the columns in the [results index](README.md), including
the [fresh tokenizer tables](tokenizer_fresh_results.md), [historical tokenizer
metrics](tokenizer_results.md), [model benchmarks](model_results.md),
[later configurations](model_configuration_results.md) and
[SFT screens](sft_configuration_screens.md). Definitions follow this repository's
evaluators and saved protocols. A better metric value means improvement on
that metric's objective, not proof of a universally better tokenizer or model.

## Direction, scale and missing values

| Label | Meaning |
| --- | --- |
| **↑ Higher is better** | A larger value is preferable for the stated objective, under a comparable evaluation. |
| **↓ Lower is better** | A smaller value is preferable for the stated objective, under a comparable evaluation. |
| **↔ Descriptive / trade-off** | There is no universal better direction. This is a setting, sample characteristic or diagnostic. |
| **Conditional** | The direction applies only to the stated objective, such as compression or runtime, and requires matching conditions. |
| **% versus 0–1** | `44.30%` equals `0.4430`. Read the column's scale before comparing numbers. |
| **Δ / pp** | Delta is candidate minus baseline. Percentage points are differences on the percentage scale: 46.18% − 45.14% = +1.04 pp, not a 1.04% relative improvement. |
| **N/A, null or —** | Not applicable, undefined or unavailable as explained beside the table. It does not mean zero. A missing model is not a zero-scoring model. |

Compare tokenizer families **within the same vocabulary tier and sample**.
Across tiers the model depth also changes. Equal training token counts do not
guarantee equal raw-text exposure. The original downstream benchmark rankings
remain provisional because of the [recovered evaluator bugs](later_model_results.md).

## Best-score highlighting

**Bold measured values** mark the best score in the table's comparable group:
minimum for a selected lower-is-better metric and maximum for a selected
higher-is-better metric. All ties at the displayed precision are highlighted;
the full-precision JSON/CSV measurements remain unchanged. A bold value is a
descriptive winner, not a statistical-significance claim.

- Tokenizer and original model comparisons stay within one vocabulary tier,
  split and sample scope. Historical word-level columns with different sample
  caps are grouped by cap; a single eligible row does not win by default.
- Per-task model tables highlight the best model for each task. They do not
  compare a QA score against a classification score or mix evaluator protocols.
- SFT screens compare candidates within the same curriculum and metric; the
  source/candidate behavior table groups rows by probe suite.
- Missing values, configuration columns, token losses across different
  tokenizers, uncontrolled runtime comparisons and standalone single-model
  category scores have no highlighted winner. Historical model rankings remain
  provisional even when their recorded best value is bold.

## Tokenization size, efficiency and coverage

A token is one vocabulary entry emitted by the tokenizer. A word here is a
Unicode regex `\w+` match; it is not necessarily a linguistic word. Document
measurements encode whole documents, while isolated-word measurements encode
each sampled word separately, without its surrounding spaces or punctuation.

| Metric / column | What it measures | Better direction and interpretation |
| --- | --- | --- |
| Vocabulary size / vocab | Total available token IDs, including nine special tokens in each local tokenizer. | **↔ Trade-off.** A larger vocabulary can shorten sequences but changes the embedding/output layers and resource costs. |
| Corpus tokens/word / token fertility | All document tokens divided by regex word occurrences. The numerator includes punctuation and spacing tokens. | **↓ Lower is better for compact document encoding**, on the same losslessly represented text. It does not measure morphological accuracy. |
| Isolated tokens/word / subword units per word | Total tokens from encoding sampled words separately, divided by sampled word occurrences. | **↓ Lower is better for isolated-word compactness.** It can favor a different tokenizer from document fertility because context changes segmentation. |
| Bytes/token (also called compression ratio here) and characters/token | Original UTF-8 bytes or Unicode characters divided by emitted tokens. | **↑ Higher is better for compact encoding**, when the same text is preserved. These are not compressed-file sizes and do not account for token-ID storage width. |
| Tokens/byte and tokens/character | Emitted tokens divided by original bytes or characters; reciprocals of the corresponding measures above. | **↓ Lower is better for compact encoding**, under the same lossless-text condition. |
| One-token words / whole-word coverage | Fraction of word occurrences encoded losslessly as exactly one non-UNK token in isolation. | **↑ Higher means more whole words fit in one token.** This is a compactness objective; a whole-word token can merge several morphemes and reduce morphology scores. |
| Character-only words | Fraction of words whose every token spans exactly one Unicode character. Includes words already one character long. | **↔ Fragmentation diagnostic.** A lower value often indicates fewer extreme splits, but trivial one-character words and byte-fragment splits make this an incomplete ranking. |
| Multi-character words split only into characters | Among words longer than one character, the fraction encoded entirely as one-character tokens. | **↓ Lower means less character-level fragmentation in this specific diagnostic.** It does not measure all fragmentation or linguistic quality. |
| Byte-only words | Fraction of words whose every token spans exactly one UTF-8 byte. | **↔ Byte-granularity diagnostic.** Includes trivial one-byte words. A byte fragment is not necessarily a complete Unicode character; this is not the same as character-only coverage. |
| UNK token count / rate | Number of emitted unknown tokens, or that count divided by all tokens in the stated word/document sample. | **↓ Lower is better for representability; zero is ideal where UNK exists.** All 12 local byte-complete tokenizers have no UNK mechanism, so their value is **N/A**, not a measured zero. |
| Words or documents with UNK | Number or fraction of sampled words/documents containing at least one unknown token. | **↓ Lower is better for representability**, only for UNK-enabled tokenizers. |
| Unknown-free word rate | Fraction of sampled word occurrences containing no UNK. | **↑ Higher is better for avoiding unknown tokens.** It does not by itself prove exact surface preservation. |
| Roundtrip failures / failure rate | Documents for which decoding the encoded text does not reproduce the evaluated text exactly. | **↓ Lower is better for exact reversibility; zero is ideal.** Normalizing tokenizers may intentionally change text and need separate interpretation. |
| Roundtrip success / lossless surface coverage | Fraction reconstructed exactly; word-level lossless coverage also requires an alignable, contiguous partition with no UNK. | **↑ Higher is better for exact representability; 100% is ideal.** It says nothing about token count or morpheme alignment. |
| Used vocabulary % / unique token rate | Distinct token IDs observed in the sample divided by total vocabulary size. | **↔ Diagnostic.** Higher usage is not automatically better: sample size, vocabulary allocation and rare tokens change this measure. |
| Vocabulary shape / token lengths / document-token distributions | Length statistics, percentiles and counts describing vocabulary pieces or encoded documents. | **↔ Descriptive.** These show where token capacity and sequence lengths are concentrated; they are not standalone quality scores. |

“Coverage” therefore needs a qualifier. All twelve local tokenizers can represent
the sampled text without document roundtrip failures, yet their one-token,
character-only and morpheme coverage values differ. Historical normalizing
references with failed roundtrips must not be treated as equivalent lossless
compression results. Historical UNK fields that were not measured stay missing.

## Morphology: what is preserved, and how consistently?

A reference morpheme is a piece supplied by the automatic morphology analyzer.
These are **automatic TRmorph references, not human-gold annotations**; this can
favor the tokenizer trained with the same analyzer. The new literal boundary
metrics and the old pairwise consistency metrics answer different questions.

| Metric / column | What it measures | Better direction and interpretation |
| --- | --- | --- |
| Individual morpheme preservation / morphemes intact | Reference morpheme spans that match exactly one token span, divided by all scored reference morphemes. Splitting a morpheme or merging it with another loses that exact match. | **↑ Higher is better for agreement with reference morpheme spans.** This is a morpheme-level rate. |
| Exact morpheme-sequence rate | Fraction of sampled words whose entire token-piece sequence equals the reference morpheme-piece sequence. | **↑ Higher is better for complete reference segmentation.** This older, stricter word-level measure is not individual-morpheme preservation. |
| Internal boundary precision (P) | Correct token boundaries divided by all predicted internal token boundaries. | **↑ Higher is better for avoiding extra splits.** A token boundary inside a morpheme counts against precision. Word edges are excluded. |
| Internal boundary recall (R) | Correct token boundaries divided by all internal reference morpheme boundaries. | **↑ Higher is better for retaining reference boundaries.** Merging across a boundary counts against recall. |
| Internal boundary F1 | `2 × matched boundaries / (predicted boundaries + reference boundaries)`, pooling the scored occurrences. | **↑ Higher is better for balancing extra splits and missed boundaries.** This is the new isolated-word boundary consistency measure. |
| Exact boundary-word rate | Fraction of scored words whose complete internal token-boundary set equals the reference-boundary set. | **↑ Higher is better for exact boundary agreement.** Unalignable words are excluded and reported separately. |
| Contextual boundary-crossing rate | Reference boundaries crossed by a token in document context, divided by reference boundaries. | **↓ Lower is better for avoiding merges across reference boundaries.** This does not penalize additional splits inside morphemes. |
| Documents with boundary crossing / crossing tokens per 1,000 | Affected documents divided by eligible boundary-bearing documents, or crossing-token count per 1,000 document tokens, respectively. | **↓ Lower means fewer crossings under the stated denominator.** Neither is the fraction of reference boundaries crossed. |
| Contextual boundary preservation | `1 − contextual boundary-crossing rate`. | **↑ Higher is better for retaining contextual reference boundaries.** It is not precision/F1 and differs from isolated-word recall because the context and denominator differ. |
| Morphological edit distance (MED) | Mean number of insertions, deletions or substitutions needed to convert the reference morpheme-piece sequence into the token-piece sequence. Each operation costs one; pieces are compared as bytes. | **↓ Lower is better; zero means exact sequence agreement.** This edits sequences of pieces, not individual letters. |
| Normalized morphological edit distance | For each word, edit distance divided by its reference morpheme count, then averaged over words. | **↓ Lower is better.** It can exceed 1; it is not an accuracy or a percentage. |
| Pairwise morphological consistency precision | Among sampled pairs of words sharing at least one token, the fraction also sharing at least one reference morpheme. | **↑ Higher is better for pairwise agreement.** The shared token need not itself equal the shared morpheme. |
| Pairwise morphological consistency recall | Among sampled pairs sharing at least one reference morpheme, the fraction also sharing at least one token. | **↑ Higher is better for retaining reference-related word pairs.** It does not count literal boundary matches. |
| Pairwise morphological consistency F1 | Harmonic mean of the pairwise precision and recall within a resample, averaged over the configured resamples. | **↑ Higher is better for this pairwise diagnostic.** Mean F1 need not equal the harmonic mean of the separately averaged P and R. |
| Unalignable / excluded words | Words whose token spans cannot be aligned losslessly to the reference surface. These are excluded from the new morphology denominator. | **↓ Fewer exclusions means broader scoring coverage**, under the same sample. Check exclusions before interpreting a high morphology score. |

**Illustrative example:** suppose the reference for `evlerimiz` is
`ev | ler | imiz`, and a tokenizer emits `evler | imiz`. It emits 2 subwords
per word. Only `imiz` is an exact morpheme/token span, so morpheme preservation
is **1/3 = 33.33%**. One predicted internal boundary matches one of the two
reference boundaries: precision **100%**, recall **50%**, and F1 **66.67%**.
This is a constructed example explaining the definitions, not a measured row.

## Pretraining and downstream LLM performance

| Metric / column | What it measures | Better direction and interpretation |
| --- | --- | --- |
| Validation BPB / bits per byte | Total negative log-probability, converted to bits, divided by evaluated UTF-8 bytes. Special-token and masked target positions contribute neither bytes nor loss. | **↓ Lower is better for predicting the evaluated text.** Byte normalization helps compare tokenizers, but the raw-text spans, split and protocol must also match. |
| Final/minimum BPB | BPB at the final checkpoint, or the lowest recorded checkpoint BPB. | **↓ Lower is better on the same evaluation.** Equality of final and minimum does not prove every preceding step improved. |
| Training loss / token cross-entropy | Prediction loss per training token, using the run's tokenization and training objective. The archived final printed value is a debiased exponential moving average, not a separate full-corpus evaluation. | **↓ Lower is better within the same tokenizer/objective.** Different token units prevent direct quality ranking across tokenizers. |
| SFT validation BPB | Byte-normalized prediction loss under a particular supervised fine-tuning dataset and masking/scoring protocol. | **↓ Lower is better on that dataset/protocol.** Different curricula, pretraining BPB and behavior-suite scores are not interchangeable. |
| Accuracy (`acc`) | Fraction of evaluation examples for which the selected answer/label is correct. | **↑ Higher is better**, on the same task, split and scoring protocol. Usually reported on 0–1 here. |
| Normalized-choice accuracy (`acc_norm`) | Accuracy after selecting the answer using the benchmark's length-normalized likelihood score. | **↑ Higher is better under that choice-scoring rule.** It is still accuracy, not F1. |
| Classification F1 | Harmonic mean of label precision and recall using the task's specified binary, macro or other averaging rule. | **↑ Higher is better under the same averaging rule.** Macro-F1 weights classes equally; do not assume every field named `f1` is macro-F1. |
| Core-11 macro | Unweighted mean of the eleven classification/likelihood task scores in the original common slice. XQuAD is excluded. | **↑ Higher is better under a valid common protocol.** The archived 0–1 values are provisional because the old likelihood adapter had boundary errors. |
| QA / XQuAD F1 | Overlap between predicted and reference answer words, combining answer precision and recall under the evaluator's normalization. | **↑ Higher is better; 100 is ideal on XQuAD's reported 0–100 scale.** Other QA tasks may use 0–1. It is unrelated to morphology F1 or LLM subword counts. Compare generation settings and evaluator versions. |
| Exact match (EM) | Fraction of predictions matching the reference exactly after the task's specified normalization or stripping. | **↑ Higher is better.** QA normalized EM and GEC stripped EM use different rules; the report retains each task's scale. |
| BLEU | Reference n-gram overlap with a brevity penalty, under the recorded translation scorer. | **↑ Higher is better for that reference-based metric.** Tokenization and scorer settings must match; it is not a complete translation-quality measure. |
| ROUGE | Reference-overlap measures, including n-gram or longest-common-subsequence variants. | **↑ Higher is better for the specified variant.** Compare the same recall/precision/F1 variant and scale. |
| Word error rate (WER) | Word substitutions + deletions + insertions, divided by the reference word count; the pinned translation task averages per-example WER. | **↓ Lower is better; zero means no word errors.** It can exceed 1 (or 100%); native values must not silently be relabeled as percentages. |
| BERTScore F1 | Similarity between prediction and reference using contextual token embeddings, combined into F1. | **↑ Higher is better for the pinned scorer.** The table-compatible summarization receipt uses TabiBERT layer 22 without baseline rescaling; it is not an accuracy percentage. |
| COMET-22 | Learned translation-quality score from the pinned COMET model/protocol. | **↑ Higher is better under that scorer.** It is not a percentage of correct translations; compare matching model revisions and scaling. |
| Table-compatible category / overall score | Task scores aggregated under the saved `cetvel-eacl2026-table4-compatible-v1` rules. Overall is the unweighted mean of seven category scores. | **↑ Higher is better within that protocol.** Its 0–100 reporting scale is not a universal accuracy scale, and it cannot be compared directly with core-11. |
| Δ BPB versus raw BPE | Candidate BPB minus the same-tier raw-BPE BPB. | **↓ Lower is better; a negative delta is a BPB improvement.** |
| Δ accuracy/core-11/F1 | Candidate score minus its same-protocol baseline, in the stated units. | **↑ Higher is better; a positive delta is a score increase.** Legacy differences remain provisional and do not establish statistical significance. |
| Per-task standard error | Estimated sampling uncertainty of the recorded task statistic. | **↔ Uncertainty, not quality.** A smaller error bar is more precise, not necessarily a better model; it does not measure variation across training seeds. |

The seven table-compatible category labels mean **QA** (question answering),
**MCQA** (the receipt's multiple-choice/judgment group, including check-worthiness
and relevance), **TC** (text classification), **NLI** (natural-language inference),
**SUM** (summarization), **MT** (machine translation) and **GEC** (grammatical
error correction). Their different scoring methods are preserved in the
[later-model report](later_model_results.md) and original receipts.

The later d32 model's **0.519040 BPB includes training text in validation**.
It is a train-overlap diagnostic, not evidence of held-out superiority.
The same-checkpoint XQuAD change after repairing the evaluator is an evaluator
correction, not a tokenizer or training gain.

## Runtime, model configuration and SFT diagnostics

| Metric / column | What it measures | Better direction and interpretation |
| --- | --- | --- |
| Encoding tokens/s, documents/s; evaluation examples/s; training tokens/s | Work units divided by the recorded time denominator. | **↑ Higher is better for throughput on matched work and hardware.** Tokens/s across tokenizers can change because one emits more tokens; document/byte throughput may be more comparable. |
| Encoding seconds (summed) | Accumulated tokenizer-encoding time across workers. | **↓ Lower is better for the same work/settings.** It is not elapsed wall time and can exceed wall time under parallelism. |
| Document-pass wall seconds (`metric_wall_seconds`) | Time for the tokenizer evaluator's document pass, including its I/O, roundtrip and contextual-boundary work; excludes later isolated-word, coverage and morphology calculations and preliminary parallel word collection. | **↓ Lower is better for that matched pass and resources.** This is not total time for all reported metrics. |
| Model-evaluator elapsed / Slurm elapsed | Wall time for the stated model evaluation or scheduler job, including its particular overheads. | **↓ Lower is better for the same completed work and resources.** Different job contents prevent a direct ranking. |
| Training-loop minutes/hours | The run's accumulated training-loop counter, distinct from evaluation, setup, checkpointing and some warmup overhead. | **↓ Lower is better for matched training work.** A continuation counter may include the earlier phase. A failed job is not a successful fast run. |
| Step milliseconds | Time per logged optimizer step. | **↓ Lower is better for matched step size/model/hardware.** Compare batch and sequence lengths too. |
| Speed/raw | Candidate throughput divided by raw-BPE throughput in the stated tier. | **↑ Higher is faster for that benchmark; 1× is equal and >1× faster.** It is not a quality score. |
| Peak memory (MiB) | Recorded peak process/device memory under the logger's scope. | **↓ Lower is preferable for fitting the same work into memory.** It is not necessarily total cluster RAM or directly comparable across model sizes. |
| MFU / BF16 MFU % | Model FLOP utilization relative to the logger's assumed hardware peak and FLOP convention. | **↔ Utilization diagnostic.** Higher can indicate better use of the same hardware, but it is not model quality or proof of lower cost. |
| Estimated FLOPs | Estimated training computation under the implementation's counting convention. | **↔ Compute-budget descriptor.** Lower cost is desirable only when comparing equivalent work and achieved quality. |
| Parameters, depth, width, heads, vocabulary and context length | Model capacity and architecture settings; context length is measured in tokenizer-specific tokens. | **↔ Configuration/trade-offs.** Larger is not automatically better; across-tier changes confound a tokenizer-only comparison. |
| Steps, checkpoint, batch, token positions, learning rate | Training progress, budget and optimizer settings. A checkpoint number identifies progress, not a performance score. | **↔ Configuration.** More steps or a larger learning rate is not intrinsically better. Token positions are not unique documents or bytes. |
| Workers, threads, GPU count and hardware | Parallel execution resources. | **↔ Conditions for interpreting cost and speed**, not tokenizer/model quality scores. |
| SFT probe passes / pass rate | Number or fraction of custom selection cases passing the recorded rules. | **↑ Higher is better on that suite.** These probes were used for model selection and are not independent held-out capability estimates. |
| Empty outputs, echo, repetition | Counts flagged by the custom behavior diagnostics. | **↓ Lower is better for avoiding those flagged behaviors**, under the same suite; independent flags need not sum to suite size. |
| Assistant-end termination | Outputs ending with the expected assistant-end marker. | **↑ Higher indicates more compliant termination** under the chat protocol, but does not establish correct or useful answers. |
| Qualified / verified / retained | Whether a checkpoint passed the specified verification or selection gates. | **↔ Status with explicit criteria.** A higher aggregate score alone can still fail a category or behavior gate. Verification is not capability evaluation. |
| Analyzer forms/s and runtime | Speed of the morphology analyzer supplying training boundaries. | **↑ Throughput / ↓ time** on identical inputs, wrappers and hardware; separate from final-tokenizer inference speed. |
| Analyzer split %, pieces/word and fallback % | Segmentation frequency/granularity and use of fallback behavior. | **↔ Diagnostics.** More splitting or less fallback does not itself prove correct morphology; identity can have zero fallback without doing analysis. |
| Blind-judge best / acceptable count or % | Preferences under the archived scripted rubric and disagreement-focused sample. | **↑ Higher is better under that rubric only.** It is not a representative human-gold corpus accuracy. |
| Sample size, scored/excluded counts, unique words, sample hashes | The denominator and identity of the evidence used for a metric. | **↔ Provenance.** Different sample sizes/caps must not silently be treated as the same experiment. |
| Resampling standard deviation / percentiles | Variation across metric resamples or the distribution explicitly named by the field. | **↔ Distribution/uncertainty.** Resampling variation is not repeated-training-seed uncertainty or a confidence interval for tokenizer differences. |

Occurrence-weighted word metrics count repeated words repeatedly; unique-type
metrics count each word type once. Full-corpus document statistics can still
use capped word samples for morphology. The per-row sample caps and source
hashes are retained so these distinctions remain visible.

## Measurement sources

- [Tokenizer evaluator](../../scripts/tokenizer_metrics.py): fertility, coverage,
  exact spans, boundary agreement, edit distance and pairwise consistency.
- [Byte-normalized model evaluator](../../nanochat/loss_eval.py): BPB calculation.
- [Model and protocol report](later_model_results.md): benchmark variants,
  corrected-generation evidence and links to original scoring receipts.
- [Pinned CETVEL translation metrics](https://github.com/KUIS-AI/cetvel/blob/6119c517e06cce23aaeac103ce3504c9380655e3/tasks/wmt2016/metrics.py)
  and [translation aggregation configuration](https://github.com/KUIS-AI/cetvel/blob/6119c517e06cce23aaeac103ce3504c9380655e3/tasks/wmt2016/tr_en_prompt.yaml),
  and [summarization metrics](https://github.com/KUIS-AI/cetvel/blob/6119c517e06cce23aaeac103ce3504c9380655e3/tasks/mlsum/metrics.py):
  BLEU/WER/ROUGE aggregation and native scales.
- [Reproduction notes](reproduction.md): saved measurements, source revisions,
  missing evidence and the conditions required for meaningful comparisons.

**Türkçe kısa rehber:** ↑ belirtilen amaç için yüksek değer daha iyi;
↓ düşük değer daha iyi; ↔ tek başına iyi/kötü sıralaması olmayan tanım veya
ödünleşim. Daha az token daha sıkı kodlama, daha yüksek morfem/sınır F1 ise
referans morfolojiyle daha iyi uyum demektir. Bunlar aynı hedef değildir;
yüksek bir tokenizer metriği otomatik olarak daha başarılı bir LLM anlamına gelmez.
