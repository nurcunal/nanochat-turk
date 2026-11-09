# Archived tokenizer and morphology screening results

Archived measurements, consolidated 9 November 2026. This report retains the earlier twelve-tokenizer sample, two lossless external references, three historical normalizing references, a separate raw-text diagnostic and upstream analyzer screening. Full precision, original metrics and SHA-256 source hashes are in [JSON](tokenizer_results.json); flat tokenizer values are in [CSV](tokenizer_results.csv). For the subsequently recovered complete training-corpus pass, use the [full-corpus report](tokenizer_full_corpus.md).

## Historical 50,000-document intrinsic comparison

**Scope:** 50,000 FineWeb-2 Turkish **training** documents, 186,328,421 bytes and 23,042,031 regex-defined words. These are corpus diagnostics, not held-out generalization results. Values below are percentages where marked `%`.

| Vocabulary | Tokenizer | Tokens/word ↓ | Isolated tokens/word ↓ | Whole word: one token % | Exact morpheme sequence % ↑ | Preserved boundaries % ↑ | Morph consistency F1 ↑ | Word sample |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.6157 | 2.0312 | 31.89 | 33.42 | 16.05 | 0.3241 | 200,000 |
| 32,768 | MorphBPE / TRmorph | 1.8166 | 1.9821 | 34.95 | 42.58 | 54.31 | 0.5129 | 200,000 |
| 32,768 | MorphBPE / Zemberek | 1.7986 | 1.9748 | 34.79 | 40.40 | 40.94 | 0.4357 | 200,000 |
| 32,768 | MorphBPE / TurkishDelight | 1.7366 | 2.0357 | 32.73 | 37.54 | 35.57 | 0.4795 | 500,000 |
| 65,536 | Raw BPE | 1.4856 | 1.8322 | 39.44 | 40.04 | 10.61 | 0.2863 | 500,000 |
| 65,536 | MorphBPE / TRmorph | 1.7121 | 1.7625 | 43.17 | 50.56 | 50.69 | 0.4421 | 500,000 |
| 65,536 | MorphBPE / Zemberek | 1.7030 | 1.7616 | 42.78 | 47.82 | 37.55 | 0.3743 | 500,000 |
| 65,536 | MorphBPE / TurkishDelight | 1.6091 | 1.8192 | 40.76 | 45.56 | 28.21 | 0.4296 | 500,000 |
| 131,072 | Raw BPE | 1.3915 | 1.6372 | 49.58 | 47.01 | 6.54 | 0.2412 | 500,000 |
| 131,072 | MorphBPE / TRmorph | 1.6304 | 1.6027 | 50.73 | 56.66 | 46.49 | 0.3786 | 500,000 |
| 131,072 | MorphBPE / Zemberek | 1.6248 | 1.6026 | 50.17 | 53.80 | 34.13 | 0.3330 | 500,000 |
| 131,072 | MorphBPE / TurkishDelight | 1.5109 | 1.6481 | 48.56 | 51.37 | 21.89 | 0.3971 | 500,000 |

**Interpretation:** Lowest corpus fertility: Raw BPE at 32k, 64k and 128k. Lowest isolated fertility: Zemberek at 32k, 64k and 128k. The lead at 64k and 128k is at most 0.001 token/word at the displayed precision. Highest one-token coverage, exact-morpheme agreement and boundary preservation: TRmorph at 32k, 64k and 128k. Highest original pairwise consistency F1: TRmorph at 32k and 64k; TurkishDelight at 128k. TurkishDelight at 128k has both higher isolated fertility and lower one-token coverage than same-size raw BPE, despite its morphology improvements. The mixed 32k word caps make its word-level rankings descriptive rather than fully matched.

### Definitions and coverage

- **Vocabulary size:** actual vocabulary including nine special tokens for every local tokenizer.
- **Subword efficiency:** total encoded document tokens divided by regex word count (`\w+`). This includes punctuation/spacing tokens in the numerator. Isolated fertility encodes each sampled word separately and is reported alongside it.
- **Whole-word coverage:** fraction of sampled word occurrences represented by exactly one token when encoded in isolation. It is sensitive to whitespace conventions and differs from context-aware word coverage.
- **Morpheme preservation:** the archived **exact morpheme-sequence rate** counts words whose token pieces exactly match the automatic morpheme sequence. It is a strict word-level proxy. Individual-morpheme intact-token preservation was absent from these June measurements and is now separately measured in the [matched evaluation](tokenizer_fresh_results.md).
- **Boundary preservation:** `(reference boundaries - crossed boundaries) / reference boundaries`, using all 12,706,795 reference boundaries. This measures retained reference boundaries; it does not penalize extra token splits, so it is not boundary precision or F1.
- **Morphological consistency:** retain the original MorphBPE shared-token/shared-morpheme pair precision, recall and F1 (`k=100`, `C=50`, `N=10`). It must not be relabelled as boundary F1. Explicit boundary precision and F1 were not measured here.
- **Character-only coverage:** fraction of words split entirely into single Unicode-character tokens; it was not measured in the archived outputs. A UTF-8 byte fragment is not a character token.
- **UNK:** N/A for the twelve local byte-complete BPE/MorphBPE tokenizers, which have no UNK mechanism. Historical UNK counts were not recorded for the external references and remain unmeasured.
- **Roundtrip coverage:** all twelve local tokenizers had zero failures in 50,000 documents (100% exact decode/encode success). This establishes observed representability, not whole-word compression or morphology quality.

## Previously reported metrics

| Vocabulary | Tokenizer | Bytes/token ↑ | Morph edit distance ↓ | Normalized edit ↓ | Consistency P ↑ | Consistency R ↑ | Boundary crossed % ↓ | Used vocab % | Encode M tokens/s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 5.0051 | 1.6836 | 1.2083 | 0.9630 | 0.1949 | 83.95 | 99.70 | 4.923 |
| 32,768 | MorphBPE / TRmorph | 4.4514 | 1.4126 | 1.0166 | 0.9865 | 0.3466 | 45.69 | 99.62 | 5.112 |
| 32,768 | MorphBPE / Zemberek | 4.4959 | 1.4817 | 1.0605 | 0.9831 | 0.2799 | 59.06 | 99.50 | 5.076 |
| 32,768 | MorphBPE / TurkishDelight | 4.6564 | 1.5544 | 1.1335 | 0.9825 | 0.3172 | 64.43 | 99.21 | 3.759 |
| 65,536 | Raw BPE | 5.4434 | 1.4627 | 1.0124 | 0.9804 | 0.1677 | 89.39 | 99.71 | 3.239 |
| 65,536 | MorphBPE / TRmorph | 4.7230 | 1.1697 | 0.7982 | 0.9919 | 0.2845 | 49.31 | 99.25 | 3.659 |
| 65,536 | MorphBPE / Zemberek | 4.7485 | 1.2527 | 0.8560 | 0.9895 | 0.2309 | 62.45 | 99.01 | 3.611 |
| 65,536 | MorphBPE / TurkishDelight | 5.0254 | 1.3071 | 0.9094 | 0.9880 | 0.2745 | 71.79 | 98.36 | 3.409 |
| 131,072 | Raw BPE | 5.8114 | 1.2623 | 0.8188 | 0.9913 | 0.1373 | 93.46 | 99.35 | 3.133 |
| 131,072 | MorphBPE / TRmorph | 4.9597 | 1.0045 | 0.6388 | 0.9955 | 0.2338 | 53.51 | 97.83 | 3.284 |
| 131,072 | MorphBPE / Zemberek | 4.9770 | 1.0859 | 0.6982 | 0.9933 | 0.2001 | 65.87 | 97.53 | 3.351 |
| 131,072 | MorphBPE / TurkishDelight | 5.3521 | 1.1345 | 0.7437 | 0.9933 | 0.2482 | 78.11 | 96.34 | 3.271 |

**Interpretation:** Best corpus compression (most bytes/token): Raw BPE at 32k, 64k and 128k. Lowest raw and normalized morphology edit distance: TRmorph at 32k, 64k and 128k. Highest pairwise consistency precision: TRmorph at 32k, 64k and 128k. Highest pairwise consistency recall: TRmorph at 32k and 64k; TurkishDelight at 128k. Lowest crossed-boundary rate: TRmorph at 32k, 64k and 128k. Recorded encoding rates are not controlled raw-text speed comparisons.

The JSON also preserves all original distributions, unique-word measurements, resampling standard deviations, vocabulary shape, worker/thread counts and timing. The original [comparison](../tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md) remains available.

## External tokenizer references

These are tokenizer-only diagnostics; their pretrained models are **not** part of the controlled LLM ablation. Kumru (`vngrs-ai/Kumru-2B`) and Cosmos (`ytu-ce-cosmos/turkish-gpt2`) use the same 50,000 documents, 200,000 isolated/alignment word occurrences and 50,000 consistency word types. The archived external-reference job is 494176. Their vocabularies and runtime settings differ from the local tokenizers.

| Tokenizer | Vocabulary | Tokens/word | Isolated tokens/word | Whole word: one token % | Exact morph sequence % | Boundaries preserved % | Morph consistency F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| kumru_2b | 50,176 | 1.6677 | 1.6648 | 50.86 | 45.15 | 22.55 | 0.3436 |
| cosmos_turkish_gpt2 | 50,257 | 1.5570 | 1.8678 | 38.47 | 39.65 | 13.06 | 0.3001 |

**Interpretation:** Cosmos uses fewer corpus tokens per word; Kumru has lower isolated fertility and higher one-token coverage, exact-morpheme agreement, boundary preservation and pairwise consistency F1. The preferred reference depends on the metric; their different vocabularies prevent treating this as a controlled tokenizer-family ablation.

| Tokenizer | Bytes/token | Morph edit ↓ | Normalized edit ↓ | Consistency P | Consistency R | Crossed boundaries % |
|---|---:|---:|---:|---:|---:|---:|
| kumru_2b | 4.8488 | 1.3500 | 0.9195 | 0.9893 | 0.2079 | 77.45 |
| cosmos_turkish_gpt2 | 5.1938 | 1.4839 | 1.0309 | 0.9768 | 0.1773 | 86.94 |

**Interpretation:** Cosmos compresses the corpus into more bytes per token. Kumru has lower morphology edit distance and stronger consistency precision/recall, with fewer crossed reference boundaries. This repeats the compression-versus-morphology trade-off rather than identifying one universal winner.

## Archived encoding and evaluator performance

These are recorded timings, not a controlled tokenizer speed benchmark. `Encode M tok/s` divides token count by accumulated encoding time across workers; `Evaluator wall s` includes the rest of metric computation. The two use different time denominators. Worker/thread settings are shown so they are not mistaken for like-for-like speed measurements.

| Vocab | Tokenizer | Encode M tok/s | Encoding s (summed) | Evaluator wall s | Workers × threads | Used vocab % | Roundtrip failures |
|---:|---|---:|---:|---:|---|---:|---:|
| 32,768 | Raw BPE | 4.923 | 7.56 | 107.78 | 1 × 8 | 99.70 | 0 |
| 32,768 | MorphBPE / TRmorph | 5.112 | 8.19 | 112.06 | 1 × 8 | 99.62 | 0 |
| 32,768 | MorphBPE / Zemberek | 5.076 | 8.16 | 108.35 | 1 × 8 | 99.50 | 0 |
| 32,768 | MorphBPE / TurkishDelight | 3.759 | 10.65 | 109.30 | 1 × 32 | 99.21 | 0 |
| 65,536 | Raw BPE | 3.239 | 10.57 | 109.00 | 1 × 32 | 99.71 | 0 |
| 65,536 | MorphBPE / TRmorph | 3.659 | 10.78 | 111.74 | 1 × 32 | 99.25 | 0 |
| 65,536 | MorphBPE / Zemberek | 3.611 | 10.87 | 110.05 | 1 × 32 | 99.01 | 0 |
| 65,536 | MorphBPE / TurkishDelight | 3.409 | 10.88 | 110.83 | 1 × 32 | 98.36 | 0 |
| 131,072 | Raw BPE | 3.133 | 10.23 | 109.81 | 1 × 32 | 99.35 | 0 |
| 131,072 | MorphBPE / TRmorph | 3.284 | 11.44 | 111.51 | 1 × 32 | 97.83 | 0 |
| 131,072 | MorphBPE / Zemberek | 3.351 | 11.17 | 110.93 | 1 × 32 | 97.53 | 0 |
| 131,072 | MorphBPE / TurkishDelight | 3.271 | 10.64 | 110.40 | 1 × 32 | 96.34 | 0 |
| 50,176 | kumru_2b | 0.414 | 92.71 | 14.96 | 32 × 1 | 98.81 | 0 |
| 50,257 | cosmos_turkish_gpt2 | 0.364 | 98.55 | 11.84 | 32 × 1 | 99.13 | 0 |

**Interpretation:** Among the local tokenizer rows, highest recorded encoding tokens/s: TRmorph at 32k and 64k; Zemberek at 128k. All rows tie at zero observed roundtrip failures. These timings use tokenizer-dependent token counts, so they do not establish a controlled raw-text-speed ranking. High vocabulary usage alone is not evidence of better morphology.

## Earlier normalizing-tokenizer references from Git history

BERTurk, TURNA and VBART were published in [commit 2ff5305](https://github.com/nurcunal/nanochat-turk/commit/2ff53058540b5d2d00a9e832e7f02a19582faed4), then removed when the primary comparison narrowed to lossless raw-text BPE. The [three original files and immutable source links](../../artifacts/tokenizer_metrics_historical_normalizing_references/) are restored separately for completeness. All use the same 50,000 training documents and 500,000 isolated word occurrences.

**These are not lossless-tokenizer comparators.** Their observed roundtrip failures reflect normalization/decoding behavior; UNK counts were not recorded. A word producing one token need not preserve that word. Thus the recorded token counts below do not establish equivalent compression, coverage or morphology quality. No model-weight evaluation was performed for these references.

| Tokenizer | Vocab | Recorded tokens/word | Isolated tokens/word | Recorded one-token words % | Bytes/token | Roundtrip failures | Failure % |
|---|---:|---:|---:|---:|---:|---:|---:|
| berturk_cased | 32,000 | 1.5888 | 1.4201 | 70.03 | 5.0897 | 49,646 | 99.292 |
| turna | 32,100 | 1.5603 | 1.3966 | 71.81 | 5.1827 | 47,607 | 95.214 |
| vbart_large_base | 32,001 | 1.5603 | 1.3966 | 71.81 | 5.1827 | 47,607 | 95.214 |

**Interpretation:** TURNA and VBART have similar recorded token counts and one-token rates, but lose exact roundtrip on 95.214% of documents; BERTurk fails on 99.292%. These values cannot establish a lossless compression or whole-word-coverage winner.

| Tokenizer | Recorded boundary crossed % | Encode M tok/s | Used vocab % |
|---|---:|---:|---:|
| berturk_cased | 80.45 | 0.863 | 95.54 |
| turna | 80.15 | 0.541 | 99.55 |
| vbart_large_base | 80.15 | 0.580 | 99.87 |

**Interpretation:** BERTurk has the highest recorded encoding rate, while the boundary figures belong to tokenizers that normalize or alter text. Neither column supports a fair morphology or speed ranking against the lossless BPE tokenizers.

Boundary-crossing fields are preserved as historical diagnostics, not used in the lossless morphology ranking. Morphological edit/consistency measurements and worker/thread settings were absent. Original IDs: `dbmdz/bert-base-turkish-cased`, `boun-tabi-LMG/TURNA`, `vngrs-ai/VBART-Large-Base`.

## Separate raw-text diagnostic

The earlier Zemberek 32k diagnostic uses **10,000 raw training documents**, 48,975,401 bytes and 6,077,942 words. It has no morphology reference; zero boundary counts in its original file mean **not applicable**, not perfect boundary preservation. Its 200,000-word isolated sample is distinct from the corpus denominator. Runtime worker/thread settings were not recorded.

| Tokenizer | Tokens/word | Isolated tokens/word | One-token words % | Bytes/token | Encode M tok/s | Roundtrip failures |
|---|---:|---:|---:|---:|---:|---:|
| MorphBPE / Zemberek 32k | 1.8268 | 1.9748 | 34.79 | 4.4109 | 9.687 | 0 |

**Interpretation:** This single Zemberek diagnostic demonstrates lossless roundtrip on its 10,000-document sample. Without other tokenizers on this same diagnostic scope, it cannot identify the best tokenizer, and its absent morphology reference cannot establish boundary quality.

## Earlier morphology analyzer screening

These measurements evaluate the upstream analyzers that supplied training boundaries. They are separate from the finished tokenizer and LLM evaluations above. Rounded screening values come from the [archived benchmark log](../tokenizer_tests/segmenter_benchmark_status.md); its large original JSON outputs are not included in this repository bundle.

| Analyzer | Sample | Unique forms | Runtime s | Forms/s | Pieces/word (weighted) | Split % (weighted) | Fallback % (weighted) | Fallback % (types) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| identity | full first shard | 10,829,544 | 35.03 | 309,135 | 1.000 | 0.0 | 0.0 | not reported |
| TRmorph | full first shard, pre-repair | 10,829,544 | 1286.69 | 8,417 | 1.474 | 29.2 | 39.0 | 87.4 |
| TRmorph | hash-100k, corrected wrapper | 100,000 | 10.84 | 9,223 | 1.542 | 33.8 | 14.5 | 81.7 |
| Zemberek | hash-1k smoke | 1,000 | 2.00 | 500 | 1.468 | 24.7 | 42.6 | 25.0 |
| Zemberek | hash-100k, corrected wrapper | 100,000 | 29.99 | 3,335 | 1.447 | 31.6 | 42.5 | 23.3 |
| TurkishDelight | hash-100k | 100,000 | 65.29 | 1,532 | 1.503 | 36.2 | 0.0 | 0.0 |

**Interpretation:** On the common hash-100k screen, TRmorph has the highest measured processing rate (9,223 forms/s), while TurkishDelight has the lowest fallback rate (0%). Identity is a no-segmentation control. These analyzer readiness results do not rank the finished tokenizers or their LLMs.

The common hash-100k screen represents 12,095,210 weighted word occurrences; the full first shard represents 1,282,426,655 and the 1k smoke sample 76,484. The full TRmorph row predates case-only surface repair and analysis-tag filtering; its fallback rate is not a current-wrapper result. Different wrappers and scopes prevent a single speed ranking.

### Archived 500-item blind segmentation judge

The [judge report](../tokenizer_tests/codex_local_judge_results.md) used a disagreement-focused sample, blind candidate labels and a local scripted rubric. It is not a human-gold or corpus-representative score. Original [scores](../tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.scores.json) and [item judgments](../tokenizer_tests/judge_results/codex_local_hash_100k_judge_500.judgments.jsonl) remain available.

| Analyzer | Best count | Best % | Acceptable count | Acceptable % |
|---|---:|---:|---:|---:|
| TRmorph | 241 | 48.2 | 255 | 51.0 |
| TurkishDelight | 173 | 34.6 | 219 | 43.8 |
| Zemberek | 86 | 17.2 | 200 | 40.0 |
| Identity | 0 | 0.0 | 4 | 0.8 |

**Interpretation:** TRmorph leads both best selections (48.2%) and acceptability (51.0%); TurkishDelight is second on both. This supports TRmorph under this disagreement-focused scripted rubric, not a universal linguistic-quality or downstream-model win.

Confidence counts: high 148, medium 287, low 65; invalid judgments: 0. Screening source hashes and the full judge summary are retained in the consolidated JSON.

## Row-level source files

Every file below is copied in full into the report JSON with its SHA-256 hash; the CSV carries the same path and hash. Original result files are preserved unchanged.

| Vocabulary | Tokenizer | Original JSON |
|---:|---|---|
| 32,768 | Raw BPE | [bpe_32768_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/bpe_32768_metrics.json) |
| 32,768 | MorphBPE / TRmorph | [morphbpe_trmorph_32768_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_32768_metrics.json) |
| 32,768 | MorphBPE / Zemberek | [morphbpe_zemberek_32768_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_32768_metrics.json) |
| 32,768 | MorphBPE / TurkishDelight | [morphbpe_tdelight_32768_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_tdelight_32768_metrics.json) |
| 65,536 | Raw BPE | [bpe_65536_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/bpe_65536_metrics.json) |
| 65,536 | MorphBPE / TRmorph | [morphbpe_trmorph_65536_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_65536_metrics.json) |
| 65,536 | MorphBPE / Zemberek | [morphbpe_zemberek_65536_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_65536_metrics.json) |
| 65,536 | MorphBPE / TurkishDelight | [morphbpe_tdelight_65536_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_tdelight_65536_metrics.json) |
| 131,072 | Raw BPE | [bpe_131072_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/bpe_131072_metrics.json) |
| 131,072 | MorphBPE / TRmorph | [morphbpe_trmorph_131072_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_131072_metrics.json) |
| 131,072 | MorphBPE / Zemberek | [morphbpe_zemberek_131072_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_131072_metrics.json) |
| 131,072 | MorphBPE / TurkishDelight | [morphbpe_tdelight_131072_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_tdelight_131072_metrics.json) |
| 50,176 | kumru_2b | [kumru_2b_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/kumru_2b_metrics.json) |
| 50,257 | cosmos_turkish_gpt2 | [cosmos_turkish_gpt2_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/cosmos_turkish_gpt2_metrics.json) |
| 32,768 | MorphBPE / Zemberek | [morphbpe_zemberek_32768_raw_metrics.json](../../docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_32768_raw_metrics.json) |
| 32,000 | berturk_cased | [berturk_cased_metrics.json](../../artifacts/tokenizer_metrics_historical_normalizing_references/berturk_cased_metrics.json) |
| 32,100 | turna | [turna_metrics.json](../../artifacts/tokenizer_metrics_historical_normalizing_references/turna_metrics.json) |
| 32,001 | vbart_large_base | [vbart_large_base_metrics.json](../../artifacts/tokenizer_metrics_historical_normalizing_references/vbart_large_base_metrics.json) |

**Reading note:** These links identify the original evidence for each row; their order is not a performance ranking.

## Interpretation and limits

At every vocabulary size, MorphBPE preserves more TRmorph reference boundaries and has better recorded morphology edit distance and consistency than raw BPE, while requiring more document tokens per word. TRmorph training and evaluation share an analyzer, so these are automatic-reference results. The 32k morphology and isolated-word caps differ across treatments, preventing a fully matched word-level estimate. Corpus-level fertility and boundary counts use the common 50,000 documents.

Isolated-word fertility often improves with MorphBPE even though document tokens/word rises; the two measures capture different contexts. Job 496882 has now been verified COMPLETED and recovered separately in the [full-corpus report](tokenizer_full_corpus.md). The [completed matched evaluation](tokenizer_fresh_results.md) adds separate training/validation results and the previously absent coverage/boundary metrics for all twelve local tokenizers; no historical null is overwritten. No significance or causal advantage is claimed from these single-run diagnostics.

Rebuild: `python -m scripts.build_tokenizer_study_results`.
