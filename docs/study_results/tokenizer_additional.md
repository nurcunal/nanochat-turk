# Additional archived tokenizer measurements

This supplement adds previously uncollected UHeM results: two complete twelve-tokenizer training-prefix comparisons and raw-BPE-only train/validation diagnostics used by d32 preflight. They are archived measurements recovered on 9 November 2026, not new model training or evaluation. [Original JSONs and scheduler evidence](../../artifacts/tokenizer_metrics_additional_2026-11-09/) are preserved byte-for-byte. Download [JSON](tokenizer_additional.json) or [CSV](tokenizer_additional.csv).

The 100k, 1m and [full-corpus](tokenizer_full_corpus.md) passes overlap. Their isolated-word/alignment metrics remain capped at 500,000 occurrences and consistency at 50,000 word types; repeated word-level results are not independent replications. Every comparison uses the automatic TRmorph reference, with its associated analyzer bias.

## 100,000-document training prefix

Job **496924**: COMPLETED, exit 0:0, elapsed 00:06:46. Every row covers 100,000 training documents, 339,412,790 UTF-8 bytes and 41,919,065 regex words. All use 32 metric workers and one tokenizer thread per worker.

| Vocab | Tokenizer | Tokens/word ↓ | Isolated tokens/word ↓ | One-token words % | Bytes/token ↑ | Exact morph sequence % | Boundaries preserved % | Morph edit ↓ | Consistency F1 ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.5976 | 2.0472 | 31.05 | 5.0681 | 32.46 | 15.75 | 1.7096 | 0.3241 |
| 32,768 | MorphBPE / TRmorph | 1.8021 | 1.9890 | 34.09 | 4.4930 | 41.54 | 54.23 | 1.4319 | 0.5129 |
| 32,768 | MorphBPE / Zemberek | 1.7845 | 1.9810 | 34.06 | 4.5373 | 39.50 | 40.89 | 1.4986 | 0.4357 |
| 32,768 | MorphBPE / TurkishDelight | 1.7217 | 2.0357 | 32.73 | 4.7029 | 37.54 | 35.52 | 1.5544 | 0.4795 |
| 65,536 | Raw BPE | 1.4689 | 1.8322 | 39.44 | 5.5123 | 40.04 | 10.33 | 1.4627 | 0.2863 |
| 65,536 | MorphBPE / TRmorph | 1.6992 | 1.7625 | 43.17 | 4.7651 | 50.56 | 50.59 | 1.1697 | 0.4421 |
| 65,536 | MorphBPE / Zemberek | 1.6899 | 1.7616 | 42.78 | 4.7915 | 47.82 | 37.47 | 1.2527 | 0.3743 |
| 65,536 | MorphBPE / TurkishDelight | 1.5953 | 1.8192 | 40.76 | 5.0755 | 45.56 | 28.13 | 1.3071 | 0.4296 |
| 131,072 | Raw BPE | 1.3770 | 1.6372 | 49.58 | 5.8801 | 47.01 | 6.32 | 1.2623 | 0.2412 |
| 131,072 | MorphBPE / TRmorph | 1.6191 | 1.6027 | 50.73 | 5.0010 | 56.66 | 46.37 | 1.0045 | 0.3786 |
| 131,072 | MorphBPE / Zemberek | 1.6133 | 1.6026 | 50.17 | 5.0188 | 53.80 | 34.03 | 1.0859 | 0.3330 |
| 131,072 | MorphBPE / TurkishDelight | 1.4985 | 1.6481 | 48.56 | 5.4035 | 51.37 | 21.75 | 1.1345 | 0.3971 |

**Interpretation:** Lowest corpus fertility: Raw BPE at 32k, 64k and 128k. Lowest isolated fertility: Zemberek at 32k, 64k and 128k. The lead at 64k and 128k is at most 0.001 token/word at the displayed precision. Lowest morphology edit distance: TRmorph at 32k, 64k and 128k. Highest one-token coverage, exact-morpheme agreement and boundary preservation: TRmorph at 32k, 64k and 128k. Highest original pairwise consistency F1: TRmorph at 32k and 64k; TurkishDelight at 128k. TurkishDelight at 128k has both higher isolated fertility and lower one-token coverage than same-size raw BPE, despite its morphology improvements.

| Vocab | Tokenizer | Normalized morph edit | Consistency P | Consistency R | F1 resample SD |
|---:|---|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.2328 | 0.9630 | 0.1949 | 0.0079 |
| 32,768 | MorphBPE / TRmorph | 1.0362 | 0.9865 | 0.3466 | 0.0046 |
| 32,768 | MorphBPE / Zemberek | 1.0773 | 0.9831 | 0.2799 | 0.0089 |
| 32,768 | MorphBPE / TurkishDelight | 1.1335 | 0.9825 | 0.3172 | 0.0072 |
| 65,536 | Raw BPE | 1.0124 | 0.9804 | 0.1677 | 0.0085 |
| 65,536 | MorphBPE / TRmorph | 0.7982 | 0.9919 | 0.2845 | 0.0043 |
| 65,536 | MorphBPE / Zemberek | 0.8560 | 0.9895 | 0.2309 | 0.0089 |
| 65,536 | MorphBPE / TurkishDelight | 0.9094 | 0.9880 | 0.2745 | 0.0060 |
| 131,072 | Raw BPE | 0.8188 | 0.9913 | 0.1373 | 0.0079 |
| 131,072 | MorphBPE / TRmorph | 0.6388 | 0.9955 | 0.2338 | 0.0036 |
| 131,072 | MorphBPE / Zemberek | 0.6982 | 0.9933 | 0.2001 | 0.0091 |
| 131,072 | MorphBPE / TurkishDelight | 0.7437 | 0.9933 | 0.2482 | 0.0062 |

**Interpretation:** Lowest normalized morphology edit distance: TRmorph at 32k, 64k and 128k. Highest pairwise consistency precision: TRmorph at 32k, 64k and 128k. Highest pairwise consistency recall: TRmorph at 32k and 64k; TurkishDelight at 128k. Lower resample SD is not a quality win or a confidence interval; it measures variability of the sampled consistency calculation.

### Archived encoding and evaluator performance

These are recorded timings, not a controlled tokenizer speed benchmark. `Encode M tok/s` divides token count by accumulated encoding time across workers; `Evaluator wall s` includes the rest of metric computation. The two use different time denominators. Worker/thread settings are shown so they are not mistaken for like-for-like speed measurements.

| Vocab | Tokenizer | Encode M tok/s | Encoding s (summed) | Evaluator wall s | Workers × threads | Used vocab % | Roundtrip failures |
|---:|---|---:|---:|---:|---|---:|---:|
| 32,768 | Raw BPE | 1.293 | 51.79 | 11.22 | 32 × 1 | 99.74 | 0 |
| 32,768 | MorphBPE / TRmorph | 1.351 | 55.92 | 11.31 | 32 × 1 | 99.68 | 0 |
| 32,768 | MorphBPE / Zemberek | 1.323 | 56.54 | 11.33 | 32 × 1 | 99.60 | 0 |
| 32,768 | MorphBPE / TurkishDelight | 1.313 | 54.97 | 11.34 | 32 × 1 | 99.39 | 0 |
| 65,536 | Raw BPE | 1.236 | 49.82 | 11.17 | 32 × 1 | 99.79 | 0 |
| 65,536 | MorphBPE / TRmorph | 1.287 | 55.35 | 11.40 | 32 × 1 | 99.53 | 0 |
| 65,536 | MorphBPE / Zemberek | 1.283 | 55.21 | 11.53 | 32 × 1 | 99.36 | 0 |
| 65,536 | MorphBPE / TurkishDelight | 1.237 | 54.06 | 11.43 | 32 × 1 | 98.83 | 0 |
| 131,072 | Raw BPE | 1.176 | 49.10 | 11.26 | 32 × 1 | 99.61 | 0 |
| 131,072 | MorphBPE / TRmorph | 1.216 | 55.79 | 11.47 | 32 × 1 | 98.82 | 0 |
| 131,072 | MorphBPE / Zemberek | 1.195 | 56.59 | 11.64 | 32 × 1 | 98.54 | 0 |
| 131,072 | MorphBPE / TurkishDelight | 1.156 | 54.34 | 11.45 | 32 × 1 | 97.45 | 0 |

**Interpretation:** Among the local tokenizer rows, highest recorded encoding tokens/s: TRmorph at 32k, 64k and 128k. All rows tie at zero observed roundtrip failures. These timings use tokenizer-dependent token counts, so they do not establish a controlled raw-text-speed ranking. High vocabulary usage alone is not evidence of better morphology.

## 1,000,000-document training prefix

Job **496925**: COMPLETED, exit 0:0, elapsed 00:24:03. Every row covers 1,000,000 training documents, 3,309,498,345 UTF-8 bytes and 405,115,789 regex words. All use 32 metric workers and one tokenizer thread per worker.

| Vocab | Tokenizer | Tokens/word ↓ | Isolated tokens/word ↓ | One-token words % | Bytes/token ↑ | Exact morph sequence % | Boundaries preserved % | Morph edit ↓ | Consistency F1 ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.5519 | 2.0472 | 31.05 | 5.2641 | 32.46 | 15.04 | 1.7096 | 0.3241 |
| 32,768 | MorphBPE / TRmorph | 1.7711 | 1.9890 | 34.09 | 4.6126 | 41.54 | 54.12 | 1.4319 | 0.5129 |
| 32,768 | MorphBPE / Zemberek | 1.7531 | 1.9810 | 34.06 | 4.6600 | 39.50 | 40.49 | 1.4986 | 0.4357 |
| 32,768 | MorphBPE / TurkishDelight | 1.6854 | 2.0357 | 32.73 | 4.8470 | 37.54 | 34.84 | 1.5544 | 0.4795 |
| 65,536 | Raw BPE | 1.4293 | 1.8322 | 39.44 | 5.7154 | 40.04 | 9.69 | 1.4627 | 0.2863 |
| 65,536 | MorphBPE / TRmorph | 1.6729 | 1.7625 | 43.17 | 4.8832 | 50.56 | 50.49 | 1.1697 | 0.4421 |
| 65,536 | MorphBPE / Zemberek | 1.6631 | 1.7616 | 42.78 | 4.9121 | 47.82 | 37.04 | 1.2527 | 0.3743 |
| 65,536 | MorphBPE / TurkishDelight | 1.5641 | 1.8192 | 40.76 | 5.2230 | 45.56 | 27.45 | 1.3071 | 0.4296 |
| 131,072 | Raw BPE | 1.3432 | 1.6372 | 49.58 | 6.0818 | 47.01 | 5.80 | 1.2623 | 0.2412 |
| 131,072 | MorphBPE / TRmorph | 1.5968 | 1.6027 | 50.73 | 5.1159 | 56.66 | 46.26 | 1.0045 | 0.3786 |
| 131,072 | MorphBPE / Zemberek | 1.5909 | 1.6026 | 50.17 | 5.1351 | 53.80 | 33.67 | 1.0859 | 0.3330 |
| 131,072 | MorphBPE / TurkishDelight | 1.4705 | 1.6481 | 48.56 | 5.5552 | 51.37 | 21.07 | 1.1345 | 0.3971 |

**Interpretation:** Lowest corpus fertility: Raw BPE at 32k, 64k and 128k. Lowest isolated fertility: Zemberek at 32k, 64k and 128k. The lead at 64k and 128k is at most 0.001 token/word at the displayed precision. Lowest morphology edit distance: TRmorph at 32k, 64k and 128k. Highest one-token coverage, exact-morpheme agreement and boundary preservation: TRmorph at 32k, 64k and 128k. Highest original pairwise consistency F1: TRmorph at 32k and 64k; TurkishDelight at 128k. TurkishDelight at 128k has both higher isolated fertility and lower one-token coverage than same-size raw BPE, despite its morphology improvements.

| Vocab | Tokenizer | Normalized morph edit | Consistency P | Consistency R | F1 resample SD |
|---:|---|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.2328 | 0.9630 | 0.1949 | 0.0079 |
| 32,768 | MorphBPE / TRmorph | 1.0362 | 0.9865 | 0.3466 | 0.0046 |
| 32,768 | MorphBPE / Zemberek | 1.0773 | 0.9831 | 0.2799 | 0.0089 |
| 32,768 | MorphBPE / TurkishDelight | 1.1335 | 0.9825 | 0.3172 | 0.0072 |
| 65,536 | Raw BPE | 1.0124 | 0.9804 | 0.1677 | 0.0085 |
| 65,536 | MorphBPE / TRmorph | 0.7982 | 0.9919 | 0.2845 | 0.0043 |
| 65,536 | MorphBPE / Zemberek | 0.8560 | 0.9895 | 0.2309 | 0.0089 |
| 65,536 | MorphBPE / TurkishDelight | 0.9094 | 0.9880 | 0.2745 | 0.0060 |
| 131,072 | Raw BPE | 0.8188 | 0.9913 | 0.1373 | 0.0079 |
| 131,072 | MorphBPE / TRmorph | 0.6388 | 0.9955 | 0.2338 | 0.0036 |
| 131,072 | MorphBPE / Zemberek | 0.6982 | 0.9933 | 0.2001 | 0.0091 |
| 131,072 | MorphBPE / TurkishDelight | 0.7437 | 0.9933 | 0.2482 | 0.0062 |

**Interpretation:** Lowest normalized morphology edit distance: TRmorph at 32k, 64k and 128k. Highest pairwise consistency precision: TRmorph at 32k, 64k and 128k. Highest pairwise consistency recall: TRmorph at 32k and 64k; TurkishDelight at 128k. Lower resample SD is not a quality win or a confidence interval; it measures variability of the sampled consistency calculation.

### Archived encoding and evaluator performance

These are recorded timings, not a controlled tokenizer speed benchmark. `Encode M tok/s` divides token count by accumulated encoding time across workers; `Evaluator wall s` includes the rest of metric computation. The two use different time denominators. Worker/thread settings are shown so they are not mistaken for like-for-like speed measurements.

| Vocab | Tokenizer | Encode M tok/s | Encoding s (summed) | Evaluator wall s | Workers × threads | Used vocab % | Roundtrip failures |
|---:|---|---:|---:|---:|---|---:|---:|
| 32,768 | Raw BPE | 1.358 | 462.96 | 96.37 | 32 × 1 | 99.82 | 0 |
| 32,768 | MorphBPE / TRmorph | 1.415 | 507.05 | 98.95 | 32 × 1 | 99.82 | 0 |
| 32,768 | MorphBPE / Zemberek | 1.420 | 500.25 | 97.79 | 32 × 1 | 99.82 | 0 |
| 32,768 | MorphBPE / TurkishDelight | 1.377 | 495.90 | 98.22 | 32 × 1 | 99.78 | 0 |
| 65,536 | Raw BPE | 1.319 | 438.93 | 95.19 | 32 × 1 | 99.91 | 0 |
| 65,536 | MorphBPE / TRmorph | 1.315 | 515.43 | 98.79 | 32 × 1 | 99.90 | 0 |
| 65,536 | MorphBPE / Zemberek | 1.354 | 497.63 | 97.64 | 32 × 1 | 99.87 | 0 |
| 65,536 | MorphBPE / TurkishDelight | 1.304 | 486.02 | 98.63 | 32 × 1 | 99.73 | 0 |
| 131,072 | Raw BPE | 1.251 | 434.92 | 95.23 | 32 × 1 | 99.95 | 0 |
| 131,072 | MorphBPE / TRmorph | 1.243 | 520.24 | 99.85 | 32 × 1 | 99.92 | 0 |
| 131,072 | MorphBPE / Zemberek | 1.265 | 509.44 | 99.11 | 32 × 1 | 99.83 | 0 |
| 131,072 | MorphBPE / TurkishDelight | 1.259 | 473.28 | 96.70 | 32 × 1 | 99.44 | 0 |

**Interpretation:** Among the local tokenizer rows, highest recorded encoding tokens/s: Zemberek at 32k, 64k and 128k. All rows tie at zero observed roundtrip failures. These timings use tokenizer-dependent token counts, so they do not establish a controlled raw-text-speed ranking. High vocabulary usage alone is not evidence of better morphology.

## Raw BPE32k baseline train/validation diagnostics

The d32 preflight records name the original `nanochat-turk-d20-bpe32k/tokenizers/bpe_32768` tokenizer path. They evaluate raw text with no morphology reference; boundary and morphology fields are not applicable. These are tokenizer-only diagnostics and do not add a new model cell or matched multi-tokenizer validation comparison. The full-train token/byte counts match the recovered full-corpus BPE32k row exactly.

| Split | Documents | UTF-8 bytes | Regex words | Tokens/word | Isolated tokens/word | One-token words % | Bytes/token | Roundtrip failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 93,211,827 | 299,173,651,770 | 36,699,889,764 | 1.5553 | 2.0472 | 31.05 | 5.2415 | 0 |
| val | 1,917,302 | 6,323,778,000 | 778,613,710 | 1.5741 | 2.0076 | 32.00 | 5.1596 | 0 |

**Interpretation:** Raw BPE uses slightly more corpus tokens per word on validation than training, despite better isolated-word one-token coverage there. Both splits have zero roundtrip failures. This is one tokenizer on different text distributions, so there is no tokenizer-family winner in this table.

Both use 500,000 isolated word occurrences. Training uses 112 workers × 1 thread; validation uses 56 × 1. Their timing is separate from the twelve-tokenizer prefix comparisons.

| Split | Encode M tok/s | Encoding s (summed) | Evaluator wall s | Used vocab % |
|---|---:|---:|---:|---:|
| train | 1.107 | 51559.81 | 1535.21 | 99.84 |
| val | 1.386 | 884.34 | 31.70 | 99.83 |

**Interpretation:** These timings describe different corpus sizes and worker counts for the same tokenizer. Neither the shorter validation runtime nor its recorded tokens/s establishes a faster tokenizer; vocabulary-use rates are scope-dependent diagnostics.

## Source index

| Scope | Tokenizer | Original JSON |
|---|---|---|
| sample_100k | bpe_32768 | [bpe_32768_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/bpe_32768_metrics.json) |
| sample_100k | morphbpe_trmorph_32768 | [morphbpe_trmorph_32768_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_trmorph_32768_metrics.json) |
| sample_100k | morphbpe_zemberek_32768 | [morphbpe_zemberek_32768_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_zemberek_32768_metrics.json) |
| sample_100k | morphbpe_tdelight_32768 | [morphbpe_tdelight_32768_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_tdelight_32768_metrics.json) |
| sample_100k | bpe_65536 | [bpe_65536_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/bpe_65536_metrics.json) |
| sample_100k | morphbpe_trmorph_65536 | [morphbpe_trmorph_65536_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_trmorph_65536_metrics.json) |
| sample_100k | morphbpe_zemberek_65536 | [morphbpe_zemberek_65536_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_zemberek_65536_metrics.json) |
| sample_100k | morphbpe_tdelight_65536 | [morphbpe_tdelight_65536_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_tdelight_65536_metrics.json) |
| sample_100k | bpe_131072 | [bpe_131072_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/bpe_131072_metrics.json) |
| sample_100k | morphbpe_trmorph_131072 | [morphbpe_trmorph_131072_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_trmorph_131072_metrics.json) |
| sample_100k | morphbpe_zemberek_131072 | [morphbpe_zemberek_131072_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_zemberek_131072_metrics.json) |
| sample_100k | morphbpe_tdelight_131072 | [morphbpe_tdelight_131072_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_100k/morphbpe_tdelight_131072_metrics.json) |
| sample_1m | bpe_32768 | [bpe_32768_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/bpe_32768_metrics.json) |
| sample_1m | morphbpe_trmorph_32768 | [morphbpe_trmorph_32768_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_trmorph_32768_metrics.json) |
| sample_1m | morphbpe_zemberek_32768 | [morphbpe_zemberek_32768_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_zemberek_32768_metrics.json) |
| sample_1m | morphbpe_tdelight_32768 | [morphbpe_tdelight_32768_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_tdelight_32768_metrics.json) |
| sample_1m | bpe_65536 | [bpe_65536_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/bpe_65536_metrics.json) |
| sample_1m | morphbpe_trmorph_65536 | [morphbpe_trmorph_65536_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_trmorph_65536_metrics.json) |
| sample_1m | morphbpe_zemberek_65536 | [morphbpe_zemberek_65536_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_zemberek_65536_metrics.json) |
| sample_1m | morphbpe_tdelight_65536 | [morphbpe_tdelight_65536_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_tdelight_65536_metrics.json) |
| sample_1m | bpe_131072 | [bpe_131072_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/bpe_131072_metrics.json) |
| sample_1m | morphbpe_trmorph_131072 | [morphbpe_trmorph_131072_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_trmorph_131072_metrics.json) |
| sample_1m | morphbpe_zemberek_131072 | [morphbpe_zemberek_131072_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_zemberek_131072_metrics.json) |
| sample_1m | morphbpe_tdelight_131072 | [morphbpe_tdelight_131072_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/sample_1m/morphbpe_tdelight_131072_metrics.json) |
| d32_preflight / train | bpe_32768 | [bpe_32768_train_raw_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/d32_preflight/bpe_32768_train_raw_metrics.json) |
| d32_preflight / val | bpe_32768 | [bpe_32768_val_metrics.json](../../artifacts/tokenizer_metrics_additional_2026-11-09/d32_preflight/bpe_32768_val_metrics.json) |

**Reading note:** These links identify the original evidence for each row; their order is not a performance ranking.

Intact-morpheme-token preservation, character-only coverage and literal boundary P/R/F1 were absent from these archives. They are now measured separately for all twelve local tokenizers in the [completed matched train/validation evaluation](tokenizer_fresh_results.md); historical missing values remain null. All original distributions, vocabulary shape, source paths, timing fields and hashes are preserved in JSON. Rebuild with `python -m scripts.build_tokenizer_study_results`.
