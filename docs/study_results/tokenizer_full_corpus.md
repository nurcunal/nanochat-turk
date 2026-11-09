# Recovered full-corpus tokenizer results

UHEM job **496882** completed on 23 June 2026 (exit 0:0; 29h 20m 05s). Results were recovered on 9 November. [Original files and scheduler evidence](../../artifacts/tokenizer_metrics_full_2026-06-23/) are preserved unchanged; [JSON](tokenizer_full_corpus.json) and [CSV](tokenizer_full_corpus.csv) contain full precision and source hashes.

All twelve rows cover the same **93,211,827 training documents**, **36,699,889,764 regex words**, and **299,173,651,770 UTF-8 bytes**. Full-corpus totals apply to compression, boundary crossing and roundtrip. Word-level metrics use **500,000 occurrences** for all rows, and consistency uses **50,000 unique words**. This is not a held-out test.

| Vocab | Tokenizer | Corpus tokens/word ↓ | Isolated tokens/word ↓ | One-token words % | Bytes/token ↑ | Exact morph sequence % ↑ | Boundaries preserved % ↑ | Morph edit ↓ | Morph consistency F1 ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.5553 | 2.0472 | 31.05 | 5.2415 | 32.46 | 14.66 | 1.7096 | 0.3241 |
| 32,768 | MorphBPE / TRmorph | 1.7698 | 1.9890 | 34.09 | 4.6061 | 41.54 | 54.18 | 1.4319 | 0.5129 |
| 32,768 | MorphBPE / Zemberek | 1.7551 | 1.9810 | 34.06 | 4.6446 | 39.50 | 40.44 | 1.4986 | 0.4357 |
| 32,768 | MorphBPE / TurkishDelight | 1.6865 | 2.0357 | 32.73 | 4.8335 | 37.54 | 34.43 | 1.5544 | 0.4795 |
| 65,536 | Raw BPE | 1.4328 | 1.8322 | 39.44 | 5.6894 | 40.04 | 9.34 | 1.4627 | 0.2863 |
| 65,536 | MorphBPE / TRmorph | 1.6712 | 1.7625 | 43.17 | 4.8780 | 50.56 | 50.59 | 1.1697 | 0.4421 |
| 65,536 | MorphBPE / Zemberek | 1.6640 | 1.7616 | 42.78 | 4.8989 | 47.82 | 36.97 | 1.2527 | 0.3743 |
| 65,536 | MorphBPE / TurkishDelight | 1.5651 | 1.8192 | 40.76 | 5.2084 | 45.56 | 27.03 | 1.3071 | 0.4296 |
| 131,072 | Raw BPE | 1.3473 | 1.6372 | 49.58 | 6.0506 | 47.01 | 5.51 | 1.2623 | 0.2412 |
| 131,072 | MorphBPE / TRmorph | 1.5952 | 1.6027 | 50.73 | 5.1101 | 56.66 | 46.32 | 1.0045 | 0.3786 |
| 131,072 | MorphBPE / Zemberek | 1.5916 | 1.6026 | 50.17 | 5.1219 | 53.80 | 33.63 | 1.0859 | 0.3330 |
| 131,072 | MorphBPE / TurkishDelight | 1.4715 | 1.6481 | 48.56 | 5.5399 | 51.37 | 20.64 | 1.1345 | 0.3971 |

All twelve rows record zero roundtrip failures. Per-morpheme intact-token preservation, character-only coverage and explicit boundary P/R/F1 were not present in these June outputs and are not inferred from the archived values. The full-corpus word caps resolve the earlier 32k 200k/500k cap mismatch, while retaining the automatic TRmorph-reference limitation. The repeated consistency values are expected because this metric remains capped at the same 50,000 word types; they are not independent replications.

## Additional morphology diagnostics

Morph consistency is shared-token/shared-morpheme pair precision/recall/F1. `F1 resample SD` is variability over the ten sampled pair sets, not a confidence interval or across-training-seed uncertainty. Boundary preservation is 1 minus crossed-boundary rate and does not penalize extra splits.

| Vocab | Tokenizer | Normalized morph edit ↓ | Consistency P | Consistency R | F1 resample SD | Crossed boundaries % |
|---:|---|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.2328 | 0.9630 | 0.1949 | 0.0079 | 85.34 |
| 32,768 | MorphBPE / TRmorph | 1.0362 | 0.9865 | 0.3466 | 0.0046 | 45.82 |
| 32,768 | MorphBPE / Zemberek | 1.0773 | 0.9831 | 0.2799 | 0.0089 | 59.56 |
| 32,768 | MorphBPE / TurkishDelight | 1.1335 | 0.9825 | 0.3172 | 0.0072 | 65.57 |
| 65,536 | Raw BPE | 1.0124 | 0.9804 | 0.1677 | 0.0085 | 90.66 |
| 65,536 | MorphBPE / TRmorph | 0.7982 | 0.9919 | 0.2845 | 0.0043 | 49.41 |
| 65,536 | MorphBPE / Zemberek | 0.8560 | 0.9895 | 0.2309 | 0.0089 | 63.03 |
| 65,536 | MorphBPE / TurkishDelight | 0.9094 | 0.9880 | 0.2745 | 0.0060 | 72.97 |
| 131,072 | Raw BPE | 0.8188 | 0.9913 | 0.1373 | 0.0079 | 94.49 |
| 131,072 | MorphBPE / TRmorph | 0.6388 | 0.9955 | 0.2338 | 0.0036 | 53.68 |
| 131,072 | MorphBPE / Zemberek | 0.6982 | 0.9933 | 0.2001 | 0.0091 | 66.37 |
| 131,072 | MorphBPE / TurkishDelight | 0.7437 | 0.9933 | 0.2482 | 0.0062 | 79.36 |

## Archived encoding and evaluator performance

These are recorded timings, not a controlled tokenizer speed benchmark. `Encode M tok/s` divides token count by accumulated encoding time across workers; `Evaluator wall s` includes the rest of metric computation. The two use different time denominators. Worker/thread settings are shown so they are not mistaken for like-for-like speed measurements.

| Vocab | Tokenizer | Encode M tok/s | Encoding s (summed) | Evaluator wall s | Workers × threads | Used vocab % | Roundtrip failures |
|---:|---|---:|---:|---:|---|---:|---:|
| 32,768 | Raw BPE | 1.217 | 46885.67 | 9061.83 | 32 × 1 | 99.84 | 0 |
| 32,768 | MorphBPE / TRmorph | 1.230 | 52806.76 | 8930.92 | 32 × 1 | 99.84 | 0 |
| 32,768 | MorphBPE / Zemberek | 1.253 | 51422.19 | 8796.46 | 32 × 1 | 99.84 | 0 |
| 32,768 | MorphBPE / TurkishDelight | 1.244 | 49774.39 | 8662.04 | 32 × 1 | 99.83 | 0 |
| 65,536 | Raw BPE | 1.159 | 45360.90 | 8532.99 | 32 × 1 | 99.92 | 0 |
| 65,536 | MorphBPE / TRmorph | 1.213 | 50546.32 | 8711.07 | 32 × 1 | 99.92 | 0 |
| 65,536 | MorphBPE / Zemberek | 1.198 | 50992.53 | 8767.68 | 32 × 1 | 99.92 | 0 |
| 65,536 | MorphBPE / TurkishDelight | 1.169 | 49152.42 | 8672.63 | 32 × 1 | 99.91 | 0 |
| 131,072 | Raw BPE | 1.063 | 46517.39 | 8564.54 | 32 × 1 | 99.96 | 0 |
| 131,072 | MorphBPE / TRmorph | 1.152 | 50814.85 | 8751.28 | 32 × 1 | 99.96 | 0 |
| 131,072 | MorphBPE / Zemberek | 1.105 | 52872.38 | 8795.26 | 32 × 1 | 99.96 | 0 |
| 131,072 | MorphBPE / TurkishDelight | 1.049 | 51477.29 | 8953.76 | 32 × 1 | 99.95 | 0 |

## Row-level source files

Every file below is copied in full into the report JSON with its SHA-256 hash; the CSV carries the same path and hash. Original result files are preserved unchanged.

| Vocabulary | Tokenizer | Original JSON |
|---:|---|---|
| 32,768 | Raw BPE | [bpe_32768_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/bpe_32768_metrics.json) |
| 32,768 | MorphBPE / TRmorph | [morphbpe_trmorph_32768_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_trmorph_32768_metrics.json) |
| 32,768 | MorphBPE / Zemberek | [morphbpe_zemberek_32768_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_zemberek_32768_metrics.json) |
| 32,768 | MorphBPE / TurkishDelight | [morphbpe_tdelight_32768_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_tdelight_32768_metrics.json) |
| 65,536 | Raw BPE | [bpe_65536_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/bpe_65536_metrics.json) |
| 65,536 | MorphBPE / TRmorph | [morphbpe_trmorph_65536_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_trmorph_65536_metrics.json) |
| 65,536 | MorphBPE / Zemberek | [morphbpe_zemberek_65536_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_zemberek_65536_metrics.json) |
| 65,536 | MorphBPE / TurkishDelight | [morphbpe_tdelight_65536_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_tdelight_65536_metrics.json) |
| 131,072 | Raw BPE | [bpe_131072_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/bpe_131072_metrics.json) |
| 131,072 | MorphBPE / TRmorph | [morphbpe_trmorph_131072_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_trmorph_131072_metrics.json) |
| 131,072 | MorphBPE / Zemberek | [morphbpe_zemberek_131072_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_zemberek_131072_metrics.json) |
| 131,072 | MorphBPE / TurkishDelight | [morphbpe_tdelight_131072_metrics.json](../../artifacts/tokenizer_metrics_full_2026-06-23/morphbpe_tdelight_131072_metrics.json) |

## Interpretation and reproduction

The recovered generated comparison uses a diagnostic sorting order, not an official paper ranking. Compare each treatment with raw BPE of the same vocabulary size; avoid a single synthetic overall winner. All nine MorphBPE variants preserve more automatic reference boundaries and improve recorded morphology edit/consistency diagnostics while using more corpus tokens per word than same-size raw BPE. These training-corpus diagnostics do not establish a downstream model advantage.

The report JSON retains original distributions, unique-word metrics, vocabulary shape, elapsed times and actual consistency clusters/pairs. Rebuild with `python -m scripts.build_tokenizer_study_results`; the builder verifies all recovered source hashes, sample counts and derived ratios before producing tables.
