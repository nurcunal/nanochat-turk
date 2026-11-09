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

All twelve rows record zero roundtrip failures. Per-morpheme intact-token preservation, character-only coverage and explicit boundary P/R/F1 were not present in these June outputs; see the fresh matched evaluation for those measurements. The full-corpus word caps resolve the earlier 32k 200k/500k cap mismatch, while retaining the automatic TRmorph-reference limitation. Original throughput is not a controlled tokenizer speed comparison.

The recovered generated comparison uses a diagnostic sorting order, not an official paper ranking. Compare each treatment with raw BPE of the same vocabulary size; avoid a single synthetic overall winner.
