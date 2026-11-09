# Fresh matched tokenizer measurements

New measurements on UHEM, 9 November 2026. All twelve original tokenizer bundles receive identical raw text within each split; segmentation is reference-only. [Run/sample provenance](fresh_tokenizer_metrics/manifest.json), [full precision JSON](tokenizer_fresh_results.json), [CSV](tokenizer_fresh_results.csv), and [complete evaluator outputs](fresh_tokenizer_metrics/) are retained.

The validation file is withheld by the repository's configured file split; training and validation sample hashes differ. This is an automatic TRmorph-reference diagnostic, not independently annotated human morphology. No model was retrained for this evaluation.

## Validation

50,000 documents; 161,056,183 UTF-8 bytes; 19,832,065 regex words. Word-level metrics use 500,000 occurrences per tokenizer. Identical word/reference hashes are required for all twelve rows.

| Vocab | Tokenizer | Corpus tok/word ↓ | Isolated tok/word ↓ | One-token words % | Character-only words % | Morphemes intact % ↑ | Isolated boundary P % ↑ | Isolated boundary R % ↑ | Isolated boundary F1 % ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.5728 | 2.0076 | 32.00 | 2.43 | 31.57 | 18.87 | 36.88 | 24.96 |
| 32,768 | TRmorph | 1.7785 | 1.9414 | 35.89 | 2.35 | 44.30 | 29.70 | 54.23 | 38.38 |
| 32,768 | Zemberek | 1.7673 | 1.9400 | 35.48 | 2.36 | 40.79 | 26.63 | 48.56 | 34.40 |
| 32,768 | TurkishDelight | 1.6989 | 1.9949 | 34.11 | 2.37 | 39.81 | 26.09 | 50.34 | 34.37 |
| 65,536 | Raw BPE | 1.4483 | 1.7850 | 41.48 | 2.31 | 37.15 | 21.99 | 33.48 | 26.54 |
| 65,536 | TRmorph | 1.6775 | 1.7121 | 45.67 | 2.25 | 50.27 | 35.88 | 49.55 | 41.62 |
| 65,536 | Zemberek | 1.6733 | 1.7176 | 44.74 | 2.25 | 46.39 | 31.77 | 44.21 | 36.97 |
| 65,536 | TurkishDelight | 1.5760 | 1.7695 | 43.09 | 2.25 | 45.27 | 30.84 | 46.02 | 36.93 |
| 131,072 | Raw BPE | 1.3611 | 1.5902 | 52.07 | 2.24 | 41.39 | 25.48 | 29.16 | 27.20 |
| 131,072 | TRmorph | 1.6001 | 1.5576 | 53.45 | 2.22 | 53.55 | 41.69 | 45.09 | 43.33 |
| 131,072 | Zemberek | 1.5989 | 1.5639 | 52.42 | 2.23 | 49.83 | 36.78 | 40.23 | 38.43 |
| 131,072 | TurkishDelight | 1.4809 | 1.6022 | 51.07 | 2.22 | 48.26 | 35.52 | 41.49 | 38.27 |

| Vocab | Tokenizer | Multi-character words split only into characters % | Byte-only words % | Roundtrip failures | UNK count | Scored / excluded morphology words |
|---:|---|---:|---:|---:|---|---|
| 32,768 | Raw BPE | 0.29 | 2.27 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 32,768 | TRmorph | 0.22 | 2.20 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 32,768 | Zemberek | 0.22 | 2.21 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 32,768 | TurkishDelight | 0.23 | 2.22 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 65,536 | Raw BPE | 0.18 | 2.16 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 65,536 | TRmorph | 0.11 | 2.11 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 65,536 | Zemberek | 0.11 | 2.11 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 65,536 | TurkishDelight | 0.11 | 2.11 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 131,072 | Raw BPE | 0.10 | 2.10 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 131,072 | TRmorph | 0.09 | 2.09 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 131,072 | Zemberek | 0.09 | 2.09 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 131,072 | TurkishDelight | 0.08 | 2.09 | 0 | N/A (byte-complete) | 500,000 / 0 |

## Training sample

50,000 documents; 186,328,421 UTF-8 bytes; 23,042,031 regex words. Word-level metrics use 500,000 occurrences per tokenizer. Identical word/reference hashes are required for all twelve rows.

| Vocab | Tokenizer | Corpus tok/word ↓ | Isolated tok/word ↓ | One-token words % | Character-only words % | Morphemes intact % ↑ | Isolated boundary P % ↑ | Isolated boundary R % ↑ | Isolated boundary F1 % ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.6157 | 2.0472 | 31.05 | 2.31 | 30.76 | 19.08 | 37.17 | 25.22 |
| 32,768 | TRmorph | 1.8166 | 1.9890 | 34.09 | 2.23 | 42.90 | 29.36 | 54.02 | 38.05 |
| 32,768 | Zemberek | 1.7986 | 1.9810 | 34.06 | 2.23 | 39.60 | 26.54 | 48.44 | 34.29 |
| 32,768 | TurkishDelight | 1.7366 | 2.0357 | 32.73 | 2.25 | 38.46 | 25.94 | 49.98 | 34.15 |
| 65,536 | Raw BPE | 1.4856 | 1.8322 | 39.44 | 2.21 | 35.59 | 21.85 | 33.83 | 26.55 |
| 65,536 | TRmorph | 1.7121 | 1.7625 | 43.17 | 2.14 | 48.16 | 34.84 | 49.42 | 40.87 |
| 65,536 | Zemberek | 1.7030 | 1.7616 | 42.78 | 2.14 | 44.50 | 31.11 | 44.08 | 36.48 |
| 65,536 | TurkishDelight | 1.6091 | 1.8192 | 40.76 | 2.15 | 43.24 | 30.12 | 45.90 | 36.37 |
| 131,072 | Raw BPE | 1.3915 | 1.6372 | 49.58 | 2.04 | 39.47 | 24.92 | 29.55 | 27.04 |
| 131,072 | TRmorph | 1.6304 | 1.6027 | 50.73 | 2.02 | 51.33 | 40.14 | 45.00 | 42.43 |
| 131,072 | Zemberek | 1.6248 | 1.6026 | 50.17 | 2.02 | 47.77 | 35.67 | 39.99 | 37.71 |
| 131,072 | TurkishDelight | 1.5109 | 1.6481 | 48.56 | 2.02 | 46.27 | 34.58 | 41.69 | 37.80 |

| Vocab | Tokenizer | Multi-character words split only into characters % | Byte-only words % | Roundtrip failures | UNK count | Scored / excluded morphology words |
|---:|---|---:|---:|---:|---|---|
| 32,768 | Raw BPE | 0.39 | 2.21 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 32,768 | TRmorph | 0.31 | 2.14 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 32,768 | Zemberek | 0.31 | 2.14 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 32,768 | TurkishDelight | 0.33 | 2.15 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 65,536 | Raw BPE | 0.28 | 2.10 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 65,536 | TRmorph | 0.21 | 2.03 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 65,536 | Zemberek | 0.21 | 2.04 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 65,536 | TurkishDelight | 0.23 | 2.05 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 131,072 | Raw BPE | 0.11 | 1.93 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 131,072 | TRmorph | 0.08 | 1.91 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 131,072 | Zemberek | 0.09 | 1.91 | 0 | N/A (byte-complete) | 500,000 / 0 |
| 131,072 | TurkishDelight | 0.09 | 1.92 | 0 | N/A (byte-complete) | 500,000 / 0 |

## Definitions

- **Vocabulary size** includes the nine special tokens in each original bundle.
- **Corpus fertility** is total document tokens / regex word count; punctuation and spacing tokens are included. **Isolated fertility** is sampled subword count / sampled word occurrences. A lower value means fewer units, not necessarily better morphology.
- **Morpheme preservation** counts reference morpheme spans that coincide exactly with one token span, divided by scored reference morphemes. It is distinct from the earlier whole-word exact-morpheme-sequence rate.
- **Boundary consistency** reports micro precision, recall and F1 from isolated-word internal token boundaries versus internal reference morpheme boundaries. Extra splits lower precision; merged morphemes lower recall. Byte boundaries inside a Unicode character are extra token boundaries. These isolated-word rates differ from contextual document-boundary preservation. The earlier pairwise morphology-consistency metric is retained separately in the raw outputs.
- **One-token coverage** counts losslessly represented, non-UNK words encoded as one token in isolation. **Character-only coverage** counts words represented entirely by one-Unicode-character tokens, including one-character words. The additional multi-character-only rate removes the trivial one-character case.
- **Byte-only coverage** instead requires each token to represent one UTF-8 byte. Unicode character and byte rates are not interchangeable.
- **UNK** is architecture-specific: all twelve local tokenizers contain the full 256-byte alphabet and no UNK mechanism, so counts are N/A. For an UNK-enabled tokenizer, the evaluator measures counts and incidence instead of treating a one-UNK word as whole-word coverage.
- Unalignable words are excluded from morphology denominators and reported explicitly. Undefined denominators produce null/N/A. Rates are occurrence-weighted, not unique-type weighted.

Source sample preparation records ordered raw/annotated document hashes. The same morphology analyzer supplies reference labels and TRmorph tokenizer training supervision; interpretation must retain that bias limitation. Earlier train-corpus diagnostics and the full-corpus archive remain separate in [the index](README.md).

Rebuild: `python -m scripts.build_fresh_tokenizer_results`.
