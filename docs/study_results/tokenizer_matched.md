# Matched tokenizer evaluation: archival mirror

**Direction legend:** ↑ means higher is better for the stated objective; ↓ means lower is better; ↔ means descriptive or a tradeoff, with no universally better direction. See the [metric guide](metric_guide.md) for every definition, denominator and limitation.

**Objectives:** lower tokens/word and higher bytes/token mean more compact lossless encoding. Higher one-token coverage helps isolated-word compression only. Higher morphology preservation, boundary agreement and consistency, or lower morphology edit distance, mean closer agreement with the automatic reference; they do not establish better LLM performance. Lower multi-character character-only coverage means less character-level fragmentation.

**Diagnostics:** vocabulary size, vocabulary use, sample counts, inclusive character-only/byte-only coverage and resample SD have no universal optimum. Zero roundtrip failures is ideal for exact text recovery; fewer UNKs and alignment exclusions are better when applicable. Speed ↑ and time ↓ apply only to matched workloads and hardware; recorded timings here do not establish a controlled speed ranking. Normalizing references' apparent compression and boundary figures remain descriptive.

This is an archival view of the same 24 measurements from job 558060 in the [canonical fresh tokenizer report](tokenizer_fresh_results.md). It preserves the recovered source snapshot and receipt checks; it is not an additional experiment or independent replication.

Fresh UHeM job **558060** completed successfully on 9 November 2026 (exit 0:0; scheduler elapsed 00:14:37). This separate evaluation adds individual-morpheme preservation, literal boundary precision/recall/F1, character-only coverage and a reserved validation-shard sample for all twelve tokenizers. It does not create new language-model results. [Original measurements and receipts](../../artifacts/tokenizer_metrics_matched_2026-11-09/), [full JSON](tokenizer_matched.json), [flat CSV](tokenizer_matched.csv).

Each split uses 50,000 documents, the same first 500,000 isolated word occurrences and morphology-reference occurrences across all tokenizers, and 50,000 unique words for the original consistency metric. The final receipt verifies matching word/reference hashes within each split. Tokenizer bytes match the twelve checked-in bundles. The train prefix and validation prefix come from distinct source shards; the latter is the final reserved source shard, not an independently human-annotated morphology dataset. Here reserved validation describes this tokenizer evaluation's sample split: the later d32 all-shard training run includes this source text, so it is not held out from every model in the repository.

## Metric definitions

- **Vocabulary (↔):** number of token IDs, including nine special tokens locally. Larger vocabularies can shorten sequences but require larger model embedding/output tables.
- **Corpus tokens/word (↓ compression):** encoded document tokens divided by regex words; punctuation/spacing tokens contribute to the numerator. **Isolated tokens/word (↓ compression)** encodes each sampled word separately. **Bytes/token (↑ compression)** is original UTF-8 bytes divided by document tokens.
- **One-token coverage (↑ isolated-word compression only)** is the fraction of sampled word occurrences represented losslessly by one non-UNK token. **Character-only coverage (↔ diagnostic)** requires every token to decode to one Unicode character and includes one-character words. **Multi-character character-only coverage (↓ fragmentation)** restricts its denominator to words longer than one character. **Byte-only coverage (↔ diagnostic)** requires one UTF-8 byte per token; byte fragments and Unicode characters differ.
- **Intact morphemes (↑ reference agreement):** fraction of reference morpheme spans that coincide exactly with one token span. Splitting a morpheme or merging multiple morphemes loses that exact match. **Exact morph sequence (↑ reference agreement)** requires every piece in a word to match the reference sequence.
- **Boundary P/R/F1 (↑ reference agreement):** micro agreement of internal token boundaries and reference boundaries on isolated words; word edges are excluded. Precision is the fraction of predicted boundaries that match; recall is the fraction of reference boundaries recovered; F1 is their harmonic mean. **Document boundaries preserved (↑ reference agreement)** checks contextual reference boundaries without penalizing extra splits, so it is a different metric.
- **Morph edit / normalized edit (↓ reference disagreement):** token-piece versus reference-piece sequence edit distance, with the normalized version divided by reference morpheme count. **Consistency P/R/F1 (↑ reference agreement)** measures shared-token/shared-morpheme word-pair agreement, separately from literal boundary F1.
- **Alignment exclusions (↓ evaluation failures):** unalignable word encodings are counted; morphology denominators use scored words. Fewer exclusions improve evaluation coverage, not necessarily linguistic quality. **Roundtrip failures (↓; zero ideal)** count documents whose decode/encode output differs from the original.
- **UNK (↓ when applicable):** local byte-complete tokenizers have no UNK mechanism, so UNK is N/A, not an observed zero.

Automatic TRmorph annotations may favor the TRmorph-trained treatment. These are single-run diagnostic measurements; consistency resample SD is not across-training-seed uncertainty or a confidence interval.

## Training prefix

50,000 documents; 186,328,421 UTF-8 bytes; 23,042,031 regex words. Source: `000_00000.segmented.parquet`. Percentages below use the denominators defined above.

| Vocab ↔ | Tokenizer | Corpus tokens/word ↓ | Isolated tokens/word ↓ | One-token words % ↑ compression | Character-only words % ↔ | Intact morphemes % ↑ | Boundary P % ↑ | Boundary R % ↑ | Boundary F1 % ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.6157 | 2.0472 | 31.05 | 2.31 | 30.76 | 19.08 | 37.17 | 25.22 |
| 32,768 | MorphBPE / TRmorph | 1.8166 | 1.9890 | 34.09 | 2.23 | 42.90 | 29.36 | 54.02 | 38.05 |
| 32,768 | MorphBPE / Zemberek | 1.7986 | 1.9810 | 34.06 | 2.23 | 39.60 | 26.54 | 48.44 | 34.29 |
| 32,768 | MorphBPE / TurkishDelight | 1.7366 | 2.0357 | 32.73 | 2.25 | 38.46 | 25.94 | 49.98 | 34.15 |
| 65,536 | Raw BPE | 1.4856 | 1.8322 | 39.44 | 2.21 | 35.59 | 21.85 | 33.83 | 26.55 |
| 65,536 | MorphBPE / TRmorph | 1.7121 | 1.7625 | 43.17 | 2.14 | 48.16 | 34.84 | 49.42 | 40.87 |
| 65,536 | MorphBPE / Zemberek | 1.7030 | 1.7616 | 42.78 | 2.14 | 44.50 | 31.11 | 44.08 | 36.48 |
| 65,536 | MorphBPE / TurkishDelight | 1.6091 | 1.8192 | 40.76 | 2.15 | 43.24 | 30.12 | 45.90 | 36.37 |
| 131,072 | Raw BPE | 1.3915 | 1.6372 | 49.58 | 2.04 | 39.47 | 24.92 | 29.55 | 27.04 |
| 131,072 | MorphBPE / TRmorph | 1.6304 | 1.6027 | 50.73 | 2.02 | 51.33 | 40.14 | 45.00 | 42.43 |
| 131,072 | MorphBPE / Zemberek | 1.6248 | 1.6026 | 50.17 | 2.02 | 47.77 | 35.67 | 39.99 | 37.71 |
| 131,072 | MorphBPE / TurkishDelight | 1.5109 | 1.6481 | 48.56 | 2.02 | 46.27 | 34.58 | 41.69 | 37.80 |

**Interpretation:** Lowest corpus fertility: Raw BPE at 32k, 64k and 128k. Lowest isolated fertility: Zemberek at 32k, 64k and 128k. The lead at 64k and 128k is at most 0.001 token/word at the displayed precision. Highest one-token coverage, intact-morpheme preservation and boundary P/R/F1: TRmorph at 32k, 64k and 128k. TurkishDelight at 128k has both higher isolated fertility and lower one-token coverage than same-size raw BPE, despite its morphology improvements. These morphology rankings use the automatic TRmorph reference.

### Compression and original morphology diagnostics

| Vocab ↔ | Tokenizer | Bytes/token ↑ | Morph edit ↓ | Normalized edit ↓ | Exact morph sequence % ↑ | Consistency P ↑ | Consistency R ↑ | Consistency F1 ↑ | Document boundaries preserved % ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 5.0051 | 1.7096 | 1.2328 | 32.46 | 0.9630 | 0.1949 | 0.3241 | 16.05 |
| 32,768 | MorphBPE / TRmorph | 4.4514 | 1.4319 | 1.0362 | 41.54 | 0.9865 | 0.3466 | 0.5129 | 54.31 |
| 32,768 | MorphBPE / Zemberek | 4.4959 | 1.4986 | 1.0773 | 39.50 | 0.9831 | 0.2799 | 0.4357 | 40.94 |
| 32,768 | MorphBPE / TurkishDelight | 4.6564 | 1.5544 | 1.1335 | 37.54 | 0.9825 | 0.3172 | 0.4795 | 35.57 |
| 65,536 | Raw BPE | 5.4434 | 1.4627 | 1.0124 | 40.04 | 0.9804 | 0.1677 | 0.2863 | 10.61 |
| 65,536 | MorphBPE / TRmorph | 4.7230 | 1.1697 | 0.7982 | 50.56 | 0.9919 | 0.2845 | 0.4421 | 50.69 |
| 65,536 | MorphBPE / Zemberek | 4.7485 | 1.2527 | 0.8560 | 47.82 | 0.9895 | 0.2309 | 0.3743 | 37.55 |
| 65,536 | MorphBPE / TurkishDelight | 5.0254 | 1.3071 | 0.9094 | 45.56 | 0.9880 | 0.2745 | 0.4296 | 28.21 |
| 131,072 | Raw BPE | 5.8114 | 1.2623 | 0.8188 | 47.01 | 0.9913 | 0.1373 | 0.2412 | 6.54 |
| 131,072 | MorphBPE / TRmorph | 4.9597 | 1.0045 | 0.6388 | 56.66 | 0.9955 | 0.2338 | 0.3786 | 46.49 |
| 131,072 | MorphBPE / Zemberek | 4.9770 | 1.0859 | 0.6982 | 53.80 | 0.9933 | 0.2001 | 0.3330 | 34.13 |
| 131,072 | MorphBPE / TurkishDelight | 5.3521 | 1.1345 | 0.7437 | 51.37 | 0.9933 | 0.2482 | 0.3971 | 21.89 |

**Interpretation:** Best corpus compression (most bytes/token): Raw BPE at 32k, 64k and 128k. Lowest raw and normalized morphology edit distance: TRmorph at 32k, 64k and 128k. Highest pairwise consistency precision: TRmorph at 32k, 64k and 128k. Highest pairwise consistency recall: TRmorph at 32k and 64k; TurkishDelight at 128k. Highest original pairwise consistency F1: TRmorph at 32k and 64k; TurkishDelight at 128k. Highest exact-morpheme agreement: TRmorph at 32k, 64k and 128k. Highest document-boundary preservation: TRmorph at 32k, 64k and 128k. This pairwise consistency metric is distinct from literal boundary F1.

### Coverage exclusions and recorded runtime

All use one metric worker and eight tokenizer threads per process, with four tokenizer processes scheduled concurrently. Encoding throughput is tokens divided by accumulated encoding time (↑ only for matched conditions); producing more tokens can raise this rate without processing text faster. Document-pass wall time (↓ time; raw field `metric_wall_seconds`) includes document I/O, roundtrip and boundary accounting but ends before isolated-word, coverage and morphology computation. Parallel preliminary word collection also precedes this timer. These are recorded runtime diagnostics, not a controlled serial speed benchmark. Multi-character character-only coverage excludes one-character words from its denominator.

| Vocab ↔ | Tokenizer | Byte-only words % ↔ | Multi-char character-only % ↓ fragmentation | Coverage unaligned ↓ | Morphology unaligned ↓ | Roundtrip failures ↓ (0 ideal) | Encode M tok/s ↑ speed | Document-pass wall s ↓ time |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 2.21 | 0.39 | 0 | 0 | 0 | 4.865 | 102.03 |
| 32,768 | MorphBPE / TRmorph | 2.14 | 0.31 | 0 | 0 | 0 | 5.224 | 102.26 |
| 32,768 | MorphBPE / Zemberek | 2.14 | 0.31 | 0 | 0 | 0 | 5.135 | 103.82 |
| 32,768 | MorphBPE / TurkishDelight | 2.15 | 0.33 | 0 | 0 | 0 | 5.084 | 101.75 |
| 65,536 | Raw BPE | 2.10 | 0.28 | 0 | 0 | 0 | 4.566 | 101.43 |
| 65,536 | MorphBPE / TRmorph | 2.03 | 0.21 | 0 | 0 | 0 | 4.981 | 101.61 |
| 65,536 | MorphBPE / Zemberek | 2.04 | 0.21 | 0 | 0 | 0 | 4.965 | 102.52 |
| 65,536 | MorphBPE / TurkishDelight | 2.05 | 0.23 | 0 | 0 | 0 | 4.764 | 102.67 |
| 131,072 | Raw BPE | 1.93 | 0.11 | 0 | 0 | 0 | 4.300 | 104.92 |
| 131,072 | MorphBPE / TRmorph | 1.91 | 0.08 | 0 | 0 | 0 | 4.533 | 106.14 |
| 131,072 | MorphBPE / Zemberek | 1.91 | 0.09 | 0 | 0 | 0 | 4.569 | 106.21 |
| 131,072 | MorphBPE / TurkishDelight | 1.92 | 0.09 | 0 | 0 | 0 | 4.623 | 98.94 |

**Interpretation:** All rows tie at zero roundtrip failures and zero alignment exclusions. Lowest multi-character character-only rate: TRmorph and Zemberek (tied at displayed precision) at 32k and 64k; TRmorph at 128k. Highest recorded encoding tokens/s: TRmorph at 32k and 64k; TurkishDelight at 128k. These timings include concurrent-process contention, and more tokens/s can reflect producing more tokens rather than processing raw text faster.

## Reserved validation-shard prefix

50,000 documents; 161,056,183 UTF-8 bytes; 19,832,065 regex words. Source: `004_00005.segmented.parquet`. Percentages below use the denominators defined above.

| Vocab ↔ | Tokenizer | Corpus tokens/word ↓ | Isolated tokens/word ↓ | One-token words % ↑ compression | Character-only words % ↔ | Intact morphemes % ↑ | Boundary P % ↑ | Boundary R % ↑ | Boundary F1 % ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 1.5728 | 2.0076 | 32.00 | 2.43 | 31.57 | 18.87 | 36.88 | 24.96 |
| 32,768 | MorphBPE / TRmorph | 1.7785 | 1.9414 | 35.89 | 2.35 | 44.30 | 29.70 | 54.23 | 38.38 |
| 32,768 | MorphBPE / Zemberek | 1.7673 | 1.9400 | 35.48 | 2.36 | 40.79 | 26.63 | 48.56 | 34.40 |
| 32,768 | MorphBPE / TurkishDelight | 1.6989 | 1.9949 | 34.11 | 2.37 | 39.81 | 26.09 | 50.34 | 34.37 |
| 65,536 | Raw BPE | 1.4483 | 1.7850 | 41.48 | 2.31 | 37.15 | 21.99 | 33.48 | 26.54 |
| 65,536 | MorphBPE / TRmorph | 1.6775 | 1.7121 | 45.67 | 2.25 | 50.27 | 35.88 | 49.55 | 41.62 |
| 65,536 | MorphBPE / Zemberek | 1.6733 | 1.7176 | 44.74 | 2.25 | 46.39 | 31.77 | 44.21 | 36.97 |
| 65,536 | MorphBPE / TurkishDelight | 1.5760 | 1.7695 | 43.09 | 2.25 | 45.27 | 30.84 | 46.02 | 36.93 |
| 131,072 | Raw BPE | 1.3611 | 1.5902 | 52.07 | 2.24 | 41.39 | 25.48 | 29.16 | 27.20 |
| 131,072 | MorphBPE / TRmorph | 1.6001 | 1.5576 | 53.45 | 2.22 | 53.55 | 41.69 | 45.09 | 43.33 |
| 131,072 | MorphBPE / Zemberek | 1.5989 | 1.5639 | 52.42 | 2.23 | 49.83 | 36.78 | 40.23 | 38.43 |
| 131,072 | MorphBPE / TurkishDelight | 1.4809 | 1.6022 | 51.07 | 2.22 | 48.26 | 35.52 | 41.49 | 38.27 |

**Interpretation:** Lowest corpus fertility: Raw BPE at 32k, 64k and 128k. Lowest isolated fertility: Zemberek at 32k; TRmorph at 64k and 128k. Highest one-token coverage, intact-morpheme preservation and boundary P/R/F1: TRmorph at 32k, 64k and 128k. TurkishDelight at 128k has both higher isolated fertility and lower one-token coverage than same-size raw BPE, despite its morphology improvements. These morphology rankings use the automatic TRmorph reference.

### Compression and original morphology diagnostics

| Vocab ↔ | Tokenizer | Bytes/token ↑ | Morph edit ↓ | Normalized edit ↓ | Exact morph sequence % ↑ | Consistency P ↑ | Consistency R ↑ | Consistency F1 ↑ | Document boundaries preserved % ↑ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 5.1636 | 1.6572 | 1.2010 | 33.54 | 0.9582 | 0.2062 | 0.3393 | 14.52 |
| 32,768 | MorphBPE / TRmorph | 4.5663 | 1.3672 | 0.9903 | 43.42 | 0.9822 | 0.3402 | 0.5054 | 54.25 |
| 32,768 | MorphBPE / Zemberek | 4.5953 | 1.4415 | 1.0396 | 41.07 | 0.9803 | 0.3002 | 0.4596 | 40.48 |
| 32,768 | MorphBPE / TurkishDelight | 4.7801 | 1.4954 | 1.0945 | 39.16 | 0.9788 | 0.3175 | 0.4794 | 34.21 |
| 65,536 | Raw BPE | 5.6074 | 1.3907 | 0.9602 | 42.25 | 0.9759 | 0.1849 | 0.3108 | 9.17 |
| 65,536 | MorphBPE / TRmorph | 4.8410 | 1.0912 | 0.7398 | 53.38 | 0.9888 | 0.2920 | 0.4508 | 50.67 |
| 65,536 | MorphBPE / Zemberek | 4.8534 | 1.1803 | 0.8048 | 50.27 | 0.9883 | 0.2679 | 0.4215 | 37.00 |
| 65,536 | MorphBPE / TurkishDelight | 5.1528 | 1.2288 | 0.8525 | 48.24 | 0.9867 | 0.2774 | 0.4330 | 26.84 |
| 131,072 | Raw BPE | 5.9666 | 1.1853 | 0.7618 | 49.65 | 0.9868 | 0.1668 | 0.2853 | 5.37 |
| 131,072 | MorphBPE / TRmorph | 5.0752 | 0.9283 | 0.5818 | 59.66 | 0.9930 | 0.2569 | 0.4082 | 46.35 |
| 131,072 | MorphBPE / Zemberek | 5.0790 | 1.0144 | 0.6469 | 56.51 | 0.9923 | 0.2361 | 0.3814 | 33.67 |
| 131,072 | MorphBPE / TurkishDelight | 5.4839 | 1.0610 | 0.6900 | 54.15 | 0.9911 | 0.2479 | 0.3965 | 20.43 |

**Interpretation:** Best corpus compression (most bytes/token): Raw BPE at 32k, 64k and 128k. Lowest raw and normalized morphology edit distance: TRmorph at 32k, 64k and 128k. Highest pairwise consistency precision: TRmorph at 32k, 64k and 128k. Highest pairwise consistency recall: TRmorph at 32k, 64k and 128k. Highest original pairwise consistency F1: TRmorph at 32k, 64k and 128k. Highest exact-morpheme agreement: TRmorph at 32k, 64k and 128k. Highest document-boundary preservation: TRmorph at 32k, 64k and 128k. This pairwise consistency metric is distinct from literal boundary F1.

### Coverage exclusions and recorded runtime

All use one metric worker and eight tokenizer threads per process, with four tokenizer processes scheduled concurrently. Encoding throughput is tokens divided by accumulated encoding time (↑ only for matched conditions); producing more tokens can raise this rate without processing text faster. Document-pass wall time (↓ time; raw field `metric_wall_seconds`) includes document I/O, roundtrip and boundary accounting but ends before isolated-word, coverage and morphology computation. Parallel preliminary word collection also precedes this timer. These are recorded runtime diagnostics, not a controlled serial speed benchmark. Multi-character character-only coverage excludes one-character words from its denominator.

| Vocab ↔ | Tokenizer | Byte-only words % ↔ | Multi-char character-only % ↓ fragmentation | Coverage unaligned ↓ | Morphology unaligned ↓ | Roundtrip failures ↓ (0 ideal) | Encode M tok/s ↑ speed | Document-pass wall s ↓ time |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | Raw BPE | 2.27 | 0.29 | 0 | 0 | 0 | 5.804 | 85.43 |
| 32,768 | MorphBPE / TRmorph | 2.20 | 0.22 | 0 | 0 | 0 | 6.118 | 87.61 |
| 32,768 | MorphBPE / Zemberek | 2.21 | 0.22 | 0 | 0 | 0 | 6.180 | 85.68 |
| 32,768 | MorphBPE / TurkishDelight | 2.22 | 0.23 | 0 | 0 | 0 | 5.946 | 85.41 |
| 65,536 | Raw BPE | 2.16 | 0.18 | 0 | 0 | 0 | 5.488 | 83.72 |
| 65,536 | MorphBPE / TRmorph | 2.11 | 0.11 | 0 | 0 | 0 | 5.621 | 87.96 |
| 65,536 | MorphBPE / Zemberek | 2.11 | 0.11 | 0 | 0 | 0 | 5.433 | 88.91 |
| 65,536 | MorphBPE / TurkishDelight | 2.11 | 0.11 | 0 | 0 | 0 | 5.464 | 87.16 |
| 131,072 | Raw BPE | 2.10 | 0.10 | 0 | 0 | 0 | 5.008 | 86.56 |
| 131,072 | MorphBPE / TRmorph | 2.09 | 0.09 | 0 | 0 | 0 | 5.466 | 86.19 |
| 131,072 | MorphBPE / Zemberek | 2.09 | 0.09 | 0 | 0 | 0 | 5.482 | 86.24 |
| 131,072 | MorphBPE / TurkishDelight | 2.09 | 0.08 | 0 | 0 | 0 | 5.216 | 87.30 |

**Interpretation:** All rows tie at zero roundtrip failures and zero alignment exclusions. Lowest multi-character character-only rate: TRmorph and Zemberek (tied at displayed precision) at 32k; TRmorph, Zemberek and TurkishDelight (tied at displayed precision) at 64k; TurkishDelight at 128k. Highest recorded encoding tokens/s: Zemberek at 32k and 128k; TRmorph at 64k. These timings include concurrent-process contention, and more tokens/s can reflect producing more tokens rather than processing raw text faster.

## Provenance and rebuild

The evaluated runtime is an isolated source snapshot with per-file hashes, based on a prior repository revision. It is not a claim that the current `main` implementation ran this experiment. The source manifest records the evaluator, tokenizer and sample-preparation code hashes; environment and sample manifests preserve package versions, selected shards, document hashes and word/reference hashes. Original per-tokenizer outputs remain unchanged.

Rebuild these tables from completed measurements with `python -m scripts.build_tokenizer_matched_results`. The builder requires terminal successful scheduler accounting, all 24 row hashes, final receipt agreement, matching document/word/reference samples, matching tokenizer bytes, and valid rate/count arithmetic. Rebuilding is separate from rerunning the evaluator.

| Split | Tokenizer | Original JSON |
|---|---|---|
| train | bpe_32768 | [bpe_32768_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/bpe_32768_metrics.json) |
| train | morphbpe_trmorph_32768 | [morphbpe_trmorph_32768_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_trmorph_32768_metrics.json) |
| train | morphbpe_zemberek_32768 | [morphbpe_zemberek_32768_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_zemberek_32768_metrics.json) |
| train | morphbpe_tdelight_32768 | [morphbpe_tdelight_32768_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_tdelight_32768_metrics.json) |
| train | bpe_65536 | [bpe_65536_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/bpe_65536_metrics.json) |
| train | morphbpe_trmorph_65536 | [morphbpe_trmorph_65536_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_trmorph_65536_metrics.json) |
| train | morphbpe_zemberek_65536 | [morphbpe_zemberek_65536_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_zemberek_65536_metrics.json) |
| train | morphbpe_tdelight_65536 | [morphbpe_tdelight_65536_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_tdelight_65536_metrics.json) |
| train | bpe_131072 | [bpe_131072_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/bpe_131072_metrics.json) |
| train | morphbpe_trmorph_131072 | [morphbpe_trmorph_131072_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_trmorph_131072_metrics.json) |
| train | morphbpe_zemberek_131072 | [morphbpe_zemberek_131072_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_zemberek_131072_metrics.json) |
| train | morphbpe_tdelight_131072 | [morphbpe_tdelight_131072_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/train/morphbpe_tdelight_131072_metrics.json) |
| val | bpe_32768 | [bpe_32768_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/bpe_32768_metrics.json) |
| val | morphbpe_trmorph_32768 | [morphbpe_trmorph_32768_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_trmorph_32768_metrics.json) |
| val | morphbpe_zemberek_32768 | [morphbpe_zemberek_32768_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_zemberek_32768_metrics.json) |
| val | morphbpe_tdelight_32768 | [morphbpe_tdelight_32768_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_tdelight_32768_metrics.json) |
| val | bpe_65536 | [bpe_65536_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/bpe_65536_metrics.json) |
| val | morphbpe_trmorph_65536 | [morphbpe_trmorph_65536_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_trmorph_65536_metrics.json) |
| val | morphbpe_zemberek_65536 | [morphbpe_zemberek_65536_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_zemberek_65536_metrics.json) |
| val | morphbpe_tdelight_65536 | [morphbpe_tdelight_65536_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_tdelight_65536_metrics.json) |
| val | bpe_131072 | [bpe_131072_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/bpe_131072_metrics.json) |
| val | morphbpe_trmorph_131072 | [morphbpe_trmorph_131072_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_trmorph_131072_metrics.json) |
| val | morphbpe_zemberek_131072 | [morphbpe_zemberek_131072_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_zemberek_131072_metrics.json) |
| val | morphbpe_tdelight_131072 | [morphbpe_tdelight_131072_metrics.json](../../artifacts/tokenizer_metrics_matched_2026-11-09/results/val/morphbpe_tdelight_131072_metrics.json) |

**Reading note:** These links identify the original evidence for each row; their order is not a performance ranking.
