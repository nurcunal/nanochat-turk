# Tokenizer results

Archived measurements, consolidated 9 November 2026. All twelve local tokenizers are included. Full precision, original metrics and SHA-256 source hashes are in [JSON](tokenizer_results.json); flat values are in [CSV](tokenizer_results.csv).

## Requested metrics

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

### Definitions and coverage

- **Vocabulary size:** actual vocabulary including nine special tokens for every local tokenizer.
- **Subword efficiency:** total encoded document tokens divided by regex word count (`\w+`). This includes punctuation/spacing tokens in the numerator. Isolated fertility encodes each sampled word separately and is reported alongside it.
- **Whole-word coverage:** fraction of sampled word occurrences represented by exactly one token when encoded in isolation. It is sensitive to whitespace conventions and differs from context-aware word coverage.
- **Morpheme preservation:** the archived **exact morpheme-sequence rate** counts words whose token pieces exactly match the automatic morpheme sequence. It is a strict word-level proxy. The requested fraction of individual morphemes preserved as intact tokens was not stored and needs the new evaluator.
- **Boundary preservation:** `(reference boundaries - crossed boundaries) / reference boundaries`, using all 12,706,795 reference boundaries. This measures retained reference boundaries; it does not penalize extra token splits, so it is not boundary precision or F1.
- **Morphological consistency:** retain the original MorphBPE shared-token/shared-morpheme pair precision, recall and F1 (`k=100`, `C=50`, `N=10`). It must not be relabelled as boundary F1. The new evaluator reports explicit boundary P/R/F1 separately.
- **Character-only coverage:** fraction of words split entirely into single Unicode-character tokens; it was not measured in the archived outputs. A UTF-8 byte fragment is not a character token. The added evaluator also separates single-byte-only segmentation.
- **UNK:** N/A for the twelve local byte-complete BPE/MorphBPE tokenizers, which have no UNK mechanism. For tokenizers with an UNK token, the evaluator records count and token/word incidence; no historical count is fabricated.
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

The JSON also preserves all original distributions, unique-word measurements, resampling standard deviations, vocabulary shape, worker/thread counts and timing. The original [comparison](../tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md) remains available.

## External tokenizer references

These are tokenizer-only diagnostics; their pretrained models are **not** part of the controlled LLM ablation.

| Tokenizer | Vocabulary | Tokens/word | Whole word: one token % | Exact morpheme sequence % | Preserved boundaries % | Morph consistency F1 |
|---|---:|---:|---:|---:|---:|---:|
| kumru_2b | 50,176 | 1.6677 | 50.86 | 45.15 | 22.55 | 0.3436 |
| cosmos_turkish_gpt2 | 50,257 | 1.5570 | 38.47 | 39.65 | 13.06 | 0.3001 |

## Interpretation and limits

At every vocabulary size, MorphBPE preserves more TRmorph reference boundaries and has better recorded morphology edit distance and consistency than raw BPE, while requiring more document tokens per word. TRmorph training and evaluation share an analyzer, so these are automatic-reference results. The 32k morphology and isolated-word caps differ across treatments, preventing a fully matched word-level estimate. Corpus-level fertility and boundary counts use the common 50,000 documents.

Isolated-word fertility often improves with MorphBPE even though document tokens/word rises; the two measures capture different contexts. Job 496882 has now been verified COMPLETED and recovered separately in the [full-corpus report](tokenizer_full_corpus.md). Fresh measurements retain corpus/split identity, equal sample caps and separate provenance. No significance or causal advantage is claimed from these single-run diagnostics.

Rebuild: `python -m scripts.build_tokenizer_study_results`.
