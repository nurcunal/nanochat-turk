# Tokenizer Ranking Methodology

This note explains the diagnostic score used in the main tokenizer comparison.
It is a transparent candidate-screening heuristic, not a claim that one
tokenizer is universally best. The final project winner must still be decided
by matched LLM validation BPB and CETVEL results.

## What The Score Is For

The ranking is designed for the specific Turkish MorphBPE study. The research
question is not simply "which tokenizer is most compact?" It is:

> Under a fixed Turkish LLM training recipe, does morphology-aware tokenization
> improve Turkish tokenizer behavior and downstream model quality compared with
> raw BPE?

For that reason, the score intentionally prioritizes morpheme-boundary
preservation while still accounting for compression, word fertility,
reversibility, and encoding speed.

## Comparable Inputs

Rows are included only when they use the same `50,000`-document
TRmorph-reference sample. The boundary marker is stripped before encoding, so
every tokenizer receives identical raw Turkish text. TRmorph boundaries are used
only as a diagnostic reference for boundary-crossing metrics.

The main README may display d20 validation BPB beside tokenizer rows when a
matched model run exists. That BPB is model-facing evidence and is not included
in the tokenizer-only diagnostic score below.

## Formula

1. For each metric, tokenizers are ranked best-to-worst.
2. Ties receive the average rank.
3. The diagnostic score is based on a weighted average rank:

```text
Diagnostic score = 100 * (1 - (weighted average rank - 1) / (N - 1))
```

4. The primary score applies a raw-text safety gate:

```text
Primary score = diagnostic score * (1 - roundtrip failure rate)
```

The round-trip factor prevents lossy or normalizing tokenizers from outranking
lossless raw-text candidates. Such tokenizers can remain useful public baselines,
but they are not safe drop-in tokenizers for raw nanochat pretraining.

## Weight Rationale

| Group | Metric | Weight | Rationale |
| --- | --- | ---: | --- |
| Morphology | Boundary crossed down | 28% | Directly measures whether reference morpheme boundaries survive tokenization. This is the core MorphBPE hypothesis. |
| Morphology | Crossing tokens per 1k down | 17% | Measures how many produced tokens violate a reference boundary, preventing the boundary rate alone from hiding frequent crossing tokens. |
| Compression | Bytes/token up | 15% | Captures how much raw text a fixed token budget can cover. Important, but not the only objective in a morphology study. |
| Word fertility | Corpus tokens/word down | 8% | Measures fragmentation on running text. |
| Word fertility | Isolated word fertility down | 8% | Measures word-level fragmentation without document context. |
| Word fertility | Single-token words up | 6% | Rewards compact handling of common whole words. |
| Word fertility | Long-word fertility down | 6% | Focuses on agglutinative forms, where Turkish tokenization failures are most visible. |
| Throughput | Encode tokens/sec up | 12% | Operationally useful for training and evaluation, but lower-weighted because it is implementation- and hardware-dependent. |

The group totals are:

- morphology preservation: `45%`;
- word fertility: `28%`;
- compression: `15%`;
- throughput: `12%`.

This weighting is defensible only because the project is explicitly testing
morphology-aware tokenization. A generic tokenizer benchmark would likely use
different weights.

## Why Rank-Based Scoring

The metrics have different units and scales: rates, tokens per word, bytes per
token, and tokens per second. Rank-based scoring keeps the formula readable,
avoids arbitrary unit normalization, and reduces sensitivity to extreme values.
The trade-off is that the score loses information about the size of gaps between
tokenizers. For paper reporting, the raw metric table should always be shown
alongside the score.

## Sensitivity Check

Using the checked-in `50,000`-document table:

| Scenario | Ranking implication |
| --- | --- |
| Current morphology-prioritized weights | TRmorph #1; Zemberek and Kumru tied #2. |
| Equal weights across the 8 metrics | Kumru rises to #1; TRmorph and Cosmos tie next; Zemberek falls behind them. |
| No throughput, other weights rescaled | Kumru slightly edges TRmorph; Zemberek remains close. |
| More morphology-heavy weights | TRmorph remains #1; Zemberek moves ahead of Kumru. |
| Compression-heavy weights | Cosmos GPT-2 and Kumru rise because compactness dominates boundary preservation. |

This sensitivity is expected. The diagnostic score encodes the project's
priorities; it does not prove an objective universal ordering. The report should
therefore say:

> Under a morphology-prioritized diagnostic score, TRmorph MorphBPE ranks first
> and Zemberek MorphBPE ties the strongest public lossless baseline. This
> supports carrying MorphBPE candidates into matched LLM training, but the final
> tokenizer claim depends on validation BPB and CETVEL.
