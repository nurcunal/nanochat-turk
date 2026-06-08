# Turkish Tokenizer Ablation Plan

This document preserves the preliminary study design for optimizing the
`nanochat-turk` tokenizer. The current UHeM production run,
`tr_d20_bpe_32768_chinchilla20`, is the first baseline cell in the study.

## Goal

The goal is to identify a tokenizer that is better suited to Turkish morphology
than raw frequency-only BPE, while keeping model training comparisons as fair as
possible. Turkish is agglutinative, so a useful tokenizer should avoid learning
tokens that cross morpheme boundaries when a reliable morpheme segmentation is
available.

The main research question is:

> Given an approximately fixed 1B-parameter Turkish LLM training budget, which
> tokenizer and vocabulary allocation yields the best intrinsic tokenizer
> metrics and CETVEL benchmark performance?

## Design Choice

The primary comparison will keep the following fixed:

- FineWeb-2 Turkish source, manifest, and document order.
- Training token positions: `17,930,649,600` tokens.
- Batch schedule: `TOTAL_BATCH_SIZE=1,048,576`, `NUM_ITERATIONS=17,100`.
- Training recipe, optimizer schedule, evaluation harness, and UHeM hardware
  profile.
- Exact model shape within each vocabulary-size tier.

Across vocabulary-size tiers, depth is adjusted to keep total parameters close
to the current 32k baseline. This makes the cross-vocabulary comparison a
fixed-budget allocation question: more vocabulary capacity versus more
transformer depth.

One caveat must be reported in the paper: fixed tokenizer positions are not the
same as fixed raw bytes or fixed document count, because tokenizers have
different fertility. For every full run, we should report raw bytes and document
counts consumed by the dataloader so readers can interpret exposure differences.

## Primary Matched Matrix

The primary model tiers are:

| Vocab size | Depth | Approx params | Training token positions | Tokens/param |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | d20 | 896.5M | 17.930B | 20.00 |
| 65,536 | d16 | 872.4M | 17.930B | 20.55 |
| 131,072 | d12 | 890.2M | 17.930B | 20.14 |

The minimum full-scale run matrix is:

| Vocab | Depth | Tokenizer | Model tag |
| ---: | ---: | --- | --- |
| 32,768 | d20 | raw BPE | `tr_d20_bpe_32768_chinchilla20` |
| 32,768 | d20 | MorphBPE + TRmorph | `tr_d20_morphbpe_trmorph_32768_chinchilla20` |
| 32,768 | d20 | MorphBPE + Zemberek | `tr_d20_morphbpe_zemberek_32768_chinchilla20` |
| 32,768 | d20 | MorphBPE + TurkishDelightNLP | `tr_d20_morphbpe_tdelight_32768_chinchilla20` |
| 65,536 | d16 | raw BPE | `tr_d16_bpe_65536_chinchilla20` |
| 65,536 | d16 | MorphBPE + TRmorph | `tr_d16_morphbpe_trmorph_65536_chinchilla20` |
| 65,536 | d16 | MorphBPE + Zemberek | `tr_d16_morphbpe_zemberek_65536_chinchilla20` |
| 65,536 | d16 | MorphBPE + TurkishDelightNLP | `tr_d16_morphbpe_tdelight_65536_chinchilla20` |
| 131,072 | d12 | raw BPE | `tr_d12_bpe_131072_chinchilla20` |
| 131,072 | d12 | MorphBPE + TRmorph | `tr_d12_morphbpe_trmorph_131072_chinchilla20` |
| 131,072 | d12 | MorphBPE + Zemberek | `tr_d12_morphbpe_zemberek_131072_chinchilla20` |
| 131,072 | d12 | MorphBPE + TurkishDelightNLP | `tr_d12_morphbpe_tdelight_131072_chinchilla20` |

Optional later controls:

- `morphbpe_hybrid`: rule-based segmentation when confident, neural fallback
  otherwise.
- `preseg_bpe_*`: BPE trained on boundary-marked text as a control that tests
  explicit morpheme markers in the model stream.
- SentencePiece BPE controls.
- SentencePiece unigram controls.
- A smaller matched-raw-byte robustness run for the best tokenizer candidates.

## Tokenizer Implementation Direction

The publishable implementation should be a true boundary-aware tokenizer, not a
model trained on visibly morpheme-spaced Turkish. The tokenizer may use
segmentation during tokenizer training, but the final tokenizer should encode
raw Turkish text without a runtime segmenter and decode back to normal Turkish.

Implementation principles:

- Segment words into morpheme spans before tokenizer training.
- Prevent BPE merges from crossing morpheme boundaries.
- Save a normal raw-text tokenizer for `morphbpe_*`; do not require users or
  CETVEL prompts to be segmented at inference.
- Keep `preseg_bpe_*` as a control ablation, not the main method.
- Keep tokenizer artifacts compatible with the existing `nanochat.tokenizer`
  interface.
- Cache segmentation and tokenizer outputs outside the GPU training hot path.
- Preserve raw document text for evaluation and generation.

## Evaluation Plan

Run tokenizer-only evaluation before committing UHeM GPU time:

- Compression: bytes/token, chars/token, fertility, tokens/word.
- Turkish morphology: boundary violation rate, token purity, morphological edit
  distance, morphological consistency F1.
- Systems: encode throughput and segmentation/cache throughput.
- Coverage: behavior on noisy web text, OOV words, punctuation, numbers, URLs,
  code-like text, and named entities.

The current segmenter smoke tests, full-shard inventory, benchmark metrics, and
local JSON output paths are tracked in
`docs/tokenizer_tests/segmenter_benchmark_status.md`.

Promote only the strongest tokenizer variants to full model training if the
tokenizer-only metrics rule out weak candidates.

Full model evaluation:

- In-training validation BPB.
- `scripts.cetvel_eval --suite fast` for iteration.
- `scripts.cetvel_eval --suite core` for primary paper results.
- `scripts.cetvel_eval --suite full` when generation-heavy tasks are stable and
  compute budget permits.
- Report raw bytes/documents consumed to account for fertility differences.

## Paper Framing

The paper should frame the study as a controlled Turkish tokenizer optimization
experiment under a fixed model budget:

- Same corpus source and ordering.
- Same training token-position budget.
- Approximately same total parameter budget.
- Same architecture and depth within each vocabulary tier.
- Same training and evaluation stack.

The central claim should be conservative and defensible:

> We introduce and evaluate a reproducible Turkish MorphBPE-style tokenizer
> pipeline, then test whether morphology-aware tokenization improves Turkish LLM
> training and CETVEL performance under a matched approximately 1B-parameter
> compute budget.

Avoid claiming that no Turkish morphology-aware tokenizer exists. The stronger
novelty claim is the controlled decoder-only LLM ablation suite for Turkish
under matched data, parameter, and evaluation conditions.
