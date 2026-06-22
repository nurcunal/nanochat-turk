# Turkish Tokenizer Ablation Plan

This document preserves the preliminary study design for optimizing the
`nanochat-turk` tokenizer. The current UHeM production run,
`tr_d20_bpe_32k`, is the first baseline cell in the study.

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
| 32k | d20 | 896.5M | 17.930B | 20.00 |
| 64k | d16 | 872.4M | 17.930B | 20.55 |
| 128k | d12 | 890.2M | 17.930B | 20.14 |

The current full-scale run matrix is:

| Vocab | Depth | Tokenizer | Model tag | Step | Final val BPB | Lowest val BPB | Current status |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 32k | d20 | raw BPE | `tr_d20_bpe_32k` | 17100 | 0.6232 | 0.6232 | CETVEL core compared; tasks 01-13 archived |
| 32k | d20 | MorphBPE + TRmorph | `tr_d20_morphbpe_trmorph_32k` | 17100 | 0.6266 | 0.6266 | CETVEL core compared |
| 32k | d20 | MorphBPE + Zemberek | `tr_d20_morphbpe_zemberek_32k` | 17100 | 0.6250 | 0.6250 | CETVEL core compared |
| 32k | d20 | MorphBPE + TurkishDelightNLP | `tr_d20_morphbpe_tdelight_32k` | - | - | - | tokenizer exists; no full checkpoint found |
| 64k | d16 | raw BPE | `tr_d16_bpe_64k` | 17100 | 0.6409 | 0.6409 | CETVEL core compared |
| 64k | d16 | MorphBPE + TRmorph | `tr_d16_morphbpe_trmorph_64k` | 17100 | 0.6521 | 0.6521 | CETVEL core compared |
| 64k | d16 | MorphBPE + Zemberek | `tr_d16_morphbpe_zemberek_64k` | 17100 | 0.6514 | 0.6514 | CETVEL core compared |
| 64k | d16 | MorphBPE + TurkishDelightNLP | `tr_d16_morphbpe_tdelight_64k` | 17100 | 0.6510 | 0.6510 | CETVEL core compared |
| 128k | d12 | raw BPE | `tr_d12_bpe_128k` | 17100 | 0.6749 | 0.6749 | CETVEL core compared |
| 128k | d12 | MorphBPE + TRmorph | `tr_d12_morphbpe_trmorph_128k` | 17100 | 0.6917 | 0.6917 | CETVEL core compared |
| 128k | d12 | MorphBPE + Zemberek | `tr_d12_morphbpe_zemberek_128k` | 17100 | 0.6940 | 0.6940 | CETVEL core compared |
| 128k | d12 | MorphBPE + TurkishDelightNLP | `tr_d12_morphbpe_tdelight_128k` | 17100 | 0.6820 | 0.6820 | CETVEL core compared |

The same model-name/BPB inventory with UHeM metadata paths is maintained in
[`docs/model_bpb_inventory.md`](model_bpb_inventory.md).

## Current Core-12 Evidence

The current model-facing comparison is documented in
[`docs/cetvel_model_comparison.md`](cetvel_model_comparison.md) and summarized
under
[`artifacts/cetvel_core12_tokenizer_ablation_2026-06-22`](../artifacts/cetvel_core12_tokenizer_ablation_2026-06-22/).

| Vocab | Run | Tokenizer | Segmenter | CETVEL job | Elapsed | ex/s up | Speed vs raw | Val BPB | Lowest BPB | Train loss | Core-11 macro | Delta | XQuAD F1 | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32k | raw BPE d20 | `bpe_32k` | none | `493293` | 50m20s | 13.06 | 1.000x | 0.6232 | 0.6232 | 2.4899 | 0.4514 | +0.0000 | 3.0985 | +0.0000 |
| 32k | MorphBPE + TRmorph d20 | `morphbpe_trmorph_32k` | TRmorph | `494056` | 52m38s | 12.49 | 0.956x | 0.6266 | 0.6266 | 2.0106 | 0.4541 | +0.0027 | 3.4786 | +0.3801 |
| 32k | MorphBPE + Zemberek d20 | `morphbpe_zemberek_32k` | Zemberek | `494057` | 50m30s | 13.02 | 0.997x | 0.6250 | 0.6250 | 2.3227 | 0.4618 | +0.0104 | 3.2633 | +0.1648 |
| 64k | raw BPE d16 | `bpe_64k` | none | `496898` | 43m27s | 15.13 | 1.000x | 0.6409 | 0.6409 | 2.5812 | 0.4590 | +0.0000 | 2.8576 | +0.0000 |
| 64k | MorphBPE + TRmorph d16 | `morphbpe_trmorph_64k` | TRmorph | `496899` | 43m09s | 15.23 | 1.007x | 0.6521 | 0.6521 | 2.2754 | 0.4532 | -0.0058 | 3.2778 | +0.4202 |
| 64k | MorphBPE + Zemberek d16 | `morphbpe_zemberek_64k` | Zemberek | `496900` | 42m45s | 15.38 | 1.016x | 0.6514 | 0.6514 | 2.3013 | 0.4568 | -0.0022 | 2.8956 | +0.0380 |
| 64k | MorphBPE + TurkishDelightNLP d16 | `morphbpe_tdelight_64k` | TurkishDelightNLP | `496901` | 40m52s | 16.09 | 1.063x | 0.6510 | 0.6510 | 2.4869 | 0.4567 | -0.0023 | 3.3280 | +0.4704 |
| 128k | raw BPE d12 | `bpe_128k` | none | `496902` | 35m13s | 18.67 | 1.000x | 0.6749 | 0.6749 | 3.0976 | 0.4651 | +0.0000 | 2.2674 | +0.0000 |
| 128k | MorphBPE + TRmorph d12 | `morphbpe_trmorph_128k` | TRmorph | `496903` | 35m09s | 18.70 | 1.002x | 0.6917 | 0.6917 | 2.5947 | 0.4503 | -0.0148 | 2.3517 | +0.0843 |
| 128k | MorphBPE + Zemberek d12 | `morphbpe_zemberek_128k` | Zemberek | `496904` | 35m19s | 18.61 | 0.997x | 0.6940 | 0.6940 | 2.6000 | 0.4618 | -0.0033 | 2.9685 | +0.7011 |
| 128k | MorphBPE + TurkishDelightNLP d12 | `morphbpe_tdelight_128k` | TurkishDelightNLP | `496905` | 33m25s | 19.67 | 1.054x | 0.6820 | 0.6820 | 2.6477 | 0.4481 | -0.0170 | 2.2498 | -0.0176 |

- `32k`: best core-11 macro is `morphbpe_zemberek_32k` (0.4618); best XQuAD F1 is `morphbpe_trmorph_32k` (3.4786); best validation BPB is `bpe_32k` (0.6232).
- `64k`: best core-11 macro is `bpe_64k` (0.4590); best XQuAD F1 is `morphbpe_tdelight_64k` (3.3280); best validation BPB is `bpe_64k` (0.6409).
- `128k`: best core-11 macro is `bpe_128k` (0.4651); best XQuAD F1 is `morphbpe_zemberek_128k` (2.9685); best validation BPB is `bpe_128k` (0.6749).

This evidence is mixed: raw BPE remains strongest by validation BPB in every completed vocabulary tier, while MorphBPE variants win selected CETVEL slices such as 32k core-11 macro or XQuAD F1 in some tiers. Final train loss is logged for operational completeness but is not the main cross-tokenizer loss metric. CETVEL examples/sec is an end-to-end inference-throughput proxy for the matched core-12 benchmark slice, not an isolated hardware benchmark.

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

## Run Safety

Each tokenizer artifact must use a unique `NANOCHAT_TOKENIZER_NAME`, and each
model should use a model tag that repeats the tokenizer name. New base
checkpoints record `tokenizer_name`, `tokenizer_dir`, and `tokenizer_config` in
`meta_*.json`. Evaluation loaders prefer the checkpoint-recorded tokenizer over
the current shell environment, so same-vocab ablations cannot silently evaluate
with the wrong tokenizer. HF upload also checks the recorded tokenizer before
publishing the checkpoint/tokenizer bundle.

## Evaluation Plan

Run tokenizer-only evaluation before committing UHeM GPU time:

- Compression: bytes/token, chars/token, fertility, tokens/word.
- Turkish MorphBPE paper metrics: fertility `phi`, Morphological Edit Distance
  `mu_e`, and Morphological Consistency precision/recall/F1 `mu_c`.
- Turkish engineering diagnostics: boundary violation rate, exact morpheme
  sequence rate, token purity, reversibility, vocabulary shape, and throughput.
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
