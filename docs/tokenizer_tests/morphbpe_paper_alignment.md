# MorphBPE Paper Alignment

This note tracks how the Turkish tokenizer work maps to the MorphBPE paper:

> MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for
> Efficient LLM Training Across Morphologies, arXiv:2502.00894.

## Implemented Method

The main `morphbpe_*` tokenizers implement the paper method:

1. Segment the tokenizer-training corpus into surface morphemes.
2. Learn BPE merges from chunks that are split at morpheme boundaries.
3. Prevent BPE merge evidence from crossing marked morpheme boundaries.
4. Save a normal raw-text BPE tokenizer.
5. Use raw Turkish text for pretraining, CETVEL evaluation, and inference.

The relevant implementation lives in:

- `nanochat/morphology/morphbpe.py`
- `scripts/tok_train.py`
- `scripts/morph_segment_corpus.py`

The `preseg_bpe_*` path is only a control: it exposes boundary-marked text to
the model stream and is not the primary paper-faithful MorphBPE method.

## Segmenters

The Turkish version supports multiple morphology providers so segmentation
quality itself can be ablated:

- `morphbpe_trmorph_*`
- `morphbpe_zemberek_*`
- `morphbpe_tdelight_*`

Each segmenter writes a distinct segmented corpus and tokenizer artifact. The
final tokenizer name must include both method and segmenter.

## Paper Intrinsic Metrics

`scripts/tokenizer_metrics.py` now implements the MorphBPE paper's tokenizer
metrics when evaluated on a boundary-marked reference corpus:

| Paper metric | Repo field | Direction |
| --- | --- | --- |
| Fertility `phi` | `tokens_per_word`, `token_fertility` | lower |
| Morphological Edit Distance `mu_e` | `morphology.alignment.morphological_edit_distance` | lower |
| Normalized edit distance | `morphology.alignment.morphological_edit_distance_normalized` | lower |
| Morphological Consistency precision | `morphology.consistency.precision_mean` | higher |
| Morphological Consistency recall | `morphology.consistency.recall_mean` | higher |
| Morphological Consistency F1 `mu_c` | `morphology.consistency.f1_mean` | higher |

The Morph-Consistency defaults follow the paper:

- `k=100` clusters: `--morph-consistency-clusters 100`
- `C=50` pairs per cluster: `--morph-consistency-pairs-per-cluster 50`
- `N=10` resamples: `--morph-consistency-resamples 10`

When `scikit-learn` is available, clustering uses `MiniBatchKMeans` over hashed
morpheme features. If it is unavailable, the job emits a deterministic
`hash_fallback` clustering label instead of failing late in a long UHeM run.

## Additional Repo Metrics

The repo also reports diagnostics not central to the paper but useful for
Turkish LLM engineering:

- bytes/token and chars/token compression;
- isolated-word fertility;
- morpheme-boundary crossing rate;
- exact morpheme-sequence match rate;
- round-trip decode failure rate;
- vocabulary byte-shape statistics;
- encode throughput.

These should be presented as engineering diagnostics, while `phi`, `mu_e`, and
`mu_c` should be presented as the paper-facing tokenizer metrics.

## Extrinsic Metrics

The paper compares cross-entropy loss for same-vocabulary BPE versus MorphBPE
models. Our Turkish study records:

- validation BPB during base pretraining;
- training cross-entropy trajectory from W&B;
- CETVEL core/full benchmark metrics after base training.

Cross-entropy or BPB should only be compared directly within a fixed vocabulary
size because vocabulary size changes the prediction space. CETVEL can be
reported across all matched parameter-budget runs, with raw bytes/documents
consumed included to account for tokenizer fertility.

## Current Gaps

The framework is method-complete for a Turkish MorphBPE reproduction. Remaining
paper-facing work is experimental rather than architectural:

- finish the full-corpus tokenizer metric job with `mu_e` and `mu_c`;
- decide whether to promote `morphbpe_tdelight_32k` to a matched d20 model;
- run raw BPE, TRmorph, Zemberek, and TurkishDelight CETVEL under matched model
  settings;
- run CETVEL for the completed `64k` and `128k` vocab-size models;
- optionally run the vocabulary-size sweep used in the paper for selecting
  morphology-aligned vocabulary sizes.
