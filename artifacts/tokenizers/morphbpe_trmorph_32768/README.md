# morphbpe_trmorph_32768

TRmorph MorphBPE tokenizer for the Turkish nanochat tokenizer ablation.

- Segmenter: TRmorph FST via `flookup`
- Tokenizer implementation: MorphBPE
- Vocabulary size: 32,768
- Morpheme boundary marker: `U+E000`
- Boundary semantics: merge constraint only
- Runtime segmentation required: no
- UHeM tokenizer source:
  `/ari/users/nunal/nanochat-turk-morphbpe-trmorph-32768/tokenizers/morphbpe_trmorph_32768`
- UHeM segmented corpus alias:
  `/ari/users/nunal/nanochat-turk-morphbpe-trmorph-32768/datasets/trmorph-segmented`
- Tokenizer-only metrics:
  `metrics/raw_vs_trmorph_reference_metrics.json`

The large segmented corpus is not stored in GitHub. It remains on UHeM under
the alias above, with manifests archived in `provenance/`.
