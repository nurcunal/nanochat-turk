# bpe_32768

Baseline raw BPE tokenizer for the Turkish nanochat runs.

- Segmenter: none
- Tokenizer implementation: BPE
- Vocabulary size: 32,768
- Training data: raw FineWeb-2 Turkish Latin corpus
- UHeM tokenizer source:
  `/ari/users/nunal/nanochat-turk-d20-bpe32k/tokenizers/bpe_32768`
- UHeM raw corpus alias:
  `/ari/users/nunal/nanochat-turk-morphbpe-trmorph-32768/datasets/raw-unsegmented-fineweb2-tur-latn`
- Tokenizer-only metrics:
  `metrics/raw_vs_trmorph_reference_metrics.json`

This is the unsegmented baseline tokenizer used by the current
`tr_d20_bpe_32768_chinchilla20`/`tr-d20-bpe32k` baseline family.
