# morphbpe_zemberek_32768

Zemberek MorphBPE tokenizer for the Turkish nanochat tokenizer ablation.

- Segmenter: Zemberek via `zemberek-python`
- Tokenizer implementation: MorphBPE
- Vocabulary size: 32,768
- Morpheme boundary marker: `U+E000`
- Boundary semantics: merge constraint only
- Runtime segmentation required: no
- UHeM tokenizer source:
  `/ari/users/nunal/nanochat-turk-morphbpe-zemberek-32768/tokenizers/morphbpe_zemberek_32768`
- UHeM segmented corpus:
  `/ari/users/nunal/nanochat-turk-morphbpe-zemberek-32768/segmented/zemberek-segmented`
- Raw tokenizer-only metric:
  `metrics/raw_metrics.json`
- TRmorph-reference tokenizer metric:
  `metrics/raw_vs_trmorph_reference_metrics.json`

The raw checked-in metric is a `10,000`-document diagnostic from the FineWeb-2
Turkish corpus. The TRmorph-reference metric uses the same `50,000`-document
boundary setup as the main tokenizer comparison table, so it is the Zemberek row
used for ranking. The large segmented corpus and model checkpoints remain on
UHeM.

## File Checksums

```text
1c4a7ee96f8e31614ffee399a3988e54eb8b8ad7cd224af877de2312c9fd9652  tokenizer.pkl
367f3c75f9bb7ec773ad97a7878ed7e9fd243b33ce71c7f6dad1e777f5d64760  tokenizer_config.json
a049a3d8aa2f227472598e73732bc72c40af10859da6198a29890de7043d7eb1  token_bytes.pt
e960427814fe90ffd2f3cbfd7f8266211a893cd186fd2f29ddd77a833643da3b  metrics/raw_metrics.json
fc8c8ac3cfd5ed9ba5872ea7c01441060cec896f815173263e86995395092e6a  metrics/raw_vs_trmorph_reference_metrics.json
```
