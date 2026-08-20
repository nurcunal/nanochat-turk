# Isolated Turkish Data Environment

This environment is only for CPU corpus preprocessing. It is deliberately
separate from Nanochat's GPU training environment; finalized training consumes
only sealed Parquet, tokenizer, and JSON manifest artifacts.

Pinned production backends:

- DataTrove at commit `a649de79c14a550dc90f48a15c025f2dd3fd3b57`
  (package version 0.10.0), using 64-bit, 5-word MinHash with 14 buckets and
  eight hashes per bucket.
- GlotLID v3 at Hugging Face commit
  `85cd6716494360367b75f642b5bc78667605d0b4`, file `model_v3.bin`, LFS
  SHA-256 `a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e`.
- `fasttext-numpy2-wheel==0.9.2`; the legacy `fasttext-wheel` distribution is
  excluded because its prediction wrapper is incompatible with NumPy 2.x.
- The Python/SQLite MinHash and lexical Turkish scorer in
  `nanochat.turkish_corpus` are reference checks for fixtures/smokes only and
  are rejected by the production command.

Create the preprocessing environment outside the repository with:

```bash
uv sync --project environments/turkish-data --locked
```

Before sealing a backend output receipt, the GlotLID gate must pass the frozen
Turkish/Azerbaijani/Turkmen/Crimean-Tatar/English confusion-set calibration.
The environment lock hash, model hash, source receipt hash, thresholds, MinHash
configuration, and backend output file hashes are recorded in that receipt.
