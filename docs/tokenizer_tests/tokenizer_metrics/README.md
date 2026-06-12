# Tokenizer Metrics

This folder stores tokenizer-only metrics for produced tokenizer artifacts.

Current comparison:

- `bpe_32768`: raw unsegmented BPE baseline.
- `morphbpe_trmorph_32768`: TRmorph-constrained MorphBPE tokenizer.
- `kumru_2b`: `vngrs-ai/Kumru-2B` public Turkish LLM tokenizer.
- `berturk_cased`: `dbmdz/bert-base-turkish-cased` WordPiece tokenizer.
- `cosmos_turkish_gpt2`: `ytu-ce-cosmos/turkish-gpt2` tokenizer.
- `turna`: `boun-tabi-LMG/TURNA` tokenizer.
- `vbart_large_base`: `vngrs-ai/VBART-Large-Base` tokenizer.

Additional checked-in single-tokenizer metrics:

- `morphbpe_zemberek_32768_raw_metrics.json`: raw-text `10,000`-document metric
  for the Zemberek MorphBPE tokenizer. This file is useful for artifact
  inspection, but it is not part of the ranked comparison because it does not use
  the same `50,000`-document TRmorph-reference boundary setup.

The checked-in comparison table is a preliminary `50,000`-document sample from
the TRmorph-segmented FineWeb-2 Turkish corpus. Paper-facing tokenizer metrics
should use the full-corpus UHeM run in
`runs/uhem_tokenizer_metrics_compare_32k.sbatch`, which defaults to
`MAX_DOCS=0` and writes to `tokenizer_metrics_32k_full`.

Operational note: the original full run is intentionally conservative and
single-process. The optimized companion
`runs/uhem_tokenizer_metrics_compare_32k_parallel.sbatch` keeps the same metric
definitions but runs `scripts.tokenizer_metrics --workers N` over parquet row
groups and writes to `tokenizer_metrics_32k_full_parallel`. Use that path when a
full-corpus run is already in flight and we want a faster side-by-side result
without overwriting the baseline job's JSON files.

The boundary marker is stripped before encoding, so all tokenizers receive
identical raw Turkish text. The segmented form is used only as a reference for
measuring whether tokenizer tokens cross TRmorph morpheme boundaries.

Public Hugging Face tokenizers are loaded from tokenizer files only; no model
weights are downloaded for these metrics.

True BPB is not a tokenizer-only metric; it requires a trained model and is
reported from validation loss after pretraining. The files here report
pretraining-independent tokenizer diagnostics: compression, fertility,
boundary-crossing behavior, reversibility, vocabulary shape, and encode speed.
