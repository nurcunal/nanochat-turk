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

Metrics are computed on the same first `50,000` train documents from the
TRmorph-segmented FineWeb-2 Turkish corpus. The boundary marker is stripped
before encoding, so both tokenizers receive identical raw Turkish text. The
segmented form is used only as a reference for measuring whether tokenizer
tokens cross TRmorph morpheme boundaries.

Public Hugging Face tokenizers are loaded from tokenizer files only; no model
weights are downloaded for these metrics.

True BPB is not a tokenizer-only metric; it requires a trained model and is
reported from validation loss after pretraining. The files here report
pretraining-independent tokenizer diagnostics: compression, fertility,
boundary-crossing behavior, reversibility, vocabulary shape, and encode speed.
