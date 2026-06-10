# Tokenizer Metrics

This folder stores tokenizer-only metrics for produced tokenizer artifacts.

Current comparison:

- `bpe_32768`: raw unsegmented BPE baseline.
- `morphbpe_trmorph_32768`: TRmorph-constrained MorphBPE tokenizer.

Metrics are computed on the same first `50,000` train documents from the
TRmorph-segmented FineWeb-2 Turkish corpus. The boundary marker is stripped
before encoding, so both tokenizers receive identical raw Turkish text. The
segmented form is used only as a reference for measuring whether tokenizer
tokens cross TRmorph morpheme boundaries.

True BPB is not a tokenizer-only metric; it requires a trained model and is
reported from validation loss after pretraining. The files here report
pretraining-independent tokenizer diagnostics: compression, fertility,
boundary-crossing behavior, reversibility, vocabulary shape, and encode speed.
