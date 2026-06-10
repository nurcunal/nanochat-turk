# Tokenizer Metrics Comparison

Tokenizer-only metrics are measured before model training. True BPB is model-dependent and should be reported from validation loss after each pretraining run; this table reports tokenizer compression, fertility, boundary behavior, reversibility, and throughput.

| Tokenizer | Impl. | Docs | Bytes | Tokens | Bytes/token ↑ | Tokens/word ↓ | Isolated word fertility ↓ | Single-token words ↑ | Long-word fertility ↓ | Boundary crossed ↓ | Crossing tok/1k ↓ | Roundtrip fail ↓ | Encode tok/s ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bpe_32768 | bpe | 50,000 | 186,328,421 | 37,228,014 | 5.0051 | 1.6157 | 2.0472 | 0.3105 | 2.8417 | 0.8395 | 210.3501 | 0.0000 | 12127686 |
| morphbpe_trmorph_32768 | morphbpe | 50,000 | 186,328,421 | 41,858,836 | 4.4514 | 1.8166 | 1.9890 | 0.3409 | 2.8046 | 0.4569 | 125.2956 | 0.0000 | 13419184 |

## Metric Notes

- `Bytes/token`: compression proxy; higher means fewer tokens for the same raw bytes.
- `Tokens/word`: corpus-level fertility; lower is usually better for Turkish.
- `Isolated word fertility`: token count when each sampled word is encoded alone.
- `Boundary crossed`: share of TRmorph morpheme boundaries crossed by at least one tokenizer token.
- `Crossing tok/1k`: tokenizer tokens per 1,000 sample tokens that cross a TRmorph boundary.
- `Roundtrip fail`: fraction of sampled documents where decode(encode(text)) differs from text.

## Source Files

- `bpe_32768`: `docs/tokenizer_tests/tokenizer_metrics/bpe_32768_metrics.json`
- `morphbpe_trmorph_32768`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_32768_metrics.json`
