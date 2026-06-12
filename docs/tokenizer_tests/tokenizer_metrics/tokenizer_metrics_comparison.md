# Tokenizer Metrics Comparison

Tokenizer-only metrics are measured before model training. True BPB is model-dependent and should be reported from validation loss after each pretraining run; this table reports tokenizer compression, fertility, boundary behavior, reversibility, and throughput.

| Tokenizer | Impl. | Docs | Bytes | Tokens | Bytes/token ↑ | Tokens/word ↓ | Isolated word fertility ↓ | Single-token words ↑ | Long-word fertility ↓ | Boundary crossed ↓ | Crossing tok/1k ↓ | Roundtrip fail ↓ | Encode tok/s ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bpe_32768 | bpe | 50,000 | 186,328,421 | 37,228,014 | 5.0051 | 1.6157 | 2.0472 | 0.3105 | 2.8417 | 0.8395 | 210.3501 | 0.0000 | 12127686 |
| morphbpe_trmorph_32768 | morphbpe | 50,000 | 186,328,421 | 41,858,836 | 4.4514 | 1.8166 | 1.9890 | 0.3409 | 2.8046 | 0.4569 | 125.2956 | 0.0000 | 13419184 |
| morphbpe_zemberek_32768 | morphbpe | 50,000 | 186,328,421 | 41,443,968 | 4.4959 | 1.7986 | 1.9810 | 0.3406 | 2.7639 | 0.5906 | 138.4285 | 0.0000 | 1472567 |
| kumru_2b | bpe | 50,000 | 186,328,421 | 38,427,427 | 4.8488 | 1.6677 | 1.6804 | 0.5003 | 2.1697 | 0.7745 | 189.9358 | 0.0000 | 818251 |
| berturk_cased | wordpiece | 50,000 | 186,328,421 | 36,609,010 | 5.0897 | 1.5888 | 1.4201 | 0.7003 | 1.8625 | 0.8045 | 205.0115 | 0.9929 | 863345 |
| cosmos_turkish_gpt2 | bpe | 50,000 | 186,328,421 | 35,875,345 | 5.1938 | 1.5570 | 1.8836 | 0.3740 | 2.6116 | 0.8694 | 220.6239 | 0.0000 | 681476 |
| turna | unigram | 50,000 | 186,328,421 | 35,952,018 | 5.1827 | 1.5603 | 1.3966 | 0.7181 | 1.7958 | 0.8015 | 203.4204 | 0.9521 | 540988 |
| vbart_large_base | unigram | 50,000 | 186,328,421 | 35,952,025 | 5.1827 | 1.5603 | 1.3966 | 0.7181 | 1.7958 | 0.8015 | 203.4204 | 0.9521 | 579818 |

## Metric Notes

- `Bytes/token`: compression proxy; higher means fewer tokens for the same raw bytes.
- `Tokens/word`: corpus-level fertility; lower is usually better for Turkish.
- `Isolated word fertility`: token count when each sampled word is encoded alone.
- `Boundary crossed`: share of TRmorph morpheme boundaries crossed by at least one tokenizer token.
- `Crossing tok/1k`: tokenizer tokens per 1,000 sample tokens that cross a TRmorph boundary.
- `Roundtrip fail`: fraction of sampled documents where decode(encode(text)) differs from text.
- High `Roundtrip fail` is expected for some encoder/seq2seq tokenizers that normalize or do not preserve raw text exactly.

## Source Files

- `bpe_32768`: `docs/tokenizer_tests/tokenizer_metrics/bpe_32768_metrics.json`
- `morphbpe_trmorph_32768`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_32768_metrics.json`
- `morphbpe_zemberek_32768`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_32768_metrics.json`
- `kumru_2b`: `docs/tokenizer_tests/tokenizer_metrics/kumru_2b_metrics.json`
- `berturk_cased`: `docs/tokenizer_tests/tokenizer_metrics/berturk_cased_metrics.json`
- `cosmos_turkish_gpt2`: `docs/tokenizer_tests/tokenizer_metrics/cosmos_turkish_gpt2_metrics.json`
- `turna`: `docs/tokenizer_tests/tokenizer_metrics/turna_metrics.json`
- `vbart_large_base`: `docs/tokenizer_tests/tokenizer_metrics/vbart_large_base_metrics.json`
