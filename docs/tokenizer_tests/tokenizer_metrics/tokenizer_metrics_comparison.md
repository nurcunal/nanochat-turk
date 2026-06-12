# Tokenizer Metrics Comparison

Tokenizer-only metrics are measured before model training. True BPB is model-dependent and should be reported from validation loss after each pretraining run; this table reports tokenizer compression, fertility, MorphBPE paper morphology metrics, boundary behavior, reversibility, and throughput.

| Tokenizer | Impl. | Docs | Bytes/token ↑ | Tokens/word phi ↓ | Isolated fertility ↓ | Morph edit mu_e ↓ | Morph edit norm ↓ | Morph exact ↑ | Morph consistency P ↑ | Morph consistency R ↑ | Morph consistency F1 mu_c ↑ | Boundary crossed ↓ | Roundtrip fail ↓ | Encode tok/s ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bpe_32768 | bpe | 50,000 | 5.0051 | 1.6157 | 2.0312 | 1.6836 | 1.2083 | 0.3342 | 0.9630 | 0.1949 | 0.3241 | 0.8395 | 0.0000 | 4922601 |
| morphbpe_trmorph_32768 | morphbpe | 50,000 | 4.4514 | 1.8166 | 1.9821 | 1.4126 | 1.0166 | 0.4258 | 0.9865 | 0.3466 | 0.5129 | 0.4569 | 0.0000 | 5111947 |
| morphbpe_zemberek_32768 | morphbpe | 50,000 | 4.4959 | 1.7986 | 1.9748 | 1.4817 | 1.0605 | 0.4040 | 0.9831 | 0.2799 | 0.4357 | 0.5906 | 0.0000 | 5075946 |

## Metric Notes

- `Bytes/token`: compression proxy; higher means fewer tokens for the same raw bytes.
- `Tokens/word phi`: corpus-level fertility, matching the MorphBPE paper's fertility metric.
- `Isolated word fertility`: token count when each sampled word is encoded alone.
- `Morph edit mu_e`: MorphBPE paper's raw average edit distance between gold morpheme sequence and tokenizer-piece sequence.
- `Morph edit norm`: same edit distance divided by the number of gold morphemes.
- `Morph exact`: share of sampled word occurrences whose tokenizer pieces exactly match gold morphemes.
- `Morph consistency P/R/F1 mu_c`: binary shared-token/shared-morpheme precision, recall, and harmonic mean; defaults follow the MorphBPE paper (`k=100`, `C=50`, `N=10`).
- `Boundary crossed`: share of TRmorph morpheme boundaries crossed by at least one tokenizer token.
- `Roundtrip fail`: fraction of sampled documents where decode(encode(text)) differs from text.
- High `Roundtrip fail` is expected for some encoder/seq2seq tokenizers that normalize or do not preserve raw text exactly.

## Morphology Metric Sample Sizes

| Tokenizer | Morph word occurrences | Consistency unique words | Consistency clustering |
|---|---:|---:|---|
| bpe_32768 | 200,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_trmorph_32768 | 200,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_zemberek_32768 | 200,000 | 50,000 | sklearn_minibatch_kmeans |

## Source Files

- `bpe_32768`: `docs/tokenizer_tests/tokenizer_metrics/bpe_32768_metrics.json`
- `morphbpe_trmorph_32768`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_32768_metrics.json`
- `morphbpe_zemberek_32768`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_32768_metrics.json`
