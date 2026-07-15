# Tokenizer Metrics Comparison

Tokenizer-only metrics are measured before model training. True BPB is model-dependent and should be reported from validation loss after each pretraining run; this table reports tokenizer compression, fertility, MorphBPE paper morphology metrics, boundary behavior, reversibility, and throughput.

Rows use a repository-defined diagnostic order: lower `mu_e`, then higher `mu_c` F1, then lower fertility `phi`, then lower boundary crossing. The MorphBPE paper defines the metrics, not this ranking. Use the order only for navigation; compare claims within a matched vocabulary and source.

| Order | Tokenizer | Source | Impl. | Vocab | Docs | Bytes/token ↑ | Tokens/word phi ↓ | Isolated fertility ↓ | Morph edit mu_e ↓ | Morph edit norm ↓ | Morph exact ↑ | Morph consistency P ↑ | Morph consistency R ↑ | Morph consistency F1 mu_c ↑ | Boundary crossed ↓ | Roundtrip fail ↓ | Encode tok/s ↑ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | morphbpe_trmorph_128k | local | morphbpe | 128k | 50,000 | 4.9597 | 1.6304 | 1.6027 | 1.0045 | 0.6388 | 0.5666 | 0.9955 | 0.2338 | 0.3786 | 0.5351 | 0.0000 | 3284424 |
| 2 | morphbpe_zemberek_128k | local | morphbpe | 128k | 50,000 | 4.9770 | 1.6248 | 1.6026 | 1.0859 | 0.6982 | 0.5380 | 0.9933 | 0.2001 | 0.3330 | 0.6587 | 0.0000 | 3350519 |
| 3 | morphbpe_tdelight_128k | local | morphbpe | 128k | 50,000 | 5.3521 | 1.5109 | 1.6481 | 1.1345 | 0.7437 | 0.5137 | 0.9933 | 0.2482 | 0.3971 | 0.7811 | 0.0000 | 3271241 |
| 4 | morphbpe_trmorph_64k | local | morphbpe | 64k | 50,000 | 4.7230 | 1.7121 | 1.7625 | 1.1697 | 0.7982 | 0.5056 | 0.9919 | 0.2845 | 0.4421 | 0.4931 | 0.0000 | 3659255 |
| 5 | morphbpe_zemberek_64k | local | morphbpe | 64k | 50,000 | 4.7485 | 1.7030 | 1.7616 | 1.2527 | 0.8560 | 0.4782 | 0.9895 | 0.2309 | 0.3743 | 0.6245 | 0.0000 | 3611250 |
| 6 | bpe_128k | local | bpe | 128k | 50,000 | 5.8114 | 1.3915 | 1.6372 | 1.2623 | 0.8188 | 0.4701 | 0.9913 | 0.1373 | 0.2412 | 0.9346 | 0.0000 | 3133245 |
| 7 | morphbpe_tdelight_64k | local | morphbpe | 64k | 50,000 | 5.0254 | 1.6091 | 1.8192 | 1.3071 | 0.9094 | 0.4556 | 0.9880 | 0.2745 | 0.4296 | 0.7179 | 0.0000 | 3409250 |
| 8 | morphbpe_trmorph_32k | local | morphbpe | 32k | 50,000 | 4.4514 | 1.8166 | 1.9821 | 1.4126 | 1.0166 | 0.4258 | 0.9865 | 0.3466 | 0.5129 | 0.4569 | 0.0000 | 5111947 |
| 9 | bpe_64k | local | bpe | 64k | 50,000 | 5.4434 | 1.4856 | 1.8322 | 1.4627 | 1.0124 | 0.4004 | 0.9804 | 0.1677 | 0.2863 | 0.8939 | 0.0000 | 3238593 |
| 10 | morphbpe_zemberek_32k | local | morphbpe | 32k | 50,000 | 4.4959 | 1.7986 | 1.9748 | 1.4817 | 1.0605 | 0.4040 | 0.9831 | 0.2799 | 0.4357 | 0.5906 | 0.0000 | 5075946 |
| 11 | morphbpe_tdelight_32k | local | morphbpe | 32k | 50,000 | 4.6564 | 1.7366 | 2.0357 | 1.5544 | 1.1335 | 0.3754 | 0.9825 | 0.3172 | 0.4795 | 0.6443 | 0.0000 | 3758650 |
| 12 | bpe_32k | local | bpe | 32k | 50,000 | 5.0051 | 1.6157 | 2.0312 | 1.6836 | 1.2083 | 0.3342 | 0.9630 | 0.1949 | 0.3241 | 0.8395 | 0.0000 | 4922601 |

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
| morphbpe_trmorph_128k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_zemberek_128k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_tdelight_128k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_trmorph_64k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_zemberek_64k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| bpe_128k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_tdelight_64k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_trmorph_32k | 200,000 | 50,000 | sklearn_minibatch_kmeans |
| bpe_64k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_zemberek_32k | 200,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_tdelight_32k | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| bpe_32k | 200,000 | 50,000 | sklearn_minibatch_kmeans |

## Source Files

- `morphbpe_trmorph_128k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_131072_metrics.json`
- `morphbpe_zemberek_128k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_131072_metrics.json`
- `morphbpe_tdelight_128k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_tdelight_131072_metrics.json`
- `morphbpe_trmorph_64k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_65536_metrics.json`
- `morphbpe_zemberek_64k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_65536_metrics.json`
- `bpe_128k`: `docs/tokenizer_tests/tokenizer_metrics/bpe_131072_metrics.json`
- `morphbpe_tdelight_64k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_tdelight_65536_metrics.json`
- `morphbpe_trmorph_32k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_32768_metrics.json`
- `bpe_64k`: `docs/tokenizer_tests/tokenizer_metrics/bpe_65536_metrics.json`
- `morphbpe_zemberek_32k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_32768_metrics.json`
- `morphbpe_tdelight_32k`: `docs/tokenizer_tests/tokenizer_metrics/morphbpe_tdelight_32768_metrics.json`
- `bpe_32k`: `docs/tokenizer_tests/tokenizer_metrics/bpe_32768_metrics.json`
