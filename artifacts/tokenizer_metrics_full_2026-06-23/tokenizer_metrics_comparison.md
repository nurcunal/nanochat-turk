# Tokenizer Metrics Comparison

Tokenizer-only metrics are measured before model training. True BPB is model-dependent and should be reported from validation loss after each pretraining run; this table reports tokenizer compression, fertility, MorphBPE paper morphology metrics, boundary behavior, reversibility, and throughput.

Rows are sorted by MorphBPE-paper-style intrinsic quality: lower `mu_e`, then higher `mu_c` F1, then lower fertility `phi` as an efficiency tie-breaker. This ordering is not a custom weighted score.

| Rank | Tokenizer | Source | Impl. | Vocab | Docs | Bytes/token ↑ | Tokens/word phi ↓ | Isolated fertility ↓ | Morph edit mu_e ↓ | Morph edit norm ↓ | Morph exact ↑ | Morph consistency P ↑ | Morph consistency R ↑ | Morph consistency F1 mu_c ↑ | Boundary crossed ↓ | Roundtrip fail ↓ | Encode tok/s ↑ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | morphbpe_trmorph_131072 | local | morphbpe | 131,072 | 93,211,827 | 5.1101 | 1.5952 | 1.6027 | 1.0045 | 0.6388 | 0.5666 | 0.9955 | 0.2338 | 0.3786 | 0.5368 | 0.0000 | 1152127 |
| 2 | morphbpe_zemberek_131072 | local | morphbpe | 131,072 | 93,211,827 | 5.1219 | 1.5916 | 1.6026 | 1.0859 | 0.6982 | 0.5380 | 0.9933 | 0.2001 | 0.3330 | 0.6637 | 0.0000 | 1104754 |
| 3 | morphbpe_tdelight_131072 | local | morphbpe | 131,072 | 93,211,827 | 5.5399 | 1.4715 | 1.6481 | 1.1345 | 0.7437 | 0.5137 | 0.9933 | 0.2482 | 0.3971 | 0.7936 | 0.0000 | 1049066 |
| 4 | morphbpe_trmorph_65536 | local | morphbpe | 65,536 | 93,211,827 | 4.8780 | 1.6712 | 1.7625 | 1.1697 | 0.7982 | 0.5056 | 0.9919 | 0.2845 | 0.4421 | 0.4941 | 0.0000 | 1213371 |
| 5 | morphbpe_zemberek_65536 | local | morphbpe | 65,536 | 93,211,827 | 4.8989 | 1.6640 | 1.7616 | 1.2527 | 0.8560 | 0.4782 | 0.9895 | 0.2309 | 0.3743 | 0.6303 | 0.0000 | 1197625 |
| 6 | bpe_131072 | local | bpe | 131,072 | 93,211,827 | 6.0506 | 1.3473 | 1.6372 | 1.2623 | 0.8188 | 0.4701 | 0.9913 | 0.1373 | 0.2412 | 0.9449 | 0.0000 | 1062937 |
| 7 | morphbpe_tdelight_65536 | local | morphbpe | 65,536 | 93,211,827 | 5.2084 | 1.5651 | 1.8192 | 1.3071 | 0.9094 | 0.4556 | 0.9880 | 0.2745 | 0.4296 | 0.7297 | 0.0000 | 1168619 |
| 8 | morphbpe_trmorph_32768 | local | morphbpe | 32,768 | 93,211,827 | 4.6061 | 1.7698 | 1.9890 | 1.4319 | 1.0362 | 0.4154 | 0.9865 | 0.3466 | 0.5129 | 0.4582 | 0.0000 | 1229981 |
| 9 | bpe_65536 | local | bpe | 65,536 | 93,211,827 | 5.6894 | 1.4328 | 1.8322 | 1.4627 | 1.0124 | 0.4004 | 0.9804 | 0.1677 | 0.2863 | 0.9066 | 0.0000 | 1159248 |
| 10 | morphbpe_zemberek_32768 | local | morphbpe | 32,768 | 93,211,827 | 4.6446 | 1.7551 | 1.9810 | 1.4986 | 1.0773 | 0.3950 | 0.9831 | 0.2799 | 0.4357 | 0.5956 | 0.0000 | 1252642 |
| 11 | morphbpe_tdelight_32768 | local | morphbpe | 32,768 | 93,211,827 | 4.8335 | 1.6865 | 2.0357 | 1.5544 | 1.1335 | 0.3754 | 0.9825 | 0.3172 | 0.4795 | 0.6557 | 0.0000 | 1243517 |
| 12 | bpe_32768 | local | bpe | 32,768 | 93,211,827 | 5.2415 | 1.5553 | 2.0472 | 1.7096 | 1.2328 | 0.3246 | 0.9630 | 0.1949 | 0.3241 | 0.8534 | 0.0000 | 1217393 |

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
| morphbpe_trmorph_131072 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_zemberek_131072 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_tdelight_131072 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_trmorph_65536 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_zemberek_65536 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| bpe_131072 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_tdelight_65536 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_trmorph_32768 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| bpe_65536 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_zemberek_32768 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| morphbpe_tdelight_32768 | 500,000 | 50,000 | sklearn_minibatch_kmeans |
| bpe_32768 | 500,000 | 50,000 | sklearn_minibatch_kmeans |

## Source Files

- `morphbpe_trmorph_131072`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_trmorph_131072_metrics.json`
- `morphbpe_zemberek_131072`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_zemberek_131072_metrics.json`
- `morphbpe_tdelight_131072`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_tdelight_131072_metrics.json`
- `morphbpe_trmorph_65536`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_trmorph_65536_metrics.json`
- `morphbpe_zemberek_65536`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_zemberek_65536_metrics.json`
- `bpe_131072`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/bpe_131072_metrics.json`
- `morphbpe_tdelight_65536`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_tdelight_65536_metrics.json`
- `morphbpe_trmorph_32768`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_trmorph_32768_metrics.json`
- `bpe_65536`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/bpe_65536_metrics.json`
- `morphbpe_zemberek_32768`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_zemberek_32768_metrics.json`
- `morphbpe_tdelight_32768`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/morphbpe_tdelight_32768_metrics.json`
- `bpe_32768`: `/ari/users/nunal/nanochat-turk/docs/tokenizer_tests/tokenizer_metrics_full/bpe_32768_metrics.json`
